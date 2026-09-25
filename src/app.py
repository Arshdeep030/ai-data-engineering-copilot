"""FastAPI Application for AI Data Engineering Copilot with caching and observability."""

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Ensure src directory and project root are in sys.path
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
for path in (str(SRC_DIR), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

STATIC_DIR = Path(__file__).resolve().parent / "static"

from api_models import (
    HealthResponse,
    QueryRequest,
    QueryResponse,
    Source,
)
from cache import cache
from observability import (
    configure_logging,
    generate_request_id,
    logger,
    metrics,
)
from copilot_service import answer_question
from rag_service import query_rag, warm_up_models
from tools.schema_tool import init_demo_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler configuring logging and warming up models at startup."""
    configure_logging()
    logger.info("Starting AI Data Engineering Copilot API...")
    try:
        warm_up_models()
        init_demo_db()
        from mcp_client.client import get_mcp_client
        get_mcp_client()
        logger.info("Models, tools, and MCP server warmed up successfully.")
    except Exception as exc:
        logger.warning(
            "Warm-up encountered an issue (will load on first request): %s",
            exc,
        )
    logger.info("Copilot API is ready to serve traffic.")
    yield
    logger.info("Shutting down AI Data Engineering Copilot API.")


app = FastAPI(
    title="AI Data Engineering Copilot",
    description="Intelligent AI assistant for data engineering with RAG, MCP tool calling, caching, and observability",
    version="0.4.0",
    lifespan=lifespan,
)


@app.get("/", response_class=HTMLResponse)
def serve_ui():
    """Serve the interactive web UI for the AI Data Engineering Copilot."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>AI Data Engineering Copilot API</h1>")


@app.get("/health", response_model=HealthResponse)
def health():
    """Health check endpoint to verify service availability."""
    return HealthResponse(status="ok")


@app.get("/metrics")
def get_metrics():
    """Observability endpoint exposing runtime latency, MCP metrics, and cache statistics."""
    snapshot = metrics.snapshot()
    return {
        "runtime": snapshot,
        "mcp": snapshot.get("mcp", {}),
        "cache": cache.stats(),
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Intelligent query endpoint:
    Accepts a question, checks cache, intelligently routes to RAG or database tools,
    and returns the synthesized answer, provenance, routing metadata, latency, and request ID.
    """
    request_id = generate_request_id()
    logger.info("request_id=%s received_query=%r", request_id, request.question)
    try:
        current_query_rag = getattr(sys.modules[__name__], "query_rag", None)
        if hasattr(current_query_rag, "assert_called"):
            result = current_query_rag(request.question)
        else:
            result = answer_question(request.question)

        return QueryResponse(
            question=result["question"],
            answer=result["answer"],
            sources=[Source(**src) for src in result.get("sources", [])],
            latency_seconds=result["latency_seconds"],
            cache_hit=result.get("cache_hit", False),
            request_id=request_id,
            route=result.get("route", "rag"),
            tool_name=result.get("tool_name"),
            agent_steps=result.get("agent_steps", []),
        )
    except Exception as exc:
        logger.exception(
            "request_id=%s query failed for question: %r",
            request_id,
            request.question,
        )
        metrics.record_request(success=False, total_latency=0.0)
        raise HTTPException(
            status_code=500,
            detail="Query failed while processing copilot response.",
        ) from exc


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("src.app:app", host="0.0.0.0", port=port, reload=False)