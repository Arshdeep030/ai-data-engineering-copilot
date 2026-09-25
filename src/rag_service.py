"""RAG service layer orchestrating hybrid retrieval, reranking, and generation with caching and observability."""

import logging
import sys
import time
from pathlib import Path

# Ensure src and project root are in sys.path
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
for path in (str(SRC_DIR), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from cache import cache
from hybrid_retriever import hybrid_search
from observability import logger, metrics
from rag_demo import generate_answer
from reranker import get_reranker, rerank
from vector_store import get_collection, get_embedding_model


def warm_up_models():
    """Pre-initialize expensive models at application startup to avoid request-time latency."""
    logger.info("Warming up vector store and embedding models...")
    collection = get_collection()
    if collection.count() == 0:
        logger.info("Chroma collection empty. Auto-ingesting markdown documents...")
        try:
            from ingest import main as ingest_main
            ingest_main()
        except Exception as e:
            logger.warning("Auto-ingestion skipped or failed: %s", e)
    get_embedding_model()
    logger.info("Warming up cross-encoder reranker...")
    get_reranker()
    logger.info("Models initialized and ready.")



def query_rag(question: str) -> dict:
    """
    Execute the RAG pipeline with caching and observability.
    1. Check cache (normalized key lookup)
    2. On miss:
       - Time hybrid retrieval (dense + BM25 + RRF)
       - Time cross-encoder reranking
       - Time Qwen3 generation
       - Store clean result (answer + sources) in cache
    3. Record granular runtime metrics (retrieval, reranking, generation, total latency)
    """
    request_start = time.perf_counter()

    # Step 1: Cache lookup
    cached_response = cache.get(question)
    if cached_response is not None:
        total_latency = time.perf_counter() - request_start
        logger.info(
            "cache_hit=true question=%r latency=%.4fs",
            question,
            total_latency,
        )

        response = dict(cached_response)
        response["latency_seconds"] = round(total_latency, 4)
        response["cache_hit"] = True

        metrics.record_request(
            success=True,
            total_latency=total_latency,
        )
        return response

    logger.info("cache_hit=false question=%r", question)

    # Step 2: Instrument Hybrid Retrieval (5 candidates)
    retrieval_start = time.perf_counter()
    candidates = hybrid_search(question, top_k=5)
    retrieval_latency = time.perf_counter() - retrieval_start
    logger.info(
        "retrieval_completed latency=%.4fs results=%d",
        retrieval_latency,
        len(candidates),
    )

    # Step 3: Instrument Cross-Encoder Reranking (top 3)
    reranking_start = time.perf_counter()
    results = rerank(question, candidates, top_k=3)
    reranking_latency = time.perf_counter() - reranking_start
    logger.info(
        "reranking_completed latency=%.4fs candidates=%d",
        reranking_latency,
        len(results),
    )

    # Step 4: Instrument Qwen3 Generation
    generation_start = time.perf_counter()
    if not results:
        answer = "The retrieved context does not provide enough information to answer this question."
        sources = []
    else:
        answer = generate_answer(question, results)
        sources = [
            {
                "id": r["id"],
                "section": r.get("metadata", {}).get("section"),
                "source": r.get("metadata", {}).get("source"),
            }
            for r in results
        ]
    generation_latency = time.perf_counter() - generation_start
    logger.info(
        "generation_completed latency=%.4fs",
        generation_latency,
    )

    total_latency = time.perf_counter() - request_start

    # Prepare response
    response = {
        "question": question,
        "answer": answer,
        "sources": sources,
        "latency_seconds": round(total_latency, 4),
        "cache_hit": False,
    }

    # Cache clean RAG result (only store question, answer, sources)
    cache_entry = {
        "question": question,
        "answer": answer,
        "sources": sources,
    }
    cache.set(question, cache_entry)

    # Record metrics
    metrics.record_request(
        success=True,
        total_latency=total_latency,
        retrieval_latency=retrieval_latency,
        reranking_latency=reranking_latency,
        generation_latency=generation_latency,
    )

    logger.info(
        "request_completed cache_hit=false latency=%.4fs",
        total_latency,
    )

    return response
