from __future__ import annotations

import json
import time
from typing import Any, Optional

try:
    from .llm_client import ask_llm
    from .observability import logger
    from .mcp_client.client import get_mcp_client
    from .tools.tool_registry import registry
except (ImportError, ValueError):
    from llm_client import ask_llm
    from observability import logger
    from mcp_client.client import get_mcp_client
    from tools.tool_registry import registry


def validate_and_execute_tool(
    tool_name: Optional[str],
    arguments: dict[str, Any],
) -> tuple[dict[str, Any], float]:
    """
    Validate tool existence and execute through the Model Context Protocol (MCP) Client.
    Returns (tool_result_dict, execution_latency_seconds).
    """
    if not tool_name:
        return (
            {
                "success": False,
                "error": "No tool name provided for tool execution.",
            },
            0.0,
        )

    client = get_mcp_client()
    start = time.perf_counter()
    result = client.call_tool(tool_name, arguments)
    latency = time.perf_counter() - start

    return result, latency


def generate_tool_answer(
    question: str,
    tool_name: str,
    tool_result: dict[str, Any],
) -> tuple[str, float]:
    """
    Synthesize a clean, natural-language response based strictly on live tool output.
    Returns (answer_text, generation_latency_seconds).
    """
    start = time.perf_counter()

    if not tool_result.get("success", False):
        error_msg = tool_result.get("error", "Unknown tool error occurred.")
        latency = time.perf_counter() - start
        return (
            f"Unable to execute tool '{tool_name}': {error_msg}",
            latency,
        )

    table_name = tool_result.get("table_name", "unknown")

    system_prompt = (
        "You are the AI Data Engineering Copilot. "
        "Answer the user's question using ONLY the provided live tool execution output. "
        "State numbers, schemas, or sample rows accurately and clearly based on the tool result. "
        "Do not invent or assume any data not present in the tool output."
    )

    user_prompt = (
        f"User question: {question}\n\n"
        f"Tool executed: {tool_name}\n"
        f"Tool output:\n{json.dumps(tool_result, indent=2)}\n\n"
        "Please provide a concise, factual answer for the user based strictly on the tool output."
    )

    try:
        answer, _ = ask_llm(user_prompt, system_prompt=system_prompt)
        generation_latency = time.perf_counter() - start
        return answer.strip(), generation_latency
    except Exception as exc:
        generation_latency = time.perf_counter() - start
        logger.error(
            "tool_answer_generation_failed tool=%s error=%s",
            tool_name,
            exc,
        )
        # Deterministic fallback formatting for each tool type
        if "row_count" in tool_result and "rows" not in tool_result:
            fallback = f"The '{table_name}' table contains {tool_result['row_count']} rows."
        elif "rows" in tool_result:
            fallback = (
                f"Sample records from '{table_name}' ({len(tool_result['rows'])} rows):\n"
                + json.dumps(tool_result["rows"], indent=2)
            )
        else:
            columns = tool_result.get("columns", [])
            col_desc = ", ".join(f"{c['name']} ({c['type']})" for c in columns)
            fallback = f"The '{table_name}' table contains {len(columns)} columns: {col_desc}."
        return fallback, generation_latency


def execute_tool_and_answer(
    tool_name: Optional[str],
    arguments: dict[str, Any],
    question: str,
) -> dict[str, Any]:
    """
    Complete tool-calling loop:
    1. Validates and executes tool via registry
    2. Passes structured result to Qwen3 for response synthesis
    3. Returns full audit and latency metrics
    """
    loop_start = time.perf_counter()

    # 1. Tool execution
    tool_result, tool_latency = validate_and_execute_tool(tool_name, arguments)

    # 2. Answer synthesis
    answer, gen_latency = generate_tool_answer(
        question=question,
        tool_name=tool_name or "unknown",
        tool_result=tool_result,
    )

    total_latency = time.perf_counter() - loop_start

    return {
        "question": question,
        "answer": answer,
        "route": "tool",
        "tool_name": tool_name,
        "arguments": arguments,
        "tool_result": tool_result,
        "sources": [],
        "tool_latency_seconds": round(tool_latency, 4),
        "generation_latency_seconds": round(gen_latency, 4),
        "latency_seconds": round(total_latency, 4),
        "success": tool_result.get("success", False),
    }
