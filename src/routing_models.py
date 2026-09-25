from __future__ import annotations

import json
import re
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


class RouteDecision(BaseModel):
    """Structured decision produced by the Copilot routing layer."""

    route: Literal["rag", "tool", "agent", "refuse"] = Field(
        ...,
        description="Selected routing path: 'rag' for documentation, 'tool'/'agent' for database tools, 'refuse' for unsupported.",
    )
    tool_name: Optional[str] = Field(
        default=None,
        description="Name of the tool to invoke if route is 'tool'.",
    )
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured key-value arguments for the selected tool.",
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Optional brief explanation of the routing choice.",
    )

    @classmethod
    def parse_from_llm_output(cls, raw_output: str, question: str = "") -> "RouteDecision":
        """
        Robustly parse a JSON RouteDecision from LLM raw output.
        Handles markdown code blocks, whitespace, or surrounding text.
        """
        text = raw_output.strip()

        # Remove markdown code fences if present (e.g. ```json ... ```)
        if "```" in text:
            # Match first json fence or general fence
            fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
            if fence_match:
                text = fence_match.group(1).strip()
            else:
                # Strip leading/trailing code blocks
                text = re.sub(r"^```(?:json)?", "", text).strip()
                text = re.sub(r"```$", "", text).strip()

        # If still not starting with {, search for first valid json object
        if not text.startswith("{"):
            json_match = re.search(r"(\{.*\})", text, re.DOTALL)
            if json_match:
                text = json_match.group(1)

        try:
            data = json.loads(text)
        except Exception:
            # Fallback heuristic if JSON parsing fails: check keywords across model output and user question
            combined = (raw_output + " " + question).lower()
            if any(term in combined for term in ["schema", "column", "table", "get_table_schema", "analyze", "order", "employee", "customer", "count", "sample", "record"]):
                table = "employees" if "employee" in combined else ("orders" if "order" in combined else ("customers" if "customer" in combined else "employees"))
                tool = "get_table_row_count" if any(k in combined for k in ["count", "how many"]) else ("get_table_sample" if any(k in combined for k in ["sample", "example", "show me"]) else "get_table_schema")
                return cls(route="tool", tool_name=tool, arguments={"table_name": table})
            elif any(term in combined for term in ["spark", "executor", "driver", "sql", "partition", "kafka", "delta"]):
                return cls(route="rag", tool_name=None, arguments={})
            else:
                return cls(route="refuse", tool_name=None, arguments={})

        # Validate route value
        route_val = str(data.get("route", "")).strip().lower()
        if route_val == "agent":
            route_val = "tool"
        elif route_val not in ("rag", "tool", "refuse"):
            if data.get("tool_name"):
                route_val = "tool"
            else:
                route_val = "refuse"

        tool_name = data.get("tool_name")
        if route_val != "tool":
            tool_name = None
        elif tool_name:
            tool_name = str(tool_name).strip()

        args = data.get("arguments", {})
        if not isinstance(args, dict):
            args = {}

        return cls(
            route=route_val,  # type: ignore
            tool_name=tool_name,
            arguments=args,
            reasoning=data.get("reasoning"),
        )
