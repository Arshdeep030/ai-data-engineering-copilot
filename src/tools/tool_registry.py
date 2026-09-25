from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

try:
    from ..observability import logger
    from .schema_tool import get_table_schema
    from .row_count_tool import get_table_row_count
    from .sample_tool import get_table_sample
except (ImportError, ValueError):
    from observability import logger
    from tools.schema_tool import get_table_schema
    from tools.row_count_tool import get_table_row_count
    from tools.sample_tool import get_table_sample


@dataclass
class ToolDefinition:
    """
    Metadata and handler contract for an executable Copilot tool.

    Designed to conform to both OpenAI function-calling specifications
    and the Model Context Protocol (MCP) tool schema.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., dict[str, Any]]
    read_only: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Returns standard dictionary metadata for internal routing or OpenAI tool formats."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "read_only": self.read_only,
        }

    def to_mcp_format(self) -> dict[str, Any]:
        """Returns Model Context Protocol (MCP) compliant tool descriptor."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


class ToolRegistry:
    """
    Central registry for managing, discovering, and executing tools.

    Provides a decoupled boundary between LLM tool selection and system execution.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        """Register a new tool definition."""
        self._tools[tool.name] = tool
        logger.info("tool_registered name=%s read_only=%s", tool.name, tool.read_only)

    def get(self, name: str) -> Optional[ToolDefinition]:
        """Retrieve a tool definition by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[dict[str, Any]]:
        """List metadata for all registered tools."""
        return [tool.to_dict() for tool in self._tools.values()]

    def list_mcp_tools(self) -> list[dict[str, Any]]:
        """List all tools in Model Context Protocol (MCP) format."""
        return [tool.to_mcp_format() for tool in self._tools.values()]

    def execute(self, tool_name: str, **kwargs: Any) -> dict[str, Any]:
        """
        Execute a registered tool safely with parameter binding and timing.
        Always returns a structured dict, never raises unhandled exceptions.
        """
        tool = self.get(tool_name)
        if not tool:
            available = list(self._tools.keys())
            logger.warning(
                "tool_execution_failed reason=not_found tool=%s available=%s",
                tool_name,
                available,
            )
            return {
                "success": False,
                "error": f"Tool '{tool_name}' is not registered. Available tools: {available}",
            }

        start_time = time.perf_counter()
        try:
            result = tool.handler(**kwargs)
            latency = time.perf_counter() - start_time

            success = result.get("success", True) if isinstance(result, dict) else True
            logger.info(
                "tool_execution_completed tool=%s success=%s latency=%.4fs",
                tool_name,
                success,
                latency,
            )
            return result
        except Exception as exc:
            latency = time.perf_counter() - start_time
            logger.error(
                "tool_execution_error tool=%s error=%s latency=%.4fs",
                tool_name,
                exc,
                latency,
            )
            return {
                "success": False,
                "error": f"Internal execution failure in tool '{tool_name}': {str(exc)}",
            }


# Instantiate default application registry
registry = ToolRegistry()

# Register Day 11's first data engineering tool
registry.register(
    ToolDefinition(
        name="get_table_schema",
        description=(
            "Returns column names and data types for an approved database table. "
            "Use this tool when answering questions about database schemas, table structure, "
            "column definitions, or data types for allowed tables ('employees', 'orders', 'customers')."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "table_name": {
                    "type": "string",
                    "description": "Name of the table to inspect (e.g. 'employees', 'orders', 'customers').",
                }
            },
            "required": ["table_name"],
        },
        handler=get_table_schema,
        read_only=True,
    )
)

registry.register(
    ToolDefinition(
        name="get_table_row_count",
        description=(
            "Returns the total number of rows in an approved database table. "
            "Use this tool when answering questions about dataset size, total records, "
            "or row counts for allowed tables ('employees', 'orders', 'customers')."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "table_name": {
                    "type": "string",
                    "description": "Name of the table to count rows in (e.g. 'employees', 'orders', 'customers').",
                }
            },
            "required": ["table_name"],
        },
        handler=get_table_row_count,
        read_only=True,
    )
)

registry.register(
    ToolDefinition(
        name="get_table_sample",
        description=(
            "Returns a sample of records from an approved database table. "
            "Use this tool when the user wants to see actual data, sample rows, "
            "or examples from allowed tables ('employees', 'orders', 'customers')."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "table_name": {
                    "type": "string",
                    "description": "Name of the table to sample (e.g. 'employees', 'orders', 'customers').",
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of rows to sample (between 1 and 20, default is 5).",
                    "default": 5,
                },
            },
            "required": ["table_name"],
        },
        handler=get_table_sample,
        read_only=True,
    )
)
