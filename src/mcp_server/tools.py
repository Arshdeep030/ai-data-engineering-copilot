from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

try:
    from ..tools.schema_tool import ALLOWED_TABLES, get_table_schema
    from ..tools.row_count_tool import get_table_row_count
    from ..tools.sample_tool import get_table_sample
except (ImportError, ValueError):
    from tools.schema_tool import ALLOWED_TABLES, get_table_schema
    from tools.row_count_tool import get_table_row_count
    from tools.sample_tool import get_table_sample


@dataclass
class MCPTool:
    """
    Representation of an MCP Tool adhering to the Model Context Protocol schema.
    """

    name: str
    description: str
    inputSchema: dict[str, Any]
    handler: Callable[[dict[str, Any]], dict[str, Any]]

    def to_mcp_dict(self) -> dict[str, Any]:
        """Serialize tool definition to standard MCP tools/list format."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.inputSchema,
        }


def _handle_get_table_schema(arguments: dict[str, Any]) -> dict[str, Any]:
    """Execute get_table_schema with validated arguments from MCP call."""
    table_name = arguments.get("table_name", "")
    return get_table_schema(table_name=table_name)


def _handle_get_table_row_count(arguments: dict[str, Any]) -> dict[str, Any]:
    """Execute get_table_row_count with validated arguments from MCP call."""
    table_name = arguments.get("table_name", "")
    return get_table_row_count(table_name=table_name)


def _handle_get_table_sample(arguments: dict[str, Any]) -> dict[str, Any]:
    """Execute get_table_sample with validated arguments from MCP call."""
    table_name = arguments.get("table_name", "")
    limit = arguments.get("limit", 5)
    return get_table_sample(table_name=table_name, limit=limit)


def get_default_mcp_tools() -> list[MCPTool]:
    """Return the list of default tools exposed by the Data Engineering MCP Server."""
    return [
        MCPTool(
            name="get_table_schema",
            description=(
                "Returns column names and data types for an approved database table. "
                "Use this tool when answering questions about database schemas, table structure, "
                "column definitions, or data types for allowed tables ('employees', 'orders', 'customers')."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": (
                            "Name of the database table to inspect (e.g. 'employees', 'orders', 'customers')."
                        ),
                    }
                },
                "required": ["table_name"],
            },
            handler=_handle_get_table_schema,
        ),
        MCPTool(
            name="get_table_row_count",
            description=(
                "Returns the total number of rows in an approved database table. "
                "Use this tool when answering questions about dataset size, total records, "
                "or how many rows exist in allowed tables ('employees', 'orders', 'customers')."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": (
                            "Name of the database table to count rows in (e.g. 'employees', 'orders', 'customers')."
                        ),
                    }
                },
                "required": ["table_name"],
            },
            handler=_handle_get_table_row_count,
        ),
        MCPTool(
            name="get_table_sample",
            description=(
                "Returns a sample of records from an approved database table. "
                "Use this tool when the user asks to see actual data rows, sample records, "
                "or examples from allowed tables ('employees', 'orders', 'customers')."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": (
                            "Name of the database table to sample (e.g. 'employees', 'orders', 'customers')."
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of sample rows to retrieve (1 to 20, default 5).",
                        "default": 5,
                    },
                },
                "required": ["table_name"],
            },
            handler=_handle_get_table_sample,
        ),
    ]
