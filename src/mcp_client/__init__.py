"""
Data Engineering MCP Client Package.

Connects to MCP servers, discovers capabilities, and dispatches tool execution requests.
"""

from .client import MCPClient, get_mcp_client

__all__ = [
    "MCPClient",
    "get_mcp_client",
]
