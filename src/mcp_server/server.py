from __future__ import annotations

import json
import sys
import time
from typing import Any, Optional

try:
    from ..observability import logger
    from .tools import MCPTool, get_default_mcp_tools
except (ImportError, ValueError):
    from observability import logger
    from mcp_server.tools import MCPTool, get_default_mcp_tools


class DataEngineeringMCPServer:
    """
    Model Context Protocol (MCP) Server for Data Engineering.

    Exposes database inspection tools and capabilities over the standardized
    JSON-RPC 2.0 MCP protocol (specification version 2024-11-05).
    """

    PROTOCOL_VERSION = "2024-11-05"

    def __init__(
        self,
        name: str = "data-engineering-mcp-server",
        version: str = "0.1.0",
    ) -> None:
        self.name = name
        self.version = version
        self._tools: dict[str, MCPTool] = {}
        self._is_running = True

    def register_tool(self, tool: MCPTool) -> None:
        """Register an MCP tool capability."""
        self._tools[tool.name] = tool
        logger.info("mcp_server_tool_registered tool=%s", tool.name)

    def register_tools(self, tools: list[MCPTool]) -> None:
        """Register multiple MCP tools."""
        for t in tools:
            self.register_tool(t)

    def get_tool(self, name: str) -> Optional[MCPTool]:
        """Look up registered tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[dict[str, Any]]:
        """Return list of registered tools serialized to MCP schema."""
        return [t.to_mcp_dict() for t in self._tools.values()]

    def handle_message(self, request: dict[str, Any]) -> dict[str, Any]:
        """
        Process incoming JSON-RPC 2.0 message and return structured JSON-RPC response.
        Handles 'initialize', 'tools/list', 'tools/call', and error states.
        """
        req_id = request.get("id")
        method = request.get("method")
        params = request.get("params") or {}

        # 1. Validate JSON-RPC version
        if request.get("jsonrpc") != "2.0":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32600,
                    "message": "Invalid Request: jsonrpc must be '2.0'",
                },
            }

        # 2. Handle 'initialize'
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": self.PROTOCOL_VERSION,
                    "capabilities": {
                        "tools": {"listChanged": False},
                    },
                    "serverInfo": {
                        "name": self.name,
                        "version": self.version,
                    },
                },
            }

        # 3. Handle 'notifications/initialized'
        if method == "notifications/initialized":
            return {"jsonrpc": "2.0", "id": req_id, "result": {}}

        # 4. Handle 'tools/list'
        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "tools": self.list_tools(),
                },
            }

        # 5. Handle 'tools/call'
        if method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments") or {}

            if not tool_name or tool_name not in self._tools:
                logger.warning(
                    "mcp_server_tool_not_found requested_tool=%s available=%s",
                    tool_name,
                    list(self._tools.keys()),
                )
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {
                        "code": -32601,
                        "message": f"Tool '{tool_name}' not found on MCP server.",
                    },
                }

            tool = self._tools[tool_name]
            start_time = time.perf_counter()
            try:
                result = tool.handler(arguments)
                latency = time.perf_counter() - start_time

                is_error = not result.get("success", False)
                content_text = json.dumps(result, indent=2)

                logger.info(
                    "mcp_server_call_completed tool=%s isError=%s latency=%.4fs",
                    tool_name,
                    is_error,
                    latency,
                )

                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": content_text,
                            }
                        ],
                        "isError": is_error,
                        "structured": result,
                    },
                }
            except Exception as exc:
                latency = time.perf_counter() - start_time
                logger.error(
                    "mcp_server_call_exception tool=%s error=%s latency=%.4fs",
                    tool_name,
                    exc,
                    latency,
                )
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": f"Tool execution failed: {str(exc)}",
                            }
                        ],
                        "isError": True,
                        "structured": {
                            "success": False,
                            "error": str(exc),
                        },
                    },
                }

        # 6. Unknown method
        logger.warning("mcp_server_unknown_method method=%s", method)
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {
                "code": -32601,
                "message": f"Method '{method}' is not supported by MCP server.",
            },
        }

    def run_stdio(self) -> None:
        """Run line-delimited JSON-RPC loop over standard input and output."""
        logger.info("mcp_server_started_stdio name=%s", self.name)
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
                response = self.handle_message(request)
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()
            except Exception as exc:
                err_resp = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": f"Parse error: {str(exc)}"},
                }
                sys.stdout.write(json.dumps(err_resp) + "\n")
                sys.stdout.flush()


def create_mcp_server() -> DataEngineeringMCPServer:
    """Factory creating and configuring the default DataEngineeringMCPServer."""
    server = DataEngineeringMCPServer()
    server.register_tools(get_default_mcp_tools())
    return server


if __name__ == "__main__":
    server = create_mcp_server()
    server.run_stdio()
