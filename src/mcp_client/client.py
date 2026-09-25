from __future__ import annotations

import time
from threading import Lock
from typing import Any, Optional

try:
    from ..observability import logger, metrics
    from ..mcp_server.server import DataEngineeringMCPServer, create_mcp_server
except (ImportError, ValueError):
    from observability import logger, metrics
    from mcp_server.server import DataEngineeringMCPServer, create_mcp_server


class MCPClient:
    """
    Model Context Protocol (MCP) Client for the AI Data Engineering Copilot.

    Connects to MCP servers, discovers exposed tools and capability schemas,
    dispatches tool execution requests via JSON-RPC 2.0, and records metrics.
    """

    def __init__(
        self,
        server: Optional[DataEngineeringMCPServer] = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        self._server = server
        self.timeout_seconds = timeout_seconds
        self._connected = False
        self._discovered_tools: dict[str, dict[str, Any]] = {}
        self._id_counter = 0
        self._lock = Lock()

    def _next_id(self) -> int:
        with self._lock:
            self._id_counter += 1
            return self._id_counter

    def _get_server(self) -> DataEngineeringMCPServer:
        if self._server is None:
            self._server = create_mcp_server()
        return self._server

    def connect(self) -> bool:
        """
        Initiate connection with MCP server, perform protocol handshake,
        and discover all exposed tools.
        """
        logger.info("mcp_client_connecting...")
        server = self._get_server()

        # 1. Initialize Handshake
        init_req = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "clientInfo": {
                    "name": "ai-data-engineering-copilot-client",
                    "version": "0.1.0",
                },
            },
        }

        init_resp = server.handle_message(init_req)
        if "error" in init_resp:
            logger.error("mcp_client_handshake_failed error=%s", init_resp["error"])
            return False

        # 2. Complete initialized notification
        notify_req = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "notifications/initialized",
            "params": {},
        }
        server.handle_message(notify_req)

        # 3. Discover exposed tools
        self._fetch_tools()
        self._connected = True
        logger.info(
            "mcp_client_connected tools_discovered=%s",
            list(self._discovered_tools.keys()),
        )
        return True

    def _fetch_tools(self) -> None:
        """Query MCP server for available tools via 'tools/list'."""
        server = self._get_server()
        list_req = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/list",
            "params": {},
        }
        resp = server.handle_message(list_req)
        tools = resp.get("result", {}).get("tools", [])

        self._discovered_tools.clear()
        for t in tools:
            self._discovered_tools[t["name"]] = t

    def list_tools(self) -> list[dict[str, Any]]:
        """
        Return metadata for all tools exposed by the connected MCP server.
        Discovers tools automatically if not yet connected.
        """
        if not self._connected or not self._discovered_tools:
            self.connect()
        return list(self._discovered_tools.values())

    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute an MCP tool via 'tools/call' JSON-RPC message.
        Validates availability, records latency, updates metrics, and returns structured result.
        """
        start_time = time.perf_counter()

        if not self._connected:
            self.connect()

        # 1. Validate tool discovery
        if name not in self._discovered_tools:
            latency = time.perf_counter() - start_time
            metrics.record_mcp_call(success=False, latency=latency)
            logger.warning(
                "mcp_client_tool_not_discovered name=%s discovered=%s",
                name,
                list(self._discovered_tools.keys()),
            )
            return {
                "success": False,
                "error": f"Tool '{name}' is not registered or exposed by the MCP server.",
            }

        server = self._get_server()
        call_req = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/call",
            "params": {
                "name": name,
                "arguments": arguments,
            },
        }

        try:
            resp = server.handle_message(call_req)
            latency = time.perf_counter() - start_time

            # Handle JSON-RPC protocol error
            if "error" in resp:
                err_msg = resp["error"].get("message", "Unknown protocol error")
                metrics.record_mcp_call(success=False, latency=latency)
                logger.error("mcp_client_protocol_error tool=%s error=%s", name, err_msg)
                return {
                    "success": False,
                    "error": f"MCP Protocol Error: {err_msg}",
                }

            # Handle CallToolResult
            result_payload = resp.get("result", {})
            is_error = result_payload.get("isError", False)
            structured = result_payload.get("structured", {})

            success = (not is_error) and structured.get("success", True)
            metrics.record_mcp_call(success=success, latency=latency)

            logger.info(
                "mcp_client_call_finished tool=%s success=%s latency=%.4fs",
                name,
                success,
                latency,
            )

            # Return structured dictionary
            if isinstance(structured, dict) and structured:
                return structured

            # Fallback if no structured output was attached
            content = result_payload.get("content", [])
            text_out = content[0].get("text", "") if content else ""
            return {
                "success": not is_error,
                "message": text_out,
            }

        except Exception as exc:
            latency = time.perf_counter() - start_time
            metrics.record_mcp_call(success=False, latency=latency)
            logger.error("mcp_client_execution_exception tool=%s error=%s", name, exc)
            return {
                "success": False,
                "error": f"MCP Client Error executing '{name}': {str(exc)}",
            }

    def close(self) -> None:
        """Close connection and reset discovery cache."""
        self._connected = False
        self._discovered_tools.clear()
        logger.info("mcp_client_closed")


# Singleton instance for application lifecycle
_global_client: Optional[MCPClient] = None
_client_lock = Lock()


def get_mcp_client() -> MCPClient:
    """Retrieve or initialize the global shared MCPClient."""
    global _global_client
    with _client_lock:
        if _global_client is None:
            _global_client = MCPClient()
            _global_client.connect()
        return _global_client
