from __future__ import annotations

import unittest
from pathlib import Path

from src.mcp_server.server import DataEngineeringMCPServer, create_mcp_server
from src.mcp_client.client import MCPClient, get_mcp_client
from src.observability import metrics
from src.tools.schema_tool import init_demo_db


class TestMCPServerAndClient(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_demo_db()

    def setUp(self):
        self.server = create_mcp_server()
        self.client = MCPClient(server=self.server)

    def tearDown(self):
        self.client.close()

    def test_server_initialize_handshake(self):
        """Test Test 1: MCP server initializes with protocolVersion, capabilities, and serverInfo."""
        resp = self.server.handle_message({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "clientInfo": {"name": "test-client", "version": "1.0"},
            },
        })

        self.assertEqual(resp["jsonrpc"], "2.0")
        self.assertEqual(resp["id"], 1)
        self.assertIn("result", resp)
        result = resp["result"]
        self.assertEqual(result["protocolVersion"], "2024-11-05")
        self.assertIn("tools", result["capabilities"])
        self.assertEqual(result["serverInfo"]["name"], "data-engineering-mcp-server")

    def test_client_connect_and_tool_discovery(self):
        """Test Test 2 & 3: Client discovers tools and verifies name, description, inputSchema."""
        connected = self.client.connect()
        self.assertTrue(connected)

        tools = self.client.list_tools()
        self.assertIsInstance(tools, list)
        self.assertGreater(len(tools), 0)

        schema_tool = next((t for t in tools if t["name"] == "get_table_schema"), None)
        self.assertIsNotNone(schema_tool)
        self.assertIn("description", schema_tool)
        self.assertIn("inputSchema", schema_tool)
        self.assertEqual(schema_tool["inputSchema"]["type"], "object")
        self.assertIn("table_name", schema_tool["inputSchema"]["properties"])

    def test_tool_execution_valid_table_employees(self):
        """Test Test 4: Call get_table_schema on 'employees' and verify schema output."""
        result = self.client.call_tool(
            "get_table_schema",
            {"table_name": "employees"},
        )

        self.assertTrue(result.get("success"))
        self.assertEqual(result.get("table_name"), "employees")
        columns = result.get("columns", [])
        self.assertGreater(len(columns), 0)

        col_names = [c["name"] for c in columns]
        self.assertIn("employee_id", col_names)
        self.assertIn("name", col_names)
        self.assertIn("department", col_names)
        self.assertIn("salary", col_names)

    def test_tool_execution_valid_tables_orders_and_customers(self):
        """Test calling get_table_schema for other allowed tables."""
        for tbl in ("orders", "customers"):
            res = self.client.call_tool("get_table_schema", {"table_name": tbl})
            self.assertTrue(res.get("success"), f"Failed for {tbl}")
            self.assertEqual(res.get("table_name"), tbl)
            self.assertGreater(len(res.get("columns", [])), 0)

    def test_disallowed_table_rejection(self):
        """Test Test 5: Rejection of non-allowlisted tables (e.g. secret_financial_table)."""
        result = self.client.call_tool(
            "get_table_schema",
            {"table_name": "secret_financial_table"},
        )

        self.assertFalse(result.get("success"))
        self.assertIn("not approved for access", result.get("error", "").lower())

    def test_sql_injection_defense(self):
        """Test Test 6: SQL injection payload is safely neutralized and rejected."""
        payloads = [
            "employees; DROP TABLE employees;--",
            "employees' OR '1'='1",
            "employees UNION SELECT * FROM sqlite_master",
        ]

        for p in payloads:
            result = self.client.call_tool("get_table_schema", {"table_name": p})
            self.assertFalse(result.get("success"), f"Injection was not blocked: {p}")
            err = result.get("error", "").lower()
            self.assertTrue("invalid table name" in err or "not approved" in err)

    def test_client_server_round_trip(self):
        """Test Test 7: Complete round trip Client -> Server -> Tool -> Result -> Client."""
        # Clean client instance
        client = MCPClient(server=self.server)
        client.connect()

        # Execute
        res = client.call_tool("get_table_schema", {"table_name": "employees"})
        self.assertTrue(res.get("success"))
        self.assertEqual(res.get("table_name"), "employees")

        client.close()

    def test_unknown_tool_rejection(self):
        """Test Test 8: Calling an unregistered tool returns clean structured error."""
        result = self.client.call_tool(
            "non_existent_tool",
            {"arg": 123},
        )
        self.assertFalse(result.get("success"))
        err_msg = result.get("error", "").lower()
        self.assertTrue("not registered" in err_msg or "not exposed" in err_msg)

    def test_protocol_error_handling(self):
        """Test invalid JSON-RPC method returns JSON-RPC error code."""
        resp = self.server.handle_message({
            "jsonrpc": "2.0",
            "id": 99,
            "method": "unsupported/method",
            "params": {},
        })
        self.assertIn("error", resp)
        self.assertEqual(resp["error"]["code"], -32601)

    def test_mcp_observability_metrics(self):
        """Test Test 10: MCP metrics tracking in RuntimeMetrics."""
        snapshot_before = metrics.snapshot()["mcp"]
        initial_calls = snapshot_before["tool_calls_total"]

        # Run 1 success and 1 error call
        self.client.call_tool("get_table_schema", {"table_name": "employees"})
        self.client.call_tool("get_table_schema", {"table_name": "disallowed_table"})

        snapshot_after = metrics.snapshot()["mcp"]
        self.assertEqual(snapshot_after["tool_calls_total"], initial_calls + 2)
        self.assertGreaterEqual(snapshot_after["successful_calls"], 1)
        self.assertGreaterEqual(snapshot_after["failed_calls"], 1)
        self.assertGreaterEqual(snapshot_after["average_tool_latency_seconds"], 0.0)


if __name__ == "__main__":
    unittest.main()
