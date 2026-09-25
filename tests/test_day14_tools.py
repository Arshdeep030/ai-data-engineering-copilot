from __future__ import annotations

import unittest
from src.tools.row_count_tool import get_table_row_count
from src.tools.sample_tool import get_table_sample
from src.tools.tool_registry import registry
from src.mcp_client.client import MCPClient, get_mcp_client
from src.mcp_server.server import create_mcp_server
from src.tools.schema_tool import init_demo_db


class TestRowCountTool(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_demo_db()

    def test_row_count_valid_tables(self):
        """Test row count for all approved tables."""
        for tbl in ("employees", "orders", "customers"):
            res = get_table_row_count(tbl)
            self.assertTrue(res["success"], f"Failed for table: {tbl}")
            self.assertEqual(res["table_name"], tbl)
            self.assertIsInstance(res["row_count"], int)
            self.assertGreater(res["row_count"], 0)

    def test_row_count_whitespace_and_case(self):
        """Test normalization for casing and leading/trailing whitespace."""
        res = get_table_row_count("   OrDeRs  ")
        self.assertTrue(res["success"])
        self.assertEqual(res["table_name"], "orders")

    def test_row_count_disallowed_table(self):
        """Test rejection of non-approved tables."""
        res = get_table_row_count("secret_salaries")
        self.assertFalse(res["success"])
        self.assertIn("not approved for access", res["error"].lower())

    def test_row_count_empty_input(self):
        """Test validation error on empty string."""
        res = get_table_row_count("")
        self.assertFalse(res["success"])
        self.assertIn("non-empty", res["error"].lower())

    def test_row_count_sql_injection_defense(self):
        """Test that SQL injection attacks are safely blocked."""
        payloads = [
            "orders; DROP TABLE orders;--",
            "orders' OR '1'='1",
            "employees UNION SELECT * FROM sqlite_master",
        ]
        for p in payloads:
            res = get_table_row_count(p)
            self.assertFalse(res["success"])


class TestSampleTool(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_demo_db()

    def test_sample_valid_table_customers(self):
        """Test fetching a limited sample of customer records."""
        res = get_table_sample("customers", limit=2)
        self.assertTrue(res["success"])
        self.assertEqual(res["table_name"], "customers")
        self.assertEqual(res["limit"], 2)
        self.assertEqual(len(res["rows"]), 2)

        first_row = res["rows"][0]
        self.assertIn("customer_id", first_row)
        self.assertIn("name", first_row)
        self.assertIn("email", first_row)
        self.assertIn("country", first_row)

    def test_sample_default_limit(self):
        """Test fetching with default limit (5)."""
        res = get_table_sample("employees")
        self.assertTrue(res["success"])
        self.assertEqual(res["limit"], 5)
        self.assertLessEqual(len(res["rows"]), 5)

    def test_sample_limit_clamping(self):
        """Test clamping limit: negative clamps to 1, > 20 clamps to 20."""
        res_min = get_table_sample("orders", limit=-5)
        self.assertTrue(res_min["success"])
        self.assertEqual(res_min["limit"], 1)
        self.assertEqual(len(res_min["rows"]), 1)

        res_max = get_table_sample("orders", limit=100)
        self.assertTrue(res_max["success"])
        self.assertEqual(res_max["limit"], 20)

    def test_sample_disallowed_table(self):
        """Test rejection of non-approved tables."""
        res = get_table_sample("secret_passwords")
        self.assertFalse(res["success"])
        self.assertIn("not approved for access", res["error"].lower())

    def test_sample_sql_injection_defense(self):
        """Test SQL injection defense on table name."""
        res = get_table_sample("employees; DELETE FROM employees;--")
        self.assertFalse(res["success"])


class TestMCPMultiToolIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_demo_db()

    def setUp(self):
        self.server = create_mcp_server()
        self.client = MCPClient(server=self.server)
        self.client.connect()

    def tearDown(self):
        self.client.close()

    def test_mcp_tool_discovery_all_three_tools(self):
        """Test MCP discovery lists get_table_schema, get_table_row_count, and get_table_sample."""
        tools = self.client.list_tools()
        names = [t["name"] for t in tools]

        self.assertIn("get_table_schema", names)
        self.assertIn("get_table_row_count", names)
        self.assertIn("get_table_sample", names)

    def test_mcp_call_row_count(self):
        """Test executing get_table_row_count over MCP."""
        res = self.client.call_tool("get_table_row_count", {"table_name": "orders"})
        self.assertTrue(res.get("success"))
        self.assertEqual(res.get("table_name"), "orders")
        self.assertEqual(res.get("row_count"), 3)

    def test_mcp_call_sample(self):
        """Test executing get_table_sample over MCP."""
        res = self.client.call_tool("get_table_sample", {"table_name": "customers", "limit": 2})
        self.assertTrue(res.get("success"))
        self.assertEqual(res.get("table_name"), "customers")
        self.assertEqual(len(res.get("rows", [])), 2)

    def test_registry_contains_all_tools(self):
        """Test internal tool registry contains all three tools."""
        self.assertIsNotNone(registry.get("get_table_schema"))
        self.assertIsNotNone(registry.get("get_table_row_count"))
        self.assertIsNotNone(registry.get("get_table_sample"))


if __name__ == "__main__":
    unittest.main()
