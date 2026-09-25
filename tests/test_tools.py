from __future__ import annotations

import unittest
from pathlib import Path

from src.tools.schema_tool import (
    ALLOWED_TABLES,
    get_table_schema,
    init_demo_db,
)
from src.tools.tool_registry import (
    ToolDefinition,
    ToolRegistry,
    registry,
)


class TestSchemaTool(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Ensure demo.db exists for testing
        init_demo_db()

    def test_valid_table_employees(self):
        """Test retrieving schema for an allowed, existing table."""
        result = get_table_schema("employees")

        self.assertTrue(result["success"])
        self.assertEqual(result["table_name"], "employees")
        self.assertIsInstance(result["columns"], list)
        self.assertGreater(len(result["columns"]), 0)

        col_names = [c["name"] for c in result["columns"]]
        self.assertIn("employee_id", col_names)
        self.assertIn("name", col_names)
        self.assertIn("department", col_names)
        self.assertIn("salary", col_names)

    def test_valid_table_orders_and_customers(self):
        """Test retrieving schema for all other allowed tables."""
        for tbl in ["orders", "customers"]:
            res = get_table_schema(tbl)
            self.assertTrue(res["success"], f"Failed for allowed table: {tbl}")
            self.assertEqual(res["table_name"], tbl)
            self.assertGreater(len(res["columns"]), 0)

    def test_table_name_case_and_whitespace_insensitivity(self):
        """Test that table names with leading/trailing spaces and mixed case work."""
        result = get_table_schema("  EmPlOyEeS  ")

        self.assertTrue(result["success"])
        self.assertEqual(result["table_name"], "employees")

    def test_disallowed_table(self):
        """Test that access to non-approved tables is strictly blocked."""
        result = get_table_schema("secret_salaries")

        self.assertFalse(result["success"])
        self.assertIn("not approved for access", result["error"].lower())

    def test_unknown_table(self):
        """Test query for a non-existent table name."""
        result = get_table_schema("does_not_exist")

        self.assertFalse(result["success"])
        self.assertIn("not approved for access", result["error"].lower())

    def test_empty_input(self):
        """Test validation failure when table_name is empty or whitespace."""
        result = get_table_schema("")
        self.assertFalse(result["success"])
        self.assertIn("non-empty", result["error"].lower())

        result_spaces = get_table_schema("   ")
        self.assertFalse(result_spaces["success"])
        self.assertIn("non-empty", result_spaces["error"].lower())

    def test_sql_injection_defense(self):
        """Test that SQL injection attacks are safely blocked before database execution."""
        injection_payloads = [
            "employees; DROP TABLE employees;--",
            "employees' OR '1'='1",
            "employees UNION SELECT * FROM sqlite_master",
            "employees; SELECT * FROM users;",
            "orders --",
            "customers/*comment*/",
        ]

        for payload in injection_payloads:
            result = get_table_schema(payload)
            self.assertFalse(
                result["success"],
                f"SQL injection payload was not blocked: {payload}",
            )
            # Must be stopped by identifier validation or allowlist
            self.assertTrue(
                "invalid table name" in result["error"].lower()
                or "not approved" in result["error"].lower()
            )


class TestToolRegistry(unittest.TestCase):
    def test_registry_contains_schema_tool(self):
        """Test that the global registry has get_table_schema registered."""
        tool = registry.get("get_table_schema")
        self.assertIsNotNone(tool)
        self.assertEqual(tool.name, "get_table_schema")
        self.assertTrue(tool.read_only)
        self.assertIn("table_name", tool.input_schema["properties"])

    def test_registry_execute_success(self):
        """Test executing a registered tool through the registry."""
        result = registry.execute("get_table_schema", table_name="employees")
        self.assertTrue(result["success"])
        self.assertEqual(result["table_name"], "employees")

    def test_registry_execute_unregistered_tool(self):
        """Test executing a non-existent tool returns a structured error."""
        result = registry.execute("drop_database_tool", force=True)
        self.assertFalse(result["success"])
        self.assertIn("not registered", result["error"].lower())

    def test_registry_mcp_format(self):
        """Test that tool definitions can be formatted for Model Context Protocol (MCP)."""
        mcp_tools = registry.list_mcp_tools()
        self.assertIsInstance(mcp_tools, list)
        self.assertGreater(len(mcp_tools), 0)

        schema_mcp = next((t for t in mcp_tools if t["name"] == "get_table_schema"), None)
        self.assertIsNotNone(schema_mcp)
        self.assertIn("inputSchema", schema_mcp)
        self.assertEqual(schema_mcp["inputSchema"]["type"], "object")
        self.assertIn("table_name", schema_mcp["inputSchema"]["properties"])


if __name__ == "__main__":
    unittest.main()
