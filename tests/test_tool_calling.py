from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from src.routing_models import RouteDecision
from src.tool_calling import (
    execute_tool_and_answer,
    generate_tool_answer,
    validate_and_execute_tool,
)
from src.copilot_service import answer_question, route_question
from src.cache import cache


class TestRoutingModels(unittest.TestCase):
    def test_parse_clean_json(self):
        raw = '{"route": "tool", "tool_name": "get_table_schema", "arguments": {"table_name": "employees"}}'
        decision = RouteDecision.parse_from_llm_output(raw)
        self.assertEqual(decision.route, "tool")
        self.assertEqual(decision.tool_name, "get_table_schema")
        self.assertEqual(decision.arguments, {"table_name": "employees"})

    def test_parse_markdown_fences(self):
        raw = """```json
{
  "route": "rag",
  "tool_name": null,
  "arguments": {}
}
```"""
        decision = RouteDecision.parse_from_llm_output(raw)
        self.assertEqual(decision.route, "rag")
        self.assertIsNone(decision.tool_name)
        self.assertEqual(decision.arguments, {})

    def test_parse_refusal_json(self):
        raw = '{"route": "refuse", "tool_name": null, "arguments": {}}'
        decision = RouteDecision.parse_from_llm_output(raw)
        self.assertEqual(decision.route, "refuse")

    def test_parse_surrounding_text(self):
        raw = 'Here is the routing decision: {"route": "tool", "tool_name": "get_table_schema", "arguments": {"table_name": "orders"}} Hope this helps!'
        decision = RouteDecision.parse_from_llm_output(raw)
        self.assertEqual(decision.route, "tool")
        self.assertEqual(decision.tool_name, "get_table_schema")
        self.assertEqual(decision.arguments.get("table_name"), "orders")


class TestToolExecution(unittest.TestCase):
    def test_validate_and_execute_valid_table(self):
        result, latency = validate_and_execute_tool(
            "get_table_schema",
            {"table_name": "employees"},
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["table_name"], "employees")
        self.assertGreater(len(result["columns"]), 0)
        self.assertGreaterEqual(latency, 0.0)

    def test_validate_and_execute_unregistered_tool(self):
        result, latency = validate_and_execute_tool(
            "unregistered_tool_name",
            {"table_name": "employees"},
        )
        self.assertFalse(result["success"])
        self.assertIn("not registered", result["error"].lower())

    def test_validate_and_execute_disallowed_table(self):
        result, latency = validate_and_execute_tool(
            "get_table_schema",
            {"table_name": "classified_financials"},
        )
        self.assertFalse(result["success"])
        self.assertIn("not approved", result["error"].lower())

    def test_validate_and_execute_sql_injection_defense(self):
        result, latency = validate_and_execute_tool(
            "get_table_schema",
            {"table_name": "employees; DROP TABLE employees;--"},
        )
        self.assertFalse(result["success"])
        self.assertTrue(
            "invalid table name" in result["error"].lower()
            or "not approved" in result["error"].lower()
        )


class TestCopilotService(unittest.TestCase):
    def setUp(self):
        cache.clear()

    @patch("src.copilot_service.route_question")
    @patch("src.copilot_service.execute_tool_and_answer")
    def test_answer_question_tool_routing(self, mock_tool_exec, mock_route):
        mock_route.return_value = RouteDecision(
            route="tool",
            tool_name="get_table_schema",
            arguments={"table_name": "employees"},
        )
        mock_tool_exec.return_value = {
            "question": "What columns are in employees?",
            "answer": "The employees table has employee_id, name, department, and salary.",
            "route": "tool",
            "tool_name": "get_table_schema",
            "arguments": {"table_name": "employees"},
            "tool_result": {"success": True},
            "sources": [],
            "tool_latency_seconds": 0.001,
            "generation_latency_seconds": 0.1,
            "latency_seconds": 0.101,
            "success": True,
        }

        resp = answer_question("What columns are in employees?")

        self.assertEqual(resp["route"], "tool")
        self.assertEqual(resp["tool_name"], "get_table_schema")
        self.assertIn("employees", resp["answer"])
        self.assertFalse(resp["cache_hit"])

    @patch("src.copilot_service.route_question")
    def test_answer_question_refusal_routing(self, mock_route):
        mock_route.return_value = RouteDecision(
            route="refuse",
            tool_name=None,
            arguments={},
        )

        resp = answer_question("How should I optimize a Snowflake warehouse?")

        self.assertEqual(resp["route"], "refuse")
        self.assertIn("not have enough information", resp["answer"].lower())
        self.assertEqual(resp["sources"], [])

    @patch("src.copilot_service.route_question")
    @patch("src.copilot_service.query_rag")
    def test_answer_question_rag_routing(self, mock_rag, mock_route):
        mock_route.return_value = RouteDecision(
            route="rag",
            tool_name=None,
            arguments={},
        )
        mock_rag.return_value = {
            "question": "What is Apache Spark?",
            "answer": "Apache Spark is a distributed computing engine.",
            "sources": [{"id": "spark_000", "section": "Intro", "source": "spark.md"}],
            "latency_seconds": 0.5,
            "cache_hit": False,
        }

        resp = answer_question("What is Apache Spark?")

        self.assertEqual(resp["route"], "rag")
        self.assertIsNone(resp["tool_name"])
        self.assertEqual(len(resp["sources"]), 1)

    @patch("src.copilot_service.route_question")
    @patch("src.copilot_service.execute_tool_and_answer")
    def test_caching_across_tool_queries(self, mock_tool_exec, mock_route):
        mock_route.return_value = RouteDecision(
            route="tool",
            tool_name="get_table_schema",
            arguments={"table_name": "orders"},
        )
        mock_tool_exec.return_value = {
            "question": "What is the schema of orders?",
            "answer": "The orders table contains order_id, customer_id, order_date, and amount.",
            "route": "tool",
            "tool_name": "get_table_schema",
            "arguments": {"table_name": "orders"},
            "tool_result": {"success": True},
            "sources": [],
            "tool_latency_seconds": 0.001,
            "generation_latency_seconds": 0.05,
            "latency_seconds": 0.051,
            "success": True,
        }

        q = "What is the schema of orders?"
        first_resp = answer_question(q)
        self.assertFalse(first_resp["cache_hit"])

        # Second query should hit cache immediately without invoking router
        second_resp = answer_question(q)
        self.assertTrue(second_resp["cache_hit"])
        self.assertEqual(mock_route.call_count, 1)


if __name__ == "__main__":
    unittest.main()
