from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from src.agent.models import AgentDecision, AgentState, AgentStep
from src.agent.controller import MAX_AGENT_STEPS, run_agent_workflow, synthesize_final_answer
from src.observability import metrics


class TestAgentModels(unittest.TestCase):
    def test_parse_clean_tool_decision(self):
        raw = (
            '{"thought": "Need schema first", "action": "tool", '
            '"tool_name": "get_table_schema", "arguments": {"table_name": "orders"}}'
        )
        dec = AgentDecision.parse_from_llm_output(raw)
        self.assertEqual(dec.action, "tool")
        self.assertEqual(dec.tool_name, "get_table_schema")
        self.assertEqual(dec.arguments, {"table_name": "orders"})
        self.assertEqual(dec.thought, "Need schema first")

    def test_parse_markdown_fences(self):
        raw = """```json
{
  "thought": "Investigation complete",
  "action": "final",
  "answer": "The orders table contains 3 records."
}
```"""
        dec = AgentDecision.parse_from_llm_output(raw)
        self.assertEqual(dec.action, "final")
        self.assertEqual(dec.answer, "The orders table contains 3 records.")

    def test_parse_refusal(self):
        raw = '{"thought": "Unauthorized table requested", "action": "refuse", "reason": "Table secret_vault is disallowed."}'
        dec = AgentDecision.parse_from_llm_output(raw)
        self.assertEqual(dec.action, "refuse")
        self.assertEqual(dec.reason, "Table secret_vault is disallowed.")

    def test_parse_fallback_text(self):
        raw = "I will check the schema of the employees table."
        dec = AgentDecision.parse_from_llm_output(raw)
        self.assertEqual(dec.action, "tool")
        self.assertEqual(dec.tool_name, "get_table_schema")
        self.assertEqual(dec.arguments.get("table_name"), "employees")

    def test_agent_state_history_rendering(self):
        state = AgentState(question="Analyze orders")
        self.assertEqual(state.format_history_for_prompt(), "No previous steps executed yet.")

        state.add_step(
            AgentStep(
                step_number=1,
                action="tool",
                tool_name="get_table_schema",
                arguments={"table_name": "orders"},
                tool_result={"success": True, "columns": [{"name": "order_id"}]},
            )
        )
        hist = state.format_history_for_prompt()
        self.assertIn("Step 1", hist)
        self.assertIn("get_table_schema", hist)
        self.assertIn("order_id", hist)


class TestAgentController(unittest.TestCase):
    @patch("src.agent.controller.ask_llm")
    def test_single_step_agent_workflow(self, mock_llm):
        # Step 1: LLM selects get_table_row_count
        # Step 2: LLM provides final answer
        mock_llm.side_effect = [
            (
                '{"thought": "Count rows in orders", "action": "tool", "tool_name": "get_table_row_count", "arguments": {"table_name": "orders"}}',
                {"input_tokens": 10, "output_tokens": 20},
            ),
            (
                '{"thought": "Got row count", "action": "final", "answer": "There are exactly 3 orders in the database."}',
                {"input_tokens": 20, "output_tokens": 15},
            ),
        ]

        result = run_agent_workflow("How many orders exist?", max_steps=4)

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["total_steps"], 2)
        self.assertIn("3 orders", result["answer"])
        self.assertEqual(result["steps"][0]["tool_name"], "get_table_row_count")
        self.assertEqual(result["steps"][1]["action"], "final")

    @patch("src.agent.controller.ask_llm")
    def test_multi_step_agent_workflow(self, mock_llm):
        # Step 1: get_table_schema
        # Step 2: get_table_row_count
        # Step 3: get_table_sample
        # Step 4: final answer
        mock_llm.side_effect = [
            (
                '{"thought": "Inspect schema first", "action": "tool", "tool_name": "get_table_schema", "arguments": {"table_name": "employees"}}',
                {},
            ),
            (
                '{"thought": "Check record count", "action": "tool", "tool_name": "get_table_row_count", "arguments": {"table_name": "employees"}}',
                {},
            ),
            (
                '{"thought": "Inspect sample records", "action": "tool", "tool_name": "get_table_sample", "arguments": {"table_name": "employees", "limit": 2}}',
                {},
            ),
            (
                '{"thought": "Synthesizing comprehensive analysis", "action": "final", "answer": "The employees table has 4 columns, 3 rows, and includes Alice and Bob."}',
                {},
            ),
        ]

        result = run_agent_workflow("Analyze the employees table", max_steps=5)

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["total_steps"], 4)
        self.assertEqual(result["steps"][0]["tool_name"], "get_table_schema")
        self.assertEqual(result["steps"][1]["tool_name"], "get_table_row_count")
        self.assertEqual(result["steps"][2]["tool_name"], "get_table_sample")
        self.assertEqual(result["steps"][3]["action"], "final")

    @patch("src.agent.controller.ask_llm")
    def test_duplicate_tool_call_loop_detection(self, mock_llm):
        # Model gets stuck asking for the same tool with identical arguments
        mock_llm.side_effect = [
            (
                '{"thought": "Count orders", "action": "tool", "tool_name": "get_table_row_count", "arguments": {"table_name": "orders"}}',
                {},
            ),
            (
                '{"thought": "Count orders again", "action": "tool", "tool_name": "get_table_row_count", "arguments": {"table_name": "orders"}}',
                {},
            ),
            # Synthesis fallback call
            (
                "Orders table contains 3 records.",
                {},
            ),
        ]

        result = run_agent_workflow("Count orders", max_steps=5)

        self.assertEqual(result["status"], "completed")
        # Loop detection should break after the duplicate attempt
        self.assertEqual(result["total_steps"], 2)
        self.assertEqual(result["steps"][1]["action"], "final")

    @patch("src.agent.controller.ask_llm")
    def test_max_steps_enforcement(self, mock_llm):
        # Model keeps requesting tools endlessly
        mock_llm.side_effect = [
            (
                '{"thought": "Step 1", "action": "tool", "tool_name": "get_table_schema", "arguments": {"table_name": "orders"}}',
                {},
            ),
            (
                '{"thought": "Step 2", "action": "tool", "tool_name": "get_table_row_count", "arguments": {"table_name": "orders"}}',
                {},
            ),
            (
                '{"thought": "Step 3", "action": "tool", "tool_name": "get_table_sample", "arguments": {"table_name": "orders", "limit": 1}}',
                {},
            ),
            # Synthesis fallback prompt
            (
                "Synthesized answer after max steps.",
                {},
            ),
        ]

        result = run_agent_workflow("Infinite request", max_steps=3)

        self.assertEqual(result["status"], "max_steps_reached")
        self.assertEqual(result["total_steps"], 3)
        self.assertIn("Synthesized answer after max steps", result["answer"])

    @patch("src.agent.controller.ask_llm")
    def test_agent_refusal_action(self, mock_llm):
        mock_llm.return_value = (
            '{"thought": "Cannot optimize Snowflake", "action": "refuse", "reason": "Snowflake is not in our data platform documentation."}',
            {},
        )

        result = run_agent_workflow("How to optimize Snowflake?")
        self.assertEqual(result["status"], "refused")
        self.assertIn("Snowflake is not in our data platform documentation", result["answer"])

    def test_observability_snapshot_has_agent_metrics(self):
        snap = metrics.snapshot()
        self.assertIn("agent", snap)
        self.assertIn("runs_total", snap["agent"])
        self.assertIn("multi_step_runs", snap["agent"])
        self.assertIn("total_steps", snap["agent"])
        self.assertIn("average_steps_per_run", snap["agent"])


if __name__ == "__main__":
    unittest.main()
