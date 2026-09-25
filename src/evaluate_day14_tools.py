from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Ensure src and root are in path
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
for p in (str(SRC_DIR), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from copilot_service import route_question
from mcp_client.client import get_mcp_client
from tools.schema_tool import init_demo_db

EVAL_PATH = PROJECT_ROOT / "data" / "evaluation" / "day14_tool_eval.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "evaluation" / "results"
OUTPUT_PATH = OUTPUT_DIR / "day14_tool_results.json"


def evaluate_day14_tools() -> dict:
    init_demo_db()
    client = get_mcp_client()

    if not EVAL_PATH.exists():
        raise FileNotFoundError(f"Evaluation dataset not found at {EVAL_PATH}")

    with open(EVAL_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)

    print(f"Loaded {len(cases)} Day 14 test cases from {EVAL_PATH.name}...")

    total_cases = len(cases)
    correct_routes = 0

    tool_cases = 0
    correct_tools = 0
    correct_arguments = 0
    successful_executions = 0

    # Per-tool breakdown tracking
    per_tool_stats = {
        "get_table_schema": {"total": 0, "correct_tool": 0, "correct_args": 0, "success_exec": 0},
        "get_table_row_count": {"total": 0, "correct_tool": 0, "correct_args": 0, "success_exec": 0},
        "get_table_sample": {"total": 0, "correct_tool": 0, "correct_args": 0, "success_exec": 0},
    }

    detailed_results = []

    print("\n" + "=" * 78)
    print(f"{'ID':<14} | {'QUERY':<32} | {'EXP TOOL':<20} | {'ACT TOOL':<20} | {'MATCH':<5}")
    print("=" * 78)

    for case in cases:
        case_id = case["id"]
        query = case["query"]
        category = case["category"]
        expected_tool = case.get("expected_tool")
        expected_args = case.get("expected_arguments") or {}

        if category in ("tool", "disallowed_tool"):
            expected_route = "tool"
        elif category == "rag":
            expected_route = "rag"
        else:
            expected_route = "refuse"

        start_time = time.perf_counter()
        decision = route_question(query)
        latency = time.perf_counter() - start_time

        route_correct = (decision.route == expected_route)
        if route_correct:
            correct_routes += 1

        tool_correct = False
        args_correct = False
        exec_success = False

        if expected_route == "tool":
            tool_cases += 1
            tool_correct = (decision.tool_name == expected_tool)
            if tool_correct:
                correct_tools += 1

            # Argument validation (table_name match + optional limit match)
            act_tbl = (decision.arguments.get("table_name") or "").strip().lower()
            exp_tbl = (expected_args.get("table_name") or "").strip().lower()
            table_match = (act_tbl == exp_tbl)

            limit_match = True
            if "limit" in expected_args:
                act_limit = decision.arguments.get("limit")
                exp_limit = expected_args.get("limit")
                limit_match = (act_limit == exp_limit)

            args_correct = table_match and limit_match
            if args_correct:
                correct_arguments += 1

            # Execute via MCP Client
            if decision.tool_name:
                exec_res = client.call_tool(decision.tool_name, decision.arguments)
                exec_success = exec_res.get("success", False)
                if exec_success:
                    successful_executions += 1

            # Update per-tool statistics
            if expected_tool in per_tool_stats:
                st = per_tool_stats[expected_tool]
                st["total"] += 1
                if tool_correct:
                    st["correct_tool"] += 1
                if args_correct:
                    st["correct_args"] += 1
                if exec_success:
                    st["success_exec"] += 1

        match_str = "PASS" if (route_correct and (expected_route != "tool" or tool_correct)) else "FAIL"
        trunc_query = (query[:30] + "..") if len(query) > 30 else query
        exp_disp = expected_tool or f"route:{expected_route}"
        act_disp = decision.tool_name or f"route:{decision.route}"
        print(f"{case_id:<14} | {trunc_query:<32} | {exp_disp:<20} | {act_disp:<20} | {match_str:<5}")

        detailed_results.append({
            "id": case_id,
            "query": query,
            "category": category,
            "expected_route": expected_route,
            "actual_route": decision.route,
            "route_correct": route_correct,
            "expected_tool": expected_tool,
            "actual_tool": decision.tool_name,
            "tool_correct": tool_correct if expected_route == "tool" else None,
            "expected_arguments": expected_args,
            "actual_arguments": decision.arguments,
            "arguments_correct": args_correct if expected_route == "tool" else None,
            "execution_success": exec_success if expected_route == "tool" else None,
            "latency_seconds": round(latency, 4),
        })

    routing_accuracy = round(correct_routes / total_cases, 4) if total_cases > 0 else 0.0
    tool_accuracy = round(correct_tools / tool_cases, 4) if tool_cases > 0 else 0.0
    arg_accuracy = round(correct_arguments / tool_cases, 4) if tool_cases > 0 else 0.0

    summary = {
        "total_cases": total_cases,
        "routing_accuracy": routing_accuracy,
        "tool_cases": tool_cases,
        "tool_selection_accuracy": tool_accuracy,
        "argument_accuracy": arg_accuracy,
        "successful_tool_executions": successful_executions,
        "per_tool_stats": per_tool_stats,
        "detailed_results": detailed_results,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 78)
    print("DAY 14 MULTI-TOOL EVALUATION SUMMARY")
    print("=" * 78)
    print(f"Total Cases:                 {total_cases}")
    print(f"Routing Accuracy:            {routing_accuracy * 100:.1f}% ({correct_routes}/{total_cases})")
    print(f"Tool Selection Accuracy:     {tool_accuracy * 100:.1f}% ({correct_tools}/{tool_cases})")
    print(f"Argument Accuracy:           {arg_accuracy * 100:.1f}% ({correct_arguments}/{tool_cases})")
    print(f"Tool Execution Success Rate: {successful_executions}/{tool_cases}")
    print("\nPer-Tool Performance:")
    for tname, stats in per_tool_stats.items():
        t_tot = stats["total"]
        if t_tot > 0:
            sel_rate = (stats["correct_tool"] / t_tot) * 100
            print(f"  - {tname:<22}: {sel_rate:.1f}% selection ({stats['correct_tool']}/{t_tot})")
    print(f"\nDetailed results saved to:   {OUTPUT_PATH}")
    print("=" * 78)

    return summary


if __name__ == "__main__":
    evaluate_day14_tools()
