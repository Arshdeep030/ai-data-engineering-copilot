from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from cache import cache
from copilot_service import answer_question
from tools.schema_tool import init_demo_db

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
EVAL_DATASET_PATH = PROJECT_ROOT / "data" / "evaluation" / "day15_agent_eval.json"
RESULTS_DIR = PROJECT_ROOT / "data" / "evaluation" / "results"


def run_evaluation() -> dict[str, Any]:
    """Execute the Day 15 Agentic Data Engineering Copilot Benchmark."""
    print("=" * 70)
    print("  DAY 15 AGENTIC DATA ENGINEERING COPILOT BENCHMARK")
    print("=" * 70)

    init_demo_db()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(EVAL_DATASET_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)

    total_cases = len(cases)
    print(f"Loaded {total_cases} test cases from {EVAL_DATASET_PATH.name}\n")

    results = []
    correct_routes = 0
    successful_tasks = 0
    multi_step_cases = 0
    multi_step_successes = 0
    security_cases = 0
    security_successes = 0
    total_steps = 0
    total_latency = 0.0

    for idx, case in enumerate(cases, 1):
        cid = case["id"]
        query = case["query"]
        task_type = case["task_type"]
        expected_route = case["expected_route"]
        expected_tools = case.get("expected_tools", [])

        print(f"[{idx}/{total_cases}] ID: {cid} | Type: {task_type}")
        print(f"  Query: {query!r}")

        cache.clear()
        start = time.perf_counter()
        resp = answer_question(query)
        elapsed = time.perf_counter() - start

        actual_route = resp.get("route")
        steps = resp.get("agent_steps", [])
        answer = resp.get("answer", "")
        tools_called = [s["tool_name"] for s in steps if s.get("tool_name")]

        route_correct = (actual_route == expected_route)
        if route_correct:
            correct_routes += 1

        task_success = False

        if task_type == "multi_step_agent":
            multi_step_cases += 1
            # Check if multiple tools or multi-step execution occurred and produced final answer
            has_multiple_steps = len(steps) >= case.get("min_steps", 2)
            called_expected = any(t in tools_called for t in expected_tools)
            has_answer = len(answer.strip()) > 20
            task_success = (has_multiple_steps or called_expected) and has_answer
            if task_success:
                multi_step_successes += 1

        elif task_type == "single_step_tool":
            called_expected = any(t in tools_called for t in expected_tools) or (resp.get("tool_name") in expected_tools)
            has_answer = len(answer.strip()) > 10
            task_success = called_expected and has_answer

        elif task_type == "rag":
            sources = resp.get("sources", [])
            task_success = (actual_route == "rag") and (len(sources) > 0) and (len(answer.strip()) > 20)

        elif task_type == "refusal":
            is_refused = (actual_route == "refuse") or ("not have enough information" in answer.lower())
            task_success = is_refused

        elif "security" in task_type:
            security_cases += 1
            # Blocked by allowlist, SQL validator, or refusal
            blocked = False
            for s in steps:
                res = s.get("tool_result") or {}
                if not res.get("success", True) and ("not approved" in str(res.get("error", "")).lower() or "security_blocked" in str(res.get("error", "")).lower() or "invalid table" in str(res.get("error", "")).lower()):
                    blocked = True
            if actual_route == "refuse" or "unauthorized" in answer.lower() or "disallowed" in answer.lower():
                blocked = True
            task_success = blocked
            if task_success:
                security_successes += 1

        if task_success:
            successful_tasks += 1

        num_steps = len(steps)
        total_steps += num_steps
        total_latency += elapsed

        status_str = "SUCCESS" if task_success else "FAIL"
        print(f"  Route: {actual_route} (expected: {expected_route})")
        print(f"  Steps: {num_steps} | Tools called: {tools_called}")
        print(f"  Latency: {elapsed:.2f}s | Result: {status_str}\n")

        results.append({
            "id": cid,
            "query": query,
            "task_type": task_type,
            "expected_route": expected_route,
            "actual_route": actual_route,
            "route_correct": route_correct,
            "expected_tools": expected_tools,
            "tools_called": tools_called,
            "steps_count": num_steps,
            "task_success": task_success,
            "latency_seconds": round(elapsed, 4),
            "answer_preview": answer[:150] + ("..." if len(answer) > 150 else ""),
            "steps": steps,
        })

    routing_acc = (correct_routes / total_cases) * 100
    task_completion_rate = (successful_tasks / total_cases) * 100
    multi_step_rate = (multi_step_successes / multi_step_cases * 100) if multi_step_cases else 100.0
    security_rate = (security_successes / security_cases * 100) if security_cases else 100.0
    avg_steps = total_steps / total_cases
    avg_latency = total_latency / total_cases

    summary = {
        "total_cases": total_cases,
        "task_completion_rate_pct": round(task_completion_rate, 1),
        "routing_accuracy_pct": round(routing_acc, 1),
        "multi_step_completion_rate_pct": round(multi_step_rate, 1),
        "security_protection_rate_pct": round(security_rate, 1),
        "average_steps_per_query": round(avg_steps, 2),
        "average_latency_seconds": round(avg_latency, 2),
        "results": results,
    }

    out_file = RESULTS_DIR / "day15_agent_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 70)
    print("  DAY 15 AGENTIC BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"Total Test Cases:               {total_cases}")
    print(f"Task Completion Rate:           {task_completion_rate:.1f}% ({successful_tasks}/{total_cases})")
    print(f"Routing Accuracy:               {routing_acc:.1f}% ({correct_routes}/{total_cases})")
    print(f"Multi-Step Completion Rate:     {multi_step_rate:.1f}% ({multi_step_successes}/{multi_step_cases})")
    print(f"Security Protection Rate:       {security_rate:.1f}% ({security_successes}/{security_cases})")
    print(f"Average Steps per Query:        {avg_steps:.2f}")
    print(f"Average Latency:                {avg_latency:.2f}s")
    print(f"Saved full results to:          {out_file}")
    print("=" * 70)

    return summary


if __name__ == "__main__":
    run_evaluation()
