from __future__ import annotations

import logging
import time
import uuid
from contextlib import contextmanager
from threading import Lock
from typing import Iterator

logger = logging.getLogger("ai_copilot")


def configure_logging() -> None:
    """Configure application-wide logging."""
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        ),
    )


def generate_request_id() -> str:
    """Generate a short unique request ID."""
    return uuid.uuid4().hex[:12]


@contextmanager
def timer() -> Iterator[dict[str, float]]:
    """
    Measure elapsed time for a block of code.

    Example:
        with timer() as timing:
            do_work()
        print(timing["elapsed_seconds"])
    """
    start = time.perf_counter()
    result = {"elapsed_seconds": 0.0}
    try:
        yield result
    finally:
        result["elapsed_seconds"] = time.perf_counter() - start


class RuntimeMetrics:
    """
    Lightweight in-memory runtime metrics.

    These metrics are intentionally simple for Day 10.
    Later they can be exported to Prometheus/OpenTelemetry.
    """

    def __init__(self) -> None:
        self._lock = Lock()

        self.requests_total = 0
        self.requests_successful = 0
        self.requests_failed = 0

        self.total_latency_seconds = 0.0

        self.total_retrieval_seconds = 0.0
        self.total_reranking_seconds = 0.0
        self.total_generation_seconds = 0.0

        # MCP observability metrics
        self.mcp_tool_calls_total = 0
        self.mcp_successful_calls = 0
        self.mcp_failed_calls = 0
        self.mcp_total_tool_latency_seconds = 0.0

        # Day 15 Agentic observability metrics
        self.agent_runs_total = 0
        self.agent_multi_step_runs = 0
        self.agent_total_steps = 0
        self.agent_max_step_violations = 0

    def record_request(
        self,
        *,
        success: bool,
        total_latency: float,
        retrieval_latency: float = 0.0,
        reranking_latency: float = 0.0,
        generation_latency: float = 0.0,
    ) -> None:
        with self._lock:
            self.requests_total += 1

            if success:
                self.requests_successful += 1
            else:
                self.requests_failed += 1

            self.total_latency_seconds += total_latency
            self.total_retrieval_seconds += retrieval_latency
            self.total_reranking_seconds += reranking_latency
            self.total_generation_seconds += generation_latency

    def record_mcp_call(
        self,
        *,
        success: bool,
        latency: float,
    ) -> None:
        """Record an MCP tool execution invocation."""
        with self._lock:
            self.mcp_tool_calls_total += 1
            if success:
                self.mcp_successful_calls += 1
            else:
                self.mcp_failed_calls += 1
            self.mcp_total_tool_latency_seconds += latency

    def record_agent_run(
        self,
        *,
        total_steps: int,
        max_steps_hit: bool = False,
    ) -> None:
        """Record a multi-step agent workflow execution."""
        with self._lock:
            self.agent_runs_total += 1
            if total_steps > 1:
                self.agent_multi_step_runs += 1
            self.agent_total_steps += total_steps
            if max_steps_hit:
                self.agent_max_step_violations += 1

    def snapshot(self) -> dict:
        with self._lock:
            total = self.requests_total

            if total:
                avg_latency = self.total_latency_seconds / total
            else:
                avg_latency = 0.0

            mcp_total = self.mcp_tool_calls_total
            mcp_avg_latency = (
                self.mcp_total_tool_latency_seconds / mcp_total
                if mcp_total
                else 0.0
            )

            agent_runs = self.agent_runs_total
            avg_steps = (
                self.agent_total_steps / agent_runs
                if agent_runs
                else 0.0
            )

            return {
                "requests_total": self.requests_total,
                "requests_successful": self.requests_successful,
                "requests_failed": self.requests_failed,
                "average_latency_seconds": round(avg_latency, 4),
                "total_retrieval_seconds": round(
                    self.total_retrieval_seconds, 4
                ),
                "total_reranking_seconds": round(
                    self.total_reranking_seconds, 4
                ),
                "total_generation_seconds": round(
                    self.total_generation_seconds, 4
                ),
                "mcp": {
                    "tool_calls_total": self.mcp_tool_calls_total,
                    "successful_calls": self.mcp_successful_calls,
                    "failed_calls": self.mcp_failed_calls,
                    "average_tool_latency_seconds": round(mcp_avg_latency, 4),
                },
                "agent": {
                    "runs_total": self.agent_runs_total,
                    "multi_step_runs": self.agent_multi_step_runs,
                    "total_steps": self.agent_total_steps,
                    "average_steps_per_run": round(avg_steps, 2),
                    "max_step_violations": self.agent_max_step_violations,
                },
            }


metrics = RuntimeMetrics()
