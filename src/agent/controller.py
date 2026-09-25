from __future__ import annotations

import json
import time
from typing import Any

try:
    from ..llm_client import ask_llm
    from ..observability import logger, metrics
    from ..tool_calling import validate_and_execute_tool
    from .models import AgentDecision, AgentState, AgentStep
    from .prompts import AGENT_SYSTEM_PROMPT, format_step_prompt
except (ImportError, ValueError):
    from llm_client import ask_llm
    from observability import logger, metrics
    from tool_calling import validate_and_execute_tool
    from agent.models import AgentDecision, AgentState, AgentStep
    from agent.prompts import AGENT_SYSTEM_PROMPT, format_step_prompt

MAX_AGENT_STEPS = 5


def synthesize_final_answer(question: str, state: AgentState) -> str:
    """Synthesize a complete natural-language answer based strictly on collected tool observations."""
    observations = []
    for s in state.steps:
        if s.tool_result:
            if s.tool_result.get("success"):
                observations.append(
                    f"Tool: {s.tool_name}\n"
                    f"Arguments: {json.dumps(s.arguments)}\n"
                    f"Result: {json.dumps(s.tool_result, indent=2)}"
                )
            else:
                observations.append(
                    f"Tool: {s.tool_name} failed with error: {s.tool_result.get('error')}"
                )

    if not observations:
        return "The investigation completed, but no relevant tool observations were collected to answer the question."

    prompt = (
        f"User question: {question}\n\n"
        f"Collected Tool Observations:\n"
        + "\n\n".join(observations)
        + "\n\nPlease provide a clear, factual, and complete final answer addressing the user's question "
        "based strictly on the above tool observations. Do not invent details not present in the results."
    )

    try:
        answer, _ = ask_llm(
            prompt,
            system_prompt=(
                "You are the AI Data Engineering Copilot. "
                "Synthesize a factual answer based solely on the provided tool observations."
            ),
        )
        return answer.strip()
    except Exception as exc:
        logger.error("agent_final_synthesis_failed error=%s", exc)
        return "\n\n".join(observations)


def run_agent_workflow(
    question: str,
    max_steps: int = MAX_AGENT_STEPS,
) -> dict[str, Any]:
    """
    Execute controlled multi-step Agentic workflow:
    1. Plan next step using Qwen3 based on question and current observations
    2. Execute MCP tools iteratively
    3. Observe outputs and update short-term state
    4. Terminate on 'final', 'refuse', loop detection, or max_steps limit
    5. Synthesize grounded answer
    """
    workflow_start = time.perf_counter()
    state = AgentState(question=question)
    executed_tool_calls: set[str] = set()

    logger.info("agent_workflow_started question=%r max_steps=%d", question, max_steps)

    for step_num in range(1, max_steps + 1):
        step_start = time.perf_counter()
        prompt = format_step_prompt(question, state)

        try:
            raw_output, _ = ask_llm(prompt, system_prompt=AGENT_SYSTEM_PROMPT)
            decision = AgentDecision.parse_from_llm_output(raw_output)
        except Exception as exc:
            logger.error("agent_decision_step_failed step=%d error=%s", step_num, exc)
            state.add_step(
                AgentStep(
                    step_number=step_num,
                    thought=f"Error generating decision: {exc}",
                    action="final",
                    latency_seconds=time.perf_counter() - step_start,
                )
            )
            state.status = "failed"
            break

        step_latency = time.perf_counter() - step_start
        logger.info(
            "agent_step_decided step=%d action=%s tool=%s thought=%r",
            step_num,
            decision.action,
            decision.tool_name,
            decision.thought,
        )

        # Case 1: Refusal (out of scope, unauthorized, etc.)
        if decision.action == "refuse":
            refusal_reason = decision.reason or "The requested operation is unauthorized or out of scope."
            state.add_step(
                AgentStep(
                    step_number=step_num,
                    thought=decision.thought,
                    action="refuse",
                    observation_summary=refusal_reason,
                    latency_seconds=step_latency,
                )
            )
            state.status = "refused"
            state.final_answer = refusal_reason
            break

        # Case 2: Final Answer
        if decision.action == "final":
            final_ans = decision.answer
            # If the model declared final but omitted an answer text, synthesize from observations
            if not final_ans or len(final_ans.strip()) < 5:
                final_ans = synthesize_final_answer(question, state)

            state.add_step(
                AgentStep(
                    step_number=step_num,
                    thought=decision.thought,
                    action="final",
                    answer=final_ans,
                    latency_seconds=step_latency,
                )
            )
            state.status = "completed"
            state.final_answer = final_ans
            break

        # Case 3: Tool Execution
        if decision.action == "tool":
            tool_name = decision.tool_name
            arguments = decision.arguments or {}

            # Loop detection: avoid executing identical tool + args twice
            call_sig = f"{tool_name}:{json.dumps(arguments, sort_keys=True)}"
            if call_sig in executed_tool_calls:
                logger.warning("agent_duplicate_tool_call_detected sig=%s concluding_workflow", call_sig)
                state.add_step(
                    AgentStep(
                        step_number=step_num,
                        thought="Duplicate tool call detected. Moving to final synthesis.",
                        action="final",
                        latency_seconds=step_latency,
                    )
                )
                state.status = "completed"
                state.final_answer = synthesize_final_answer(question, state)
                break

            executed_tool_calls.add(call_sig)

            # Execute via MCP Client through tool_calling layer
            tool_start = time.perf_counter()
            tool_result, tool_exec_latency = validate_and_execute_tool(tool_name, arguments)
            total_step_latency = step_latency + tool_exec_latency

            logger.info(
                "agent_tool_executed step=%d tool=%s success=%s latency=%.4fs",
                step_num,
                tool_name,
                tool_result.get("success", False),
                total_step_latency,
            )

            state.add_step(
                AgentStep(
                    step_number=step_num,
                    thought=decision.thought,
                    action="tool",
                    tool_name=tool_name,
                    arguments=arguments,
                    tool_result=tool_result,
                    observation_summary=json.dumps(tool_result),
                    latency_seconds=total_step_latency,
                )
            )

            # If tool execution failed due to unauthorized access or security block, let model observe or break if critical
            if not tool_result.get("success", False) and "access_denied" in str(tool_result.get("error", "")).lower():
                # Allow agent next step to explain or conclude
                pass

    # If loop ended without reaching 'final' or 'refuse'
    if state.status == "in_progress":
        logger.warning("agent_max_steps_reached steps=%d concluding", len(state.steps))
        state.status = "max_steps_reached"
        state.final_answer = synthesize_final_answer(question, state)

    total_workflow_latency = time.perf_counter() - workflow_start
    state.total_latency_seconds = total_workflow_latency

    # Observability metric update
    metrics.record_agent_run(
        total_steps=len(state.steps),
        max_steps_hit=(state.status == "max_steps_reached"),
    )

    primary_tool = None
    for step in state.steps:
        if step.tool_name:
            primary_tool = step.tool_name
            break

    logger.info(
        "agent_workflow_finished status=%s steps=%d latency=%.4fs",
        state.status,
        len(state.steps),
        total_workflow_latency,
    )

    return {
        "question": question,
        "answer": state.final_answer or "No answer could be determined.",
        "status": state.status,
        "total_steps": len(state.steps),
        "steps": [step.model_dump() for step in state.steps],
        "route": "tool",
        "tool_name": primary_tool,
        "sources": [],
        "latency_seconds": round(total_workflow_latency, 4),
        "cache_hit": False,
    }
