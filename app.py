"""
AI Data Engineering Copilot - Hugging Face Space Entrypoint
Autonomous Agentic Copilot with MCP Database Tools & Hybrid RAG
"""
import os
import sys
from pathlib import Path

# Ensure root and src are in sys.path
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
for p in (str(ROOT_DIR), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import gradio as gr
from src.copilot_service import answer_question


def copilot_response(message, history):
    if not message or not message.strip():
        return "Please ask a question about Spark/data engineering or database tables."

    result = answer_question(message.strip())
    answer = result.get("answer", "")
    route = result.get("route", "rag").upper()
    tool = result.get("tool_name")
    steps = result.get("agent_steps", [])
    cache_hit = result.get("cache_hit", False)
    latency = result.get("latency_seconds", 0.0)

    # Provenance and telemetry footer
    badges = []
    badges.append(f"`ROUTE: {route}`")
    if tool:
        badges.append(f"`TOOL: {tool}`")
    if cache_hit:
        badges.append("`⚡ CACHE HIT`")
    badges.append(f"`LATENCY: {latency}s`")

    meta_footer = "\n\n---\n" + " • ".join(badges)

    # Multi-step agent timeline if applicable
    if steps and len(steps) > 1:
        timeline = "\n\n### 🤖 Agent Execution Timeline\n"
        for s in steps:
            action = s.get("action", "")
            t_name = s.get("tool_name", "")
            thought = s.get("thought", "")
            s_lat = s.get("latency_seconds", 0.0)
            timeline += f"- **Step {s.get('step_number', 1)}** ({s_lat:.2f}s): `{action}` {f'(`{t_name}`)' if t_name else ''}\n"
            if thought:
                timeline += f"  *Thought: {thought}*\n"
        meta_footer += timeline

    # Sources if RAG
    sources = result.get("sources", [])
    if sources:
        src_text = "\n\n### 📚 Retrieved Documentation Sources\n"
        for src in sources:
            src_text += f"- **{src.get('source', 'Doc')}** (Section: *{src.get('section', 'General')}*) — `{src.get('id', '')}`\n"
        meta_footer += src_text

    return answer + meta_footer


demo = gr.ChatInterface(
    fn=copilot_response,
    title="⚡ AI Data Engineering Copilot",
    description="Autonomous Agentic Copilot with MCP database tools, Hybrid RAG, and query caching.",
    examples=[
        "Analyze the employees table and tell me what columns it has, how many records it contains, and show me sample records.",
        "Tell me how many orders exist and show me some examples.",
        "What columns are in customers?",
        "What is Apache Spark?",
        "How many rows are in secret_financial_table?",
        "How should I optimize a Snowflake warehouse?",
    ],
    theme=gr.themes.Soft(primary_hue="indigo"),
)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port)
