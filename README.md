# ⚡ AI Data Engineering Copilot
### Production-Grade Agentic Copilot with Model Context Protocol (MCP), Hybrid RAG, & Multi-Step Reasoning

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![MCP](https://img.shields.io/badge/Protocol-MCP%20JSON--RPC%202.0-indigo.svg)](https://modelcontextprotocol.io/)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-orange.svg)](https://www.trychroma.com/)
[![Tests](https://img.shields.io/badge/Tests-68%2F68%20Passing-brightgreen.svg)](tests/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Executive Summary

The **AI Data Engineering Copilot** is an enterprise-grade AI system designed for modern data engineering teams. While traditional RAG applications are limited to passive document lookup, this Copilot bridges **technical documentation knowledge** with **active operational access to production databases and pipelines** through the **Model Context Protocol (MCP)** and an **autonomous multi-step reasoning agent**.

It runs locally on **Qwen3 8B** (or cloud via **Groq LPU**) and features a complete production software architecture with enterprise security boundaries, reciprocal rank fusion, cross-encoder reranking, semantic hallucination checks, query caching, and real-time observability.

---

## 🏛️ End-to-End System Architecture

```text
                                  User Request
                                       │
                                       ▼
                       FastAPI Server (HTTP & UI)
                                       │
                      ┌────────────────┴────────────────┐
                      ▼                                 ▼
             Semantic Cache Lookup             Request Tracking (UUID)
               /            \
            HIT              MISS
             │                │
             │                ▼
             │         Intelligent LLM Router (Qwen3 8B / Groq)
             │        /       │                \
             │       /        │                 \
             │   RAG Route  Tool Route      Refusal Route
             │     │          │                   │
             │     │          ▼                   ▼
             │     │   Agent Controller    Domain Boundary Guard
             │     │   (Max 5 Steps)
             │     │          │
             │     │          ▼
             │     │   MCP JSON-RPC Client
             │     │          │
             │     │          ▼
             │     │   MCP Server & Tools
             │     │   ├── get_table_schema
             │     │   ├── get_table_row_count
             │     │   └── get_table_sample
             │     │          │
             │     │          ▼
             │     │     SQLite Database (Allowlisted & Sanitized)
             │     │
             │     ▼
             │  Hybrid Retrieval Engine
             │  ├── Dense Semantic (BGE Embeddings + ChromaDB)
             │  └── Sparse Lexical (BM25)
             │          │
             │          ▼
             │     Reciprocal Rank Fusion (RRF)
             │          │
             │          ▼
             │  Cross-Encoder Reranker (BAAI/bge-reranker-base)
             │          │
             │          ▼
             │  Faithfulness & Citation Grounding (DeBERTa NLI)
             │
             ▼          ▼
     Telemetry & Observability (Metrics, Latency, Audit Logs)
                        │
                        ▼
           Modern Glassmorphic Web Dashboard / REST API
```

---

## 🚀 Key Technical Innovations

### 1. Autonomous Multi-Step Database Agent
* **Sequential Loop Controller:** Deconstructs complex user requests (e.g., *"Analyze the employees table, show me how many records exist and sample records"*) into distinct observation-action-thought steps.
* **Infinite Loop Safeguards:** Has a hard execution boundary (`MAX_AGENT_STEPS = 5`) and runtime signature hashing that detects and aborts duplicate tool invocations immediately.
* **Provenance Timeline:** Captures and displays every step's action, tool signature, and latency for full explainability.

### 2. Model Context Protocol (MCP) Client & Server
* **Standardized JSON-RPC 2.0:** Completely separates the AI reasoning engine from tool execution via standard MCP protocol envelopes (`tools/list`, `tools/call`).
* **Tool Registry:** Houses safe operational database tools:
  * `get_table_schema`: Inspects column definitions, types, and primary keys.
  * `get_table_row_count`: Retrieves exact table cardinality.
  * `get_table_sample`: Retrieves bounded record samples (`limit <= 50`).
* **Enterprise Security & SQL Injection Shield:** Strict table allowlist (`employees`, `orders`, `customers`) and regex identifier validation (`^[a-zA-Z0-9_]+$`) blocking comment dashes, semicolons, and UNION attacks.

### 3. State-of-the-Art Hybrid RAG Pipeline
* **Dense + Sparse Fusion:** Combines **BAAI/bge-base-en-v1.5** embeddings in ChromaDB with **BM25** lexical scoring via **Reciprocal Rank Fusion (RRF)**.
* **Cross-Encoder Reranking:** Applies **BAAI/bge-reranker-base** to score the top candidates for optimal passage precision.
* **Semantic Faithfulness & Grounding:** Natural Language Inference (NLI) with **DeBERTa-v3** verifies claims against source chunks, eliminating hallucinations.

### 4. Production Observability & High-Performance Caching
* **Semantic & Exact Query Cache:** Normalized question caching reduces repeat query latency from ~25s down to **< 10ms** (recorded in metrics).
* **Distributed Request Tracking:** Generates unique hexadecimal `request_id` hashes propagated across all logs.
* **Telemetry & Metrics Endpoint:** Exposes live Prometheus-compatible metrics (`/metrics`) covering cache hit rates, average tool latency, agent step counts, and error rates.

---

## 📊 Architecture Components & Technical Specifications

| Subsystem | Technology Stack | Capabilities & Guarantees |
| :--- | :--- | :--- |
| **Agent Controller** | Python 3.11+, Qwen3 8B, Pydantic | Multi-step sequential reasoning, loop detection aborts, `MAX_STEPS = 5` boundary |
| **Protocol Layer** | Model Context Protocol (MCP) | JSON-RPC 2.0 client/server decoupling, dynamic tool discovery (`tools/list`, `tools/call`) |
| **Operational Tools** | SQLite, Parameterized SQL | Column schema discovery, exact row counts, bounded sampling with identifier allowlists |
| **Hybrid Retrieval** | BGE-base + BM25 + ChromaDB | Dense vector search fused with lexical BM25 using Reciprocal Rank Fusion (RRF) |
| **Reranking Engine** | BAAI/bge-reranker-base | Cross-Encoder scoring of top-k passages for high-precision context selection |
| **Hallucination Shield** | DeBERTa-v3 NLI | Natural language inference verifying citation claims against source documents |
| **Semantic Caching** | SQLite & In-Memory LRU | Normalized query hashing delivering sub-10ms response times on repeat queries |
| **API & Dashboard** | FastAPI, Uvicorn, Vanilla CSS | REST API, Prometheus-compatible `/metrics`, and custom dark-mode glassmorphic UI |

---

## 💻 Interactive Web Dashboard

The Copilot serves a custom, responsive, dark-mode glassmorphic dashboard directly from FastAPI at `/`:
* **1-Click Query Presets:** Multi-step database investigations, RAG architecture questions, security defense demos, and refusal guardrails.
* **Live Execution Timer:** Real-time feedback while multi-step local reasoning is underway.
* **Interactive Step Timeline:** Expandable accordion showing exact agent reasoning chains, tool parameters, and raw JSON observations.
* **Sources Grid:** Clickable citation provenance linking back to indexed documentation chunks.
* **Telemetry Modal:** Live dashboard displaying runtime latency, MCP tool health, and cache hit ratios.

---

## 🛠️ Quickstart Guide

### 1. Prerequisites
* Python 3.11+
* [Ollama](https://ollama.ai/) with `qwen3:8b`:
  ```bash
  ollama pull qwen3:8b
  ```
  *(Or use free [Groq Cloud API](https://console.groq.com) by setting `export GROQ_API_KEY="gsk_..."`)*

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/Arshdeep030/ai-data-engineering-copilot.git
cd ai-data-engineering-copilot

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the Application
```bash
python app.py
```
Open your browser and navigate to **[http://127.0.0.1:8000](http://127.0.0.1:8000)**.

### 4. Run the Full Test Suite
```bash
python -m unittest discover tests
```
```text
Ran 68 tests in 0.025s
OK
```
All **68 unit tests** pass with 100% success rate across agent controllers, MCP client/server protocols, security allowlists, and hybrid RAG caching.

---

## 🌐 Public Demo via Cloudflare Tunnel (100% Free)

To share a live public link during an interview or demo without paying for cloud hosting:
```bash
# Install Cloudflare Tunnel
brew install cloudflared

# Expose your running local server
cloudflared tunnel --url http://127.0.0.1:8000
```
Cloudflare will output a public HTTPS link (e.g. `https://xxxx.trycloudflare.com`) that anyone can open to interact with your Copilot live!

---

## 📡 API Reference

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| **`/`** | `GET` | Interactive Glassmorphic Web Dashboard |
| **`/query`** | `POST` | Core Copilot query pipeline (Agent + RAG + Cache) |
| **`/health`** | `GET` | Service health check and warmed component status |
| **`/metrics`** | `GET` | Runtime latency, MCP tool health, and cache telemetry |
| **`/docs`** | `GET` | Interactive Swagger API documentation |

### Example Query Request:
```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Analyze the employees table and tell me what columns it has and how many records exist."}'
```

### Example JSON Response:
```json
{
  "question": "Analyze the employees table and tell me what columns it has and how many records exist.",
  "answer": "The employees table contains 5 columns: id (INTEGER), name (TEXT), department (TEXT), salary (REAL), and hire_date (TEXT). It currently contains 8 total records.",
  "sources": [],
  "latency_seconds": 1.42,
  "cache_hit": false,
  "request_id": "9b3d755e6b40",
  "route": "tool",
  "tool_name": "get_table_schema",
  "agent_steps": [
    {
      "step_number": 1,
      "action": "tool",
      "tool_name": "get_table_schema",
      "thought": "Need to retrieve schema for column definitions.",
      "observation_summary": "{\"columns\": [\"id\", \"name\", \"department\", \"salary\", \"hire_date\"]}",
      "latency_seconds": 0.45
    },
    {
      "step_number": 2,
      "action": "tool",
      "tool_name": "get_table_row_count",
      "thought": "Now retrieve record count.",
      "observation_summary": "{\"table\": \"employees\", \"row_count\": 8}",
      "latency_seconds": 0.42
    },
    {
      "step_number": 3,
      "action": "final",
      "tool_name": null,
      "thought": "Information gathered, synthesizing response.",
      "observation_summary": null,
      "latency_seconds": 0.55
    }
  ]
}
```

---

## 🔒 Security & Defense-in-Depth

* **Identifier Sanitation:** Enforces strict regex validation on all table parameters (`^[a-zA-Z0-9_]+$`).
* **Table Allowlisting:** Blocks unauthorized access to system catalogs or unapproved tables (`sqlite_master`, `passwords`, `salaries`), returning structured `AccessDenied` errors.
* **SQL Injection Defense:** All row count and sample queries use parameterized SQL execution (`LIMIT ?`) rather than string concatenation.
* **Domain Guardrails:** Automatically identifies out-of-scope questions and returns structured refusals.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

Developed with passion by **Arshdeep Singh**.
