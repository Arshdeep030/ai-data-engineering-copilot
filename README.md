---
title: AI Data Engineering Copilot
emoji: ⚡
colorFrom: indigo
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
---

# ⚡ AI Data Engineering Copilot

> An enterprise-grade, agentic AI Copilot for Data Engineering architectures, technical documentation retrieval, and secure multi-step database investigations over **Model Context Protocol (MCP)**.

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141-009688.svg)](https://fastapi.tiangolo.com/)
[![MCP](https://img.shields.io/badge/Protocol-MCP%20JSON--RPC%202.0-indigo.svg)](https://modelcontextprotocol.io/)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-orange.svg)](https://www.trychroma.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🏛️ System Architecture

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
             │         LLM Router (Qwen3 / Llama 3.1)
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
             │     │     SQLite Database
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
                 JSON API / Modern Web UI
```

---

## ✨ Key Capabilities

1. **Autonomous Multi-Step Database Investigations**
   - Implements sequential agent reasoning loop with strict safety bounds (`MAX_AGENT_STEPS = 5`).
   - Loop detection safeguard automatically aborts duplicate consecutive tool signatures.
   - Model Context Protocol (MCP) tool execution for schema inspection, row counts, and data sampling.

2. **State-of-the-Art Hybrid RAG Pipeline**
   - Dense vector search with **BAAI/bge-small-en-v1.5** embeddings.
   - Lexical search with **BM25**.
   - **Reciprocal Rank Fusion (RRF)** combines dense and sparse candidate pools.
   - **Cross-Encoder Reranker** (`BAAI/bge-reranker-base`) scores context candidates with high precision.
   - Hallucination prevention and NLI verification with **DeBERTa-v3**.

3. **Enterprise Security & Guardrails**
   - Strict SQL injection prevention (identifier allowlisting, regex validation, parameterized queries).
   - Strict table allowlist (`employees`, `orders`, `customers`). Disallowed tables reject immediately with `AccessDenied`.
   - Out-of-domain query refusal mechanism (e.g. general chat or competitor platforms).

4. **Production Observability & UI**
   - Glassmorphic dark-mode web dashboard served directly from FastAPI at `/`.
   - Real-time **Agent Execution Timeline** displaying thoughts, tool calls, and observations.
   - Live latency tracking, cache hit indicators, and Prometheus-compatible metrics at `/metrics`.

---

## 🚀 Running Locally

### 1. Prerequisites
- Python 3.11+
- [Ollama](https://ollama.ai/) with `qwen3:8b`:
  ```bash
  ollama pull qwen3:8b
  ```

### 2. Setup & Installation
```bash
# Clone the repository
git clone https://github.com/Arshdeep030/ai-data-engineering-copilot.git
cd ai-data-engineering-copilot

# Create virtual environment & install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Launch the Server
```bash
uvicorn src.app:app --host 127.0.0.1 --port 8000
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser.

### 4. Run Unit Tests
```bash
python -m unittest discover tests
```
*(All 68 unit tests pass in < 0.1s).*

---

## 🌐 100% Free 24/7 Cloud Deployment (Hugging Face Spaces)

This repository is pre-configured for **Hugging Face Spaces** free tier (2 vCPU, 16GB RAM) using the **Gradio SDK** (100% Free, no credit card or subscription needed) paired with **Groq's Free API** (ultra-fast 500 tokens/sec, $0 cost).

### Step 1: Get a Free Groq API Key
1. Go to [console.groq.com/keys](https://console.groq.com/keys) (No credit card required).
2. Create an account and click **Create API Key**.
3. Copy your key (starts with `gsk_...`).

### Step 2: Create a Hugging Face Space (100% Free)
1. Go to [huggingface.co/new-space](https://huggingface.co/new-space).
2. Name your space: `ai-data-engineering-copilot`.
3. Select **Gradio** as the Space SDK (*Notice: Do NOT select Docker as HF made Docker paid; Gradio is 100% FREE*).
4. Select the **Free CPU (2 vCPU • 16 GB RAM)** hardware tier.
5. Click **Create Space**.

### Step 3: Add Your Free API Key as a Secret
1. Inside your new Space, click **Settings** (top right).
2. Scroll to **Variables and secrets** -> **New secret**.
3. Name: `GROQ_API_KEY`
4. Value: `gsk_your_groq_api_key_here`
5. Click **Save**.

### Step 4: Push the Code
Add your Space as a git remote and push:
```bash
git remote add space https://huggingface.co/spaces/<YOUR_HF_USERNAME>/ai-data-engineering-copilot
git push space main
```
Hugging Face will automatically install dependencies from `requirements.txt` and launch your live application at:
`https://huggingface.co/spaces/<YOUR_HF_USERNAME>/ai-data-engineering-copilot`!

---

## 🧪 API Endpoints

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/` | `GET` | Interactive Web Dashboard |
| `/query` | `POST` | Primary Copilot query endpoint (Agent + RAG + Cache) |
| `/health` | `GET` | Service health status and warmed component check |
| `/metrics` | `GET` | Runtime observability, latency, and MCP tool metrics |
| `/docs` | `GET` | Interactive Swagger API documentation |

---

## 📄 License
MIT License. Created by Arshdeep Singh.
