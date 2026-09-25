"""
AI Data Engineering Copilot - Application Entrypoint
Launches the FastAPI server with the interactive Web Dashboard, REST API routes, and Swagger docs.
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

from src.app import app

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "127.0.0.1")
    print(f"\n🚀 Launching AI Data Engineering Copilot at http://{host}:{port}")
    uvicorn.run("src.app:app", host=host, port=port, reload=False)
