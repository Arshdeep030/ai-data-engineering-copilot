"""
Entrypoint for Hugging Face Spaces (Gradio SDK) and production server.
Mounts the FastAPI application with custom UI and full API functionality.
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

# Mount Gradio if running in Hugging Face Gradio Space environment
try:
    import gradio as gr
    with gr.Blocks(title="AI Data Engineering Copilot") as demo:
        gr.Markdown(
            "### ⚡ AI Data Engineering Copilot is Running!\n"
            "Open the [Dashboard](/) directly to interact with the full UI."
        )
    app = gr.mount_gradio_app(app, demo, path="/gradio")
except Exception:
    pass

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)
