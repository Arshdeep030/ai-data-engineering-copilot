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

from src.app import app, STATIC_DIR

# ZeroGPU compatibility decorator (satisfies ZeroGPU scanner if enabled)
try:
    import spaces

    @spaces.GPU
    def gpu_query_handler(question: str):
        from src.copilot_service import answer_question
        return answer_question(question)
except Exception:
    pass

import gradio as gr

# Read custom glassmorphic dark-mode dashboard
index_file = STATIC_DIR / "index.html"
ui_html = (
    index_file.read_text(encoding="utf-8")
    if index_file.exists()
    else "<h1>AI Data Engineering Copilot</h1>"
)

with gr.Blocks(title="AI Data Engineering Copilot", css="footer {display: none !important;}") as demo:
    gr.HTML(ui_html)

# Mount FastAPI app onto Gradio
app = gr.mount_gradio_app(app, demo, path="/gradio")

if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port)
