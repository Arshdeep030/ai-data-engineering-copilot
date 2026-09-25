FROM python:3.11-slim

# Environment settings for Hugging Face Spaces & Python
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=7860 \
    HF_HOME=/tmp/huggingface \
    TRANSFORMERS_CACHE=/tmp/huggingface

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces uses UID 1000 for non-root containers
RUN useradd -m -u 1000 user
WORKDIR /app

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY --chown=user:user . .

# Set permissions for cache and database directories
RUN mkdir -p /tmp/huggingface /app/data && chown -R user:user /tmp/huggingface /app/data

USER user

EXPOSE 7860

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "7860"]
