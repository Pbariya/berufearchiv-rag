# Dockerfile
# Two-stage build:
#   Stage 1 (builder): installs C compilers + compiles heavy packages
#   Stage 2 (runtime): copies only compiled result — no build tools in final image

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install C build tools needed to compile faiss-cpu, numpy, tokenizers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first — Docker caches this layer
# If requirements.txt hasn't changed, pip install is skipped next build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy compiled packages from builder (no build tools in final image)
COPY --from=builder /install /usr/local

# Copy your application code
COPY rag_evaluation_berufearchiv.py .
COPY templates/ ./templates/

# Create empty mount points — real data comes via docker-compose volumes
# Faiss_Metadata/ and ckpts/ are too large to bake into the image
RUN mkdir -p Faiss_Metadata exports_v6 ckpts

ENV PORT=5000
ENV PYTHONUNBUFFERED=1
ENV USE_NGROK=false

# Health check — Docker uses this to confirm app started correctly
# start-period=90s gives time for FAISS + ML models to load
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c \
    "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')" \
    || exit 1

# Start Flask app
CMD ["python", "rag_evaluation_berufearchiv.py"]
