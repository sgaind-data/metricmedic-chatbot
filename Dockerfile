# syntax=docker/dockerfile:1
# MetricMedic RAG API image.
# Stage 1 installs dependencies and downloads the embedding model.
# Stage 2 copies only what is needed to run, as a non-root user.

############################
# Stage 1: builder
############################
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf-cache

# Isolated virtual environment we can copy into the final image.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# CPU-only PyTorch first. The default PyPI build on x86_64 bundles CUDA (GPU)
# libraries worth several GB that this service never uses.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

# Bake the embedding model into the image so containers start without
# downloading anything from Hugging Face.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

############################
# Stage 2: runtime
############################
FROM python:3.12-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/opt/hf-cache \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    LOG_FORMAT=json \
    LOG_LEVEL=INFO

# Unprivileged user: if the app is ever compromised, it is not root.
RUN useradd --create-home --uid 10001 appuser

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/hf-cache /opt/hf-cache

WORKDIR /app
COPY metricmedic/ ./metricmedic/
COPY data/ ./data/

# Pre-compute the knowledge-base embedding cache at build time (no API key needed).
RUN python -c "from metricmedic.retriever import MetricMedicRetriever; MetricMedicRetriever()" \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# slim images have no curl, so the healthcheck uses Python's standard library.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "metricmedic.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
