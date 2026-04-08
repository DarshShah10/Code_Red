# ══════════════════════════════════════════════════════════════════════════════
#  CodeRedEnv — Emergency Medical Coordination Simulation
#  Build:   docker build -t codered-env .
#  Run:     docker run -p 8000:8000 --env-file .env codered-env
#  HF Spaces: push to darshshah1012/coderedenv (auto-builds from Dockerfile)
# ══════════════════════════════════════════════════════════════════════════════

FROM python:3.11-slim

WORKDIR /app

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Install uv ────────────────────────────────────────────────────────────────
RUN pip install --no-cache-dir uv

# ── Copy source ────────────────────────────────────────────────────────────────
COPY . .

# ── Install dependencies via pyproject.toml ───────────────────────────────────
RUN uv pip install --system --no-cache \
    "openenv-core[core]>=0.2.2" \
    "fastapi>=0.109.0" \
    "uvicorn[standard]>=0.27.0" \
    "pydantic>=2.5.0" \
    "openai>=1.56.0" \
    "httpx>=0.27.0" \
    "networkx>=3.0" \
    "python-dotenv>=1.0.0" \
    "anthropic>=0.40.0"

# ── Install this package in editable mode ─────────────────────────────────────
RUN uv pip install --system --no-cache -e .

# ── Runtime config (HF_SPACE=1 enables OpenEnv special behaviors) ─────────────
ENV PYTHONUNBUFFERED=1
ENV HF_SPACE=1
ENV PYTHONPATH=/app

# ── Health check ───────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# ── Run server (app.py contains uvicorn.run in main()) ────────────────────────
CMD ["python", "-m", "server.app"]
