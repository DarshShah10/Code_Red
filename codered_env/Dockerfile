# Dockerfile for CodeRedEnv
# Build from repo root:
#   docker build -t codered-env . -f Dockerfile

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
RUN pip install --no-cache-dir uv

# Copy dependency spec first (better layer caching)
COPY pyproject.toml ./

# Install openenv-core and all dependencies via uv (no --frozen for cross-platform)
RUN uv pip install --system --no-cache \
    "openenv-core[core]>=0.2.2" \
    "fastapi>=0.109.0" \
    "uvicorn[standard]>=0.27.0" \
    "pydantic>=2.5.0" \
    "openai>=1.56.0" \
    "httpx>=0.27.0" \
    "networkx>=3.0" \
    "python-dotenv>=1.0.0"

# Copy application source
COPY . .

# Install this package in editable mode
RUN uv pip install --system --no-cache -e .

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV HF_SPACE=1

# Health check: /tasks is lightweight and always returns 200 when server is up
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || curl -f http://localhost:8000/tasks || exit 1

EXPOSE 8000

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
