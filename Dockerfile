FROM ghcr.io/facebookresearch/openenv/openenv-base:latest

WORKDIR /app

# Install dependencies
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-install-project || true

# Copy application
COPY . .

# Environment variables for HF Spaces
ENV HOST=0.0.0.0
ENV PORT=8000
ENV HF_SPACE=1
ENV OPENAI_API_KEY=${OPENAI_API_KEY:-}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/tasks')" || exit 1

EXPOSE 8000

# Run with uvicorn
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
