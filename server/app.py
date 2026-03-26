"""FastAPI application for CodeRedEnv (minimal scaffold)."""

try:
    from openenv.core.env_server.http_server import create_app
except ImportError as e:
    raise ImportError("openenv is required. Install with: uv sync") from e

# Will be updated in Task 4
app = create_app(
    None,  # Placeholder — CodeRedEnvironment
    None,  # Placeholder — CodeRedAction
    None,  # Placeholder — CodeRedObservation
    env_name="codered_env",
    max_concurrent_envs=4,
)
