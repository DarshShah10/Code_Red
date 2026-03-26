"""FastAPI application for CodeRedEnv."""

try:
    from openenv.core.env_server.http_server import create_app
except ImportError as e:
    raise ImportError(
        "openenv is required. Install with: uv sync"
    ) from e

from codered_env.server.codered_environment import CodeRedEnvironment
from codered_env.server.models import CodeRedAction, CodeRedObservation

app = create_app(
    CodeRedEnvironment,
    CodeRedAction,
    CodeRedObservation,
    env_name="codered_env",
    max_concurrent_envs=4,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
