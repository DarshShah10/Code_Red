"""CodeRedEnv EnvClient — thin wrapper for remote HF Space environments."""
from openenv.core.env_server.interfaces import Environment


class CodeRedEnv(Environment):
    """Placeholder for remote EnvClient. Used by inference.py when running against a deployed HF Space."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, base_url: str | None = None):
        self._base_url = base_url or "http://localhost:8000"
        self._session_id: str | None = None
        self._client = None  # lazily initialized

    def _get_client(self):
        if self._client is None:
            import httpx
            self._client = httpx.Client(base_url=self._base_url, timeout=60.0)
        return self._client

    def reset(self, *, seed: int | None = None, task_id: str = "task1", **kwargs):
        import httpx
        client = self._get_client()
        payload = {"seed": seed, "task_id": task_id}
        payload.update(kwargs)
        resp = client.post("/reset", json=payload)
        resp.raise_for_status()
        data = resp.json()
        self._session_id = data.get("session_id")
        return data

    def step(self, action, timeout_s: float | None = None, **kwargs):
        import httpx
        client = self._get_client()
        payload = {"action": action, **kwargs}
        resp = client.post("/step", json=payload, timeout=timeout_s)
        resp.raise_for_status()
        return resp.json()

    def state(self):
        import httpx
        client = self._get_client()
        resp = client.get("/state")
        resp.raise_for_status()
        return resp.json()

    def close(self):
        if self._client:
            self._client.close()
            self._client = None

    @classmethod
    def from_docker_image(cls, image_name: str | None = None, **kwargs):
        raise NotImplementedError(
            "CodeRedEnv.from_docker_image is not yet implemented. "
            "Use CodeRedEnv(base_url='http://<space-url>') to connect to a deployed HF Space."
        )
