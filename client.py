"""CodeRedEnv — OpenEnv-compliant async environment client."""

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from server.models import CodeRedAction, CodeRedObservation, CodeRedState


class CodeRedEnv(EnvClient[CodeRedAction, CodeRedObservation, CodeRedState]):
    """
    Async client for CodeRedEnv.

    Inherits from_docker_image() and from_env() from EnvClient.

    Usage with Docker:
        async with CodeRedEnv.from_docker_image("codered-env:latest") as env:
            result = await env.reset(task_id="task1")
            ...

    Usage with HF Space:
        async with CodeRedEnv.from_env("darshshah1012/coderedenv") as env:
            result = await env.reset(task_id="task1")
            ...

    Usage with direct HTTP (no Docker):
        env = CodeRedEnv(base_url="http://localhost:8000")
        await env.connect()
        result = await env.reset(task_id="task1")
    """

    def _step_payload(self, action: CodeRedAction) -> Dict[str, Any]:
        """
        Convert CodeRedAction to JSON payload for the server.

        CodeRedAction is a discriminated union of action variants.
        Pydantic's model_dump() produces the right format.
        """
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[CodeRedObservation]:
        """
        Parse server response into StepResult[CodeRedObservation].

        The server returns:
        {
            "observation": {...},   # CodeRedObservation data
            "reward": float,
            "done": bool
        }
        """
        obs_data = payload.get("observation", {})
        observation = CodeRedObservation(**obs_data)
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> CodeRedState:
        """Parse server state response into CodeRedState."""
        return CodeRedState(**payload)
