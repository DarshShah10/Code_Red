"""State model for CodeRedEnv."""

from typing import List, Optional

from pydantic import Field

from openenv.core.env_server.types import State


class DisruptionState(State):
    """State of an active disruption."""
    disruption_type: str
    target: str
    remaining_steps: int


class CodeRedState(State):
    """Episode-level state for CodeRedEnv."""
    episode_id: Optional[str] = None
    step_count: int = Field(default=0, ge=0)
    task_id: str = "task1"
    cum_reward: float = 0.0
    max_steps: int = 30
    mutual_aid_used: int = 0
    mutual_aid_available: int = 0
    disruptions_active: List[DisruptionState] = Field(default_factory=list)
    all_patients_terminal: bool = False

    model_config = {"extra": "forbid"}
