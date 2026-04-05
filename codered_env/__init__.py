"""CodeRedEnv — Emergency Medical Coordination Environment."""

from server.models import (
    CodeRedAction,
    CodeRedObservation,
    CodeRedState,
)
from client import CodeRedEnv

try:
    from inference import run_agent
    run_baseline_agent = run_agent  # Alias for backward compat
except ImportError:
    run_agent = None
    run_baseline_agent = None

__all__ = [
    "CodeRedAction",
    "CodeRedObservation",
    "CodeRedState",
    "CodeRedEnv",
    "run_agent",
    "run_baseline_agent",
]
