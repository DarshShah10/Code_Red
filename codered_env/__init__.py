"""CodeRedEnv — Emergency Medical Coordination Environment."""

from .models import (
    CodeRedAction,
    CodeRedObservation,
    CodeRedState,
)
from .client import CodeRedEnv

try:
    from . import inference
    run_baseline_agent = inference.run_baseline_agent
except ImportError:
    run_baseline_agent = None

__all__ = [
    "CodeRedAction",
    "CodeRedObservation",
    "CodeRedState",
    "CodeRedEnv",
    "run_baseline_agent",
]
