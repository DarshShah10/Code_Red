# codered_env/tests/test_client.py
import pytest

from client import CodeRedEnv


def test_client_has_from_docker_image():
    """CodeRedEnv must have from_docker_image classmethod from EnvClient."""
    assert hasattr(CodeRedEnv, "from_docker_image")
    assert callable(CodeRedEnv.from_docker_image)


def test_client_has_from_env():
    """CodeRedEnv must have from_env classmethod for HF Spaces."""
    assert hasattr(CodeRedEnv, "from_env")
    assert callable(CodeRedEnv.from_env)


def test_client_inherits_from_env_client():
    """CodeRedEnv must inherit from openenv EnvClient."""
    from openenv.core import EnvClient
    assert issubclass(CodeRedEnv, EnvClient)


def test_client_type_params():
    """CodeRedEnv must be typed with Action, Observation, State."""
    from client import CodeRedEnv
    assert hasattr(CodeRedEnv, "_step_payload")
    assert hasattr(CodeRedEnv, "_parse_result")
    assert hasattr(CodeRedEnv, "_parse_state")
