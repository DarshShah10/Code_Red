"""Test that done flag flows correctly through step() for HF Space version."""
import pytest
from server.codered_environment import CodeRedEnvironment


def test_step_returns_done_true_when_terminated():
    """When _check_done() returns True, step() must pass done=True to observation."""
    env = CodeRedEnvironment()
    obs = env.reset(seed=0, task_id="task1")

    # Run to termination (step1 cardiac patient: 30 steps or all treated)
    for _ in range(31):
        from server.models.actions import MaintainPlan
        obs = env.step(MaintainPlan())
        if env._check_done():
            # CRITICAL: done field in observation must be True
            assert obs.done is True, (
                f"Expected obs.done=True when episode terminates, got obs.done={obs.done}"
            )
            break
    else:
        pytest.fail("Episode did not terminate after 31 steps")


def test_reset_returns_done_false():
    """reset() should always return observation with done=False."""
    env = CodeRedEnvironment()
    obs = env.reset(seed=0, task_id="task1")
    assert obs.done is False, "reset() must return done=False"


def test_build_observation_accepts_done_parameter():
    """_build_observation() must accept a done kwarg."""
    env = CodeRedEnvironment()
    env.reset(seed=0, task_id="task1")
    # Must not raise TypeError
    obs = env._build_observation(done=True)
    assert obs.done is True
    obs2 = env._build_observation(done=False)
    assert obs2.done is False
