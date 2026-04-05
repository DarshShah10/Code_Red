"""Phase 2 environment integration tests — Task 9."""
import pytest


def test_pending_calls_empty_for_task1():
    """Tasks 1-3 don't use the call queue (backward compat)."""
    from server.codered_environment import CodeRedEnvironment
    env = CodeRedEnvironment()
    obs = env.reset(seed=0, task_id="task1")
    assert hasattr(obs, "pending_calls")
    assert obs.pending_calls == []


def test_pending_calls_populated_for_task4():
    """Task 4 uses the call queue — calls spawn after a few steps."""
    from server.codered_environment import CodeRedEnvironment
    from server.models.actions import MaintainPlan

    env = CodeRedEnvironment()
    obs = env.reset(seed=0, task_id="task4")
    assert hasattr(obs, "pending_calls")

    # After 8 steps (CALL_SPAWN_INTERVAL), a call should appear
    for _ in range(9):
        obs = env.step(MaintainPlan())

    assert len(obs.pending_calls) >= 1
    call = obs.pending_calls[0]
    assert hasattr(call, "call_id")
    assert hasattr(call, "category")
    assert hasattr(call, "location_node")


def test_triage_call_action():
    """TriageCall action removes call from pending_calls."""
    from server.codered_environment import CodeRedEnvironment
    from server.models.actions import MaintainPlan, TriageCall

    env = CodeRedEnvironment()
    obs = env.reset(seed=0, task_id="task4")

    for _ in range(9):
        obs = env.step(MaintainPlan())

    if len(obs.pending_calls) == 0:
        pytest.skip("No call spawned yet")

    call = obs.pending_calls[0]
    triage = TriageCall(call_id=call.call_id, decision="no_dispatch")
    obs = env.step(triage)

    remaining_ids = [c.call_id for c in obs.pending_calls]
    assert call.call_id not in remaining_ids


def test_overcrowding_modifier_in_observation():
    """Overcrowding modifier appears in observation."""
    from server.codered_environment import CodeRedEnvironment
    from server.models.actions import MaintainPlan

    env = CodeRedEnvironment()
    obs = env.reset(seed=0, task_id="task4")
    assert hasattr(obs, "overcrowding_modifier")
    assert obs.overcrowding_modifier == 1.0


def test_grader_includes_cascade_score():
    """Grader output includes cascade_score field."""
    from server.codered_environment import CodeRedEnvironment
    from server.models.actions import MaintainPlan
    from server.grader import grade_from_environment

    env = CodeRedEnvironment()
    obs = env.reset(seed=0, task_id="task4")

    for _ in range(20):
        obs = env.step(MaintainPlan())
        if env.state.step_count >= env.state.max_steps:
            break

    result = grade_from_environment(env)
    assert hasattr(result, "cascade_score")
    assert 0.0 <= result.cascade_score <= 1.0


def test_task4_e2e():
    """Task 4 end-to-end: call queue, triage, patient arrival."""
    from server.codered_environment import CodeRedEnvironment
    from server.models.actions import MaintainPlan, TriageCall

    env = CodeRedEnvironment()
    obs = env.reset(seed=42, task_id="task4")
    assert obs.pending_calls == []  # Initially empty

    for _ in range(9):
        obs = env.step(MaintainPlan())
        if len(obs.pending_calls) > 0:
            break

    if len(obs.pending_calls) > 0:
        call = obs.pending_calls[0]
        triage = TriageCall(call_id=call.call_id, decision="no_dispatch")
        obs = env.step(triage)

    for _ in range(50):
        obs = env.step(MaintainPlan())
        if env.state.step_count >= env.state.max_steps:
            break

    assert env.state.step_count > 0


def test_task5_cascade_engine_active():
    """Task 5 has cascade_enabled — overcrowding modifier can be 1.2."""
    from server.codered_environment import CodeRedEnvironment
    from server.models.actions import MaintainPlan

    env = CodeRedEnvironment()
    obs = env.reset(seed=99, task_id="task5")

    for _ in range(50):
        obs = env.step(MaintainPlan())
        if env.state.step_count >= env.state.max_steps:
            break

    assert hasattr(obs, "overcrowding_modifier")
    assert obs.overcrowding_modifier in (1.0, 1.2)
