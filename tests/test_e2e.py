"""End-to-end integration tests for all 3 tasks."""

from server.codered_environment import CodeRedEnvironment
from server.models.actions import MaintainPlan
from server.grader import grade_from_environment


def test_task1_e2e():
    """Task 1: single cardiac patient, no disruptions."""
    env = CodeRedEnvironment()
    obs = env.reset(seed=42, task_id="task1")
    assert obs.step == 1
    assert len(obs.patients) == 1
    assert obs.patients[0].condition.value == "cardiac"
    assert obs.patients[0].vitals_score == 1.0  # Phase 1: starts at 1.0

    # Take 10 steps with maintain_plan dummy
    for _ in range(10):
        obs = env.step(MaintainPlan())
        if env.state.step_count >= env.state.max_steps:
            break
    assert env.state.step_count > 0


def test_task2_e2e():
    """Task 2: multi-patient with disruptions."""
    env = CodeRedEnvironment()
    obs = env.reset(seed=0, task_id="task2")
    assert obs.step == 1
    assert len(obs.patients) >= 2  # cardiac + stroke

    for _ in range(10):
        obs = env.step(MaintainPlan())
        if env.state.step_count >= env.state.max_steps:
            break
    assert env.state.step_count > 0


def test_task3_e2e():
    """Task 3: crisis surge, 5 patients, mutual aid available."""
    env = CodeRedEnvironment()
    obs = env.reset(seed=7, task_id="task3")
    assert obs.step == 1
    assert len(obs.patients) >= 5

    for _ in range(15):
        obs = env.step(MaintainPlan())
        if env.state.step_count >= env.state.max_steps:
            break
    assert env.state.step_count > 0


def test_all_tasks_reset_and_step():
    """All 3 tasks reset and step without crashing."""
    for task_id in ["task1", "task2", "task3"]:
        env = CodeRedEnvironment()
        obs = env.reset(seed=42, task_id=task_id)
        assert obs.step == 1
        # One step should not crash
        obs = env.step(MaintainPlan())
        assert env.state.step_count == 1


def test_grader_computes_score():
    """Grader produces valid scores for all tasks."""
    for task_id in ["task1", "task2", "task3"]:
        env = CodeRedEnvironment()
        env.reset(seed=0, task_id=task_id)
        for _ in range(10):
            env.step(MaintainPlan())
            if env.state.step_count >= env.state.max_steps:
                break
        result = grade_from_environment(env)
        assert 0.0 <= result.final_score <= 1.0
        assert 0.0 <= result.time_score <= 1.0
        assert 0.0 <= result.efficiency <= 1.0
        assert 0.0 <= result.secondary_harm <= 1.0
        assert 0.0 <= result.prep_ready <= 1.0
        assert 0.0 <= result.mutual_aid_penalty <= 1.0
        assert 0.0 <= result.vitals_score_avg <= 1.0  # Phase 1
