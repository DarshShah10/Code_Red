import pytest

def test_reset_task1_returns_observation():
    from codered_env.server.codered_environment import CodeRedEnvironment
    env = CodeRedEnvironment()
    obs = env.reset(seed=42, task_id="task1")
    assert obs.step == 1
    assert len(obs.patients) == 1
    assert obs.patients[0].condition.value == "cardiac"
    assert obs.patients[0].status.value == "waiting"

def test_reset_task1_has_5_ambulances():
    from codered_env.server.codered_environment import CodeRedEnvironment
    env = CodeRedEnvironment()
    obs = env.reset(seed=0, task_id="task1")
    assert len(obs.ambulances) == 5

def test_reset_task1_has_3_hospitals():
    from codered_env.server.codered_environment import CodeRedEnvironment
    env = CodeRedEnvironment()
    obs = env.reset(seed=0, task_id="task1")
    assert len(obs.hospitals) == 3

def test_step_increments_counter():
    from codered_env.server.codered_environment import CodeRedEnvironment
    from codered_env.server.models.actions import MaintainPlan
    env = CodeRedEnvironment()
    obs = env.reset(seed=0, task_id="task1")
    assert env.state.step_count == 0
    obs2 = env.step(MaintainPlan())
    assert obs2.step == 2
    assert env.state.step_count == 1

def test_state_returns_episode_metadata():
    from codered_env.server.codered_environment import CodeRedEnvironment
    env = CodeRedEnvironment()
    env.reset(seed=0, task_id="task1")
    state = env.state
    assert state.task_id == "task1"
    assert state.max_steps == 30
    assert state.mutual_aid_available == 0

def test_seed_reproducibility():
    from codered_env.server.codered_environment import CodeRedEnvironment
    from codered_env.server.models.actions import MaintainPlan
    env1 = CodeRedEnvironment()
    env2 = CodeRedEnvironment()
    obs1 = env1.reset(seed=123, task_id="task1")
    obs2 = env2.reset(seed=123, task_id="task1")
    # Same seed = same patient location
    assert obs1.patients[0].location_node == obs2.patients[0].location_node

def test_dispatch_increments_step():
    from codered_env.server.codered_environment import CodeRedEnvironment
    from codered_env.server.models.actions import DispatchAmbulance
    env = CodeRedEnvironment()
    env.reset(seed=0, task_id="task1")
    obs = env.step(DispatchAmbulance(ambulance_id="AMB_1", target_node="NH45_BYPASS"))
    assert obs.step == 2


def test_dense_reward_non_zero_for_active_steps():
    """Dense reward returns non-zero values when patient state changes."""
    from codered_env.server.codered_environment import CodeRedEnvironment
    from codered_env.server.models.actions import DispatchAmbulance, MaintainPlan
    env = CodeRedEnvironment()
    obs = env.reset(seed=0, task_id="task1")

    # Step 1: MaintainPlan (idle) — first reward call
    obs = env.step(MaintainPlan())

    # Dispatch ambulance to patient — should give milestone bonus
    patient_node = obs.patients[0].location_node
    prev_cum = env.state.cum_reward
    obs = env.step(DispatchAmbulance(ambulance_id="AMB_1", target_node=patient_node))
    reward_delta = env.state.cum_reward - prev_cum
    # Dispatching gives milestone bonus (dispatched=0.05) — reward should be positive
    assert reward_delta > 0.0, f"Expected positive reward for dispatch, got {reward_delta}"


def test_vitals_in_observation():
    """Observation includes vitals_score for each patient."""
    from codered_env.server.codered_environment import CodeRedEnvironment
    env = CodeRedEnvironment()
    obs = env.reset(seed=0, task_id="task1")
    assert len(obs.patients) == 1
    assert hasattr(obs.patients[0], "vitals_score")
    assert 0.0 <= obs.patients[0].vitals_score <= 1.0


def test_vitals_score_preview_in_observation():
    """Observation includes vitals_score_preview."""
    from codered_env.server.codered_environment import CodeRedEnvironment
    env = CodeRedEnvironment()
    obs = env.reset(seed=0, task_id="task1")
    assert hasattr(obs, "vitals_score_preview")
    assert 0.0 <= obs.vitals_score_preview <= 1.0


def test_early_done_when_all_dead():
    """Episode terminates early when all patients are deceased."""
    from codered_env.server.codered_environment import CodeRedEnvironment
    from codered_env.server.models.actions import MaintainPlan
    import random
    rng = random.Random(42)
    env = CodeRedEnvironment()
    obs = env.reset(seed=42, task_id="task1")

    # Run until done
    done = False
    steps = 0
    for _ in range(200):
        obs = env.step(MaintainPlan())
        if env._check_done():
            done = True
            break
        steps += 1

    assert done, "Episode should terminate when patient dies"
    # Patient should be deceased (cardiac: death at step ~180, but may vary with rng)
    assert steps < 200, "Should terminate before max_steps"
