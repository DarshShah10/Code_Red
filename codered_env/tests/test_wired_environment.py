"""Integration tests for wired CodeRedEnvironment with subsystems."""

import pytest


def test_dispatch_routes_ambulance():
    from server.codered_environment import CodeRedEnvironment
    from server.models.actions import DispatchAmbulance
    env = CodeRedEnvironment()
    env.reset(seed=0, task_id="task1")
    obs = env.step(DispatchAmbulance(ambulance_id="AMB_1", target_node="NH45_BYPASS"))
    amb = next(a for a in obs.ambulances if a.id == "AMB_1")
    assert len(amb.route) > 0
    assert amb.eta_minutes > 0


def test_prepare_or_increments_prep_countdown():
    from server.codered_environment import CodeRedEnvironment
    from server.models.actions import PrepareOR
    env = CodeRedEnvironment()
    env.reset(seed=0, task_id="task1")
    obs = env.step(PrepareOR(hospital_id="HOSP_A", procedure_type="cardiac"))
    hosp = next(h for h in obs.hospitals if h.id == "HOSP_A")
    prep_or = next((o for o in hosp.operating_rooms if o.status.value == "prep"), None)
    assert prep_or is not None


def test_assign_hospital_sets_patient_destination():
    from server.codered_environment import CodeRedEnvironment
    from server.models.actions import AssignHospital
    env = CodeRedEnvironment()
    env.reset(seed=0, task_id="task1")
    obs = env.step(AssignHospital(patient_id="P1", hospital_id="HOSP_A"))
    patient = next(p for p in obs.patients if p.patient_id == "P1")
    assert patient.assigned_hospital == "HOSP_A"


def test_patient_cannot_be_assigned_to_unsuitable_hospital():
    from server.codered_environment import CodeRedEnvironment
    from server.models.actions import AssignHospital
    env = CodeRedEnvironment()
    env.reset(seed=0, task_id="task1")
    obs = env.step(AssignHospital(patient_id="P1", hospital_id="HOSP_C"))
    patient = next(p for p in obs.patients if p.patient_id == "P1")
    assert patient.assigned_hospital is None  # Assignment failed
    assert any("cannot treat" in a for a in obs.alerts)


def test_mutual_aid_not_available_task1():
    from server.codered_environment import CodeRedEnvironment
    from server.models.actions import RequestMutualAid
    env = CodeRedEnvironment()
    obs = env.reset(seed=0, task_id="task1")
    assert obs.mutual_aid_remaining == 0
    obs2 = env.step(RequestMutualAid())
    assert any("no calls remaining" in a for a in obs2.alerts)


def test_mutual_aid_assigns_patient_and_logs_arrival():
    """MA ambulance arrival should auto-assign highest-priority patient and log patient_id."""
    from server.codered_environment import CodeRedEnvironment
    from server.models.actions import RequestMutualAid, MaintainPlan
    env = CodeRedEnvironment()
    env.reset(seed=0, task_id="task2")  # task2 has 1 MA call
    obs = env.reset(seed=0, task_id="task2")
    assert obs.mutual_aid_remaining == 1

    # Call mutual aid
    obs = env.step(RequestMutualAid())
    assert obs.mutual_aid_remaining == 0

    # Episode log should have mutual_aid_called with patient_id
    log = env.get_episode_log()
    ma_called = next(e for e in log if e["event"] == "mutual_aid_called")
    assert "patient_id" in ma_called
    assert ma_called["patient_id"] is not None  # Patient assigned at call time
    assert "optimal_arrival_step" in ma_called
    assert ma_called["optimal_arrival_step"] > env._state.step_count

    # Fast-forward to MA arrival (compute how many steps needed)
    arrival_step = ma_called["optimal_arrival_step"]
    for _ in range(arrival_step - env._state.step_count):
        obs = env.step(MaintainPlan())

    # After arrival: episode log should have mutual_aid_arrived with patient_id
    log = env.get_episode_log()
    ma_arrived = next(e for e in log if e["event"] == "mutual_aid_arrived")
    assert "patient_id" in ma_arrived
    assert ma_arrived["patient_id"] == ma_called["patient_id"]
    assert "actual_arrival_step" in ma_arrived
    assert ma_arrived["actual_arrival_step"] == arrival_step


def test_blood_emergency_release():
    from server.codered_environment import CodeRedEnvironment
    from server.models.actions import AllocateBlood
    env = CodeRedEnvironment()
    env.reset(seed=0, task_id="task1")
    obs = env.step(AllocateBlood(
        hospital_id="HOSP_A", patient_id="P1",
        blood_type="O_NEG", units=2, emergency=True
    ))
    assert any("Emergency blood" in a for a in obs.alerts)


def test_timestep_increments_patient_time():
    from server.codered_environment import CodeRedEnvironment
    from server.models.actions import MaintainPlan
    env = CodeRedEnvironment()
    env.reset(seed=0, task_id="task1")
    patient = env._patients[0]
    # Subsystem patient tracks onset_step; time_since_onset is calculated in observation
    assert patient.onset_step == 0
    env.step(MaintainPlan())
    assert env._state.step_count == 1
    env.step(MaintainPlan())
    assert env._state.step_count == 2


def test_hospital_mortality_roll_hosp_c_emergent_always_dies():
    """
    HOSP_C cardiac mortality = 1.0 → patient always dies even after 'treatment'.
    HOSP_C has 0 ORs, so surgery can't start there. We test the mortality roll
    logic directly by patching self._rng to return 0.5 (below threshold=1.0).
    """
    from server.codered_environment import CodeRedEnvironment
    from server.models.actions import MaintainPlan

    env = CodeRedEnvironment()
    env.reset(seed=42, task_id="task1")

    patient = env._patient_manager.patients[0]
    assert patient.condition == "cardiac"

    # Use HOSP_A (has ORs and cardiologists) but set assigned_hospital to HOSP_C
    # for the mortality rate lookup. Override the RNG to force death (random() < 1.0).
    patient.assigned_hospital = "HOSP_C"
    patient.status = "in_treatment"

    hosp_a = env._hospital_system.get("HOSP_A")
    idle_or = next((o for o in hosp_a.operating_rooms if o.status == "idle"), None)
    assert idle_or is not None, "HOSP_A should have an idle OR"

    env._hospital_system.start_surgery(
        "HOSP_A", idle_or.index, "cardiac", patient.id, duration_minutes=30
    )
    env._active_surgeries[patient.id] = {
        "hospital_id": "HOSP_A",
        "or_index": idle_or.index,
        "start_step": env._state.step_count,
        "duration_minutes": 30,
    }

    # Override RNG so mortality roll always returns death (random() < 1.0 always)
    import random
    original_choice = env._rng.random
    env._rng.random = lambda: 0.5  # 0.5 < 1.0 → patient dies

    try:
        for _ in range(35):
            env.step(MaintainPlan())

        assert patient.outcome == "deceased", (
            f"Expected patient to die (mortality=1.0 for HOSP_C/cardiac), got outcome={patient.outcome}"
        )
    finally:
        env._rng.random = original_choice


def test_hospital_mortality_roll_survives_with_low_rate():
    """
    With HOSP_A cardiac (8% mortality) and rng forced above threshold, patient survives.
    """
    from server.codered_environment import CodeRedEnvironment
    from server.models.actions import MaintainPlan

    env = CodeRedEnvironment()
    env.reset(seed=42, task_id="task1")

    patient = env._patient_manager.patients[0]
    patient.assigned_hospital = "HOSP_A"
    patient.status = "in_treatment"

    hosp = env._hospital_system.get("HOSP_A")
    idle_or = next((o for o in hosp.operating_rooms if o.status == "idle"), None)
    assert idle_or is not None

    env._hospital_system.start_surgery(
        "HOSP_A", idle_or.index, "cardiac", patient.id, duration_minutes=30
    )
    env._active_surgeries[patient.id] = {
        "hospital_id": "HOSP_A",
        "or_index": idle_or.index,
        "start_step": env._state.step_count,
        "duration_minutes": 30,
    }

    # Override RNG so mortality roll always returns survival (random() >= 0.08)
    saved_random = env._rng.random
    env._rng.random = lambda: 0.5  # 0.5 >= 0.08 → patient survives

    try:
        for _ in range(35):
            env.step(MaintainPlan())

        assert patient.outcome == "saved", (
            f"Expected patient to survive (random=0.5 >= mortality=0.08), got outcome={patient.outcome}"
        )
    finally:
        env._rng.random = saved_random


def test_hospital_mortality_roll_seed_reproducibility():
    """
    Mortality roll is seeded — same seed → same survival outcome.
    """
    from server.codered_environment import CodeRedEnvironment
    from server.models.actions import MaintainPlan

    outcomes_seed42 = []
    outcomes_seed99 = []

    for seed in [42, 99]:
        for repeat in range(3):
            env = CodeRedEnvironment()
            env.reset(seed=seed, task_id="task1")
            patient = env._patient_manager.patients[0]
            patient.assigned_hospital = "HOSP_A"
            patient.status = "in_treatment"

            hosp = env._hospital_system.get("HOSP_A")
            idle_or = next((o for o in hosp.operating_rooms if o.status == "idle"), None)
            if idle_or:
                env._hospital_system.start_surgery("HOSP_A", idle_or.index, "cardiac", patient.id, duration_minutes=30)
                env._active_surgeries[patient.id] = {
                    "hospital_id": "HOSP_A",
                    "or_index": idle_or.index,
                    "start_step": env._state.step_count,
                    "duration_minutes": 30,
                }

            for _ in range(35):
                env.step(MaintainPlan())

            if seed == 42:
                outcomes_seed42.append(patient.outcome)
            else:
                outcomes_seed99.append(patient.outcome)

    # Same seed always produces the same outcome
    assert len(set(outcomes_seed42)) == 1, "HOSP_A mortality should be deterministic per seed"


def test_icu_bed_consumed_on_arrival():
    """
    consume_icu_bed decrements count and returns True when beds available.
    """
    from server.subsystems.hospital_system import HospitalSystem

    hs = HospitalSystem()
    hosp_a = hs.get("HOSP_A")
    initial = hosp_a.icu_beds["available"]
    assert initial > 0

    result = hs.consume_icu_bed("HOSP_A")
    assert result is True
    assert hosp_a.icu_beds["available"] == initial - 1

    # Release restores the bed
    hs.release_icu_bed("HOSP_A")
    assert hosp_a.icu_beds["available"] == initial


def test_icu_boarding_when_beds_exhausted():
    """
    When all ICU beds are consumed, consume_icu_bed returns False.
    Patient should get icu_status='boarding' in the environment.
    """
    from server.subsystems.hospital_system import HospitalSystem

    hs = HospitalSystem()
    hosp_a = hs.get("HOSP_A")

    # Exhaust all beds
    while hs.consume_icu_bed("HOSP_A"):
        pass
    assert hosp_a.icu_beds["available"] == 0

    result = hs.consume_icu_bed("HOSP_A")
    assert result is False


def test_grader_icu_boarding_penalty():
    """
    grade_from_environment applies ICU boarding penalty.
    """
    from server.grader import grade_from_environment

    class MockPM:
        patients = []

        def get_all(self):
            return self.patients

    class MockEnv:
        _episode_log = []
        _patient_manager = MockPM()

        def get_episode_log(self):
            return self._episode_log

    env = MockEnv()
    env._episode_log = [
        {"step": 0, "patient_id": "P1", "event": "patient_created", "condition": "cardiac"},
        {"step": 10, "patient_id": "P1", "event": "icu_boarding", "hospital_id": "HOSP_A"},
        {"step": 30, "patient_id": "P1", "event": "treatment_complete",
         "effective_time": 30, "target_time": 90, "vitals_at_treatment": 0.9},
    ]
    env._patient_manager.patients = []

    result = grade_from_environment(env)
    assert result.breakdown["icu_boarding_events"] == 1
    assert result.breakdown["icu_boarding_penalty"] == 0.05


def test_dispatch_als_pending_call_ground_truth_backfill():
    """
    Regression: when a pending outcome entry (true_condition=None) exists from a
    prior triage decision, and _spawn_patient_from_call fires, it must backfill
    that entry in-place. Previously this produced two entries for the same call_id
    (one with None, one with correct truth), poisoning the cascade classification
    reward.
    """
    from server.codered_environment import CodeRedEnvironment
    from server.models.actions import MaintainPlan

    env = CodeRedEnvironment()
    env.reset(seed=0, task_id="task4")  # task4 has use_call_queue=True

    # Advance until at least one pending call exists
    for _ in range(12):
        env.step(MaintainPlan())

    pending = env._pending_calls
    assert len(pending) > 0
    call = pending[0]
    call_id = call["call_id"]
    location_node = call["location_node"]
    category = call["category"]

    # Manually inject a pending outcome (simulates prior triage decision)
    env._dispatch_outcomes_history.append({
        "call_id": call_id,
        "decision": "no_dispatch",
        "category": category.value if hasattr(category, "value") else category,
        "true_condition": None,
        "als_needed": None,
        "revealed_at_step": None,
    })

    pending_outcomes = [o for o in env._dispatch_outcomes_history if o["call_id"] == call_id]
    assert len(pending_outcomes) == 1
    assert pending_outcomes[0]["true_condition"] is None

    # Manually trigger the spawn (this is what tick does via countdown expiry)
    env._spawn_patient_from_call(call_id)

    # Must STILL be exactly one entry (backfilled, not duplicated)
    final_outcomes = [o for o in env._dispatch_outcomes_history if o["call_id"] == call_id]
    assert len(final_outcomes) == 1, (
        f"Expected 1 outcome for {call_id}, got {len(final_outcomes)} — "
        f"duplicate was created instead of backfilling. Entries: {final_outcomes}"
    )
    assert final_outcomes[0]["true_condition"] is not None, (
        f"Ground truth still None after spawn — backfill failed. Entry: {final_outcomes[0]}"
    )
    assert final_outcomes[0]["als_needed"] is not None
    assert final_outcomes[0]["revealed_at_step"] is not None


def test_spawned_secondary_patient_bootstrap_reward():
    """
    Regression: mid-episode secondary patients (cascade/surge spawns) must be
    added to _prev_vitals so the agent gets a non-zero reward signal.
    Previously, new patients had no prev_vitals entry → zero delta → agent
    could not learn to respond to secondary patients via reward shaping.
    """
    from server.codered_environment import CodeRedEnvironment
    from server.models.actions import MaintainPlan

    env = CodeRedEnvironment()
    env.reset(seed=0, task_id="task1")

    # Ensure prev_vitals is bootstrapped for initial patients
    initial_patient_ids = {p.id for p in env._patient_manager.patients}
    for pid in initial_patient_ids:
        assert pid in env._prev_vitals

    # Spawn a secondary patient mid-episode (simulating cascade/surge)
    new_patient = env._patient_manager.spawn_secondary(
        condition="trauma",
        onset_step=env._state.step_count,
        reason="test_spawn",
    )
    assert new_patient.id not in env._prev_vitals, "Should not yet be in prev_vitals"

    cum_before = env._state.cum_reward

    # Take a step — the bootstrap runs inside _compute_step_reward before return
    env.step(MaintainPlan())

    # New patient must now be in _prev_vitals (bootstrap added it)
    assert new_patient.id in env._prev_vitals, (
        f"Secondary patient {new_patient.id} not bootstrapped into _prev_vitals"
    )
    assert env._prev_vitals[new_patient.id] == new_patient.vitals_score

    # Cumulative reward must have increased (includes "dispatched" milestone for new patient)
    assert env._state.cum_reward > cum_before, (
        f"Expected cumulative reward to increase for new secondary patient; "
        f"before={cum_before}, after={env._state.cum_reward}"
    )

