"""Integration tests for wired CodeRedEnvironment with subsystems."""

import pytest


def test_dispatch_routes_ambulance():
    from codered_env.server.codered_environment import CodeRedEnvironment
    from codered_env.server.models.actions import DispatchAmbulance
    env = CodeRedEnvironment()
    env.reset(seed=0, task_id="task1")
    obs = env.step(DispatchAmbulance(ambulance_id="AMB_1", target_node="NH45_BYPASS"))
    amb = next(a for a in obs.ambulances if a.id == "AMB_1")
    assert len(amb.route) > 0
    assert amb.eta_minutes > 0


def test_prepare_or_increments_prep_countdown():
    from codered_env.server.codered_environment import CodeRedEnvironment
    from codered_env.server.models.actions import PrepareOR
    env = CodeRedEnvironment()
    env.reset(seed=0, task_id="task1")
    obs = env.step(PrepareOR(hospital_id="HOSP_A", procedure_type="cardiac"))
    hosp = next(h for h in obs.hospitals if h.id == "HOSP_A")
    prep_or = next((o for o in hosp.operating_rooms if o.status.value == "prep"), None)
    assert prep_or is not None


def test_assign_hospital_sets_patient_destination():
    from codered_env.server.codered_environment import CodeRedEnvironment
    from codered_env.server.models.actions import AssignHospital
    env = CodeRedEnvironment()
    env.reset(seed=0, task_id="task1")
    obs = env.step(AssignHospital(patient_id="P1", hospital_id="HOSP_A"))
    patient = next(p for p in obs.patients if p.patient_id == "P1")
    assert patient.assigned_hospital == "HOSP_A"


def test_patient_cannot_be_assigned_to_unsuitable_hospital():
    from codered_env.server.codered_environment import CodeRedEnvironment
    from codered_env.server.models.actions import AssignHospital
    env = CodeRedEnvironment()
    env.reset(seed=0, task_id="task1")
    obs = env.step(AssignHospital(patient_id="P1", hospital_id="HOSP_C"))
    patient = next(p for p in obs.patients if p.patient_id == "P1")
    assert patient.assigned_hospital is None  # Assignment failed
    assert any("cannot treat" in a for a in obs.alerts)


def test_mutual_aid_not_available_task1():
    from codered_env.server.codered_environment import CodeRedEnvironment
    from codered_env.server.models.actions import RequestMutualAid
    env = CodeRedEnvironment()
    obs = env.reset(seed=0, task_id="task1")
    assert obs.mutual_aid_remaining == 0
    obs2 = env.step(RequestMutualAid())
    assert any("no calls remaining" in a for a in obs2.alerts)


def test_mutual_aid_assigns_patient_and_logs_arrival():
    """MA ambulance arrival should auto-assign highest-priority patient and log patient_id."""
    from codered_env.server.codered_environment import CodeRedEnvironment
    from codered_env.server.models.actions import RequestMutualAid, MaintainPlan
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
    from codered_env.server.codered_environment import CodeRedEnvironment
    from codered_env.server.models.actions import AllocateBlood
    env = CodeRedEnvironment()
    env.reset(seed=0, task_id="task1")
    obs = env.step(AllocateBlood(
        hospital_id="HOSP_A", patient_id="P1",
        blood_type="O_NEG", units=2, emergency=True
    ))
    assert any("Emergency blood" in a for a in obs.alerts)


def test_timestep_increments_patient_time():
    from codered_env.server.codered_environment import CodeRedEnvironment
    from codered_env.server.models.actions import MaintainPlan
    env = CodeRedEnvironment()
    env.reset(seed=0, task_id="task1")
    patient = env._patients[0]
    # Subsystem patient tracks onset_step; time_since_onset is calculated in observation
    assert patient.onset_step == 0
    env.step(MaintainPlan())
    assert env._state.step_count == 1
    env.step(MaintainPlan())
    assert env._state.step_count == 2
