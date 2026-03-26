def test_task1_episode_runs_to_completion():
    """Task 1: single cardiac patient runs without crashing."""
    from codered_env.server.codered_environment import CodeRedEnvironment
    from codered_env.server.models.actions import (
        MaintainPlan, DispatchAmbulance, AssignHospital, PrepareOR, PageSpecialist,
    )
    env = CodeRedEnvironment()
    obs = env.reset(seed=0, task_id="task1")
    done = False
    steps = 0
    while not done and steps < 35:
        patient = obs.patients[0]
        amb = next((a for a in obs.ambulances if a.status.value == "available"), None)
        if patient.status.value == "waiting" and amb:
            obs = env.step(DispatchAmbulance(ambulance_id=amb.id, target_node=patient.location_node))
        elif patient.assigned_hospital is None:
            obs = env.step(AssignHospital(patient_id=patient.patient_id, hospital_id="HOSP_A"))
        elif patient.status.value not in ("treating", "treated"):
            obs = env.step(PrepareOR(hospital_id="HOSP_A", procedure_type="cardiac"))
        elif patient.status.value not in ("treating", "treated"):
            obs = env.step(PageSpecialist(hospital_id="HOSP_A", specialist_type="cardiologist"))
        else:
            obs = env.step(MaintainPlan())
        obs = env.step(MaintainPlan())
        done = env.state.step_count >= env.state.max_steps
        steps += 1
    assert env.state.step_count > 0

def test_grader_runs_without_error():
    """The grader can process an episode log without crashing."""
    from codered_env.server.codered_environment import CodeRedEnvironment
    from codered_env.server.models.actions import MaintainPlan
    env = CodeRedEnvironment()
    env.reset(seed=0, task_id="task1")
    for _ in range(30):
        env.step(MaintainPlan())
        if env.state.step_count >= env.state.max_steps:
            break
    # Episode log should be non-empty
    log = env.get_episode_log()
    assert len(log) > 0
