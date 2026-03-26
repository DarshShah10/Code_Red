from codered_env.server.subsystems.patient_manager import PatientManager

def test_patient_created():
    pm = PatientManager()
    pm.reset(task_id="task1", rng=None)
    assert len(pm.patients) == 1
    assert pm.patients[0].condition == "cardiac"
    assert pm.patients[0].treatment_complete_time is None  # Not yet treated

def test_mark_treated_sets_complete_time():
    pm = PatientManager()
    pm.reset(task_id="task1", rng=None)
    patient = pm.patients[0]
    patient.status = "in_treatment"
    pm.mark_treated(patient.id, treatment_complete_time=14)
    assert patient.treatment_complete_time == 14
    assert patient.outcome == "saved"

def test_mark_deceased_sets_outcome():
    pm = PatientManager()
    pm.reset(task_id="task1", rng=None)
    patient = pm.patients[0]
    pm.mark_deceased(patient.id, reason="timeout")
    assert patient.outcome == "deceased"

def test_patients_from_task2():
    pm = PatientManager()
    import random
    rng = random.Random(0)
    pm.reset(task_id="task2", rng=rng)
    assert len(pm.patients) >= 2  # cardiac + stroke
