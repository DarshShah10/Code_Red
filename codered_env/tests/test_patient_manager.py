import random

import pytest
from server.subsystems.patient_manager import PatientManager

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
    rng = random.Random(0)
    pm.reset(task_id="task2", rng=rng)
    assert len(pm.patients) >= 2  # cardiac + stroke


def test_patient_vitals_decline_post_target():
    """Vitals decay after effective_time exceeds target_time."""
    pm = PatientManager()
    rng = random.Random(0)
    pm.reset(task_id="task1", rng=rng)  # task1: cardiac, target=90min
    patient = pm.patients[0]
    assert patient.vitals_score == 1.0

    onset_steps = pm.get_onset_steps()

    # Steps 0-90: vitals stay near 1.0 (stable window)
    for step in range(1, 91):
        pm.tick(onset_steps, step)
    assert patient.vitals_score == pytest.approx(1.0, abs=1e-6)

    # Step 91: first post-target tick — vitals should be ~0.99 (22.5/90 overtime)
    pm.tick(onset_steps, 91)
    assert 0.9 < patient.vitals_score < 1.0

    # Step 135: overtime_ratio = 45/90 = 0.5, vitals = 0.5
    pm.tick(onset_steps, 135)
    assert patient.vitals_score == pytest.approx(0.5, abs=0.05)

    # Step 180: overtime_ratio = 90/90 = 1.0, vitals = 0.0
    pm.tick(onset_steps, 180)
    assert patient.vitals_score == pytest.approx(0.0, abs=0.05)


def test_patient_vitals_freeze_on_treatment():
    """Vitals freeze at treatment time, no further decay."""
    pm = PatientManager()
    rng = random.Random(0)
    pm.reset(task_id="task1", rng=rng)
    patient = pm.patients[0]
    onset_steps = pm.get_onset_steps()

    # Move to post-target (vitals declining)
    for step in range(1, 100):
        pm.tick(onset_steps, step)

    vitals_at_treatment = patient.vitals_score
    assert 0.0 < vitals_at_treatment < 1.0

    # Treat the patient
    pm.mark_treated(patient.id, treatment_complete_time=100)
    assert patient.status == "treated"
    assert patient.vitals_score == vitals_at_treatment

    # Tick several more times — vitals should NOT change
    for _ in range(10):
        pm.tick(onset_steps, 200)
    assert patient.vitals_score == vitals_at_treatment


def test_patient_status_escalation():
    """Patient status escalates from waiting→deteriorating→critical at thresholds."""
    pm = PatientManager()
    rng = random.Random(0)
    pm.reset(task_id="task1", rng=rng)  # cardiac target=90
    patient = pm.patients[0]
    onset_steps = pm.get_onset_steps()

    # Initially waiting
    assert patient.status == "waiting"

    # Advance to deteriorating threshold (vitals <= 0.75)
    for step in range(1, 120):
        pm.tick(onset_steps, step)
    assert patient.status == "deteriorating", f"Expected deteriorating at step 119, got {patient.status} (vitals={patient.vitals_score:.3f})"

    # Advance to critical threshold (vitals <= 0.4)
    for step in range(120, 160):
        pm.tick(onset_steps, step)
    assert patient.status == "critical", f"Expected critical at step 159, got {patient.status} (vitals={patient.vitals_score:.3f})"


def test_patient_death_at_zero_vitals():
    """Patient marked deceased when vitals reach 0.0."""
    pm = PatientManager()
    rng = random.Random(0)
    pm.reset(task_id="task1", rng=rng)  # cardiac target=90, death at step 180
    patient = pm.patients[0]
    onset_steps = pm.get_onset_steps()

    # Tick until death
    for step in range(1, 181):
        pm.tick(onset_steps, step)

    assert patient.status == "deceased"
    assert patient.vitals_score == 0.0
    assert patient.outcome == "deceased"
