"""Phase 2 patient manager tests — Task 7."""
import pytest


def test_patient_dataclass_has_phase2_fields():
    """Patient dataclass has Phase 2 fields."""
    from dataclasses import fields
    from server.subsystems.patient_manager import Patient
    field_names = {f.name for f in fields(Patient)}
    assert "dispatch_call_id" in field_names
    assert "is_secondary" in field_names
    assert "cascade_trigger_reason" in field_names
    assert "observed_condition" in field_names


def test_patient_default_phase2_fields():
    """Phase 2 fields have correct defaults."""
    from server.subsystems.patient_manager import Patient
    p = Patient(id="P1", condition="cardiac", status="waiting", location_node="node_A", onset_step=0)
    assert p.dispatch_call_id is None
    assert p.is_secondary is False
    assert p.cascade_trigger_reason is None
    assert p.observed_condition is None


def test_spawn_secondary_accepts_new_params():
    """spawn_secondary accepts triggered_by, reason, spawn_node params."""
    from server.subsystems.patient_manager import PatientManager
    import random
    pm = PatientManager()
    pm._rng = random.Random(42)
    pm._patient_counter = 0
    patient = pm.spawn_secondary(
        condition="cardiac", onset_step=10,
        triggered_by="P_orig", reason="psychogenic_cascade", spawn_node="node_B",
    )
    assert patient.is_secondary is True
    assert patient.cascade_trigger_reason == "psychogenic_cascade"
    assert patient.condition == "cardiac"
    assert patient.location_node == "node_B"


def test_spawn_secondary_sets_observed_condition():
    """Secondary patients have observed_condition set immediately."""
    from server.subsystems.patient_manager import PatientManager
    import random
    pm = PatientManager()
    pm._rng = random.Random(99)
    pm._patient_counter = 0
    patient = pm.spawn_secondary(condition="trauma", onset_step=0, reason="news_cycle")
    assert patient.observed_condition == "trauma"


def test_tick_accepts_overcrowding_modifier():
    """tick() accepts overcrowding_modifier parameter without TypeError."""
    from server.subsystems.patient_manager import PatientManager
    import random
    pm = PatientManager()
    pm._rng = random.Random(42)
    pm._patient_counter = 0
    patient = pm.spawn_secondary(condition="cardiac", onset_step=0)
    onset_steps = {patient.id: 0}
    pm.tick(onset_steps, step_count=200, overcrowding_modifier=1.2)  # must not raise


def test_patient_with_dispatch_call_id():
    """Patient can be created with dispatch_call_id."""
    from server.subsystems.patient_manager import Patient
    p = Patient(id="P_new", condition="general", status="waiting", location_node="node_A",
                onset_step=5, dispatch_call_id="CALL_0001")
    assert p.dispatch_call_id == "CALL_0001"
