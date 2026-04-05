"""Phase 2 observation tests — Task 4."""
import pytest


def test_pending_calls_field_exists():
    """CodeRedObservation has pending_calls field."""
    from server.models.observations import CodeRedObservation
    obs = CodeRedObservation(
        step=0,
        patients=[],
        ambulances=[],
        hospitals=[],
        blood_banks=[],
        road_network={},
        alerts=[],
        mutual_aid_remaining=1,
        time_score_preview=1.0,
        patients_remaining=0,
        vitals_score_preview=1.0,
    )
    assert hasattr(obs, "pending_calls")
    assert obs.pending_calls == []


def test_recent_dispatch_outcomes_field_exists():
    """CodeRedObservation has recent_dispatch_outcomes field."""
    from server.models.observations import CodeRedObservation
    obs = CodeRedObservation(
        step=0,
        patients=[],
        ambulances=[],
        hospitals=[],
        blood_banks=[],
        road_network={},
        alerts=[],
        mutual_aid_remaining=1,
        time_score_preview=1.0,
        patients_remaining=0,
        vitals_score_preview=1.0,
    )
    assert hasattr(obs, "recent_dispatch_outcomes")
    assert obs.recent_dispatch_outcomes == []


def test_overcrowding_modifier_field_exists():
    """CodeRedObservation has overcrowding_modifier field with bounds."""
    from server.models.observations import CodeRedObservation
    obs = CodeRedObservation(
        step=0,
        patients=[],
        ambulances=[],
        hospitals=[],
        blood_banks=[],
        road_network={},
        alerts=[],
        mutual_aid_remaining=1,
        time_score_preview=1.0,
        patients_remaining=0,
        vitals_score_preview=1.0,
    )
    assert hasattr(obs, "overcrowding_modifier")
    assert obs.overcrowding_modifier == 1.0


def test_dispatch_call_pydantic_model_importable():
    """DispatchCall can be instantiated."""
    from server.models.entities import DispatchCall
    from server.subsystems.constants import DispatchCategory
    call = DispatchCall(
        call_id="CALL_0001",
        category=DispatchCategory.CHEST_PAIN,
        location_node="node_A",
    )
    assert call.call_id == "CALL_0001"
    assert call.category == DispatchCategory.CHEST_PAIN
    assert call.time_waiting == 0
    assert call.estimated_severity == 0.1
    assert call.spawned_patient_id is None


def test_dispatch_outcome_pydantic_model_importable():
    """DispatchOutcome can be instantiated."""
    from server.models.entities import DispatchOutcome
    outcome = DispatchOutcome(
        call_id="CALL_0001",
        decision="als",
        category="chest_pain",
    )
    assert outcome.call_id == "CALL_0001"
    assert outcome.decision == "als"
    assert outcome.true_condition is None
    assert outcome.als_needed is False
    assert outcome.revealed_at_step is None
