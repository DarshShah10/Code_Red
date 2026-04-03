from dataclasses import dataclass, field
from typing import Optional

from codered_env.server.grader import grade_episode, grade_from_environment


@dataclass
class MockPatient:
    id: str
    outcome: Optional[str] = None


class MockPatientManager:
    def __init__(self, patients):
        self.patients = patients

    def get_all(self):
        return self.patients


class MockEnv:
    def __init__(self, episode_log, patients):
        self._episode_log = episode_log
        self._patient_manager = MockPatientManager(patients)

    def get_episode_log(self):
        return self._episode_log

def test_all_patients_treated_on_time():
    episode_log = [
        {"step": 0, "patient_id": "P0", "event": "patient_created", "condition": "CARDIAC"},
        {"step": 1, "patient_id": "P0", "event": "dispatch", "ambulance_id": "AMB_1", "target_time": 90, "effective_time": 14},
        {"step": 15, "patient_id": "P0", "event": "treatment_complete"},
    ]
    result = grade_episode(episode_log)
    # 14 min vs 90 min target: score = 1.0 - (14-90)/90 = clamped to 1.0
    assert result.time_score == 1.0
    assert result.efficiency == 1.0  # no wasted actions
    assert result.final_score >= 0.80

def test_patient_not_treated():
    episode_log = [
        {"step": 0, "patient_id": "P0", "event": "patient_created", "condition": "CARDIAC"},
        # patient never dispatched, never treated
    ]
    result = grade_episode(episode_log)
    assert result.time_score == 0.0

def test_efficiency_wasted_or_prep():
    # Two wasted OR preps: penalty = -0.15 * 2 = -0.30 → efficiency = 0.70
    episode_log = [
        {"step": 0, "patient_id": "P0", "event": "patient_created"},
        {"step": 1, "patient_id": "P0", "event": "action_prepare_or", "result": "wasted"},
        {"step": 2, "patient_id": "P0", "event": "action_prepare_or", "result": "wasted"},
        {"step": 3, "patient_id": "P0", "event": "treatment_complete"},
    ]
    result = grade_episode(episode_log)
    assert result.efficiency == 0.70

def test_efficiency_unused_specialist_pages():
    # One unused specialist page: penalty = -0.1 * 1 = -0.10 → efficiency = 0.90
    episode_log = [
        {"step": 0, "patient_id": "P0", "event": "patient_created"},
        {"step": 1, "patient_id": "P0", "event": "action_page_specialist", "result": "wasted"},
        {"step": 2, "patient_id": "P0", "event": "treatment_complete"},
    ]
    result = grade_episode(episode_log)
    assert result.efficiency == 0.90

def test_mutual_aid_on_time_no_penalty():
    # MA arrives exactly at optimal: 0 penalty
    episode_log = [
        {"step": 0, "patient_id": "P0", "event": "patient_created"},
        {"step": 5, "patient_id": "P0", "event": "mutual_aid_called", "ambulance_id": "MUTUAL_1",
         "patient_id": "P0", "optimal_arrival_step": 17},
        {"step": 17, "patient_id": "P0", "event": "mutual_aid_arrived",
         "ambulance_id": "MUTUAL_1", "patient_id": "P0", "actual_arrival_step": 17},
        {"step": 20, "patient_id": "P0", "event": "treatment_complete"},
    ]
    result = grade_episode(episode_log)
    assert result.mutual_aid_penalty == 0.0

def test_mutual_aid_late_penalty():
    # MA arrives 5 steps late: penalty = 0.2 * 5 / 1 call = 1.0 → capped to 1.0
    episode_log = [
        {"step": 0, "patient_id": "P0", "event": "patient_created"},
        {"step": 5, "patient_id": "P0", "event": "mutual_aid_called", "ambulance_id": "MUTUAL_1",
         "patient_id": "P0", "optimal_arrival_step": 17},
        {"step": 22, "patient_id": "P0", "event": "mutual_aid_arrived",
         "ambulance_id": "MUTUAL_1", "patient_id": "P0", "actual_arrival_step": 22},
        {"step": 30, "patient_id": "P0", "event": "treatment_complete"},
    ]
    result = grade_episode(episode_log)
    assert result.mutual_aid_penalty > 0.0
    assert result.final_score < 1.0

def test_secondary_harm_partial():
    # 1 of 2 secondary patients died → harm = 1 - 0.5 = 0.5
    episode_log = [
        {"step": 0, "patient_id": "P0", "event": "patient_created", "condition": "CARDIAC"},
        {"step": 1, "patient_id": "P1", "event": "patient_created", "is_secondary": True},
        {"step": 2, "patient_id": "P2", "event": "patient_created", "is_secondary": True},
        {"step": 3, "patient_id": "P2", "event": "patient_deceased", "reason": "secondary"},
        {"step": 20, "patient_id": "P0", "event": "treatment_complete"},
    ]
    result = grade_episode(episode_log)
    assert result.secondary_harm == 0.5


# =============================================================================
# Cross-validation tests (Task 11)
# =============================================================================

def test_cross_validation_no_mismatch():
    """No mismatches → no cross-validation penalty."""
    log = [
        {"step": 0, "patient_id": "P1", "event": "patient_created", "condition": "CARDIAC"},
        {"step": 10, "patient_id": "P1", "event": "treatment_complete",
         "effective_time": 10, "target_time": 90, "vitals_at_treatment": 1.0},
    ]
    env = MockEnv(log, [MockPatient("P1", outcome="saved")])
    result = grade_from_environment(env)
    assert result.breakdown["cross_validation_mismatches"] == 0
    assert result.breakdown["cross_validation_penalty"] == 0.0


def test_cross_validation_treated_missing_log():
    """Patient marked saved but no treatment_complete event → 1 mismatch, 0.2 penalty."""
    log = [
        {"step": 0, "patient_id": "P1", "event": "patient_created", "condition": "CARDIAC"},
        # No treatment_complete event for P1
    ]
    env = MockEnv(log, [MockPatient("P1", outcome="saved")])
    result = grade_from_environment(env)
    assert result.breakdown["cross_validation_mismatches"] == 1
    assert result.breakdown["cross_validation_penalty"] == 0.2
    assert "P1" in result.breakdown["treated_missing_log"]
    # Score reduced by penalty
    original = grade_episode(log)
    assert result.final_score < original.final_score


def test_cross_validation_treated_but_deceased():
    """Patient has treatment_complete event but outcome=deceased → 1 mismatch."""
    log = [
        {"step": 0, "patient_id": "P1", "event": "patient_created", "condition": "CARDIAC"},
        {"step": 50, "patient_id": "P1", "event": "treatment_complete",
         "effective_time": 50, "target_time": 90, "vitals_at_treatment": 0.8},
    ]
    env = MockEnv(log, [MockPatient("P1", outcome="deceased")])
    result = grade_from_environment(env)
    assert result.breakdown["cross_validation_mismatches"] == 1
    assert "P1" in result.breakdown["treated_but_deceased"]


def test_cross_validation_multiple_mismatches():
    """Two mismatches → 0.4 penalty, capped at 1.0."""
    log = [
        {"step": 0, "patient_id": "P1", "event": "patient_created", "condition": "CARDIAC"},
        {"step": 0, "patient_id": "P2", "event": "patient_created", "condition": "STROKE"},
        # P1: saved but no treatment_complete
        # P2: has treatment_complete but outcome=deceased
        {"step": 30, "patient_id": "P2", "event": "treatment_complete",
         "effective_time": 30, "target_time": 60, "vitals_at_treatment": 0.9},
    ]
    env = MockEnv(log, [MockPatient("P1", outcome="saved"), MockPatient("P2", outcome="deceased")])
    result = grade_from_environment(env)
    assert result.breakdown["cross_validation_mismatches"] == 2
    assert result.breakdown["cross_validation_penalty"] == 0.4


def test_cross_validation_all_patients_deceased_no_treated():
    """All patients deceased with no treatment_complete → no mismatch (expected outcome)."""
    log = [
        {"step": 0, "patient_id": "P1", "event": "patient_created", "condition": "CARDIAC"},
        {"step": 10, "patient_id": "P1", "event": "patient_deceased", "reason": "timeout"},
    ]
    env = MockEnv(log, [MockPatient("P1", outcome="deceased")])
    result = grade_from_environment(env)
    assert result.breakdown["cross_validation_mismatches"] == 0
    assert result.breakdown["cross_validation_penalty"] == 0.0

