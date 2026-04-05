"""Grader cascade_score tests — Task 8."""
import pytest


def test_cascade_score_all_saved():
    """Perfect cascade: all secondary patients treated."""
    episode_log = [
        {"step": 0, "patient_id": "P1", "event": "secondary_patient_spawned",
         "condition": "cardiac", "is_secondary": True},
        {"step": 10, "patient_id": "P1", "event": "treatment_complete"},
    ]
    from server.grader import grade_cascade_score
    score = grade_cascade_score(episode_log)
    assert score == 1.0


def test_cascade_score_all_dead():
    """Worst cascade: all secondary patients deceased."""
    episode_log = [
        {"step": 0, "patient_id": "P1", "event": "secondary_patient_spawned",
         "condition": "trauma", "is_secondary": True},
        {"step": 100, "patient_id": "P1", "event": "patient_deceased", "reason": "secondary"},
    ]
    from server.grader import grade_cascade_score
    score = grade_cascade_score(episode_log)
    assert score == 0.5  # secondary_score=0.0, but overcrowding=1.0, news=1.0


def test_cascade_score_partial():
    """Mixed cascade: 1 saved, 1 dead out of 2."""
    episode_log = [
        {"step": 0, "patient_id": "P1", "event": "secondary_patient_spawned",
         "condition": "cardiac", "is_secondary": True},
        {"step": 10, "patient_id": "P1", "event": "treatment_complete"},
        {"step": 0, "patient_id": "P2", "event": "secondary_patient_spawned",
         "condition": "trauma", "is_secondary": True},
        {"step": 100, "patient_id": "P2", "event": "patient_deceased", "reason": "secondary"},
    ]
    from server.grader import grade_cascade_score
    score = grade_cascade_score(episode_log)
    # secondary_score = 0.5 (1 death out of 2)
    # No overcrowding -> 1.0, no news -> 1.0
    # cascade = 0.5*0.5 + 0.3*1.0 + 0.2*1.0 = 0.75
    assert 0.7 < score < 0.8


def test_cascade_score_no_cascades():
    """No cascade events = perfect score (no penalty for nothing to manage)."""
    episode_log = [
        {"step": 0, "patient_id": "P1", "event": "patient_created", "condition": "cardiac"},
        {"step": 10, "patient_id": "P1", "event": "treatment_complete"},
    ]
    from server.grader import grade_cascade_score
    score = grade_cascade_score(episode_log)
    assert score == 1.0


def test_cascade_score_overcrowding_penalty():
    """Overcrowding events reduce score."""
    episode_log = [
        {"step": 0, "patient_id": "P1", "event": "secondary_patient_spawned",
         "condition": "cardiac", "is_secondary": True},
        {"step": 10, "patient_id": "P1", "event": "treatment_complete"},
        {"step": 5, "event": "overcrowding_started", "active_patient_count": 5},
        {"step": 15, "event": "overcrowding_started", "active_patient_count": 6},
    ]
    from server.grader import grade_cascade_score
    score = grade_cascade_score(episode_log)
    # 2 overcrowding events -> 1.0 - 2/5 = 0.6
    # cascade = 0.5*1.0 + 0.3*0.6 + 0.2*1.0 = 0.88
    assert score > 0.8


def test_rubric_result_has_cascade_score():
    """RubricResult dataclass has cascade_score field."""
    from server.grader import RubricResult
    result = RubricResult(
        time_score=0.8,
        efficiency=0.9,
        secondary_harm=1.0,
        prep_ready=1.0,
        mutual_aid_penalty=0.0,
        final_score=0.85,
        breakdown={},
        cascade_score=0.75,
    )
    assert result.cascade_score == 0.75


def test_rubric_result_cascade_score_defaults_to_one():
    """RubricResult cascade_score defaults to 1.0 when not provided."""
    from server.grader import RubricResult
    result = RubricResult(
        time_score=0.8,
        efficiency=0.9,
        secondary_harm=1.0,
        prep_ready=1.0,
        mutual_aid_penalty=0.0,
        final_score=0.85,
        breakdown={},
    )
    assert result.cascade_score == 1.0
