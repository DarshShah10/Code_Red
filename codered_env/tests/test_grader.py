from codered_env.server.grader import grade_episode

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
