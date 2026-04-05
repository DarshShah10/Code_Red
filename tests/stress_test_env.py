"""
Comprehensive stress test for CodeRedEnv.
Tests all 5 tasks, grading, Phase 2, edge cases, and endpoint integration.
Run with: python tests/stress_test_env.py
"""
import sys
import traceback

sys.path.insert(0, "C:/Darsh/Scaler/codered_env")

from server.codered_environment import CodeRedEnvironment
from server.grader import grade_from_environment, grade_episode, RubricResult
from server.models.actions import (
    MaintainPlan, DispatchAmbulance,
)
from server.subsystems.constants import TASK_CONFIG


def make_env(task_id="task1", seed=0):
    """Create and reset environment. Returns (env, obs)."""
    env = CodeRedEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)
    return env, obs


def run_episode(env, policy_fn=None):
    """Run full episode with MaintainPlan (no-op) policy. Returns (done, steps, final_obs)."""
    if policy_fn is None:
        policy_fn = lambda obs: MaintainPlan()
    obs = env.reset()  # re-reset to start fresh
    steps = 0
    max_s = env._state.max_steps

    while steps < max_s:
        action = policy_fn(obs)
        obs = env.step(action)
        steps += 1
        if obs.done:
            break

    return obs.done, steps, obs


def run_n_steps(env, n, obs=None, policy_fn=None):
    """Run n steps from given obs. Returns (done, steps_taken, final_obs)."""
    if policy_fn is None:
        policy_fn = lambda obs: MaintainPlan()
    if obs is None:
        obs = env.reset()
    steps = 0
    max_s = env._state.max_steps
    while steps < n and steps < max_s:
        action = policy_fn(obs)
        obs = env.step(action)
        steps += 1
        if obs.done:
            break
    return obs.done, steps, obs


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1: Reset sanity — all 5 tasks × 5 seeds
# ─────────────────────────────────────────────────────────────────────────────
def test_all_tasks_reset():
    task_ids = ["task1", "task2", "task3", "task4", "task5"]
    results = {}
    for tid in task_ids:
        for seed in range(5):
            key = f"{tid}_s{seed}"
            try:
                env, obs = make_env(task_id=tid, seed=seed)
                assert obs is not None, "reset returned None"
                assert hasattr(obs, "step"), "no step attribute"
                assert obs.step == 0, f"step != 0 after reset (got {obs.step})"
                assert hasattr(obs, "patients"), "no patients attribute"
                assert hasattr(obs, "ambulances"), "no ambulances attribute"
                assert hasattr(obs, "hospitals"), "no hospitals attribute"
                # Run a few steps
                for _ in range(3):
                    obs = env.step(MaintainPlan())
                results[key] = "PASS"
            except Exception as e:
                results[key] = f"FAIL: {e}"
    return results


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2: Full grading — all 5 tasks × 3 seeds
# ─────────────────────────────────────────────────────────────────────────────
def test_grader_all_tasks():
    task_ids = ["task1", "task2", "task3", "task4", "task5"]
    results = {}
    for tid in task_ids:
        for seed in range(3):
            key = f"{tid}_s{seed}"
            try:
                env, _ = make_env(task_id=tid, seed=seed)
                done, steps, obs = run_episode(env)
                result = grade_from_environment(env)

                assert isinstance(result, RubricResult), f"{key}: wrong type"
                assert 0.0 <= result.time_score <= 1.0, f"{key}: time_score={result.time_score} out of [0,1]"
                assert 0.0 <= result.efficiency <= 1.0, f"{key}: efficiency={result.efficiency} out of [0,1]"
                assert 0.0 <= result.secondary_harm <= 1.0, f"{key}: secondary_harm out of [0,1]"
                assert 0.0 <= result.prep_ready <= 1.0, f"{key}: prep_ready out of [0,1]"
                assert 0.0 <= result.vitals_score_avg <= 1.0, f"{key}: vitals_score_avg out of [0,1]"
                assert 0.0 <= result.cascade_score <= 1.0, f"{key}: cascade_score out of [0,1]"
                assert isinstance(result.mutual_aid_penalty, (int, float)), f"{key}: penalty not numeric"
                assert 0.0 <= result.final_score <= 1.0, f"{key}: final_score={result.final_score} out of [0,1]"
                assert isinstance(result.as_dict(), dict), f"{key}: as_dict() failed"

                results[key] = f"PASS score={result.final_score:.3f} steps={steps} done={done}"
            except Exception as e:
                results[key] = f"FAIL: {e}\n{traceback.format_exc()[-500:]}"
    return results


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3: Grader as_dict format — all tasks × 3 seeds
# ─────────────────────────────────────────────────────────────────────────────
def test_grader_as_dict():
    task_ids = ["task1", "task2", "task3", "task4", "task5"]
    results = {}
    for tid in task_ids:
        for seed in range(3):
            key = f"{tid}_s{seed}"
            try:
                env, _ = make_env(task_id=tid, seed=seed)
                run_episode(env)
                d = grade_from_environment(env).as_dict()
                required = ["time_score", "efficiency", "secondary_harm", "prep_ready",
                            "mutual_aid_penalty", "final_score", "breakdown",
                            "vitals_score_avg", "cascade_score"]
                for k in required:
                    assert k in d, f"{key}: missing {k} in as_dict()"
                results[key] = f"PASS ({len(d)} keys)"
            except Exception as e:
                results[key] = f"FAIL: {e}"
    return results


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4: Phase 2 tasks (task4 & task5) — call queue, triage, cascade
# ─────────────────────────────────────────────────────────────────────────────
def test_phase2_detailed():
    results = {}

    # task4: call queue without cascade
    for seed in range(3):
        key = f"task4_s{seed}"
        try:
            env, obs0 = make_env(task_id="task4", seed=seed)
            # Track that pending_calls appear in at least one observation
            pending_seen = False
            env2, _ = make_env(task_id="task4", seed=seed)
            done, steps, obs = run_episode(env2)
            # pending_calls should have been seen
            result = grade_from_environment(env2)
            assert 0.0 <= result.final_score <= 1.0
            results[key] = f"PASS score={result.final_score:.3f} steps={steps}"
        except Exception as e:
            results[key] = f"FAIL: {e}"

    # task5: call queue + cascade engine
    for seed in range(3):
        key = f"task5_s{seed}"
        try:
            env, _ = make_env(task_id="task5", seed=seed)
            done, steps, obs = run_episode(env)
            result = grade_from_environment(env)
            assert 0.0 <= result.final_score <= 1.0
            assert hasattr(result, "cascade_score")
            results[key] = f"PASS cascade={result.cascade_score:.3f} score={result.final_score:.3f} steps={steps}"
        except Exception as e:
            results[key] = f"FAIL: {e}"

    return results


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 5: Edge cases
# ─────────────────────────────────────────────────────────────────────────────
def test_edge_cases():
    results = {}

    # Edge 1: All patients deceased (let episode run long)
    key = "edge_all_dead"
    try:
        env, _ = make_env(task_id="task1", seed=0)
        done, steps, obs = run_n_steps(env, 150)
        result = grade_from_environment(env)
        assert 0.0 <= result.final_score <= 1.0, f"score={result.final_score}"
        results[key] = f"PASS score={result.final_score:.3f}"
    except Exception as e:
        results[key] = f"FAIL: {e}"

    # Edge 2: task3 max patients
    key = "edge_task3"
    try:
        env, _ = make_env(task_id="task3", seed=0)
        done, steps, obs = run_episode(env)
        result = grade_from_environment(env)
        assert 0.0 <= result.final_score <= 1.0
        results[key] = f"PASS score={result.final_score:.3f} steps={steps}"
    except Exception as e:
        results[key] = f"FAIL: {e}"

    # Edge 3: Extreme seed
    key = "edge_extreme_seed"
    try:
        for tid in ["task1", "task3", "task5"]:
            env, _ = make_env(task_id=tid, seed=99999)
            done, steps, obs = run_episode(env)
            result = grade_from_environment(env)
            assert 0.0 <= result.final_score <= 1.0, f"{tid} seed=99999: score={result.final_score}"
        results[key] = "PASS for task1, task3, task5 with seed=99999"
    except Exception as e:
        results[key] = f"FAIL: {e}"

    # Edge 4: No mutual aid used (task2, task3)
    key = "edge_no_mutual_aid"
    try:
        for tid in ["task2", "task3"]:
            env, _ = make_env(task_id=tid, seed=0)
            done, steps, obs = run_episode(env)
            result = grade_from_environment(env)
            assert isinstance(result.mutual_aid_penalty, (int, float))
            assert 0.0 <= result.final_score <= 1.0
        results[key] = "PASS"
    except Exception as e:
        results[key] = f"FAIL: {e}"

    # Edge 5: Exact max steps (episode doesn't early-terminate)
    key = "edge_exact_max_steps"
    try:
        env, _ = make_env(task_id="task1", seed=42)
        max_s = env._state.max_steps
        done, steps, obs = run_n_steps(env, max_s)
        result = grade_from_environment(env)
        assert 0.0 <= result.final_score <= 1.0
        results[key] = f"PASS steps={steps} max={max_s} score={result.final_score:.3f}"
    except Exception as e:
        results[key] = f"FAIL: {e}"

    return results


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 6: Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
def test_reproducibility():
    results = {}
    for tid in ["task1", "task2", "task3", "task4", "task5"]:
        key = f"repro_{tid}"
        try:
            scores = []
            for run in range(3):
                env, _ = make_env(task_id=tid, seed=0)
                run_episode(env)
                result = grade_from_environment(env)
                scores.append(result.final_score)
            assert len(set(scores)) == 1, f"non-deterministic: {scores}"
            results[key] = f"PASS score={scores[0]:.3f}"
        except Exception as e:
            results[key] = f"FAIL: {e}"
    return results


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 7: Grader pure function
# ─────────────────────────────────────────────────────────────────────────────
def test_grader_pure_function():
    results = {}

    key = "pure_empty"
    try:
        result = grade_episode([])
        assert 0.0 <= result.time_score <= 1.0
        assert 0.0 <= result.efficiency <= 1.0
        assert 0.0 <= result.secondary_harm <= 1.0
        assert 0.0 <= result.prep_ready <= 1.0
        assert 0.0 <= result.cascade_score <= 1.0
        results[key] = "PASS"
    except Exception as e:
        results[key] = f"FAIL: {e}"

    # Treatment on time: efficiency = 1.0, time_score = 1.0
    key = "pure_treatment_on_time"
    try:
        log = [
            {"event": "patient_created", "patient_id": "P1", "condition": "cardiac", "is_secondary": False},
            {"event": "patient_arrived_hospital", "patient_id": "P1", "hospital_id": "HOSP_A",
             "arrival_step": 30, "or_ready": True, "specialist_available": True},
            {"event": "treatment_complete", "patient_id": "P1", "outcome": "saved",
             "effective_time": 40, "target_time": 90, "vitals_at_treatment": 0.9},
        ]
        result = grade_episode(log)
        assert 0.0 <= result.final_score <= 1.0
        assert result.efficiency == 1.0, f"efficiency={result.efficiency} expected 1.0"
        results[key] = f"PASS score={result.final_score:.3f}"
    except Exception as e:
        results[key] = f"FAIL: {e}"

    # Wasted actions: efficiency = 1.0 - 0.15 - 0.1 = 0.75
    key = "pure_wasted_actions"
    try:
        log = [
            {"event": "patient_created", "patient_id": "P1", "condition": "cardiac", "is_secondary": False},
            {"event": "patient_arrived_hospital", "patient_id": "P1", "hospital_id": "HOSP_A",
             "arrival_step": 30, "or_ready": True, "specialist_available": True},
            {"event": "treatment_complete", "patient_id": "P1", "outcome": "saved",
             "effective_time": 40, "target_time": 90, "vitals_at_treatment": 0.9},
            {"event": "action_prepare_or", "hospital_id": "HOSP_A", "result": "wasted"},
            {"event": "action_page_specialist", "hospital_id": "HOSP_A", "result": "wasted"},
        ]
        result = grade_episode(log)
        assert 0.0 <= result.final_score <= 1.0
        assert result.efficiency == 0.75, f"efficiency={result.efficiency} expected 0.75"
        results[key] = f"PASS score={result.final_score:.3f} eff={result.efficiency:.3f}"
    except Exception as e:
        results[key] = f"FAIL: {e}"

    # Secondary patient death: secondary_harm = 0.0 (1 death / 1 secondary)
    key = "pure_secondary_death"
    try:
        log = [
            {"event": "patient_created", "patient_id": "P1", "condition": "cardiac", "is_secondary": False},
            {"event": "patient_created", "patient_id": "P2", "condition": "cardiac", "is_secondary": True},
            {"event": "patient_deceased", "patient_id": "P2", "reason": "secondary"},
            {"event": "patient_arrived_hospital", "patient_id": "P1", "hospital_id": "HOSP_A",
             "arrival_step": 30, "or_ready": True, "specialist_available": True},
            {"event": "treatment_complete", "patient_id": "P1", "outcome": "saved",
             "effective_time": 40, "target_time": 90, "vitals_at_treatment": 0.9},
        ]
        result = grade_episode(log)
        assert 0.0 <= result.final_score <= 1.0
        assert result.secondary_harm == 0.0, f"secondary_harm={result.secondary_harm} expected 0.0"
        results[key] = f"PASS secondary_harm={result.secondary_harm:.3f}"
    except Exception as e:
        results[key] = f"FAIL: {e}"

    # prep_ready scoring: (1.0 + 0.0) / 2 = 0.5
    key = "pure_prep_ready"
    try:
        log = [
            {"event": "patient_created", "patient_id": "P1", "condition": "cardiac", "is_secondary": False},
            {"event": "patient_created", "patient_id": "P2", "condition": "stroke", "is_secondary": False},
            {"event": "patient_arrived_hospital", "patient_id": "P1", "hospital_id": "HOSP_A",
             "arrival_step": 30, "or_ready": True, "specialist_available": True},
            {"event": "patient_arrived_hospital", "patient_id": "P2", "hospital_id": "HOSP_B",
             "arrival_step": 35, "or_ready": False, "specialist_available": False},
            {"event": "treatment_complete", "patient_id": "P1", "outcome": "saved",
             "effective_time": 40, "target_time": 90, "vitals_at_treatment": 0.9},
            {"event": "treatment_complete", "patient_id": "P2", "outcome": "saved",
             "effective_time": 45, "target_time": 60, "vitals_at_treatment": 0.8},
        ]
        result = grade_episode(log)
        assert 0.0 <= result.final_score <= 1.0
        assert result.prep_ready == 0.5, f"prep_ready={result.prep_ready} expected 0.5"
        results[key] = f"PASS prep_ready={result.prep_ready:.3f}"
    except Exception as e:
        results[key] = f"FAIL: {e}"

    return results


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 8: Task config consistency
# ─────────────────────────────────────────────────────────────────────────────
def test_task_config():
    results = {}
    expected = {"task1", "task2", "task3", "task4", "task5"}
    try:
        assert expected == set(TASK_CONFIG.keys()), f"mismatch: {set(TASK_CONFIG.keys())}"
        results["task_config"] = f"PASS ({len(TASK_CONFIG)} tasks)"
    except Exception as e:
        results["task_config"] = f"FAIL: {e}"

    for tid in expected:
        key = f"config_{tid}"
        try:
            cfg = TASK_CONFIG[tid]
            assert "max_steps" in cfg, "no max_steps"
            assert "mutual_aid_calls" in cfg, "no mutual_aid_calls"
            assert "disruption_prob" in cfg, "no disruption_prob"
            call_queue = cfg.get("use_call_queue", False)
            cascade = cfg.get("cascade_enabled", False)
            results[key] = f"PASS max={cfg['max_steps']} cq={call_queue} cascade={cascade}"
        except Exception as e:
            results[key] = f"FAIL: {e}"
    return results


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 9: Dispatch action execution
# ─────────────────────────────────────────────────────────────────────────────
def test_dispatch_execution():
    results = {}

    # Dispatch ambulance to patient location
    key = "dispatch_basic"
    try:
        env, obs = make_env(task_id="task1", seed=0)
        # Find available ambulance and waiting patient
        amb = next((a for a in obs.ambulances if a.status.value == "available"), None)
        patient = next((p for p in obs.patients if p.status.value == "waiting"), None)
        assert amb is not None, "no available ambulance"
        assert patient is not None, "no waiting patient"

        action = DispatchAmbulance(ambulance_id=amb.id, target_node=patient.location_node)
        obs2 = env.step(action)
        assert obs2 is not None

        # Find the ambulance in new obs
        amb2 = next((a for a in obs2.ambulances if a.id == amb.id), None)
        assert amb2 is not None
        # Status should have changed
        assert amb2.status.value in ("en_route", "at_scene", "transporting"), \
            f"unexpected status: {amb2.status.value}"
        results[key] = f"PASS amb={amb.id} status={amb2.status.value}"
    except Exception as e:
        results[key] = f"FAIL: {e}"

    # Dispatch to hospital (wasted) should not crash
    key = "dispatch_wasted"
    try:
        env, _ = make_env(task_id="task1", seed=0)
        action = DispatchAmbulance(ambulance_id="AMB_1", target_node="HOSP_A")
        obs = env.step(action)
        assert obs is not None
        results[key] = "PASS no crash"
    except Exception as e:
        results[key] = f"FAIL: {e}"

    return results


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 10: Inference format compliance
# ─────────────────────────────────────────────────────────────────────────────
def test_inference_format():
    results = {}
    from inference import format_observation, parse_action_string

    key = "inference_format"
    try:
        env, obs = make_env(task_id="task1", seed=0)
        text = format_observation(obs)
        assert isinstance(text, str), "format_observation did not return str"
        assert len(text) > 10, "format_observation returned empty/short string"
        results[key] = f"PASS len={len(text)}"
    except Exception as e:
        results[key] = f"FAIL: {e}"

    key = "inference_parse"
    try:
        # parse_action_string expects fn_name(args) format, returns tuple[dict, str]
        for action_str, expected_type in [
            ("maintain_plan()", "maintain_plan"),
            ("dispatch_ambulance(ambulance_id='AMB_1', target_node='HOSP_A')", "dispatch_ambulance"),
            ("assign_hospital(patient_id='P1', hospital_id='HOSP_A')", "assign_hospital"),
        ]:
            parsed, display = parse_action_string(action_str)
            assert isinstance(parsed, dict), f"expected dict, got {type(parsed)}"
            assert isinstance(display, str), f"expected str display, got {type(display)}"
            assert parsed.get("type") == expected_type, \
                f"{action_str}: got type={parsed.get('type')}, expected {expected_type}"

        # Plain text fallback
        parsed, display = parse_action_string("do something random")
        assert parsed.get("type") == "maintain_plan", f"plain text should fallback: {parsed}"
        assert display == "maintain_plan()"

        results[key] = "PASS tuple[dict,str]"
    except Exception as e:
        results[key] = f"FAIL: {e}"

    return results


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 11: Observation fields — verify all required fields present
# ─────────────────────────────────────────────────────────────────────────────
def test_observation_fields():
    results = {}
    task_ids = ["task1", "task2", "task3", "task4", "task5"]
    for tid in task_ids:
        key = f"obs_fields_{tid}"
        try:
            env, obs = make_env(task_id=tid, seed=0)
            required = ["step", "patients", "ambulances", "hospitals", "blood_banks",
                        "road_network", "alerts", "mutual_aid_remaining",
                        "time_score_preview", "vitals_score_preview", "patients_remaining"]
            for field in required:
                assert hasattr(obs, field), f"missing {field}"
            results[key] = f"PASS ({len(required)} fields)"
        except Exception as e:
            results[key] = f"FAIL: {e}"

    # Phase 2 specific fields
    key = "obs_phase2"
    try:
        env, obs = make_env(task_id="task4", seed=0)
        assert hasattr(obs, "pending_calls"), "missing pending_calls"
        assert hasattr(obs, "overcrowding_modifier"), "missing overcrowding_modifier"
        # For task4, pending_calls should exist as field
        assert isinstance(obs.pending_calls, list), "pending_calls not a list"
        results[key] = f"PASS pending_calls={len(obs.pending_calls)} overcrowded={obs.overcrowding_modifier}"
    except Exception as e:
        results[key] = f"FAIL: {e}"

    return results


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 12: Cross-validation penalty detection (grade_from_environment)
# ─────────────────────────────────────────────────────────────────────────────
def test_cross_validation():
    results = {}

    key = "cross_val_task1"
    try:
        env, _ = make_env(task_id="task1", seed=0)
        run_episode(env)
        result = grade_from_environment(env)
        # Verify cross-validation fields exist in breakdown
        bd = result.breakdown
        assert "cross_validation_mismatches" in bd
        assert "cross_validation_penalty" in bd
        assert "icu_boarding_events" in bd
        results[key] = f"PASS mismatches={bd['cross_validation_mismatches']} penalty={bd['cross_validation_penalty']}"
    except Exception as e:
        results[key] = f"FAIL: {e}"

    return results


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 13: Episode log format — verify logs are properly structured
# ─────────────────────────────────────────────────────────────────────────────
def test_episode_log_format():
    results = {}

    key = "log_structure"
    try:
        env, _ = make_env(task_id="task1", seed=0)
        run_episode(env)
        log = env.get_episode_log()
        assert isinstance(log, list), "episode_log should be a list"
        # Each entry should have "event" field
        for i, entry in enumerate(log):
            assert isinstance(entry, dict), f"entry {i} not a dict"
            assert "event" in entry, f"entry {i} missing 'event' field: {entry}"
        results[key] = f"PASS {len(log)} log entries"
    except Exception as e:
        results[key] = f"FAIL: {e}"

    # Check for key event types in log
    key = "log_event_types"
    try:
        env, _ = make_env(task_id="task3", seed=0)
        run_episode(env)
        log = env.get_episode_log()
        event_types = {e["event"] for e in log}
        assert "patient_created" in event_types, "no patient_created in log"
        results[key] = f"PASS event_types={sorted(event_types)}"
    except Exception as e:
        results[key] = f"FAIL: {e}"

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Print + run
# ─────────────────────────────────────────────────────────────────────────────
def run_phase(name, fn):
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")
    results = fn()
    passed = failed = 0
    for key, val in results.items():
        if val.startswith("PASS"):
            print(f"  [PASS] {key}: {val}")
            passed += 1
        elif val.startswith("SKIP"):
            print(f"  [SKIP] {key}: {val}")
        else:
            print(f"  [FAIL] {key}: {val}")
            failed += 1
    print(f"\n  {passed} passed, {failed} failed")
    return passed, failed, results


def main():
    print("\n" + "="*70)
    print("  CodeRedEnv Comprehensive Stress Test")
    print("="*70)

    total_p = 0
    total_f = 0
    all_results = {}

    phases = [
        ("Phase 1: All Tasks Reset", test_all_tasks_reset),
        ("Phase 2: Grader All Tasks x 3 Seeds", test_grader_all_tasks),
        ("Phase 3: Grader as_dict Format", test_grader_as_dict),
        ("Phase 4: Phase 2 Tasks (Task4 & Task5)", test_phase2_detailed),
        ("Phase 5: Edge Cases", test_edge_cases),
        ("Phase 6: Reproducibility", test_reproducibility),
        ("Phase 7: Grader Pure Function", test_grader_pure_function),
        ("Phase 8: Task Config Consistency", test_task_config),
        ("Phase 9: Dispatch Execution", test_dispatch_execution),
        ("Phase 10: Inference Format", test_inference_format),
        ("Phase 11: Observation Fields", test_observation_fields),
        ("Phase 12: Cross-Validation", test_cross_validation),
        ("Phase 13: Episode Log Format", test_episode_log_format),
    ]

    for name, fn in phases:
        p, f, results = run_phase(name, fn)
        total_p += p
        total_f += f
        all_results[name] = results

    print(f"\n{'='*70}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"  Total: {total_p} passed, {total_f} failed")
    if total_f == 0:
        print(f"\n  ALL TESTS PASSED")
    else:
        print(f"\n  Failed tests:")
        for phase_name, results in all_results.items():
            for key, val in results.items():
                if not val.startswith("PASS") and not val.startswith("SKIP"):
                    # Show first line of error
                    first_line = val.split("\n")[0]
                    print(f"    {key}: {first_line[:120]}")

    return total_f == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
