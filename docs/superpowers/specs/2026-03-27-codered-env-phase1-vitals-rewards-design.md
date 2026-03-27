# CodeRedEnv Phase 1: Patient Vitals + Dense Rewards

**Date:** 2026-03-27
**Status:** Approved
**Scope:** Patient deterioration model + dense reward signal (Phase 1 of 4)

---

## Overview

Layer patient deterioration and a dense reward signal into the existing environment without changing the grader. The existing `PATIENT_TARGET_TIMES` becomes the mortality clock. All changes are additive — existing tests remain valid.

---

## 1. Patient Vitals Model

### Field Addition

Add `vitals_score: float` to the `Patient` Pydantic model in `server/models/entities.py`:

- Range: `1.0` (perfect health) → `0.0` (deceased)
- Default on spawn: `1.0`
- Frozen (no further update) once `status` is `treated` or `deceased`
- Readable in observation and by the grader

### PatientStatus Enum

Add `critical` escalation status. Final enum:

```
waiting → deteriorating → critical → dispatched → in_treatment → treated / deceased
```

Transitions:
- `vitals_score <= 0.75`: status = `deteriorating`
- `vitals_score <= 0.4`: status = `critical`
- `vitals_score <= 0.0`: patient dies (see death rule below)

---

## 2. PatientManager Tick — Deterioration Logic

Replace the empty `PatientManager.tick()` with the following. It receives `onset_steps: dict[str, int]` (patient_id → step when patient spawned) and `step_count` from the environment.

### Deterioration Formula

For each patient where `status` is not `treated` or `deceased`:

```
effective_time = step_count - onset_steps[patient_id]   # minutes (1 step = 1 min)
target_time    = PATIENT_TARGET_TIMES[condition]         # cardiac=90, stroke=60, trauma=60, general=120

if effective_time <= target_time:
    # Stable window: slight natural decay (~0.2% per step), clamped to 1.0
    patient.vitals_score = min(1.0, patient.vitals_score + 0.002)
else:
    # Post-target deterioration: linear fall from 1.0 to 0.0 over one target_time window
    overtime_ratio = (effective_time - target_time) / target_time
    patient.vitals_score = max(0.0, 1.0 - overtime_ratio)
```

At `effective_time = 2 * target_time`, vitals_score = 0.0 and patient dies.

### Status Escalation

After vitals update, apply status escalation:

```
if patient.vitals_score <= 0.75: patient.status = deteriorating
if patient.vitals_score <= 0.4: patient.status = critical
if patient.vitals_score <= 0.0 and patient.status not in [treated, deceased]:
    mark_deceased(patient_id, reason="cardiac_arrest")
```

### Treatment Freeze

`mark_treated()` freezes `vitals_score` at its current value. No further deterioration. Patient is counted as saved regardless of remaining vitals.

### Death

`mark_deceased()` sets `status = deceased` and `vitals_score = 0.0`. Patient is removed from active deterioration tracking.

---

## 3. Environment Integration

### Environment Calls PatientManager.tick()

In `CodeRedEnvironment.step()`, call `patient_manager.tick(onset_steps, env.state.step_count)` each step, after the action is executed but before observation building.

Store `onset_steps` dict at `reset()` time — `patient_manager.patients[pid].onset_step` for each spawned patient.

### Dense Reward Computation

Add `prev_vitals: dict[str, float]` as instance state on `CodeRedEnvironment`. Snapshot before `tick()` runs, compute reward after:

```python
def _compute_step_reward(self) -> float:
    reward = 0.0
    for pid, prev in self._prev_vitals.items():
        curr = self.patient_manager.patients[pid].vitals_score
        # Vitals delta shaping
        reward += (curr - prev) * 0.5

    # Milestone bonuses
    for pid, prev_status in self._prev_patient_status.items():
        curr_status = self.patient_manager.patients[pid].status.value
        if prev_status == "waiting" and curr_status == "dispatched":
            reward += 0.05
        if prev_status == "dispatched" and curr_status == "in_treatment":
            reward += 0.10
        if curr_status == "treated":
            reward += 0.20
        if curr_status == "deceased":
            reward -= 0.30

    self._prev_vitals = {pid: p.vitals_score for pid, p in self.patient_manager.patients.items()}
    self._prev_patient_status = {pid: p.status.value for pid, p in self.patient_manager.patients.items()}

    return max(-1.0, min(1.0, reward))  # per-step clamp
```

### Early Termination

Add to `_check_done()`: if all non-deceased patients have `vitals_score <= 0.0`, return `done = True`.

### Observation Update

In `_build_observation()`, add to patient display lines:
```
f"  {p.patient_id}: ... | vitals={p.vitals_score:.2f}"
```

Also add `vitals_score_preview: float` to `CodeRedObservation` — average vitals of all active patients.

---

## 4. Grader Changes

No change to the existing 4-axis rubric (time_score, efficiency, secondary_harm, prep_ready).

Add informational axis reported in `RubricResult`:

```
vitals_score_avg: float  # average vitals at treatment time across all treated patients
```

This is reported but not yet weighted into `final_score`. Can be added in a future phase.

---

## 5. Constants

New constants in `subsystems/constants.py`:

```python
VITALS_STABLE_DECAY_RATE = 0.002   # per step in stable window (1 step = 1 min)
VITALS_DETERIORATING_THRESHOLD = 0.75
VITALS_CRITICAL_THRESHOLD = 0.4
VITALS_DEAD_THRESHOLD = 0.0
MILESTONE_REWARDS = {
    "dispatched": 0.05,
    "in_treatment": 0.10,
    "treated": 0.20,
    "deceased": -0.30,
}
VITALS_DELTA_WEIGHT = 0.5
REWARD_CLAMP = (-1.0, 1.0)
```

---

## 6. Tests

### Unit Tests

| Test File | What It Tests |
|-----------|--------------|
| `test_patient_manager.py` | `test_patient_vitals_decline_post_target` — verifies vitals decay after `effective_time > target_time`, freeze on treatment, death at 0.0 |
| `test_patient_manager.py` | `test_patient_status_escalation` — verifies waiting→deteriorating→critical transitions at correct thresholds |
| `test_patient_manager.py` | `test_patient_death_at_zero_vitals` — patient marked deceased when vitals reach 0 |
| `test_environment.py` | `test_dense_reward_computed_per_step` — `_compute_step_reward` returns non-zero for active steps, zero for MaintainPlan when idle |
| `test_environment.py` | `test_early_done_when_all_dead` — episode terminates early when all patients are deceased |

### E2E Test Updates

- `test_e2e.py::test_task1_e2e` — update to verify positive `time_score` for patient treated before target_time
- `test_e2e.py::test_grader_computes_score` — add assertion for `vitals_score_avg` field in result

### All Existing Tests

Pass unchanged — vitals default to 1.0, deterioration only activates post-target_time, all existing status transitions and action effects preserved.

---

## 7. Files to Modify

| File | Changes |
|------|---------|
| `server/models/entities.py` | Add `vitals_score` field to `Patient`, add `critical` to `PatientStatus` |
| `server/models/observations.py` | Add `vitals_score_preview` to `CodeRedObservation` |
| `subsystems/constants.py` | Add vitals/reward constants |
| `subsystems/patient_manager.py` | Replace empty `tick()` with deterioration logic, add `mark_deceased` with reason parameter |
| `server/codered_environment.py` | Add `vitals_score` to observation, `_prev_vitals` tracking, `_compute_step_reward()`, early termination, pass `onset_steps` to tick |
| `server/grader.py` | Add `vitals_score_avg` informational axis |
| `tests/test_patient_manager.py` | New deterioration/status/death tests |
| `tests/test_environment.py` | New reward/termination tests |
| `tests/test_e2e.py` | Update assertions for vitals + time_score |

---

## 8. Backward Compatibility

- `PATIENT_TARGET_TIMES` is the single source of truth for deterioration timing
- `mark_treated()` and `mark_deceased()` signatures unchanged (reason param is new but optional)
- All existing tests pass without modification
- Grader behavior unchanged — vitals are parallel to the existing scoring
- Phase 2 (resource constraints), 3 (emergent complexity), and 4 (additional improvements) build on top of this foundation

---

## 9. Open Questions (Deferred to Future Phases)

- Should the agent see individual vitals or just aggregate `vitals_score_preview`? (Agent sees per-patient vitals in Phase 1)
- Blood type urgency correlation (trauma → likely need O_NEG)? → Phase 2 (blood bank)
- Variable surgery duration by condition/severity? → Phase 2 (hospital)
- Cascading disruptions (accident → secondary patients)? → Phase 3
- Stochastic mortality roll (D2)? → Phase 2
