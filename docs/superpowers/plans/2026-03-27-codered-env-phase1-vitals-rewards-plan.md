# CodeRedEnv Phase 1: Patient Vitals + Dense Rewards Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add patient deterioration (vitals_score 1.0→0.0), status escalation (waiting→deteriorating→critical), dense milestone rewards, early termination, and vitals_score_avg to the grader.

**Architecture:** Add `vitals_score` to the internal `Patient` dataclass (patient_manager) and the Pydantic `Patient` model (entities). Replace the empty `PatientManager.tick()` with deterioration logic. Snapshot vitals before/after tick to compute reward deltas. All changes are additive — existing tests remain valid.

**Tech Stack:** Python dataclasses (patient_manager), Pydantic v2 (entities/observations), pytest

---

## Files to Modify

| File | Changes |
|------|---------|
| `server/subsystems/constants.py` | Add vitals/reward constants |
| `server/subsystems/patient_manager.py` | Add `vitals_score` to Patient dataclass, replace empty `tick()` with deterioration logic, update `mark_deceased()` |
| `server/models/entities.py` | Add `vitals_score` field to Patient Pydantic model, add `DETERIORATING`/`CRITICAL` to PatientStatus enum |
| `server/models/observations.py` | Add `vitals_score_preview` to CodeRedObservation |
| `server/codered_environment.py` | Add `_prev_vitals`/`_prev_patient_status` state, call `patient_manager.tick()` with onset_steps, implement `_compute_step_reward()`, add early done check, update `_build_observation()` with vitals + preview |
| `server/grader.py` | Add `vitals_score_avg` to RubricResult dataclass and `grade_episode()` |
| `tests/test_patient_manager.py` | New tests for vitals decline, status escalation, death |
| `tests/test_environment.py` | New tests for dense reward, early termination |
| `tests/test_e2e.py` | Update assertions for vitals + time_score |

---

## Task 1: Add Constants

**Files:**
- Modify: `server/subsystems/constants.py`

Add these constants at the end of the file (after `TASK_CONFIG`):

- [ ] **Step 1: Add vitals and reward constants**

```python
# =============================================================================
# PATIENT VITALS — Phase 1
# =============================================================================

VITALS_INITIAL = 1.0
VITALS_STABLE_DECAY_RATE = 0.002   # per step in stable window (1 step = 1 min)
VITALS_DETERIORATING_THRESHOLD = 0.75
VITALS_CRITICAL_THRESHOLD = 0.4
VITALS_DEAD_THRESHOLD = 0.0

# Reward shaping
VITALS_DELTA_WEIGHT = 0.5
MILESTONE_REWARDS = {
    "dispatched": 0.05,
    "in_treatment": 0.10,
    "treated": 0.20,
    "deceased": -0.30,
}
REWARD_STEP_CLAMP = (-1.0, 1.0)
```

- [ ] **Step 2: Run tests to confirm nothing broke**

Run: `cd codered_env && uv run pytest tests/test_constants.py tests/test_patient_manager.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add server/subsystems/constants.py
git commit -m "feat(constants): add vitals and reward constants for phase 1"
```

---

## Task 2: Add Vitals to Patient Dataclass

**Files:**
- Modify: `server/subsystems/patient_manager.py`

- [ ] **Step 1: Add vitals_score field to Patient dataclass**

In the `@dataclass Patient` class, add after `arrival_hospital_step`:

```python
    vitals_score: float = 1.0
    _vitals_frozen: bool = False
```

- [ ] **Step 2: Add status constants for escalation**

At module level (after the `Patient` dataclass, before `PatientManager`):

```python
TERMINAL_STATUSES = frozenset({"treated", "deceased"})
```

- [ ] **Step 3: Replace empty tick() with deterioration logic**

Replace the empty `def tick(self) -> None: pass` with:

```python
    def tick(self, onset_steps: dict[str, int], step_count: int) -> None:
        """Advance patient vitals deterioration. Call once per environment step."""
        for patient in self.patients:
            if patient.status in TERMINAL_STATUSES or patient._vitals_frozen:
                continue

            effective_time = step_count - onset_steps.get(patient.id, patient.onset_step)
            target_time = {"cardiac": 90, "stroke": 60, "trauma": 60, "general": 120}.get(
                patient.condition, 60
            )

            if effective_time <= target_time:
                # Stable window: slow recovery, clamped to 1.0
                patient.vitals_score = min(1.0, patient.vitals_score + VITALS_STABLE_DECAY_RATE)
            else:
                # Post-target: linear fall from 1.0 to 0.0 over one target_time window
                overtime_ratio = (effective_time - target_time) / target_time
                patient.vitals_score = max(0.0, 1.0 - overtime_ratio)

            # Status escalation
            if patient.vitals_score <= VITALS_DETERIORATING_THRESHOLD:
                patient.status = "deteriorating"
            if patient.vitals_score <= VITALS_CRITICAL_THRESHOLD:
                patient.status = "critical"
            if patient.vitals_score <= VITALS_DEAD_THRESHOLD:
                self.mark_deceased(patient.id, reason="cardiac_arrest")
```

Note: the `tick()` signature changes from `() -> None` to `(onset_steps, step_count) -> None`. This is the ONLY breaking change.

- [ ] **Step 4: Update mark_treated() to freeze vitals**

```python
    def mark_treated(self, patient_id: str, treatment_complete_time: int) -> None:
        p = self.get(patient_id)
        if p:
            p.status = "treated"
            p.treatment_complete_time = treatment_complete_time
            p.outcome = "saved"
            p._vitals_frozen = True
```

- [ ] **Step 5: Update mark_deceased() to freeze vitals and set to 0**

```python
    def mark_deceased(self, patient_id: str, reason: str = "timeout") -> None:
        p = self.get(patient_id)
        if p:
            p.status = "deceased"
            p.outcome = "deceased"
            p.vitals_score = 0.0
            p._vitals_frozen = True
```

- [ ] **Step 6: Add onset_steps helper at reset()**

In `reset()`, after spawning patients, build and store onset_steps. Add to `__init__`:

```python
self._onset_steps: dict[str, int] = {}
```

And update `reset()` to populate it:

```python
def reset(self, task_id: str, rng: Optional[random.Random]) -> None:
    self._rng = rng or random.Random()
    self._task_id = task_id
    self._patient_counter = 0
    self.patients = []
    self._onset_steps = {}
    self._spawn_patients()
    # Populate onset_steps from spawned patients
    for p in self.patients:
        self._onset_steps[p.id] = p.onset_step
```

Also add a getter for onset_steps at the end of the class:

```python
    def get_onset_steps(self) -> dict[str, int]:
        return self._onset_steps.copy()
```

- [ ] **Step 7: Run tests**

Run: `cd codered_env && uv run pytest tests/test_patient_manager.py -v`
Expected: PASS (existing tests still work, new ones fail until Task 7)

- [ ] **Step 8: Commit**

```bash
git add server/subsystems/patient_manager.py
git commit -m "feat(patient_manager): add vitals_score and deterioration tick()"
```

---

## Task 3: Add Vitals to Pydantic Models

**Files:**
- Modify: `server/models/entities.py`

- [ ] **Step 1: Add DETERIORATING and CRITICAL to PatientStatus enum**

```python
class PatientStatus(str, Enum):
    WAITING = "waiting"
    DETERIORATING = "deteriorating"  # Phase 1
    CRITICAL = "critical"              # Phase 1
    DISPATCHED = "dispatched"
    TRANSPORTING = "transporting"
    TREATING = "treating"
    TREATED = "treated"
    DECEASED = "deceased"
```

- [ ] **Step 2: Add vitals_score to Patient Pydantic model**

```python
class Patient(BaseModel):
    # ... existing fields ...
    is_secondary: bool = False
    vitals_score: float = Field(default=1.0, ge=0.0, le=1.0)

    model_config = {"extra": "forbid"}
```

- [ ] **Step 3: Run tests**

Run: `cd codered_env && uv run pytest tests/test_openenv_validate.py tests/test_entities.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add server/models/entities.py
git commit -m "feat(models): add deteriorating/critical status and vitals_score to Patient"
```

---

## Task 4: Add vitals_score_preview to Observation

**Files:**
- Modify: `server/models/observations.py`

- [ ] **Step 1: Add vitals_score_preview to CodeRedObservation**

```python
    vitals_score_preview: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Average vitals of all active (non-terminal) patients"
    )
```

- [ ] **Step 2: Run tests**

Run: `cd codered_env && uv run pytest tests/test_openenv_validate.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add server/models/observations.py
git commit -m "feat(observation): add vitals_score_preview to CodeRedObservation"
```

---

## Task 5: Integrate Tick into Environment + Dense Rewards

**Files:**
- Modify: `server/codered_environment.py`

- [ ] **Step 1: Add prev_vitals and prev_patient_status to __init__**

In `__init__`, after `self._pending_mutual_aid`:

```python
self._prev_vitals: Dict[str, float] = {}
self._prev_patient_status: Dict[str, str] = {}
```

- [ ] **Step 2: Initialize prev snapshots in reset()**

After `self._pending_mutual_aid = {}`, add:

```python
self._prev_vitals = {}
self._prev_patient_status = {}
```

Then after the patient creation loop (after the `_episode_log.append` for `patient_created`), populate the snapshots:

```python
for p in self._patients:
    self._prev_vitals[p.id] = p.vitals_score
    self._prev_patient_status[p.id] = p.status
```

- [ ] **Step 3: Update _advance_time() to call patient_manager.tick()**

Find the line `self._patient_manager.tick()` in `_advance_time()`. Replace it with:

```python
self._patient_manager.tick(
    self._patient_manager.get_onset_steps(),
    self._state.step_count
)
```

- [ ] **Step 4: Implement _compute_step_reward()**

Replace the existing stub:

```python
def _compute_step_reward(self) -> float:
    """Dense reward: vitals delta shaping + milestone bonuses/penalties."""
    reward = 0.0
    from .subsystems.constants import (
        VITALS_DELTA_WEIGHT, MILESTONE_REWARDS, REWARD_STEP_CLAMP
    )

    for pid, prev in self._prev_vitals.items():
        curr = self._patient_manager.patients_dict.get(pid)
        if curr is None:
            continue
        # Vitals delta shaping
        reward += (curr.vitals_score - prev) * VITALS_DELTA_WEIGHT

    # Milestone bonuses/penalties
    for pid, prev_status in self._prev_patient_status.items():
        curr_patient = self._patient_manager.patients_dict.get(pid)
        if curr_patient is None:
            continue
        curr_status = curr_patient.status
        if prev_status == "waiting" and curr_status == "dispatched":
            reward += MILESTONE_REWARDS["dispatched"]
        if prev_status == "dispatched" and curr_status == "in_treatment":
            reward += MILESTONE_REWARDS["in_treatment"]
        if curr_status == "treated":
            reward += MILESTONE_REWARDS["treated"]
        if curr_status == "deceased":
            reward += MILESTONE_REWARDS["deceased"]

    # Snapshot for next step
    self._prev_vitals = {
        p.id: p.vitals_score for p in self._patient_manager.patients
    }
    self._prev_patient_status = {
        p.id: p.status for p in self._patient_manager.patients
    }

    lo, hi = REWARD_STEP_CLAMP
    return max(lo, min(hi, reward))
```

Note: `PatientManager.patients` is a `list[Patient]` dataclass, not a dict. Change `patients_dict` to access by ID. Add a helper property to `PatientManager`:

In `patient_manager.py`, add after `get()`:

```python
    @property
    def patients_dict(self) -> dict[str, Patient]:
        return {p.id: p for p in self.patients}
```

- [ ] **Step 5: Update _check_done() for early termination**

Find `_check_done()`. After the existing `non_terminal` logic, add:

```python
    # Early termination: all non-deceased patients have vitals_score <= 0
    critical_patients = [
        p for p in self._patients
        if p.status != "deceased" and p.vitals_score <= 0.0
    ]
    if critical_patients:
        alive = [p for p in self._patients if p.status not in ("treated", "deceased")]
        if all(p.vitals_score <= 0.0 for p in alive):
            self._state.all_patients_terminal = True
            return True
```

- [ ] **Step 6: Update _build_observation() to include vitals**

In `_build_observation()`, inside the `Patient(...)` construction for each patient, add:

```python
vitals_score=p.vitals_score,
```

Also, add `vitals_score_preview` to the `CodeRedObservation(...)` return:

```python
vitals_score_preview=round(
    sum(p.vitals_score for p in self._patients if p.status not in ("treated", "deceased"))
    / max(1, sum(1 for p in self._patients if p.status not in ("treated", "deceased"))),
    4
),
```

And add the vitals line to the patient observation text format (the `format_observation` in baseline.py reads from observation patients, not text — but for debug clarity, the patient objects in observation already have vitals_score):

```python
# In the text format (for baseline.py format_observation which reads from patient attrs):
# Already included via p.vitals_score in the Patient Pydantic model
# No text format change needed — baseline reads from Patient object fields
```

Also update the patient status mapping in `_map_patient_status` to handle new statuses:

```python
    def _map_patient_status(s):
        mapping = {
            "waiting": PatientStatus.WAITING,
            "deteriorating": PatientStatus.DETERIORATING,
            "critical": PatientStatus.CRITICAL,
            "dispatched": PatientStatus.DISPATCHED,
            "transporting": PatientStatus.TRANSPORTING,
            "in_treatment": PatientStatus.TREATING,
            "treating": PatientStatus.TREATING,
            "treated": PatientStatus.TREATED,
            "deceased": PatientStatus.DECEASED,
        }
        return mapping.get(s, PatientStatus.WAITING)
```

- [ ] **Step 7: Add patients_dict property to PatientManager**

In `server/subsystems/patient_manager.py`, add after `get()`:

```python
    @property
    def patients_dict(self) -> dict[str, Patient]:
        return {p.id: p for p in self.patients}
```

- [ ] **Step 8: Run tests**

Run: `cd codered_env && uv run pytest tests/test_environment.py tests/test_wired_environment.py -v`
Expected: Most PASS, some new tests fail until Task 7

- [ ] **Step 9: Commit**

```bash
git add server/codered_environment.py server/subsystems/patient_manager.py
git commit -m "feat(environment): integrate patient deterioration tick and dense reward"
```

---

## Task 6: Add vitals_score_avg to Grader

**Files:**
- Modify: `server/grader.py`

- [ ] **Step 1: Add vitals_score_avg to RubricResult dataclass**

```python
@dataclass
class RubricResult:
    time_score: float      # 0.0–1.0
    efficiency: float       # 0.0–1.0
    secondary_harm: float   # 0.0–1.0
    prep_ready: float      # 0.0–1.0
    mutual_aid_penalty: float  # 0.0–1.0, subtracted from final_score
    final_score: float    # weighted sum minus mutual_aid_penalty
    breakdown: dict        # human-readable per-axis details
    vitals_score_avg: float = 0.0  # Phase 1: informational only

    def as_dict(self) -> dict:
        return {
            "time_score": self.time_score,
            "efficiency": self.efficiency,
            "secondary_harm": self.secondary_harm,
            "prep_ready": self.prep_ready,
            "mutual_aid_penalty": self.mutual_aid_penalty,
            "final_score": self.final_score,
            "breakdown": self.breakdown,
            "vitals_score_avg": self.vitals_score_avg,
        }
```

- [ ] **Step 2: Compute vitals_score_avg in grade_episode()**

In `grade_episode()`, after building `patient_times` dict, add:

```python
    # =========================================================================
    # VITALS SCORE AVERAGE (informational — not yet in final_score)
    # =========================================================================
    # Capture vitals at treatment from treatment_complete events
    vitals_scores = []
    for entry in episode_log:
        if entry.get("event") == "treatment_complete":
            vitals_scores.append(entry.get("vitals_at_treatment", 1.0))
    vitals_score_avg = sum(vitals_scores) / len(vitals_scores) if vitals_scores else 1.0
```

- [ ] **Step 3: Pass vitals_score_avg to RubricResult constructor**

In the `return RubricResult(...)` call at the end of `grade_episode()`, add:

```python
    return RubricResult(
        time_score=round(time_score, 4),
        efficiency=round(efficiency, 4),
        secondary_harm=round(secondary_harm, 4),
        prep_ready=round(prep_ready, 4),
        mutual_aid_penalty=round(mutual_aid_penalty, 4),
        final_score=round(final_score, 4),
        breakdown=breakdown,
        vitals_score_avg=round(vitals_score_avg, 4),
    )
```

- [ ] **Step 4: Capture vitals_at_treatment in environment**

In `server/codered_environment.py`, when logging `treatment_complete` in `_advance_time()` (around line 240), add the current vitals:

```python
                        "vitals_at_treatment": patient.vitals_score,
```

Find the `treatment_complete` log entry and add:
```python
                        "vitals_at_treatment": patient.vitals_score,
```

Look for the `treatment_complete` event in `_advance_time()` around the surgery completion detection. It should look like:

```python
                        self._episode_log.append({
                            "step": self._state.step_count,
                            "patient_id": patient_id,
                            "event": "treatment_complete",
                            "effective_time": effective_time,
                            "target_time": target_time,
                            "vitals_at_treatment": patient.vitals_score,
                        })
```

- [ ] **Step 5: Run tests**

Run: `cd codered_env && uv run pytest tests/test_grader.py tests/test_integration.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add server/grader.py server/codered_environment.py
git commit -m "feat(grader): add vitals_score_avg informational axis"
```

---

## Task 7: Write Unit Tests for PatientManager

**Files:**
- Modify: `tests/test_patient_manager.py`

- [ ] **Step 1: Write test for vitals decline post-target**

Add to `tests/test_patient_manager.py`:

```python
def test_patient_vitals_decline_post_target():
    """Vitals decay after effective_time exceeds target_time."""
    from codered_env.server.subsystems.patient_manager import PatientManager
    pm = PatientManager()
    import random
    rng = random.Random(0)
    pm.reset(task_id="task1", rng=rng)  # task1: cardiac, target=90min
    patient = pm.patients[0]
    assert patient.vitals_score == 1.0

    onset_steps = {patient.id: patient.onset_step}

    # Steps 0-90: vitals stay near 1.0 (stable window)
    for step in range(1, 91):
        pm.tick(onset_steps, step)
    assert patient.vitals_score == pytest.approx(1.0, abs=0.05)

    # Steps 91-135: vitals decline linearly post-target
    pm.tick(onset_steps, 91)
    assert 0.9 < patient.vitals_score < 1.0

    pm.tick(onset_steps, 135)
    assert patient.vitals_score == pytest.approx(0.0, abs=0.05)


def test_patient_vitals_freeze_on_treatment():
    """Vitals freeze at treatment time, no further decay."""
    from codered_env.server.subsystems.patient_manager import PatientManager
    pm = PatientManager()
    import random
    rng = random.Random(0)
    pm.reset(task_id="task1", rng=rng)
    patient = pm.patients[0]
    onset_steps = {patient.id: patient.onset_step}

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
    from codered_env.server.subsystems.patient_manager import PatientManager
    pm = PatientManager()
    import random
    rng = random.Random(0)
    pm.reset(task_id="task1", rng=rng)  # cardiac target=90
    patient = pm.patients[0]
    onset_steps = {patient.id: patient.onset_step}

    # Initially waiting
    assert patient.status == "waiting"

    # Advance to deteriorating threshold (vitals <= 0.75)
    # At step 90+22.5 = ~112-113, vitals crosses 0.75
    for step in range(1, 120):
        pm.tick(onset_steps, step)
    assert patient.status in ("waiting", "deteriorating")

    # Advance to critical threshold (vitals <= 0.4)
    for step in range(120, 160):
        pm.tick(onset_steps, step)
    assert patient.status in ("deteriorating", "critical")


def test_patient_death_at_zero_vitals():
    """Patient marked deceased when vitals reach 0.0."""
    from codered_env.server.subsystems.patient_manager import PatientManager
    pm = PatientManager()
    import random
    rng = random.Random(0)
    pm.reset(task_id="task1", rng=rng)  # cardiac target=90, death at step 180
    patient = pm.patients[0]
    onset_steps = {patient.id: patient.onset_step}

    # Tick until death
    for step in range(1, 181):
        pm.tick(onset_steps, step)

    assert patient.status == "deceased"
    assert patient.vitals_score == 0.0
    assert patient.outcome == "deceased"
```

- [ ] **Step 2: Run tests**

Run: `cd codered_env && uv run pytest tests/test_patient_manager.py -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_patient_manager.py
git commit -m "test(patient_manager): add vitals decline, status escalation, and death tests"
```

---

## Task 8: Write Environment Tests + Update E2E

**Files:**
- Modify: `tests/test_environment.py`
- Modify: `tests/test_e2e.py`

- [ ] **Step 1: Write test for dense reward**

Add to `tests/test_environment.py`:

```python
def test_dense_reward_non_zero_for_active_steps():
    """Dense reward returns non-zero values when patient state changes."""
    from codered_env.server.codered_environment import CodeRedEnvironment
    from codered_env.server.models.actions import DispatchAmbulance
    env = CodeRedEnvironment()
    obs = env.reset(seed=0, task_id="task1")

    # Step 1: MaintainPlan (idle) — reward should be near 0
    prev_cum = env.state.cum_reward
    obs = env.step(DispatchAmbulance(ambulance_id="AMB_1", target_node=obs.patients[0].location_node))
    reward_delta = env.state.cum_reward - prev_cum
    # Dispatching gives milestone bonus — reward should be positive
    assert reward_delta > 0.0


def test_vitals_in_observation():
    """Observation includes vitals_score for each patient."""
    from codered_env.server.codered_environment import CodeRedEnvironment
    env = CodeRedEnvironment()
    obs = env.reset(seed=0, task_id="task1")
    assert len(obs.patients) == 1
    assert hasattr(obs.patients[0], "vitals_score")
    assert 0.0 <= obs.patients[0].vitals_score <= 1.0


def test_vitals_score_preview_in_observation():
    """Observation includes vitals_score_preview."""
    from codered_env.server.codered_environment import CodeRedEnvironment
    env = CodeRedEnvironment()
    obs = env.reset(seed=0, task_id="task1")
    assert hasattr(obs, "vitals_score_preview")
    assert 0.0 <= obs.vitals_score_preview <= 1.0
```

- [ ] **Step 2: Update E2E tests**

Open `tests/test_e2e.py` and update the grader assertion to check vitals_score_avg:

```python
def test_grader_computes_score():
    for task_id in ["task1", "task2", "task3"]:
        env = CodeRedEnvironment()
        env.reset(seed=0, task_id=task_id)
        for _ in range(10):
            env.step(MaintainPlan())
            if env.state.step_count >= env.state.max_steps:
                break
        result = grade_from_environment(env)
        assert 0.0 <= result.final_score <= 1.0
        assert 0.0 <= result.time_score <= 1.0
        assert 0.0 <= result.efficiency <= 1.0
        assert 0.0 <= result.secondary_harm <= 1.0
        assert 0.0 <= result.prep_ready <= 1.0
        assert 0.0 <= result.mutual_aid_penalty <= 1.0
        assert 0.0 <= result.vitals_score_avg <= 1.0  # Phase 1
```

Also update `test_task1_e2e` to verify vitals_score is present:

```python
def test_task1_e2e():
    env = CodeRedEnvironment()
    obs = env.reset(seed=42, task_id="task1")
    assert obs.step == 1
    assert len(obs.patients) == 1
    assert obs.patients[0].condition.value == "cardiac"
    assert obs.patients[0].vitals_score == 1.0  # Phase 1: starts at 1.0
    for _ in range(10):
        obs = env.step(MaintainPlan())
        if env.state.step_count >= env.state.max_steps:
            break
    assert env.state.step_count > 0
```

- [ ] **Step 3: Run full test suite**

Run: `cd codered_env && uv run pytest tests/ -v --tb=short 2>&1 | tail -20`
Expected: ALL 89+ tests PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_environment.py tests/test_e2e.py
git commit -m "test: add environment vitals tests and update E2E assertions"
```

---

## Self-Review Checklist

1. **Spec coverage:** All 5 spec sections implemented?
   - Section 1 (PatientStatus + vitals field): Tasks 3, 2 ✓
   - Section 2 (PatientManager.tick + deterioration): Task 2 ✓
   - Section 3 (Environment integration + dense rewards): Task 5 ✓
   - Section 4 (Grader vitals_score_avg): Task 6 ✓
   - Section 5 (Constants): Task 1 ✓
   - Section 6 (Tests): Tasks 7, 8 ✓

2. **Placeholder scan:** No TBD, TODO, or "fill in later" in any step.

3. **Type consistency:**
   - `PatientManager.tick(onset_steps, step_count)` — consistent across Tasks 2 and 5
   - `Patient.vitals_score` — added to both dataclass and Pydantic model
   - `PatientStatus` enum additions `DETERIORATING`/`CRITICAL` — mapped in `_map_patient_status()`
   - `RubricResult.vitals_score_avg` — added to dataclass + `as_dict()` + `grade_episode()`
   - `patients_dict` property added to PatientManager for O(1) lookup in reward computation

4. **Breaking changes:** Only one — `PatientManager.tick()` signature changes from `() -> None` to `(onset_steps, step_count) -> None`. All callers updated in Task 5.
