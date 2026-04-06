# CodeRedEnv — Audit Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all 5 critical blockers, 9 major issues, 8 minor issues, and 5 exploit patches from the production audit, ensuring OpenEnv API compliance and reproducible deterministic scoring.

**Architecture:** Each task is self-contained with a failing test first (TDD). Fixes are applied in dependency order: P0 blockers first, then P1, then P2, then exploit hardening. The HF Space version gets its own parallel track.

**Tech Stack:** Python 3.10+, pytest, Pydantic v2, FastAPI, OpenEnv core

---

## File Map

| File | Role |
|------|------|
| `codered_env/server/codered_environment.py` | Local environment — gets ALL fixes |
| `codered-hfspace/server/codered_environment.py` | HF Space — gets `done` fix + sync |
| `codered_env/server/grader.py` | Grading — empty episode guard, preempt_or, cascade |
| `codered_env/server/subsystems/constants.py` | Static config — cascade_enabled flags |
| `codered_env/server/subsystems/patient_manager.py` | Patient lifecycle — onset_step sync |
| `codered_env/server/subsystems/ambulance_manager.py` | Ambulance fleet — patient linkage, empty scene |
| `codered_env/server/subsystems/blood_bank.py` | Blood supply — emergency caps |
| `codered_env/server/subsystems/cascade_engine.py` | Cascades — unbounded surge cap |
| `codered_env/server/subsystems/mutual_aid.py` | Mutual aid — arrival step logging |
| `codered_env/inference.py` | LLM agent — temperature fix |
| `codered_env/tests/test_grader.py` | Existing grader tests |
| `codered_env/tests/test_environment.py` | Existing env tests |
| `codered_env/tests/test_exploit_*` | New exploit detection tests (create) |

---

## Phase 0: Baseline — Verify Tests Pass First

Before touching anything, confirm the existing test suite passes.

- [ ] **Step 1: Run full test suite and record results**

```bash
cd C:/Darsh/Scaler/codered_env
python -m pytest tests/ -x -q --tb=short 2>&1 | tail -20
```

Expected: All tests pass (183 tests per session memory S925).

---

## Phase 1: P0 — Critical Blockers

---

### Task 1: HF Space `_build_observation()` missing `done` parameter

**Files:**
- Modify: `codered-hfspace/server/codered_environment.py:1042`
- Modify: `codered-hfspace/server/codered_environment.py:166` (reset return)
- Modify: `codered-hfspace/server/codered_environment.py:192` (step return)

- [ ] **Step 1: Write the failing test**

Create `codered_env/tests/test_hfspace_done_flag.py`:

```python
"""Test that done flag flows correctly through step() for both local and HF Space."""
import pytest
from server.codered_environment import CodeRedEnvironment


def test_step_returns_done_true_when_terminated():
    """When _check_done() returns True, step() must pass done=True to observation."""
    env = CodeRedEnvironment()
    obs = env.reset(seed=0, task_id="task1")

    # Run to termination (step1 cardiac patient: 30 steps or all treated)
    for _ in range(31):
        from server.models.actions import MaintainPlan
        obs = env.step(MaintainPlan())
        if env._check_done():
            # CRITICAL: done field in observation must be True
            assert obs.done is True, (
                f"Expected obs.done=True when episode terminates, got obs.done={obs.done}"
            )
            break
    else:
        pytest.fail("Episode did not terminate after 31 steps")


def test_reset_returns_done_false():
    """reset() should always return observation with done=False."""
    env = CodeRedEnvironment()
    obs = env.reset(seed=0, task_id="task1")
    assert obs.done is False, "reset() must return done=False"


def test_build_observation_accepts_done_parameter():
    """_build_observation() must accept a done kwarg."""
    env = CodeRedEnvironment()
    env.reset(seed=0, task_id="task1")
    # Must not raise TypeError
    obs = env._build_observation(done=True)
    assert obs.done is True
    obs2 = env._build_observation(done=False)
    assert obs2.done is False
```

- [ ] **Step 2: Run test on local version — should PASS (local already fixed)**

```bash
cd C:/Darsh/Scaler/codered_env
python -m pytest tests/test_hfspace_done_flag.py -v 2>&1
```

Expected: PASS (local already has the fix from session memory S919-921).

- [ ] **Step 3: Verify HF Space fails before fix**

```bash
# Temporarily monkey-patch to simulate HF Space behavior
cd C:/Darsh/Scaler/codered-hfspace
python -c "
from server.codered_environment import CodeRedEnvironment
from server.models.actions import MaintainPlan
env = CodeRedEnvironment()
obs = env.reset(seed=0, task_id='task1')
# Run to step 5, then check done flag
for i in range(5):
    obs = env.step(MaintainPlan())
print('obs.done at step 5:', obs.done)
print('env._check_done():', env._check_done())
"
```

Expected: `obs.done=False` even when `_check_done()=True` (if we can force it), or `TypeError` if signature doesn't accept `done`.

- [ ] **Step 4: Fix HF Space codered_environment.py**

**Edit 1:** Line 1042 — change signature:
```python
# BEFORE:
def _build_observation(self) -> CodeRedObservation:
# AFTER:
def _build_observation(self, done: bool = False) -> CodeRedObservation:
```

**Edit 2:** Line 1195 (approximately) — add `done=done` to the returned CodeRedObservation inside `_build_observation()`. Find the `return CodeRedObservation(` line and add `done=done,` as the first field.

**Edit 3:** Line 166 — reset return:
```python
# BEFORE:
return self._build_observation()
# AFTER:
return self._build_observation(done=False)
```

**Edit 4:** Line 192 — step return:
```python
# BEFORE:
return self._build_observation()
# AFTER:
return self._build_observation(done=done)
```

- [ ] **Step 5: Run test on HF Space version**

```bash
cd C:/Darsh/Scaler/codered-hfspace
python -m pytest ../codered_env/tests/test_hfspace_done_flag.py -v 2>&1
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
cd C:/Darsh/Scaler
git add codered-hfspace/server/codered_environment.py codered_env/tests/test_hfspace_done_flag.py
git commit -m "fix(hfspace): add done parameter to _build_observation() — fixes OpenEnv termination detection
"
```

---

### Task 2: Phase 2 ambulance-patient linkage broken

**Files:**
- Modify: `codered_env/server/codered_environment.py`
- Create test: `codered_env/tests/test_phase2_dispatch.py`

- [ ] **Step 1: Write the failing test**

```python
"""Test Phase 2 dispatch: ambulance must carry patient on arrival."""
import pytest
from server.codered_environment import CodeRedEnvironment
from server.models.actions import DispatchALS, DispatchBLS, TriageCall, MaintainPlan


def _run_phase2_episode(task_id="task4", seed=0, max_steps=10):
    """Run a minimal Phase 2 episode and return patient statuses."""
    env = CodeRedEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)

    # Find first pending call
    call_id = None
    if obs.pending_calls:
        call_id = obs.pending_calls[0].call_id

    ambulance_id = None
    if call_id:
        # Find an ALS ambulance
        for a in obs.ambulances:
            if a.equipment.value == "ALS" and a.status.value == "available":
                ambulance_id = a.id
                break

    if call_id and ambulance_id:
        # Triage + dispatch
        obs = env.step(TriageCall(call_id=call_id, decision="dispatch_als", ambulance_id=ambulance_id))

    # Advance steps until ambulance arrives
    for _ in range(max_steps):
        done = env._check_done()
        if done:
            break
        obs = env.step(MaintainPlan())

    return env


def test_phase2_dispatch_patient_has_ambulance():
    """After dispatching ALS to a call, the spawned patient must have assigned_ambulance set."""
    env = _run_phase2_episode(seed=42)

    # Check all patients with dispatch_call_id have assigned_ambulance set
    patients_with_call = [p for p in env._patients if getattr(p, 'dispatch_call_id', None) is not None]

    for p in patients_with_call:
        # The ambulance should have this patient's ID OR the patient should have ambulance assigned
        # BUG: Neither is currently set — this test FAILS
        assert p.assigned_ambulance is not None, (
            f"Patient {p.id} spawned from call {p.dispatch_call_id} has no assigned ambulance. "
            f"Ambulance-patient linkage is broken in Phase 2 dispatch."
        )


def test_phase2_dispatch_ambulance_carries_patient():
    """Dispatched ambulance must have patient_id set so arrival detection works."""
    env = _run_phase2_episode(seed=42)

    # Find ambulance that was dispatched to a call
    for amb_id, amb in env._ambulance_manager.all().items():
        if getattr(amb, 'patient_id', None) is not None:
            # Ambulance has a patient — check if it's linked to a patient in the system
            patient = env._patient_manager.get(amb.patient_id)
            if patient and patient.status not in ("treated", "deceased"):
                assert patient is not None  # reachable
                return  # found valid linkage

    # BUG: If no ambulance has a patient_id, the dispatch was broken
    pytest.fail(
        "No ambulance has patient_id set after Phase 2 dispatch. "
        "dispatch_als/bls must set ambulance.patient_id so arrival detection works."
    )
```

- [ ] **Step 2: Run test — expect FAIL**

```bash
cd C:/Darsh/Scaler/codered_env
python -m pytest tests/test_phase2_dispatch.py -v 2>&1
```

Expected: FAIL — `assigned_ambulance is None` and `patient_id not set on ambulance`.

- [ ] **Step 3: Fix `_do_triage_call()` — Phase 2 ambulance-patient linkage**

In `codered_env/server/codered_environment.py`, find `_do_triage_call()`. After the successful `dispatch_als/bls` block, add patient linking:

```python
# Find the _do_triage_call() method, around line ~850
# In the block handling decision in ("dispatch_als", "dispatch_bls"):
# AFTER result = self._ambulance_manager.dispatch(...)
# ADD the following before the return:

# FIX: Track call→patient linkage for arrival detection
# The patient hasn't spawned yet (will spawn when ambulance arrives or countdown expires).
# We store a pending linkage that the ambulance tick resolves.
self._pending_call_to_patient[action.call_id] = {
    "ambulance_id": action.ambulance_id,
    "location_node": call["location_node"],
    "spawned": False,
}

# Also set ambulance's target_node so it knows where it's going
amb = self._ambulance_manager.get(action.ambulance_id)
if amb:
    amb.target_node = call["location_node"]
    # Link ambulance to call via patient_id placeholder
    amb.patient_id = f"CALL:{action.call_id}"  # resolved when patient spawns

# And set ambulance on the call dict for lookup
call["assigned_ambulance_id"] = action.ambulance_id
```

Also initialize `self._pending_call_to_patient: Dict[str, Dict] = {}` in `__init__()`.

- [ ] **Step 4: Add `_pending_call_to_patient` initialization to `__init__()`**

```python
# In __init__(), add:
self._pending_call_to_patient: Dict[str, Dict] = {}
```

- [ ] **Step 5: Fix ambulance `tick()` to resolve pending patient linkage**

In `server/subsystems/ambulance_manager.py`, modify `tick()`:

```python
def tick(self, pending_call_to_patient=None) -> None:
    """
    pending_call_to_patient: dict of call_id → {ambulance_id, location_node, spawned}
    If provided, resolves CALL:xxx placeholder patient_ids to actual patient IDs.
    """
    for amb in self._ambulances.values():
        if amb.status == "on_scene" and amb.scene_minutes_remaining > 0:
            amb.scene_minutes_remaining -= 1
            if amb.scene_minutes_remaining == 0:
                # Scene time expired
                if amb.patient_id is not None:
                    amb.arrived_with_patient = True
                # FIX: If patient_id is a placeholder (CALL:xxx), resolve it
                    if pending_call_to_patient and amb.patient_id.startswith("CALL:"):
                        call_id = amb.patient_id.replace("CALL:", "")
                        entry = pending_call_to_patient.get(call_id, {})
                        if not entry.get("spawned"):
                            # Patient hasn't spawned yet — spawn them now synchronously
                            # Signal back to environment via a callback or flag
                            pass
                amb.status = "returning"
                amb.route = []
                amb.target_node = amb.base_node
        elif amb.status in ("en_route", "returning") and amb.eta_minutes > 0:
            amb.eta_minutes -= 1
            if amb.eta_minutes == 0:
                prev_status = amb.status
                amb.status = "on_scene"
                if amb.patient_id is not None and prev_status == "en_route":
                    amb.scene_minutes_remaining = SCENE_TIME
                # FIX: resolve CALL:xxx placeholder
                if pending_call_to_patient and amb.patient_id and amb.patient_id.startswith("CALL:"):
                    call_id = amb.patient_id.replace("CALL:", "")
                    # Try to find spawned patient for this call
                    entry = pending_call_to_patient.get(call_id, {})
                    if entry.get("spawned"):
                        amb.patient_id = entry["spawned_patient_id"]
                    # Otherwise leave placeholder — will be resolved by force-spawn or environment
```

- [ ] **Step 6: Update `mutual_aid_manager.tick()` call signature**

In `codered_environment.py` `_advance_time()`, update the `mutual_aid_manager.tick()` call to pass the callback. Actually the callback already handles arrivals. The key fix is that `_do_triage_call()` must set `ambulance.patient_id` — which I've added.

- [ ] **Step 7: Fix `_spawn_patient_from_call()` to back-link to ambulance**

```python
# In _spawn_patient_from_call(), ADD:
# Link the patient to the ambulance that was dispatched
call_assigned_amb = call.get("assigned_ambulance_id")
if call_assigned_amb:
    patient.assigned_ambulance = call_assigned_amb
    amb = self._ambulance_manager.get(call_assigned_amb)
    if amb:
        amb.patient_id = patient.id  # resolve the placeholder
```

Also mark the pending entry as spawned:
```python
pending_entry = self._pending_call_to_patient.get(call_id)
if pending_entry:
    pending_entry["spawned"] = True
    pending_entry["spawned_patient_id"] = patient.id
```

- [ ] **Step 8: Run test again**

```bash
cd C:/Darsh/Scaler/codered_env
python -m pytest tests/test_phase2_dispatch.py -v 2>&1
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add codered_env/server/codered_environment.py codered_env/server/subsystems/ambulance_manager.py codered_env/tests/test_phase2_dispatch.py
git commit -m "fix(phase2): ambulance-patient linkage in dispatch_als/bls — set ambulance.patient_id and resolve on patient spawn
"
```

---

### Task 3: Empty episode perfect score exploit

**Files:**
- Modify: `codered_env/server/grader.py`
- Create test: `codered_env/tests/test_grader_empty_episode.py`

- [ ] **Step 1: Write the failing test**

```python
"""Test that episodes with zero patients get score 0, not 1.0."""
import pytest
from server.grader import grade_episode


def test_empty_episode_gets_zero_score():
    """Episode with no patients should NOT get perfect score."""
    # Simulate an episode where no patients were ever created or processed
    empty_log = [
        {"step": 0, "event": "episode_started"},
        # No patient_created, no treatment_complete, no patient_deceased
    ]

    result = grade_episode(empty_log)

    # BUG: Currently returns time_score=1.0, efficiency=1.0, secondary_harm=1.0,
    # prep_ready=1.0, final_score=1.0
    # FIX: Should return final_score=0.0 with reason "no_patients_processed"
    assert result.final_score < 0.1, (
        f"Empty episode got final_score={result.final_score}, expected < 0.1. "
        f"An agent doing nothing must not get a perfect score."
    )
    assert "no_patients_processed" in str(result.breakdown.get("reason", "")), (
        f"Empty episode breakdown: {result.breakdown}"
    )


def test_noop_agent_task4_gets_low_score():
    """A maintain_plan-only agent on task4 (call queue) should get near-zero."""
    # Simulate: 5 calls arrive, no dispatches, all calls expire and spawn patients who die
    log = []
    for step in range(5):
        log.append({
            "step": step * 8,
            "call_id": f"CALL_{step}",
            "event": "call_received",
            "category": "chest_pain",
            "location": "LAJPAT_NAGAR",
        })
    # All patients die (timeout)
    for i in range(5):
        log.append({
            "step": 45,
            "patient_id": f"P{i+1}",
            "event": "patient_created",
            "condition": "cardiac",
            "is_secondary": False,
            "target_time": 90,
        })
        log.append({
            "step": 45,
            "patient_id": f"P{i+1}",
            "event": "patient_deceased",
            "reason": "timeout",
            "effective_time": 0,
            "target_time": 90,
        })

    result = grade_episode(log)
    # Time score: all 5 patients dead → 0.0. But no dispatch actions → efficiency=1.0.
    # final_score ≈ 0.32*0 + 0.16*1 + 0.16*1 + 0.16*0 + 0.10*1 + 0.10*1 = 0.68
    # This is too high for a no-op agent.
    # After the fix: no_patients_processed → score = 0.0
    assert result.final_score < 0.3, (
        f"No-op agent got final_score={result.final_score}, expected < 0.3"
    )
```

- [ ] **Step 2: Run test — expect FAIL on first test**

```bash
cd C:/Darsh/Scaler/codered_env
python -m pytest tests/test_grader_empty_episode.py -v 2>&1
```

Expected: FAIL — `final_score=1.0` for empty episode.

- [ ] **Step 3: Fix `grade_episode()` in grader.py**

At the start of `grade_episode()`:

```python
def grade_episode(episode_log: list[dict]) -> RubricResult:
    """Grade a CodeRedEnv episode using the 4-axis rubric."""

    # =========================================================================
    # ANTI-EXPLOIT: Detect empty/no-action episodes
    # =========================================================================
    patient_events = {
        e["patient_id"]
        for e in episode_log
        if e.get("event") in ("patient_created", "treatment_complete", "patient_deceased")
    }

    # If no patients were ever created (e.g., task4 before first call spawns):
    if len(patient_events) == 0:
        return RubricResult(
            time_score=0.0,
            efficiency=1.0,
            secondary_harm=1.0,
            prep_ready=1.0,
            mutual_aid_penalty=0.0,
            final_score=0.0,
            breakdown={"reason": "no_patients_processed", "anti_exploit": True},
            vitals_score_avg=1.0,
            cascade_score=1.0,
        )

    # Track whether any meaningful action was taken
    dispatch_events = [
        e for e in episode_log
        if e.get("event") in ("dispatch", "action_dispatch")
    ]
    treatment_events = [
        e for e in episode_log
        if e.get("event") in ("treatment_complete", "treatment_started")
    ]

    # If patients exist but no dispatches and no treatments (pure no-op):
    if len(dispatch_events) == 0 and len(treatment_events) == 0:
        all_deceased = all(
            e.get("event") == "patient_deceased"
            for e in episode_log
            if e.get("event") in ("patient_deceased", "treatment_complete")
        )
        if all_deceased and len(patient_events) > 0:
            # All patients died with no agent action → severe penalty
            time_score = 0.0
            breakdown_extra = {"anti_exploit": "no_action_all_deceased"}
        else:
            breakdown_extra = {"anti_exploit": "no_dispatches"}
    else:
        breakdown_extra = {}

    # ... rest of grading logic ...
```

- [ ] **Step 4: Run test**

```bash
cd C:/Darsh/Scaler/codered_env
python -m pytest tests/test_grader_empty_episode.py -v 2>&1
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add codered_env/server/grader.py codered_env/tests/test_grader_empty_episode.py
git commit -m "fix(grader): zero-patient episode guard — no-op agent gets 0.0 not 1.0
"
```

---

## Phase 2: P1 — Major Issues

---

### Task 4: ICU bed leak on patient death

**Files:**
- Modify: `codered_env/server/codered_environment.py`
- Modify: `codered_env/tests/stress_test_env.py`

- [ ] **Step 1: Write the failing test**

```python
"""Test that ICU beds are released when patients die."""
import pytest
from server.codered_environment import CodeRedEnvironment
from server.models.actions import MaintainPlan


def test_icu_bed_released_on_death():
    """When a patient with admitted ICU status dies, the ICU bed must be released."""
    env = CodeRedEnvironment()
    env.reset(seed=0, task_id="task1")

    # Manually consume an ICU bed and mark patient as admitted
    hosp_a = env._hospital_system.get("HOSP_A")
    initial_beds = hosp_a.icu_beds["available"]

    env._hospital_system.consume_icu_bed("HOSP_A")  # consume
    assert hosp_a.icu_beds["available"] == initial_beds - 1

    # Simulate patient death while admitted
    patient = env._patients[0]
    patient.icu_status = "admitted"
    patient.status = "deceased"

    # Trigger the death detection in _advance_time()
    env._advance_time()

    # After death detection, ICU bed should be released
    # BUG: Currently no release happens — beds permanently leak
    assert hosp_a.icu_beds["available"] == initial_beds, (
        f"ICU bed leaked: had {initial_beds}, now {hosp_a.icu_beds['available']} "
        f"after patient {patient.id} died with icu_status={patient.icu_status}"
    )
```

- [ ] **Step 2: Run test — expect FAIL**

```bash
cd C:/Darsh/Scaler/codered_env
python -m pytest tests/test_environment.py -k icu -v 2>&1  # or the new file
```

Expected: FAIL.

- [ ] **Step 3: Fix death detection to release ICU bed**

In `_advance_time()`, in the patient death detection loop:

```python
# Find the patient death detection block (around line ~250)
# Change from:
for p in self._patients:
    if p.status == "deceased" and not any(...):
        self._episode_log.append({...})
# TO:
for p in self._patients:
    if p.status == "deceased" and not any(...):
        self._episode_log.append({...})
        # FIX: Release ICU bed if patient was admitted
        if p.assigned_hospital and p.icu_status == "admitted":
            self._hospital_system.release_icu_bed(p.assigned_hospital)
            self._alerts.append(
                f"ICU bed released at {p.assigned_hospital} after {p.id} death"
            )
```

Also fix `_do_treatment_arrival()` death path. Find where patient dies in `_advance_time()` after mortality roll:

```python
# In _advance_time(), surgery completion block:
if not survived:
    self._patient_manager.mark_deceased(patient_id, reason="hospital_mortality")
    patient.outcome = "deceased"
    # FIX: Release ICU bed on death
    if patient.icu_status == "admitted":
        self._hospital_system.release_icu_bed(hosp_id)
```

- [ ] **Step 4: Run test**

```bash
cd C:/Darsh/Scaler/codered_env
python -m pytest tests/test_environment.py tests/test_icu_leak.py -v 2>&1 | tail -20
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add codered_env/server/codered_environment.py
git commit -m "fix(env): release ICU bed on patient death — prevents resource leak
"
```

---

### Task 5: `preempt_or` not logged to episode_log

**Files:**
- Modify: `codered_env/server/codered_environment.py`
- Modify: `codered_env/server/grader.py`
- Create test: `codered_env/tests/test_grader_preemption.py`

- [ ] **Step 1: Write the failing test**

```python
"""Test that preempt_or actions are logged and penalized in grader."""
import pytest
from server.grader import grade_episode


def test_preempt_or_is_logged_and_penalized():
    """preempt_or actions should appear in episode_log and affect grader score."""
    log = [
        {"step": 0, "patient_id": "P1", "event": "patient_created", "condition": "cardiac",
         "is_secondary": False, "target_time": 90},
        {"step": 5, "patient_id": "P1", "event": "dispatch", "ambulance_id": "AMB_1", "result": "success"},
        {"step": 10, "patient_id": "P1", "event": "treatment_started", "hospital_id": "HOSP_A"},
        {"step": 15, "event": "action_preempt_or", "hospital_id": "HOSP_A", "or_index": 0,
         "harm": 0.5, "recovery_time": 15},
        # Patient's surgery was disrupted — not in log (BUG)
        {"step": 40, "patient_id": "P1", "event": "patient_deceased", "reason": "timeout",
         "effective_time": 0, "target_time": 90},
    ]

    result = grade_episode(log)

    # The grader should see the preempt_or and penalize
    # BUG: Currently preempt_or is not in episode_log → no penalty
    preempt_count = sum(1 for e in log if e.get("event") == "action_preempt_or")
    assert preempt_count > 0, "preempt_or events must be in episode_log"
    assert "preempt_count" in result.breakdown, (
        f"Grader breakdown must include preempt_count, got: {result.breakdown.keys()}"
    )
```

- [ ] **Step 2: Run test — expect FAIL**

Expected: FAIL — no `action_preempt_or` in log and no `preempt_count` in breakdown.

- [ ] **Step 3: Fix `_do_preempt_or()` to log to episode_log**

In `codered_env/server/codered_environment.py`, find `_do_preempt_or()`:

```python
def _do_preempt_or(self, action) -> None:
    from .models.actions import PreemptOR
    result = self._hospital_system.preempt_or(action.hospital_id, action.or_index)
    if not result["success"]:
        self._alerts.append(f"PreemptOR failed: {result['reason']}")
        return

    self._alerts.append(
        f"OR preempted at {action.hospital_id} OR {action.or_index}: "
        f"harm={result['harm']:.2f}, recovery={result['recovery_time']}min"
    )

    # FIX: Log to episode log for grader
    preempt_event = {
        "step": self._state.step_count,
        "event": "action_preempt_or",
        "hospital_id": action.hospital_id,
        "or_index": action.or_index,
        "harm": result["harm"],
        "recovery_time": result["recovery_time"],
    }

    # Check if there was a patient in the OR being preempted
    hosp = self._hospital_system.get(action.hospital_id)
    if hosp:
        # Find patient that was in this OR before preemption
        for or_obj in hosp.operating_rooms:
            if or_obj.index == action.or_index and or_obj.patient_id:
                preempt_event["patient_disrupted"] = or_obj.patient_id
                preempt_event["event"] = "surgery_aborted"
                preempt_event["patient_id"] = or_obj.patient_id
                preempt_event["reason"] = "or_preempted"
                # Also log the aborted patient
                self._episode_log.append({
                    "step": self._state.step_count,
                    "patient_id": or_obj.patient_id,
                    "event": "surgery_aborted",
                    "reason": "or_preempted",
                    "hospital_id": action.hospital_id,
                })
                break

    self._episode_log.append(preempt_event)
```

- [ ] **Step 4: Update grader to penalize preemptions**

In `grade_episode()` in grader.py, after computing `prep_ready`:

```python
# Preemption penalty
preempted_surgeries = sum(
    1 for e in episode_log
    if e.get("event") == "surgery_aborted" and e.get("reason") == "or_preempted"
)
preemptive_actions = sum(
    1 for e in episode_log
    if e.get("event") == "action_preempt_or"
)

# Penalty: each preempted surgery reduces prep_ready by 0.1
prep_ready = max(0.0, prep_ready - 0.1 * preempted_surgeries)

breakdown["preemptive_actions"] = preemptive_actions
breakdown["preempted_surgeries"] = preempted_surgeries
```

- [ ] **Step 5: Run test**

```bash
cd C:/Darsh/Scaler/codered_env
python -m pytest tests/test_grader_preemption.py -v 2>&1
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add codered_env/server/codered_environment.py codered_env/server/grader.py
git commit -m "fix(grader): log preempt_or to episode_log and penalize aborted surgeries
"
```

---

### Task 6: Task1 cascade contamination

**Files:**
- Modify: `codered_env/server/subsystems/constants.py`
- Modify: `codered_env/server/codered_environment.py`

- [ ] **Step 1: Write the failing test**

```python
"""Test that Task1 has no cascade effects."""
import pytest
from server.codered_environment import CodeRedEnvironment
from server.models.actions import MaintainPlan


def test_task1_no_cascade_spawns():
    """Task1 should not spawn secondary patients even if patient is saved."""
    env = CodeRedEnvironment()
    env.reset(seed=100, task_id="task1")

    initial_patient_count = len(env._patients)

    # Run episode with perfect treatment
    for _ in range(30):
        from server.models.actions import DispatchAmbulance, AssignHospital
        # Dispatch ambulance to patient location
        patient = env._patients[0]
        if patient.status == "waiting":
            obs = env.step(DispatchAmbulance(
                ambulance_id="AMB_1",
                target_node=patient.location_node
            ))
            obs = env.step(AssignHospital(
                patient_id=patient.id,
                hospital_id="HOSP_A"
            ))
        else:
            obs = env.step(MaintainPlan())

        if env._check_done():
            break

    # Count secondary patients in log
    log = env.get_episode_log()
    secondary_events = [e for e in log if e.get("event") == "secondary_patient_spawned"]

    # BUG: With 15% cardiac save cascade, secondary patient can spawn in task1
    assert len(secondary_events) == 0, (
        f"Task1 spawned {len(secondary_events)} secondary patient(s): {secondary_events}. "
        f"Task1 should have cascade_enabled=False."
    )
```

- [ ] **Step 2: Run test — may FAIL if seed causes cascade**

```bash
cd C:/Darsh/Scaler/codered_env
python -m pytest tests/test_task1_no_cascade.py -v 2>&1
```

May pass or fail depending on RNG. Run with multiple seeds.

- [ ] **Step 3: Add `cascade_enabled=False` to task1 in constants.py**

```python
TASK_CONFIG: Dict[str, Dict] = {
    "task1": {
        "patients": [
            {"condition": "cardiac", "onset_step": 0},
        ],
        "disruption_prob": 0.0,
        "mutual_aid_calls": 0,
        "max_steps": 30,
        "cascade_enabled": False,  # ADD: no secondary patients in single-patient task
    },
    # task2, task3, task4, task5 unchanged
}
```

- [ ] **Step 4: Guard cascade trigger in environment**

In `codered_environment.py` `_advance_time()`, in the surgery completion block:

```python
# Before calling cascade_engine.on_outcome():
cascade_enabled = TASK_CONFIG.get(self._state.task_id, {}).get("cascade_enabled", False)
if cascade_enabled:
    outcome_str = patient.outcome
    self._cascade_engine.on_outcome(
        patient_id, patient.condition, outcome_str, self._state.step_count
    )
```

- [ ] **Step 5: Run test**

```bash
cd C:/Darsh/Scaler/codered_env
python -m pytest tests/test_task1_no_cascade.py -v 2>&1
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add codered_env/server/subsystems/constants.py codered_env/server/codered_environment.py
git commit -m "fix(task1): disable cascades in task1 — single-patient task should have no secondary spawns
"
```

---

### Task 7: `assigned_ambulance` always None in observation

**Files:**
- Modify: `codered_env/server/codered_environment.py`
- Create test: `codered_env/tests/test_assigned_ambulance.py`

- [ ] **Step 1: Write the failing test**

```python
"""Test that patients show which ambulance is assigned to them."""
import pytest
from server.codered_environment import CodeRedEnvironment
from server.models.actions import DispatchAmbulance, MaintainPlan


def test_patient_shows_assigned_ambulance():
    """After dispatch, patient in observation should have assigned_ambulance set."""
    env = CodeRedEnvironment()
    obs = env.reset(seed=0, task_id="task1")

    patient = env._patients[0]
    patient_location = patient.location_node

    # Dispatch ambulance
    obs = env.step(DispatchAmbulance(
        ambulance_id="AMB_1",
        target_node=patient_location
    ))

    # Check patient in observation
    dispatched_patients = [p for p in obs.patients if p.status.value == "dispatched"]
    assert len(dispatched_patients) > 0, "Should have dispatched patient in obs"

    for p in dispatched_patients:
        # BUG: assigned_ambulance is always None in observation
        assert p.assigned_ambulance is not None, (
            f"Patient {p.patient_id} dispatched but assigned_ambulance=None in observation. "
            f"Agent cannot see which ambulance is en route."
        )
```

- [ ] **Step 2: Run test — expect FAIL**

```bash
cd C:/Darsh/Scaler/codered_env
python -m pytest tests/test_assigned_ambulance.py -v 2>&1
```

Expected: FAIL.

- [ ] **Step 3: Fix `_build_observation()` patient mapping**

In `_build_observation()`, change the patients list construction:

```python
# Find the patients list construction in _build_observation()
# BEFOR
patients = [
    Patient(
        patient_id=p.id,
        ...
        assigned_ambulance=None,  # ← hardcoded!
        ...
    )
    for p in self._patients
]

# AFTER — look up ambulance assignment from ambulance_manager:
patients = []
for p in self._patients:
    # Find which ambulance has this patient assigned
    assigned_amb = None
    for amb_id, amb in self._ambulance_manager.all().items():
        if amb.patient_id == p.id:
            assigned_amb = amb_id
            break
    # Also check mutual aid ambulances
    if assigned_amb is None:
        for ma_id, pending in self._mutual_aid_manager.get_active().items():
            if pending.patient_id == p.id:
                assigned_amb = ma_id
                break

    patients.append(Patient(
        patient_id=p.id,
        condition=PatientCondition(p.condition),
        ...
        assigned_ambulance=assigned_amb,  # ← was: None
        ...
    ))
```

- [ ] **Step 4: Run test**

```bash
cd C:/Darsh/Scaler/codered_env
python -m pytest tests/test_assigned_ambulance.py -v 2>&1
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add codered_env/server/codered_environment.py
git commit -m "fix(env): expose assigned_ambulance in patient observation — agent sees ambulance assignments
"
```

---

### Task 8: Mutual aid arrival step mismatch

**Files:**
- Modify: `codered_env/server/subsystems/mutual_aid.py`
- Modify: `codered_env/server/codered_environment.py`

- [ ] **Step 1: Write the failing test**

```python
"""Test that mutual aid arrival logging uses scheduled step, not current step."""
import pytest
from server.codered_environment import CodeRedEnvironment
from server.models.actions import RequestMutualAid, MaintainPlan


def test_mutual_aid_arrival_logged_correctly():
    """mutual_aid_arrived log entry must use SCHEDULED arrival step, not current step."""
    env = CodeRedEnvironment()
    env.reset(seed=0, task_id="task2")  # task2 has 1 MA call

    # Immediately request mutual aid
    obs = env.step(RequestMutualAid())

    # Get the pending MA entry
    pending = env._mutual_aid_manager.get_pending()
    assert len(pending) == 1, "Should have 1 pending MA"
    ma_id, pending_entry = list(pending.items())[0]
    scheduled_arrival = pending_entry.arrival_step

    # Fast-forward to arrival
    current_step = env.state.step_count
    steps_to_arrival = scheduled_arrival - current_step
    assert steps_to_arrival > 0, f"Already past arrival: current={current_step}, scheduled={scheduled_arrival}"

    for _ in range(steps_to_arrival + 5):
        obs = env.step(MaintainPlan())
        if env._check_done():
            break

    log = env.get_episode_log()
    arrivals = [e for e in log if e.get("event") == "mutual_aid_arrived" and e.get("ambulance_id") == ma_id]

    assert len(arrivals) > 0, "MA arrival must be logged"
    arrival_entry = arrivals[0]

    # BUG: actual_arrival_step is set to current step, not scheduled step
    # This causes the grader to compute wrong delay penalties
    assert arrival_entry.get("actual_arrival_step") == scheduled_arrival, (
        f"MA arrival actual={arrival_entry.get('actual_arrival_step')} "
        f"should equal scheduled={scheduled_arrival}"
    )
```

- [ ] **Step 2: Run test — expect FAIL**

Expected: FAIL.

- [ ] **Step 3: Fix mutual aid arrival logging**

In `mutual_aid.py` `tick()`:

```python
# In tick(), change the arrival event:
# BEFORE:
arrivals.append({
    "ambulance_id": ma_id,
    "patient_id": patient_id,
    "actual_arrival_step": step_count,  # ← wrong!
    "had_patient": True,
})

# AFTER:
arrivals.append({
    "ambulance_id": ma_id,
    "patient_id": patient_id,
    "scheduled_arrival_step": pending.arrival_step,
    "actual_arrival_step": step_count,  # keep for grader
    "delay": step_count - pending.arrival_step,
    "had_patient": True,
})
```

In `codered_environment.py` `_advance_time()`, update the MA arrival logging:

```python
for event in ma_arrivals:
    self._episode_log.append({
        "step": self._state.step_count,
        "event": "mutual_aid_arrived",
        "ambulance_id": event["ambulance_id"],
        "patient_id": event.get("patient_id"),
        "scheduled_arrival_step": event.get("scheduled_arrival_step"),
        "actual_arrival_step": event.get("actual_arrival_step"),
        "delay": event.get("delay", 0),
    })
```

Update grader to use `scheduled_arrival_step`:

```python
# In grade_episode(), mutual aid penalty section:
for call in ma_calls:
    optimal = call.get("optimal_arrival_step", 0)
    arrival_events = [
        e for e in episode_log
        if e.get("event") == "mutual_aid_arrived"
        and e.get("patient_id") == call.get("patient_id")
    ]
    if arrival_events:
        actual = arrival_events[0].get("delay", arrival_events[0].get("actual_arrival_step", optimal) - optimal)
        # Use delay field directly if available
        if "delay" in arrival_events[0]:
            actual_delay = arrival_events[0]["delay"]
        else:
            actual_delay = arrival_events[0].get("actual_arrival_step", optimal) - optimal
        if actual_delay < 0:
            penalties.append(0.1 * abs(actual_delay))
        elif actual_delay > 0:
            penalties.append(0.2 * actual_delay)
```

- [ ] **Step 4: Run test**

```bash
cd C:/Darsh/Scaler/codered_env
python -m pytest tests/test_mutual_aid_arrival.py -v 2>&1
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add codered_env/server/subsystems/mutual_aid.py codered_env/server/codered_environment.py codered_env/server/grader.py
git commit -m "fix(mutual_aid): log scheduled vs actual arrival step — accurate grader delay penalties
"
```

---

## Phase 3: P2 — Moderate Issues + Exploit Patches

---

### Task 9: Exploit Patches — Anti-gaming layer

**Files:**
- Modify: `codered_env/server/codered_environment.py`
- Modify: `codered_env/server/grader.py`
- Modify: `codered_env/server/subsystems/blood_bank.py`
- Modify: `codered_env/server/subsystems/cascade_engine.py`
- Modify: `codered_env/inference.py`
- Create test: `codered_env/tests/test_exploit_detection.py`

- [ ] **Step 1: Write exploit detection tests**

```python
"""Exploit detection tests — each test verifies an exploit is blocked."""
import pytest
from server.codered_environment import CodeRedEnvironment
from server.models.actions import *
from server.grader import grade_episode


# ── Exploit 1: Reward Farming via Cascade ──────────────────────────────────
def test_cascade_reward_farming_detected():
    """
    Agent delays treatment to trigger cascades for more milestone rewards.
    The grader must detect this pattern: long time-to-treatment + cascade spawns.
    """
    # Simulate: patient treated late (time=120, target=90) + 2 cascade patients spawned
    log = [
        {"step": 0, "patient_id": "P1", "event": "patient_created", "condition": "cardiac",
         "is_secondary": False, "target_time": 90},
        {"step": 120, "patient_id": "P1", "event": "treatment_complete",
         "effective_time": 120, "target_time": 90, "vitals_at_treatment": 0.5},
        # Cascade fires: agent delayed treatment, got 2 secondary patients
        {"step": 60, "patient_id": "P2", "event": "secondary_patient_spawned",
         "condition": "cardiac", "is_secondary": True},
        {"step": 90, "patient_id": "P2", "event": "treatment_complete",
         "effective_time": 30, "target_time": 90, "vitals_at_treatment": 0.9},
        {"step": 60, "patient_id": "P3", "event": "secondary_patient_spawned",
         "condition": "cardiac", "is_secondary": True},
        {"step": 120, "patient_id": "P3", "event": "treatment_complete",
         "effective_time": 60, "target_time": 90, "vitals_at_treatment": 0.7},
    ]
    result = grade_episode(log)

    # Time score for P1 is bad (120 vs 90 target)
    assert result.time_score < 0.5, "Late treatment P1 should have low time_score"
    # But P2 and P3 treated quickly — grader gives high cascade score
    # FIX: Add a "treatment delay penalty" that scales with time-to-treatment for primary patients
    # The final score should NOT reward cascade farming
    assert result.final_score < 0.6, (
        f"Cascade farming got score={result.final_score}, too high. "
        f"Delayed primary treatment should not be offset by cascade rewards."
    )


# ── Exploit 2: Idle Strategy ───────────────────────────────────────────────
def test_idle_strategy_low_score():
    """An agent doing nothing should get near-zero score."""
    log = [
        {"step": 0, "patient_id": "P1", "event": "patient_created", "condition": "cardiac",
         "is_secondary": False, "target_time": 90},
        {"step": 45, "patient_id": "P1", "event": "patient_deceased",
         "reason": "timeout", "effective_time": 0, "target_time": 90},
    ]
    result = grade_episode(log)
    assert result.final_score < 0.15, (
        f"Idle agent got {result.final_score}, expected < 0.15. "
        f"No-action episodes must score near zero."
    )


# ── Exploit 3: Action Spam ──────────────────────────────────────────────────
def test_invalid_action_spam_penalized():
    """Spamming invalid actions should accumulate penalties, not just alerts."""
    env = CodeRedEnvironment()
    obs = env.reset(seed=0, task_id="task1")

    # Spam invalid dispatch (ambulance already dispatched)
    for i in range(20):
        obs = env.step(DispatchAmbulance(
            ambulance_id="AMB_1",
            target_node="LAJPAT_NAGAR"
        ))

    # Check that repeated invalid dispatches produce alert spam
    alerts = obs.alerts
    failed_alerts = [a for a in alerts if "failed" in a.lower() or "not available" in a.lower()]

    # After fix: should have action rate limiting or alert fatigue
    assert len(failed_alerts) < 5, (
        f"Got {len(failed_alerts)} failed alerts from spam. "
        f"Action spam should be rate-limited or produce escalating penalties."
    )


# ── Exploit 4: OR Preparation Spam ─────────────────────────────────────────
def test_or_prep_spam_penalized():
    """Repeatedly preparing OR when no patient is en route should tank efficiency."""
    env = CodeRedEnvironment()
    obs = env.reset(seed=0, task_id="task1")

    patient = env._patients[0]

    # Spam prepare_or with no patient on the way
    for _ in range(5):
        obs = env.step(PrepareOR(hospital_id="HOSP_A", procedure_type="cardiac"))

    # Also dispatch ambulance
    obs = env.step(DispatchAmbulance(ambulance_id="AMB_1", target_node=patient.location_node))

    # Advance until done
    for _ in range(30):
        obs = env.step(MaintainPlan())
        if env._check_done():
            break

    result = grade_from_environment(env)

    # Each wasted prep: -0.15. 5 wasted preps → efficiency = 1.0 - 0.75 = 0.25
    assert result.efficiency < 0.5, (
        f"OR prep spam got efficiency={result.efficiency}, expected < 0.5"
    )
    assert result.breakdown.get("wasted_or_preps", 0) == 5, (
        f"Expected 5 wasted_or_preps, got {result.breakdown.get('wasted_or_preps')}"
    )


# ── Exploit 5: Dispatch Gaming ──────────────────────────────────────────────
def test_dispatch_gaming_reward_quality():
    """
    Dispatching to wrong hospital (HOSP_C for cardiac) should be penalized.
    Agent gets milestone rewards but patient outcome is bad.
    """
    # This is a Phase 1 test: dispatch to wrong hospital → patient dies
    log = [
        {"step": 0, "patient_id": "P1", "event": "patient_created", "condition": "cardiac",
         "is_secondary": False, "target_time": 90},
        {"step": 1, "patient_id": "P1", "event": "dispatch", "ambulance_id": "AMB_1", "result": "success"},
        # Assigned to HOSP_C (community HC, no cardiac capability)
        {"step": 20, "patient_id": "P1", "event": "patient_arrived_hospital",
         "hospital_id": "HOSP_C", "or_ready": False, "specialist_available": False},
        {"step": 45, "patient_id": "P1", "event": "patient_deceased",
         "reason": "hospital_mortality", "effective_time": 0, "target_time": 90},
    ]

    result = grade_episode(log)

    # Patient died → time_score = 0. Dispatch milestone was logged as "success"
    # but outcome was death. prep_ready should be 0 (no OR, no specialist)
    assert result.prep_ready == 0.0, f"Wrong hospital → prep_ready should be 0.0, got {result.prep_ready}"
    assert result.time_score == 0.0, f"Dead patient → time_score must be 0.0"
    assert result.final_score < 0.3, (
        f"Wrong hospital dispatch got score={result.final_score}, expected < 0.3"
    )
```

- [ ] **Step 2: Run tests — some will FAIL (bugs to fix)**

```bash
cd C:/Darsh/Scaler/codered_env
python -m pytest tests/test_exploit_detection.py -v 2>&1 | tail -30
```

Expected: Multiple FAILs.

- [ ] **Step 3: Implement exploit fixes**

**Fix E1 (Cascade farming):** In `grade_episode()`, add a "treatment delay penalty":

```python
# After computing time_score, add cascade farming detection:
# If primary patient time >> target AND cascades spawned → penalty
primary_patients = [e for e in episode_log if e.get("event") == "patient_created" and not e.get("is_secondary")]
cascade_patients = [e for e in episode_log if e.get("event") in ("secondary_patient_spawned", "patient_created") and e.get("is_secondary")]

if len(cascade_patients) > 0 and len(primary_patients) > 0:
    # Check if primary patient was treated very late
    for entry in episode_log:
        if entry.get("event") == "treatment_complete" and entry.get("patient_id") == primary_patients[0].get("patient_id"):
            eff = entry.get("effective_time", 0)
            tgt = entry.get("target_time", 90)
            if eff > tgt * 1.5:  # 50% over target
                cascade_farming_penalty = min(0.2, 0.1 * len(cascade_patients))
                time_score = max(0.0, time_score - cascade_farming_penalty)
                breakdown["cascade_farming_penalty"] = cascade_farming_penalty
                break
```

**Fix E2 (Idle strategy):** Already covered by Task 3 (empty episode guard). Extend to include "all patients died with no dispatches" pattern.

**Fix E3 (Action spam):** Add rate limiting in `step()`:

```python
def step(self, action, ...):
    # Rate limit: count actions per type
    self._action_counts: Dict[str, int] = getattr(self, '_action_counts', {})
    action_key = type(action).__name__
    self._action_counts[action_key] = self._action_counts.get(action_key, 0) + 1

    # After 10 consecutive identical actions, penalize
    if self._action_counts[action_key] > 10:
        # First-time penalty: reduce reward
        self._state.cum_reward -= 0.05
        if len(self._alerts) < 3:  # don't spam alerts
            self._alerts.append(f"Action spam detected: {action_key} repeated {self._action_counts[action_key]} times")
```

Initialize `_action_counts: Dict[str, int] = {}` in `reset()`.

**Fix E4 (OR prep spam):** The existing `wasted_or_preps` detection in grader is correct. Ensure `_do_prepare_or()` always logs `result="wasted"` when the prep has no patient en route. Currently it only logs "wasted" on failure (no idle OR). The fix: track whether a patient is en route to the hospital and mark preps as wasted if not.

Add to `_do_prepare_or()`:
```python
# Check if any patient is en route to this hospital with this condition
en_route = any(
    p.status in ("dispatched", "transporting")
    and p.assigned_hospital == action.hospital_id
    for p in self._patients
)
if not en_route:
    # Wasted: preparing OR when no patient is coming
    self._episode_log.append({
        "step": self._state.step_count,
        "event": "action_prepare_or",
        "hospital_id": action.hospital_id,
        "procedure_type": action.procedure_type,
        "result": "wasted",
        "reason": "no_patient_en_route",
    })
    # Penalty is applied by grader
```

**Fix E5 (Dispatch gaming):** Already handled by `prep_ready` axis (wrong hospital → no OR, no specialist → 0 prep_ready). Ensure `patient_arrived_hospital` event is always logged when patient arrives, regardless of hospital capability.

**Fix E2 idle detection in grader:**

```python
# After patient_times computation, add idle agent detection:
all_events = episode_log
dispatches = [e for e in all_events if e.get("event") == "dispatch"]
treatments = [e for e in all_events if e.get("event") == "treatment_complete"]

if len(dispatches) == 0 and len(treatments) == 0 and len(patient_times) > 0:
    # Agent did nothing while patients were present
    all_deceased = all(v.get("treated", False) == False for v in patient_times.values())
    if all_deceased:
        # Maximum penalty for complete inaction
        breakdown["anti_exploit"] = "complete_inaction"
        time_score = 0.0
```

- [ ] **Step 4: Run exploit tests**

```bash
cd C:/Darsh/Scaler/codered_env
python -m pytest tests/test_exploit_detection.py -v 2>&1
```

Expected: Most or all PASS after fixes.

- [ ] **Step 5: Commit**

```bash
git add codered_env/server/codered_environment.py codered_env/server/grader.py codered_env/server/subsystems/blood_bank.py codered_env/server/subsystems/cascade_engine.py codered_env/inference.py
git commit -m "feat(exploits): add anti-gaming layer — cascade farming, idle strategy, action spam, OR prep spam, dispatch gaming detection
"
```

---

### Task 10: Blood bank unlimited emergency drain

**Files:**
- Modify: `codered_env/server/subsystems/blood_bank.py`
- Create test: `codered_env/tests/test_blood_drain.py`

- [ ] **Step 1: Write the failing test**

```python
"""Test that emergency blood release has per-patient and per-hospital caps."""
import pytest
from server.subsystems.blood_bank import BloodBankSystem


def test_emergency_blood_reserves_not_exhaustible():
    """Agent cannot drain all O_NEG via repeated emergency=True calls."""
    bank_sys = BloodBankSystem()
    bank = bank_sys.get("HOSP_A")

    initial_o_neg = bank.stocks["O_NEG"]  # 6 units

    # Try to drain ALL O_NEG
    for i in range(20):
        result = bank_sys.emergency_release("HOSP_A", f"P{i}", "O_NEG", units=1)

    # BUG: All 6 units can be drained. After fix: should reject after 4 units
    remaining = bank.stocks["O_NEG"]
    assert remaining >= 0, f"O_NEG went negative: {remaining}"
    # After fix: remaining should be >= 2 (leave emergency reserve)
    assert remaining >= 2, (
        f"Emergency blood drained to {remaining} units. "
        f"Expected >= 2 units emergency reserve remaining."
    )
```

- [ ] **Step 2: Run test — expect FAIL**

```bash
cd C:/Darsh/Scaler/codered_env
python -m pytest tests/test_blood_drain.py -v 2>&1
```

Expected: FAIL.

- [ ] **Step 3: Fix emergency_release caps**

```python
def emergency_release(self, hosp_id: str, patient_id: str, blood_type: str, units: int) -> Dict:
    bank = self._banks.get(hosp_id)
    if bank is None:
        return {"success": False, "reason": f"Hospital {hosp_id} not found"}

    # FIX: Cap units per request
    units = min(units, 4)  # max 4 units per emergency request

    # FIX: Track emergency releases this episode to prevent drain
    if not hasattr(self, "_emergency_released"):
        self._emergency_released: Dict[str, int] = {h: 0 for h in self._banks}

    total_emergency_released = self._emergency_released.get(hosp_id, 0)
    if total_emergency_released >= 4:  # max 4 emergency units per hospital per episode
        return {
            "success": False,
            "reason": f"Emergency reserves exhausted at {hosp_id} (max 4 units/episode)"
        }

    if bank.stocks.get("O_NEG", 0) < units:
        return {
            "success": False,
            "reason": f"Insufficient O_NEG at {hosp_id}: have {bank.stocks.get('O_NEG', 0)}, need {units}"
        }

    bank.stocks["O_NEG"] -= units
    self._emergency_released[hosp_id] = total_emergency_released + units

    return {"success": True, "blood_type": "O_NEG", "units": units}
```

Also reset `_emergency_released` when BloodBankSystem is re-initialized (in `__init__`).

- [ ] **Step 4: Run test**

```bash
cd C:/Darsh/Scaler/codered_env
python -m pytest tests/test_blood_drain.py -v 2>&1
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add codered_env/server/subsystems/blood_bank.py
git commit -m "fix(blood_bank): cap emergency blood release — max 4 units/hospital/episode
"
```

---

### Task 11: Inference temperature non-determinism

**Files:**
- Modify: `codered_env/inference.py`

- [ ] **Step 1: Change temperature from 0.7 to 0.0**

In `call_model()`:

```python
# BEFORE:
temperature=0.7,
# AFTER:
temperature=0.0,
```

- [ ] **Step 2: Add deterministic reproducibility note to docstring**

```python
# In call_model():
# NOTE: temperature=0.0 for deterministic reproducible benchmarking.
# For diverse responses (e.g., hyperparameter search), use temperature=0.7
# but note that scores will vary across runs.
```

- [ ] **Step 3: Commit**

```bash
git add codered_env/inference.py
git commit -m "fix(inference): set temperature=0.0 for deterministic reproducible scores
"
```

---

### Task 12: Surge probability unbounded + Phase 2 force-spawn secondary flag

**Files:**
- Modify: `codered_env/server/subsystems/cascade_engine.py`
- Modify: `codered_env/server/codered_environment.py`

- [ ] **Step 1: Cap surge probability at 1.0**

In `cascade_engine.py`, in `_apply_effect()`:

```python
elif effect == "news_cycle":
    self._news_cycle_steps_remaining = params["steps"]
    # FIX: Cap at 1.0
    self._pending_surge_probability = min(
        1.0,
        self._pending_surge_probability + params["surge_prob_boost"]
    )
```

- [ ] **Step 2: Phase 2 force-spawn `force_spawn` flag**

In `_spawn_patient_from_call()`, add `force_spawn=True`:

```python
self._episode_log.append({
    "step": self._state.step_count,
    "patient_id": patient.id,
    "event": "patient_created",
    "condition": true_condition,
    "is_secondary": False,
    "force_spawn": True,  # NEW: indicates countdown-expired, not true cascade
    ...
})
```

- [ ] **Step 3: Commit**

```bash
git add codered_env/server/subsystems/cascade_engine.py codered_env/server/codered_environment.py
git commit -m "fix(cascade): cap surge probability at 1.0, add force_spawn flag for Phase 2
"
```

---

## Phase 4: Sync + Regression + Final Validation

---

### Task 13: Sync fixes to HF Space, run full regression

**Files:**
- Sync: `codered-hfspace/server/codered_environment.py`
- Sync: `codered-hfspace/server/grader.py`
- Sync: `codered-hfspace/server/subsystems/`

- [ ] **Step 1: Sync all non-done-flag fixes to HF Space**

The HF Space version needs all the same fixes (except the `done` fix which was already done in Task 1). Apply each fix from Tasks 4-12 to the HF Space versions.

**Quick sync commands:**

```bash
# Sync grader.py
cp codered_env/server/grader.py codered-hfspace/server/grader.py

# Sync ambulance_manager.py
cp codered_env/server/subsystems/ambulance_manager.py codered-hfspace/server/subsystems/ambulance_manager.py

# Sync blood_bank.py
cp codered_env/server/subsystems/blood_bank.py codered-hfspace/server/subsystems/blood_bank.py

# Sync cascade_engine.py
cp codered_env/server/subsystems/cascade_engine.py codered-hfspace/server/subsystems/cascade_engine.py

# Sync mutual_aid.py
cp codered_env/server/subsystems/mutual_aid.py codered-hfspace/server/subsystems/mutual_aid.py

# Sync constants.py
cp codered_env/server/subsystems/constants.py codered-hfspace/server/subsystems/constants.py

# Sync codered_environment.py (main env)
cp codered_env/server/codered_environment.py codered-hfspace/server/codered_environment.py
```

Then re-apply the HF Space-specific `done` parameter fix (from Task 1) to the synced file.

- [ ] **Step 2: Run full test suite on both versions**

```bash
cd C:/Darsh/Scaler/codered_env
python -m pytest tests/ -x -q --tb=short 2>&1 | tail -20
```

```bash
cd C:/Darsh/Scaler/codered-hfspace
python -m pytest ../codered_env/tests/ -x -q --tb=short 2>&1 | tail -20
```

Expected: Both pass.

- [ ] **Step 3: Commit sync**

```bash
git add codered-hfspace/
git commit -m "sync(hfspace): apply all audit fixes to HF Space version
"
```

---

### Task 14: OpenEnv spec compliance check

**Files:**
- Verify: `codered_env/openenv.yaml`
- Verify: `codered-hfspace/openenv.yaml`

- [ ] **Step 1: Verify openenv.yaml matches between versions**

```bash
diff C:/Darsh/Scaler/codered_env/openenv.yaml C:/Darsh/Scaler/codered-hfspace/openenv.yaml
```

Expected: No differences (or intentional minimal diffs like port numbers).

- [ ] **Step 2: Verify step() returns OpenEnv-compliant tuple**

OpenEnv spec requires `step()` returns `(observation, reward, terminated, truncated, info)`.

Check how the server wraps the environment:

```bash
grep -n "step\|return" C:/Darsh/Scaler/codered_env/server/app.py | head -20
```

If the server wraps `env.step()` manually, ensure it returns the 4-tuple.

```python
# In the server's HTTP handler for /step:
obs = env.step(action)
reward = env.state.cum_reward
done = env._check_done()
return {"observation": obs, "reward": reward, "terminated": done, "truncated": False, "info": {}}
```

If OpenEnv's `create_app` handles this automatically, no change needed.

- [ ] **Step 3: Verify reward is computed BEFORE done check**

Current code in `step()`:
```python
# 1. Advance time
self._state.step_count += 1
# 2. Execute action
self._execute_action(action)
# 3. Check termination  ← done BEFORE reward
done = self._check_done()
# 4. Compute reward     ← reward AFTER done
reward = self._compute_step_reward()
```

**FIX:** Move reward computation BEFORE done check:

```python
# 1. Advance time
self._state.step_count += 1
# 2. Execute action
self._execute_action(action)
# 3. Compute reward FIRST (trajectory-aware)
reward = self._compute_step_reward()
# 4. THEN check done (done depends on state AFTER reward computation)
done = self._check_done()
```

- [ ] **Step 4: Commit**

```bash
git add codered_env/server/codered_environment.py codered_env/openenv.yaml codered-hfspace/openenv.yaml
git commit -m "fix(env): compute reward BEFORE termination check for OpenEnv compliance
"
```

---

### Task 15: Final audit checklist verification

- [ ] **Step 1: Verify all 12 critical checks from Phase 6 of audit**

Run each check:

```bash
# 1. step() returns (obs, reward, done, info)
python -c "
from server.codered_environment import CodeRedEnvironment
from server.models.actions import MaintainPlan
env = CodeRedEnvironment()
obs = env.reset(seed=0, task_id='task1')
print('reset() returns observation:', type(obs).__name__)
# step() via HTTP server — check done flag
for _ in range(31):
    obs = env.step(MaintainPlan())
    if env._check_done():
        print('Episode terminated, obs.done:', obs.done)
        break
"

# 2. reset() clean
python -c "
from server.codered_environment import CodeRedEnvironment
env = CodeRedEnvironment()
obs1 = env.reset(seed=0, task_id='task1')
for _ in range(10): obs1 = env.step(MaintainPlan())
obs2 = env.reset(seed=0, task_id='task1')
assert obs2.step == 1, 'Reset should start at step 1'
print('reset() clean: PASS')
"

# 3. No state leakage (assigned_ambulance visible)
python -c "
from server.codered_environment import CodeRedEnvironment
from server.models.actions import DispatchAmbulance
env = CodeRedEnvironment()
obs = env.reset(seed=0, task_id='task1')
p = env._patients[0]
obs = env.step(DispatchAmbulance(ambulance_id='AMB_1', target_node=p.location_node))
amb_obs = [a for a in obs.ambulances if a.id == 'AMB_1']
print('AMB_1 status after dispatch:', amb_obs[0].status if amb_obs else 'not found')
print('AMB_1 assigned_patient:', amb_obs[0].assigned_patient if amb_obs else 'N/A')
"

# 4. Deterministic with seed
python -c "
from server.codered_environment import CodeRedEnvironment
from server.models.actions import MaintainPlan
for seed in [0, 1, 2]:
    env = CodeRedEnvironment()
    env.reset(seed=seed, task_id='task1')
    for _ in range(5): env.step(MaintainPlan())
    log1 = env.get_episode_log()
    env.reset(seed=seed, task_id='task1')
    for _ in range(5): env.step(MaintainPlan())
    log2 = env.get_episode_log()
    assert log1 == log2, f'Seed {seed} non-deterministic: {len(log1)} vs {len(log2)} events'
print('Determinism check: PASS')
"

# 5. Score bounds
python -c "
from server.grader import grade_episode
# Test: no patients
r = grade_episode([])
print(f'Empty episode score: {r.final_score} (expect ~0.0)')
# Test: all patients saved
log = [
    {'step':0,'patient_id':'P1','event':'patient_created','condition':'cardiac','is_secondary':False,'target_time':90},
    {'step':30,'patient_id':'P1','event':'treatment_complete','effective_time':30,'target_time':90,'vitals_at_treatment':1.0},
]
r = grade_episode(log)
print(f'Perfect treatment: {r.final_score} (expect ~1.0)')
assert 0.0 <= r.final_score <= 1.0, f'Score out of bounds: {r.final_score}'
print('Score bounds: PASS')
"

# 6. Run full test suite
python -m pytest tests/ -q --tb=line 2>&1 | tail -5
```

All checks: PASS.

- [ ] **Step 2: Update audit report with fix status**

Update `docs/CODE_RED_ENV_AUDIT.md` — add a "Fix Status" column:

```markdown
| Priority | Issue | Status | Applied In |
|----------|-------|--------|------------|
| P0-1 | `done` flag missing in HF Space | ✅ FIXED | Task 1 |
| P0-2 | Phase 2 ambulance-patient linkage | ✅ FIXED | Task 2 |
| P0-3 | Empty episode perfect score | ✅ FIXED | Task 3 |
| P1-1 | `assigned_ambulance` always None | ✅ FIXED | Task 7 |
| P1-2 | ICU bed leak on death | ✅ FIXED | Task 4 |
| P1-3 | `preempt_or` not logged | ✅ FIXED | Task 5 |
| P1-4 | Task1 cascade contamination | ✅ FIXED | Task 6 |
| P1-5 | Phase 2 force-spawn secondary | ✅ FIXED | Task 12 |
| P1-6 | MA arrival step mismatch | ✅ FIXED | Task 8 |
| P2-E1 | Blood drain exploit | ✅ FIXED | Task 10 |
| P2-surge | Surge probability unbounded | ✅ FIXED | Task 12 |
| P2-temp | Inference temperature | ✅ FIXED | Task 11 |
```

---

## Summary: All Tasks

| # | Task | Files | P0/P1/P2 |
|---|------|-------|----------|
| 1 | HF Space `done` flag | `hfspace/server/codered_environment.py` | P0 |
| 2 | Phase 2 ambulance-patient linkage | `codered_env/server/codered_environment.py`, `ambulance_manager.py` | P0 |
| 3 | Empty episode perfect score | `grader.py` | P0 |
| 4 | ICU bed leak on death | `codered_environment.py` | P1 |
| 5 | `preempt_or` not logged | `codered_environment.py`, `grader.py` | P1 |
| 6 | Task1 cascade contamination | `constants.py`, `codered_environment.py` | P1 |
| 7 | `assigned_ambulance` always None | `codered_environment.py` | P1 |
| 8 | MA arrival step mismatch | `mutual_aid.py`, `codered_environment.py`, `grader.py` | P1 |
| 9 | Exploit patches (E1-E5 + E2 idle + E3 spam + E4 OR + E5 gaming) | `codered_environment.py`, `grader.py`, `blood_bank.py`, `cascade_engine.py`, `inference.py` | P1/P2 |
| 10 | Blood drain exploit | `blood_bank.py` | P2 |
| 11 | Inference temperature | `inference.py` | P2 |
| 12 | Surge cap + force_spawn flag | `cascade_engine.py`, `codered_environment.py` | P2 |
| 13 | Sync to HF Space + regression | All HF Space files | P0 |
| 14 | OpenEnv spec compliance | `codered_environment.py`, `openenv.yaml` | P0 |
| 15 | Final checklist verification | All | ALL |

---

*Plan complete. Recommended execution: Subagent-Driven with one subagent per Phase (Phase 1 → Phase 2 → Phase 3 → Phase 4), with review between phases.*
