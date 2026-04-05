# CodeRedEnv Phase 2: Pre-Dispatch Uncertainty + Causal Cascade

**Date:** 2026-04-01
**Status:** Draft
**Scope:** Phase 2 of CodeRed emergency response environment upgrade
**Supersedes:** 2026-03-27-codered-env-phase1-vitals-rewards-design.md
**Dependencies:** Phase 1 (Tasks 1-16) — fully implemented and committed

---

## Overview

Phase 1 (Tasks 1-16) built the foundational environment: patient deterioration model, dense reward shaping, shift-based staffing, time-of-day congestion, on-scene time, ICU bed constraints, hospital quality variance, and mutual aid. All 129 tests pass.

Phase 2 adds two new core mechanics that together create a genuinely novel RL benchmark:

1. **Pre-Dispatch Uncertainty** — incoming 911 calls arrive with vague dispatch categories (not true conditions). The agent must decide ambulance level (ALS/BLS) and transport method before ground truth is revealed.
2. **Causal Cascade Outcomes** — patient outcomes are interdependent. Deaths spawn secondary patients, overcrowding accelerates deterioration, successful saves trigger news-cycle surges.

These two mechanics are synergistic: dispatch classification errors (Mechanic 1) propagate into cascade triggers (Mechanic 2), and cascade outcomes create new dispatch decisions. The agent must reason at decision time AND over time.

**Grading targets:** time_score (40%), efficiency (20%), secondary_harm (20%), prep_ready (20%), with new cascade_score axis. The 9/10 creativity threshold is met on domain novelty, reward design, and emergent mechanic depth.

---

## Phase 1: Current Architecture Summary

### What Exists (16 Tasks Complete)

| Component | File | Key Details |
|-----------|------|-------------|
| Main environment | `codered_environment.py` | `step()`/`reset()`/`_advance_time()`, 1129 lines |
| Patient lifecycle | `patient_manager.py` | Vitals deterioration, status escalation, `tick()` |
| Ambulance fleet | `ambulance_manager.py` | Dispatch, ETA, scene time countdown |
| Hospital system | `hospital_system.py` | OR prep/use, specialist paging, ICU beds, shifts |
| Road network | `road_network.py` | Dijkstra routing, TOD congestion, disruptions |
| Blood bank | `blood_bank.py` | Crossmatch queue, emergency O-neg release |
| Disruption engine | `disruption_engine.py` | Seeded disruption scheduler (5 types) |
| Grader | `grader.py` | 4-axis rubric + ICU boarding penalty |
| Constants | `constants.py` | City graph, hospital defs, task configs |
| Models | `entities.py`, `state.py`, `actions.py`, `observations.py` | Pydantic types |

### Current Action Space (11 actions)
`DispatchAmbulance`, `PrepareOR`, `PageSpecialist`, `AssignHospital`, `PreemptOR`, `AllocateBlood`, `TransferBlood`, `RequestMutualAid`, `QueryBloodType`, `QueryORStatus`, `MaintainPlan`

### Current Observation Space
`CodeRedObservation`: step, patients list (condition, status, vitals_score, location, hospital), ambulances list (status, location, ETA), hospitals list (OR status, specialists, ICU beds, diversion), blood banks list, road network state, alerts, time_score_preview, vitals_score_preview, patients_remaining

### Current Reward Function
Vitals delta (×0.5) + milestone bonuses: dispatched (+0.20), in_treatment (+0.40), treated (+0.80), deceased (−0.80). Clamped to [−1.0, 1.0].

### Grading Rubric
`final = 0.36×time_score + 0.18×efficiency + 0.18×secondary_harm + 0.18×prep_ready + 0.10×vitals_avg − mutual_aid_penalty`

### Current Data Flow
```
reset() → spawn patients (full truth known)
step(action) → tick all subsystems → execute action → check done → compute reward → return observation
patient outcome → logged → secondary_harm axis scores survival
```

---

## Phase 2: Gap Analysis

### Where We Are (5/10 Profile)

| Dimension | Current State | Gap to 9/10 |
|-----------|--------------|-------------|
| Domain novelty | Emergency coordination is novel domain ✓ | Already met |
| Observation space | Full ground-truth state ✓ | Missing pre-dispatch call queue |
| Action space | 11 typed actions, all state known ✓ | Missing triage-level decisions |
| Reward design | Dense + milestones ✓ | Missing post-hoc dispatch classification feedback |
| Mechanic novelty | Standard RL applied to new domain ✓ | Missing interdependent outcomes |
| Emergent behavior | Disruptions are independent events ✓ | Missing cascade propagation |

**Critical gap:** Every patient is known at spawn with full condition. The agent never has to *decide who gets an ambulance*. This is the central strategic decision in real EMS dispatch — not "which hospital" but "which calls deserve a response at all."

**Critical gap:** Patient outcomes are independent. A death doesn't create new patients. Overcrowding doesn't slow care. This misses the core feedback loop that makes emergency medicine genuinely hard to optimize.

---

## Phase 3: Design Options

### Mechanic 1 — Pre-Dispatch Uncertainty

#### Option A: Selective Concealment (Recommended)
Only the **dispatch category** is concealed pre-dispatch. The agent sees `pending_calls` with categories: `"chest_pain"`, `"altered_consciousness"`, `"fall"`, `"breathing_difficulty"`, `"general"`. Each category maps to a probability distribution over true conditions.

**Pros:** Simple POMDP, clear reward signal, agent can learn fast, maintainable.
**Cons:** Some agents may learn to ignore categories entirely.
**Compatibility:** High. No changes to patient_manager tick logic.
**RL stability:** High. Probabilities are static constants, not changing between episodes.

#### Option B: Multi-Layered Concealment
Above + blood type hints only (no full reveal until QueryBloodType) + noisy hospital OR counts. Harder to train against, but more realistic.

**Decision:** Defer to Phase 3. Phase 2 uses Option A.

#### Option C: Full POMDP
Agent sees no per-patient vitals, no OR counts, no ambulance locations. Must build internal world model. Too aggressive for Phase 2.

**Decision:** Not planned.

### Mechanic 2 — Causal Cascade Outcomes

#### Option A: Distributed Triggers (Rejected)
Cascade rules embedded in `patient_manager.tick()` and `_advance_time()` near each outcome site.

**Pros:** No new subsystem.
**Cons:** Logic scattered across files, hard to test in isolation, violates single-responsibility.
**RL stability:** Low. Logic entangled with core tick makes tuning cascade parameters risky.

#### Option B: Event-Driven CascadeEngine (Recommended)
New `CascadeEngine` subsystem. Environment publishes outcome events via `cascade_engine.on_outcome(event_type, patient)`. Engine evaluates probability rules and calls back to `patient_manager.spawn_secondary()` or `environment._add_overcrowding_modifier()`.

**Pros:** Isolated, testable, extensible, follows existing pattern of DisruptionEngine.
**Cons:** Adds one new subsystem.
**Compatibility:** High. Mirrors DisruptionEngine architecture exactly.
**RL stability:** High. Probability rules isolated and tunable independently.

#### Option C: Centralized CascadeManager in Environment
Single `CascadeManager` class in `codered_environment.py` owning all cascade state.

**Pros:** No new subsystem file.
**Cons:** Environment file already 1129 lines; this adds more complexity there.
**Decision:** Rejected.

### Final Design Choices

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Uncertainty depth | Option A — Selective concealment | Trainable POMDP, meets novelty bar |
| Cascade architecture | Option B — Event-driven CascadeEngine | Isolated, testable, mirrors DisruptionEngine |
| Action space change | Split actions — ALS/BLS + TriageCall | Clean separation, replaces ambiguous DispatchAmbulance |
| Task integration | Option B — Graduated (Task 4 uncertainty, Task 5 cascades) | Progressive difficulty, cleaner evaluation |
| Mechanic depth | Option B — Full probabilistic | Emergent behavior without maximum complexity |

---

## Phase 4: Final Architecture

### 4.1 New Data Structures

#### DispatchCall Model (new in `entities.py`)
```python
class DispatchCategory(str, Enum):
    CHEST_PAIN = "chest_pain"
    ALTERED_CONSCIOUSNESS = "altered_consciousness"
    FALL = "fall"
    BREATHING_DIFFICULTY = "breathing_difficulty"
    GENERAL = "general"

# Dispatch category → condition probability table
DISPATCH_CATEGORY_MAP: dict[DispatchCategory, list[tuple[str, float]]] = {
    # (true_condition, probability)
    DispatchCategory.CHEST_PAIN: [("cardiac", 0.45), ("anxiety", 0.25), ("gerd", 0.20), ("panic", 0.10)],
    DispatchCategory.ALTERED_CONSCIOUSNESS: [("stroke", 0.30), ("hypoglycemia", 0.35), ("intoxication", 0.25), ("seizure", 0.10)],
    DispatchCategory.FALL: [("trauma", 0.30), ("fracture", 0.35), ("syncope", 0.25), ("minor", 0.10)],
    DispatchCategory.BREATHING_DIFFICULTY: [("respiratory", 0.40), ("cardiac", 0.25), ("asthma", 0.25), ("panic", 0.10)],
    DispatchCategory.GENERAL: [("general", 0.60), ("dehydration", 0.25), ("viral", 0.15)],
}

# ALS necessity probability per category (for reward shaping hints)
ALS_NEEDED_PROB: dict[DispatchCategory, float] = {
    DispatchCategory.CHEST_PAIN: 0.60,
    DispatchCategory.ALTERED_CONSCIOUSNESS: 0.50,
    DispatchCategory.FALL: 0.30,
    DispatchCategory.BREATHING_DIFFICULTY: 0.45,
    DispatchCategory.GENERAL: 0.10,
}

@dataclass
class DispatchCall:
    call_id: str
    category: DispatchCategory
    location_node: str
    time_waiting: int  # steps since call arrived
    estimated_severity: float  # 0.0–1.0, based on time_waiting
    spawned_patient_id: Optional[str] = None  # set when patient spawns on-scene
```

#### DispatchOutcome Model (new in `entities.py`)
```python
@dataclass
class DispatchOutcome:
    call_id: str
    decision: str  # "als", "bls", "self_transport", "callback", "no_dispatch"
    category: DispatchCategory  # what was reported
    true_condition: Optional[str] = None  # revealed on-scene arrival
    als_needed: bool = False  # ground truth: was ALS actually needed?
    revealed_at_step: Optional[int] = None
```

#### Patient Extension (`entities.py` Patient model — add fields)
```python
@dataclass
class Patient:
    # ... existing fields ...
    # New fields:
    dispatch_call_id: Optional[str] = None  # links patient to originating call
    is_secondary: bool = False  # True if spawned by cascade
    cascade_trigger_reason: Optional[str] = None  # "witnessed_death", "overcrowding_surge", "news_cycle"
    severity_modifier: float = 1.0  # hidden: 0.7–1.3, multiplies deterioration rate
    observed_condition: Optional[str] = None  # shown to agent pre-arrival
```

#### Action Changes (`actions.py`)

**REMOVE:** `DispatchAmbulance`

**ADD:**
```python
class DispatchALS(Action):
    """Dispatch an ALS ambulance to a dispatch call. Commits ALS resource."""
    ambulance_id: str
    call_id: str  # must reference a pending DispatchCall
    target_node: str  # from the call's location_node

class DispatchBLS(Action):
    """Dispatch a BLS ambulance to a dispatch call."""
    ambulance_id: str
    call_id: str
    target_node: str

class TriageCall(Action):
    """Decide what to do with a pending dispatch call (ALS/BLS/self-transport/callback)."""
    call_id: str
    decision: Literal["dispatch_als", "dispatch_bls", "self_transport", "callback", "no_dispatch"]
    # For dispatch_als/dispatch_bls: also supply ambulance_id
    ambulance_id: Optional[str] = None

class MaintainPlan(Action):
    """No-op action. Keep current state."""
    pass  # existing
```

**Existing actions unchanged** except: `AssignHospital` must look up `patient.observed_condition` (shown post-arrival) instead of `patient.condition` for the capability check, until ground truth is confirmed.

#### Observation Changes (`observations.py`)

**ADD to `CodeRedObservation`:**
```python
class CodeRedObservation(BaseModel):
    # ... existing fields ...
    pending_calls: List[DispatchCall]  # NEW: calls awaiting triage decision
    recent_dispatch_outcomes: List[DispatchOutcome]  # NEW: resolved call outcomes (last 5)
    overcrowding_modifier: float  # NEW: current deterioration multiplier (1.0 or 1.2)
    alerts: List[str]  # UPDATED: now includes cascade events
```

**UPDATE `Patient` display line in observation:**
```
Patient fields now include: observed_condition (pre-reveal) or condition (post-reveal)
is_secondary flag shown in patient line
```

### 4.2 New Subsystem: CascadeEngine

#### File: `server/subsystems/cascade_engine.py`

```python
"""Event-driven causal cascade engine for interdependent patient outcomes."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
import random

@dataclass
class CascadeRule:
    trigger: str  # "patient_deceased", "patient_saved", "overcrowded", "news_cycle"
    condition_filter: Optional[str] = None  # e.g., "trauma", "cardiac"
    probability: float  # 0.0–1.0
    effect: str  # "spawn_secondary", "increase_surge_prob", "apply_overcrowding"
    effect_params: Dict = field(default_factory=dict)


class CascadeEngine:
    """
    Subscribes to patient outcome events and evaluates probabilistic cascade rules.
    Follows the same architecture as DisruptionEngine — isolated, seeded, testable.
    """

    def __init__(self):
        self._rules: List[CascadeRule] = []
        self._rng: Optional[random.Random] = None
        self._overcrowding_active: bool = False
        self._pending_surge_probability: float = 0.0
        self._news_cycle_steps_remaining: int = 0
        self._callback: Optional[Callable] = None  # environment callback

    def reset(self, seed: int, episode_config: dict) -> None:
        """Initialize engine with seed. Call at environment reset()."""
        self._rng = random.Random(seed)
        self._overcrowding_active = False
        self._pending_surge_probability = 0.0
        self._news_cycle_steps_remaining = 0
        self._load_rules(episode_config)

    def set_callback(self, callback: Callable) -> None:
        """Environment provides a callback for spawning patients, raising alerts."""
        self._callback = callback

    def _load_rules(self, config: dict) -> None:
        """Load cascade rules from config. Extensible."""
        self._rules = [
            # Witnessed trauma death → secondary cardiac (psychogenic)
            CascadeRule(
                trigger="patient_deceased",
                condition_filter="trauma",
                probability=0.30,
                effect="spawn_secondary",
                effect_params={"secondary_condition": "cardiac", "reason": "psychogenic_cascade"},
            ),
            # Witnessed cardiac death → secondary cardiac (sympathy effect)
            CascadeRule(
                trigger="patient_deceased",
                condition_filter="cardiac",
                probability=0.20,
                effect="spawn_secondary",
                effect_params={"secondary_condition": "cardiac", "reason": "sympathy_cascade"},
            ),
            # Successful cardiac resuscitation → news cycle → increased call volume
            CascadeRule(
                trigger="patient_saved",
                condition_filter="cardiac",
                probability=0.15,
                effect="news_cycle",
                effect_params={"steps": 10, "surge_prob_boost": 0.20},
            ),
            # Successful trauma resuscitation → news cycle → moderate surge
            CascadeRule(
                trigger="patient_saved",
                condition_filter="trauma",
                probability=0.10,
                effect="news_cycle",
                effect_params={"steps": 8, "surge_prob_boost": 0.15},
            ),
            # Overcrowded ED (active patients > 3) — handled in tick, not trigger
            CascadeRule(
                trigger="overcrowded",
                probability=1.0,
                effect="apply_overcrowding",
                effect_params={"threshold": 3, "modifier": 1.2},
            ),
        ]

    def on_outcome(self, patient_id: str, condition: str, outcome: str, step: int) -> None:
        """
        Called by environment when a patient outcome is determined.
        Evaluates cascade rules and triggers effects.
        """
        if self._rng is None:
            return

        trigger = f"patient_{outcome}"  # "patient_deceased" or "patient_saved"

        for rule in self._rules:
            if rule.trigger != trigger:
                continue
            if rule.condition_filter is not None and rule.condition_filter != condition:
                continue

            if self._rng.random() < rule.probability:
                self._apply_effect(rule.effect, rule.effect_params, step)

    def check_overcrowding(self, active_patient_count: int) -> float:
        """
        Called each tick by environment.
        Returns overcrowding modifier (1.0 or 1.2) based on active patient count.
        """
        overcrowded_threshold = 3
        modifier = 1.2  # 20% faster deterioration when overcrowded

        if active_patient_count > overcrowded_threshold:
            if not self._overcrowding_active:
                self._overcrowding_active = True
                if self._callback:
                    self._callback("overcrowding_started", active_patient_count)
            return modifier
        else:
            self._overcrowding_active = False
            return 1.0

    def tick(self) -> None:
        """Advance news cycle timers."""
        if self._news_cycle_steps_remaining > 0:
            self._news_cycle_steps_remaining -= 1
        if self._news_cycle_steps_remaining == 0:
            self._pending_surge_probability = max(0.0, self._pending_surge_probability - 0.10)

    def get_surge_probability(self) -> float:
        """Returns current additional surge probability from news cycles."""
        return self._pending_surge_probability

    def _apply_effect(self, effect: str, params: Dict, step: int) -> None:
        if effect == "spawn_secondary":
            if self._callback:
                self._callback(
                    "spawn_secondary",
                    condition=params["secondary_condition"],
                    reason=params["reason"],
                    triggered_at_step=step,
                )
        elif effect == "news_cycle":
            self._news_cycle_steps_remaining = params["steps"]
            self._pending_surge_probability += params["surge_prob_boost"]
            if self._callback:
                self._callback(
                    "news_cycle",
                    message=f"News cycle: successful {params.get('condition', 'emergency')} save draws attention",
                    steps=params["steps"],
                )

    # =========================================================================
    # For testing
    # =========================================================================

    @property
    def overcrowding_modifier(self) -> float:
        return 1.2 if self._overcrowding_active else 1.0

    @property
    def news_cycle_steps_remaining(self) -> int:
        return self._news_cycle_steps_remaining
```

### 4.3 PatientManager Changes

#### File: `server/subsystems/patient_manager.py`

**Changes to `_spawn_patients()`:**
- For each patient: pick `dispatch_category` from weighted distribution, pick `true_condition` from `DISPATCH_CATEGORY_MAP[category]`, set `observed_condition = None` (hidden until on-scene)
- Set `severity_modifier = rng.uniform(0.7, 1.3)` (hidden from agent)
- Set `is_secondary = False` for primary patients

**Changes to `tick()` deterioration formula:**
```python
# OVERCROWDING MODIFIER: applied before deterioration rate
deterioration_rate = DETERIORATION_BASE_RATE * patient.severity_modifier
if overcrowding_modifier > 1.0:
    deterioration_rate *= overcrowding_modifier  # 1.2x when overcrowded
```

New `spawn_secondary()` signature:
```python
def spawn_secondary(
    self,
    condition: str,
    triggered_by: Optional[str] = None,
    reason: Optional[str] = None,
    onset_step: int = 0,
    spawn_node: Optional[str] = None,
) -> Patient:
    """Spawn a cascade-triggered secondary patient."""
    # is_secondary = True
    # cascade_trigger_reason = reason
    # observed_condition = condition (shown immediately — secondary patients are known)
    # severity_modifier = rng.uniform(0.7, 1.3)
```

### 4.4 Environment Integration

#### File: `server/codered_environment.py`

**New state fields:**
```python
self._pending_calls: List[DispatchCall] = []    # calls awaiting triage
self._dispatch_outcomes: List[DispatchOutcome] = []  # resolved outcomes
self._pending_call_countdown: Dict[str, int] = {}  # call_id → steps until spawn
self._dispatch_outcomes_history: List[DispatchOutcome] = []  # all outcomes for grader
self._cascade_engine: CascadeEngine = CascadeEngine()
```

**New `reset()` logic:**
```python
# After subsystem init:
self._cascade_engine.reset(seed=seed, episode_config=TASK_CONFIG[self._task_id])
self._cascade_engine.set_callback(self._cascade_callback)
self._pending_calls = []
self._pending_call_countdown = {}
self._dispatch_outcomes_history = []
```

**New `_cascade_callback()` method:**
```python
def _cascade_callback(self, event_type: str, **kwargs) -> None:
    if event_type == "spawn_secondary":
        condition = kwargs["condition"]
        reason = kwargs["reason"]
        triggered_at_step = kwargs["triggered_at_step"]
        node = kwargs.get("spawn_node", self._rng.choice(_SPAWN_NODES))

        patient = self.patient_manager.spawn_secondary(
            condition=condition,
            reason=reason,
            onset_step=triggered_at_step,
            spawn_node=node,
        )
        self._patients.append(patient)
        self._prev_vitals[patient.id] = 1.0
        self._prev_patient_status[patient.id] = "waiting"
        self._alerts.append(
            f"CASCADE: Secondary {condition} patient spawned (reason: {reason})"
        )
        self._log_event("secondary_patient_spawned", {
            "patient_id": patient.id,
            "condition": condition,
            "reason": reason,
            "triggered_at_step": triggered_at_step,
        })

    elif event_type == "overcrowding_started":
        count = kwargs["active_patient_count"]
        self._alerts.append(f"OVERCROWDING: ED at {count} active patients — deterioration accelerated")

    elif event_type == "news_cycle":
        msg = kwargs["message"]
        steps = kwargs["steps"]
        self._alerts.append(f"NEWS: {msg} (surge expected over next {steps} steps)")
        self._log_event("news_cycle", {"message": msg, "steps": steps})
```

**New `_advance_time()` integration:**
```python
# After patient_manager.tick():
overcrowding_modifier = self._cascade_engine.check_overcrowding(
    active_patient_count=len([p for p in self._patients if p.status not in TERMINAL])
)
# Pass to patient_manager.tick():
self._patient_manager.tick(onset_steps, step_count, overcrowding_modifier=overcrowding_modifier)

# After disruption_engine.roll():
self._cascade_engine.tick()

# During pending call countdown:
for call_id in list(self._pending_call_countdown.keys()):
    self._pending_call_countdown[call_id] -= 1
    if self._pending_call_countdown[call_id] <= 0:
        # Patient has been waiting too long — force spawn
        self._spawn_patient_from_call(call_id)
```

**New `_spawn_patient_from_call(call_id)` method:**
```python
def _spawn_patient_from_call(self, call_id: str) -> None:
    """Force-spawn a patient when a pending call's timer expires."""
    call = next((c for c in self._pending_calls if c.call_id == call_id), None)
    if call is None:
        return

    # Find or create matching patient
    patient = next(
        (p for p in self._patients if p.dispatch_call_id == call_id),
        None
    )
    if patient is None:
        # Create patient from call data
        patient = self.patient_manager.spawn_primary_from_call(call)
        self._patients.append(patient)
        self._prev_vitals[patient.id] = 1.0
        self._prev_patient_status[patient.id] = "waiting"

    # Reveal condition to agent (update call + patient observed_condition)
    call.true_condition = patient.condition
    call.als_needed = patient.condition in ("cardiac", "stroke", "trauma")
    call.revealed_at_step = self._state.step_count
    patient.observed_condition = patient.condition

    self._alerts.append(f"ON-SCENE: {call.category.value} call {call_id} arrived — condition: {patient.condition}")
    self._log_event("call_arrived_on_scene", {
        "call_id": call_id,
        "condition": patient.condition,
        "step": self._state.step_count,
    })
```

**Action handlers:**

```python
def _do_triage_call(self, action: TriageCall) -> None:
    call = next((c for c in self._pending_calls if c.call_id == action.call_id), None)
    if call is None:
        self._alerts.append(f"TriageCall: call_id {action.call_id} not found")
        return

    if action.decision == "dispatch_als":
        if action.ambulance_id is None:
            self._alerts.append("DispatchALS requires ambulance_id")
            return
        # Check ambulance is ALS and available
        amb = self._ambulance_manager.get(action.ambulance_id)
        if amb is None or amb.equipment != "ALS" or amb.status != "available":
            self._alerts.append(f"ALS ambulance {action.ambulance_id} not available")
            return
        result = self._ambulance_manager.dispatch(
            action.ambulance_id, call.location_node, self._road_network
        )
        if result["success"]:
            # Track dispatch for post-hoc reward
            outcome = DispatchOutcome(
                call_id=call.call_id,
                decision="als",
                category=call.category,
                true_condition=None,  # revealed later
            )
            self._dispatch_outcomes_history.append(outcome)
            self._pending_calls.remove(call)
            if call.call_id in self._pending_call_countdown:
                del self._pending_call_countdown[call.call_id]
            self._alerts.append(f"ALS dispatched to {call.category.value} call {call.call_id}")
        else:
            self._alerts.append(f"Dispatch failed: {result.get('reason', 'unknown')}")

    elif action.decision == "dispatch_bls":
        # Similar logic for BLS, ambulance check equipment == "BLS"
        pass

    elif action.decision == "self_transport":
        # Patient advised to self-transport — no ambulance used
        outcome = DispatchOutcome(
            call_id=call.call_id,
            decision="self_transport",
            category=call.category,
            true_condition=None,
        )
        self._dispatch_outcomes_history.append(outcome)
        self._pending_calls.remove(call)
        self._alerts.append(f"Self-transport advised for {call.category.value} call {call.call_id}")

    elif action.decision == "callback":
        # Non-urgent, call back later — put back in queue with longer wait
        call.time_waiting = 0  # reset timer
        self._pending_call_countdown[action.call_id] = 15  # callback in 15 steps
        self._alerts.append(f"Callback scheduled for {call.category.value} call {action.call_id}")

    elif action.decision == "no_dispatch":
        # Refuse ambulance — patient may deteriorate
        outcome = DispatchOutcome(
            call_id=call.call_id,
            decision="no_dispatch",
            category=call.category,
            true_condition=None,
        )
        self._dispatch_outcomes_history.append(outcome)
        self._pending_calls.remove(call)
        self._alerts.append(f"No dispatch for {call.category.value} call {action.call_id} — deferred")
```

**On ambulance arrival (in `_do_treatment_arrival()`):**
```python
# Reveal true condition for patients that came from dispatch calls
if patient.dispatch_call_id:
    for outcome in reversed(self._dispatch_outcomes_history):
        if outcome.call_id == patient.dispatch_call_id and outcome.true_condition is None:
            outcome.true_condition = patient.condition
            outcome.als_needed = patient.condition in ("cardiac", "stroke", "trauma")
            outcome.revealed_at_step = self._state.step_count
            patient.observed_condition = patient.condition
            break
```

**Post-hoc dispatch classification reward:**
```python
def _compute_dispatch_classification_reward(self) -> float:
    """Additional reward for dispatch accuracy. Called at end of episode."""
    reward = 0.0
    for outcome in self._dispatch_outcomes_history:
        if outcome.true_condition is None:
            continue  # Outcome never resolved

        als_needed = outcome.als_needed
        als_dispatched = outcome.decision == "als"

        if als_needed and als_dispatched:
            reward += 0.05   # correct ALS dispatch
        elif not als_needed and not als_dispatched:
            reward += 0.02   # correct non-ALS decision
        elif als_needed and not als_dispatched:
            reward -= 0.20   # BLS/taxi sent to cardiac/stroke/trauma — patient dies
        elif not als_needed and als_dispatched:
            reward -= 0.05   # ALS over-dispatched — resource wasted

        if outcome.decision in ("self_transport", "callback") and als_needed:
            reward -= 0.30   # refused ambulance to emergency patient — large penalty

    return reward
```

### 4.5 Reward Function Changes

#### Step-level reward additions:
- **Dispatch milestone**: `waiting→dispatched` remains at +0.20 (no change)
- **Pre-existing** milestone bonuses unchanged

#### Episode-level reward additions:
- **Dispatch classification reward**: Computed at episode end via `_compute_dispatch_classification_reward()`. Added to `cum_reward`.
- **Secondary patient bonus**: +0.30 for each secondary patient treated before their target time. −0.40 for each secondary patient deceased.

#### Dense reward changes:
- Overcrowding modifier (1.2×) naturally accelerates vitals decline → accelerates negative reward signal when overcrowded → agents learn to reduce load

### 4.6 Grading Changes

#### File: `server/grader.py`

**New axis: `cascade_score`** (10% weight, replacing part of secondary_harm):
```python
@dataclass
class RubricResult:
    # ... existing fields ...
    cascade_score: float = 0.0  # NEW

def grade_cascade_score(episode_log: list[dict]) -> float:
    """
    Scores how well the agent managed cascade effects.
    - Secondary patients saved / total secondary patients (50% weight)
    - Overcrowding events prevented (30% weight)
    - News cycle surge handled (20% weight)
    """
    secondary_patient_ids = {e["patient_id"] for e in episode_log if e.get("event") == "secondary_patient_spawned"}
    secondary_saves = [e for e in episode_log if e.get("event") == "treatment_complete"
                       and e.get("patient_id") in secondary_patient_ids]

    overcrowding_events = [e for e in episode_log if e.get("event") == "overcrowding_started"]
    news_cycles = [e for e in episode_log if e.get("event") == "news_cycle"]

    if len(secondary_patients) == 0:
        return 1.0  # No cascades = no penalty

    secondary_score = 1.0 - (len([p for p in secondary_patients
                                    if not p.get("saved", False)]) / len(secondary_patients))

    # Overcrowding: fewer events = better
    overcrowding_score = max(0.0, 1.0 - len(overcrowding_events) / 5.0)

    # News cycle: can't prevent, but how many secondary patients did you handle?
    news_score = 1.0 - (len(news_cycles) / 10.0)

    return 0.5 * secondary_score + 0.3 * overcrowding_score + 0.2 * news_score
```

**Updated final score formula:**
```python
final_score = (
    0.32 * time_score +
    0.16 * efficiency +
    0.16 * secondary_harm +
    0.16 * prep_ready +
    0.08 * vitals_score_avg +
    0.10 * cascade_score +
    0.02 * dispatch_classification_score
    - mutual_aid_penalty
    - cross_validation_penalty
    - icu_boarding_penalty
)
```

### 4.7 Call Spawning System

New calls arrive over time, not all at once at reset. The environment generates new dispatch calls at configurable intervals.

**In `constants.py`:**
```python
CALL_SPAWN_INTERVAL = 8  # steps between new call spawns
MAX_PENDING_CALLS = 5     # max calls in queue before forced dispatch
FORCE_SPAWN_THRESHOLD = 20  # steps waiting → force spawn patient

CALL_SEVERITY_ESCALATION = 0.05  # +5% severity per step waiting
```

**In `environment._advance_time()`:**
```python
# Spawn new dispatch calls periodically
self._step_count_since_last_call += 1
if self._step_count_since_last_call >= CALL_SPAWN_INTERVAL:
    self._spawn_dispatch_call()
    self._step_count_since_last_call = 0
```

**New `environment._spawn_dispatch_call()`:**
```python
def _spawn_dispatch_call(self) -> None:
    """Spawn a new dispatch call with random category."""
    call_id = f"CALL_{self._state.step_count:04d}"
    category = self._rng.choices(
        population=list(DISPATCH_CATEGORY_MAP.keys()),
        weights=[0.20, 0.15, 0.25, 0.20, 0.20],  # weighted distribution
    )[0]
    location = self._rng.choice(_SPAWN_NODES)

    call = DispatchCall(
        call_id=call_id,
        category=category,
        location_node=location,
        time_waiting=0,
        estimated_severity=0.1,
    )
    self._pending_calls.append(call)
    self._pending_call_countdown[call_id] = FORCE_SPAWN_THRESHOLD
    self._log_event("call_received", {
        "call_id": call_id,
        "category": category.value,
        "location": location,
        "step": self._state.step_count,
    })
```

---

## Phase 5: Research Comparison

### Existing RL Environments with Partial Observability

| Environment | Partial Observability | Cascade Dynamics | Novelty vs CodeRed |
|------------|----------------------|-----------------|-------------------|
| POMDPy / POMDPs.jl | Full POMDP support | None | Abstract, no domain novelty |
| BabyAI | Partial vision | None | Gridworld, synthetic language |
| NetHack | Partial memory | None | Single-agent, no inter-outcome links |
| Hanabi | Partial information | None | Cooperative, no cascade |
| Overcooked | Partial communication | None | Cooperative, no domain novelty |
| Traffic environments (SUMO) | Traffic state only | None | Optimization, not emergent |

**Our differentiation:** CodeRed combines imperfect information at dispatch (who gets an ambulance?) with interdependent outcomes (a death creates a new patient). No existing RL benchmark has this combination.

### Existing RL Environments with Interdependent Dynamics

| Environment | Interdependent Outcomes | Novelty vs CodeRed |
|------------|------------------------|-------------------|
| WarpDrive (multi-agent) | Agents interact via env | No cascade mechanic |
| MaCAO | Causal consequences | Abstract, not real-world |
| EMSNet | Patient prioritization | No cascade, deterministic outcomes |

**Our differentiation:** Causal cascade in CodeRed is grounded in real emergency medicine literature (psychogenic cardiac events after witnessed death, ED overcrowding increasing mortality). This makes it interpretable and evaluable by domain experts.

### What Is Genuinely New

1. **Vague dispatch categories → true condition reveal post-arrival**: No RL benchmark gives the agent a probability distribution over conditions at dispatch decision time.
2. **Patient outcome → secondary patient spawn**: No benchmark has outcome-dependent patient population changes.
3. **Dispatch classification reward post-hoc**: No benchmark evaluates the agent's dispatch-level decision accuracy against ground truth revealed later.
4. **Combined: POMDP dispatch + interdependent outcomes**: This specific combination of imperfect information at the input layer and emergent consequences at the output layer is novel.

---

## Phase 6: Creative Extensions (Future Phases)

### Extension 1: Dispatch Accuracy Feedback Loop
After each episode, the grader generates a "dispatch accuracy report" showing every dispatch call, the agent's decision, the true condition, and the outcome. This report is fed as additional context to the next episode's observation. Agents that learn from this feedback outperform those that don't.

### Extension 2: Hospital Reputation System
Builds on cascade_score. Each hospital has a `reputation_score`. Good outcomes at HOSP_A → reputation +0.05, bad outcomes → reputation −0.05. Reputation affects patient willingness to go there (some patients self-divert). Creates inter-hospital competition dynamics.

### Extension 3: Adaptive Call Volume
Based on the agent's last N episodes, adjust the call spawn rate. High-performing agents get more calls (success breeds demand). Low-performing agents get fewer (community loses trust). This creates a "success trap" where competence generates new challenges.

### Extension 4: Blood Type Cascade
Untreated trauma patients may need emergency blood. If HOSP_A runs out of O-NEG (emergency reserve depleted), it triggers a hospital-to-hospital transfer request. If no hospital has stock, the patient dies. This chains blood_bank into the cascade system.

---

## Phase 7: Implementation Roadmap

### Task 17: Pre-Dispatch Uncertainty Foundation
1. Add `DispatchCategory` enum and `DISPATCH_CATEGORY_MAP` to `constants.py`
2. Add `DispatchCall` and `DispatchOutcome` dataclasses to `entities.py`
3. Add `DispatchALS`, `DispatchBLS`, `TriageCall` action types to `actions.py`
4. Add `pending_calls`, `recent_dispatch_outcomes`, `overcrowding_modifier` to `observations.py`
5. Remove `DispatchAmbulance` action (replace with ALS/BLS)
6. Add `_pending_calls`, `_pending_call_countdown`, `_dispatch_outcomes_history` to environment
7. Implement `_spawn_dispatch_call()` and call spawning logic in `_advance_time()`
8. Implement `_do_triage_call()` action handler
9. Update ambulance dispatch to link to `call_id`
10. Implement `_spawn_patient_from_call()` — condition reveal on-scene
11. Implement `_compute_dispatch_classification_reward()`
12. Add unit tests for call spawning, triage action, condition reveal
13. Update e2e tests for new action space

**Files modified:** `constants.py`, `entities.py`, `actions.py`, `observations.py`, `codered_environment.py`
**Files created:** None (no new subsystem in this task)
**Test files:** `test_environment.py` (new tests), `test_e2e.py` (updated)

### Task 18: Causal Cascade Engine
1. Create `server/subsystems/cascade_engine.py` — `CascadeRule`, `CascadeEngine` class
2. Implement 5 cascade rules (witnessed death → cardiac, news cycle → surge, overcrowding)
3. Integrate `CascadeEngine` into environment `reset()` and `_advance_time()`
4. Implement `_cascade_callback()` for secondary patient spawning
5. Add `overcrowding_modifier` to patient_manager `tick()` (1.2× deterioration rate)
6. Update `_log_event()` to track cascade events for grader
7. Add unit tests for `CascadeEngine` (probability rolls, overcrowding check, news cycle)
8. Add integration test for secondary patient spawning on death event
9. Update e2e tests to verify secondary patients appear and are scored

**Files modified:** `codered_environment.py`, `patient_manager.py`
**Files created:** `server/subsystems/cascade_engine.py`
**Test files:** `test_cascade_engine.py` (new), `test_environment.py` (update), `test_e2e.py` (update)

### Task 19: Grader Integration + Cascade Scoring
1. Add `cascade_score` axis to `RubricResult` in `grader.py`
2. Implement `grade_cascade_score()` in `grader.py`
3. Update `grade_episode()` and `grade_from_environment()` to compute `cascade_score`
4. Update final score formula to include `cascade_score` and `dispatch_classification_score`
5. Add grader unit tests for `cascade_score` computation
6. Update `RubricResult` in `observations.py` if needed (add `cascade_score_preview`)

**Files modified:** `grader.py`, `observations.py`
**Test files:** `test_grader.py` (update)

### Task 20: Baseline Inference + Documentation
1. Update `inference.py` for new action space (TriageCall, DispatchALS, DispatchBLS)
2. Verify baseline scores on Tasks 1-3 still pass (backwards compatibility)
3. Update `README.md` with new action/observation space descriptions
4. Update `openenv.yaml` if needed (task metadata changes)
5. Run full test suite: all 129 + new tests must pass
6. Docker build verification

**Files modified:** `inference.py`, `README.md`, `openenv.yaml`
**Test files:** Full suite regression test

---

## Backward Compatibility

- Tasks 1-3 continue to work with existing action space. New tasks (4, 5) use the new mechanics.
- `DispatchAmbulance` is **replaced**, not added. Existing agents using it will break in Phase 2. This is intentional — the action space change is the mechanic.
- All patient fields added are optional with defaults. Existing episode logs are compatible with the grader (new fields ignored).
- Cascade engine is inactive in tasks without cascade configuration.
- `vitals_score` field existing from Phase 1 is unchanged.

---

## Open Questions (Deferred)

- Should `TriageCall` be mandatory each step, or can the agent `MaintainPlan` instead? (Decision: MaintainPlan allowed — agent can skip triage decisions)
- Should callback patients that re-enter the queue be tracked separately in grading? (Decision: Deferred — not scored in Phase 2)
- What's the maximum number of pending calls? (Decision: 5, enforced by dropping oldest non-urgent call if exceeded)
- How does mutual aid interact with dispatch calls? (Decision: Mutual aid ambulances are dispatched to existing patients, not pending calls — separate pool)
- Should BLS dispatch to a cardiac call trigger an automatic upgrade to ALS mid-transport? (Decision: No — wrong-level dispatch is penalized post-hoc in grading, not corrected mid-episode)
