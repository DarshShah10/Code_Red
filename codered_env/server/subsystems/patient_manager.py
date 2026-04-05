"""Patient manager subsystem: patient creation, state transitions, treatment completion."""

import random
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Patient:
    id: str
    condition: str  # CARDIAC | STROKE | TRAUMA | GENERAL
    status: str  # waiting | dispatched | in_treatment | treated | deceased
    blood_type: Optional[str] = None
    assigned_hospital: Optional[str] = None
    location_node: str = ""
    onset_step: int = 0
    treatment_complete_time: Optional[int] = None  # MUST be set on treatment
    outcome: Optional[str] = None  # saved | deceased
    arrival_hospital_step: Optional[int] = None
    vitals_score: float = 1.0
    _vitals_frozen: bool = False
    icu_status: Optional[str] = None  # "admitted" | "boarding" | None
    # Phase 2 fields:
    dispatch_call_id: Optional[str] = None
    is_secondary: bool = False
    cascade_trigger_reason: Optional[str] = None
    observed_condition: Optional[str] = None


TERMINAL_STATUSES = frozenset({"treated", "deceased"})


class PatientManager:
    """
    Manages patient lifecycle: creation, state transitions, treatment completion.

    Called by the environment's reset() and step().
    """

    def __init__(self):
        self.patients: list[Patient] = []
        self._rng: Optional[random.Random] = None
        self._task_id: str = "task1"
        self._patient_counter: int = 0
        self._onset_steps: dict[str, int] = {}
        self._patients_dict_cache: Optional[dict[str, Patient]] = None

    def reset(self, task_id: str, rng: Optional[random.Random]) -> None:
        self._rng = rng or random.Random()
        self._task_id = task_id
        self._patient_counter = 0
        self.patients = []
        self._onset_steps = {}
        self._patients_dict_cache = None
        self._spawn_patients()
        for p in self.patients:
            self._onset_steps[p.id] = p.onset_step

    # Nodes where patients can spawn (incident scenes — not hospital nodes)
    _SPAWN_NODES = [
        "RAJIV_CHOWK", "LAJPAT_NAGAR", "CHOWKHA", "RAILWAY_XING",
        "NH45_BYPASS", "IT_HUB", "MG_CHOWK", "SECTOR_12", "RING_ROAD",
    ]

    def _spawn_patients(self) -> None:
        """Spawn patients based on task configuration with randomized locations and onset steps."""
        from .constants import TASK_CONFIG  # string keys: "task1", "task2", "task3"

        config = TASK_CONFIG.get(self._task_id, TASK_CONFIG["task1"])
        for pdef in config["patients"]:
            self._patient_counter += 1
            # Randomize patient spawn location (incident scenes only, not hospital nodes)
            location = self._rng.choice(self._SPAWN_NODES)
            # Jitter onset_step for multi-patient tasks (task2/task3); task1 stays deterministic
            base_onset = pdef.get("onset_step", 0)
            if self._task_id in ("task2", "task3"):
                onset_jitter = self._rng.randint(-3, 3)
                onset = max(0, base_onset + onset_jitter)
            else:
                onset = base_onset
            patient = Patient(
                id=f"P{self._patient_counter}",
                condition=pdef["condition"],
                status="waiting",
                location_node=location,
                onset_step=onset,
            )
            self.patients.append(patient)

    def get(self, patient_id: str) -> Optional[Patient]:
        return self.patients_dict.get(patient_id)

    def mark_treated(self, patient_id: str, treatment_complete_time: int) -> None:
        """Mark patient as treated. CRITICAL: sets treatment_complete_time for grader."""
        p = self.get(patient_id)
        if p:
            p.status = "treated"
            p.treatment_complete_time = treatment_complete_time
            p.outcome = "saved"
            p._vitals_frozen = True

    def mark_deceased(self, patient_id: str, reason: str = "timeout") -> None:
        p = self.get(patient_id)
        if p:
            p.status = "deceased"
            p.outcome = "deceased"
            p.vitals_score = 0.0
            p._vitals_frozen = True

    def tick(self, onset_steps: dict[str, int], step_count: int, overcrowding_modifier: float = 1.0) -> None:
        """Advance patient vitals deterioration. Call once per environment step."""
        from .constants import (
            VITALS_STABLE_DECAY_RATE, VITALS_DETERIORATING_THRESHOLD,
            VITALS_CRITICAL_THRESHOLD, VITALS_DEAD_THRESHOLD,
            PATIENT_TARGET_TIMES,
        )
        for patient in self.patients:
            if patient.status in TERMINAL_STATUSES or patient._vitals_frozen:
                continue

            effective_time = step_count - onset_steps.get(patient.id, patient.onset_step)
            target_time = PATIENT_TARGET_TIMES.get(patient.condition, 60)

            if effective_time <= target_time:
                # Stable window: slow recovery, clamped to 1.0
                patient.vitals_score = min(1.0, patient.vitals_score + VITALS_STABLE_DECAY_RATE)
            else:
                # Post-target: linear fall from 1.0 to 0.0 over one target_time window
                # Overcrowding_modifier speeds up deterioration (multiplies effective overtime)
                overtime_ratio = ((effective_time - target_time) * overcrowding_modifier) / target_time
                patient.vitals_score = max(0.0, 1.0 - overtime_ratio)

            # Status escalation (ordered: dead is subset of critical is subset of deteriorating)
            if patient.vitals_score <= VITALS_DEAD_THRESHOLD:
                self.mark_deceased(patient.id, reason="cardiac_arrest")
            elif patient.vitals_score <= VITALS_CRITICAL_THRESHOLD:
                patient.status = "critical"
            elif patient.vitals_score <= VITALS_DETERIORATING_THRESHOLD:
                patient.status = "deteriorating"

    def get_onset_steps(self) -> dict[str, int]:
        return self._onset_steps.copy()

    @property
    def patients_dict(self) -> dict[str, Patient]:
        if self._patients_dict_cache is None:
            self._patients_dict_cache = {p.id: p for p in self.patients}
        return self._patients_dict_cache

    def get_all(self) -> list["Patient"]:
        """Return all patients. Used by grader for cross-validation."""
        return self.patients

    def spawn_secondary(
        self,
        condition: str,
        onset_step: int,
        triggered_by: Optional[str] = None,
        reason: Optional[str] = None,
        spawn_node: Optional[str] = None,
    ) -> Patient:
        """
        Spawn a secondary (surge) patient at a random incident scene node.
        Secondary patients are counted in the secondary_harm grader axis.
        """
        location = spawn_node or self._rng.choice(self._SPAWN_NODES)
        self._patient_counter += 1
        patient = Patient(
            id=f"P{self._patient_counter}",
            condition=condition,
            status="waiting",
            location_node=location,
            onset_step=onset_step,
            is_secondary=True,
            cascade_trigger_reason=reason,
            observed_condition=condition,
        )
        self.patients.append(patient)
        self._onset_steps[patient.id] = onset_step
        self._patients_dict_cache = None
        return patient
