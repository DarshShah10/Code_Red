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

    def reset(self, task_id: str, rng: Optional[random.Random]) -> None:
        self._rng = rng or random.Random()
        self._task_id = task_id
        self._patient_counter = 0
        self.patients = []
        self._onset_steps = {}
        self._spawn_patients()
        for p in self.patients:
            self._onset_steps[p.id] = p.onset_step

    def _spawn_patients(self) -> None:
        """Spawn patients based on task configuration."""
        from .constants import TASK_CONFIG  # string keys: "task1", "task2", "task3"

        config = TASK_CONFIG.get(self._task_id, TASK_CONFIG["task1"])
        for pdef in config["patients"]:
            self._patient_counter += 1
            patient = Patient(
                id=f"P{self._patient_counter}",
                condition=pdef["condition"],
                status="waiting",
                location_node=pdef.get("location_node", "NH45_BYPASS"),
                onset_step=pdef.get("onset_step", 0),
            )
            self.patients.append(patient)

    def get(self, patient_id: str) -> Optional[Patient]:
        for p in self.patients:
            if p.id == patient_id:
                return p
        return None

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

    def tick(self, onset_steps: dict[str, int], step_count: int) -> None:
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
                overtime_ratio = (effective_time - target_time) / target_time
                patient.vitals_score = max(0.0, 1.0 - overtime_ratio)

            # Status escalation
            if patient.vitals_score <= VITALS_DETERIORATING_THRESHOLD:
                patient.status = "deteriorating"
            if patient.vitals_score <= VITALS_CRITICAL_THRESHOLD:
                patient.status = "critical"
            if patient.vitals_score <= VITALS_DEAD_THRESHOLD:
                self.mark_deceased(patient.id, reason="cardiac_arrest")

    def get_onset_steps(self) -> dict[str, int]:
        return self._onset_steps.copy()

    @property
    def patients_dict(self) -> dict[str, Patient]:
        return {p.id: p for p in self.patients}
