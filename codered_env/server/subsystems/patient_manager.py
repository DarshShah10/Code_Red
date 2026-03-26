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

    def reset(self, task_id: str, rng: Optional[random.Random]) -> None:
        self._rng = rng or random.Random()
        self._task_id = task_id
        self._patient_counter = 0
        self.patients = []
        self._spawn_patients()

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

    def mark_deceased(self, patient_id: str, reason: str) -> None:
        p = self.get(patient_id)
        if p:
            p.status = "deceased"
            p.outcome = "deceased"

    def tick(self) -> None:
        """Called each step to advance patient deterioration."""
        pass
