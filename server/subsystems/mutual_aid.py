"""Mutual aid ambulance subsystem — manages MA request lifecycle, arrival, and patient assignment."""

from dataclasses import dataclass, field
from typing import Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .patient_manager import PatientManager
    from .hospital_system import HospitalSystem


@dataclass
class MAAmbulance:
    """A temporary mutual-aid ambulance entry."""
    ambulance_id: str
    patient_id: Optional[str] = None
    hospital_id: Optional[str] = None
    status: str = "pending"  # pending | assigned | transporting | delivered


@dataclass
class MAPending:
    """A mutual-aid request that hasn't arrived yet."""
    ambulance_id: str
    patient_id: Optional[str]
    arrival_step: int
    source_node: str
    request_step: int = 0


class MutualAidManager:
    """
    Manages mutual-aid ambulance lifecycle.

    Lifecycle:
      request() → pending entry created
      tick() → checks for arrivals, promotes to active
      environment calls on_arrival() → triggers treatment arrival
    """

    def __init__(
        self,
        available_calls: int = 0,
        seed: Optional[int] = None,
    ):
        import random
        self._rng = random.Random(seed)
        self._available_calls: int = available_calls
        self._used_calls: int = 0
        self._pending: dict[str, MAPending] = {}   # ma_id → pending record
        self._active: dict[str, MAAmbulance] = {}   # ma_id → active record

    def get_available(self) -> int:
        return self._available_calls

    def get_used(self) -> int:
        return self._used_calls

    def request(
        self,
        step_count: int,
        road_network,
        patients: list,
    ) -> Optional[str]:
        """
        Request a mutual-aid ambulance.
        Returns the MA ambulance ID, or None if no calls available.
        Auto-assigns to highest-priority waiting patient.
        """
        from .constants import MA_SOURCE_NODE, SCENE_TIME

        if self._available_calls <= 0:
            return None

        self._available_calls -= 1
        self._used_calls += 1
        ma_id = f"MUTUAL_{self._used_calls}"

        # Auto-assign to highest-priority waiting patient
        priority_order = ["cardiac", "stroke", "trauma", "general"]
        target_patient = None
        for cond in priority_order:
            candidates = [
                p for p in patients
                if getattr(p, "status", None) == "waiting" and getattr(p, "condition", None) == cond
            ]
            if candidates:
                target_patient = candidates[0]
                break

        if target_patient is None:
            from .constants import MA_BASE_TRAVEL_TIME
            arrival_step = step_count + MA_BASE_TRAVEL_TIME
            self._pending[ma_id] = MAPending(
                ambulance_id=ma_id,
                patient_id=None,
                arrival_step=arrival_step,
                source_node=MA_SOURCE_NODE,
                request_step=step_count,
            )
            return ma_id

        route = road_network.shortest_path(MA_SOURCE_NODE, target_patient.location_node)
        travel_time = road_network.route_travel_time(route) if route else 12
        arrival_step = step_count + travel_time + SCENE_TIME
        self._pending[ma_id] = MAPending(
            ambulance_id=ma_id,
            patient_id=target_patient.id,
            arrival_step=arrival_step,
            source_node=MA_SOURCE_NODE,
            request_step=step_count,
        )
        return ma_id

    def tick(
        self,
        step_count: int,
        patient_manager: "PatientManager",
        hospital_system: "HospitalSystem",
        arrival_callback: Callable[[str, str], None],  # (ma_id, patient_id) → env delivers patient
    ) -> list[dict]:
        """
        Advance MA state by one step.
        Returns list of arrival event dicts to be logged by the environment.
        """
        arrivals = []

        for ma_id, pending in list(self._pending.items()):
            if step_count < pending.arrival_step:
                continue

            patient_id = pending.patient_id
            original_patient_id = pending.patient_id
            patient_changed = False

            # Auto-assign to highest-priority waiting patient if none assigned at request time
            if patient_id is None:
                priority_order = ["cardiac", "stroke", "trauma", "general"]
                for cond in priority_order:
                    candidates = [
                        p for p in patient_manager.patients
                        if getattr(p, "status", None) == "waiting" and getattr(p, "condition", None) == cond
                    ]
                    if candidates:
                        patient_id = candidates[0].id
                        pending.patient_id = patient_id
                        patient_changed = True
                        break

            if patient_id:
                patient = patient_manager.get(patient_id)
                if patient and getattr(patient, "status", None) == "waiting":
                    # Auto-assign hospital
                    hosp_id = self._auto_assign_hospital(patient.condition, hospital_system)
                    patient.assigned_hospital = hosp_id
                    patient.status = "transporting"
                    patient.assigned_ambulance = ma_id
                    self._active[ma_id] = MAAmbulance(
                        ambulance_id=ma_id,
                        patient_id=patient_id,
                        hospital_id=hosp_id,
                        status="transporting",
                    )

                    arrivals.append({
                        "ambulance_id": ma_id,
                        "patient_id": patient_id,
                        "hospital_id": hosp_id,
                        "actual_arrival_step": step_count,
                        "had_patient": True,
                        "patient_changed": patient_changed,
                        "original_patient_id": original_patient_id,
                    })
                    # Trigger instant delivery: MA picks up and brings to hospital
                    arrival_callback(ma_id, patient_id)
                    self._active.pop(ma_id, None)
                else:
                    arrivals.append({
                        "ambulance_id": ma_id,
                        "patient_id": patient_id,
                        "actual_arrival_step": step_count,
                        "had_patient": False,
                        "patient_changed": patient_changed,
                        "original_patient_id": original_patient_id,
                    })
            else:
                arrivals.append({
                    "ambulance_id": ma_id,
                    "patient_id": None,
                    "actual_arrival_step": step_count,
                    "had_patient": False,
                    "patient_changed": patient_changed,
                    "original_patient_id": original_patient_id,
                })

            del self._pending[ma_id]

        return arrivals

    def _auto_assign_hospital(
        self,
        condition: str,
        hospital_system: "HospitalSystem",
    ) -> str:
        """Auto-assign hospital for a MA patient based on condition."""
        if condition in ("cardiac", "stroke") and hospital_system.can_treat("HOSP_A", condition):
            return "HOSP_A"
        if condition in ("cardiac", "trauma") and hospital_system.can_treat("HOSP_B", condition):
            return "HOSP_B"
        if hospital_system.can_treat("HOSP_C", condition):
            return "HOSP_C"
        # Fallback: first hospital that can treat
        for hosp_id in ("HOSP_A", "HOSP_B", "HOSP_C"):
            if hospital_system.can_treat(hosp_id, condition):
                return hosp_id
        return "HOSP_A"

    def get_pending(self) -> dict[str, MAPending]:
        return dict(self._pending)

    def get_active(self) -> dict[str, MAAmbulance]:
        return dict(self._active)
