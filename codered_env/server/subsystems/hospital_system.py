"""Hospital resource management subsystem."""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional


@dataclass
class OR:
    index: int
    status: Literal["idle", "in_use", "prep"] = "idle"
    procedure_type: Optional[str] = None
    minutes_remaining: Optional[int] = None
    patient_id: Optional[str] = None
    assigned_specialist: Optional[str] = None


@dataclass
class Specialist:
    available: int
    total: int
    status: Literal["available", "paged", "en_route", "busy"] = "available"
    minutes_until_available: int = 0


@dataclass
class Hospital:
    id: str
    node_id: str
    capabilities: List[str]
    specialists: Dict[str, Specialist]
    operating_rooms: List[OR]
    icu_beds: Dict[str, int]
    blood_stock: Dict[str, int]
    on_diversion: bool = False
    or_prep_countdowns: Dict[int, int] = field(default_factory=dict)
    surgery_queue: Dict[str, int] = field(default_factory=dict)
    equipment_failures: Dict[int, int] = field(default_factory=dict)


class HospitalSystem:
    """Manages hospital resources: ORs, specialists, ICU beds."""

    def __init__(self, episode_start_hour: int = 8):
        from .constants import HOSPITALS, SHIFT_CONFIG, get_current_shift

        self._hospitals: Dict[str, Hospital] = {}
        self._episode_start_hour: int = episode_start_hour
        self._step_count: int = 0
        self._current_shift: str = get_current_shift(episode_start_hour, 0)

        for h in HOSPITALS:
            specialists = {}
            hosp_shift = SHIFT_CONFIG.get(self._current_shift, {}).get(h["id"], {})
            for role, data in h["specialists"].items():
                available = hosp_shift.get(role, data["available"])
                specialists[role] = Specialist(available=available, total=data["total"])
            ors = [OR(index=i) for i in range(h["num_or"])]
            self._hospitals[h["id"]] = Hospital(
                id=h["id"],
                node_id=h["node_id"],
                capabilities=h["capabilities"],
                specialists=specialists,
                operating_rooms=ors,
                icu_beds=dict(h["icu_beds"]),
                blood_stock=dict(h["blood_stock"]),
            )

    def _update_shift(self) -> None:
        """Detect and handle shift transitions, adjusting specialist availability."""
        from .constants import SHIFT_CONFIG, get_current_shift

        new_shift = get_current_shift(self._episode_start_hour, self._step_count)
        if new_shift == self._current_shift:
            return

        self._current_shift = new_shift
        shift_hospitals = SHIFT_CONFIG.get(new_shift, {})

        for h in self._hospitals.values():
            hosp_shift = shift_hospitals.get(h.id, {})
            for role, specialist in h.specialists.items():
                if specialist.total == 0:
                    continue
                shift_available = hosp_shift.get(role, specialist.available)
                # Don't reduce below the number already committed (in surgery)
                committed = specialist.total - specialist.available
                new_available = min(shift_available, specialist.total - committed)
                specialist.available = max(0, new_available)

    def get_current_shift(self) -> str:
        """Return the current shift name ('day', 'evening', 'night')."""
        return self._current_shift

    def get(self, hosp_id: str) -> Optional[Hospital]:
        return self._hospitals.get(hosp_id)

    def all(self) -> Dict[str, Hospital]:
        return self._hospitals

    def set_diversion(self, hosp_id: str, on_diversion: bool) -> None:
        h = self._hospitals.get(hosp_id)
        if h:
            h.on_diversion = on_diversion

    def set_equipment_failure(self, hosp_id: str, or_index: int, duration: int) -> None:
        """Make an OR unavailable for a set duration (disruption)."""
        h = self._hospitals.get(hosp_id)
        if h and 0 <= or_index < len(h.operating_rooms):
            h.equipment_failures[or_index] = duration

    def can_treat(self, hosp_id: str, condition: str) -> bool:
        """Check if hospital can treat a condition."""
        h = self._hospitals.get(hosp_id)
        if h is None or h.on_diversion:
            return False
        return condition in h.capabilities

    # =========================================================================
    # OR Management
    # =========================================================================

    def prepare_or(
        self,
        hosp_id: str,
        procedure_type: str,
    ) -> Dict:
        """
        Begin OR preparation. Returns dict with 'success' bool and optional 'reason'.
        """
        h = self._hospitals.get(hosp_id)
        if h is None:
            return {"success": False, "reason": f"Hospital {hosp_id} not found"}
        if h.on_diversion:
            return {"success": False, "reason": f"{hosp_id} is on diversion"}
        if procedure_type not in h.capabilities:
            return {
                "success": False,
                "reason": f"{hosp_id} cannot treat {procedure_type}"
            }
        idle_or = next((or_obj for or_obj in h.operating_rooms if or_obj.status == "idle"), None)
        if idle_or is None:
            return {"success": False, "reason": f"no idle OR at {hosp_id}"}

        idle_or.status = "prep"
        idle_or.procedure_type = procedure_type
        h.or_prep_countdowns[idle_or.index] = 10  # 10-minute prep
        return {"success": True, "or_index": idle_or.index}

    def start_surgery(
        self,
        hosp_id: str,
        or_index: int,
        procedure_type: str,
        patient_id: str,
        duration_minutes: int = 30,
    ) -> Dict:
        """Start a surgery in an OR."""
        h = self._hospitals.get(hosp_id)
        if h is None:
            return {"success": False, "reason": f"Hospital {hosp_id} not found"}
        or_obj = next((o for o in h.operating_rooms if o.index == or_index), None)
        if or_obj is None:
            return {"success": False, "reason": f"OR {or_index} not found"}
        if or_obj.status not in ("idle", "prep"):
            return {"success": False, "reason": f"OR {or_index} is {or_obj.status}"}

        or_obj.status = "in_use"
        or_obj.procedure_type = procedure_type
        or_obj.patient_id = patient_id
        or_obj.minutes_remaining = duration_minutes
        if or_index in h.or_prep_countdowns:
            del h.or_prep_countdowns[or_index]
        return {"success": True}

    def preempt_or(self, hosp_id: str, or_index: int) -> Dict:
        """Preempt an OR. Returns harm score and recovery time."""
        h = self._hospitals.get(hosp_id)
        if h is None:
            return {"success": False, "reason": f"Hospital {hosp_id} not found"}
        or_obj = next((o for o in h.operating_rooms if o.index == or_index), None)
        if or_obj is None:
            return {"success": False, "reason": f"OR {or_index} not found"}
        if or_obj.status == "idle":
            return {"success": False, "reason": f"OR {or_index} is already idle"}

        harm = min(1.0, (or_obj.minutes_remaining or 0) / 30.0)
        recovery_time = or_obj.minutes_remaining or 0

        or_obj.status = "idle"
        or_obj.procedure_type = None
        or_obj.minutes_remaining = None
        or_obj.patient_id = None

        return {
            "success": True,
            "harm": harm,
            "recovery_time": recovery_time,
        }

    def is_prepared(self, hosp_id: str) -> bool:
        """Check if hospital has at least one idle OR."""
        h = self._hospitals.get(hosp_id)
        if h is None:
            return False
        return any(or_obj.status == "idle" for or_obj in h.operating_rooms)

    def get_idle_or(self, hosp_id: str) -> Optional[OR]:
        """Return first idle OR or None."""
        h = self._hospitals.get(hosp_id)
        if h is None:
            return None
        return next((o for o in h.operating_rooms if o.status == "idle"), None)

    def get_prep_or(self, hosp_id: str) -> Optional[OR]:
        """Return first OR in prep phase or None."""
        h = self._hospitals.get(hosp_id)
        if h is None:
            return None
        return next((o for o in h.operating_rooms if o.status == "prep"), None)

    # =========================================================================
    # Specialist Management
    # =========================================================================

    def page_specialist(self, hosp_id: str, specialist_type: str) -> Dict:
        """Page a specialist. Returns success."""
        h = self._hospitals.get(hosp_id)
        if h is None:
            return {"success": False, "reason": f"Hospital {hosp_id} not found"}
        spec = h.specialists.get(specialist_type)
        if spec is None or spec.total == 0:
            return {"success": False, "reason": f"No {specialist_type} at {hosp_id}"}
        if spec.available <= 0:
            return {"success": False, "reason": f"No {specialist_type} available at {hosp_id}"}
        spec.available -= 1
        spec.status = "paged"
        spec.minutes_until_available = 8  # 8-minute page latency
        return {"success": True}

    def is_specialist_available(self, hosp_id: str, specialist_type: str) -> bool:
        h = self._hospitals.get(hosp_id)
        if h is None:
            return False
        spec = h.specialists.get(specialist_type)
        if spec is None:
            return False
        return spec.status == "available" and spec.available > 0

    # =========================================================================
    # ICU Bed Management (Task 13)
    # =========================================================================

    def consume_icu_bed(self, hosp_id: str) -> bool:
        """
        Try to consume an ICU bed for an incoming patient.
        Returns True if bed was consumed, False if ICU is full.
        """
        h = self._hospitals.get(hosp_id)
        if h is None:
            return False
        if h.icu_beds["available"] <= 0:
            return False
        h.icu_beds["available"] -= 1
        return True

    def release_icu_bed(self, hosp_id: str) -> None:
        """Release an ICU bed when a patient is discharged or deceased."""
        h = self._hospitals.get(hosp_id)
        if h is None:
            return
        if h.icu_beds["available"] < h.icu_beds["total"]:
            h.icu_beds["available"] += 1

    # =========================================================================
    # Time Advance
    # =========================================================================

    def tick(self) -> None:
        """Advance all hospital timers by 1 minute and handle shift transitions."""
        self._step_count += 1
        self._update_shift()

        for h in self._hospitals.values():
            # OR prep countdown
            for or_idx in list(h.or_prep_countdowns.keys()):
                remaining = h.or_prep_countdowns[or_idx]
                if remaining <= 1:
                    del h.or_prep_countdowns[or_idx]
                    for or_obj in h.operating_rooms:
                        if or_obj.index == or_idx:
                            or_obj.status = "idle"
                else:
                    h.or_prep_countdowns[or_idx] = remaining - 1

            # OR in-use countdown
            for or_obj in h.operating_rooms:
                if or_obj.status == "in_use" and or_obj.minutes_remaining:
                    if or_obj.minutes_remaining <= 1:
                        or_obj.minutes_remaining = None
                        or_obj.status = "idle"
                        or_obj.procedure_type = None
                        or_obj.patient_id = None
                    else:
                        or_obj.minutes_remaining -= 1

            # Specialist recovery
            for spec in h.specialists.values():
                if spec.status in ("paged", "en_route") and spec.minutes_until_available > 0:
                    if spec.minutes_until_available <= 1:
                        spec.status = "available"
                        if spec.available < spec.total:
                            spec.available += 1
                        spec.minutes_until_available = 0
                    else:
                        spec.minutes_until_available -= 1

            # Equipment failure countdown
            for or_idx in list(h.equipment_failures.keys()):
                h.equipment_failures[or_idx] -= 1
                if h.equipment_failures[or_idx] <= 0:
                    del h.equipment_failures[or_idx]
