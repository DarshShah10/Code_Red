"""Blood bank subsystem: crossmatch, emergency release, transfers."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class BloodBank:
    hospital_id: str
    stocks: Dict[str, int]  # blood_type -> units
    crossmatch_queue: List[Dict] = field(default_factory=list)


class BloodBankSystem:
    """Manages blood supply, crossmatching, and transfers across hospitals."""

    def __init__(self):
        from .constants import HOSPITALS

        self._banks: Dict[str, BloodBank] = {}
        for h in HOSPITALS:
            self._banks[h["id"]] = BloodBank(
                hospital_id=h["id"],
                stocks=dict(h["blood_stock"]),
            )
        self._pending_completed: List[Dict] = []  # completed crossmatches from last tick()
        self._emergency_released: Dict[str, int] = {h["id"]: 0 for h in HOSPITALS}

    def get(self, hosp_id: str) -> Optional[BloodBank]:
        return self._banks.get(hosp_id)

    def all(self) -> Dict[str, BloodBank]:
        return self._banks

    # =========================================================================
    # Blood Operations
    # =========================================================================

    def emergency_release(
        self,
        hosp_id: str,
        patient_id: str,
        blood_type: str,
        units: int,
    ) -> Dict:
        """Instant O-Neg emergency release with per-call and per-episode caps."""
        bank = self._banks.get(hosp_id)
        if bank is None:
            return {"success": False, "reason": f"Hospital {hosp_id} not found"}

        # Per-call cap: max 4 units per request
        units = min(units, 4)

        # Per-episode cap: max 4 units total per hospital
        total_released = self._emergency_released.get(hosp_id, 0)
        if total_released >= 4:
            return {
                "success": False,
                "reason": f"Emergency reserves exhausted at {hosp_id} (max 4 units/episode)"
            }

        available = bank.stocks.get("O_NEG", 0)
        if available < units:
            return {
                "success": False,
                "reason": f"Insufficient O_NEG at {hosp_id}: have {available}, need {units}"
            }

        bank.stocks["O_NEG"] -= units
        self._emergency_released[hosp_id] = total_released + units
        return {"success": True, "blood_type": "O_NEG", "units": units}

    def start_crossmatch(
        self,
        hosp_id: str,
        patient_id: str,
        blood_type: str,
        units: int,
    ) -> Dict:
        """Add to crossmatch queue (15-minute delay)."""
        bank = self._banks.get(hosp_id)
        if bank is None:
            return {"success": False, "reason": f"Hospital {hosp_id} not found"}
        if bank.stocks.get(blood_type, 0) < units:
            return {
                "success": False,
                "reason": f"Insufficient {blood_type} at {hosp_id}: have {bank.stocks.get(blood_type, 0)}, need {units}"
            }
        bank.crossmatch_queue.append({
            "patient_id": patient_id,
            "blood_type": blood_type,
            "units": units,
            "time_remaining": 15,
        })
        return {"success": True}

    def flush_completed_crossmatches(self) -> List[Dict]:
        """Drain and return completed crossmatches from the last tick(). Idempotent."""
        completed = self._pending_completed
        self._pending_completed = []
        return completed

    def transfer(
        self,
        from_hosp: str,
        to_hosp: str,
        blood_type: str,
        units: int,
    ) -> Dict:
        """Transfer blood between hospitals. Instant (simplification)."""
        bank_from = self._banks.get(from_hosp)
        bank_to = self._banks.get(to_hosp)
        if bank_from is None or bank_to is None:
            return {"success": False, "reason": "Hospital not found"}
        if bank_from.stocks.get(blood_type, 0) < units:
            return {
                "success": False,
                "reason": f"Insufficient {blood_type} at {from_hosp}"
            }
        bank_from.stocks[blood_type] -= units
        bank_to.stocks[blood_type] = bank_to.stocks.get(blood_type, 0) + units
        return {"success": True}

    def tick(self) -> None:
        """Advance crossmatch timers and collect newly-completed entries in one pass."""
        self._pending_completed = []
        for bank in self._banks.values():
            still_pending = []
            for entry in bank.crossmatch_queue:
                entry["time_remaining"] -= 1
                if entry["time_remaining"] <= 0:
                    bank.stocks[entry["blood_type"]] -= entry["units"]
                    self._pending_completed.append(dict(entry))
                else:
                    still_pending.append(entry)
            bank.crossmatch_queue = still_pending
