"""Ambulance fleet manager subsystem."""

from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class Ambulance:
    id: str
    equipment: Literal["ALS", "BLS"]
    base_node: str
    status: Literal["available", "en_route", "on_scene", "returning", "off_duty"] = "available"
    target_node: Optional[str] = None
    eta_minutes: int = 0
    route: list[str] = field(default_factory=list)
    patient_id: Optional[str] = None
    scene_minutes_remaining: int = 0  # countdown while on-scene (Task 16)
    arrived_with_patient: bool = False  # signals environment to call _do_treatment_arrival


class AmbulanceManager:
    """
    Manages all ambulance state and movement.

    Responsibilities:
    - Dispatch: set route from base to target via RoadNetwork
    - Move: decrement ETA each tick
    - Arrive: transition status when ETA reaches 0
    - Query: available ambulances by equipment type
    """

    def __init__(self, ambulance_defs: list[dict]):
        self._ambulances: dict[str, Ambulance] = {
            d["id"]: Ambulance(id=d["id"], equipment=d["equipment"], base_node=d["base_node"])
            for d in ambulance_defs
        }

    def dispatch(
        self,
        ambulance_id: str,
        target_node: str,
        road_network,  # RoadNetwork instance for routing
        patient_id: Optional[str] = None,
    ) -> dict:
        """
        Dispatch an ambulance to a target node.
        Computes shortest path via road_network.shortest_path().
        On arrival, the environment sets amb.patient_id and starts the scene
        time countdown via tick().
        Returns {"success": bool, "reason": str | None}
        """
        from .constants import SCENE_TIME

        amb = self._ambulances.get(ambulance_id)
        if amb is None:
            return {"success": False, "reason": "ambulance not found"}
        if amb.status != "available":
            return {"success": False, "reason": "ambulance not available"}

        if road_network is not None:
            route = road_network.shortest_path(amb.base_node, target_node)
            eta = road_network.route_travel_time(route)
            if patient_id is not None:
                eta += SCENE_TIME  # Signal: ambulance will have patient on board at arrival
        else:
            route = []
            eta = 0

        amb.status = "en_route"
        amb.target_node = target_node
        amb.route = route
        amb.eta_minutes = eta
        if patient_id is not None:
            amb.patient_id = patient_id
        return {"success": True}

    def arrive(self, ambulance_id: str) -> None:
        """Mark ambulance as arrived at destination (called by environment after ETA reaches 0)."""
        amb = self._ambulances.get(ambulance_id)
        if amb:
            amb.status = "on_scene"
            amb.eta_minutes = 0

    def return_to_base(self, ambulance_id: str, road_network) -> None:
        """Begin return journey to base."""
        amb = self._ambulances.get(ambulance_id)
        if amb and road_network:
            route = road_network.shortest_path(amb.target_node, amb.base_node)
            eta = road_network.route_travel_time(route)
            amb.status = "returning"
            amb.route = route
            amb.eta_minutes = eta
            amb.target_node = amb.base_node

    def mark_available(self, ambulance_id: str) -> None:
        amb = self._ambulances.get(ambulance_id)
        if amb:
            amb.status = "available"
            amb.target_node = None
            amb.route = []
            amb.eta_minutes = 0
            amb.patient_id = None
            amb.scene_minutes_remaining = 0
            amb.arrived_with_patient = False

    def tick(self) -> None:
        """Advance all ambulances by 1 minute. Auto-return when scene time expires."""
        from .constants import SCENE_TIME

        for amb in self._ambulances.values():
            if amb.status == "on_scene" and amb.scene_minutes_remaining > 0:
                # Task 16: countdown on-scene time
                amb.scene_minutes_remaining -= 1
                if amb.scene_minutes_remaining == 0:
                    # Scene time expired — signal environment to deliver patient
                    if amb.patient_id is not None:
                        amb.arrived_with_patient = True
                    amb.status = "returning"
                    amb.route = []
                    amb.target_node = amb.base_node
            elif amb.status in ("en_route", "returning") and amb.eta_minutes > 0:
                amb.eta_minutes -= 1
                if amb.eta_minutes == 0:
                    prev_status = amb.status
                    amb.status = "on_scene"
                    # Task 16: if ambulance arrived with a patient, start scene countdown
                    if amb.patient_id is not None and prev_status == "en_route":
                        amb.scene_minutes_remaining = SCENE_TIME

    def get_available(self, equipment: Optional[str] = None) -> list[str]:
        ids = [
            amb_id for amb_id, amb in self._ambulances.items()
            if amb.status == "available"
        ]
        if equipment:
            ids = [aid for aid in ids if self._ambulances[aid].equipment == equipment]
        return ids

    def get(self, ambulance_id: str) -> Optional[Ambulance]:
        return self._ambulances.get(ambulance_id)

    def all(self) -> dict[str, Ambulance]:
        return self._ambulances.copy()
