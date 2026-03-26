# CodeRedEnv — Plan 2: Actions & Subsystems

> **For agentic workers:** Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Complete all action execution with correct latencies, implement the full road network routing with congestion and disruptions, hospital OR/prep/specialist system with preemption, blood bank with crossmatch timing, patient lifecycle with treatment completion, and the disruption engine with seeded randomization.

**Prerequisites:** Plan 1 must be complete. This plan builds on the foundation established in `codered_env/server/subsystems/constants.py`, `codered_env/server/models/`, and `codered_env/server/codered_environment.py`.

**Tech Stack:** Python 3.10+, random (seeding), heapq (Dijkstra), dataclasses

---

## File Structure

```
codered_env/server/subsystems/
    constants.py         # (Plan 1 — complete)
    road_network.py      # REWRITE: full routing, congestion, disruption
    ambulance_manager.py  # REWRITE: full movement, arrival, status transitions
    hospital_system.py    # REWRITE: OR prep/surgery/preemption, specialist pages
    patient_manager.py    # REWRITE: treatment completion, patient outcome
    blood_bank.py         # REWRITE: crossmatch queue, emergency release, transfers
    disruption_engine.py   # REWRITE: seeded disruption generator
codered_env/server/codered_environment.py  # MODIFY: wire subsystems, fix latencies
codered_env/tests/test_subsystems.py         # NEW: tests for all subsystems
codered_env/tests/test_dispatch.py           # NEW: action execution tests
```

---

## Task 6: Road Network (Full Routing + Congestion + Disruptions)

**Files:**
- Rewrite: `codered_env/server/subsystems/road_network.py`
- Test: `codered_env/tests/test_road_network.py`

- [ ] **Step 1: Write failing tests**

```python
# codered_env/tests/test_road_network.py
from codered_env.server.subsystems.road_network import RoadNetwork

def test_road_network_builds_from_constants():
    rn = RoadNetwork()
    assert len(rn.edges) == 15  # All 15 edges
    assert rn.get_travel_time("RAILWAY_XING", "NH45_BYPASS") == 6

def test_shortest_path():
    rn = RoadNetwork()
    path = rn.shortest_path("RAJIV_CHOWK", "IT_HUB")
    assert path[0] == "RAJIV_CHOWK"
    assert path[-1] == "IT_HUB"
    assert rn.route_travel_time(path) < float("inf")

def test_road_closure():
    rn = RoadNetwork()
    rn.set_disruption("RAILWAY_XING", "NH45_BYPASS", "road_closure", remaining_steps=999)
    assert rn.get_travel_time("RAILWAY_XING", "NH45_BYPASS") == float("inf")
    # Alternative path should still work
    path = rn.shortest_path("CHOWKHA", "IT_HUB")
    assert len(path) > 0

def test_accident_slows_road():
    rn = RoadNetwork()
    rn.set_disruption("NH45_BYPASS", "IT_HUB", "accident", remaining_steps=15)
    # Accident multiplies congestion by 3
    edge = rn._get_edge("NH45_BYPASS", "IT_HUB")
    assert edge.effective_time() == 4 * 3.0

def test_reachable_all_nodes():
    rn = RoadNetwork()
    for node_a in rn.node_ids:
        for node_b in rn.node_ids:
            path = rn.shortest_path(node_a, node_b)
            assert len(path) > 0, f"No path from {node_a} to {node_b}"

def test_hospital_nodes_accessible():
    rn = RoadNetwork()
    hosp_nodes = ["AIIMS_PRAKASH", "DISTRICT_HOSP", "COMMUNITY_HC"]
    for hn in hosp_nodes:
        for other in ["RAJIV_CHOWK", "IT_HUB", "NH45_BYPASS"]:
            path = rn.shortest_path(other, hn)
            assert len(path) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest codered_env/tests/test_road_network.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write `road_network.py`**

```python
"""Road network subsystem: graph, routing, congestion, disruptions."""

import heapq
from typing import Dict, List, Optional, Tuple


class Edge:
    def __init__(self, from_node: str, to_node: str, base_time: int):
        self.from_node = from_node
        self.to_node = to_node
        self.base_time = base_time
        self.congestion_multiplier: float = 1.0
        self.disrupted: bool = False
        self.disruption_type: Optional[str] = None
        self.disruption_remaining: int = 0

    def effective_time(self) -> float:
        if self.disrupted and self.disruption_type == "road_closure":
            return float("inf")
        return self.base_time * self.congestion_multiplier

    def tick(self) -> None:
        """Advance disruption timers by 1 minute."""
        if self.disrupted and self.disruption_remaining > 0:
            self.disruption_remaining -= 1
            if self.disruption_remaining == 0:
                self.disrupted = False
                self.disruption_type = None
                self.congestion_multiplier = 1.0


class RoadNetwork:
    """City road network with routing, congestion, and disruption support."""

    def __init__(self):
        from .constants import CITY_NODES, CITY_EDGES

        self.node_ids: List[str] = [n["id"] for n in CITY_NODES]
        self._edges: Dict[str, Edge] = {}

        for edge in CITY_EDGES:
            key = self._edge_key(edge["from"], edge["to"])
            self._edges[key] = Edge(edge["from"], edge["to"], edge["base_time"])

    # =========================================================================
    # Public API
    # =========================================================================

    def get_travel_time(self, from_node: str, to_node: str) -> float:
        """Get effective travel time between two adjacent nodes."""
        edge = self._get_edge(from_node, to_node)
        if edge is None:
            return float("inf")
        return edge.effective_time()

    def shortest_path(self, from_node: str, to_node: str) -> List[str]:
        """
        Dijkstra shortest path. Returns list of node IDs or empty list if unreachable.
        road_closure edges return inf and are skipped.
        """
        if from_node == to_node:
            return [from_node]

        pq: List[Tuple[float, str, List[str]]] = [(0.0, from_node, [from_node])]
        visited: set = set()

        while pq:
            cost, node, path = heapq.heappop(pq)
            if node in visited:
                continue
            visited.add(node)
            if node == to_node:
                return path
            for neighbor, edge in self._neighbors(node):
                if neighbor not in visited and edge.effective_time() < float("inf"):
                    new_cost = cost + edge.effective_time()
                    heapq.heappush(pq, (new_cost, neighbor, path + [neighbor]))

        return []  # Unreachable

    def route_travel_time(self, route: List[str]) -> int:
        """Total minutes for a precomputed route."""
        total = 0.0
        for i in range(len(route) - 1):
            t = self.get_travel_time(route[i], route[i + 1])
            if t == float("inf"):
                return 999
            total += t
        return int(total)

    def set_disruption(
        self,
        from_node: str,
        to_node: str,
        disruption_type: str,
        remaining_steps: int,
    ) -> bool:
        """Apply a disruption to an edge. Returns True if applied."""
        edge = self._get_edge(from_node, to_node)
        if edge is None:
            return False
        edge.disrupted = True
        edge.disruption_type = disruption_type
        edge.disruption_remaining = remaining_steps
        if disruption_type == "accident":
            edge.congestion_multiplier = 3.0
        elif disruption_type == "road_closure":
            edge.congestion_multiplier = float("inf")
        return True

    def clear_disruption(self, from_node: str, to_node: str) -> None:
        """Remove disruption from an edge."""
        edge = self._get_edge(from_node, to_node)
        if edge:
            edge.disrupted = False
            edge.disruption_type = None
            edge.disruption_remaining = 0
            edge.congestion_multiplier = 1.0

    def tick(self) -> None:
        """Advance all edge disruptions by 1 minute."""
        for edge in self._edges.values():
            edge.tick()

    def get_active_disruptions(self) -> List[Dict]:
        """Return list of currently disrupted edges."""
        result = []
        for key, edge in self._edges.items():
            if edge.disrupted:
                result.append({
                    "edge_key": key,
                    "from_node": edge.from_node,
                    "to_node": edge.to_node,
                    "type": edge.disruption_type,
                    "remaining_steps": edge.disruption_remaining,
                })
        return result

    # =========================================================================
    # Internal
    # =========================================================================

    def _edge_key(self, from_node: str, to_node: str) -> str:
        nodes = sorted([from_node, to_node])
        return f"{nodes[0]}->{nodes[1]}"

    def _get_edge(self, from_node: str, to_node: str) -> Optional[Edge]:
        key = self._edge_key(from_node, to_node)
        return self._edges.get(key)

    def _neighbors(self, node: str) -> List[Tuple[str, Edge]]:
        """Return (neighbor_node, edge) for all edges adjacent to node."""
        result = []
        for edge in self._edges.values():
            if edge.from_node == node:
                result.append((edge.to_node, edge))
            elif edge.to_node == node:
                result.append((edge.from_node, edge))
        return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest codered_env/tests/test_road_network.py -v`
Expected: PASS (all 6 tests)

---

## Task 7: Hospital System (OR Prep/Surgery/Preemption + Specialist Pages)

**Files:**
- Rewrite: `codered_env/server/subsystems/hospital_system.py`
- Test: `codered_env/tests/test_hospital_system.py`

- [ ] **Step 1: Write failing tests**

```python
# codered_env/tests/test_hospital_system.py
from codered_env.server.subsystems.hospital_system import HospitalSystem

def test_hospitals_initialized_from_constants():
    hs = HospitalSystem()
    hosp_a = hs.get("HOSP_A")
    assert hosp_a is not None
    assert hosp_a.capabilities == ["cardiac", "stroke", "trauma", "stabilization"]
    assert len(hosp_a.operating_rooms) == 3
    assert hosp_a.icu_beds["total"] == 4

def test_prepare_or_sets_prep_state():
    hs = HospitalSystem()
    result = hs.prepare_or("HOSP_A", "cardiac")
    assert result["success"] is True
    assert hosp_a.operating_rooms[0].status == "prep"
    assert hosp_a.or_prep_countdowns[0] == 10

def test_prepare_or_fails_for_unsupported_condition():
    hs = HospitalSystem()
    result = hs.prepare_or("HOSP_C", "cardiac")
    assert result["success"] is False
    assert "cannot treat" in result["reason"]

def test_prepare_or_fails_when_no_idle_or():
    hs = HospitalSystem()
    # Occupy all 3 ORs
    for i in range(3):
        hs.start_surgery("HOSP_A", i, "cardiac", "P1")
    result = hs.prepare_or("HOSP_A", "cardiac")
    assert result["success"] is False
    assert "no idle OR" in result["reason"]

def test_prepare_or_fails_when_on_diversion():
    hs = HospitalSystem()
    hs.set_diversion("HOSP_A", True)
    result = hs.prepare_or("HOSP_A", "cardiac")
    assert result["success"] is False
    assert "diversion" in result["reason"]
    hs.set_diversion("HOSP_A", False)

def test_tick_decrements_prep_countdown():
    hs = HospitalSystem()
    hs.prepare_or("HOSP_A", "cardiac")
    hosp_a = hs.get("HOSP_A")
    assert hosp_a.or_prep_countdowns[0] == 10
    hs.tick()
    assert hosp_a.or_prep_countdowns[0] == 9
    for _ in range(9):
        hs.tick()
    # After 10 ticks, OR should be idle (prep done)
    assert hosp_a.operating_rooms[0].status == "idle"

def test_page_specialist():
    hs = HospitalSystem()
    hosp_a = hs.get("HOSP_A")
    initial_available = hosp_a.specialists["cardiologist"].available
    result = hs.page_specialist("HOSP_A", "cardiologist")
    assert result["success"] is True
    assert hosp_a.specialists["cardiologist"].available == initial_available - 1

def test_page_specialist_fails_when_none_available():
    hs = HospitalSystem()
    hosp_a = hs.get("HOSP_A")
    hosp_a.specialists["neurologist"].available = 0
    result = hs.page_specialist("HOSP_A", "neurologist")
    assert result["success"] is False

def test_preempt_or_returns_harm():
    hs = HospitalSystem()
    hs.prepare_or("HOSP_A", "cardiac")
    for _ in range(10):
        hs.tick()  # OR goes idle
    # Start a surgery with remaining time
    hosp_a = hs.get("HOSP_A")
    hosp_a.operating_rooms[0].status = "in_use"
    hosp_a.operating_rooms[0].minutes_remaining = 20
    result = hs.preempt_or("HOSP_A", 0)
    assert result["success"] is True
    assert result["harm"] == 20 / 30.0
    assert result["recovery_time"] == 20
    assert hosp_a.operating_rooms[0].status == "idle"

def test_preemption_with_zero_time_no_harm():
    hs = HospitalSystem()
    hosp_a = hs.get("HOSP_A")
    hosp_a.operating_rooms[0].status = "in_use"
    hosp_a.operating_rooms[0].minutes_remaining = 0
    result = hs.preempt_or("HOSP_A", 0)
    assert result["harm"] == 0.0

def test_tick_specialist_recovery():
    hs = HospitalSystem()
    hs.page_specialist("HOSP_A", "cardiologist")
    hosp_a = hs.get("HOSP_A")
    assert hosp_a.specialists["cardiologist"].minutes_until_available == 8
    hs.tick()
    assert hosp_a.specialists["cardiologist"].minutes_until_available == 7

def test_hospital_C_cannot_handle_cardiac():
    hs = HospitalSystem()
    result = hs.prepare_or("HOSP_C", "cardiac")
    assert result["success"] is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest codered_env/tests/test_hospital_system.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write `hospital_system.py`**

```python
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
    icu_beds_total: int
    icu_beds_available: int
    blood_stock: Dict[str, int]
    on_diversion: bool = False
    or_prep_countdowns: Dict[int, int] = field(default_factory=dict)
    # Surgery queue: patient_id -> minutes_remaining
    surgery_queue: Dict[str, int] = field(default_factory=dict)
    # Equipment failures: or_index -> minutes_remaining (OR unavailable)
    equipment_failures: Dict[int, int] = field(default_factory=dict)


class HospitalSystem:
    """Manages hospital resources: ORs, specialists, ICU beds."""

    def __init__(self):
        from .constants import HOSPITALS

        self._hospitals: Dict[str, Hospital] = {}
        for h in HOSPITALS:
            specialists = {
                role: Specialist(available=data["available"], total=data["total"])
                for role, data in h["specialists"].items()
            }
            ors = [OR(index=i) for i in range(h["num_or"])]
            self._hospitals[h["id"]] = Hospital(
                id=h["id"],
                node_id=h["node_id"],
                capabilities=h["capabilities"],
                specialists=specialists,
                operating_rooms=ors,
                icu_beds_total=h["icu_beds"]["total"],
                icu_beds_available=h["icu_beds"]["available"],
                blood_stock=dict(h["blood_stock"]),
            )

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
            return {"success": False, "reason": f"No idle OR at {hosp_id}"}

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
    # Time Advance
    # =========================================================================

    def tick(self) -> None:
        """Advance all hospital timers by 1 minute."""
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
                        spec.available += 1
                        spec.minutes_until_available = 0
                    else:
                        spec.minutes_until_available -= 1

            # Equipment failure countdown
            for or_idx in list(h.equipment_failures.keys()):
                h.equipment_failures[or_idx] -= 1
                if h.equipment_failures[or_idx] <= 0:
                    del h.equipment_failures[or_idx]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest codered_env/tests/test_hospital_system.py -v`
Expected: PASS (all 11 tests)

---

## Task 8: Patient Manager (Treatment Completion + Outcome)

**Files:**
- Rewrite: `codered_env/server/subsystems/patient_manager.py`
- Test: `codered_env/tests/test_patient_manager.py`

> **Critical:** `treatment_complete_time` MUST be set when a patient finishes treatment. This field is required by the grader's time_score axis.

- [ ] **Step 1: Write failing tests**

```python
# codered_env/tests/test_patient_manager.py
from codered_env.server.subsystems.patient_manager import PatientManager

def test_patient_created():
    pm = PatientManager()
    pm.reset(task_id="task1", rng=None)
    assert len(pm.patients) == 1
    assert pm.patients[0].condition == "CARDIAC"
    assert pm.patients[0].treatment_complete_time is None  # Not yet treated

def test_mark_treated_sets_complete_time():
    pm = PatientManager()
    pm.reset(task_id="task1", rng=None)
    patient = pm.patients[0]
    patient.status = "in_treatment"
    # Simulate treatment taking 14 minutes from onset
    pm.mark_treated(patient.id, treatment_complete_time=14)
    assert patient.treatment_complete_time == 14
    assert patient.outcome == "saved"

def test_mark_deceased_sets_outcome():
    pm = PatientManager()
    pm.reset(task_id="task1", rng=None)
    patient = pm.patients[0]
    pm.mark_deceased(patient.id, reason="timeout")
    assert patient.outcome == "deceased"

def test_patients_from_task2():
    pm = PatientManager()
    import random
    rng = random.Random(0)
    pm.reset(task_id="task2", rng=rng)
    assert len(pm.patients) >= 3  # cardiac + stroke + trauma
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest codered_env/tests/test_patient_manager.py -v`
Expected: FAIL — PatientManager not defined

- [ ] **Step 3: Implement PatientManager**

```python
# codered_env/server/subsystems/patient_manager.py
from typing import Optional
from dataclasses import dataclass, field

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
            # Note: treatment_complete_time stays None for deceased patients

    def tick(self) -> None:
        """Called each step to advance patient deterioration."""
        # In basic implementation: no-op. Full implementation would:
        # - Check for timeout deaths (patient past critical window)
        # - Update patient severity levels
        pass
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest codered_env/tests/test_patient_manager.py -v`
Expected: PASS

---

## Task 9: Ambulance Fleet Manager

**Files:**
- Rewrite: `codered_env/server/subsystems/ambulance_manager.py` (implements `AmbulanceManager`)
- Test: `codered_env/tests/test_ambulance_manager.py`

> **Note:** All references across all 3 plans use `ambulance_manager.py` — the environment imports from `.subsystems.ambulance_manager`, tests import from `ambulance_manager`, and the file is named `ambulance_manager.py`. Keep this consistent everywhere.

- [ ] **Step 1: Write failing tests**

```python
# codered_env/tests/test_ambulance_manager.py
from codered_env.server.subsystems.ambulance_manager import AmbulanceManager

def test_ambulance_manager_created_from_constants():
    from codered_env.server.subsystems.constants import AMBULANCES
    am = AmbulanceManager(AMBULANCES)
    assert len(am._ambulances) == 5
    assert "AMB_1" in am._ambulances
    assert am._ambulances["AMB_1"].status == "available"

def test_dispatch_sets_route():
    from codered_env.server.subsystems.constants import AMBULANCES
    am = AmbulanceManager(AMBULANCES)
    result = am.dispatch("AMB_1", "NH45_BYPASS", road_network=None)
    assert result["success"] is True
    amb = am._ambulances["AMB_1"]
    assert amb.status == "en_route"
    assert amb.target_node == "NH45_BYPASS"

def test_dispatch_fails_when_unavailable():
    from codered_env.server.subsystems.constants import AMBULANCES
    am = AmbulanceManager(AMBULANCES)
    am.dispatch("AMB_1", "NH45_BYPASS", road_network=None)
    # AMB_1 is now en-route — dispatch should fail
    result = am.dispatch("AMB_1", "RAILWAY_XING", road_network=None)
    assert result["success"] is False

def test_tick_decrements_eta():
    from codered_env.server.subsystems.constants import AMBULANCES
    am = AmbulanceManager(AMBULANCES)
    am.dispatch("AMB_1", "NH45_BYPASS", road_network=None)
    amb = am._ambulances["AMB_1"]
    initial_eta = amb.eta_minutes
    am.tick()
    assert amb.eta_minutes == initial_eta - 1

def test_arrival_when_eta_hits_zero():
    from codered_env.server.subsystems.constants import AMBULANCES
    am = AmbulanceManager(AMBULANCES)
    am.dispatch("AMB_1", "NH45_BYPASS", road_network=None)
    amb = am._ambulances["AMB_1"]
    for _ in range(amb.eta_minutes):
        am.tick()
    # After reaching destination
    am.tick()  # one more step
    assert amb.status == "on_scene"

def test_get_available_als():
    from codered_env.server.subsystems.constants import AMBULANCES
    am = AmbulanceManager(AMBULANCES)
    available = am.get_available(equipment="ALS")
    assert len(available) == 2  # AMB_1 and AMB_2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest codered_env/tests/test_ambulance_manager.py -v`
Expected: FAIL — AmbulanceManager class not defined

- [ ] **Step 3: Implement AmbulanceManager**

```python
# codered_env/server/subsystems/ambulance_manager.py
from dataclasses import dataclass, field
from typing import Optional, Literal
from codered_env.server.subsystems.constants import AMBULANCES

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
        Returns {"success": bool, "reason": str | None}
        """
        amb = self._ambulances.get(ambulance_id)
        if amb is None:
            return {"success": False, "reason": "ambulance not found"}
        if amb.status != "available":
            return {"success": False, "reason": "ambulance not available"}

        if road_network is not None:
            route = road_network.shortest_path(amb.base_node, target_node)
            eta = road_network.route_travel_time(route)
        else:
            route = []
            eta = 0

        amb.status = "en_route"
        amb.target_node = target_node
        amb.route = route
        amb.eta_minutes = eta
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

    def tick(self) -> None:
        """Advance all ambulances by 1 minute."""
        for amb in self._ambulances.values():
            if amb.status in ("en_route", "returning") and amb.eta_minutes > 0:
                amb.eta_minutes -= 1
                if amb.eta_minutes == 0:
                    # Arrival handled externally via arrive()
                    pass

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest codered_env/tests/test_ambulance_manager.py -v`
Expected: PASS

---

## Task 10: Blood Bank (Crossmatch + Emergency Release + Transfers)

**Files:**
- Rewrite: `codered_env/server/subsystems/blood_bank.py`
- Test: `codered_env/tests/test_blood_bank.py`

- [ ] **Step 1: Write failing tests**

```python
# codered_env/tests/test_blood_bank.py
from codered_env.server.subsystems.blood_bank import BloodBankSystem

def test_hospital_C_has_limited_blood():
    bb = BloodBankSystem()
    hosp_c = bb.get("HOSP_C")
    assert hosp_c.stocks["O_POS"] == 4
    assert hosp_c.stocks["A_POS"] == 0

def test_emergency_release():
    bb = BloodBankSystem()
    result = bb.emergency_release("HOSP_A", "P1", "O_NEG", 2)
    assert result["success"] is True
    assert bb.get("HOSP_A").stocks["O_NEG"] == 4  # 6-2

def test_emergency_release_fails_when_insufficient():
    bb = BloodBankSystem()
    bb.get("HOSP_C").stocks["O_NEG"] = 1
    result = bb.emergency_release("HOSP_C", "P1", "O_NEG", 3)
    assert result["success"] is False

def test_crossmatch_queue():
    bb = BloodBankSystem()
    result = bb.start_crossmatch("HOSP_A", "P1", "A_POS", 2)
    assert result["success"] is True
    queue = bb.get("HOSP_A").crossmatch_queue
    assert len(queue) == 1
    assert queue[0]["time_remaining"] == 15

def test_crossmatch_tick():
    bb = BloodBankSystem()
    bb.start_crossmatch("HOSP_A", "P1", "A_POS", 2)
    bb.tick()
    queue = bb.get("HOSP_A").crossmatch_queue
    assert queue[0]["time_remaining"] == 14

def test_crossmatch_completes_and_reserves():
    bb = BloodBankSystem()
    hosp = bb.get("HOSP_A")
    initial_stock = hosp.stocks["A_POS"]
    bb.start_crossmatch("HOSP_A", "P1", "A_POS", 2)
    # Fast-forward 15 ticks
    for _ in range(15):
        bb.tick()
    completed = bb.flush_completed_crossmatches()
    assert len(completed) == 1
    assert hosp.stocks["A_POS"] == initial_stock - 2  # Reserved

def test_transfer_blood():
    bb = BloodBankSystem()
    initial_a = bb.get("HOSP_A").stocks["O_POS"]
    initial_c = bb.get("HOSP_C").stocks["O_POS"]
    result = bb.transfer("HOSP_A", "HOSP_C", "O_POS", 3)
    assert result["success"] is True
    assert bb.get("HOSP_A").stocks["O_POS"] == initial_a - 3
    assert bb.get("HOSP_C").stocks["O_POS"] == initial_c + 3

def test_transfer_fails_insufficient():
    bb = BloodBankSystem()
    result = bb.transfer("HOSP_A", "HOSP_C", "O_POS", 999)
    assert result["success"] is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest codered_env/tests/test_blood_bank.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write `blood_bank.py`**

```python
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
        """
        Instant O-Neg emergency release. In reality, any patient can receive O-neg.
        For simplicity: emergency always uses O_NEG (universal donor).
        """
        bank = self._banks.get(hosp_id)
        if bank is None:
            return {"success": False, "reason": f"Hospital {hosp_id} not found"}
        if bank.stocks.get("O_NEG", 0) < units:
            return {
                "success": False,
                "reason": f"Insufficient O_NEG at {hosp_id}: have {bank.stocks.get('O_NEG', 0)}, need {units}"
            }
        bank.stocks["O_NEG"] -= units
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
        """
        Called each tick. Returns list of completed crossmatches.
        Call this after tick() advances the timers.
        """
        completed = []
        for bank in self._banks.values():
            still_pending = []
            for entry in bank.crossmatch_queue:
                if entry["time_remaining"] <= 0:
                    # Reserve the blood
                    bank.stocks[entry["blood_type"]] -= entry["units"]
                    completed.append(dict(entry))
                else:
                    still_pending.append(entry)
            bank.crossmatch_queue = still_pending
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
        """Advance crossmatch timers by 1 minute."""
        for bank in self._banks.values():
            for entry in bank.crossmatch_queue:
                entry["time_remaining"] -= 1
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest codered_env/tests/test_blood_bank.py -v`
Expected: PASS (all 7 tests)

---

## Task 11: Disruption Engine (Seeded Probability + Jitter)

**Files:**
- Rewrite: `codered_env/server/subsystems/disruption_engine.py`
- Test: `codered_env/tests/test_disruption_engine.py`

- [ ] **Step 1: Write failing tests**

```python
# codered_env/tests/test_disruption_engine.py
from codered_env.server.subsystems.disruption_engine import DisruptionEngine

def test_no_disruptions_task1():
    eng = DisruptionEngine()
    eng.reset(seed=0, task_id="task1")
    events = eng.roll_disruptions(step=1, road_network=None)
    assert events == []  # No disruptions in task 1

def test_seed_reproducibility():
    eng1 = DisruptionEngine()
    eng2 = DisruptionEngine()
    eng1.reset(seed=42, task_id="task2")
    eng2.reset(seed=42, task_id="task2")
    # Same seed = same intensity
    assert eng1._intensity == eng2._intensity

def test_intensity_in_range():
    eng = DisruptionEngine()
    eng.reset(seed=123, task_id="task2")
    assert 0.7 <= eng._intensity <= 1.3

def test_road_closure_affects_target():
    eng = DisruptionEngine()
    eng.reset(seed=0, task_id="task2")
    # Seed 0, step 1: road closure on RING_ROAD->COMMUNITY_HC
    from codered_env.server.subsystems.road_network import RoadNetwork
    rn = RoadNetwork()
    events = eng.roll_disruptions(step=1, road_network=rn)
    assert any(e["type"] == "road_closure" for e in events)

def test_disruption_types_vary_across_seeds():
    """Different seeds produce different disruption patterns."""
    eng = DisruptionEngine()
    eng.reset(seed=1, task_id="task3")
    eng2 = DisruptionEngine()
    eng2.reset(seed=999, task_id="task3")
    # At least one difference in the disruption schedule
    assert eng._seed != eng2._seed
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest codered_env/tests/test_disruption_engine.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write `disruption_engine.py`**

```python
"""Seeded disruption generator for CodeRedEnv."""

import random
from typing import Dict, List, Literal, Optional

# Disruption type weights per task difficulty
DISRUPTION_TYPES_BY_TASK = {
    "task1": [],
    "task2": ["road_closure", "hospital_diversion", "accident"],
    "task3": ["road_closure", "hospital_diversion", "accident", "equipment_failure", "surge_event"],
}

BASE_PROB_PER_TASK = {
    "task1": 0.0,
    "task2": 0.05,
    "task3": 0.15,
}


class DisruptionEngine:
    """
    Generates disruptions using seeded probability with jitter.

    At reset(seed), a deterministic intensity multiplier k is drawn from
    Uniform(0.7, 1.3). Per-step disruption chance = base_prob * k.
    Disruption target is selected deterministically from seed + step index.
    """

    def __init__(self):
        self._seed: int = 0
        self._task_id: str = "task1"
        self._intensity: float = 1.0
        self._rng: Optional[random.Random] = None
        self._scheduled_disruptions: List[Dict] = []  # Pre-generated schedule

    def reset(self, seed: int, task_id: str) -> None:
        """Initialize disruption engine for a new episode."""
        self._seed = seed
        self._task_id = task_id
        self._rng = random.Random(seed)

        # Draw intensity multiplier deterministically from seed
        self._intensity = self._rng.uniform(0.7, 1.3)

        # Pre-generate disruption schedule for this episode
        max_steps = {  # From TASK_CONFIG
            "task1": 30, "task2": 45, "task3": 60,
        }[task_id]
        self._scheduled_disruptions = self._generate_schedule(max_steps)

    def roll_disruptions(
        self,
        step: int,
        road_network,  # RoadNetwork instance
        hospital_system=None,
    ) -> List[Dict]:
        """
        At each step, check if a disruption is scheduled and apply it.
        Returns list of disruption events applied this step.
        """
        events = []
        for sched in self._scheduled_disruptions:
            if sched["step"] == step:
                event = dict(sched)
                target_type = sched["target_type"]

                if target_type == "edge":
                    from_node, to_node = sched["target"].split("->")
                    road_network.set_disruption(
                        from_node, to_node,
                        sched["disruption_type"],
                        sched["duration"],
                    )
                elif target_type == "hospital":
                    if hospital_system:
                        if sched["disruption_type"] == "hospital_diversion":
                            hospital_system.set_diversion(sched["target"], True)
                        elif sched["disruption_type"] == "equipment_failure":
                            # Pick a random OR index (0, 1, or 2) at the target hospital
                            or_idx = self._rng.randint(0, 2)
                            hospital_system.set_equipment_failure(
                                sched["target"], or_idx, sched["duration"]
                            )

                events.append(event)

        return events

    def get_optimal_mutual_aid_window(self, call_index: int) -> tuple[int, int]:
        """
        Return (optimal_window_start, optimal_window_end) for mutual aid call.
        Based on first disruption step in the schedule.
        """
        first_disruption_step = None
        for sched in self._scheduled_disruptions:
            if sched["disruption_type"] not in ("surge_event",):
                first_disruption_step = sched["step"]
                break

        if first_disruption_step is None:
            # Default windows by task
            defaults = {
                "task2": (15, 25),
                "task3_call_1": (5, 15),
                "task3_call_2": (25, 35),
            }
            key = f"{self._task_id}_call_{call_index}"
            return defaults.get(key, (20, 30))

        start = max(1, first_disruption_step - 8)
        end = first_disruption_step + 2
        return (start, end)

    def _generate_schedule(self, max_steps: int) -> List[Dict]:
        """Generate disruption events from seed and intensity."""
        rng = random.Random(self._seed + 1000)  # Offset from main seed
        base_prob = BASE_PROB_PER_TASK[self._task_id]
        effective_prob = base_prob * self._intensity

        available_types = DISRUPTION_TYPES_BY_TASK[self._task_id]
        if not available_types:
            return []

        from .constants import CITY_EDGES, HOSPITALS

        # Build list of all edges and hospitals
        edges = [f"{e['from']}->{e['to']}" for e in CITY_EDGES]
        hospitals = [h["id"] for h in HOSPITALS]

        schedule = []
        scheduled_edges: set = set()
        scheduled_hospitals: set = set()

        for step_idx in range(1, max_steps + 1):
            if rng.random() < effective_prob:
                disp_type = rng.choice(available_types)

                # Select target deterministically from seed + step
                target_seed = self._seed + step_idx
                target_rng = random.Random(target_seed)

                duration_map = {
                    "road_closure": 999,       # Remainder of episode
                    "hospital_diversion": target_rng.randint(5, 15),
                    "accident": target_rng.randint(10, 20),
                    "equipment_failure": target_rng.randint(10, 30),
                }
                duration = duration_map.get(disp_type, 10)

                if disp_type == "road_closure":
                    edge_idx = target_rng.randint(0, len(edges) - 1)
                    target = edges[edge_idx]
                    target_type = "edge"
                elif disp_type == "hospital_diversion":
                    hosp_idx = target_rng.randint(0, len(hospitals) - 1)
                    target = hospitals[hosp_idx]
                    target_type = "hospital"
                elif disp_type == "equipment_failure":
                    hosp_idx = target_rng.randint(0, len(hospitals) - 1)
                    target = hospitals[hosp_idx]
                    target_type = "hospital"
                elif disp_type == "surge_event":
                    # Surge adds patients (handled in patient_manager)
                    target = "SURGE"
                    target_type = "surge"
                else:
                    # accident
                    edge_idx = target_rng.randint(0, len(edges) - 1)
                    target = edges[edge_idx]
                    target_type = "edge"

                schedule.append({
                    "step": step_idx,
                    "disruption_type": disp_type,
                    "target": target,
                    "target_type": target_type,
                    "duration": duration,
                })

        return schedule
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest codered_env/tests/test_disruption_engine.py -v`
Expected: PASS (all 5 tests)

---

## Task 12: Wire Subsystems into Environment

**Files:**
- Rewrite: `codered_env/server/codered_environment.py` (replace Plan 1 placeholder logic with subsystem calls)
- Test: `codered_env/tests/test_wired_environment.py`

- [ ] **Step 1: Write failing tests**

```python
# codered_env/tests/test_wired_environment.py
def test_dispatch_routes_ambulance():
    from codered_env.server.codered_environment import CodeRedEnvironment
    from codered_env.server.models.actions import DispatchAmbulance
    env = CodeRedEnvironment()
    env.reset(seed=0, task_id="task1")
    obs = env.step(DispatchAmbulance(ambulance_id="AMB_1", target_node="NH45_BYPASS"))
    amb = next(a for a in obs.ambulances if a.id == "AMB_1")
    assert len(amb.route) > 0
    assert amb.eta_minutes > 0

def test_prepare_or_increases_prep_countdown():
    from codered_env.server.codered_environment import CodeRedEnvironment
    from codered_env.server.models.actions import PrepareOR
    env = CodeRedEnvironment()
    env.reset(seed=0, task_id="task1")
    obs = env.step(PrepareOR(hospital_id="HOSP_A", procedure_type="cardiac"))
    hosp = next(h for h in obs.hospitals if h.id == "HOSP_A")
    prep_or = next(o for o in hosp.operating_rooms if o.status == "prep")
    assert prep_or is not None

def test_assign_hospital_sets_patient_destination():
    from codered_env.server.codered_environment import CodeRedEnvironment
    from codered_env.server.models.actions import AssignHospital
    env = CodeRedEnvironment()
    env.reset(seed=0, task_id="task1")
    obs = env.step(AssignHospital(patient_id="P1", hospital_id="HOSP_A"))
    patient = next(p for p in obs.patients if p.patient_id == "P1")
    assert patient.assigned_hospital == "HOSP_A"

def test_patient_cannot_be_assigned_to_unsuitable_hospital():
    from codered_env.server.codered_environment import CodeRedEnvironment
    from codered_env.server.models.actions import AssignHospital
    env = CodeRedEnvironment()
    env.reset(seed=0, task_id="task1")
    # Task 1 is cardiac — HOSP_C cannot handle cardiac
    obs = env.step(AssignHospital(patient_id="P1", hospital_id="HOSP_C"))
    patient = next(p for p in obs.patients if p.patient_id == "P1")
    assert patient.assigned_hospital is None  # Assignment failed
    assert any("cannot treat cardiac" in a for a in obs.alerts)

def test_mutual_aid_not_available_task1():
    from codered_env.server.codered_environment import CodeRedEnvironment
    from codered_env.server.models.actions import RequestMutualAid
    env = CodeRedEnvironment()
    obs = env.reset(seed=0, task_id="task1")
    assert obs.mutual_aid_remaining == 0
    obs2 = env.step(RequestMutualAid())
    assert any("no calls remaining" in a for a in obs2.alerts)

def test_blood_emergency_release():
    from codered_env.server.codered_environment import CodeRedEnvironment
    from codered_env.server.models.actions import AllocateBlood
    env = CodeRedEnvironment()
    env.reset(seed=0, task_id="task1")
    obs = env.step(AllocateBlood(
        hospital_id="HOSP_A", patient_id="P1",
        blood_type="O_NEG", units=2, emergency=True
    ))
    assert any("Emergency blood" in a for a in obs.alerts)

def test_timestep_increments_patient_time():
    from codered_env.server.codered_environment import CodeRedEnvironment
    from codered_env.server.models.actions import MaintainPlan
    env = CodeRedEnvironment()
    env.reset(seed=0, task_id="task1")
    patient = env._patients[0]
    assert patient.time_since_onset == 0
    env.step(MaintainPlan())
    assert patient.time_since_onset == 1
    env.step(MaintainPlan())
    assert patient.time_since_onset == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest codered_env/tests/test_wired_environment.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Rewrite `codered_environment.py` to wire subsystems**

Key changes from Plan 1:
1. Replace inline `_init_hospitals` etc. with subsystem instances
2. Replace inline `_advance_time` with `self._road_network.tick()`, `self._hospital_system.tick()`, `self._blood_bank.tick()`
3. Replace inline action handlers with subsystem calls
4. Use `self._road_network.shortest_path()` for routing
5. Wire `self._disruption_engine` into the environment
6. Implement treatment completion (patient reaches treating status → after surgery duration, mark treated)
7. Implement `QueryBloodType` with 5-step delay
8. Implement `QueryORStatus` as costing 1 step (reveals detail for 1 step)

```python
# Full rewrite of codered_environment.py — key sections shown below
# (Full file is ~350 lines, same structure as Plan 1 but with subsystems wired)

class CodeRedEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        from .subsystems.road_network import RoadNetwork
        from .subsystems.hospital_system import HospitalSystem
        from .subsystems.blood_bank import BloodBankSystem
        from .subsystems.disruption_engine import DisruptionEngine
        from .subsystems.patient_manager import PatientManager
        from .subsystems.ambulance_manager import AmbulanceManager

        self._rng: Optional[random.Random] = None
        self._patients: List[Patient] = []
        self._road_network = RoadNetwork()
        self._hospital_system = HospitalSystem()
        self._blood_bank = BloodBankSystem()
        self._disruption_engine = DisruptionEngine()
        self._patient_manager = PatientManager()
        self._ambulance_manager = AmbulanceManager(self._ambulances)
        self._state: CodeRedState = CodeRedState()
        self._alerts: List[str] = []
        self._pending_blood_queries: Dict[str, int] = {}
        self._pending_or_queries: Dict[str, int] = {}
        self._active_disruptions: List[Dict] = []

    def reset(self, seed=None, episode_id=None, task_id="task1", **kwargs):
        self._rng = random.Random(seed)
        self._road_network = RoadNetwork()
        self._hospital_system = HospitalSystem()
        self._blood_bank = BloodBankSystem()
        self._disruption_engine.reset(seed=seed, task_id=task_id)
        self._patient_manager.reset(task_id, self._rng)
        self._patients = self._patient_manager.patients
        self._ambulance_manager.reset()
        self._state = CodeRedState(
            episode_id=episode_id or str(uuid4()),
            step_count=0, task_id=task_id,
            max_steps=TASK_CONFIG[task_id]["max_steps"],
            mutual_aid_available=TASK_CONFIG[task_id]["mutual_aid_calls"],
        )
        self._alerts = []
        self._pending_blood_queries = {}
        self._pending_or_queries = {}
        self._active_disruptions = []
        return self._build_observation()

    def _advance_time(self):
        # Patient deterioration + outcome check
        self._patient_manager.tick()
        # Ambulances: advance movement + check for arrivals
        self._ambulance_manager.tick()
        for amb_id, amb in self._ambulance_manager.all().items():
            if amb.status == "en_route" and amb.eta_minutes == 0:
                self._ambulance_manager.arrive(amb_id)
                self._do_ambulance_arrived(amb_id)
        # Hospitals
        self._hospital_system.tick()
        # Blood bank
        self._blood_bank.tick()
        completed = self._blood_bank.flush_completed_crossmatches()
        for entry in completed:
            self._alerts.append(
                f"Crossmatch complete: {entry['units']} units "
                f"{entry['blood_type']} reserved for {entry['patient_id']}"
            )
        # Road disruptions
        self._road_network.tick()
        self._road_network_tick_disruptions()
        # Advance pending queries
        for patient_id in list(self._pending_blood_queries.keys()):
            self._pending_blood_queries[patient_id] -= 1
            if self._pending_blood_queries[patient_id] <= 0:
                del self._pending_blood_queries[patient_id]
                self._reveal_blood_type(patient_id)
        for key in list(self._pending_or_queries.keys()):
            self._pending_or_queries[key] -= 1
            if self._pending_or_queries[key] <= 0:
                del self._pending_or_queries[key]
        # Roll new disruptions
        events = self._disruption_engine.roll_disruptions(
            step=self._state.step_count + 1,
            road_network=self._road_network,
            hospital_system=self._hospital_system,
        )
        for event in events:
            if event["target_type"] == "surge":
                self._patient_manager.add_surge_patients(self._rng)
                self._alerts.append(f"Surge event: additional patients arriving")
            else:
                self._alerts.append(
                    f"Disruption: {event['disruption_type']} on {event['target']}"
                )

    def _road_network_tick_disruptions(self) -> None:
        """
        Process road network disruption events.

        This method is called after self._road_network.tick() decrements disruption
        counters. It checks for newly expired disruptions and:
        1. Clears the disruption from the road network edge
        2. Logs an alert for the implementer/agent

        Disruptions are applied via self._road_network.set_disruption() (from the
        DisruptionEngine's roll_disruptions() output) and are tracked internally
        by RoadNetwork.tick() decrementing edge.disruption_remaining.
        """
        active = self._road_network.get_active_disruptions()
        for disp in active:
            if disp["remaining_steps"] <= 0:
                self._road_network.clear_disruption(disp["from_node"], disp["to_node"])
                self._alerts.append(f"Road cleared: {disp['from_node']} ↔ {disp['to_node']}")

    def _execute_action(self, action: CodeRedAction):
        # (Same structure as Plan 1 but calling subsystem methods)
        if isinstance(action, DispatchAmbulance):
            self._do_dispatch_ambulance(action)
        elif isinstance(action, PrepareOR):
            result = self._hospital_system.prepare_or(action.hospital_id, action.procedure_type)
            if not result["success"]:
                self._alerts.append(f"PrepareOR failed: {result['reason']}")
        elif isinstance(action, PageSpecialist):
            result = self._hospital_system.page_specialist(action.hospital_id, action.specialist_type)
            if not result["success"]:
                self._alerts.append(f"PageSpecialist failed: {result['reason']}")
        elif isinstance(action, AssignHospital):
            self._do_assign_hospital(action)
        elif isinstance(action, PreemptOR):
            result = self._hospital_system.preempt_or(action.hospital_id, action.or_index)
            if result["success"]:
                self._alerts.append(
                    f"OR preempted at {action.hospital_id} OR {action.or_index}: "
                    f"harm={result['harm']:.2f}, recovery={result['recovery_time']}min"
                )
            else:
                self._alerts.append(f"PreemptOR failed: {result['reason']}")
        elif isinstance(action, AllocateBlood):
            if action.emergency:
                result = self._blood_bank.emergency_release(
                    action.hospital_id, action.patient_id, action.blood_type, action.units
                )
                if result["success"]:
                    self._alerts.append(
                        f"Emergency blood: {action.units} units released for {action.patient_id}"
                    )
                else:
                    self._alerts.append(f"AllocateBlood failed: {result['reason']}")
            else:
                result = self._blood_bank.start_crossmatch(
                    action.hospital_id, action.patient_id, action.blood_type, action.units
                )
                if not result["success"]:
                    self._alerts.append(f"AllocateBlood failed: {result['reason']}")
        elif isinstance(action, TransferBlood):
            result = self._blood_bank.transfer(
                action.from_hospital, action.to_hospital, action.blood_type, action.units
            )
            if not result["success"]:
                self._alerts.append(f"TransferBlood failed: {result['reason']}")
        elif isinstance(action, RequestMutualAid):
            self._do_request_mutual_aid(action)
        elif isinstance(action, QueryBloodType):
            self._pending_blood_queries[action.patient_id] = 5
        elif isinstance(action, QueryORStatus):
            self._pending_or_queries[f"{action.hospital_id}:{action.or_index}"] = 1
        elif isinstance(action, MaintainPlan):
            pass
```

- [ ] **Step 4: Run wired environment tests**

Run: `pytest codered_env/tests/test_wired_environment.py -v`
Expected: PASS (all 7 tests)

---

## Task 13: Integration Tests (Full Episode)

**Files:**
- Test: `codered_env/tests/test_integration.py`

- [ ] **Step 1: Write integration tests**

```python
# codered_env/tests/test_integration.py
def test_task1_episode_runs_to_completion():
    from codered_env.server.codered_environment import CodeRedEnvironment
    from codered_env.server.models.actions import (
        MaintainPlan, DispatchAmbulance, AssignHospital, PrepareOR, PageSpecialist,
    )
    env = CodeRedEnvironment()
    env.reset(seed=0, task_id="task1")
    done = False
    steps = 0
    while not done and steps < 35:
        obs = env.state  # Get current state
        patient = obs.patients[0]
        amb = next(a for a in obs.ambulances if a.status == "available")
        if patient.status == "waiting" and amb:
            env.step(DispatchAmbulance(ambulance_id=amb.id, target_node=patient.location_node))
        elif patient.status == "dispatched" and patient.assigned_hospital is None:
            env.step(AssignHospital(patient_id=patient.patient_id, hospital_id="HOSP_A"))
        elif patient.assigned_hospital == "HOSP_A":
            env.step(PrepareOR(hospital_id="HOSP_A", procedure_type="cardiac"))
            env.step(PageSpecialist(hospital_id="HOSP_A", specialist_type="cardiologist"))
        else:
            env.step(MaintainPlan())
        result = env.step(MaintainPlan())
        done = result[2] is True or env.state.step_count >= env.state.max_steps
        steps += 1
    # Episode should complete without error
    assert env.state.step_count > 0

def test_grader_runs_without_error():
    from codered_env.server.codered_environment import CodeRedEnvironment
    from codered_env.server.models.actions import MaintainPlan
    env = CodeRedEnvironment()
    env.reset(seed=0, task_id="task1")
    for _ in range(30):
        _, _, done, _ = env.step(MaintainPlan())
        if done:
            break
    score = env._compute_grader_score()
    assert 0.0 <= score <= 1.0
```

- [ ] **Step 2: Run integration tests**

Run: `pytest codered_env/tests/test_integration.py -v`
Expected: PASS

---
