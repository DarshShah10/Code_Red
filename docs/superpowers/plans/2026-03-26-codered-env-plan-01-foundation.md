# CodeRedEnv — Plan 1: Foundation

> **For agentic workers:** Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Build the project scaffold, static data (city, hospitals, ambulances), entity models (Patient, Hospital, Ambulance, RoadNode, BloodBank), and the core environment class with working `reset()`, `step()`, `state()` for Task 1 (single cardiac patient, no disruptions).

**Architecture:** OpenEnv Gym-style environment extending `Environment`. Pydantic models for all entities, actions, observations, and state. Subsystems are placeholder classes initially; full logic added in Plan 2.

**Tech Stack:** Python 3.10+, Pydantic, OpenEnv (openenv-core), FastAPI, uvicorn, pytest

---

## File Structure

```
codered_env/
├── __init__.py
├── models.py                    # Top-level exports
├── client.py                    # EnvClient subclass
├── openenv.yaml                 # Manifest
├── pyproject.toml               # Dependencies
├── README.md                    # (Plan 3)
├── server/
│   ├── __init__.py
│   ├── app.py                   # FastAPI + create_app (Plan 1: minimal, Plan 3: full)
│   ├── codered_environment.py   # Core Environment class
│   ├── Dockerfile               # (Plan 1: scaffold, Plan 3: complete)
│   └── subsystems/
│       ├── __init__.py
│       ├── constants.py         # City graph, hospital defs, ambulance defs — all static data
│       ├── patient_manager.py   # Patient generation + deterioration (placeholder)
│       ├── road_network.py      # Graph + routing (placeholder)
│       ├── ambulance_manager.py  # Ambulance state (placeholder)
│       ├── hospital_system.py   # Hospital state + OR/prep (placeholder)
│       ├── blood_bank.py        # Blood stocks (placeholder)
│       └── disruption_engine.py # (Plan 2 — placeholder returns empty)
└── baseline.py                  # (Plan 3)
# Note: Grader (server/grader.py) is scaffolded as a stub in Plan 1 Task 3
# and fully implemented in Plan 3 Task 14.
```

---

## Task 1: Project Scaffold

**Files:**
- Create: `codered_env/__init__.py`
- Create: `codered_env/pyproject.toml`
- Create: `codered_env/openenv.yaml`
- Create: `codered_env/server/__init__.py`
- Create: `codered_env/server/Dockerfile`
- Create: `codered_env/server/subsystems/__init__.py`
- Create: `codered_env/server/grader.py`   # stub grader (full impl in Plan 3)

- [ ] **Step 1: Create `codered_env/pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "openenv-codered"
version = "0.1.0"
description = "CodeRedEnv — Emergency Medical Coordination Environment for OpenEnv"
requires-python = ">=3.10"
dependencies = [
    "openenv-core[core]>=0.2.1",
    "fastapi>=0.109.0",
    "uvicorn>=0.27.0",
    "pydantic>=2.5.0",
    "openai>=1.56.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0.0", "pytest-cov>=4.0.0"]

[project.scripts]
server = "codered_env.server.app:main"

[tool.setuptools]
include-package-data = true
packages = ["codered_env", "codered_env.server", "codered_env.server.subsystems"]
package-dir = { "codered_env" = ".", "codered_env.server" = "server" }
```

- [ ] **Step 2: Create `codered_env/openenv.yaml`**

```yaml
spec_version: 1
name: codered_env
type: space
runtime: fastapi
app: server.app:app
port: 8000
```

- [ ] **Step 3: Create `codered_env/__init__.py`**

```python
"""CodeRedEnv — Emergency Medical Coordination Environment."""

from .models import (
    CodeRedAction,
    CodeRedObservation,
    CodeRedState,
)
from .client import CodeRedEnv

__all__ = [
    "CodeRedAction",
    "CodeRedObservation",
    "CodeRedState",
    "CodeRedEnv",
]
```

- [ ] **Step 4: Create server Dockerfile scaffold (copy from template, minimal)**

```dockerfile
ARG BASE_IMAGE=ghcr.io/facebookresearch/openenv/openenv-base:latest
FROM ${BASE_IMAGE} AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
ARG BUILD_MODE=in-repo
ARG ENV_NAME=codered_env
COPY . /app/env
WORKDIR /app/env
RUN if ! command -v uv >/dev/null 2>&1; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh && \
        mv /root/.local/bin/uv /usr/local/bin/uv && \
        mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-editable

FROM ${BASE_IMAGE}
WORKDIR /app
COPY --from=builder /app/env/.venv /app/.venv
COPY --from=builder /app/env /app/env
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/env:$PYTHONPATH"
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
CMD ["sh", "-c", "cd /app/env && uvicorn server.app:app --host 0.0.0.0 --port 8000"]
```

- [ ] **Step 5: Create empty `__init__.py` files**

Create empty `__init__.py` files for: `codered_env/server/__init__.py`, `codered_env/server/subsystems/__init__.py`. Create `server/grader.py` with a stub that returns `RubricResult(time_score=1.0, efficiency=1.0, secondary_harm=1.0, prep_ready=1.0, mutual_aid_penalty=0.0, final_score=1.0, breakdown={})`.

- [ ] **Step 6: Verify scaffold builds**

Run: `cd codered_env && uv sync`
Expected: Dependencies install, no import errors

---

## Task 2: Static City Data (Constants)

**Files:**
- Create: `codered_env/server/subsystems/constants.py`
- Test: `codered_env/tests/test_constants.py`

- [ ] **Step 1: Write the failing test**

```python
# codered_env/tests/test_constants.py
from codered_env.server.subsystems.constants import (
    CITY_NODES,
    CITY_EDGES,
    HOSPITALS,
    AMBULANCES,
    PATIENT_TARGET_TIMES,
    BLOOD_TYPES,
)

def test_city_has_12_nodes():
    assert len(CITY_NODES) == 12

def test_city_edges_make_graph_connected():
    """All nodes must be reachable from each other."""
    import networkx as nx
    G = nx.Graph()
    for node in CITY_NODES:
        G.add_node(node["id"])
    for edge in CITY_EDGES:
        G.add_edge(edge["from"], edge["to"])
    assert nx.is_connected(G), "City graph is not fully connected"

def test_hospital_a_can_handle_cardiac():
    hosp_a = next(h for h in HOSPITALS if h["id"] == "HOSP_A")
    assert "cardiac" in hosp_a["capabilities"]

def test_hospital_c_cannot_handle_cardiac():
    hosp_c = next(h for h in HOSPITALS if h["id"] == "HOSP_C")
    assert "cardiac" not in hosp_c["capabilities"]
    assert "stroke" not in hosp_c["capabilities"]

def test_fleet_has_5_ambulances():
    assert len(AMBULANCES) == 5

def test_ambulance_ids_unique():
    ids = [a["id"] for a in AMBULANCES]
    assert len(ids) == len(set(ids))

def test_blood_types_listed():
    assert "O_POS" in BLOOD_TYPES
    assert "AB_NEG" in BLOOD_TYPES
    assert len(BLOOD_TYPES) == 8

def test_target_times_defined():
    assert PATIENT_TARGET_TIMES["cardiac"] == 90
    assert PATIENT_TARGET_TIMES["stroke"] == 60
    assert PATIENT_TARGET_TIMES["trauma"] == 60
    assert PATIENT_TARGET_TIMES["general"] == 120
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest codered_env/tests/test_constants.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write `constants.py` with all static data**

```python
"""Static data for CodeRedEnv — Prakashnagar city layout, hospital definitions, ambulance fleet."""

from typing import Dict, List

# =============================================================================
# CITY ROAD NETWORK — Prakashnagar
# Hub-and-spoke city: central NH45 bypass + old city + IT corridor
# 12 nodes, 15 bidirectional edges
# =============================================================================

CITY_NODES: List[Dict] = [
    {"id": "RAJIV_CHOWK", "name": "Rajiv Chowk", "type": "intersection"},
    {"id": "LAJPAT_NAGAR", "name": "Lajpat Nagar", "type": "intersection"},
    {"id": "CHOWKHA", "name": "Chowkha", "type": "intersection", "notes": "old city, narrow streets"},
    {"id": "RAILWAY_XING", "name": "Railway Crossing", "type": "intersection"},
    {"id": "NH45_BYPASS", "name": "NH45 Bypass", "type": "arterial"},
    {"id": "IT_HUB", "name": "IT Hub", "type": "intersection"},
    {"id": "AIIMS_PRAKASH", "name": "AIIMS Prakash", "type": "hospital"},
    {"id": "DISTRICT_HOSP", "name": "District Hospital", "type": "hospital"},
    {"id": "COMMUNITY_HC", "name": "Community Health Centre", "type": "hospital"},
    {"id": "MG_CHOWK", "name": "MG Chowk", "type": "intersection"},
    {"id": "SECTOR_12", "name": "Sector 12", "type": "intersection"},
    {"id": "RING_ROAD", "name": "Ring Road", "type": "intersection"},
]

CITY_EDGES: List[Dict] = [
    # Old city cluster
    {"from": "RAJIV_CHOWK", "to": "LAJPAT_NAGAR", "base_time": 3, "notes": "narrow street"},
    {"from": "LAJPAT_NAGAR", "to": "CHOWKHA", "base_time": 4, "notes": "old city"},
    {"from": "LAJPAT_NAGAR", "to": "RING_ROAD", "base_time": 6, "notes": "outer ring"},
    {"from": "CHOWKHA", "to": "DISTRICT_HOSP", "base_time": 2, "notes": "hospital access"},
    {"from": "CHOWKHA", "to": "RAILWAY_XING", "base_time": 5, "notes": "congestion-prone"},
    {"from": "RAILWAY_XING", "to": "NH45_BYPASS", "base_time": 6, "notes": "highway segment"},
    {"from": "RAILWAY_XING", "to": "RING_ROAD", "base_time": 4, "notes": "ring connector"},
    # NH45 bypass (main artery)
    {"from": "NH45_BYPASS", "to": "IT_HUB", "base_time": 4, "notes": "fast road"},
    {"from": "NH45_BYPASS", "to": "MG_CHOWK", "base_time": 8, "notes": "long connector"},
    {"from": "NH45_BYPASS", "to": "RING_ROAD", "base_time": 3, "notes": "ring junction"},
    # IT corridor
    {"from": "IT_HUB", "to": "SECTOR_12", "base_time": 3, "notes": "IT corridor"},
    {"from": "IT_HUB", "to": "COMMUNITY_HC", "base_time": 7, "notes": "indirect clinic access"},
    # MG Chowk connections
    {"from": "MG_CHOWK", "to": "AIIMS_PRAKASH", "base_time": 5, "notes": "hospital access"},
    {"from": "MG_CHOWK", "to": "SECTOR_12", "base_time": 4, "notes": "IT corridor"},
    # Community HC access
    {"from": "RING_ROAD", "to": "COMMUNITY_HC", "base_time": 5, "notes": "clinic access"},
]

# =============================================================================
# HOSPITALS
# =============================================================================

BLOOD_TYPES: List[str] = [
    "A_POS", "A_NEG", "B_POS", "B_NEG",
    "AB_POS", "AB_NEG", "O_POS", "O_NEG",
]

HOSPITALS: List[Dict] = [
    {
        "id": "HOSP_A",
        "name": "AIIMS Prakash",
        "node_id": "AIIMS_PRAKASH",
        "capabilities": ["cardiac", "stroke", "trauma", "stabilization"],
        "specialists": {
            "cardiologist": {"available": 2, "total": 2},
            "neurologist": {"available": 1, "total": 1},
            "trauma_surgeon": {"available": 2, "total": 2},
        },
        "num_or": 3,
        "icu_beds": {"total": 4, "available": 4},
        "blood_stock": {
            "A_POS": 10, "A_NEG": 5, "B_POS": 10, "B_NEG": 5,
            "AB_POS": 5, "AB_NEG": 3, "O_POS": 12, "O_NEG": 6,
        },
    },
    {
        "id": "HOSP_B",
        "name": "District Hospital",
        "node_id": "DISTRICT_HOSP",
        "capabilities": ["cardiac", "trauma", "stabilization"],
        "specialists": {
            "cardiologist": {"available": 1, "total": 1},
            "neurologist": {"available": 0, "total": 0},
            "trauma_surgeon": {"available": 1, "total": 1},
        },
        "num_or": 2,
        "icu_beds": {"total": 2, "available": 2},
        "blood_stock": {
            "A_POS": 6, "A_NEG": 3, "B_POS": 6, "B_NEG": 3,
            "AB_POS": 3, "AB_NEG": 2, "O_POS": 8, "O_NEG": 4,
        },
    },
    {
        "id": "HOSP_C",
        "name": "Community Health Centre",
        "node_id": "COMMUNITY_HC",
        "capabilities": ["stabilization"],
        "specialists": {
            "cardiologist": {"available": 0, "total": 0},
            "neurologist": {"available": 0, "total": 0},
            "trauma_surgeon": {"available": 0, "total": 0},
        },
        "num_or": 0,
        "icu_beds": {"total": 1, "available": 1},
        "blood_stock": {
            "O_POS": 4, "O_NEG": 2,
            "A_POS": 0, "A_NEG": 0, "B_POS": 0, "B_NEG": 0,
            "AB_POS": 0, "AB_NEG": 0,
        },
    },
]

# =============================================================================
# AMBULANCE FLEET
# =============================================================================

AMBULANCES: List[Dict] = [
    {"id": "AMB_1", "equipment": "ALS", "base_node": "RAILWAY_XING"},
    {"id": "AMB_2", "equipment": "ALS", "base_node": "NH45_BYPASS"},
    {"id": "AMB_3", "equipment": "BLS", "base_node": "LAJPAT_NAGAR"},
    {"id": "AMB_4", "equipment": "BLS", "base_node": "IT_HUB"},
    {"id": "AMB_5", "equipment": "BLS", "base_node": "RAJIV_CHOWK"},
]

# =============================================================================
# PATIENT CONFIGURATION
# =============================================================================

PATIENT_TARGET_TIMES: Dict[str, int] = {
    "cardiac": 90,    # Door-to-balloon (AHA standard)
    "stroke": 60,     # Door-to-needle (AHA standard)
    "trauma": 60,     # Golden hour (ATLS standard)
    "general": 120,   # Stabilization target
}

PATIENT_CONDITION_REQUIREMENTS: Dict[str, List[str]] = {
    "cardiac": ["cardiac"],
    "stroke": ["stroke"],
    "trauma": ["trauma"],
    "general": ["stabilization"],
}

# =============================================================================
# TASK CONFIGURATION
# =============================================================================

TASK_CONFIG: Dict[str, Dict] = {
    "task1": {
        "patients": [
            {"condition": "cardiac", "onset_step": 0},
        ],
        "disruption_prob": 0.0,
        "mutual_aid_calls": 0,
        "max_steps": 30,
    },
    "task2": {
        "patients": [
            {"condition": "cardiac", "onset_step": 0},
            {"condition": "stroke", "onset_step": 5},  # 5 minutes after first
        ],
        "disruption_prob": 0.05,
        "mutual_aid_calls": 1,
        "max_steps": 45,
    },
    "task3": {
        "patients": [
            {"condition": "cardiac", "onset_step": 0},
            {"condition": "cardiac", "onset_step": 3},
            {"condition": "stroke", "onset_step": 5},
            {"condition": "trauma", "onset_step": 8},
            {"condition": "general", "onset_step": 12},
        ],
        "disruption_prob": 0.15,
        "mutual_aid_calls": 2,
        "max_steps": 60,
    },
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest codered_env/tests/test_constants.py -v`
Expected: PASS (all 8 tests)

---

## Task 3: Entity Pydantic Models

**Files:**
- Create: `codered_env/server/subsystems/entities.py`
- Create: `codered_env/server/models/__init__.py`
- Create: `codered_env/server/models/entities.py`
- Create: `codered_env/server/models/actions.py`
- Create: `codered_env/server/models/observations.py`
- Create: `codered_env/server/models/state.py`
- Create: `codered_env/models.py`
- Test: `codered_env/tests/test_entities.py`

**Note:** All models live in `server/models/` and are re-exported from the top-level `models.py`.

- [ ] **Step 1: Write the failing test**

```python
# codered_env/tests/test_entities.py
import pytest
from pydantic import ValidationError

def test_patient_valid():
    from codered_env.server.models.entities import Patient
    p = Patient(
        patient_id="P1",
        condition="cardiac",
        tier="critical",
        location_node="NH45_BYPASS",
        time_since_onset=5,
    )
    assert p.patient_id == "P1"
    assert p.status == "waiting"

def test_patient_unknown_blood_type():
    from codered_env.server.models.entities import Patient
    p = Patient(
        patient_id="P1",
        condition="cardiac",
        tier="critical",
        location_node="NH45_BYPASS",
        time_since_onset=0,
    )
    assert p.blood_type is None  # Unknown by default

def test_hospital_state_has_correct_defaults():
    from codered_env.server.models.entities import HospitalState
    h = HospitalState(id="HOSP_A", node_id="AIIMS_PRAKASH")
    assert len(h.operating_rooms) == 3  # Default 3 ORs

def test_ambulance_state_defaults():
    from codered_env.server.models.entities import AmbulanceState
    a = AmbulanceState(id="AMB_1", node_id="RAILWAY_XING", equipment="ALS")
    assert a.status == "available"
    assert a.route == []

def test_road_network_state_has_edges():
    from codered_env.server.models.entities import RoadNetworkState
    r = RoadNetworkState()
    assert len(r.edges) > 0
    assert r.edges["RAILWAY_XING->NH45_BYPASS"]["base_time"] == 6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest codered_env/tests/test_entities.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write `server/models/entities.py`**

```python
"""Pydantic entity models for CodeRedEnv."""

from enum import Enum
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class PatientStatus(str, Enum):
    WAITING = "waiting"
    DISPATCHED = "dispatched"
    TRANSPORTING = "transporting"
    TREATING = "treating"
    TREATED = "treated"
    DECEASED = "deceased"


class PatientCondition(str, Enum):
    CARDIAC = "cardiac"
    STROKE = "stroke"
    TRAUMA = "trauma"
    GENERAL = "general"


class PatientTier(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"


class Patient(BaseModel):
    """A patient requiring emergency medical response."""
    patient_id: str
    condition: PatientCondition
    tier: PatientTier
    location_node: str
    time_since_onset: int = 0
    assigned_ambulance: Optional[str] = None
    assigned_hospital: Optional[str] = None
    status: PatientStatus = PatientStatus.WAITING
    blood_type: Optional[str] = None  # None = unknown until QueryBloodType
    treatment_start_time: Optional[int] = None
    treatment_complete_time: Optional[int] = None
    outcome: Optional[Literal["saved", "deceased"]] = None
    is_secondary: bool = False  # True for patients who arrive after the primary emergency

    model_config = {"extra": "forbid"}


class ORStatus(str, Enum):
    IDLE = "idle"
    IN_USE = "in_use"
    PREP = "prep"


class OperatingRoom(BaseModel):
    """An operating room in a hospital."""
    index: int
    status: ORStatus = ORStatus.IDLE
    procedure_type: Optional[str] = None
    minutes_remaining: Optional[int] = None
    patient_id: Optional[str] = None
    assigned_specialist: Optional[str] = None


class SpecialistStatus(BaseModel):
    available: int
    total: int
    status: Literal["available", "paged", "en_route", "busy"] = "available"
    minutes_until_available: int = 0


class HospitalState(BaseModel):
    """State of a hospital including resources."""
    id: str
    node_id: str
    capabilities: List[Literal["cardiac", "stroke", "trauma", "stabilization"]]
    specialists: Dict[str, SpecialistStatus]
    operating_rooms: List[OperatingRoom]
    icu_beds: Dict[Literal["total", "available"], int]
    blood_stock: Dict[str, int]  # blood_type -> units
    on_diversion: bool = False
    # Mutable: prep countdowns tracked here
    or_prep_countdowns: Dict[int, int] = Field(default_factory=dict)  # or_index -> minutes_left

    model_config = {"extra": "forbid"}


class AmbulanceStatus(str, Enum):
    AVAILABLE = "available"
    DISPATCHED = "dispatched"
    TRANSPORTING = "transporting"
    RETURNING = "returning"
    OFFLINE = "offline"


class AmbulanceEquipment(str, Enum):
    ALS = "ALS"
    BLS = "BLS"


class AmbulanceState(BaseModel):
    """State of an ambulance."""
    id: str
    node_id: str
    equipment: AmbulanceEquipment
    status: AmbulanceStatus = AmbulanceStatus.AVAILABLE
    assigned_patient: Optional[str] = None
    route: List[str] = Field(default_factory=list)
    eta_minutes: int = 0
    destination_type: Optional[Literal["patient", "hospital", "base"]] = None

    model_config = {"extra": "forbid"}


class EdgeState(BaseModel):
    """State of a road segment."""
    from_node: str
    to_node: str
    base_time: int
    congestion_multiplier: float = 1.0
    disrupted: bool = False
    disruption_type: Optional[str] = None

    model_config = {"extra": "forbid"}

    def effective_time(self) -> float:
        if self.disrupted and self.disruption_type == "road_closure":
            return float("inf")
        return self.base_time * self.congestion_multiplier


class RoadNetworkState(BaseModel):
    """The city road network with current travel times."""
    edges: Dict[str, EdgeState] = Field(default_factory=dict)  # "A->B" -> EdgeState

    model_config = {"extra": "forbid"}

    def get_edge_key(self, from_node: str, to_node: str) -> str:
        # Normalize to lexicographically smaller first
        nodes = sorted([from_node, to_node])
        return f"{nodes[0]}->{nodes[1]}"

    def get_travel_time(self, from_node: str, to_node: str) -> float:
        key = self.get_edge_key(from_node, to_node)
        edge = self.edges.get(key)
        if edge is None:
            return float("inf")
        return edge.effective_time()


class BloodBankState(BaseModel):
    """Blood bank state for a hospital."""
    hospital_id: str
    stocks: Dict[str, int]  # blood_type -> units
    crossmatch_queue: List[Dict] = Field(default_factory=list)

    model_config = {"extra": "forbid"}
```

- [ ] **Step 4: Write `server/models/actions.py`**

```python
"""Typed action models for CodeRedEnv."""

from typing import Literal, Optional

from pydantic import Field

from openenv.core.env_server.types import Action


class DispatchAmbulance(Action):
    """Dispatch an ambulance to a target node."""
    ambulance_id: str = Field(..., description="Ambulance ID to dispatch")
    target_node: str = Field(..., description="Target node ID")


class PrepareOR(Action):
    """Begin OR preparation for a procedure type."""
    hospital_id: str = Field(..., description="Hospital ID")
    procedure_type: Literal["cardiac", "stroke", "trauma", "general"] = Field(
        ..., description="Type of procedure being prepared for"
    )


class PageSpecialist(Action):
    """Page a specialist at a hospital."""
    hospital_id: str = Field(..., description="Hospital ID")
    specialist_type: Literal["cardiologist", "neurologist", "trauma_surgeon"] = Field(
        ..., description="Type of specialist to page"
    )


class AssignHospital(Action):
    """Assign a patient to a destination hospital."""
    patient_id: str = Field(..., description="Patient ID")
    hospital_id: str = Field(..., description="Destination hospital ID")


class PreemptOR(Action):
    """Preempt (clear) an operating room for emergency use."""
    hospital_id: str = Field(..., description="Hospital ID")
    or_index: int = Field(..., ge=0, description="OR index to preempt")


class AllocateBlood(Action):
    """Allocate blood units for a patient."""
    hospital_id: str = Field(..., description="Hospital ID")
    patient_id: str = Field(..., description="Patient ID")
    blood_type: str = Field(..., description="Blood type to allocate")
    units: int = Field(..., ge=1, description="Number of units")
    emergency: bool = Field(
        default=False,
        description="If True, use emergency O_NEG release (instant); if False, use crossmatch (15 min)"
    )


class TransferBlood(Action):
    """Transfer blood units between hospitals."""
    from_hospital: str = Field(..., description="Source hospital ID")
    to_hospital: str = Field(..., description="Destination hospital ID")
    blood_type: str = Field(..., description="Blood type")
    units: int = Field(..., ge=1, description="Number of units")


class RequestMutualAid(Action):
    """Request mutual aid ambulance (Task 2/3 only). Has 12-minute arrival latency."""
    pass  # No fields needed


class QueryBloodType(Action):
    """Query a patient's blood type. Takes 5 minutes to complete."""
    patient_id: str = Field(..., description="Patient ID to test")


class QueryORStatus(Action):
    """Query detailed OR status. Takes 1 step to complete."""
    hospital_id: str = Field(..., description="Hospital ID")
    or_index: int = Field(..., ge=0, description="OR index to query")


class MaintainPlan(Action):
    """No-op: continue current plan without changes."""
    pass


# Union of all action types
CodeRedAction = (
    DispatchAmbulance
    | PrepareOR
    | PageSpecialist
    | AssignHospital
    | PreemptOR
    | AllocateBlood
    | TransferBlood
    | RequestMutualAid
    | QueryBloodType
    | QueryORStatus
    | MaintainPlan
)
```

- [ ] **Step 5: Write `server/models/observations.py`**

```python
"""Typed observation and state models for CodeRedEnv."""

from typing import Any, Dict, List, Optional

from pydantic import Field

from openenv.core.env_server.types import Observation

from .entities import (
    AmbulanceState,
    BloodBankState,
    HospitalState,
    Patient,
    RoadNetworkState,
)


class CodeRedObservation(Observation):
    """Full observation of the emergency coordination state."""
    step: int = Field(default=0, description="Current timestep (1-indexed)")
    patients: List[Patient] = Field(default_factory=list)
    ambulances: List[AmbulanceState] = Field(default_factory=list)
    hospitals: List[HospitalState] = Field(default_factory=list)
    blood_banks: List[BloodBankState] = Field(default_factory=list)
    road_network: RoadNetworkState = Field(default_factory=RoadNetworkState)
    alerts: List[str] = Field(
        default_factory=list,
        description="New events and action failures this timestep"
    )
    mutual_aid_remaining: int = Field(
        default=0,
        description="Alias for mutual_aid_available for API compatibility"
    )
    time_score_preview: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Running time-score estimate"
    )
    patients_remaining: int = Field(
        default=0,
        description="Non-terminal patient count"
    )

    model_config = {"extra": "forbid"}
```

- [ ] **Step 6: Write `server/models/state.py`**

```python
"""State model for CodeRedEnv."""

from typing import Any, Dict, List, Optional

from pydantic import Field

from openenv.core.env_server.types import State


class DisruptionState(State):
    """State of an active disruption."""
    disruption_type: str
    target: str  # edge key or hospital_id
    remaining_steps: int


class CodeRedState(State):
    """Episode-level state for CodeRedEnv."""
    episode_id: Optional[str] = None
    step_count: int = Field(default=0, ge=0)
    task_id: str = "task1"
    cum_reward: float = 0.0
    max_steps: int = 30
    mutual_aid_used: int = 0
    mutual_aid_available: int = 0
    disruptions_active: List[DisruptionState] = Field(default_factory=list)
    all_patients_terminal: bool = False

    model_config = {"extra": "allow"}  # Allow extra fields for flexibility
```

- [ ] **Step 7: Write `server/models/__init__.py` and top-level `models.py`**

```python
# codered_env/server/models/__init__.py
"""CodeRedEnv data models."""
from .entities import (
    Patient,
    PatientCondition,
    PatientStatus,
    PatientTier,
    HospitalState,
    OperatingRoom,
    ORStatus,
    SpecialistStatus,
    AmbulanceState,
    AmbulanceEquipment,
    AmbulanceStatus,
    RoadNetworkState,
    EdgeState,
    BloodBankState,
)
from .actions import (
    CodeRedAction,
    DispatchAmbulance,
    PrepareOR,
    PageSpecialist,
    AssignHospital,
    PreemptOR,
    AllocateBlood,
    TransferBlood,
    RequestMutualAid,
    QueryBloodType,
    QueryORStatus,
    MaintainPlan,
)
from .observations import CodeRedObservation
from .state import CodeRedState

__all__ = [
    # Entities
    "Patient",
    "PatientCondition",
    "PatientStatus",
    "PatientTier",
    "HospitalState",
    "OperatingRoom",
    "ORStatus",
    "SpecialistStatus",
    "AmbulanceState",
    "AmbulanceEquipment",
    "AmbulanceStatus",
    "RoadNetworkState",
    "EdgeState",
    "BloodBankState",
    # Actions
    "CodeRedAction",
    "DispatchAmbulance",
    "PrepareOR",
    "PageSpecialist",
    "AssignHospital",
    "PreemptOR",
    "AllocateBlood",
    "TransferBlood",
    "RequestMutualAid",
    "QueryBloodType",
    "QueryORStatus",
    "MaintainPlan",
    # Observation / State
    "CodeRedObservation",
    "CodeRedState",
]
```

```python
# codered_env/models.py — top-level re-exports
"""CodeRedEnv — top-level model exports."""
from .server.models import (
    CodeRedAction,
    CodeRedObservation,
    CodeRedState,
)

__all__ = ["CodeRedAction", "CodeRedObservation", "CodeRedState"]
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `pytest codered_env/tests/test_entities.py -v`
Expected: PASS

---

## Task 4: Core Environment Class

**Files:**
- Create: `codered_env/server/codered_environment.py`
- Modify: `codered_env/server/app.py` (minimal)
- Test: `codered_env/tests/test_environment.py`

- [ ] **Step 1: Write the failing test**

```python
# codered_env/tests/test_environment.py
import pytest

def test_reset_task1_returns_observation():
    from codered_env.server.codered_environment import CodeRedEnvironment
    env = CodeRedEnvironment()
    obs = env.reset(seed=42, task_id="task1")
    assert obs.step == 1
    assert len(obs.patients) == 1
    assert obs.patients[0].condition == "cardiac"
    assert obs.patients[0].status == "waiting"

def test_reset_task1_has_5_ambulances():
    from codered_env.server.codered_environment import CodeRedEnvironment
    env = CodeRedEnvironment()
    obs = env.reset(seed=0, task_id="task1")
    assert len(obs.ambulances) == 5

def test_reset_task1_has_3_hospitals():
    from codered_env.server.codered_environment import CodeRedEnvironment
    env = CodeRedEnvironment()
    obs = env.reset(seed=0, task_id="task1")
    assert len(obs.hospitals) == 3

def test_step_increments_counter():
    from codered_env.server.codered_environment import CodeRedEnvironment
    from codered_env.server.models.actions import MaintainPlan
    env = CodeRedEnvironment()
    obs = env.reset(seed=0, task_id="task1")
    assert env.state.step_count == 0
    obs2 = env.step(MaintainPlan())
    assert obs2.step == 2
    assert env.state.step_count == 1

def test_state_returns_episode_metadata():
    from codered_env.server.codered_environment import CodeRedEnvironment
    env = CodeRedEnvironment()
    env.reset(seed=0, task_id="task1")
    state = env.state
    assert state.task_id == "task1"
    assert state.max_steps == 30
    assert state.mutual_aid_available == 0

def test_seed_reproducibility():
    from codered_env.server.codered_environment import CodeRedEnvironment
    from codered_env.server.models.actions import MaintainPlan
    env1 = CodeRedEnvironment()
    env2 = CodeRedEnvironment()
    obs1 = env1.reset(seed=123, task_id="task1")
    obs2 = env2.reset(seed=123, task_id="task1")
    # Same seed = same patient location
    assert obs1.patients[0].location_node == obs2.patients[0].location_node

def test_dispatch_increments_step():
    from codered_env.server.codered_environment import CodeRedEnvironment
    from codered_env.server.models.actions import DispatchAmbulance
    env = CodeRedEnvironment()
    env.reset(seed=0, task_id="task1")
    obs = env.step(DispatchAmbulance(ambulance_id="AMB_1", target_node="NH45_BYPASS"))
    assert obs.step == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest codered_env/tests/test_environment.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write `codered_environment.py`**

```python
"""CodeRedEnvironment — Core OpenEnv environment."""

from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

from .models import (
    CodeRedAction,
    CodeRedObservation,
    CodeRedState,
)
from .models.entities import (
    AmbulanceEquipment,
    AmbulanceState,
    AmbulanceStatus,
    BloodBankState,
    EdgeState,
    HospitalState,
    OperatingRoom,
    ORStatus,
    Patient,
    PatientCondition,
    PatientStatus,
    PatientTier,
    RoadNetworkState,
    SpecialistStatus,
)
from .subsystems.constants import (
    AMBULANCES,
    BLOOD_TYPES,
    CITY_EDGES,
    CITY_NODES,
    HOSPITALS,
    PATIENT_CONDITION_REQUIREMENTS,
    PATIENT_TARGET_TIMES,
    TASK_CONFIG,
)


class CodeRedEnvironment(Environment):
    """OpenEnv environment for emergency medical coordination."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True  # Each WS session gets own env instance

    def __init__(self):
        self._patients: List[Patient] = []
        self._ambulances: Dict[str, AmbulanceState] = {}
        self._hospitals: Dict[str, HospitalState] = {}
        self._blood_banks: Dict[str, BloodBankState] = {}
        self._road_network: RoadNetworkState = RoadNetworkState()
        self._state: CodeRedState = CodeRedState()
        self._alerts: List[str] = []
        self._action_results: Dict[int, str] = {}  # action_index -> failure reason
        self._rng_seed: Optional[int] = None

    # =========================================================================
    # Public API
    # =========================================================================

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: str = "task1",
        **kwargs: Any,
    ) -> CodeRedObservation:
        """Reset the environment for a new episode."""
        import random

        self._rng_seed = seed
        rng = random.Random(seed)

        self._state = CodeRedState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=task_id,
            cum_reward=0.0,
            max_steps=TASK_CONFIG[task_id]["max_steps"],
            mutual_aid_available=TASK_CONFIG[task_id]["mutual_aid_calls"],
            mutual_aid_used=0,
        )

        self._alerts = []
        self._action_results = {}

        # Initialize subsystems
        self._init_road_network()
        self._init_hospitals()
        self._init_ambulances()
        self._init_blood_banks()
        self._init_patients(task_id, rng)

        return self._build_observation()

    def step(
        self,
        action: CodeRedAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Tuple[CodeRedObservation, Optional[float], bool, Dict[str, Any]]:
        """Execute one timestep."""
        self._state.step_count += 1
        self._alerts = []

        # Advance all systems by 1 minute
        self._advance_time()

        # Execute action(s) — action is a union, dispatch to handler
        self._execute_action(action)

        # Check termination
        done = self._check_done()

        # Compute reward
        reward = self._compute_step_reward()

        self._state.cum_reward += reward

        return self._build_observation(), reward, done, {}

    @property
    def state(self) -> CodeRedState:
        return self._state

    # =========================================================================
    # Initialization
    # =========================================================================

    def _init_road_network(self) -> None:
        """Build the road network state from constants."""
        self._road_network = RoadNetworkState()
        for edge in CITY_EDGES:
            key = self._road_network.get_edge_key(edge["from"], edge["to"])
            self._road_network.edges[key] = EdgeState(
                from_node=edge["from"],
                to_node=edge["to"],
                base_time=edge["base_time"],
                congestion_multiplier=1.0,
                disrupted=False,
            )

    def _init_hospitals(self) -> None:
        """Initialize hospital states from constants."""
        self._hospitals = {}
        for hosp in HOSPITALS:
            ors = [
                OperatingRoom(index=i, status=ORStatus.IDLE)
                for i in range(hosp["num_or"])
            ]
            specialists = {
                role: SpecialistStatus(
                    available=data["available"],
                    total=data["total"],
                )
                for role, data in hosp["specialists"].items()
            }
            self._hospitals[hosp["id"]] = HospitalState(
                id=hosp["id"],
                node_id=hosp["node_id"],
                capabilities=hosp["capabilities"],
                specialists=specialists,
                operating_rooms=ors,
                icu_beds=hosp["icu_beds"],
                blood_stock=dict(hosp["blood_stock"]),
            )

    def _init_ambulances(self) -> None:
        """Initialize ambulance fleet from constants."""
        self._ambulances = {}
        for amb in AMBULANCES:
            self._ambulances[amb["id"]] = AmbulanceState(
                id=amb["id"],
                node_id=amb["base_node"],
                equipment=AmbulanceEquipment(amb["equipment"]),
                status=AmbulanceStatus.AVAILABLE,
                route=[],
                eta_minutes=0,
            )

    def _init_blood_banks(self) -> None:
        """Initialize blood banks from hospital states."""
        self._blood_banks = {}
        for hosp_id, hosp in self._hospitals.items():
            self._blood_banks[hosp_id] = BloodBankState(
                hospital_id=hosp_id,
                stocks=dict(hosp.blood_stock),
                crossmatch_queue=[],
            )

    def _init_patients(self, task_id: str, rng) -> None:
        """Generate patients for the task from constants."""
        self._patients = []
        config = TASK_CONFIG[task_id]

        for i, pconfig in enumerate(config["patients"]):
            condition = PatientCondition(pconfig["condition"])
            tier = self._condition_to_tier(condition)

            # Choose a random non-hospital node for patient location
            non_hospital_nodes = [
                n["id"] for n in CITY_NODES if n["type"] != "hospital"
            ]
            location = rng.choice(non_hospital_nodes)

            patient = Patient(
                patient_id=f"P{i+1}",
                condition=condition,
                tier=tier,
                location_node=location,
                time_since_onset=0,
                status=PatientStatus.WAITING,
            )
            self._patients.append(patient)

    # =========================================================================
    # Time Advancement
    # =========================================================================

    def _advance_time(self) -> None:
        """Advance all systems by 1 minute."""
        # Patient deterioration
        for p in self._patients:
            if p.status not in (PatientStatus.TREATED, PatientStatus.DECEASED):
                p.time_since_onset += 1
                self._check_patient_outcome(p)

        # Ambulance movement
        for amb in self._ambulances.values():
            if amb.status in (AmbulanceStatus.DISPATCHED, AmbulanceStatus.TRANSPORTING):
                if amb.eta_minutes > 0:
                    amb.eta_minutes -= 1
                if amb.eta_minutes == 0:
                    self._arrive_at_destination(amb)

        # OR prep countdowns
        for hosp in self._hospitals.values():
            for or_index, mins_left in list(hosp.or_prep_countdowns.items()):
                if mins_left <= 1:
                    del hosp.or_prep_countdowns[or_index]
                    # OR is now idle
                    for or_obj in hosp.operating_rooms:
                        if or_obj.index == or_index:
                            or_obj.status = ORStatus.IDLE
                else:
                    hosp.or_prep_countdowns[or_index] = mins_left - 1

            # OR in-use countdowns
            for or_obj in hosp.operating_rooms:
                if or_obj.status == ORStatus.IN_USE and or_obj.minutes_remaining:
                    if or_obj.minutes_remaining <= 1:
                        or_obj.minutes_remaining = None
                        or_obj.status = ORStatus.IDLE
                        or_obj.procedure_type = None
                        or_obj.patient_id = None
                    else:
                        or_obj.minutes_remaining -= 1

        # Blood crossmatch queue
        for hosp_id, bb in self._blood_banks.items():
            for entry in list(bb.crossmatch_queue):
                entry["time_remaining"] -= 1
                if entry["time_remaining"] <= 0:
                    # Reserve the blood
                    bb.stocks[entry["blood_type"]] -= entry["units"]
                    self._alerts.append(
                        f"Crossmatch complete: {entry['units']} units "
                        f"{entry['blood_type']} reserved for "
                        f"{entry['patient_id']} at {hosp_id}"
                    )

    def _check_patient_outcome(self, patient: Patient) -> None:
        """Check if a patient has exceeded their survival window."""
        target = PATIENT_TARGET_TIMES[patient.condition.value]
        if patient.time_since_onset >= target + 15:  # 15 min grace after strict window
            patient.status = PatientStatus.DECEASED
            patient.outcome = "deceased"

    def _arrive_at_destination(self, amb: AmbulanceState) -> None:
        """Handle ambulance arriving at its destination."""
        if amb.status == AmbulanceStatus.DISPATCHED:
            # Arrived at patient pickup location
            patient = next(
                (p for p in self._patients if p.patient_id == amb.assigned_patient),
                None,
            )
            if patient and patient.status == PatientStatus.WAITING:
                patient.status = PatientStatus.TRANSPORTING
                amb.node_id = patient.location_node
                if patient.assigned_hospital:
                    hosp = self._hospitals[patient.assigned_hospital]
                    route = self._compute_route(amb.node_id, hosp.node_id)
                    amb.route = route
                    amb.eta_minutes = self._route_travel_time(route)
                    amb.destination_type = "hospital"
                else:
                    # No hospital assigned — stay here
                    amb.status = AmbulanceStatus.AVAILABLE
                    amb.assigned_patient = None

        elif amb.status == AmbulanceStatus.TRANSPORTING:
            # Arrived at hospital
            patient = next(
                (p for p in self._patients if p.patient_id == amb.assigned_patient),
                None,
            )
            if patient and patient.assigned_hospital:
                patient.status = PatientStatus.TREATING
                patient.treatment_start_time = self._state.step_count
                amb.status = AmbulanceStatus.AVAILABLE
                amb.assigned_patient = None
                amb.route = []

    # =========================================================================
    # Action Execution
    # =========================================================================

    def _execute_action(self, action: CodeRedAction) -> None:
        """Dispatch action to the appropriate handler."""
        # Dispatch on action type using Pydantic discriminator
        if isinstance(action, DispatchAmbulance):
            self._do_dispatch_ambulance(action)
        elif isinstance(action, PrepareOR):
            self._do_prepare_or(action)
        elif isinstance(action, PageSpecialist):
            self._do_page_specialist(action)
        elif isinstance(action, AssignHospital):
            self._do_assign_hospital(action)
        elif isinstance(action, PreemptOR):
            self._do_preempt_or(action)
        elif isinstance(action, AllocateBlood):
            self._do_allocate_blood(action)
        elif isinstance(action, TransferBlood):
            self._do_transfer_blood(action)
        elif isinstance(action, RequestMutualAid):
            self._do_request_mutual_aid(action)
        elif isinstance(action, QueryBloodType):
            self._do_query_blood_type(action)
        elif isinstance(action, QueryORStatus):
            self._do_query_or_status(action)
        elif isinstance(action, MaintainPlan):
            pass  # No-op

    def _do_dispatch_ambulance(self, action: DispatchAmbulance) -> None:
        amb = self._ambulances.get(action.ambulance_id)
        if amb is None:
            self._alerts.append(f"Dispatch failed: ambulance {action.ambulance_id} not found")
            return
        if amb.status != AmbulanceStatus.AVAILABLE:
            self._alerts.append(
                f"Dispatch failed: ambulance {action.ambulance_id} is {amb.status.value}"
            )
            return

        amb.status = AmbulanceStatus.DISPATCHED
        amb.assigned_patient = None
        route = self._compute_route(amb.node_id, action.target_node)
        amb.route = route
        amb.eta_minutes = self._route_travel_time(route)
        amb.destination_type = "patient"

        # Check if a waiting patient is at this node
        waiting_patient = next(
            (p for p in self._patients
             if p.status == PatientStatus.WAITING
             and p.location_node == action.target_node),
            None,
        )
        if waiting_patient:
            amb.assigned_patient = waiting_patient.patient_id
            waiting_patient.assigned_ambulance = action.ambulance_id
            waiting_patient.status = PatientStatus.DISPATCHED

    def _do_prepare_or(self, action: PrepareOR) -> None:
        hosp = self._hospitals.get(action.hospital_id)
        if hosp is None:
            self._alerts.append(f"PrepareOR failed: hospital {action.hospital_id} not found")
            return
        if action.procedure_type not in hosp.capabilities:
            self._alerts.append(
                f"PrepareOR failed: {action.hospital_id} does not have "
                f"{action.procedure_type} capability"
            )
            return
        # Find an idle OR
        idle_or = next((or_obj for or_obj in hosp.operating_rooms if or_obj.status == ORStatus.IDLE), None)
        if idle_or is None:
            self._alerts.append(f"PrepareOR failed: no idle OR at {action.hospital_id}")
            return
        idle_or.status = ORStatus.PREP
        hosp.or_prep_countdowns[idle_or.index] = 10  # 10-minute prep

    def _do_page_specialist(self, action: PageSpecialist) -> None:
        hosp = self._hospitals.get(action.hospital_id)
        if hosp is None:
            self._alerts.append(f"PageSpecialist failed: hospital {action.hospital_id} not found")
            return
        spec = hosp.specialists.get(action.specialist_type)
        if spec is None or spec.total == 0:
            self._alerts.append(
                f"PageSpecialist failed: no {action.specialist_type} at {action.hospital_id}"
            )
            return
        if spec.available <= 0:
            self._alerts.append(
                f"PageSpecialist failed: no {action.specialist_type} available at {action.hospital_id}"
            )
            return
        spec.available -= 1
        spec.status = "paged"
        spec.minutes_until_available = 8  # 8-minute page latency

    def _do_assign_hospital(self, action: AssignHospital) -> None:
        patient = next((p for p in self._patients if p.patient_id == action.patient_id), None)
        if patient is None:
            self._alerts.append(f"AssignHospital failed: patient {action.patient_id} not found")
            return
        hosp = self._hospitals.get(action.hospital_id)
        if hosp is None:
            self._alerts.append(f"AssignHospital failed: hospital {action.hospital_id} not found")
            return
        if hosp.on_diversion:
            self._alerts.append(f"AssignHospital failed: {action.hospital_id} is on diversion")
            return
        # Check capability
        required = PATIENT_CONDITION_REQUIREMENTS[patient.condition.value][0]
        if required not in hosp.capabilities:
            self._alerts.append(
                f"AssignHospital failed: {action.hospital_id} cannot treat "
                f"{patient.condition.value} (needs {required})"
            )
            return
        patient.assigned_hospital = action.hospital_id

    def _do_preempt_or(self, action: PreemptOR) -> None:
        hosp = self._hospitals.get(action.hospital_id)
        if hosp is None:
            self._alerts.append(f"PreemptOR failed: hospital {action.hospital_id} not found")
            return
        or_obj = next(
            (or_o for or_o in hosp.operating_rooms if or_o.index == action.or_index),
            None,
        )
        if or_obj is None:
            self._alerts.append(f"PreemptOR failed: OR {action.or_index} not found at {action.hospital_id}")
            return
        if or_obj.status == ORStatus.IDLE:
            self._alerts.append(f"PreemptOR failed: OR {action.or_index} is already idle")
            return
        harm = (or_obj.minutes_remaining or 0) / 30.0
        harm = min(1.0, harm)
        recovery_time = or_obj.minutes_remaining or 0
        or_obj.status = ORStatus.IDLE
        or_obj.procedure_type = None
        or_obj.minutes_remaining = None
        or_obj.patient_id = None
        self._alerts.append(
            f"OR preempted at {action.hospital_id} OR {action.or_index}: "
            f"harm={harm:.2f}, recovery={recovery_time}min"
        )

    def _do_allocate_blood(self, action: AllocateBlood) -> None:
        hosp = self._hospitals.get(action.hospital_id)
        if hosp is None:
            self._alerts.append(f"AllocateBlood failed: hospital {action.hospital_id} not found")
            return
        bb = self._blood_banks.get(action.hospital_id)
        if bb is None:
            return
        if action.emergency:
            # Instant O_NEG release
            if bb.stocks.get("O_NEG", 0) < action.units:
                self._alerts.append(
                    f"AllocateBlood failed: insufficient O_NEG at {action.hospital_id}"
                )
                return
            bb.stocks["O_NEG"] -= action.units
            self._alerts.append(
                f"Emergency blood: {action.units} units O_NEG released for {action.patient_id} "
                f"at {action.hospital_id}"
            )
        else:
            # Crossmatch (15 min delay)
            if bb.stocks.get(action.blood_type, 0) < action.units:
                self._alerts.append(
                    f"AllocateBlood failed: insufficient {action.blood_type} at {action.hospital_id}"
                )
                return
            bb.crossmatch_queue.append({
                "patient_id": action.patient_id,
                "blood_type": action.blood_type,
                "units": action.units,
                "time_remaining": 15,
            })

    def _do_transfer_blood(self, action: TransferBlood) -> None:
        bb_from = self._blood_banks.get(action.from_hospital)
        bb_to = self._blood_banks.get(action.to_hospital)
        if bb_from is None or bb_to is None:
            self._alerts.append("TransferBlood failed: hospital not found")
            return
        if bb_from.stocks.get(action.blood_type, 0) < action.units:
            self._alerts.append(
                f"TransferBlood failed: insufficient {action.blood_type} at {action.from_hospital}"
            )
            return
        # Deduct immediately; in a full impl, this would be in-transit state
        bb_from.stocks[action.blood_type] -= action.units
        bb_to.stocks[action.blood_type] = bb_to.stocks.get(action.blood_type, 0) + action.units
        self._alerts.append(
            f"Blood transfer: {action.units} units {action.blood_type} "
            f"from {action.from_hospital} to {action.to_hospital}"
        )

    def _do_request_mutual_aid(self, action: RequestMutualAid) -> None:
        if self._state.mutual_aid_available <= 0:
            self._alerts.append("Mutual aid failed: no calls remaining")
            return
        self._state.mutual_aid_available -= 1
        self._state.mutual_aid_used += 1
        # Add new BLS ambulance at random location
        import random
        non_hospital = [n["id"] for n in CITY_NODES if n["type"] != "hospital"]
        new_id = f"MUTUAL_{self._state.mutual_aid_used}"
        self._ambulances[new_id] = AmbulanceState(
            id=new_id,
            node_id=random.choice(non_hospital),
            equipment=AmbulanceEquipment.BLS,
            status=AmbulanceStatus.AVAILABLE,
        )
        self._alerts.append(
            f"Mutual aid requested: {new_id} arriving in 12 min (available now as reserve)"
        )

    def _do_query_blood_type(self, action: QueryBloodType) -> None:
        patient = next((p for p in self._patients if p.patient_id == action.patient_id), None)
        if patient is None:
            self._alerts.append(f"QueryBloodType failed: patient {action.patient_id} not found")
            return
        # Simulate: blood type becomes known after 5 min
        # For simplicity in Plan 1, reveal immediately (full impl in Plan 2)
        blood_types = list(BLOOD_TYPES)
        import random
        patient.blood_type = random.choice(blood_types)
        self._alerts.append(
            f"Blood type revealed for {action.patient_id}: {patient.blood_type}"
        )

    def _do_query_or_status(self, action: QueryORStatus) -> None:
        hosp = self._hospitals.get(action.hospital_id)
        if hosp is None:
            return
        # In Plan 1, OR detail is always visible; full query in Plan 2
        self._alerts.append(
            f"OR status at {action.hospital_id}: this action costs 1 step (Plan 2)"
        )

    # =========================================================================
    # Routing
    # =========================================================================

    def _compute_route(self, from_node: str, to_node: str) -> List[str]:
        """Simple BFS shortest path. Returns list of node IDs including start and end."""
        import heapq
        if from_node == to_node:
            return [from_node]
        # Dijkstra on road network
        pq = [(0, from_node, [from_node])]
        visited = set()
        while pq:
            cost, node, path = heapq.heappop(pq)
            if node in visited:
                continue
            visited.add(node)
            if node == to_node:
                return path
            for edge_key, edge in self._road_network.edges.items():
                if edge.disrupted and edge.disruption_type == "road_closure":
                    continue
                next_node = None
                if edge.from_node == node:
                    next_node = edge.to_node
                elif edge.to_node == node:
                    next_node = edge.from_node
                if next_node and next_node not in visited:
                    new_cost = cost + edge.effective_time()
                    heapq.heappush(pq, (new_cost, next_node, path + [next_node]))
        return [from_node]  # No path found

    def _route_travel_time(self, route: List[str]) -> int:
        """Compute total travel time for a route."""
        total = 0
        for i in range(len(route) - 1):
            t = self._road_network.get_travel_time(route[i], route[i + 1])
            if t == float("inf"):
                return 999
            total += int(t)
        return total

    # =========================================================================
    # Termination & Reward
    # =========================================================================

    def _check_done(self) -> bool:
        """Check if episode should terminate."""
        if self._state.step_count >= self._state.max_steps:
            return True
        non_terminal = [
            p for p in self._patients
            if p.status not in (PatientStatus.TREATED, PatientStatus.DECEASED)
        ]
        if len(non_terminal) == 0:
            self._state.all_patients_terminal = True
            return True
        return False

    def _compute_step_reward(self) -> float:
        """Dense reward based on time_score_preview delta."""
        prev = self._compute_time_score_preview()
        # Small step cost
        return -0.01

    def _compute_time_score_preview(self) -> float:
        """Running estimate of time_score axis."""
        scores = []
        for p in self._patients:
            if p.status == PatientStatus.TREATED and p.treatment_complete_time is not None:
                target = PATIENT_TARGET_TIMES[p.condition.value]
                actual = p.treatment_complete_time
                score = max(0.0, min(1.0, 1.0 - (actual - target) / target))
                scores.append(score)
            elif p.status == PatientStatus.DECEASED:
                scores.append(0.0)
            else:
                # Projected
                target = PATIENT_TARGET_TIMES[p.condition.value]
                projected = p.time_since_onset + 5 + 10 + 10  # dispatch + travel + prep
                score = max(0.0, min(1.0, 1.0 - (projected - target) / target))
                scores.append(score)
        return sum(scores) / len(scores) if scores else 1.0

    # =========================================================================
    # Observation
    # =========================================================================

    def _build_observation(self) -> CodeRedObservation:
        """Build the current observation."""
        self._state.cum_reward = round(self._state.cum_reward, 4)
        return CodeRedObservation(
            step=self._state.step_count + 1,  # 1-indexed for agent
            patients=list(self._patients),
            ambulances=list(self._ambulances.values()),
            hospitals=list(self._hospitals.values()),
            blood_banks=list(self._blood_banks.values()),
            road_network=self._road_network,
            alerts=list(self._alerts),
            mutual_aid_remaining=self._state.mutual_aid_available,
            time_score_preview=round(self._compute_time_score_preview(), 4),
            patients_remaining=len([
                p for p in self._patients
                if p.status not in (PatientStatus.TREATED, PatientStatus.DECEASED)
            ]),
        )

    # =========================================================================
    # Helpers
    # =========================================================================

    def _condition_to_tier(self, condition: PatientCondition) -> PatientTier:
        mapping = {
            PatientCondition.CARDIAC: PatientTier.CRITICAL,
            PatientCondition.STROKE: PatientTier.CRITICAL,
            PatientCondition.TRAUMA: PatientTier.HIGH,
            PatientCondition.GENERAL: PatientTier.MEDIUM,
        }
        return mapping[condition]
```

- [ ] **Step 4: Write minimal `app.py`**

```python
"""FastAPI application for CodeRedEnv."""

try:
    from openenv.core.env_server.http_server import create_app
except ImportError as e:
    raise ImportError(
        "openenv is required. Install with: uv sync"
    ) from e

from codered_env.server.codered_environment import CodeRedEnvironment
from codered_env.server.models import CodeRedAction, CodeRedObservation

app = create_app(
    CodeRedEnvironment,
    CodeRedAction,
    CodeRedObservation,
    env_name="codered_env",
    max_concurrent_envs=4,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest codered_env/tests/test_environment.py -v`
Expected: PASS (all 7 tests)

- [ ] **Step 6: Verify server starts**

Run: `cd codered_env && python -c "from server.codered_environment import CodeRedEnvironment; e = CodeRedEnvironment(); o = e.reset(seed=0, task_id='task1'); print(f'Patients: {len(o.patients)}, Ambulances: {len(o.ambulances)}, Hospitals: {len(o.hospitals)}')"`
Expected: `Patients: 1, Ambulances: 5, Hospitals: 3`

- [ ] **Step 7: Verify OpenEnv integration**

Run: `cd codered_env && python -c "from server.app import app; print('FastAPI app created successfully')"`
Expected: `FastAPI app created successfully`

---

## Task 5: OpenEnv Validation

**Files:**
- Modify: `codered_env/openenv.yaml` (update with proper metadata)
- Test: `codered_env/tests/test_openenv_validate.py`

- [ ] **Step 1: Update `openenv.yaml` with proper metadata**

```yaml
spec_version: 1
name: codered_env
type: space
runtime: fastapi
app: server.app:app
port: 8000
version: "0.1.0"
description: "CodeRedEnv — Emergency Medical Coordination Environment for OpenEnv. Simulates ambulance dispatch, hospital resource preparation, and multi-patient triage in Prakashnagar."
```

- [ ] **Step 2: Write validation test**

```python
# codered_env/tests/test_openenv_validate.py
def test_openenv_yaml_valid():
    """Verify openenv.yaml is parseable and has required fields."""
    import yaml
    with open("openenv.yaml") as f:
        data = yaml.safe_load(f)
    assert data["spec_version"] == 1
    assert data["name"] == "codered_env"
    assert data["app"] == "server.app:app"
    assert data["port"] == 8000

def test_environment_subclass():
    """Verify CodeRedEnvironment is a valid OpenEnv Environment."""
    from openenv.core.env_server.interfaces import Environment
    from codered_env.server.codered_environment import CodeRedEnvironment
    assert issubclass(CodeRedEnvironment, Environment)

def test_all_action_types_importable():
    """All action types can be imported."""
    from codered_env.server.models.actions import (
        DispatchAmbulance, PrepareOR, PageSpecialist, AssignHospital,
        PreemptOR, AllocateBlood, TransferBlood, RequestMutualAid,
        QueryBloodType, QueryORStatus, MaintainPlan,
    )
    # Smoke test instantiation
    DispatchAmbulance(ambulance_id="AMB_1", target_node="NH45_BYPASS")
    MaintainPlan()
    RequestMutualAid()
```

- [ ] **Step 3: Run validation tests**

Run: `pytest codered_env/tests/test_openenv_validate.py -v`
Expected: PASS

---
