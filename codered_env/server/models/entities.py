"""Pydantic entity models for CodeRedEnv."""

from enum import Enum
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from ..subsystems.constants import DispatchCategory


class PatientStatus(str, Enum):
    WAITING = "waiting"
    DETERIORATING = "deteriorating"  # Phase 1
    CRITICAL = "critical"           # Phase 1
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
    # Phase 2: Additional conditions from dispatch calls
    ANXIETY = "anxiety"
    GERD = "gerd"
    PANIC = "panic"
    HYPOGLYCEMIA = "hypoglycemia"
    INTOXICATION = "intoxication"
    SEIZURE = "seizure"
    FRACTURE = "fracture"
    SYNCOPE = "syncope"
    MINOR = "minor"
    RESPIRATORY = "respiratory"
    ASTHMA = "asthma"
    DEHYDRATION = "dehydration"
    VIRAL = "viral"


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
    vitals_score: float = Field(default=1.0, ge=0.0, le=1.0)
    blood_type: Optional[str] = None  # None = unknown until QueryBloodType
    treatment_start_time: Optional[int] = None
    treatment_complete_time: Optional[int] = None
    outcome: Optional[Literal["saved", "deceased"]] = None
    is_secondary: bool = False
    icu_status: Optional[str] = None  # "admitted" | "boarding" | None

    # Phase 2: Dispatch cascade fields
    dispatch_call_id: Optional[str] = None  # links patient to originating call
    cascade_trigger_reason: Optional[str] = None  # "witnessed_death", "news_cycle"
    observed_condition: Optional[str] = None  # shown pre-reveal (None = hidden)

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
    blood_stock: Dict[str, int]
    on_diversion: bool = False
    or_prep_countdowns: Dict[int, int] = Field(default_factory=dict)


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
    edges: Dict[str, EdgeState] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}

    def get_edge_key(self, from_node: str, to_node: str) -> str:
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
    stocks: Dict[str, int]
    crossmatch_queue: List[Dict] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


class DispatchCall(BaseModel):
    """A 911 call awaiting a triage decision. True condition is hidden until on-scene."""
    call_id: str
    category: DispatchCategory
    location_node: str
    time_waiting: int = 0  # steps since call arrived
    estimated_severity: float = 0.1  # 0.0-1.0, escalates with time_waiting
    spawned_patient_id: Optional[str] = None  # set when patient spawns on-scene

    model_config = {"extra": "forbid"}


class DispatchOutcome(BaseModel):
    """Result of a triage decision on a dispatch call. Ground truth revealed later."""
    call_id: str
    decision: str  # "als", "bls", "self_transport", "callback", "no_dispatch"
    category: str  # dispatch category reported
    true_condition: Optional[str] = None  # revealed on-scene arrival
    als_needed: bool = False  # ground truth: was ALS actually required?
    revealed_at_step: Optional[int] = None

    model_config = {"extra": "forbid"}
