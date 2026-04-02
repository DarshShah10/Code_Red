"""Typed action models for CodeRedEnv."""

from typing import Literal, Optional

from pydantic import Field

from openenv.core.env_server.types import Action


class DispatchAmbulance(Action):
    """Dispatch an ambulance to a target node."""
    ambulance_id: str = Field(..., description="Ambulance ID to dispatch")
    target_node: str = Field(..., description="Target node ID")


class DispatchALS(Action):
    """Dispatch an ALS ambulance to a pending dispatch call. Commits ALS resource."""
    ambulance_id: str = Field(..., description="ALS ambulance ID to dispatch")
    call_id: str = Field(..., description="Dispatch call ID to respond to")


class DispatchBLS(Action):
    """Dispatch a BLS ambulance to a pending dispatch call."""
    ambulance_id: str = Field(..., description="BLS ambulance ID to dispatch")
    call_id: str = Field(..., description="Dispatch call ID to respond to")


class TriageCall(Action):
    """Decide what to do with a pending dispatch call."""
    call_id: str = Field(..., description="Dispatch call ID to triage")
    decision: Literal["dispatch_als", "dispatch_bls", "self_transport", "callback", "no_dispatch"] = Field(
        ...,
        description="Triage decision"
    )
    ambulance_id: Optional[str] = Field(
        default=None,
        description="Ambulance ID (required when decision is dispatch_als or dispatch_bls)"
    )


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
    pass


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
    DispatchAmbulance    # kept for backward compat with tasks 1-3
    | DispatchALS
    | DispatchBLS
    | TriageCall
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
