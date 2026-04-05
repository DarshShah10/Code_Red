"""Typed action models for CodeRedEnv — OpenEnv spec compliant."""

from typing import Annotated, Literal, Union

from pydantic import Field

from openenv.core.env_server.types import Action


class DispatchAmbulance(Action):
    """Dispatch an ambulance to a target node (Phase 1)."""
    type: Literal["dispatch_ambulance"] = "dispatch_ambulance"
    ambulance_id: str = Field(..., description="Ambulance ID to dispatch")
    target_node: str = Field(..., description="Target node ID")


class DispatchALS(Action):
    """Dispatch an ALS ambulance to a pending 911 call (Phase 2)."""
    type: Literal["dispatch_als"] = "dispatch_als"
    ambulance_id: str = Field(..., description="ALS ambulance ID to dispatch")
    call_id: str = Field(..., description="Dispatch call ID to respond to")


class DispatchBLS(Action):
    """Dispatch a BLS ambulance to a pending 911 call (Phase 2)."""
    type: Literal["dispatch_bls"] = "dispatch_bls"
    ambulance_id: str = Field(..., description="BLS ambulance ID to dispatch")
    call_id: str = Field(..., description="Dispatch call ID to respond to")


class TriageCall(Action):
    """Decide what to do with a pending dispatch call."""
    type: Literal["triage_call"] = "triage_call"
    call_id: str = Field(..., description="Dispatch call ID to triage")
    decision: Literal["dispatch_als", "dispatch_bls", "self_transport", "callback", "no_dispatch"] = Field(
        ..., description="Triage decision"
    )
    ambulance_id: str | None = Field(
        default=None,
        description="Required when decision is dispatch_als or dispatch_bls",
    )


class PrepareOR(Action):
    """Begin OR preparation at a hospital for a procedure type."""
    type: Literal["prepare_or"] = "prepare_or"
    hospital_id: str = Field(..., description="Hospital ID")
    procedure_type: Literal["cardiac", "stroke", "trauma", "general"] = Field(
        ..., description="Type of procedure being prepared for"
    )


class PageSpecialist(Action):
    """Page a specialist at a hospital."""
    type: Literal["page_specialist"] = "page_specialist"
    hospital_id: str = Field(..., description="Hospital ID")
    specialist_type: Literal["cardiologist", "neurologist", "trauma_surgeon"] = Field(
        ..., description="Type of specialist to page"
    )


class AssignHospital(Action):
    """Assign a patient to a destination hospital."""
    type: Literal["assign_hospital"] = "assign_hospital"
    patient_id: str = Field(..., description="Patient ID")
    hospital_id: str = Field(..., description="Destination hospital ID")


class PreemptOR(Action):
    """Preempt (clear) an operating room for emergency use."""
    type: Literal["preempt_or"] = "preempt_or"
    hospital_id: str = Field(..., description="Hospital ID")
    or_index: int = Field(..., ge=0, description="OR index to preempt")


class AllocateBlood(Action):
    """Allocate blood units for a patient."""
    type: Literal["allocate_blood"] = "allocate_blood"
    hospital_id: str = Field(..., description="Hospital ID")
    patient_id: str = Field(..., description="Patient ID")
    blood_type: str = Field(..., description="Blood type to allocate")
    units: int = Field(..., ge=1, description="Number of units")
    emergency: bool = Field(
        default=False,
        description="If True, use emergency O_NEG release (instant)",
    )


class TransferBlood(Action):
    """Transfer blood units between hospitals."""
    type: Literal["transfer_blood"] = "transfer_blood"
    from_hospital: str = Field(..., description="Source hospital ID")
    to_hospital: str = Field(..., description="Destination hospital ID")
    blood_type: str = Field(..., description="Blood type")
    units: int = Field(..., ge=1, description="Number of units")


class RequestMutualAid(Action):
    """Request mutual aid ambulance (Task 2/3 only). Has 12-minute arrival latency."""
    type: Literal["request_mutual_aid"] = "request_mutual_aid"


class QueryBloodType(Action):
    """Query a patient's blood type. Takes 5 minutes to complete."""
    type: Literal["query_blood_type"] = "query_blood_type"
    patient_id: str = Field(..., description="Patient ID to test")


class QueryORStatus(Action):
    """Query detailed OR status. Takes 1 step to complete."""
    type: Literal["query_or_status"] = "query_or_status"
    hospital_id: str = Field(..., description="Hospital ID")
    or_index: int = Field(..., ge=0, description="OR index to query")


class MaintainPlan(Action):
    """No-op: continue current plan without changes."""
    type: Literal["maintain_plan"] = "maintain_plan"


# Union of all action types — used as action_cls in create_app()
# The `type` field on each class acts as a discriminator for OpenEnv's
# model_validate() and provides model_json_schema() for the /schema endpoint.
CodeRedAction = Union[
    DispatchAmbulance,
    DispatchALS,
    DispatchBLS,
    TriageCall,
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
]
