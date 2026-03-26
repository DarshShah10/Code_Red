"""Typed observation and state models for CodeRedEnv."""

from typing import List

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
