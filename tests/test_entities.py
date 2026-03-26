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
    from codered_env.server.models.entities import HospitalState, OperatingRoom
    from codered_env.server.models.entities import SpecialistStatus
    h = HospitalState(
        id="HOSP_A",
        node_id="AIIMS_PRAKASH",
        capabilities=["cardiac"],
        specialists={"cardiologist": SpecialistStatus(available=2, total=2)},
        operating_rooms=[OperatingRoom(index=i) for i in range(3)],
        icu_beds={"total": 4, "available": 4},
        blood_stock={"O_POS": 10},
    )
    assert len(h.operating_rooms) == 3  # Default 3 ORs

def test_ambulance_state_defaults():
    from codered_env.server.models.entities import AmbulanceState
    a = AmbulanceState(id="AMB_1", node_id="RAILWAY_XING", equipment="ALS")
    assert a.status == "available"
    assert a.route == []

def test_road_network_state_has_edges():
    from codered_env.server.models.entities import RoadNetworkState, EdgeState
    r = RoadNetworkState()
    r.edges["RAILWAY_XING->NH45_BYPASS"] = EdgeState(
        from_node="RAILWAY_XING",
        to_node="NH45_BYPASS",
        base_time=6,
    )
    assert len(r.edges) > 0
    assert r.edges["RAILWAY_XING->NH45_BYPASS"].base_time == 6
