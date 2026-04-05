"""Regression tests for MutualAidManager subsystem."""
import pytest

from server.subsystems.mutual_aid import MutualAidManager, MAAmbulance, MAPending
from server.subsystems.constants import SCENE_TIME


class MockPatient:
    """Minimal patient mock for MutualAidManager testing."""
    def __init__(self, pid: str, condition: str, status: str = "waiting", location_node: str = "RAJIV_CHOWK"):
        self.id = pid
        self.condition = condition
        self.status = status
        self.assigned_hospital = None
        self.assigned_ambulance = None
        self.location_node = location_node


class MockHospitalSystem:
    def __init__(self):
        self._treat_map = {
            "HOSP_A": {"cardiac": True, "stroke": True, "trauma": True, "general": True},
            "HOSP_B": {"cardiac": True, "stroke": False, "trauma": True, "general": True},
            "HOSP_C": {"cardiac": False, "stroke": False, "trauma": False, "general": True},
        }

    def can_treat(self, hosp_id: str, condition: str) -> bool:
        return self._treat_map.get(hosp_id, {}).get(condition, False)


class MockPatientManager:
    def __init__(self, patients):
        self.patients = patients

    def get(self, patient_id: str):
        for p in self.patients:
            if p.id == patient_id:
                return p
        return None


class MockRoadNetwork:
    """Road network with travel_time=6 (used for init/request tests)."""
    def shortest_path(self, from_node: str, to_node: str) -> list:
        return [from_node, to_node]

    def route_travel_time(self, route: list) -> int:
        return 6  # fixed travel time


class MockRoadNetworkInstant:
    """Road network with travel_time=0 so arrivals fire at tick(step_count=SCENE_TIME)."""
    def shortest_path(self, from_node: str, to_node: str) -> list:
        return [from_node, to_node]

    def route_travel_time(self, route: list) -> int:
        return 0


class MockHospitalSystem2:
    def can_treat(self, hosp_id: str, condition: str) -> bool:
        return hosp_id in ("HOSP_A", "HOSP_B", "HOSP_C")


class TestMutualAidManagerInit:
    def test_available_calls_starts_correctly(self):
        ma = MutualAidManager(available_calls=2)
        assert ma.get_available() == 2
        assert ma.get_used() == 0

    def test_request_decrements_available(self):
        ma = MutualAidManager(available_calls=1)
        rn = MockRoadNetwork()
        patients = []
        ma_id = ma.request(step_count=0, road_network=rn, patients=patients)
        assert ma_id == "MUTUAL_1"
        assert ma.get_available() == 0
        assert ma.get_used() == 1

    def test_request_returns_none_when_no_calls(self):
        ma = MutualAidManager(available_calls=0)
        rn = MockRoadNetwork()
        patients = []
        ma_id = ma.request(step_count=0, road_network=rn, patients=patients)
        assert ma_id is None

    def test_multiple_requests_use_sequential_ids(self):
        ma = MutualAidManager(available_calls=3)
        rn = MockRoadNetwork()
        patients = []
        ids = [ma.request(0, rn, patients) for _ in range(3)]
        assert ids == ["MUTUAL_1", "MUTUAL_2", "MUTUAL_3"]
        assert ma.get_available() == 0


class TestMutualAidManagerAutoAssign:
    def test_auto_assigns_cardiac_over_stroke(self):
        ma = MutualAidManager(available_calls=1)
        rn = MockRoadNetwork()
        patients = [
            MockPatient("P2", "stroke"),
            MockPatient("P1", "cardiac"),
        ]
        ma_id = ma.request(step_count=0, road_network=rn, patients=patients)
        pending = ma.get_pending()[ma_id]
        assert pending.patient_id == "P1"  # cardiac is higher priority

    def test_auto_assigns_stroke_over_trauma(self):
        ma = MutualAidManager(available_calls=1)
        rn = MockRoadNetwork()
        patients = [
            MockPatient("P1", "trauma"),
            MockPatient("P2", "stroke"),
        ]
        ma_id = ma.request(step_count=0, road_network=rn, patients=patients)
        pending = ma.get_pending()[ma_id]
        assert pending.patient_id == "P2"  # stroke is higher priority

    def test_auto_assigns_no_waiting_patients(self):
        ma = MutualAidManager(available_calls=1)
        rn = MockRoadNetwork()
        patients = []  # no waiting patients
        ma_id = ma.request(step_count=0, road_network=rn, patients=patients)
        pending = ma.get_pending()[ma_id]
        assert pending.patient_id is None


class TestMutualAidManagerTick:
    def test_arrival_with_patient_auto_assigns_hospital(self):
        ma = MutualAidManager(available_calls=1)
        rn = MockRoadNetworkInstant()
        hosp_sys = MockHospitalSystem2()
        patients = [MockPatient("P1", "cardiac")]
        pm = MockPatientManager(patients)
        ma.request(step_count=0, road_network=rn, patients=patients)
        arrivals = ma.tick(
            step_count=SCENE_TIME,
            patient_manager=pm,
            hospital_system=hosp_sys,
            arrival_callback=lambda *_: None,
        )
        assert len(arrivals) == 1
        assert arrivals[0]["had_patient"] is True
        assert arrivals[0]["hospital_id"] == "HOSP_A"  # cardiac → HOSP_A
        assert patients[0].assigned_hospital == "HOSP_A"
        assert patients[0].status == "transporting"

    def test_arrival_without_patient(self):
        ma = MutualAidManager(available_calls=1)
        rn = MockRoadNetworkInstant()
        hosp_sys = MockHospitalSystem2()
        patients = []
        pm = MockPatientManager(patients)
        ma.request(step_count=0, road_network=rn, patients=patients)
        arrivals = ma.tick(
            step_count=SCENE_TIME,
            patient_manager=pm,
            hospital_system=hosp_sys,
            arrival_callback=lambda *_: None,
        )
        assert len(arrivals) == 1
        assert arrivals[0]["had_patient"] is False

    def test_tick_does_not_fire_before_arrival_step(self):
        ma = MutualAidManager(available_calls=1)
        rn = MockRoadNetworkInstant()
        hosp_sys = MockHospitalSystem2()
        patients = [MockPatient("P1", "cardiac")]
        pm = MockPatientManager(patients)
        ma.request(step_count=0, road_network=rn, patients=patients)
        arrivals = ma.tick(
            step_count=SCENE_TIME - 1,  # one step before arrival
            patient_manager=pm,
            hospital_system=hosp_sys,
            arrival_callback=lambda *_: None,
        )
        assert len(arrivals) == 0

    def test_auto_assign_falls_back_to_HOSP_B_for_trauma(self):
        ma = MutualAidManager(available_calls=1)
        rn = MockRoadNetworkInstant()
        hosp_sys = MockHospitalSystem2()
        patients = [MockPatient("P1", "trauma")]
        pm = MockPatientManager(patients)
        ma.request(step_count=0, road_network=rn, patients=patients)
        arrivals = ma.tick(
            step_count=SCENE_TIME,
            patient_manager=pm,
            hospital_system=hosp_sys,
            arrival_callback=lambda *_: None,
        )
        assert arrivals[0]["hospital_id"] == "HOSP_B"  # trauma → HOSP_B


class TestMutualAidManagerArrivalCallback:
    def test_callback_called_on_arrival(self):
        ma = MutualAidManager(available_calls=1)
        rn = MockRoadNetworkInstant()
        hosp_sys = MockHospitalSystem2()
        patients = [MockPatient("P1", "cardiac")]
        pm = MockPatientManager(patients)
        ma.request(step_count=0, road_network=rn, patients=patients)
        called = []
        def callback(ma_id, patient_id):
            called.append((ma_id, patient_id))
        ma.tick(step_count=SCENE_TIME, patient_manager=pm, hospital_system=hosp_sys, arrival_callback=callback)
        assert len(called) == 1
        assert called[0][0] == "MUTUAL_1"
        assert called[0][1] == "P1"
