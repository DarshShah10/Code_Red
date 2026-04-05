from server.subsystems.ambulance_manager import AmbulanceManager
from server.subsystems.road_network import RoadNetwork

def test_ambulance_manager_created_from_constants():
    from server.subsystems.constants import AMBULANCES
    am = AmbulanceManager(AMBULANCES)
    assert len(am._ambulances) == 5
    assert "AMB_1" in am._ambulances
    assert am._ambulances["AMB_1"].status == "available"

def test_dispatch_sets_route():
    from server.subsystems.constants import AMBULANCES
    rn = RoadNetwork()
    am = AmbulanceManager(AMBULANCES)
    result = am.dispatch("AMB_1", "NH45_BYPASS", road_network=rn)
    assert result["success"] is True
    amb = am._ambulances["AMB_1"]
    assert amb.status == "en_route"
    assert amb.target_node == "NH45_BYPASS"

def test_dispatch_fails_when_unavailable():
    from server.subsystems.constants import AMBULANCES
    rn = RoadNetwork()
    am = AmbulanceManager(AMBULANCES)
    am.dispatch("AMB_1", "NH45_BYPASS", road_network=rn)
    result = am.dispatch("AMB_1", "RAILWAY_XING", road_network=rn)
    assert result["success"] is False

def test_tick_decrements_eta():
    from server.subsystems.constants import AMBULANCES
    rn = RoadNetwork()
    am = AmbulanceManager(AMBULANCES)
    am.dispatch("AMB_1", "NH45_BYPASS", road_network=rn)
    amb = am._ambulances["AMB_1"]
    initial_eta = amb.eta_minutes
    am.tick()
    assert amb.eta_minutes == initial_eta - 1

def test_arrival_when_eta_hits_zero():
    from server.subsystems.constants import AMBULANCES
    rn = RoadNetwork()
    am = AmbulanceManager(AMBULANCES)
    am.dispatch("AMB_1", "NH45_BYPASS", road_network=rn)
    amb = am._ambulances["AMB_1"]
    for _ in range(amb.eta_minutes):
        am.tick()
    am.tick()  # one more step triggers auto-arrival
    assert amb.status == "on_scene"

def test_get_available_als():
    from server.subsystems.constants import AMBULANCES
    am = AmbulanceManager(AMBULANCES)
    available = am.get_available(equipment="ALS")
    assert len(available) == 2  # AMB_1 and AMB_2


# =============================================================================
# Scene time tests — Task 16
# =============================================================================

def test_dispatch_with_patient_adds_scene_time():
    """Dispatching with patient_id should add SCENE_TIME to ETA."""
    from server.subsystems.constants import AMBULANCES, SCENE_TIME
    rn = RoadNetwork()
    am = AmbulanceManager(AMBULANCES)
    # AMB_1 base = RAILWAY_XING, route to NH45_BYPASS = 6 min (with congestion)
    route = rn.shortest_path("RAILWAY_XING", "NH45_BYPASS")
    travel_time = rn.route_travel_time(route)
    result = am.dispatch("AMB_1", "NH45_BYPASS", road_network=rn, patient_id="P1")
    assert result["success"] is True
    amb = am._ambulances["AMB_1"]
    assert amb.eta_minutes == travel_time + SCENE_TIME
    assert amb.patient_id == "P1"


def test_scene_countdown_then_auto_return():
    """After arriving on-scene with patient, countdown fires and auto-returns."""
    from server.subsystems.constants import AMBULANCES, SCENE_TIME
    rn = RoadNetwork()
    am = AmbulanceManager(AMBULANCES)
    # Dispatch with patient, forcing arrival
    am.dispatch("AMB_1", "RAILWAY_XING", road_network=rn, patient_id="P1")
    amb = am._ambulances["AMB_1"]
    # Fast-forward to arrival
    while amb.status == "en_route":
        am.tick()
    # Should be on_scene with scene_minutes_remaining
    assert amb.status == "on_scene"
    assert amb.scene_minutes_remaining == SCENE_TIME
    # Tick SCENE_TIME more times → should auto-return
    for _ in range(SCENE_TIME - 1):
        am.tick()
    assert amb.status == "on_scene"
    am.tick()  # last tick
    assert amb.status == "returning"
    assert amb.target_node == amb.base_node


def test_mark_available_resets_scene_minutes():
    """mark_available should reset scene_minutes_remaining."""
    from server.subsystems.constants import AMBULANCES
    rn = RoadNetwork()
    am = AmbulanceManager(AMBULANCES)
    am.dispatch("AMB_1", "NH45_BYPASS", road_network=rn, patient_id="P1")
    amb = am._ambulances["AMB_1"]
    # Force arrival
    while amb.status == "en_route":
        am.tick()
    am.tick()  # arrive
    assert amb.scene_minutes_remaining > 0
    am.mark_available("AMB_1")
    assert amb.scene_minutes_remaining == 0
    assert amb.status == "available"
