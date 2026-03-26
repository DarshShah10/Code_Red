from codered_env.server.subsystems.ambulance_manager import AmbulanceManager
from codered_env.server.subsystems.road_network import RoadNetwork

def test_ambulance_manager_created_from_constants():
    from codered_env.server.subsystems.constants import AMBULANCES
    am = AmbulanceManager(AMBULANCES)
    assert len(am._ambulances) == 5
    assert "AMB_1" in am._ambulances
    assert am._ambulances["AMB_1"].status == "available"

def test_dispatch_sets_route():
    from codered_env.server.subsystems.constants import AMBULANCES
    rn = RoadNetwork()
    am = AmbulanceManager(AMBULANCES)
    result = am.dispatch("AMB_1", "NH45_BYPASS", road_network=rn)
    assert result["success"] is True
    amb = am._ambulances["AMB_1"]
    assert amb.status == "en_route"
    assert amb.target_node == "NH45_BYPASS"

def test_dispatch_fails_when_unavailable():
    from codered_env.server.subsystems.constants import AMBULANCES
    rn = RoadNetwork()
    am = AmbulanceManager(AMBULANCES)
    am.dispatch("AMB_1", "NH45_BYPASS", road_network=rn)
    result = am.dispatch("AMB_1", "RAILWAY_XING", road_network=rn)
    assert result["success"] is False

def test_tick_decrements_eta():
    from codered_env.server.subsystems.constants import AMBULANCES
    rn = RoadNetwork()
    am = AmbulanceManager(AMBULANCES)
    am.dispatch("AMB_1", "NH45_BYPASS", road_network=rn)
    amb = am._ambulances["AMB_1"]
    initial_eta = amb.eta_minutes
    am.tick()
    assert amb.eta_minutes == initial_eta - 1

def test_arrival_when_eta_hits_zero():
    from codered_env.server.subsystems.constants import AMBULANCES
    rn = RoadNetwork()
    am = AmbulanceManager(AMBULANCES)
    am.dispatch("AMB_1", "NH45_BYPASS", road_network=rn)
    amb = am._ambulances["AMB_1"]
    for _ in range(amb.eta_minutes):
        am.tick()
    am.tick()  # one more step triggers auto-arrival
    assert amb.status == "on_scene"

def test_get_available_als():
    from codered_env.server.subsystems.constants import AMBULANCES
    am = AmbulanceManager(AMBULANCES)
    available = am.get_available(equipment="ALS")
    assert len(available) == 2  # AMB_1 and AMB_2
