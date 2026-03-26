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
