from codered_env.server.subsystems.road_network import RoadNetwork


def test_road_network_builds_from_constants():
    rn = RoadNetwork()
    assert len(rn.edges) == 15  # All 15 edges
    # Base time is 6; TOD congestion applied at init makes effective_time > base_time
    edge = rn._get_edge("RAILWAY_XING", "NH45_BYPASS")
    assert edge.base_time == 6


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


def test_tod_congestion_applied_at_init():
    """At step_count=0, episode starts at EPISODE_START_HOUR=8, NH45_BYPASS curve peak."""
    from codered_env.server.subsystems.constants import CONGESTION_CURVES
    rn = RoadNetwork()
    edge = rn._get_edge("NH45_BYPASS", "IT_HUB")
    curve = CONGESTION_CURVES.get("NH45_BYPASS", [])
    from codered_env.server.subsystems.constants import interpolate_congestion
    expected_mult = interpolate_congestion(curve, 8.0)
    assert edge.congestion_multiplier == expected_mult
    assert edge.congestion_multiplier > 1.0  # Peak hour → > 1.0


def test_tod_congestion_changes_over_time():
    """After 60 steps, hour increments and congestion multipliers update."""
    rn = RoadNetwork()
    edge = rn._get_edge("NH45_BYPASS", "IT_HUB")
    initial_mult = edge.congestion_multiplier
    rn.tick()  # step 1
    rn.tick()  # step 2
    # ... tick 58 more times to reach hour 9
    for _ in range(58):
        rn.tick()
    # At hour 9, NH45_BYPASS multiplier should differ from hour 8
    assert edge.congestion_multiplier != initial_mult


def test_disruption_overrides_tod_congestion():
    """Active disruption's congestion multiplier is not overwritten by TOD."""
    rn = RoadNetwork()
    edge = rn._get_edge("NH45_BYPASS", "IT_HUB")
    mult_before = edge.congestion_multiplier
    rn.set_disruption("NH45_BYPASS", "IT_HUB", "accident", remaining_steps=30)
    assert edge.disrupted is True
    # Accident sets multiplier to 3.0; subsequent ticks must not reset it
    assert edge.congestion_multiplier == 3.0
    rn.tick()
    assert edge.congestion_multiplier == 3.0  # Still 3.0, not TOD value
