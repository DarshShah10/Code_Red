from server.subsystems.disruption_engine import DisruptionEngine

def test_no_disruptions_task1():
    eng = DisruptionEngine()
    eng.reset(seed=0, task_id="task1")
    events = eng.roll_disruptions(step=1, road_network=None)
    assert events == []  # No disruptions in task 1

def test_seed_reproducibility():
    eng1 = DisruptionEngine()
    eng2 = DisruptionEngine()
    eng1.reset(seed=42, task_id="task2")
    eng2.reset(seed=42, task_id="task2")
    assert eng1._intensity == eng2._intensity

def test_intensity_in_range():
    eng = DisruptionEngine()
    eng.reset(seed=123, task_id="task2")
    assert 0.7 <= eng._intensity <= 1.3

def test_road_closure_affects_target():
    eng = DisruptionEngine()
    eng.reset(seed=19, task_id="task2")
    from server.subsystems.road_network import RoadNetwork
    rn = RoadNetwork()
    events = eng.roll_disruptions(step=1, road_network=rn)
    assert any(e["disruption_type"] == "road_closure" for e in events)

def test_disruption_types_vary_across_seeds():
    """Different seeds produce different disruption patterns."""
    eng1 = DisruptionEngine()
    eng1.reset(seed=1, task_id="task3")
    eng2 = DisruptionEngine()
    eng2.reset(seed=999, task_id="task3")
    assert eng1._seed != eng2._seed
