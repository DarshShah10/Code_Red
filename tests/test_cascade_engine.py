from codered_env.server.subsystems.cascade_engine import CascadeEngine, CascadeRule


def test_engine_initializes_with_seed():
    """CascadeEngine creates a seeded RNG at reset()."""
    eng = CascadeEngine()
    eng.reset(seed=0, episode_config={})
    assert eng._rng is not None


def test_trauma_death_spawns_cardiac_with_probability():
    """A trauma death triggers secondary cardiac with 30% probability."""
    spawned = []
    def callback(event_type, **kwargs):
        if event_type == "spawn_secondary":
            spawned.append(kwargs)

    count = 0
    for i in range(100):
        eng2 = CascadeEngine()
        eng2.reset(seed=42 + i * 1000, episode_config={})
        eng2.set_callback(callback)
        eng2.on_outcome("P1", "trauma", "deceased", step=10)
        count += len(spawned)
        spawned.clear()

    # At 30% probability over 100 trials, expect at least some
    assert count > 5  # At least some should fire


def test_overcrowding_modifier_at_threshold():
    """Active patients > 3 returns 1.2 modifier."""
    eng = CascadeEngine()
    eng.reset(seed=0, episode_config={})
    modifier = eng.check_overcrowding(4)
    assert modifier == 1.2
    assert eng.overcrowding_modifier == 1.2


def test_overcrowding_modifier_below_threshold():
    """Active patients <= 3 returns 1.0 modifier."""
    eng = CascadeEngine()
    eng.reset(seed=0, episode_config={})
    modifier = eng.check_overcrowding(3)
    assert modifier == 1.0
    assert eng.overcrowding_modifier == 1.0


def test_overcrowding_callback_fires_on_first_crossing():
    """Callback fires when overcrowding first activates, not on subsequent ticks."""
    calls = []
    def callback(event_type, **kwargs):
        calls.append(event_type)

    eng = CascadeEngine()
    eng.reset(seed=0, episode_config={})
    eng.set_callback(callback)

    eng.check_overcrowding(4)  # First crossing → fires
    eng.check_overcrowding(5)  # Still overcrowded → no callback
    eng.check_overcrowding(6)  # Still overcrowded → no callback
    eng.check_overcrowding(2)  # Cleared

    assert calls == ["overcrowding_started"]


def test_tick_decrements_news_cycle():
    """tick() decrements news cycle counter."""
    eng = CascadeEngine()
    eng.reset(seed=0, episode_config={})
    eng._news_cycle_steps_remaining = 5

    eng.tick()
    assert eng.news_cycle_steps_remaining == 4

    eng.tick()
    eng.tick()
    assert eng.news_cycle_steps_remaining == 2


def test_surge_probability_resets_after_news_cycle():
    """After news cycle expires, surge probability decays by 0.10."""
    import math
    eng = CascadeEngine()
    eng.reset(seed=0, episode_config={})
    eng._pending_surge_probability = 0.30
    eng._news_cycle_steps_remaining = 1

    eng.tick()  # expiry tick
    assert eng.news_cycle_steps_remaining == 0
    assert math.isclose(eng.pending_surge_probability, 0.20, rel_tol=1e-9)  # 0.30 - 0.10


def test_outcome_ignores_wrong_condition():
    """Rules with condition_filter only fire for matching conditions."""
    eng = CascadeEngine()
    eng.reset(seed=0, episode_config={})
    events = []
    eng.set_callback(lambda e, **kw: events.append(e))

    # A "general" save should NOT trigger the cardiac news cycle rule
    eng.on_outcome("P1", "general", "saved", step=5)
    assert all(e != "news_cycle" for e in events)
