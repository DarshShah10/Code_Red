from codered_env.server.subsystems.hospital_system import HospitalSystem

def test_hospitals_initialized_from_constants():
    hs = HospitalSystem()
    hosp_a = hs.get("HOSP_A")
    assert hosp_a is not None
    assert hosp_a.capabilities == ["cardiac", "stroke", "trauma", "stabilization"]
    assert len(hosp_a.operating_rooms) == 3
    assert hosp_a.icu_beds["total"] == 4

def test_prepare_or_sets_prep_state():
    hs = HospitalSystem()
    result = hs.prepare_or("HOSP_A", "cardiac")
    assert result["success"] is True
    hosp_a = hs.get("HOSP_A")
    assert hosp_a.operating_rooms[0].status == "prep"
    assert hosp_a.or_prep_countdowns[0] == 10

def test_prepare_or_fails_for_unsupported_condition():
    hs = HospitalSystem()
    result = hs.prepare_or("HOSP_C", "cardiac")
    assert result["success"] is False
    assert "cannot treat" in result["reason"]

def test_prepare_or_fails_when_no_idle_or():
    hs = HospitalSystem()
    for i in range(3):
        hs.start_surgery("HOSP_A", i, "cardiac", "P1")
    result = hs.prepare_or("HOSP_A", "cardiac")
    assert result["success"] is False
    assert "no idle OR" in result["reason"]

def test_prepare_or_fails_when_on_diversion():
    hs = HospitalSystem()
    hs.set_diversion("HOSP_A", True)
    result = hs.prepare_or("HOSP_A", "cardiac")
    assert result["success"] is False
    assert "diversion" in result["reason"]
    hs.set_diversion("HOSP_A", False)

def test_tick_decrements_prep_countdown():
    hs = HospitalSystem()
    hs.prepare_or("HOSP_A", "cardiac")
    hosp_a = hs.get("HOSP_A")
    assert hosp_a.or_prep_countdowns[0] == 10
    hs.tick()
    assert hosp_a.or_prep_countdowns[0] == 9

def test_page_specialist():
    hs = HospitalSystem()
    hosp_a = hs.get("HOSP_A")
    initial_available = hosp_a.specialists["cardiologist"].available
    result = hs.page_specialist("HOSP_A", "cardiologist")
    assert result["success"] is True
    assert hosp_a.specialists["cardiologist"].available == initial_available - 1

def test_page_specialist_fails_when_none_available():
    hs = HospitalSystem()
    hosp_a = hs.get("HOSP_A")
    hosp_a.specialists["neurologist"].available = 0
    result = hs.page_specialist("HOSP_A", "neurologist")
    assert result["success"] is False

def test_preempt_or_returns_harm():
    hs = HospitalSystem()
    hosp_a = hs.get("HOSP_A")
    hosp_a.operating_rooms[0].status = "in_use"
    hosp_a.operating_rooms[0].minutes_remaining = 20
    result = hs.preempt_or("HOSP_A", 0)
    assert result["success"] is True
    assert result["harm"] == 20 / 30.0
    assert result["recovery_time"] == 20
    assert hosp_a.operating_rooms[0].status == "idle"

def test_preemption_with_zero_time_no_harm():
    hs = HospitalSystem()
    hosp_a = hs.get("HOSP_A")
    hosp_a.operating_rooms[0].status = "in_use"
    hosp_a.operating_rooms[0].minutes_remaining = 0
    result = hs.preempt_or("HOSP_A", 0)
    assert result["harm"] == 0.0

def test_tick_specialist_recovery():
    hs = HospitalSystem()
    hs.page_specialist("HOSP_A", "cardiologist")
    hosp_a = hs.get("HOSP_A")
    assert hosp_a.specialists["cardiologist"].minutes_until_available == 8
    hs.tick()
    assert hosp_a.specialists["cardiologist"].minutes_until_available == 7

def test_hospital_C_cannot_handle_cardiac():
    hs = HospitalSystem()
    result = hs.prepare_or("HOSP_C", "cardiac")
    assert result["success"] is False


def test_specialist_available_never_exceeds_total():
    """Specialist available count should never exceed total (recovery guard)."""
    hs = HospitalSystem()
    hosp_a = hs.get("HOSP_A")
    # Force all cardiologists to be paged (available=0)
    hosp_a.specialists["cardiologist"].available = 0
    hosp_a.specialists["cardiologist"].status = "paged"
    hosp_a.specialists["cardiologist"].minutes_until_available = 1
    # Tick: specialist recovers — available should go to 1, not 2
    hs.tick()
    assert hosp_a.specialists["cardiologist"].available == 1
    assert hosp_a.specialists["cardiologist"].status == "available"
