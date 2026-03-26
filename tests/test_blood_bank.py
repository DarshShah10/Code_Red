from codered_env.server.subsystems.blood_bank import BloodBankSystem

def test_hospital_C_has_limited_blood():
    bb = BloodBankSystem()
    hosp_c = bb.get("HOSP_C")
    assert hosp_c.stocks["O_POS"] == 4
    assert hosp_c.stocks["A_POS"] == 0

def test_emergency_release():
    bb = BloodBankSystem()
    result = bb.emergency_release("HOSP_A", "P1", "O_NEG", 2)
    assert result["success"] is True
    assert bb.get("HOSP_A").stocks["O_NEG"] == 4  # 6-2

def test_emergency_release_fails_when_insufficient():
    bb = BloodBankSystem()
    bb.get("HOSP_C").stocks["O_NEG"] = 1
    result = bb.emergency_release("HOSP_C", "P1", "O_NEG", 3)
    assert result["success"] is False

def test_crossmatch_queue():
    bb = BloodBankSystem()
    result = bb.start_crossmatch("HOSP_A", "P1", "A_POS", 2)
    assert result["success"] is True
    queue = bb.get("HOSP_A").crossmatch_queue
    assert len(queue) == 1
    assert queue[0]["time_remaining"] == 15

def test_crossmatch_tick():
    bb = BloodBankSystem()
    bb.start_crossmatch("HOSP_A", "P1", "A_POS", 2)
    bb.tick()
    queue = bb.get("HOSP_A").crossmatch_queue
    assert queue[0]["time_remaining"] == 14

def test_crossmatch_completes_and_reserves():
    bb = BloodBankSystem()
    hosp = bb.get("HOSP_A")
    initial_stock = hosp.stocks["A_POS"]
    bb.start_crossmatch("HOSP_A", "P1", "A_POS", 2)
    for _ in range(15):
        bb.tick()
    completed = bb.flush_completed_crossmatches()
    assert len(completed) == 1
    assert hosp.stocks["A_POS"] == initial_stock - 2  # Reserved

def test_transfer_blood():
    bb = BloodBankSystem()
    initial_a = bb.get("HOSP_A").stocks["O_POS"]
    initial_c = bb.get("HOSP_C").stocks["O_POS"]
    result = bb.transfer("HOSP_A", "HOSP_C", "O_POS", 3)
    assert result["success"] is True
    assert bb.get("HOSP_A").stocks["O_POS"] == initial_a - 3
    assert bb.get("HOSP_C").stocks["O_POS"] == initial_c + 3

def test_transfer_fails_insufficient():
    bb = BloodBankSystem()
    result = bb.transfer("HOSP_A", "HOSP_C", "O_POS", 999)
    assert result["success"] is False
