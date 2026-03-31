from codered_env.server.subsystems.constants import (
    CITY_NODES,
    CITY_EDGES,
    HOSPITALS,
    AMBULANCES,
    PATIENT_TARGET_TIMES,
    BLOOD_TYPES,
    HOSPITAL_MORTALITY_RATES,
)

def test_city_has_12_nodes():
    assert len(CITY_NODES) == 12

def test_city_edges_make_graph_connected():
    """All nodes must be reachable from each other."""
    import networkx as nx
    G = nx.Graph()
    for node in CITY_NODES:
        G.add_node(node["id"])
    for edge in CITY_EDGES:
        G.add_edge(edge["from"], edge["to"])
    assert nx.is_connected(G), "City graph is not fully connected"

def test_hospital_a_can_handle_cardiac():
    hosp_a = next(h for h in HOSPITALS if h["id"] == "HOSP_A")
    assert "cardiac" in hosp_a["capabilities"]

def test_hospital_c_cannot_handle_cardiac():
    hosp_c = next(h for h in HOSPITALS if h["id"] == "HOSP_C")
    assert "cardiac" not in hosp_c["capabilities"]
    assert "stroke" not in hosp_c["capabilities"]

def test_fleet_has_5_ambulances():
    assert len(AMBULANCES) == 5

def test_ambulance_ids_unique():
    ids = [a["id"] for a in AMBULANCES]
    assert len(ids) == len(set(ids))

def test_blood_types_listed():
    assert "O_POS" in BLOOD_TYPES
    assert "AB_NEG" in BLOOD_TYPES
    assert len(BLOOD_TYPES) == 8

def test_target_times_defined():
    assert PATIENT_TARGET_TIMES["cardiac"] == 90
    assert PATIENT_TARGET_TIMES["stroke"] == 60
    assert PATIENT_TARGET_TIMES["trauma"] == 60
    assert PATIENT_TARGET_TIMES["general"] == 120


def test_hospital_mortality_rates_all_hospitals():
    """All 3 hospitals must have mortality rates for all 4 conditions."""
    for hosp_id in ["HOSP_A", "HOSP_B", "HOSP_C"]:
        assert hosp_id in HOSPITAL_MORTALITY_RATES, f"{hosp_id} missing from mortality rates"
        rates = HOSPITAL_MORTALITY_RATES[hosp_id]
        for condition in ["cardiac", "stroke", "trauma", "general"]:
            assert condition in rates, f"{hosp_id} missing rate for {condition}"
            rate = rates[condition]
            assert 0.0 <= rate <= 1.0, f"{hosp_id}/{condition} rate {rate} out of [0,1]"


def test_hospital_c_emergent_conditions_100_mortality():
    """HOSP_C has no OR or specialists — emergent conditions always die."""
    rates = HOSPITAL_MORTALITY_RATES["HOSP_C"]
    assert rates["cardiac"] == 1.0
    assert rates["stroke"] == 1.0
    assert rates["trauma"] == 1.0


def test_hospital_a_best_outcomes():
    """HOSP_A (AIIMS) has lowest mortality rates."""
    hosp_a = HOSPITAL_MORTALITY_RATES["HOSP_A"]
    hosp_b = HOSPITAL_MORTALITY_RATES["HOSP_B"]
    for cond in ["cardiac", "stroke", "trauma", "general"]:
        assert hosp_a[cond] <= hosp_b[cond], f"HOSP_A/{cond} not better than HOSP_B"
