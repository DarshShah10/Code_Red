from codered_env.server.subsystems.constants import (
    CITY_NODES,
    CITY_EDGES,
    HOSPITALS,
    AMBULANCES,
    PATIENT_TARGET_TIMES,
    BLOOD_TYPES,
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
