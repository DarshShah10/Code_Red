"""Static data for CodeRedEnv — Prakashnagar city layout, hospital definitions, ambulance fleet."""

from typing import Dict, List

# =============================================================================
# CITY ROAD NETWORK — Prakashnagar
# Hub-and-spoke city: central NH45 bypass + old city + IT corridor
# 12 nodes, 15 bidirectional edges
# =============================================================================

CITY_NODES: List[Dict] = [
    {"id": "RAJIV_CHOWK", "name": "Rajiv Chowk", "type": "intersection"},
    {"id": "LAJPAT_NAGAR", "name": "Lajpat Nagar", "type": "intersection"},
    {"id": "CHOWKHA", "name": "Chowkha", "type": "intersection", "notes": "old city, narrow streets"},
    {"id": "RAILWAY_XING", "name": "Railway Crossing", "type": "intersection"},
    {"id": "NH45_BYPASS", "name": "NH45 Bypass", "type": "arterial"},
    {"id": "IT_HUB", "name": "IT Hub", "type": "intersection"},
    {"id": "AIIMS_PRAKASH", "name": "AIIMS Prakash", "type": "hospital"},
    {"id": "DISTRICT_HOSP", "name": "District Hospital", "type": "hospital"},
    {"id": "COMMUNITY_HC", "name": "Community Health Centre", "type": "hospital"},
    {"id": "MG_CHOWK", "name": "MG Chowk", "type": "intersection"},
    {"id": "SECTOR_12", "name": "Sector 12", "type": "intersection"},
    {"id": "RING_ROAD", "name": "Ring Road", "type": "intersection"},
]

CITY_EDGES: List[Dict] = [
    # Old city cluster
    {"from": "RAJIV_CHOWK", "to": "LAJPAT_NAGAR", "base_time": 3, "notes": "narrow street"},
    {"from": "LAJPAT_NAGAR", "to": "CHOWKHA", "base_time": 4, "notes": "old city"},
    {"from": "LAJPAT_NAGAR", "to": "RING_ROAD", "base_time": 6, "notes": "outer ring"},
    {"from": "CHOWKHA", "to": "DISTRICT_HOSP", "base_time": 2, "notes": "hospital access"},
    {"from": "CHOWKHA", "to": "RAILWAY_XING", "base_time": 5, "notes": "congestion-prone"},
    {"from": "RAILWAY_XING", "to": "NH45_BYPASS", "base_time": 6, "notes": "highway segment"},
    {"from": "RAILWAY_XING", "to": "RING_ROAD", "base_time": 4, "notes": "ring connector"},
    # NH45 bypass (main artery)
    {"from": "NH45_BYPASS", "to": "IT_HUB", "base_time": 4, "notes": "fast road"},
    {"from": "NH45_BYPASS", "to": "MG_CHOWK", "base_time": 8, "notes": "long connector"},
    {"from": "NH45_BYPASS", "to": "RING_ROAD", "base_time": 3, "notes": "ring junction"},
    # IT corridor
    {"from": "IT_HUB", "to": "SECTOR_12", "base_time": 3, "notes": "IT corridor"},
    {"from": "IT_HUB", "to": "COMMUNITY_HC", "base_time": 7, "notes": "indirect clinic access"},
    # MG Chowk connections
    {"from": "MG_CHOWK", "to": "AIIMS_PRAKASH", "base_time": 5, "notes": "hospital access"},
    {"from": "MG_CHOWK", "to": "SECTOR_12", "base_time": 4, "notes": "IT corridor"},
    # Community HC access
    {"from": "RING_ROAD", "to": "COMMUNITY_HC", "base_time": 5, "notes": "clinic access"},
]

# =============================================================================
# HOSPITALS
# =============================================================================

BLOOD_TYPES: List[str] = [
    "A_POS", "A_NEG", "B_POS", "B_NEG",
    "AB_POS", "AB_NEG", "O_POS", "O_NEG",
]

HOSPITALS: List[Dict] = [
    {
        "id": "HOSP_A",
        "name": "AIIMS Prakash",
        "node_id": "AIIMS_PRAKASH",
        "capabilities": ["cardiac", "stroke", "trauma", "stabilization"],
        "specialists": {
            "cardiologist": {"available": 2, "total": 2},
            "neurologist": {"available": 1, "total": 1},
            "trauma_surgeon": {"available": 2, "total": 2},
        },
        "num_or": 3,
        "icu_beds": {"total": 4, "available": 4},
        "blood_stock": {
            "A_POS": 10, "A_NEG": 5, "B_POS": 10, "B_NEG": 5,
            "AB_POS": 5, "AB_NEG": 3, "O_POS": 12, "O_NEG": 6,
        },
    },
    {
        "id": "HOSP_B",
        "name": "District Hospital",
        "node_id": "DISTRICT_HOSP",
        "capabilities": ["cardiac", "trauma", "stabilization"],
        "specialists": {
            "cardiologist": {"available": 1, "total": 1},
            "neurologist": {"available": 0, "total": 0},
            "trauma_surgeon": {"available": 1, "total": 1},
        },
        "num_or": 2,
        "icu_beds": {"total": 2, "available": 2},
        "blood_stock": {
            "A_POS": 6, "A_NEG": 3, "B_POS": 6, "B_NEG": 3,
            "AB_POS": 3, "AB_NEG": 2, "O_POS": 8, "O_NEG": 4,
        },
    },
    {
        "id": "HOSP_C",
        "name": "Community Health Centre",
        "node_id": "COMMUNITY_HC",
        "capabilities": ["stabilization"],
        "specialists": {
            "cardiologist": {"available": 0, "total": 0},
            "neurologist": {"available": 0, "total": 0},
            "trauma_surgeon": {"available": 0, "total": 0},
        },
        "num_or": 0,
        "icu_beds": {"total": 1, "available": 1},
        "blood_stock": {
            "O_POS": 4, "O_NEG": 2,
            "A_POS": 0, "A_NEG": 0, "B_POS": 0, "B_NEG": 0,
            "AB_POS": 0, "AB_NEG": 0,
        },
    },
]

# =============================================================================
# AMBULANCE FLEET
# =============================================================================

AMBULANCES: List[Dict] = [
    {"id": "AMB_1", "equipment": "ALS", "base_node": "RAILWAY_XING"},
    {"id": "AMB_2", "equipment": "ALS", "base_node": "NH45_BYPASS"},
    {"id": "AMB_3", "equipment": "BLS", "base_node": "LAJPAT_NAGAR"},
    {"id": "AMB_4", "equipment": "BLS", "base_node": "IT_HUB"},
    {"id": "AMB_5", "equipment": "BLS", "base_node": "RAJIV_CHOWK"},
]

# =============================================================================
# PATIENT CONFIGURATION
# =============================================================================

PATIENT_TARGET_TIMES: Dict[str, int] = {
    "cardiac": 90,    # Door-to-balloon (AHA standard)
    "stroke": 60,     # Door-to-needle (AHA standard)
    "trauma": 60,     # Golden hour (ATLS standard)
    "general": 120,   # Stabilization target
}

PATIENT_CONDITION_REQUIREMENTS: Dict[str, List[str]] = {
    "cardiac": ["cardiac"],
    "stroke": ["stroke"],
    "trauma": ["trauma"],
    "general": ["stabilization"],
}

# =============================================================================
# TASK CONFIGURATION
# =============================================================================

TASK_CONFIG: Dict[str, Dict] = {
    "task1": {
        "patients": [
            {"condition": "cardiac", "onset_step": 0},
        ],
        "disruption_prob": 0.0,
        "mutual_aid_calls": 0,
        "max_steps": 30,
    },
    "task2": {
        "patients": [
            {"condition": "cardiac", "onset_step": 0},
            {"condition": "stroke", "onset_step": 5},
        ],
        "disruption_prob": 0.05,
        "mutual_aid_calls": 1,
        "max_steps": 45,
    },
    "task3": {
        "patients": [
            {"condition": "cardiac", "onset_step": 0},
            {"condition": "cardiac", "onset_step": 3},
            {"condition": "stroke", "onset_step": 5},
            {"condition": "trauma", "onset_step": 8},
            {"condition": "general", "onset_step": 12},
        ],
        "disruption_prob": 0.15,
        "mutual_aid_calls": 2,
        "max_steps": 60,
    },
}

# =============================================================================
# HOSPITAL QUALITY VARIANCE — Task 12
# =============================================================================

# Mortality rate per hospital × condition (probability patient dies even after treatment).
# HOSP_A = tertiary referral (AIIMS) — best outcomes
# HOSP_B = district hospital — moderate outcomes
# HOSP_C = community HC — stabilisation only; no OR, no surgery → 100% mortality for
#          conditions requiring procedures (cardiac, stroke, trauma)
HOSPITAL_MORTALITY_RATES: dict[str, dict[str, float]] = {
    "HOSP_A": {
        "cardiac": 0.08,
        "stroke": 0.12,
        "trauma": 0.15,
        "general": 0.05,
    },
    "HOSP_B": {
        "cardiac": 0.15,
        "stroke": 0.25,   # no neurologist → worse stroke outcomes
        "trauma": 0.20,
        "general": 0.08,
    },
    "HOSP_C": {
        # Community HC has no OR and no specialists — cannot treat emergent conditions
        "cardiac": 1.00,
        "stroke": 1.00,
        "trauma": 1.00,
        "general": 0.10,
    },
}

# =============================================================================
# PATIENT VITALS — Phase 1
# =============================================================================

VITALS_INITIAL = 1.0
VITALS_STABLE_DECAY_RATE = 0.0     # flat in stable window — no recovery without treatment
VITALS_DETERIORATING_THRESHOLD = 0.75
VITALS_CRITICAL_THRESHOLD = 0.4
VITALS_DEAD_THRESHOLD = 0.0

# Reward shaping
VITALS_DELTA_WEIGHT = 0.5
MILESTONE_REWARDS = {
    "dispatched": 0.20,
    "in_treatment": 0.40,
    "treated": 0.80,
    "deceased": -0.80,
}
REWARD_STEP_CLAMP = (-1.0, 1.0)
