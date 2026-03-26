# CodeRedEnv — OpenEnv Emergency Medical Coordination Environment

## 1. Overview

### Purpose
CodeRedEnv is an OpenEnv-compatible RL environment that simulates the emergency medical response chain of a city. An AI agent acts as a central emergency coordination system responsible for managing ambulance dispatch, hospital resource preparation, specialist coordination, operating room allocation, blood bank management, and multi-patient triage — all while responding to real-world disruptions.

### Why This Domain
Emergency medical coordination is a genuine real-world crisis — particularly in Indian cities where ambulance response times, hospital overcrowding, and resource shortages cause preventable deaths daily. The domain has:
- **Genuine time pressure** — patient survival decays with time to treatment
- **Scarce persistent resources** — ambulances, ORs, specialists, blood cannot be in two places at once
- **Delayed consequences** — actions taken now (preparing an OR) affect availability minutes later
- **Stochastic disruptions** — road closures, hospital diversions, equipment failures create unexpected pressure
- **Competing objectives** — saving one patient may cost resources needed for another

### Why This Is a Natural RL Problem
Optimal policies require anticipating future resource conflicts, not simply reacting to current conditions. Greedy approaches fail because dispatching an ambulance now reduces fleet availability for the next emergency. This makes the environment ideal for RL training.

### Scope
This document describes the environment's behavior, data models, subsystems, API, grading system, and deployment requirements. It specifies what the environment does — internal implementation choices are left to developers.

---

## 2. City Layout — Prakashnagar

### Geography
Prakashnagar is a fictional but India-flavored medium-sized city with a ring road, national highway bypass, old city narrow streets, and a newer IT corridor. The asymmetry in travel times creates meaningful routing decisions.

### Road Network Graph
12 nodes connected by named roads:

```
                          [AIIMS_PRAKASH] ←→ [MG_CHOWK] ←→ [SECTOR_12]
                                  ↑               ↑               ↑
    [LAJPAT_NAGAR] ←→ [CHOWKHA] ←→ [RAILWAY_XING] ←→ [NH45_BYPASS] ←→ [IT_HUB]
         ↑               ↑               ↑               ↑
    [RAJIV_CHOWK]  [DISTRICT_HOSP]   [RING_ROAD]    [COMMUNITY_HC]
```

**Node list (12 nodes):**
| Node ID | Name | Type |
|---------|------|------|
| `NODE_1` | RAJIV_CHOWK | Intersection |
| `NODE_2` | LAJPAT_NAGAR | Intersection |
| `NODE_3` | CHOWKHA | Intersection (old city, narrow) |
| `NODE_4` | RAILWAY_XING | Intersection |
| `NODE_5` | NH45_BYPASS | Arterial road |
| `NODE_6` | IT_HUB | Intersection |
| `NODE_7` | AIIMS_PRAKASH | Hospital A |
| `NODE_8` | DISTRICT_HOSPITAL | Hospital B |
**Edge list (base travel times in minutes, all 12 nodes fully connected):**
| From | To | Base Time | Notes |
|------|----|-----------|-------|
| RAJIV_CHOWK | LAJPAT_NAGAR | 3 | Narrow street |
| LAJPAT_NAGAR | CHOWKHA | 4 | Old city |
| LAJPAT_NAGAR | RING_ROAD | 6 | Outer ring |
| CHOWKHA | DISTRICT_HOSPITAL | 2 | Short access road |
| CHOWKHA | RAILWAY_XING | 5 | Congestion-prone |
| RAILWAY_XING | NH45_BYPASS | 6 | Highway segment |
| RAILWAY_XING | RING_ROAD | 4 | Ring connector |
| NH45_BYPASS | IT_HUB | 4 | Fast road |
| NH45_BYPASS | MG_CHOWK | 8 | Long connector |
| NH45_BYPASS | RING_ROAD | 3 | Ring junction |
| IT_HUB | SECTOR_12 | 3 | IT corridor |
| AIIMS_PRAKASH | MG_CHOWK | 5 | Hospital access |
| MG_CHOWK | SECTOR_12 | 4 | IT corridor |
| RING_ROAD | COMMUNITY_HC | 5 | Community clinic access |
| IT_HUB | COMMUNITY_HC | 7 | Indirect clinic access |

Each edge also has a `congestion_multiplier` (1.0 by default, increases during disruptions). All edges are bidirectional with the same travel time.

---

## 3. Hospitals

### Hospital Definitions

**Hospital A — AIIMS_PRAKASH (Node: AIIMS_PRAKASH)**
- Capabilities: cardiac, stroke, trauma, stabilization
- Specialists: 2 cardiologists, 1 neurologist, 2 trauma surgeons
- Operating rooms: 3
- ICU beds: 4
- Blood bank: A_POS=10, A_NEG=5, B_POS=10, B_NEG=5, AB_POS=5, AB_NEG=3, O_POS=12, O_NEG=6

**Hospital B — DISTRICT_HOSPITAL (Node: DISTRICT_HOSPITAL)**
- Capabilities: cardiac, trauma, stabilization (no stroke)
- Specialists: 1 cardiologist, 1 trauma surgeon
- Operating rooms: 2
- ICU beds: 2
- Blood bank: A_POS=6, A_NEG=3, B_POS=6, B_NEG=3, AB_POS=3, AB_NEG=2, O_POS=8, O_NEG=4

**Hospital C — COMMUNITY_HC (Node: COMMUNITY_HC)**
- Capabilities: stabilization only (no cardiac, no stroke, no trauma surgery)
- Specialists: none
- Operating rooms: 0
- ICU beds: 1
- Blood bank: O_POS=4, O_NEG=2 (all other types = 0)

### Hospital State Fields
- `hospital_id`: str
- `node_id`: str (maps to road network)
- `capabilities`: Set[Literal["cardiac", "stroke", "trauma", "stabilization"]]
- `specialists`: Dict[str, Dict["available": int, "total": int]]
- `operating_rooms`: List[Dict["status": str, "procedure_type": str | None, "minutes_remaining": int | None, "patient_id": str | None]]
- `icu_beds`: Dict["total": int, "available": int]
- `blood_stock`: Dict[str, int]  # 8 blood types
- `on_diversion`: bool (when True, hospital accepts no new patients)

### OR Preemption
When the agent issues `PreemptOR(hospital_id, or_index)`:
1. Current procedure is interrupted
2. Harm score computed: `harm = min(1.0, interrupted_procedure_minutes_remaining / 30)`
3. Recovery delay before OR is ready: `recovery_time = interrupted_procedure_minutes_remaining`
4. Interrupted patient's status is unchanged but OR slot is cleared
5. The preempting agent gets the OR available for the new patient

---

## 4. Ambulance Fleet

### Fleet Composition
5 ambulances total:
- `AMB_1`, `AMB_2`: ALS (Advanced Life Support) — faster treatment, more equipment
- `AMB_3`, `AMB_4`, `AMB_5`: BLS (Basic Life Support)

### Ambulance State Fields
- `ambulance_id`: str
- `node_id`: str (current position in road network)
- `status`: Literal["available", "dispatched", "transporting", "returning", "offline"]
- `equipment`: Literal["ALS", "BLS"]
- `assigned_patient`: str | None
- `route`: List[str] (ordered list of node IDs)
- `eta_minutes`: int (estimated time to destination)
- `destination_type`: Literal["patient", "hospital", "base"] | None

---

## 5. Patient Model

### Patient Conditions and Thresholds

| Condition | Tier | Target Treatment Time | Strict Decay Starts |
|-----------|------|---------------------|---------------------|
| Cardiac (STEMI) | Critical | 90 min (door-to-balloon) | After 90 min |
| Stroke | Critical | 60 min (door-to-needle) | After 60 min |
| Trauma | High | 60 min (golden hour) | After 60 min |
| General | Medium | 120 min (stabilization) | After 120 min |

### Patient State Fields
- `patient_id`: str
- `condition`: Literal["cardiac", "stroke", "trauma", "general"]
- `tier`: Literal["critical", "high", "medium"]
- `location_node`: str
- `time_since_onset`: int (minutes since emergency started)
- `assigned_ambulance`: str | None
- `assigned_hospital`: str | None
- `status`: Literal["waiting", "dispatched", "transporting", "treating", "treated", "deceased"]
- `blood_type`: str | None  # None = unknown until QueryBloodType
- `treatment_start_time`: int | None
- `treatment_complete_time`: int | None
- `outcome`: Literal["saved", "deceased"] | None

### Patient Lifecycle
1. `waiting` — patient appears, awaiting ambulance dispatch
2. `dispatched` — ambulance assigned and en route
3. `transporting` — ambulance has patient, en route to hospital
4. `treating` — patient admitted, receiving treatment
5. `treated` — treatment complete, stable (terminal state, positive outcome)
6. `deceased` — exceeded survival window or died en route (terminal state, zero time score)

Episode ends when all patients are in terminal states or `step_count >= max_steps`.

### Partial Observability
- `blood_type` is `None` until agent issues `QueryBloodType`
- Detailed OR status requires `QueryORStatus` action

---

## 6. Blood Bank

### Blood Types
8 types: `A_POS`, `A_NEG`, `B_POS`, `B_NEG`, `AB_POS`, `AB_NEG`, `O_POS`, `O_NEG`

### Blood Bank Actions
| Action | Latency | Effect |
|--------|---------|--------|
| AllocateBlood (crossmatch) | 15 min | Blood reserved, wait for crossmatch completion |
| AllocateBlood (emergency) | 0 min | O_NEG released immediately, depletes universal donor stock |
| TransferBlood | 20 min | Units moved between hospitals |
| QueryBloodType | 5 min | Reveals patient blood type |

### Blood Bank State Fields
- `hospital_id`: str
- `stocks`: Dict[str, int]  # units per blood type
- `crossmatch_queue`: List[Dict["patient_id": str, "blood_type": str, "units": int, "time_remaining": int]]

### Crossmatch Completion
When a crossmatch entry's `time_remaining` reaches 0:
1. The blood units are reserved for the patient (deducted from `stocks`)
2. An alert is added to the observation: `"Crossmatch complete: {units} units {blood_type} reserved for {patient_id} at {hospital_id}"`
3. The agent must still issue `AllocateBlood` to actually release the blood to the patient
4. If the patient has already been treated or deceased, the reserved units are returned to stock (wasted — efficiency penalty applies)

---

## 7. Disruption Engine

### Disruption Types
| Type | Effect | Duration |
|------|--------|----------|
| `road_closure` | Edge travel time → infinity | Remainder of episode |
| `accident` | Edge congestion multiplier × 3 | 10–20 minutes |
| `hospital_diversion` | Hospital on_diversion = True | 5–15 minutes |
| `equipment_failure` | One OR in hospital becomes unavailable | 10–30 minutes |
| `surge_event` | 1–2 additional patients spawn simultaneously | Immediate |

### Seeding and Regularization
Each task has a base disruption probability per timestep:
- **Task 1:** 0.0 (no disruptions)
- **Task 2:** 0.05 per timestep
- **Task 3:** 0.15 per timestep

At `reset(seed)`:
1. Base probability is fixed per task
2. Intensity multiplier `k` is drawn from `Uniform(0.7, 1.3)` deterministically from seed
3. Per-timestep disruption chance = `base_prob * k`
4. Disruption type is sampled from available types weighted by task difficulty
5. Disruption target (which road, which hospital) is seeded per episode

This is reproducible per seed (enabling RL replay) but varies across seeds (preventing pattern memorization).

**Disruption target selection:** For each disruption event, the target is selected deterministically from the seed:
- `road_closure`: `edge_index = (seed + event_index) % num_edges`
- `accident`: `edge_index = (seed + event_index + 1) % num_edges`
- `hospital_diversion`: `hospital_index = (seed + event_index) % num_hospitals`
- `equipment_failure`: `hospital_index = (seed + event_index + 2) % num_hospitals`, `or_index = (seed + event_index) % num_ors_in_hospital`
- `surge_event`: additional patients are sampled from the available patient pool

---

## 8. Mutual Aid

### Mutual Aid Properties
| Task | Available | Latency | Ambulance Type |
|------|-----------|---------|----------------|
| Task 1 | No | — | — |
| Task 2 | 1 call | 12 min | BLS |
| Task 3 | 2 calls | 12 min each | BLS |

### Mutual Aid Timing Penalty
The grader computes an optimal call window per mutual aid use:
- `RequestMutualAid` action has a 12-minute latency — the new ambulance arrives 12 steps after the call
- `arrival_step = call_step + 12`
- Early penalty: `0.1 * (optimal_window_start - arrival_step)` if ambulance arrives before window
- Late penalty: `0.2 * (arrival_step - optimal_window_end)` if ambulance arrives after window
- Calls are tracked: once used, `RequestMutualAid` has no further effect until episode reset
- Task 3 has two calls: the grader computes the penalty for each call independently against its respective window

---

## 9. Action Space

All actions are typed Pydantic models inheriting from `Action`. Each action has latency and a defined consequence.

### Action Definitions

| Action | Fields | Latency | Consequence |
|--------|--------|---------|------------|
| `DispatchAmbulance` | `ambulance_id`, `target_node` | Instant | Ambulance begins routing to target node |
| `PrepareOR` | `hospital_id`, `procedure_type` | 10 min | OR reserved; specialist paged; prep countdown starts |
| `PageSpecialist` | `hospital_id`, `specialist_type` | 8 min | Specialist status → paged; available after latency |
| `AssignHospital` | `patient_id`, `hospital_id` | Instant | Hospital flagged to prepare for incoming patient |
| `PreemptOR` | `hospital_id`, `or_index` | Instant | OR cleared; harm to interrupted patient computed |
| `AllocateBlood` | `hospital_id`, `blood_type`, `units`, `emergency: bool` | 0 or 15 min | Emergency: instant O_NEG release. Crossmatch: queued |
| `TransferBlood` | `from_hospital`, `to_hospital`, `blood_type`, `units` | 20 min | Units in transit; deducted from source |
| `RequestMutualAid` | *(none)* | 12 min | +1 BLS ambulance added to fleet; mutual_aid_remaining – 1 |
| `QueryBloodType` | `patient_id` | 5 min | Patient blood_type revealed |
| `QueryORStatus` | `hospital_id`, `or_index` | 1 step | Full OR detail revealed: procedure_type, minutes_remaining, specialist_assigned |
| `MaintainPlan` | *(none)* | — | No-op; all active routes/plans continue |

### Action Execution Order
Actions in a bundle execute in submission order. Contradictory actions resolve as follows:
- If a second `DispatchAmbulance` targets the same ambulance, the second is rejected
- Alert is added to next observation: `"action {n} failed: {reason}"`

---

## 10. Observation Model

### CodeRedObservation Fields
```python
class CodeRedObservation(Observation):
    step: int                                    # Current timestep (1-indexed)
    patients: List[PatientState]                 # All patients and current states
    ambulances: List[AmbulanceState]             # All ambulance positions/routes/status
    hospitals: List[HospitalState]               # All hospital resource states (OR status summarized)
    blood_banks: List[BloodBankState]            # Blood stocks per hospital
    road_network: RoadNetworkState               # Current edge travel times, active disruptions
    alerts: List[str]                            # New events this timestep + action failures
    mutual_aid_remaining: int                   # Calls left this episode
    time_score_preview: float                    # Running time-score estimate (0–1, per-patient average)
    patients_remaining: int                     # Non-terminal patients count
```

### Default OR Visibility
Without `QueryORStatus`, agent sees: `or_status: Literal["idle", "in_use", "prep"]`
With `QueryORStatus`, agent additionally sees: `procedure_type`, `minutes_remaining`, `specialist_assigned`

---

## 11. State Model

```python
class CodeRedState(State):
    episode_id: str
    task_id: str
    step_count: int
    cum_reward: float
    max_steps: int
    mutual_aid_used: int
    disruptions_active: List[DisruptionState]
    all_patients_terminal: bool
```

---

## 12. Task Definitions

**Note on timestep duration:** Each step represents 1 simulation minute. All action latencies are expressed in minutes and consume the equivalent number of steps. `time_since_onset` increments by 1 per step.

### Task 1 — Basic Emergency
- **Patients:** 1 cardiac patient
- **Disruptions:** None
- **Mutual Aid:** None
- **Max Steps:** 30 (30 minutes)
- **Focus:** Learn the coordination pipeline — dispatch ambulance, assign hospital, prepare OR, page specialist

### Task 2 — Concurrent Emergencies
- **Patients:** 2 patients (cardiac + stroke, staggered onset 5–10 minutes apart)
- **Disruptions:** Road closure and/or hospital diversion (intensity 0.7–1.3× seed multiplier)
- **Mutual Aid:** 1 call available
- **Max Steps:** 45 (45 minutes)
- **Focus:** Coordinate parallel treatment pipelines; manage resource conflicts

### Task 3 — Major Incident
- **Patients:** 5 patients (mixed conditions: 2 cardiac, 1 stroke, 1 trauma, 1 general)
- **Disruptions:** Multiple disruptions including surge event, hospital diversion, road closures (intensity 0.7–1.3×)
- **Mutual Aid:** 2 calls available
- **Max Steps:** 60 (60 minutes)
- **Focus:** Full triage, resource prioritization, staggered mutual aid timing

---

## 13. Grading System

### Rubric Axes (all normalized 0–1)

```
final_score =
    time_score     * 0.40 +
    efficiency     * 0.20 +
    secondary_harm * 0.20 +
    prep_ready     * 0.20
```

Coherence is handled at runtime via action conflict alerts, not as a grader axis.

### Axis Definitions

**time_score (40%)**
For each patient:
```
if outcome == "treated":
    time_score = max(0.0, min(1.0, 1.0 - (actual_treatment_time - target_time) / target_time))
else:  # deceased or never treated
    time_score = 0.0

episode_time_score = mean([patient.time_score for all patients])
```
Note: `min(1.0, ...)` clamps the score so arriving before the target time caps at 1.0 (no bonus above perfect).

**efficiency (20%)**
Penalizes wasted resources:
- Unused specialist pages: −0.1 each (page was issued but patient never arrived at that hospital)
- Wasted OR preparations: −0.15 each (prep countdown completed without a patient assigned)
- Emergency blood releases when crossmatch was feasible: −0.1 each
```
efficiency = max(0.0, 1.0 + total_penalties)
```
Mutual aid timing is **not** in the efficiency axis — it is tracked separately (see Mutual Aid Timing Penalty below).

**secondary_harm (20%)**
Penalizes harm to existing patients from resource reallocation:
- OR preemption harm = `min(1.0, interrupted_procedure_minutes_remaining / 30)` per preemption
```
secondary_harm = max(0, 1.0 - total_preemption_harm)
```

**prep_ready (20%)**
For each patient who reaches treatment:
```
prep_score = 1.0 if OR is idle and specialist is available when patient arrives
           = 0.5 if OR is in prep phase (countdown active, ≥1 min remaining) when patient arrives
           = 0.0 otherwise (no prep, OR in use, or specialist not paged)
episode_prep_ready = mean([patient.prep_score for all treated patients])
```
Only treated patients count toward this axis. Deceased patients contribute 0.

### Mutual Aid Timing Penalty (computed separately from base score)
The optimal call window is determined at `reset(seed)` from the disruption schedule:
- At reset, the first expected disruption timestep is computed from the seeded disruption sequence
- `optimal_window_start = max(1, disruption_timestep - 8)` (8 minutes before first disruption)
- `optimal_window_end = disruption_timestep + 2` (2 minutes after disruption hits)

If no disruptions occur in the episode, the optimal window defaults to the mid-episode range:
- Task 2: `[15, 25]`
- Task 3 call 1: `[5, 15]`
- Task 3 call 2: `[25, 35]`

```
arrival_step = call_step + 12  # mutual aid latency is 12 minutes

if arrival_step < optimal_window_start:
    penalty = 0.1 * (optimal_window_start - arrival_step)
elif arrival_step > optimal_window_end:
    penalty = 0.2 * (arrival_step - optimal_window_end)
else:
    penalty = 0.0

mutual_aid_penalty = sum(penalties) / num_calls

base_score = time_score * 0.40 + efficiency * 0.20 + secondary_harm * 0.20 + prep_ready * 0.20
final_score = base_score - mutual_aid_penalty
```
`final_score` is clamped to `[0.0, 1.0]`.

---

## 14. Reward Signal

### Continuous Feedback
At each step, the agent receives `time_score_preview` in the observation — a running estimate of the time_score axis based on current patient states and trajectories. This provides dense reward signal throughout the trajectory, not just at episode end.

**`time_score_preview` formula:**
```
for each patient:
    if status == "treated":
        preview = max(0.0, min(1.0, 1.0 - (treatment_complete_time - target_time) / target_time))
    elif status == "deceased":
        preview = 0.0
    elif assigned_ambulance is not None:
        # Projected: current time + remaining travel + OR prep
        projected_time = time_since_onset + ambulance.eta_minutes + 10
        preview = max(0.0, min(1.0, 1.0 - (projected_time - target_time) / target_time))
    else:
        # Waiting: projected as current + estimated dispatch + travel + prep
        projected_time = time_since_onset + 5 + 10 + 10  # 5 min dispatch + 10 travel + 10 prep
        preview = max(0.0, min(1.0, 1.0 - (projected_time - target_time) / target_time))

time_score_preview = mean([patient.preview for all patients])
```

### Per-Step Reward (intermediate signal)
```
step_reward = (current_time_score_preview - previous_time_score_preview)
            - 0.01 * step  # small step cost to discourage stalling
```
Note: efficiency penalties (unused pages, wasted preps) are tracked cumulatively and applied at episode end in the efficiency axis. They are not deducted per-step to avoid noisy dense signal.

---

## 15. OpenEnv API

### Required Functions

```python
class CodeRedEnvironment(Environment):
    ActT = CodeRedAction      # Union of all action types
    ObsT = CodeRedObservation

    def reset(self, seed: int | None = None, task_id: str = "task1") -> CodeRedObservation
    def step(self, action: CodeRedAction) -> Tuple[CodeRedObservation, float, bool, dict]
    @property
    def state(self) -> CodeRedState
```

### Endpoint Routes

**GET /tasks**
Returns task definitions, action schemas, and required fields.

**POST /baseline**
Runs the baseline agent (OpenAI API) on tasks 1–3 with seeds [0, 1, 2]. Returns:
```json
{
  "task1": {"mean": 0.XX, "per_seed": [0.XX, 0.XX, 0.XX]},
  "task2": {"mean": 0.XX, "per_seed": [0.XX, 0.XX, 0.XX]},
  "task3": {"mean": 0.XX, "per_seed": [0.XX, 0.XX, 0.XX]}
}
```

**POST /grader**
Returns grader scores for the most recent episode (all 5 axes and final score).

---

## 16. Baseline Script

The baseline script uses the OpenAI API (reads `OPENAI_API_KEY`). It implements a simple structured-prompt agent:

1. Formats the current `CodeRedObservation` as a structured text summary
2. Provides the full action schema
3. Instructs the agent to select one or more actions from the valid set
4. Calls the OpenAI API and parses the response into typed actions
5. Runs for `max_steps` or until all patients are terminal

The `run_baseline_agent(task_id, seed)` function is importable and synchronous — callable from both the `/baseline` endpoint and from a local CLI script.

---

## 17. File Structure

```
codered_env/
├── server/
│   ├── app.py                          # FastAPI entry point, /tasks, /baseline, /grader routes
│   ├── __init__.py
│   ├── codered_environment.py           # Core Environment: step(), reset(), state()
│   ├── models/
│   │   ├── __init__.py
│   │   ├── entities.py                 # Hospital, Ambulance, Patient, RoadNode, BloodBank
│   │   ├── actions.py                  # Typed action union (all action subclasses)
│   │   ├── observations.py             # Typed observation models
│   │   └── state.py                    # Typed state model
│   ├── subsystems/
│   │   ├── __init__.py
│   │   ├── road_network.py             # Graph, routing, congestion, disruptions
│   │   ├── ambulance_fleet.py          # Ambulance state, routes, positions
│   │   ├── hospital_system.py         # ORs, specialists, ICU, preemption logic
│   │   ├── patient_manager.py         # Patient generation, deterioration, lifecycle
│   │   ├── blood_bank.py              # Blood stocks, crossmatch, transfers
│   │   └── disruption_engine.py       # Seeded probability + jitter generator
│   └── graders/
│       ├── __init__.py
│       └── task_grader.py              # 4-axis rubric, mutual aid timing, final score
├── models.py                           # Top-level Action, Observation, State exports
├── client.py                           # EnvClient subclass for OpenEnv
├── baseline.py                         # Baseline inference script (OpenAI API)
├── openenv.yaml                        # Manifest with metadata, tasks, grading schema
├── pyproject.toml                      # Dependencies
└── README.md                           # Documentation
```

### Design Principle
Each subsystem is independently testable. `RoadNetwork`, `AmbulanceFleet`, `HospitalSystem`, `PatientManager`, `BloodBank`, and `DisruptionEngine` each have a clear interface. `CodeRedEnvironment` orchestrates them in `step()`.

---

## 18. Docker and Deployment

### Dockerfile
Based on `ghcr.io/meta-pytorch/openenv-base:latest`:
- Multi-stage build
- uv for dependency management
- Health check against `/health`
- Exposes port 8000

### Hugging Face Space
- Tagged with `openenv`
- Space name: `<username>/codered-env`
- Responds to `reset()` with HTTP 200

### Local Development
```bash
# Install dependencies
uv sync

# Run server
uvicorn server.app:app --reload --port 8000

# Run baseline
python baseline.py --task task1 --seed 0

# Validate
openenv validate
```

---

## 19. Implementation Priorities

### Phase 1 — Core (Must Have)
1. City graph, hospital definitions, ambulance fleet — static data
2. `reset()` with patient generation for Task 1
3. `step()` with dispatch, routing, transport, treatment actions
4. Tiered patient thresholds and deterioration
5. Basic grader (time_score axis only)
6. OpenEnv integration, openenv.yaml, Docker

### Phase 2 — Complexity (Should Have)
7. All 3 tasks with correct patient counts and disruption engine
8. All grader axes
9. Hospital capabilities and specialist routing
10. Blood bank model
11. OR preemption with harm computation
12. Mutual aid with timing penalty

### Phase 3 — Polish (Nice to Have)
13. Baseline inference script
14. `/tasks`, `/baseline`, `/grader` endpoints
15. HF Spaces deployment
16. Comprehensive README and documentation

---

## 20. Acceptance Criteria

- [ ] Environment passes `openenv validate`
- [ ] `reset()` returns valid initial observation for all 3 tasks
- [ ] `step()` executes all 10 action types correctly
- [ ] Patient deterioration follows tiered thresholds
- [ ] Hospital capability routing works (cardiac → Hospital A/B)
- [ ] Disruption engine produces seeded, reproducible events
- [ ] Mutual aid arrives after 12 minutes, penalized for early/late use
- [ ] Grader outputs scores 0.0–1.0 on all axes for all tasks
- [ ] All patients terminal (treated or deceased) → episode done
- [ ] Dockerfile builds and runs
- [ ] Baseline script completes on all 3 tasks
- [ ] HF Space responds to `reset()` with HTTP 200
