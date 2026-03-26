# CodeRedEnv — Plan 3: Grading, API Endpoints & Deployment

> **For agentic workers:** Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Implement the 4-axis rubric grader with mutual aid penalty, the `/tasks` and `/grader` endpoints, the `/baseline` endpoint with OpenAI API agent, the `baseline.py` importable script, finish the Dockerfile, and prepare the HF Spaces deployment.

**Prerequisites:** Plan 1 and Plan 2 must be complete. The `CodeRedEnvironment` class, all subsystem classes, and all Pydantic models must exist with correct interfaces.

**Architecture:** Rubric is a pure function `grade(episode_log) -> RubricResult`. The grader endpoint runs the full RL episode internally. The baseline endpoint uses `OpenAIResponsesAPI` with function-calling tools matching all 10 action types. HF Spaces deployment via `huggingface_hub` with `openenv` tag.

**Tech Stack:** Python 3.10+, OpenAI SDK (`openai>=1.56.0`), `huggingface_hub`, FastAPI, pydantic, pytest

---

## File Structure

```
codered_env/
├── server/
│   ├── app.py                      # MODIFY: add /tasks, /grader, /baseline endpoints
│   ├── grader.py                   # NEW: Rubric class, 4-axis scoring, mutual aid penalty
│   └── codered_environment.py      # MODIFY: expose episode_log for grader access
├── baseline.py                     # NEW: importable run_baseline_agent(task_id, seed, api_key)
├── server/Dockerfile               # MODIFY: add HEALTHCHECK, ENV vars, huggingface metadata
├── openenv.yaml                    # MODIFY: add HF Spaces metadata
├── codered_env/tests/test_grader.py      # NEW: rubric unit tests
└── codered_env/tests/test_baseline.py   # NEW: baseline agent tests (mocked)
```

---

## Task 14: Rubric Grading System

**Files:**
- Create: `codered_env/server/grader.py`
- Create: `codered_env/tests/test_grader.py`

### RubricResult Model

```python
# codered_env/server/grader.py

from dataclasses import dataclass

@dataclass
class RubricResult:
    time_score: float      # 0.0–1.0
    efficiency: float      # 0.0–1.0
    secondary_harm: float  # 0.0–1.0
    prep_ready: float       # 0.0–1.0
    mutual_aid_penalty: float  # 0.0–1.0, subtracted from final_score
    final_score: float     # weighted sum minus mutual_aid_penalty
    breakdown: dict         # human-readable per-axis details
```

### Grading Implementation

The rubric iterates over `episode_log` — a list of `{step, patient_id, event, ...}` dicts captured during the episode. Each event may carry extra fields (e.g., `reason`, `ambulance_id`).

**IMPORTANT — match spec exactly.** The grader implementation MUST match the formulas in the design spec (`codered-env-design.md`). The episode_log event fields are defined in Task 13.

**time_score (40% weight):** Per the spec:

```
for each patient:
    if treated:
        time_score_patient = max(0.0, min(1.0, 1.0 - (effective_time - target_time) / target_time))
    else:
        time_score_patient = 0.0
time_score = mean(time_score_patient for all patients)
```

Where `effective_time = treatment_complete_step - dispatch_step`. Both values come from episode_log events.

**efficiency (20% weight):** Per the spec — three weighted penalty categories:

```
unused_specialist_pages = count of "action_page_specialist" where specialist_id not in ON_CALL/ON_WAY at action time
wasted_or_preps         = count of "action_prepare_or" where or_id already in PREPPING/READY at action time
premature_blood_emerg   = count of "action_allocate_blood" where crossmatch was feasible (crossmatch_queue contains patient)

total_penalty = -0.1 * unused_specialist_pages + -0.15 * wasted_or_preps + -0.1 * premature_blood_emerg
efficiency = max(0.0, min(1.0, 1.0 + total_penalty))
```

Each action in episode_log carries a `result` field: `"success"`, `"wasted"`, or `"failed"`. The implementer writes the logic to classify each.

**secondary_harm (20% weight):** Per the spec — secondary deaths vs. secondary patients:

```
num_secondary_deaths   = count of "patient_deceased" where reason == "secondary"
num_secondary_patients = count of "patient_created" where is_secondary == True
secondary_harm = 1.0 - (num_secondary_deaths / num_secondary_patients)  if num_secondary_patients > 0 else 1.0
```

Primary patient deaths are NOT counted here (they affect time_score instead).

**prep_ready (20% weight):** Per the spec — ternary at patient arrival:

```
for each patient that arrived at hospital:
    state_at_arrival = or_state + specialist_state from episode_log at arrival_step
    if OR_READY and specialist_ON_WAY/ON_CALL:  score = 1.0
    elif OR_READY or specialist_available:       score = 0.5
    else:                                        score = 0.0
prep_ready = mean(scores for arrived patients)
# Patients who never arrived: skip (don't penalize agent for patient not reaching hospital)
```

The episode_log must emit a `hospital_state_at_arrival` event with `or_state` and `specialist_state` fields when a patient arrives.

**Mutual Aid Penalty:** Per the spec — arrival-step comparison, divided by num_calls:

```
mutual_aid_calls = [log for log in episode_log if log["event"] == "mutual_aid_called"]
penalties = []
for call in mutual_aid_calls:
    optimal_arrival = call["optimal_arrival_step"]   # from disruption_engine schedule
    actual_arrival  = call["actual_arrival_step"]   # from "mutual_aid_arrived" event
    if actual_arrival < optimal_arrival:
        penalties.append(0.1 * (optimal_arrival - actual_arrival))
    elif actual_arrival > optimal_arrival:
        penalties.append(0.2 * (actual_arrival - optimal_arrival))
num_calls = len(mutual_aid_calls)
mutual_aid_penalty = sum(penalties) / num_calls if num_calls > 0 else 0.0
```

**final_score:** Per the spec:

```
raw = 0.40 * time_score + 0.20 * efficiency + 0.20 * secondary_harm + 0.20 * prep_ready
final_score = max(0.0, min(1.0, raw - mutual_aid_penalty))
```

**Important note for implementer:** The mutual aid penalty computation requires `optimal_arrival_step` from the disruption schedule. Pass the disruption schedule into the grader or store it in the episode_log at `mutual_aid_called` time.

- [ ] **Step 1: Write rubric unit tests**

```python
# codered_env/tests/test_grader.py
import pytest
from codered_env.server.grader import grade_episode

def test_all_patients_treated_on_time():
    episode_log = [
        {"step": 0, "patient_id": "P0", "event": "patient_created", "condition": "CARDIAC"},
        {"step": 1, "patient_id": "P0", "event": "dispatch", "ambulance_id": "AMB_1", "target_time": 90, "effective_time": 14},
        {"step": 15, "patient_id": "P0", "event": "treatment_complete"},
    ]
    result = grade_episode(episode_log)
    # 14 min vs 90 min target: score = 1.0 - (14-90)/90 = 1.0 - (-76/90) = clamped to 1.0
    assert result.time_score == 1.0
    assert result.efficiency == 1.0  # no wasted actions
    assert result.final_score >= 0.80

def test_patient_not_treated():
    episode_log = [
        {"step": 0, "patient_id": "P0", "event": "patient_created", "condition": "CARDIAC"},
        # patient never dispatched, never treated
    ]
    result = grade_episode(episode_log)
    assert result.time_score == 0.0

def test_efficiency_wasted_or_prep():
    # Two wasted OR preps: penalty = -0.15 * 2 = -0.30 → efficiency = 0.70
    episode_log = [
        {"step": 0, "patient_id": "P0", "event": "patient_created"},
        {"step": 1, "patient_id": "P0", "event": "action_prepare_or", "result": "wasted"},
        {"step": 2, "patient_id": "P0", "event": "action_prepare_or", "result": "wasted"},
        {"step": 3, "patient_id": "P0", "event": "treatment_complete"},
    ]
    result = grade_episode(episode_log)
    assert result.efficiency == 0.70

def test_efficiency_unused_specialist_pages():
    # Two unused specialist pages: penalty = -0.1 * 2 = -0.20 → efficiency = 0.80
    episode_log = [
        {"step": 0, "patient_id": "P0", "event": "patient_created"},
        {"step": 1, "patient_id": "P0", "event": "action_page_specialist", "result": "wasted"},
        {"step": 2, "patient_id": "P0", "event": "treatment_complete"},
    ]
    result = grade_episode(episode_log)
    assert result.efficiency == 0.80

def test_mutual_aid_on_time_no_penalty():
    # MA arrives exactly at optimal: 0 penalty
    episode_log = [
        {"step": 0, "patient_id": "P0", "event": "patient_created"},
        {"step": 5, "patient_id": "P0", "event": "mutual_aid_called", "optimal_arrival_step": 17},
        {"step": 17, "patient_id": "P0", "event": "mutual_aid_arrived", "actual_arrival_step": 17},
        {"step": 20, "patient_id": "P0", "event": "treatment_complete"},
    ]
    result = grade_episode(episode_log)
    assert result.mutual_aid_penalty == 0.0

def test_mutual_aid_late_penalty():
    # MA arrives 5 steps late: penalty = 0.2 * 5 / 1 call = 1.0 → capped to 1.0
    episode_log = [
        {"step": 0, "patient_id": "P0", "event": "patient_created"},
        {"step": 5, "patient_id": "P0", "event": "mutual_aid_called", "optimal_arrival_step": 17},
        {"step": 22, "patient_id": "P0", "event": "mutual_aid_arrived", "actual_arrival_step": 22},
        {"step": 30, "patient_id": "P0", "event": "treatment_complete"},
    ]
    result = grade_episode(episode_log)
    assert result.mutual_aid_penalty > 0.0
    assert result.final_score < 1.0

def test_secondary_harm_partial():
    # 1 of 2 secondary patients died → harm = 1 - 0.5 = 0.5
    episode_log = [
        {"step": 0, "patient_id": "P0", "event": "patient_created", "condition": "CARDIAC"},   # primary
        {"step": 1, "patient_id": "P1", "event": "patient_created", "is_secondary": True},     # secondary survived
        {"step": 2, "patient_id": "P2", "event": "patient_created", "is_secondary": True},     # secondary died
        {"step": 3, "patient_id": "P2", "event": "patient_deceased", "reason": "secondary"},
        {"step": 20, "patient_id": "P0", "event": "treatment_complete"},
    ]
    result = grade_episode(episode_log)
    assert result.secondary_harm == 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest codered_env/tests/test_grader.py -v`
Expected: FAIL — grader module doesn't exist yet

- [ ] **Step 3: Write minimal grader implementation**

```python
# codered_env/server/grader.py
from dataclasses import dataclass

@dataclass
class RubricResult:
    time_score: float
    efficiency: float
    secondary_harm: float
    prep_ready: float
    mutual_aid_penalty: float
    final_score: float
    breakdown: dict

def grade_episode(episode_log: list[dict]) -> RubricResult:
    # placeholders — full implementation below
    time_score = 0.0
    efficiency = 0.0
    secondary_harm = 0.0
    prep_ready = 0.0
    mutual_aid_penalty = 0.0
    final_score = (
        0.40 * time_score +
        0.20 * efficiency +
        0.20 * secondary_harm +
        0.20 * prep_ready
    ) - mutual_aid_penalty
    return RubricResult(
        time_score=time_score, efficiency=efficiency,
        secondary_harm=secondary_harm, prep_ready=prep_ready,
        mutual_aid_penalty=mutual_aid_penalty,
        final_score=max(0.0, min(1.0, final_score)),
        breakdown={}
    )
```

- [ ] **Step 4: Run test to verify it fails with partial assertions**

Run: `pytest codered_env/tests/test_grader.py -v`
Expected: FAIL — stub returns 0.0, assertions check for real values

- [ ] **Step 5: Implement full rubric logic**

Replace the stub with the complete implementation per the design above.

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest codered_env/tests/test_grader.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add codered_env/server/grader.py codered_env/tests/test_grader.py
git commit -m "feat: implement 4-axis rubric grader with mutual aid penalty"
```

---

## Task 15: Episode Logging Infrastructure

**Files:**
- Modify: `codered_env/server/codered_environment.py`
- Modify: `codered_env/tests/test_grader.py`

The grader needs structured episode data. The environment logs events during `reset()` and `step()` calls.

- [ ] **Step 1: Add episode_log to CodeRedEnvironment**

Add a `_episode_log: list[dict]` field initialized in `reset()`. After each `step()`, append structured events:

```python
# In reset():
self._episode_log: list[dict] = []

# In step(), after each action/event:
self._episode_log.append({
    "step": self._state.step_count,
    "patient_id": patient_id,
    "event": "dispatch",          # or "treatment_complete", "death", etc.
    "ambulance_id": ambulance_id,
    "target_time": target_time,
    "effective_time": effective_time,
    "or_ready": True,             # whether OR was ready at arrival
    "mutual_aid_arrival_step": arrival_step,
})
```

Key events to log:
- `patient_created`: when a new patient is spawned (includes condition, target_time)
- `dispatch`: ambulance dispatched (includes ambulance_id)
- `action_<action_type>`: each action taken (includes action type)
- `action_wasted`: when an action had no effect (includes reason)
- `patient_arrived_hospital`: patient arrived at hospital
- `or_prep_started`: OR prep initiated
- `or_ready`: OR became ready
- `surgery_started`: patient started surgery
- `treatment_complete`: patient treatment completed (includes effective_time)
- `patient_deceased`: patient died (includes reason: timeout | secondary)
- `mutual_aid_called`: mutual aid requested (includes call_step)
- `mutual_aid_arrived`: mutual aid arrived (includes arrival_step)

- [ ] **Step 2: Add `get_episode_log()` to environment**

```python
def get_episode_log(self) -> list[dict]:
    """Return the full episode log for grading."""
    return self._episode_log.copy()
```

- [ ] **Step 3: Update grader to use episode_log from environment**

```python
def grade_from_environment(env: CodeRedEnvironment) -> RubricResult:
    return grade_episode(env.get_episode_log())
```

- [ ] **Step 4: Run grader tests**

Run: `pytest codered_env/tests/test_grader.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add codered_env/server/codered_environment.py codered_env/tests/test_grader.py
git commit -m "feat: add episode logging infrastructure for grader"
```

---

## Task 16: API Endpoints — `/tasks`, `/grader`, `/baseline`

**Files:**
- Modify: `codered_env/server/app.py`
- Create: `codered_env/tests/test_endpoints.py`

### Endpoint Design

**`GET /tasks`** — Returns task definitions for all 3 tasks, including:
- Task ID and name
- Patient scenarios (conditions, count, spawn steps)
- Disruption configuration
- Action schema (OpenAPI-compatible)
- Max episode steps

```python
@app.get("/tasks")
async def get_tasks() -> dict:
    return {
        "tasks": [
            {
                "task_id": "task1",
                "name": "Cardiac Emergency — Single Patient",
                "description": "...",
                "max_steps": 30,
                "disruption_schedule": [],
                "patients": [
                    {"condition": "CARDIAC", "spawn_step": 1}
                ],
                "action_schema": {...}  # all 10 actions
            },
            {
                "task_id": "task2",
                "name": "Multi-Patient Emergency",
                "description": "...",
                "max_steps": 30,
                "disruption_schedule": [...],  # from constants
                "patients": [
                    {"condition": "CARDIAC", "spawn_step": 1},
                    {"condition": "STROKE", "spawn_step": 5},
                    {"condition": "TRAUMA", "spawn_step": 8}
                ]
            },
            {
                "task_id": "task3",
                "name": "Crisis Surge — Mass Casualty",
                "description": "...",
                "max_steps": 30,
                "disruption_schedule": [...],
                "patients": [
                    {"condition": "CARDIAC", "spawn_step": 1},
                    {"condition": "STROKE", "spawn_step": 2},
                    {"condition": "TRAUMA", "spawn_step": 3},
                    {"condition": "GENERAL", "spawn_step": 4},
                    {"condition": "CARDIAC", "spawn_step": 10}
                ],
                "mutual_aid": {"calls": 2, "stagger_min_steps": 8}
            }
        ]
    }
```

**`POST /grader`** — Runs a full episode and returns rubric score.

```python
class GraderRequest(BaseModel):
    task_id: Literal["task1", "task2", "task3"]  # Must match TASK_CONFIG keys in constants.py
    seed: int
    # Optional: override openai_api_key for agent-based grading
    agent_type: Literal["dummy", "openai"] = "dummy"
    openai_api_key: str | None = None

@app.post("/grader")
async def grade_task(req: GraderRequest) -> dict:
    env = CodeRedEnvironment()
    obs = env.reset(seed=req.seed, task_id=req.task_id)

    # Run dummy agent (or OpenAI agent if specified)
    for _ in range(30):
        action = dummy_agent(obs)  # or openai_agent(obs)
        obs = env.step(action)
        if obs.done:
            break

    result = grade_from_environment(env)
    return asdict(result)
```

**`POST /baseline`** — Runs OpenAI API agent on seeds [0, 1, 2] and returns scores.

```python
class BaselineRequest(BaseModel):
    task_id: Literal["task1", "task2", "task3"]  # Must match TASK_CONFIG keys in constants.py
    openai_api_key: str

@app.post("/baseline")
async def run_baseline(req: BaselineRequest) -> dict:
    scores = []
    for seed in [0, 1, 2]:
        score = run_baseline_agent(task_id=req.task_id, seed=seed, api_key=req.openai_api_key)
        scores.append(score)
    return {
        "task_id": req.task_id,
        "scores": scores,
        "mean": sum(scores) / len(scores)
    }
```

- [ ] **Step 1: Write endpoint tests**

```python
# codered_env/tests/test_endpoints.py
import pytest
from fastapi.testclient import TestClient

def test_get_tasks(client):
    response = client.get("/tasks")
    assert response.status_code == 200
    data = response.json()
    assert len(data["tasks"]) == 3
    assert data["tasks"][0]["task_id"] == "task1"
    assert data["tasks"][1]["task_id"] == "task2"
    assert data["tasks"][2]["task_id"] == "task3"

def test_grader_endpoint_task1(client):
    response = client.post("/grader", json={"task_id": "task1", "seed": 42})
    assert response.status_code == 200
    data = response.json()
    assert "final_score" in data
    assert 0.0 <= data["final_score"] <= 1.0

def test_baseline_endpoint_requires_api_key(client):
    response = client.post("/baseline", json={"task_id": "task1", "openai_api_key": "sk-test"})
    assert response.status_code in [200, 400]  # 400 if key invalid
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest codered_env/tests/test_endpoints.py -v`
Expected: FAIL — endpoints don't exist yet

- [ ] **Step 3: Implement all endpoints in app.py**

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest codered_env/tests/test_endpoints.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add codered_env/server/app.py codered_env/tests/test_endpoints.py
git commit -m "feat: add /tasks, /grader, /baseline API endpoints"
```

---

## Task 17: Baseline Agent — OpenAI API Integration

**Files:**
- Create: `codered_env/baseline.py`
- Create: `codered_env/tests/test_baseline.py`

### Baseline Agent Design

The baseline agent uses OpenAI's Responses API with function-calling tools. It receives the full observation at each step and must output one of the 10 typed actions.

**System prompt** (abbreviated — full prompt in spec):

```
You are a medical emergency coordination AI. You manage ambulance dispatch, hospital preparation,
and patient triage in the city of Prakashnagar.

Your tools:
- dispatch_ambulance(ambulance_id, patient_id, destination_node)
- prepare_or(hospital_id)
- page_specialist(hospital_id, specialty)
- assign_hospital(patient_id, hospital_id)
- preempt_or(hospital_id)
- allocate_blood(hospital_id, blood_type, units)
- transfer_blood(source_hospital_id, dest_hospital_id, blood_type, units)
- request_mutual_aid(ambulance_id, target_hospital_id)
- query_blood_type(hospital_id)
- query_or_status(hospital_id)

City layout: [nodes table from constants]
Hospitals: A(AIIMS), B(District), C(Community HC)
Ambulances: AMB_1, AMB_2 (ALS — advanced life support), AMB_3, AMB_4, AMB_5 (BLS — basic life support)

Step every minute. Take action or "maintain_plan" if no action needed.
Prioritize by condition severity: CARDIAC > STROKE > TRAUMA > GENERAL.
```

**Function definitions** (one per action type):

```python
FUNCTIONS = [
    {
        "type": "function",
        "name": "dispatch_ambulance",
        "description": "Dispatch an ambulance to pick up a patient...",
        "parameters": {
            "type": "object",
            "properties": {
                "ambulance_id": {"type": "string", "enum": ["AMB_1", "AMB_2", "AMB_3", "AMB_4", "AMB_5"]},
                "patient_id": {"type": "string"},
                "destination_node": {"type": "string"}
            },
            "required": ["ambulance_id", "patient_id", "destination_node"]
        }
    },
    # ... all 10 actions
]
```

**Agent loop:**

```python
def format_observation(obs: CodeRedObservation) -> str:
    """
    Convert CodeRedObservation into a readable text prompt for the LLM.
    Keep it concise — aim for ~2000 tokens max.
    """
    lines = [f"=== Step {obs.step} ==="]
    # Patients
    if obs.patients:
        lines.append("\n## Active Patients:")
        for p in obs.patients:
            lines.append(
                f"  {p.id}: {p.condition} | "
                f"status={p.status} | "
                f"blood={p.blood_type or '?'} | "
                f"assigned_to={p.assigned_hospital or 'unassigned'}"
            )
    # Ambulances
    if obs.ambulances:
        lines.append("\n## Ambulances:")
        for a in obs.ambulances:
            lines.append(
                f"  {a.id} ({a.equipment}): status={a.status}"
                + (f" | heading to {a.target_node}" if a.target_node else "")
                + (f" | eta={a.eta_steps}min" if a.eta_steps else "")
            )
    # Hospitals
    if obs.hospitals:
        lines.append("\n## Hospitals:")
        for h in obs.hospitals:
            lines.append(
                f"  {h.id}: OR={h.or_status} | specialists={[s.status for s in h.specialists]}"
            )
    # Blood banks
    if obs.blood_banks:
        lines.append("\n## Blood Banks:")
        for b in obs.blood_banks:
            lines.append(
                f"  {b.hospital_id}: " +
                " | ".join(f"{bt}={q}" for bt, q in b.stock.items())
            )
    # Alerts
    if obs.alerts:
        lines.append("\n## Alerts:")
        for a in obs.alerts[-5:]:  # last 5 alerts
            lines.append(f"  {a}")
    lines.append(f"\nTime score preview: {obs.time_score_preview:.2f}")
    lines.append(f"Mutual aid remaining: {obs.mutual_aid_remaining}")
    return "\n".join(lines)


def parse_agent_response(response) -> CodeRedAction:
    """
    Extract a CodeRedAction from the OpenAI Responses API output.

    Args:
        response: openai.responses.create() response object
            - response.output is a list of OutputItem
            - each OutputItem.content is a list of ContentPart
            - ContentPart.type == "function_call" has name + arguments (JSON string)

    Returns:
        CodeRedAction subclass instance
    """
    import json

    for item in response.output:
        if hasattr(item, "content"):
            for part in item.content:
                if part.type == "function_call":
                    name = part.name
                    args = json.loads(part.arguments) if isinstance(part.arguments, str) else part.arguments
                    return _name_to_action(name, args)
    # Fallback: maintain plan
    return {"action_type": "maintain_plan"}


def _name_to_action(name: str, args: dict) -> CodeRedAction:
    """Map OpenAI function name to CodeRedAction."""
    from codered_env.models import (
        DispatchAmbulance, PrepareOR, PageSpecialist, AssignHospital,
        PreemptOR, AllocateBlood, TransferBlood, RequestMutualAid,
        QueryBloodType, QueryORStatus, MaintainPlan,
    )
    dispatch_map = {
        "dispatch_ambulance": lambda a: DispatchAmbulance(**a),
        "prepare_or": lambda a: PrepareOR(**a),
        "page_specialist": lambda a: PageSpecialist(**a),
        "assign_hospital": lambda a: AssignHospital(**a),
        "preempt_or": lambda a: PreemptOR(**a),
        "allocate_blood": lambda a: AllocateBlood(**a),
        "transfer_blood": lambda a: TransferBlood(**a),
        "request_mutual_aid": lambda a: RequestMutualAid(**a),
        "query_blood_type": lambda a: QueryBloodType(**a),
        "query_or_status": lambda a: QueryORStatus(**a),
    }
    if name == "maintain_plan":
        return MaintainPlan()
    if name in dispatch_map:
        return dispatch_map[name](args)
    return MaintainPlan()


def run_baseline_agent(task_id: Literal["task1", "task2", "task3"], seed: int, api_key: str) -> float:
    """
    Run the OpenAI API baseline agent on a given task and seed.
    Returns the rubric final_score (0.0–1.0).

    Args:
        task_id: Task identifier matching TASK_CONFIG keys ("task1", "task2", "task3")
        seed: Random seed for environment reproducibility
        api_key: OpenAI API key

    Returns:
        Final score from the grader
    """
    import openai
    from openenv import EnvClient
    from codered_env.server.grader import grade_from_environment

    openai.api_key = api_key

    env = EnvClient("codered_env")
    obs = env.reset(seed=seed, task_id=task_id)

    for _ in range(30):
        prompt = format_observation(obs)

        response = openai.responses.create(
            model="gpt-4o",
            input=prompt,
            tools=FUNCTIONS,
            tool_choice="auto",
            instructions="You are a medical emergency coordinator. Use the available tools to save patients."
        )

        action = parse_agent_response(response)
        obs = env.step(action)
        if getattr(obs, "done", False):
            break

    # Grade the episode
    result = grade_from_environment(env)
    return result.final_score
```

- [ ] **Step 1: Write baseline tests with mocked OpenAI**

```python
# codered_env/tests/test_baseline.py
from unittest.mock import patch, MagicMock

def test_baseline_agent_runs_episode():
    mock_response = MagicMock()
    mock_response.output = [
        MagicMock(
            content=[{"type": "function_call", "name": "dispatch_ambulance", "arguments": '{"ambulance_id":"AMB_1","patient_id":"P0","destination_node":"INDUSTRIAL_ZONE"}'}]
        )
    ]

    with patch("openai.responses.create", return_value=mock_response):
        score = run_baseline_agent(task_id="task1", seed=0, api_key="sk-test")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest codered_env/tests/test_baseline.py -v`
Expected: FAIL

- [ ] **Step 3: Implement baseline.py**

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest codered_env/tests/test_baseline.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add codered_env/baseline.py codered_env/tests/test_baseline.py
git commit -m "feat: add OpenAI baseline agent with function-calling tools"
```

---

## Task 18: Dockerfile Completion & HF Spaces Deployment

**Files:**
- Modify: `codered_env/server/Dockerfile`
- Modify: `codered_env/openenv.yaml`
- Create: `codered_env/README.md`

### Dockerfile Completion

```dockerfile
# codered_env/server/Dockerfile

FROM ghcr.io/facebookresearch/openenv/openenv-base:latest  # NOTE: Use meta-pytorch variant if deploying on Meta infrastructure; facebookresearch is canonical

WORKDIR /app

# Install dependencies
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-install-project

# Copy application
COPY . .

# Environment variables for HF Spaces
ENV HOST=0.0.0.0
ENV PORT=8000
ENV HF_SPACE=1
ENV OPENAI_API_KEY=${OPENAI_API_KEY:-}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/tasks')"

EXPOSE 8000

# Run with uvicorn
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### openenv.yaml — HF Spaces Metadata

```yaml
# codered_env/openenv.yaml
spec_version: "1"
app: server.app:app

meta:
  name: CodeRedEnv
  description: Emergency medical coordination simulation for RL agents
  author: Darsh
  license: Apache-2.0
  tags:
    - openenv
    - healthcare
    - reinforcement-learning
    - emergency-response
    - medical
  hyperparameters:
    task_ids: ["task1", "task2", "task3"]
    max_episode_steps: 30
    observation:
      type: object
      properties:
        patients: {...}
        ambulances: {...}
        hospitals: {...}
        blood_banks: {...}
        step_count: integer
        time_score_preview: float
```

### README.md

```markdown
# CodeRedEnv — Emergency Medical Coordination Simulation

An OpenEnv-compatible RL environment where an AI agent coordinates emergency medical
response across a fictional Indian city (Prakashnagar).

## Quick Start

### Local Development

uv sync
uv run pytest codered_env/tests/ -v

### Docker

docker build -f server/Dockerfile -t codered-env .
docker run -p 8000:8000 -e OPENAI_API_KEY=sk-... codered-env

### Validate

openenv validate codered_env/

## Environment Overview

- **City**: 12-node hub-and-spoke network in Prakashnagar
- **Hospitals**: 3-tiered (AIIMS, District, Community HC)
- **Ambulances**: 5 vehicles (2 ALS, 3 BLS)
- **Patients**: Cardiac, Stroke, Trauma, General emergencies

## Tasks

| Task | Description | Patients | Disruptions | Mutual Aid |
|------|-------------|----------|-------------|------------|
| 1 | Single cardiac emergency | 1 cardiac | None | No |
| 2 | Multi-patient emergency | 3 mixed | 2 events | 1 call (12 min) |
| 3 | Crisis surge | 5 patients | 3 events | 2 staggered calls |

## Grading

Final score = 0.40×time_score + 0.20×efficiency + 0.20×secondary_harm + 0.20×prep_ready − mutual_aid_penalty

## Baseline Agent

```python
from codered_env import run_baseline_agent

score = run_baseline_agent(
    task_id="task1",
    seed=0,
    api_key="sk-your-key"
)
```

Or use the `/baseline` endpoint on the deployed Space.

## Citation

@misc{coderedenv2026,
  title={CodeRedEnv: Emergency Medical Coordination Simulation},
  author={Darsh},
  year={2026}
}
```

- [ ] **Step 1: Update Dockerfile with HEALTHCHECK and ENV vars**

- [ ] **Step 2: Update openenv.yaml with HF Spaces metadata**

- [ ] **Step 3: Write README.md**

- [ ] **Step 4: Test Docker build locally**

Run: `docker build -f codered_env/server/Dockerfile -t codered-env-test .`
Expected: Build succeeds, image contains app

- [ ] **Step 5: Test openenv validate**

Run: `openenv validate codered_env/`
Expected: PASS — all schema checks green

- [ ] **Step 6: Commit**

```bash
git add codered_env/server/Dockerfile codered_env/openenv.yaml codered_env/README.md
git commit -m "feat: complete Dockerfile, HF Spaces metadata, and README"
```

---

## Task 19: End-to-End Integration Test

**Files:**
- Create: `codered_env/tests/test_e2e.py`

- [ ] **Step 1: Write e2e test**

```python
# codered_env/tests/test_e2e.py
import pytest
from openenv import EnvClient

def test_task1_e2e():
    """Task 1: single cardiac patient, no disruptions."""
    env = EnvClient("codered_env")
    obs = env.reset(seed=42, task_id="task1")
    assert obs.step == 1
    assert len(obs.patients) == 1
    assert obs.patients[0].condition == "CARDIAC"

    # Take 10 steps with maintain_plan dummy
    for _ in range(10):
        obs = env.step({"action_type": "maintain_plan"})
        if obs.done:
            break
    # Environment should not crash
    assert obs.step >= 1

def test_task2_e2e():
    """Task 2: multi-patient with disruptions."""
    env = EnvClient("codered_env")
    obs = env.reset(seed=0, task_id="task2")
    assert obs.step == 1
    assert len(obs.patients) >= 3  # cardiac + stroke + trauma

    for _ in range(10):
        obs = env.step({"action_type": "maintain_plan"})
        if obs.done:
            break
    assert obs.step >= 1

def test_task3_e2e():
    """Task 3: crisis surge, 5 patients, mutual aid available."""
    env = EnvClient("codered_env")
    obs = env.reset(seed=7, task_id="task3")
    assert obs.step == 1
    assert len(obs.patients) >= 5

    for _ in range(15):
        obs = env.step({"action_type": "maintain_plan"})
        if obs.done:
            break
    assert obs.step >= 1

def test_all_tasks_reset_and_step():
    """All 3 tasks reset and step without crashing."""
    env = EnvClient("codered_env")
    for task_id in ["task1", "task2", "task3"]:
        obs = env.reset(seed=42, task_id=task_id)
        assert obs.step == 1
        # One step should not crash
        obs = env.step({"action_type": "maintain_plan"})
        assert obs.step == 2
```

- [ ] **Step 2: Run e2e tests**

Run: `pytest codered_env/tests/test_e2e.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add codered_env/tests/test_e2e.py
git commit -m "test: add end-to-end integration tests for all tasks"
```

---

## Summary: All Plans Together

| Plan | Tasks | Focus |
|------|-------|-------|
| **Plan 1** | 1–5 | Foundation: scaffold, data, models, env, OpenEnv validation |
| **Plan 2** | 6–13 | Actions & Subsystems: routing, patient lifecycle, ambulance fleet, hospitals, blood, disruptions, wiring, integration |
| **Plan 3** | 14–19 | Grading, API, Baseline, Deployment |

**All 19 tasks across 3 plans = complete CodeRedEnv implementation.**
