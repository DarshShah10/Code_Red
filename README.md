# CodeRedEnv — Emergency Medical Coordination Simulation

**A multi-subsystem OpenEnv-compatible RL benchmark** where an AI agent coordinates emergency medical response across Prakashnagar, India — a 12-node hub-and-spoke city with 3 hospital tiers and 5 ambulances.

> **Live:** [huggingface.co/spaces/darshshah1012/coderedenv](https://huggingface.co/spaces/darshshah1012/coderedenv) |
> **Grading:** Rubric-based (0.0–1.0), never escapes bounds

---

## Tasks

| ID | Name | Patients | Disruptions | Mutual Aid | Max Steps |
|----|------|----------|-------------|-----------|-----------|
| `task1` | Cardiac Emergency — Single Patient | 1 cardiac | None | No | 30 |
| `task2` | Multi-Patient Emergency | 2 (cardiac+stroke) | 1 road closure | 1 call | 45 |
| `task3` | Crisis Surge — Mass Casualty | 5 patients | 2 events | 2 calls | 60 |
| `task4` | Call Queue — Dispatch Triage | 911 call queue | 1 road closure | 1 call | 45 |
| `task5` | Cascade Crisis — Full Phase 2 | Queue + cascades | 2 events | 2 calls | 60 |

## Actions (14 total)

**Phase 1** (`task1`–`task3`): direct patient dispatch
`dispatch_ambulance` · `assign_hospital` · `prepare_or` · `page_specialist` · `preempt_or` · `allocate_blood` · `transfer_blood` · `request_mutual_aid` · `query_blood_type` · `query_or_status` · `maintain_plan`

**Phase 2** (`task4`–`task5`): 911 call triage before dispatch
`dispatch_als` · `dispatch_bls` · `triage_call` · (all above actions)

## Grading Rubric

Score always in `[0.0, 1.0]` — never escapes bounds.

```
final_score = max(0.0, min(1.0, raw − deductions))
raw = 0.30 × time_score
    + 0.20 × efficiency
    + 0.20 × secondary_harm
    + 0.15 × prep_ready
    + 0.15 × cascade_score
```

- **time_score (30%):** Treatment time vs. clinical target per patient
- **efficiency (20%):** Ratio of successful actions to wasted actions
- **secondary_harm (20%):** Survival rate of cascade-spawned secondary patients
- **prep_ready (15%):** OR + specialist readiness when patients arrive at hospital
- **cascade_score (15%):** Secondary patient management, overcrowding, news cycles

**Deductions (subtracted):** Mutual aid timing penalty, triage misclassification penalty, cross-validation penalty, ICU boarding penalty. Success threshold: **≥ 0.20**.

## Quick Start

### Local (Python)

```bash
cd codered_env
uv sync
python inference.py --task task1 --max-steps 5 --provider anthropic
```

### Docker

```bash
docker build -t codered-env:latest .
docker run -p 8000:8000 --env-file .env codered-env:latest
```

Set API keys in `.env` (copy from `.env.sample`). Provider priority: **OpenAI → Anthropic → HF fallback**.

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/tasks` | All 5 task definitions |
| `POST` | `/reset` | Start new episode (seed, task_id) |
| `POST` | `/step` | Execute action → observation |
| `GET` | `/state` | Current environment state |
| `POST` | `/grade` | Run baseline agent, return rubric score |
| `POST` | `/baseline` | Run agent on seeds [0,1,2], return mean score |
| `POST` | `/inference` | **Streaming SSE** — run agent with real-time output |

### Streaming Inference (`/inference`)

```bash
curl -N http://localhost:8000/inference \
  -X POST -H "Content-Type: application/json" \
  -d '{"task_id":"task1","max_steps":5,"provider":"anthropic"}'
```

Output streams as Server-Sent Events:
```
data: [START] task=task1 env=codered_env model=claude-sonnet-4-6
data: [STEP] step=1 action=dispatch_ambulance(...) reward=0.20 done=false
data: [STEP] step=2 action=assign_hospital(...) reward=0.20 done=false
...
data: [END] success=true steps=5 rewards=0.20,0.20,0.20,0.20,0.20
data: [GRADE] score=0.5500
event: done
```

## Inference CLI

```bash
# Default: OpenAI (OpenRouter), 30 steps
python inference.py --task task1

# Force Anthropic, 5 steps
python inference.py --task task1 --max-steps 5 --provider anthropic

# HuggingFace fallback
python inference.py --task task2 --provider hf_fallback

# Multi-seed benchmark
for seed in 0 1 2; do
  python inference.py --task task1 --seed $seed --max-steps 30
done
```

## Environment Variables

```bash
# Provider order: OPENAI > ANTHROPIC > HF_FALLBACK
OPENAI_API_KEY=           # OpenRouter / Groq key (priority 1)
OPENAI_API_BASE_URL=https://openrouter.ai/api/v1
OPENAI_MODEL_NAME=qwen/qwen3.6-plus

ANTHROPIC_API_KEY=        # OpusCode / Anthropic direct (priority 2)
ANTHROPIC_API_BASE_URL=https://claude.opuscode.pro/api
ANTHROPIC_MODEL_NAME=claude-sonnet-4-6

HF_TOKEN=                 # HuggingFace (last resort fallback)

BENCHMARK=codered_env
TASK_NAME=task1
MAX_STEPS=30
SUCCESS_THRESHOLD=0.2
LOCAL_IMAGE_NAME=codered-env:latest
HF_SPACE_REPO=darshshah1012/coderedenv
```

## HuggingFace Space Deployment

The Space at `darshshah1012/coderedenv` is set to **Docker hardware**. Push updated code to the Space repo — HF auto-rebuilds the Docker image from `Dockerfile`.

```bash
# Authenticate
hf login

# Push codered_env/ directory to the Space
hf upload . --repo-id darshshah1012/coderedenv --repo-type space

# Or clone, commit, push
git clone https://huggingface.co/spaces/darshshah1012/coderedenv
# copy updated code →
git add . && git commit -m "update" && git push
```

Set secrets (API keys) via **HF Spaces UI → Settings → Repository secrets** — keys are never stored in code.

## Action Space

### Phase 1 — Direct Patient Dispatch (`task1`–`task3`)
| Function | Description |
|----------|-------------|
| `dispatch_ambulance(ambulance_id, patient_id, target_node)` | Send ambulance to patient location |
| `assign_hospital(patient_id, hospital_id)` | Assign destination hospital |
| `prepare_or(hospital_id, procedure_type)` | Begin OR prep (cardiac/stroke/trauma/general) |
| `page_specialist(hospital_id, specialist_type)` | Page specialist (cardiologist/neurologist/trauma_surgeon) |
| `preempt_or(hospital_id, or_index)` | Clear OR room for emergency |
| `allocate_blood(hospital_id, patient_id, blood_type, units)` | Allocate blood units |
| `transfer_blood(from_hospital, to_hospital, blood_type, units)` | Transfer blood between hospitals |
| `request_mutual_aid()` | Request mutual aid ambulance |
| `query_blood_type(patient_id)` | Query patient blood type |
| `query_or_status(hospital_id, or_index)` | Query OR room status |
| `maintain_plan()` | No-op: continue without changes |

### Phase 2 — Call Triage (`task4`–`task5`)
| Function | Description |
|----------|-------------|
| `dispatch_als(ambulance_id, call_id)` | Dispatch ALS ambulance (AMB_1/AMB_2) to 911 call |
| `dispatch_bls(ambulance_id, call_id)` | Dispatch BLS ambulance (AMB_3/4/5) to 911 call |
| `triage_call(call_id, decision, ambulance_id?)` | Classify and dispatch 911 call |

## Observation Space

Key fields returned by the environment:
- `step`: Current simulation step (int)
- `patients[]`: Active patients with condition, status, vitals_score, blood_type, assigned_hospital, location_node
- `ambulances[]`: Ambulance status (idle/en_route/to_hospital/at_hospital), equipment (als/bls), assigned_patient, eta_minutes
- `hospitals[]`: Hospital capabilities, operating_rooms (idle/prep/surgery/occupied), specialists on-call
- `pending_calls[]`: 911 calls with category, severity_est, time_waiting, location_node
- `alerts[]`: System alerts (last 3 shown)
- `time_score_preview`: Preview of time component score
- `vitals_score_preview`: Preview of vitals component score
- `patients_remaining`: Patients still needing treatment
- `mutual_aid_remaining`: Mutual aid requests available
- `overcrowding_modifier`: ED overcrowding factor (>1.0 means crowded)

## Task Difficulty

| Task | Difficulty | Description | Success Threshold |
|------|-----------|-------------|-------------------|
| `task1` | Easy | Single cardiac patient, no disruptions | Score ≥ 0.20 |
| `task2` | Medium | 2 patients + 1 road closure | Score ≥ 0.20 |
| `task3` | Hard | 5 patients + 2 disruptions + mutual aid | Score ≥ 0.20 |
| `task4` | Medium | 911 call queue triage | Score ≥ 0.20 |
| `task5` | Hard | Full cascade crisis | Score ≥ 0.20 |

## Baseline Scores

| Task | Model | Score (0.0–1.0) |
|------|-------|-----------------|
| task1 | qwen/qwen3.6-plus | ~0.35 |
| task2 | qwen/qwen3.6-plus | ~0.25 |
| task3 | qwen/qwen3.6-plus | ~0.20 |

## OpenEnv Validation

```bash
uv run openenv validate
```
