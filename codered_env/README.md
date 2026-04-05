# CodeRedEnv — Emergency Medical Coordination Simulation

**A multi-subsystem OpenEnv-compatible RL benchmark** where an AI agent coordinates emergency
medical response across Prakashnagar, India — a 12-node hub-and-spoke city with 3 hospital tiers
and 5 ambulances.

## Environment Description

CodeRedEnv simulates the critical first-30-minutes of a mass-casualty emergency response.
The agent must prioritize, dispatch, triage, and resource-manage under time pressure and
disruption events. It combines ambulance routing, hospital OR preparation, specialist paging,
blood bank allocation, and mutual aid coordination into a single RL benchmark.

**Motivation:** Emergency coordination is a real-world domain where sequential
decision-making under uncertainty directly maps to RL formulation. Unlike toy benchmarks,
CodeRedEnv has non-trivial state, cascading consequences, and resource contention — all
real challenges faced by EMS dispatchers.

## Action & Observation Spaces

### Actions (14 total)

| Action | Description | Phase |
|--------|-------------|-------|
| `dispatch_ambulance` | Dispatch ambulance to patient node (Phase 1) | 1 |
| `dispatch_als` | Dispatch ALS ambulance to pending 911 call (Phase 2) | 2 |
| `dispatch_bls` | Dispatch BLS ambulance to pending 911 call (Phase 2) | 2 |
| `triage_call` | Classify a pending 911 call | 2 |
| `assign_hospital` | Assign patient to destination hospital | 1+2 |
| `prepare_or` | Begin OR preparation at hospital | 1+2 |
| `page_specialist` | Page specialist at hospital | 1+2 |
| `preempt_or` | Clear an OR for emergency use | 1+2 |
| `allocate_blood` | Allocate blood units for patient | 1+2 |
| `transfer_blood` | Transfer blood between hospitals | 1+2 |
| `request_mutual_aid` | Request mutual aid ambulance | 1+2 |
| `query_blood_type` | Query patient blood type (5-min delay) | 1+2 |
| `query_or_status` | Query detailed OR status | 1+2 |
| `maintain_plan` | No-op | 1+2 |

### Observations

Full state includes: active patients with vitals/location/status, ambulance positions/status/ETAs,
hospital OR/specialist/ICU availability, pending 911 calls, blood bank stock, road network status,
disruption alerts, and score previews.

## Tasks

| ID | Name | Patients | Disruptions | Mutual Aid | Difficulty |
|----|------|----------|-------------|-----------|-----------|
| `task1` | Cardiac Emergency | 1 cardiac | None | No | Beginner |
| `task2` | Multi-Patient Emergency | 2 (cardiac+stroke) | 1 event | 1 call | Intermediate |
| `task3` | Crisis Surge | 5 patients | 2 events | 2 calls | Advanced |
| `task4` | Call Queue — Triage | Queue (Phase 2) | 1 event | 1 call | Advanced |
| `task5` | Cascade Crisis | Queue + cascades | 2 events | 2 calls | Expert |

## Grading Rubric

```
final_score = 0.32 × time_score
            + 0.16 × efficiency
            + 0.16 × secondary_harm
            + 0.16 × prep_ready
            + 0.10 × vitals_score_avg
            + 0.10 × cascade_score
            − mutual_aid_penalty
```

- **Time score (32%):** How close to the target time each patient was treated
- **Efficiency (16%):** Avoid wasted specialist pages, unused OR preps, premature blood draws
- **Secondary harm (16%):** Whether secondary patients spawned by cascade events survived
- **Prep ready (16%):** Whether hospitals had ORs and specialists ready when patients arrived
- **Vitals score (10%):** Average vitals score of patients at time of treatment
- **Cascade score (10%):** Secondary patient management, overcrowding prevention, news cycle handling

## Setup & Usage

### Local Development

```bash
cd codered_env
uv sync
uv run python -m pytest tests/ -v
uv run python inference.py --task task1 --seed 0
```

### Docker (recommended for deployment)

```bash
docker build -f Dockerfile -t codered-env .
docker run -p 8000:8000 \
  -e HF_TOKEN=hf_... \
  -e API_BASE_URL=https://router.huggingface.co/v1 \
  -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
  codered-env
```

### OpenEnv Validate

```bash
uv run openenv validate
```

## Inference

### Environment Variables

Before running inference, set these variables:

```bash
# Required
export HF_TOKEN=hf_...                        # Your API key

# Optional (defaults shown)
export API_BASE_URL=https://router.huggingface.co/v1  # LLM endpoint
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct   # Model identifier
export BENCHMARK=codered_env                   # Benchmark name for logs
export MAX_STEPS=30                            # Max episode steps
export SUCCESS_THRESHOLD=0.1                   # Score threshold for success
```

### Running

```bash
HF_TOKEN=... python inference.py --task task1 --seed 0 --max-steps 30
HF_TOKEN=... python inference.py --task task2 --seed 0 --max-steps 45
HF_TOKEN=... python inference.py --task task3 --seed 0 --max-steps 60
HF_TOKEN=... python inference.py --task task4 --seed 0 --max-steps 45
HF_TOKEN=... python inference.py --task task5 --seed 0 --max-steps 60
```

### STDOUT Format

The inference script emits structured logs for automated evaluation:

```
[START] task=task1 env=codered_env model=Qwen/Qwen2.5-72B-Instruct
[STEP]  step=1 action=dispatch_ambulance(...) reward=0.00 done=false error=null
[STEP]  step=2 action=assign_hospital(...) reward=0.50 done=false error=null
...
[END]   success=true steps=8 rewards=0.00,0.00,0.50,0.50,0.50,0.50,0.50,0.50
```

## Baseline Scores

Expected baseline (gpt-5-nano, seeds 0-2, no tuning):

| Task | Mean Score | Notes |
|------|-----------|-------|
| task1 | ~0.3–0.6 | Simple single-patient dispatch |
| task2 | ~0.2–0.5 | Multi-patient coordination |
| task3 | ~0.1–0.3 | Mass casualty, high disruption |
| task4 | ~0.2–0.4 | Dispatch triage learning |
| task5 | ~0.1–0.2 | Full cascade management |

## Deployment

The environment deploys as a **containerized Hugging Face Space** tagged with `openenv`:

```bash
# Build and push
docker build -f codered_env/Dockerfile -t ghcr.io/<user>/codered-env:latest .
docker push ghcr.io/<user>/codered-env:latest

# Or use the HF Space SDK
huggingface_hub.upload_folder(...)
```

Set `HF_TOKEN` as a secret in your HF Space settings for gated model access.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/reset` | Start new episode (seed, task_id) |
| `POST` | `/step` | Execute action, return observation |
| `GET` | `/state` | Current environment state |
| `GET` | `/tasks` | List all 5 task definitions |
| `POST` | `/grader` | Run dummy agent and grade episode |
| `POST` | `/baseline` | Run OpenAI baseline on seeds 0-2 |

## Citation

```
@misc{coderedenv2026,
  title={CodeRedEnv: Emergency Medical Coordination Simulation},
  author={Darsh},
  year={2026}
}
```
