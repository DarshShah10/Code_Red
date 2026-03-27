# CodeRedEnv — Emergency Medical Coordination Simulation

An OpenEnv-compatible RL environment where an AI agent coordinates emergency medical
response across a fictional Indian city (Prakashnagar).

## Quick Start

### Local Development

```bash
cd codered_env
uv sync
uv run pytest tests/ -v
```

### Docker

```bash
docker build -f Dockerfile -t codered-env .
docker run -p 8000:8000 -e OPENAI_API_KEY=sk-... codered-env
```

### Validate

```bash
openenv validate codered_env/
```

## Environment Overview

- **City**: 12-node hub-and-spoke network in Prakashnagar
- **Hospitals**: 3-tiered (AIIMS, District, Community HC)
- **Ambulances**: 5 vehicles (2 ALS, 3 BLS)
- **Patients**: Cardiac, Stroke, Trauma, General emergencies

## Tasks

| Task | Description | Patients | Disruptions | Mutual Aid |
|------|-------------|----------|-------------|------------|
| 1 | Single cardiac emergency | 1 cardiac | None | No |
| 2 | Multi-patient emergency | 2 mixed | 1 event | 1 call |
| 3 | Crisis surge | 5 patients | 2 events | 2 staggered calls |

## Grading

Final score = 0.40×time_score + 0.20×efficiency + 0.20×secondary_harm + 0.20×prep_ready − mutual_aid_penalty

## Baseline Agent

```python
from codered_env import run_baseline_agent

score = run_baseline_agent(
    task_id="task1",
    seed=0,
    # api_key and model read from .env if not passed
)
```

Or use the `/baseline` endpoint on the deployed Space.

## Citation

@misc{coderedenv2026,
  title={CodeRedEnv: Emergency Medical Coordination Simulation},
  author={Darsh},
  year={2026}
}
