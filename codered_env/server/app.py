"""FastAPI application for CodeRedEnv with API endpoints."""

import os
from dataclasses import asdict
from typing import Literal

try:
    from openenv.core.env_server.http_server import create_app
except ImportError as e:
    raise ImportError("openenv is required. Install with: uv sync") from e

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from server.codered_environment import CodeRedEnvironment
from server.grader import grade_from_environment, RubricResult
from server.models import CodeRedAction, CodeRedObservation
from server.subsystems.constants import TASK_CONFIG


# Create the base app with OpenEnv
_base_app = create_app(
    CodeRedEnvironment,
    CodeRedAction,
    CodeRedObservation,
    env_name="codered_env",
    max_concurrent_envs=4,
)

# Create a new app that includes both OpenEnv routes and our custom endpoints
app = FastAPI(title="CodeRedEnv API")

# Include OpenEnv routes
app.include_router(_base_app.router)


# =============================================================================
# Task definitions
# =============================================================================

TASK_DEFINITIONS = {
    "task1": {
        "task_id": "task1",
        "name": "Cardiac Emergency — Single Patient",
        "description": "A single cardiac patient requires immediate emergency response. No disruptions, no secondary patients.",
        "max_steps": 30,
        "disruption_schedule": [],
        "patients": [{"condition": "cardiac", "spawn_step": 1}],
    },
    "task2": {
        "task_id": "task2",
        "name": "Multi-Patient Emergency",
        "description": "Cardiac and stroke patients arrive in quick succession. One disruption event occurs.",
        "max_steps": 45,
        "disruption_schedule": [{"step": 20, "type": "road_closure"}],
        "patients": [
            {"condition": "cardiac", "spawn_step": 1},
            {"condition": "stroke", "spawn_step": 5},
        ],
        "mutual_aid": {"calls": 1, "stagger_min_steps": 8},
    },
    "task3": {
        "task_id": "task3",
        "name": "Crisis Surge — Mass Casualty",
        "description": "Five patients arrive in rapid succession with multiple disruptions. Two mutual aid calls available.",
        "max_steps": 60,
        "disruption_schedule": [
            {"step": 15, "type": "accident"},
            {"step": 30, "type": "road_closure"},
        ],
        "patients": [
            {"condition": "cardiac", "spawn_step": 1},
            {"condition": "cardiac", "spawn_step": 3},
            {"condition": "stroke", "spawn_step": 5},
            {"condition": "trauma", "spawn_step": 8},
            {"condition": "general", "spawn_step": 12},
        ],
        "mutual_aid": {"calls": 2, "stagger_min_steps": 8},
    },
    "task4": {
        "task_id": "task4",
        "name": "Call Queue — Dispatch Triage (Phase 2)",
        "description": "Incoming 911 calls arrive via dispatch queue. Agent must triage and dispatch appropriately. One disruption event.",
        "max_steps": 45,
        "disruption_schedule": [{"step": 20, "type": "road_closure"}],
        "patients": [],
        "use_call_queue": True,
        "mutual_aid": {"calls": 1, "stagger_min_steps": 8},
    },
    "task5": {
        "task_id": "task5",
        "name": "Cascade Crisis — Full Phase 2",
        "description": "Full Phase 2 simulation with dispatch triage, secondary patient cascades, overcrowding effects, and news cycle surges.",
        "max_steps": 60,
        "disruption_schedule": [
            {"step": 15, "type": "accident"},
            {"step": 30, "type": "road_closure"},
        ],
        "patients": [],
        "use_call_queue": True,
        "cascade_enabled": True,
        "mutual_aid": {"calls": 2, "stagger_min_steps": 8},
    },
}


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/tasks")
async def get_tasks() -> dict:
    """Return task definitions for all 5 tasks."""
    return {
        "tasks": [TASK_DEFINITIONS[tid] for tid in ["task1", "task2", "task3", "task4", "task5"]]
    }


class GraderRequest(BaseModel):
    task_id: Literal["task1", "task2", "task3", "task4", "task5"]
    seed: int


@app.post("/grader")
async def grade_task(req: GraderRequest) -> dict:
    """Run a dummy agent episode and return the rubric score."""
    env = CodeRedEnvironment()
    env.reset(seed=req.seed, task_id=req.task_id)

    # Run a simple baseline agent
    from server.models.actions import MaintainPlan
    done = False
    steps = 0
    while not done and steps < env.state.max_steps:
        env.step(MaintainPlan())
        done = env._check_done()
        steps += 1

    result = grade_from_environment(env)
    return asdict(result)


class BaselineRequest(BaseModel):
    task_id: Literal["task1", "task2", "task3", "task4", "task5"]
    openai_api_key: str | None = None
    model: str | None = None


@app.post("/baseline")
async def run_baseline(req: BaselineRequest) -> dict:
    """Run OpenAI baseline agent on seeds [0, 1, 2] and return scores."""
    try:
        from inference import run_agent as run_baseline_agent
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="Inference agent not implemented yet. Run manually with inference.py"
        )

    scores = []
    for seed in [0, 1, 2]:
        try:
            score = run_baseline_agent(
                task_id=req.task_id, seed=seed,
                model=req.model or None,
            )
            scores.append(score)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Baseline failed: {e}")

    return {
        "task_id": req.task_id,
        "model": os.environ.get("OPENAI_MODEL", "gpt-5-nano"),
        "scores": scores,
        "mean": sum(scores) / len(scores) if scores else 0.0,
    }


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main()  # main() call for openenv validate
