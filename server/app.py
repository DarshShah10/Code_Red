import os
from dataclasses import asdict
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from server.codered_environment import CodeRedEnvironment
from server.grader import grade_from_environment, RubricResult
from server.models import CodeRedAction, CodeRedObservation
from server.subsystems.constants import TASK_CONFIG

# Create the base app with OpenEnv (disabled — gradio pulls in heavy deps that block startup on HF)
_base_app = None

# Create the main app
app = FastAPI(title="CodeRedEnv API")

# Include OpenEnv routes if available
if _base_app is not None:
    app.include_router(_base_app.router)


# =============================================================================
# ENVIRONMENT STATE (GLOBAL INSTANCE)
# =============================================================================

env = CodeRedEnvironment()


# Root endpoint providing a quick overview of all available API routes
@app.get("/")
async def root() -> dict:
    """Return a JSON overview of the API endpoints."""
    return {
        "endpoints": [
            "/health",
            "/info",
            "/tasks",
            "/reset",
            "/step",
            "/state",
            "/grade",
            "/inference",
        ]
    }


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

@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {"status": "healthy", "environment": "CodeRedEnv"}


@app.get("/info")
async def info():
    """Return application metadata."""
    return {
        "name": "CodeRedEnv",
        "description": "Emergency Medical Coordination Environment for OpenEnv",
        "version": "0.1.0",
        "endpoints": ["/health", "/reset", "/step", "/state", "/grade", "/info"],
        "tasks": ["task1", "task2", "task3", "task4", "task5"],
    }


@app.get("/tasks")
async def get_tasks() -> dict:
    """Return task definitions for all 5 tasks."""
    return {
        "tasks": [TASK_DEFINITIONS[tid] for tid in ["task1", "task2", "task3", "task4", "task5"]]
    }


# ---------------- RESET ----------------
class ResetRequest(BaseModel):
    task_id: Literal["task1", "task2", "task3", "task4", "task5"] = "task1"
    seed: int = 0


@app.post("/reset")
async def reset_env(req: ResetRequest):
    """
    Reset the environment and return initial observation.
    """
    try:
        obs = env.reset(seed=req.seed, task_id=req.task_id)

        return {
            "observation": obs,
            "done": False,
            "info": {
                "task_id": req.task_id,
                "seed": req.seed
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------- STEP ----------------
class StepRequest(BaseModel):
    action: dict


@app.post("/step")
async def step_env(req: StepRequest):
    """
    Apply action → return (obs, reward, done, info)
    """
    try:
        action = CodeRedAction(**req.action)

        obs, reward, done, info = env.step(action)

        return {
            "observation": obs,
            "reward": reward,
            "done": done,
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------- STATE ----------------
@app.get("/state")
async def get_state():
    """
    Return current environment state WITHOUT stepping.
    """
    try:
        return {
            "state": env.state,
            "done": env._check_done() if hasattr(env, "_check_done") else False
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------- GRADING ----------------
class GraderRequest(BaseModel):
    task_id: Literal["task1", "task2", "task3", "task4", "task5"]
    seed: int


@app.post("/grade")
async def grade_task(req: GraderRequest) -> dict:
    """Run a dummy agent episode and return the rubric score."""
    test_env = CodeRedEnvironment()
    test_env.reset(seed=req.seed, task_id=req.task_id)

    # Run a simple baseline agent
    from server.models.actions import MaintainPlan
    done = False
    steps = 0
    while not done and steps < test_env.state.max_steps:
        test_env.step(MaintainPlan())
        done = test_env._check_done()
        steps += 1

    result = grade_from_environment(test_env)
    return asdict(result)


# Alias for backward compat with spec
@app.post("/grader", include_in_schema=False)
async def grade_task_alias(req: GraderRequest) -> dict:
    """Alias for /grade. DEPRECATED — use /grade instead."""
    return await grade_task(req)


# =============================================================================
# Inference endpoint — streams real-time agent output + final grade
# =============================================================================

class InferenceRequest(BaseModel):
    task_id: Literal["task1", "task2", "task3", "task4", "task5"] = "task1"
    seed: int = 0
    max_steps: int = 30
    provider: Literal["openai", "anthropic", "hf_fallback", "auto"] = "auto"
    model: str | None = None


@app.post("/inference")
async def run_inference(req: InferenceRequest):
    """
    Run the LLM agent on a task and stream output line-by-line (SSE).
    Each line is one of: [START], [STEP], [END], [GRADE], [BREAKDOWN], [ERROR].
    Connect via EventSource in the browser, or curl with --no-buffer.
    """
    import asyncio
    import subprocess

    cmd = [
        "python", "inference.py",
        "--task", req.task_id,
        "--seed", str(req.seed),
        "--max-steps", str(req.max_steps),
    ]
    if req.provider != "auto":
        cmd.extend(["--provider", req.provider])
    if req.model:
        cmd.extend(["--model", req.model])

    async def event_stream():
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=os.path.dirname(os.path.dirname(__file__)) or "/app",
            )
            # Stream stdout lines as SSE events
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                decoded = line.decode("utf-8", errors="replace").rstrip("\n")
                if decoded:
                    yield f"data: {decoded}\n\n"
            # Stream stderr (warnings/errors) as error events
            stderr = await proc.stderr.read()
            if stderr:
                decoded_err = stderr.decode("utf-8", errors="replace").rstrip("\n")
                for err_line in decoded_err.splitlines():
                    if err_line.strip():
                        yield f"event: error\ndata: {err_line}\n\n"
            # Send final done event
            yield "event: done\ndata: \n\n"
        except Exception as e:
            yield f"event: error\ndata: {str(e)}\n\n"

    from starlette.responses import StreamingResponse
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no"},
    )


def main(host: str = "0.0.0.0", port: int = 7860):
    import uvicorn
    uvicorn.run(app, host=host, port=port, workers=1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    main(port=args.port)  # main() call for openenv validate