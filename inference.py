#!/usr/bin/env python3
"""
CodeRedEnv Inference Script — OpenEnv Spec Compliant

Mandatory environment variables:
    API_BASE_URL   Endpoint for the LLM (default: https://router.huggingface.co/v1)
    MODEL_NAME     Model identifier (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN       HuggingFace / API key (set HF_TOKEN or API_KEY env var)

CLI usage:
    HF_TOKEN=... python inference.py --task task1 --benchmark codered_env
    python inference.py --task task1 --seed 0 --max-steps 30

STDOUT FORMAT (exact — field names, ordering, formatting are enforced):
    [START] task=<task> env=<benchmark> model=<model>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>

All fields on single lines. reward and rewards to 2 decimal places.
done and success are lowercase booleans: true or false.
error is the raw last_action_error string or null.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Config from environment — MUST be read from env vars
# ---------------------------------------------------------------------------
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN: str | None = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
BENCHMARK: str = os.getenv("BENCHMARK", "codered_env")
MAX_STEPS_DEFAULT: int = int(os.getenv("MAX_STEPS", "30"))
SUCCESS_SCORE_THRESHOLD: float = float(os.getenv("SUCCESS_THRESHOLD", "0.1"))


def _get_client():
    """Lazily create the OpenAI client."""
    from openai import OpenAI

    return OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN, timeout=120.0)


# ---------------------------------------------------------------------------
# Function definitions — must match server/models/actions.py exactly
# ---------------------------------------------------------------------------
FUNCTIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "dispatch_ambulance",
        "description": "Dispatch an ambulance to a patient location (Phase 1). Also assign hospital with assign_hospital.",
        "parameters": {
            "type": "object",
            "properties": {
                "ambulance_id": {"type": "string", "enum": ["AMB_1", "AMB_2", "AMB_3", "AMB_4", "AMB_5"]},
                "patient_id": {"type": "string"},
                "target_node": {"type": "string"},
            },
            "required": ["ambulance_id", "patient_id", "target_node"],
        },
    },
    {
        "type": "function",
        "name": "dispatch_als",
        "description": "Dispatch an ALS ambulance to a pending 911 call (Phase 2). Use triage_call first.",
        "parameters": {
            "type": "object",
            "properties": {
                "ambulance_id": {"type": "string", "enum": ["AMB_1", "AMB_2"]},
                "call_id": {"type": "string"},
            },
            "required": ["ambulance_id", "call_id"],
        },
    },
    {
        "type": "function",
        "name": "dispatch_bls",
        "description": "Dispatch a BLS ambulance to a pending 911 call (Phase 2). Use triage_call first.",
        "parameters": {
            "type": "object",
            "properties": {
                "ambulance_id": {"type": "string", "enum": ["AMB_3", "AMB_4", "AMB_5"]},
                "call_id": {"type": "string"},
            },
            "required": ["ambulance_id", "call_id"],
        },
    },
    {
        "type": "function",
        "name": "triage_call",
        "description": "Decide what to do with a pending 911 call: dispatch_als, dispatch_bls, self_transport, callback, no_dispatch.",
        "parameters": {
            "type": "object",
            "properties": {
                "call_id": {"type": "string"},
                "decision": {
                    "type": "string",
                    "enum": ["dispatch_als", "dispatch_bls", "self_transport", "callback", "no_dispatch"],
                },
                "ambulance_id": {"type": "string", "description": "Required when decision is dispatch_als or dispatch_bls"},
            },
            "required": ["call_id", "decision"],
        },
    },
    {
        "type": "function",
        "name": "prepare_or",
        "description": "Begin OR preparation at a hospital for a procedure type.",
        "parameters": {
            "type": "object",
            "properties": {
                "hospital_id": {"type": "string", "enum": ["HOSP_A", "HOSP_B", "HOSP_C"]},
                "procedure_type": {"type": "string", "enum": ["cardiac", "stroke", "trauma", "general"]},
            },
            "required": ["hospital_id", "procedure_type"],
        },
    },
    {
        "type": "function",
        "name": "page_specialist",
        "description": "Page a specialist at a hospital.",
        "parameters": {
            "type": "object",
            "properties": {
                "hospital_id": {"type": "string", "enum": ["HOSP_A", "HOSP_B"]},
                "specialist_type": {"type": "string", "enum": ["cardiologist", "neurologist", "trauma_surgeon"]},
            },
            "required": ["hospital_id", "specialist_type"],
        },
    },
    {
        "type": "function",
        "name": "assign_hospital",
        "description": "Assign a patient to a destination hospital.",
        "parameters": {
            "type": "object",
            "properties": {
                "patient_id": {"type": "string"},
                "hospital_id": {"type": "string", "enum": ["HOSP_A", "HOSP_B", "HOSP_C"]},
            },
            "required": ["patient_id", "hospital_id"],
        },
    },
    {
        "type": "function",
        "name": "preempt_or",
        "description": "Preempt (clear) an operating room for emergency use.",
        "parameters": {
            "type": "object",
            "properties": {
                "hospital_id": {"type": "string", "enum": ["HOSP_A", "HOSP_B"]},
                "or_index": {"type": "integer", "minimum": 0},
            },
            "required": ["hospital_id", "or_index"],
        },
    },
    {
        "type": "function",
        "name": "allocate_blood",
        "description": "Allocate blood units for a patient.",
        "parameters": {
            "type": "object",
            "properties": {
                "hospital_id": {"type": "string", "enum": ["HOSP_A", "HOSP_B", "HOSP_C"]},
                "patient_id": {"type": "string"},
                "blood_type": {"type": "string"},
                "units": {"type": "integer", "minimum": 1},
                "emergency": {"type": "boolean", "default": False},
            },
            "required": ["hospital_id", "patient_id", "blood_type", "units"],
        },
    },
    {
        "type": "function",
        "name": "transfer_blood",
        "description": "Transfer blood units between hospitals.",
        "parameters": {
            "type": "object",
            "properties": {
                "from_hospital": {"type": "string", "enum": ["HOSP_A", "HOSP_B", "HOSP_C"]},
                "to_hospital": {"type": "string", "enum": ["HOSP_A", "HOSP_B", "HOSP_C"]},
                "blood_type": {"type": "string"},
                "units": {"type": "integer", "minimum": 1},
            },
            "required": ["from_hospital", "to_hospital", "blood_type", "units"],
        },
    },
    {
        "type": "function",
        "name": "request_mutual_aid",
        "description": "Request mutual aid ambulance (Task 2/3 only). Has 12-minute arrival latency.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "type": "function",
        "name": "query_blood_type",
        "description": "Query a patient's blood type (takes 5 minutes).",
        "parameters": {
            "type": "object",
            "properties": {"patient_id": {"type": "string"}},
            "required": ["patient_id"],
        },
    },
    {
        "type": "function",
        "name": "query_or_status",
        "description": "Query detailed OR status at a hospital.",
        "parameters": {
            "type": "object",
            "properties": {
                "hospital_id": {"type": "string"},
                "or_index": {"type": "integer", "minimum": 0},
            },
            "required": ["hospital_id", "or_index"],
        },
    },
    {
        "type": "function",
        "name": "maintain_plan",
        "description": "No-op: continue current plan without changes. Use when there is nothing useful to do.",
        "parameters": {"type": "object", "properties": {}},
    },
]

SYSTEM_PROMPT = textwrap.dedent("""\
    You are a medical emergency coordination AI managing ambulance dispatch,
    hospital preparation, and patient triage in Prakashnagar, India.

    City layout:
    - HOSP_A (AIIMS): full capability (cardiac, stroke, trauma, stabilization)
    - HOSP_B (District): cardiac/trauma (no neurologist)
    - HOSP_C (Community HC): stabilization only
    - Ambulances: AMB_1/AMB_2 are ALS (advanced); AMB_3/4/5 are BLS (basic)
    - City nodes: RAJIV_CHOWK, LAJPAT_NAGAR, CHOWKHA, RAILWAY_XING, NH45_BYPASS,
      IT_HUB, MG_CHOWK, SECTOR_12, RING_ROAD

    Priority: CARDIAC > STROKE > TRAUMA > GENERAL

    Phase 2 — Call Queue (task4/task5):
    - Incoming 911 calls appear as pending_calls.
    - Use triage_call to classify each call before dispatching.
    - Ground truth condition is hidden until ambulance arrives.
    - dispatch_als commits ALS (AMB_1/AMB_2); dispatch_bls commits BLS (AMB_3/4/5).

    Phase 1 (task1/task2/task3):
    - Patients spawn directly on the map.
    - Use dispatch_ambulance to send an ambulance to patient location.
    - Also call assign_hospital to assign destination.

    General strategy:
    1. Dispatch nearest available ambulance to each patient.
    2. Call assign_hospital immediately (HOSP_A for cardiac/stroke, HOSP_B for trauma).
    3. Call prepare_or (procedure_type=cardiac/stroke/trauma) at destination hospital.
    4. Call page_specialist if time allows.
    5. Use maintain_plan only when no useful action is available.

    Respond with exactly one function call using the available tools.
    Never leave a patient waiting without an ambulance dispatched.
    """).strip()


# ---------------------------------------------------------------------------
# STDOUT logging — exact format per OpenEnv spec
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)


# ---------------------------------------------------------------------------
# Action parsing
# ---------------------------------------------------------------------------

def _build_action_dict(name: str, args: dict) -> dict:
    """Map function call name to env action dict."""
    mapping = {
        "dispatch_ambulance": lambda a: {"type": "dispatch_ambulance", **a},
        "dispatch_als": lambda a: {"type": "dispatch_als", **a},
        "dispatch_bls": lambda a: {"type": "dispatch_bls", **a},
        "triage_call": lambda a: {"type": "triage_call", **a},
        "prepare_or": lambda a: {"type": "prepare_or", **a},
        "page_specialist": lambda a: {"type": "page_specialist", **a},
        "assign_hospital": lambda a: {"type": "assign_hospital", **a},
        "preempt_or": lambda a: {"type": "preempt_or", **a},
        "allocate_blood": lambda a: {"type": "allocate_blood", **a},
        "transfer_blood": lambda a: {"type": "transfer_blood", **a},
        "request_mutual_aid": lambda _: {"type": "request_mutual_aid"},
        "query_blood_type": lambda a: {"type": "query_blood_type", **a},
        "query_or_status": lambda a: {"type": "query_or_status", **a},
        "maintain_plan": lambda _: {"type": "maintain_plan"},
    }
    return mapping.get(name, lambda _: {"type": "maintain_plan"})(args)


def parse_action_string(action_str: str) -> tuple[dict, str]:
    """
    Parse an LLM response string into (action_dict, display_str).
    Handles: fn_name(k=v, ...) or plain text (fallback to maintain_plan).
    """
    fn_match = re.match(r"^(\w+)\((.*)\)$", action_str.strip())
    if fn_match:
        fn_name = fn_match.group(1)
        args_str = fn_match.group(2)
        fn_args: dict = {}
        if args_str.strip():
            try:
                # Try JSON first
                fn_args = json.loads(f"{{{args_str}}}")
            except Exception:
                # Fallback: parse key=value pairs (handles single-quoted strings)
                for m in re.finditer(r"(\w+)=(['\"]?)([\w_/]+)\2", args_str):
                    fn_args[m.group(1)] = m.group(3)
        action_dict = _build_action_dict(fn_name, fn_args)
        display_str = action_str.strip()
    else:
        action_dict = {"type": "maintain_plan"}
        display_str = "maintain_plan()"
    return action_dict, display_str


# ---------------------------------------------------------------------------
# Environment interaction
# ---------------------------------------------------------------------------

def execute_action(env, action_dict: dict) -> tuple[Any, Optional[str]]:
    """
    Execute action on environment and return (observation, error_or_none).
    error is the alert text if action had a failure.
    """
    from server.models.actions import (
        DispatchAmbulance, DispatchALS, DispatchBLS, TriageCall,
        PrepareOR, PageSpecialist, AssignHospital, PreemptOR,
        AllocateBlood, TransferBlood, RequestMutualAid,
        QueryBloodType, QueryORStatus, MaintainPlan,
    )

    at = action_dict.get("type", "maintain_plan")
    args = {k: v for k, v in action_dict.items() if k != "type"}

    try:
        action: Any
        if at == "dispatch_ambulance":
            action = DispatchAmbulance(**args)
        elif at == "dispatch_als":
            action = DispatchALS(**args)
        elif at == "dispatch_bls":
            action = DispatchBLS(**args)
        elif at == "triage_call":
            action = TriageCall(**args)
        elif at == "prepare_or":
            action = PrepareOR(**args)
        elif at == "page_specialist":
            action = PageSpecialist(**args)
        elif at == "assign_hospital":
            action = AssignHospital(**args)
        elif at == "preempt_or":
            action = PreemptOR(**args)
        elif at == "allocate_blood":
            action = AllocateBlood(**args)
        elif at == "transfer_blood":
            action = TransferBlood(**args)
        elif at == "request_mutual_aid":
            action = RequestMutualAid()
        elif at == "query_blood_type":
            action = QueryBloodType(**args)
        elif at == "query_or_status":
            action = QueryORStatus(**args)
        else:
            action = MaintainPlan()

        obs = env.step(action)

        # Surface failure-type alerts as error strings
        error: Optional[str] = None
        if obs.alerts:
            for alert in obs.alerts:
                if any(kw in alert.lower() for kw in ["failed", "error", "no ", "cannot", "unable", "not available"]):
                    error = alert
                    break
        return obs, error

    except Exception as exc:
        return None, str(exc)


# ---------------------------------------------------------------------------
# Observation formatting
# ---------------------------------------------------------------------------

def format_observation(obs) -> str:
    """Format CodeRedObservation as a concise text prompt for the LLM."""
    lines = [f"=== Step {obs.step} ==="]

    if obs.patients:
        lines.append("## Active Patients:")
        for p in obs.patients:
            if p.status not in ("treated", "deceased"):
                lines.append(
                    f"  {p.patient_id}: {p.condition.value} | "
                    f"status={p.status} | blood={p.blood_type or '?'} | "
                    f"assigned={p.assigned_hospital or 'unassigned'} | "
                    f"location={p.location_node} | vitals={p.vitals_score:.2f}"
                )

    if obs.ambulances:
        lines.append("## Ambulances:")
        for a in obs.ambulances:
            if a.status.value not in ("idle", "at_hospital"):
                lines.append(
                    f"  {a.id} ({a.equipment.value}): {a.status.value}"
                    + (f" | patient={a.assigned_patient}" if a.assigned_patient else "")
                    + (f" | eta={a.eta_minutes}min" if a.eta_minutes else "")
                )
            elif a.status.value == "idle":
                lines.append(f"  {a.id} ({a.equipment.value}): idle at {a.location_node}")

    if obs.hospitals:
        lines.append("## Hospitals:")
        for h in obs.hospitals:
            idle_ors = sum(1 for o in h.operating_rooms if o.status.value == "idle")
            lines.append(
                f"  {h.id}: idle_ors={idle_ors}/{len(h.operating_rooms)}"
                f" | {', '.join(h.capabilities)}"
            )

    if obs.pending_calls:
        lines.append("## Pending 911 Calls:")
        for c in obs.pending_calls:
            lines.append(
                f"  {c.call_id}: {c.category.value} | waiting={c.time_waiting}min"
                f" | severity_est={c.estimated_severity:.1f} | loc={c.location_node}"
            )

    if obs.alerts:
        lines.append("## Alerts:")
        for a in obs.alerts[-3:]:
            lines.append(f"  {a}")

    lines.append(
        f"\nTime score: {obs.time_score_preview:.2f} | "
        f"Vitals: {obs.vitals_score_preview:.2f} | "
        f"Patients remaining: {obs.patients_remaining} | "
        f"Mutual aid remaining: {obs.mutual_aid_remaining}"
    )
    if obs.overcrowding_modifier > 1.0:
        lines.append(f"WARNING: ED overcrowding (modifier={obs.overcrowding_modifier:.1f}x)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def call_model(prompt: str, model: str | None = None) -> str:
    """Call the LLM with tools and return the function call as a string."""
    client = _get_client()
    actual_model = model if model else MODEL_NAME
    try:
        completion = client.chat.completions.create(
            model=actual_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            tools=FUNCTIONS,
            tool_choice="auto",
            temperature=0.0,  # deterministic for reproducible benchmarking
            max_tokens=200,
        )
        msg = completion.choices[0].message
        if msg.tool_calls:
            tc = msg.tool_calls[0]
            fn_name = tc.function.name
            fn_args_str = tc.function.arguments or ""
            try:
                fn_args = json.loads(fn_args_str)
                args_parts = [f"{k}={v!r}" for k, v in fn_args.items()]
                return f"{fn_name}({', '.join(args_parts)})"
            except Exception:
                return f"{fn_name}()"
        text = (msg.content or "").strip()
        return text if text else "maintain_plan()"
    except Exception as exc:
        sys.stderr.write(f"[WARN] Model call failed: {exc}\n")
        return "maintain_plan()"


# ---------------------------------------------------------------------------
# Main run function
# ---------------------------------------------------------------------------

def run_agent(
    task_id: str,
    seed: int = 0,
    max_steps: int = MAX_STEPS_DEFAULT,
    task_name: str | None = None,
    benchmark: str = BENCHMARK,
    model: str | None = None,
    api_key: str | None = None,
) -> float:
    """
    Run the OpenAI-powered agent on the given task.

    Mandatory env vars (overridden by parameters):
        API_BASE_URL   LLM endpoint
        MODEL_NAME     Model identifier
        HF_TOKEN       API key

    Returns the normalized score (0.0-1.0) from the rubric grader.
    """
    from server.codered_environment import CodeRedEnvironment
    from server.grader import grade_from_environment

    actual_model = model or MODEL_NAME
    task = task_name or task_id
    log_start(task=task, env=benchmark, model=actual_model)

    rewards: list[float] = []
    last_error: Optional[str] = None
    env: Any = None

    try:
        env = CodeRedEnvironment()
        obs = env.reset(seed=seed, task_id=task_id)

        for step in range(1, max_steps + 1):
            if env._check_done():
                break

            prompt = format_observation(obs)
            action_str = call_model(prompt, model=actual_model)
            action_dict, display_str = parse_action_string(action_str)

            obs, last_error = execute_action(env, action_dict)

            # If step returned None obs (action failed), skip
            if obs is None:
                last_error = last_error or "action failed"
                rewards.append(0.0)
                log_step(step=step, action=display_str, reward=0.0, done=True, error=last_error)
                break

            reward = env.state.cum_reward
            done = env._check_done()

            rewards.append(round(reward, 2))
            log_step(
                step=step,
                action=display_str,
                reward=reward,
                done=done,
                error=last_error,
            )

            if done:
                break

    except Exception as exc:
        sys.stderr.write(f"[ERROR] Episode crashed: {exc}\n")
        last_error = str(exc)

    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

    # Grade the episode
    try:
        result = grade_from_environment(env)
        score = result.final_score
    except Exception:
        # Fallback: normalize cumulative reward
        if rewards:
            score = min(max(sum(rewards) / 10.0, 0.0), 1.0)
        else:
            score = 0.0

    success = score >= SUCCESS_SCORE_THRESHOLD
    log_end(success=success, steps=len(rewards), rewards=rewards)
    return score


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="CodeRedEnv inference — OpenEnv spec")
    parser.add_argument("--task", default=os.getenv("TASK_NAME", "task1"),
                        help="Task ID (task1–task5). Default: task1")
    parser.add_argument("--task-name", default=None,
                        help="Human-readable name for [START] log line")
    parser.add_argument("--seed", type=int, default=0,
                        help="Episode random seed. Default: 0")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS_DEFAULT,
                        help=f"Max episode steps. Default: {MAX_STEPS_DEFAULT}")
    parser.add_argument("--benchmark", default=BENCHMARK,
                        help=f"Benchmark name for logs. Default: {BENCHMARK}")
    parser.add_argument("--model", default=None,
                        help=f"Model name. Default: {MODEL_NAME}")
    args = parser.parse_args()

    if not HF_TOKEN:
        sys.stderr.write(
            "[ERROR] HF_TOKEN (or API_KEY / OPENAI_API_KEY) env var must be set.\n"
            "Example: HF_TOKEN=sk-... python inference.py --task task1\n"
        )
        sys.exit(1)

    run_agent(
        task_id=args.task,
        seed=args.seed,
        max_steps=args.max_steps,
        task_name=args.task_name,
        benchmark=args.benchmark,
        model=args.model,
    )


if __name__ == "__main__":
    main()


# Backward-compatible alias
run_baseline_agent = run_agent

