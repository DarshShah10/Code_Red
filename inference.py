#!/usr/bin/env python3
"""
CodeRedEnv Inference Script — OpenEnv Spec Compliant

MANDATORY environment variables:
  API_BASE_URL   The API endpoint for the LLM (default: https://openrouter.ai/api/v1)
  MODEL_NAME     The model identifier (default: qwen/qwen3.6-plus)
  HF_TOKEN       Hugging Face / API key (optional)
  LOCAL_IMAGE_NAME  Docker image name for from_docker_image() (optional)

If LOCAL_IMAGE_NAME is set: uses CodeRedEnv.from_docker_image() (async, Docker)
Otherwise: imports CodeRedEnvironment directly (sync, no Docker needed)

STDOUT FORMAT (exact — enforced by validator):
  [START] task=<task> env=<benchmark> model=<model>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import textwrap
from typing import Any, Optional

from dotenv import load_dotenv
load_dotenv()

# ── Env var config ──────────────────────────────────────────────────────────
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://openrouter.ai/api/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "qwen/qwen3.6-plus")
HF_TOKEN: str | None = os.getenv("HF_TOKEN")
ANTHROPIC_API_KEY: str | None = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
BENCHMARK: str = os.getenv("BENCHMARK", "codered_env")
MAX_STEPS_DEFAULT: int = int(os.getenv("MAX_STEPS", "30"))
SUCCESS_SCORE_THRESHOLD: float = float(os.getenv("SUCCESS_THRESHOLD", "0.2"))
LOCAL_IMAGE_NAME: str | None = os.getenv("LOCAL_IMAGE_NAME")

# ── Detect provider ──────────────────────────────────────────────────────────
def _detect_provider(explicit: str | None) -> str:
    if explicit:
        return explicit.lower()
    if OPENAI_API_KEY:
        return "openai"
    if ANTHROPIC_API_KEY:
        return "anthropic"
    return "hf_fallback"

def _get_provider_config(explicit: str | None = None) -> tuple[str, str, str, str | None]:
    """Return (provider, base_url, model, api_key)."""
    provider = _detect_provider(explicit)
    if provider == "openai":
        return provider, API_BASE_URL, MODEL_NAME, OPENAI_API_KEY
    elif provider == "anthropic":
        return (
            provider,
            os.getenv("ANTHROPIC_API_BASE_URL", "https://claude.opuscode.pro/api"),
            os.getenv("ANTHROPIC_MODEL_NAME", "claude-sonnet-4-6"),
            ANTHROPIC_API_KEY,
        )
    else:
        return provider, "https://router.huggingface.co", "Qwen/Qwen2.5-72B-Instruct", HF_TOKEN

# ── Function definitions (must match server/models/actions.py) ─────────────
FUNCTIONS: list[dict[str, Any]] = [
    {"type": "function", "function": {"name": "dispatch_ambulance", "description": "Dispatch an ambulance to a target node (Phase 1). Also assign hospital with assign_hospital.", "parameters": {"type": "object", "properties": {"ambulance_id": {"type": "string", "enum": ["AMB_1", "AMB_2", "AMB_3", "AMB_4", "AMB_5"]}, "patient_id": {"type": "string"}, "target_node": {"type": "string"}}, "required": ["ambulance_id", "patient_id", "target_node"]}}},
    {"type": "function", "function": {"name": "dispatch_als", "description": "Dispatch an ALS ambulance to a pending 911 call (Phase 2).", "parameters": {"type": "object", "properties": {"ambulance_id": {"type": "string", "enum": ["AMB_1", "AMB_2"]}, "call_id": {"type": "string"}}, "required": ["ambulance_id", "call_id"]}}},
    {"type": "function", "function": {"name": "dispatch_bls", "description": "Dispatch a BLS ambulance to a pending 911 call (Phase 2).", "parameters": {"type": "object", "properties": {"ambulance_id": {"type": "string", "enum": ["AMB_3", "AMB_4", "AMB_5"]}, "call_id": {"type": "string"}}, "required": ["ambulance_id", "call_id"]}}},
    {"type": "function", "function": {"name": "triage_call", "description": "Decide what to do with a pending 911 call.", "parameters": {"type": "object", "properties": {"call_id": {"type": "string"}, "decision": {"type": "string", "enum": ["dispatch_als", "dispatch_bls", "self_transport", "callback", "no_dispatch"]}, "ambulance_id": {"type": "string"}}, "required": ["call_id", "decision"]}}},
    {"type": "function", "function": {"name": "prepare_or", "description": "Begin OR preparation at a hospital for a procedure type.", "parameters": {"type": "object", "properties": {"hospital_id": {"type": "string", "enum": ["HOSP_A", "HOSP_B", "HOSP_C"]}, "procedure_type": {"type": "string", "enum": ["cardiac", "stroke", "trauma", "general"]}}, "required": ["hospital_id", "procedure_type"]}}},
    {"type": "function", "function": {"name": "page_specialist", "description": "Page a specialist at a hospital.", "parameters": {"type": "object", "properties": {"hospital_id": {"type": "string", "enum": ["HOSP_A", "HOSP_B"]}, "specialist_type": {"type": "string", "enum": ["cardiologist", "neurologist", "trauma_surgeon"]}}, "required": ["hospital_id", "specialist_type"]}}},
    {"type": "function", "function": {"name": "assign_hospital", "description": "Assign a patient to a destination hospital.", "parameters": {"type": "object", "properties": {"patient_id": {"type": "string"}, "hospital_id": {"type": "string", "enum": ["HOSP_A", "HOSP_B", "HOSP_C"]}}, "required": ["patient_id", "hospital_id"]}}},
    {"type": "function", "function": {"name": "preempt_or", "description": "Preempt (clear) an operating room for emergency use.", "parameters": {"type": "object", "properties": {"hospital_id": {"type": "string", "enum": ["HOSP_A", "HOSP_B"]}, "or_index": {"type": "integer", "minimum": 0}}, "required": ["hospital_id", "or_index"]}}},
    {"type": "function", "function": {"name": "allocate_blood", "description": "Allocate blood units for a patient.", "parameters": {"type": "object", "properties": {"hospital_id": {"type": "string", "enum": ["HOSP_A", "HOSP_B", "HOSP_C"]}, "patient_id": {"type": "string"}, "blood_type": {"type": "string"}, "units": {"type": "integer", "minimum": 1}, "emergency": {"type": "boolean", "default": False}}, "required": ["hospital_id", "patient_id", "blood_type", "units"]}}},
    {"type": "function", "function": {"name": "transfer_blood", "description": "Transfer blood units between hospitals.", "parameters": {"type": "object", "properties": {"from_hospital": {"type": "string", "enum": ["HOSP_A", "HOSP_B", "HOSP_C"]}, "to_hospital": {"type": "string", "enum": ["HOSP_A", "HOSP_B", "HOSP_C"]}, "blood_type": {"type": "string"}, "units": {"type": "integer", "minimum": 1}}, "required": ["from_hospital", "to_hospital", "blood_type", "units"]}}},
    {"type": "function", "function": {"name": "request_mutual_aid", "description": "Request mutual aid ambulance (Task 2/3 only).", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "query_blood_type", "description": "Query a patient's blood type.", "parameters": {"type": "object", "properties": {"patient_id": {"type": "string"}}, "required": ["patient_id"]}}},
    {"type": "function", "function": {"name": "query_or_status", "description": "Query detailed OR status at a hospital.", "parameters": {"type": "object", "properties": {"hospital_id": {"type": "string"}, "or_index": {"type": "integer", "minimum": 0}}, "required": ["hospital_id", "or_index"]}}},
    {"type": "function", "function": {"name": "maintain_plan", "description": "No-op: continue current plan without changes.", "parameters": {"type": "object", "properties": {}}}},
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

# ── STDOUT logging (exact format per spec) ────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error if error else 'null'}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

# ── Action parsing ────────────────────────────────────────────────────────────

ACTION_TYPE_MAP: dict[str, str] = {
    "dispatch_ambulance": "dispatch_ambulance",
    "dispatch_als": "dispatch_als",
    "dispatch_bls": "dispatch_bls",
    "triage_call": "triage_call",
    "prepare_or": "prepare_or",
    "page_specialist": "page_specialist",
    "assign_hospital": "assign_hospital",
    "preempt_or": "preempt_or",
    "allocate_blood": "allocate_blood",
    "transfer_blood": "transfer_blood",
    "request_mutual_aid": "request_mutual_aid",
    "query_blood_type": "query_blood_type",
    "query_or_status": "query_or_status",
    "maintain_plan": "maintain_plan",
}

def parse_action_string(action_str: str) -> tuple[dict[str, Any], str]:
    """
    Parse LLM response into (action_dict, display_str).
    Handles: fn_name(k=v, ...) or plain text (fallback to maintain_plan).
    """
    fn_match = re.match(r"^(\w+)\((.*)\)$", action_str.strip())
    if fn_match:
        fn_name = fn_match.group(1)
        args_str = fn_match.group(2)
        fn_args: dict = {}
        if args_str.strip():
            try:
                fn_args = json.loads(f"{{{args_str}}}")
            except Exception:
                for m in re.finditer(r"(\w+)=(['\"]?)([\w_/]+)\2", args_str):
                    fn_args[m.group(1)] = m.group(3)
        action_dict = {"type": fn_name, **fn_args}
        display_str = action_str.strip()
    else:
        action_dict = {"type": "maintain_plan"}
        display_str = "maintain_plan()"
    return action_dict, display_str

# ── Observation formatting ────────────────────────────────────────────────────

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
            prep_ors = [(o.index, o.status.value, o.procedure_type, o.minutes_remaining)
                        for o in h.operating_rooms if o.status.value != "idle"]
            or_info = ""
            if prep_ors:
                or_info = " | " + " | ".join(
                    f"OR{o[0]}={o[1]}({o[2] or '?'}, {o[3]}min)" for o in prep_ors
                )
            lines.append(
                f"  {h.id}: idle_ors={idle_ors}/{len(h.operating_rooms)}"
                f"{or_info} | {', '.join(h.capabilities)}"
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

# ── LLM calls ─────────────────────────────────────────────────────────────────

def _convert_to_anthropic_tools(functions: list[dict]) -> list[dict]:
    anthropic_tools = []
    for fn in functions:
        func = fn.get("function", fn)
        anthropic_tools.append({
            "name": func["name"],
            "description": func.get("description", ""),
            "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
        })
    return anthropic_tools

def call_model(messages: list[dict], explicit_provider: str | None = None) -> str:
    """Call LLM. Provider auto-detected from env vars or --provider flag."""
    provider, base_url, model, api_key = _get_provider_config(explicit_provider)

    if provider == "anthropic":
        return _call_anthropic(messages, model, base_url, api_key)
    return _call_openai(messages, model, base_url, api_key)

def _call_anthropic(messages: list[dict], model: str, base_url: str, api_key: str | None) -> str:
    import anthropic
    client = anthropic.Anthropic(base_url=base_url, api_key=api_key, timeout=120.0, max_retries=2)
    system_msg = ""
    chat_messages = []
    for msg in messages:
        if msg["role"] == "system":
            system_msg = msg["content"]
        else:
            chat_messages.append(msg)

    anthropic_tools = _convert_to_anthropic_tools(FUNCTIONS)
    try:
        response = client.messages.create(
            model=model, max_tokens=2048, system=system_msg,
            messages=chat_messages, tools=anthropic_tools,
        )
        for block in response.content:
            if block.type == "tool_use":
                fn_name = block.name
                fn_args = dict(block.input) if block.input else {}
                args_parts = [f"{k}={v!r}" for k, v in fn_args.items()]
                return f"{fn_name}({', '.join(args_parts)})"
            elif block.type == "text":
                text = block.text.strip()
                if text:
                    return text
        return "maintain_plan()"
    except Exception as exc:
        sys.stderr.write(f"[WARN] Anthropic model call failed: {exc}\n")
        return "maintain_plan()"

def _call_openai(messages: list[dict], model: str, base_url: str, api_key: str | None) -> str:
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=120.0)
    try:
        completion = client.chat.completions.create(
            model=model, messages=messages, tools=FUNCTIONS,
            tool_choice="auto", max_completion_tokens=2048,
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
        return (msg.content or "").strip() or "maintain_plan()"
    except Exception as exc:
        sys.stderr.write(f"[WARN] OpenAI model call failed: {exc}\n")
        return "maintain_plan()"

# ── Environment setup ────────────────────────────────────────────────────────

def _build_action_obj(action_dict: dict):
    """Convert action dict to typed CodeRedAction object."""
    from server.models.actions import (
        DispatchAmbulance, DispatchALS, DispatchBLS, TriageCall,
        PrepareOR, PageSpecialist, AssignHospital, PreemptOR,
        AllocateBlood, TransferBlood, RequestMutualAid,
        QueryBloodType, QueryORStatus, MaintainPlan,
    )
    at = action_dict.get("type", "maintain_plan")
    args = {k: v for k, v in action_dict.items() if k != "type"}
    mapping = {
        "dispatch_ambulance": DispatchAmbulance,
        "dispatch_als": DispatchALS,
        "dispatch_bls": DispatchBLS,
        "triage_call": TriageCall,
        "prepare_or": PrepareOR,
        "page_specialist": PageSpecialist,
        "assign_hospital": AssignHospital,
        "preempt_or": PreemptOR,
        "allocate_blood": AllocateBlood,
        "transfer_blood": TransferBlood,
        "request_mutual_aid": RequestMutualAid,
        "query_blood_type": QueryBloodType,
        "query_or_status": QueryORStatus,
    }
    cls = mapping.get(at, MaintainPlan)
    try:
        return cls(**args)
    except Exception:
        valid = set(cls.model_fields.keys())
        return cls(**{k: v for k, v in args.items() if k in valid})

# ── Main run ─────────────────────────────────────────────────────────────────

def run_episode(
    task_id: str, seed: int = 0, max_steps: int = MAX_STEPS_DEFAULT,
    benchmark: str = BENCHMARK, explicit_provider: str | None = None,
) -> float:
    """
    Run one episode using Docker (from_docker_image) if LOCAL_IMAGE_NAME is set,
    otherwise use direct Python import (no Docker).
    """
    provider, _, model_name, _ = _get_provider_config(explicit_provider)
    actual_model = os.getenv("MODEL_NAME") or model_name

    log_start(task=task_id, env=benchmark, model=actual_model)

    rewards: list[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    last_error: str | None = None

    try:
        if LOCAL_IMAGE_NAME:
            # ── Docker mode: use async CodeRedEnv.from_docker_image() ──
            env, obs, done = _run_docker_episode(
                LOCAL_IMAGE_NAME, task_id, seed, max_steps,
                actual_model, provider, log_step,
            )
        else:
            # ── Local mode: import CodeRedEnvironment directly ──
            env, obs, done = _run_local_episode(
                task_id, seed, max_steps,
                actual_model, provider, log_step,
            )

        rewards = env._rewards if hasattr(env, "_rewards") else []
        steps_taken = env.state.step_count if hasattr(env, "state") and env.state else len(rewards)

    except Exception as exc:
        sys.stderr.write(f"[ERROR] Episode crashed: {exc}\n")
        last_error = str(exc)

    # Grade the episode
    breakdown: dict | None = None
    try:
        from server.grader import grade_from_environment
        result = grade_from_environment(env)
        score = result.final_score
        breakdown = result.breakdown
    except Exception:
        if rewards:
            score = min(max(sum(rewards) / 10.0, 0.0), 1.0)
        else:
            score = 0.0

    success = score >= SUCCESS_SCORE_THRESHOLD
    log_end(success=success, steps=len(rewards), score=score, rewards=rewards)

    if breakdown:
        print(f"[GRADE] score={score:.4f}", flush=True)

    return score


async def _run_docker_episode(
    image_name: str, task_id: str, seed: int, max_steps: int,
    model: str, provider: str, log_step_fn,
):
    """Run episode via CodeRedEnv.from_docker_image() (async, Docker)."""
    from client import CodeRedEnv

    async with CodeRedEnv.from_docker_image(image_name) as env:
        result = await env.reset(seed=seed, task_id=task_id)
        obs = result.observation
        rewards: list[float] = []
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        for step in range(1, max_steps + 1):
            if result.done:
                break

            prompt = format_observation(obs)
            messages.append({"role": "user", "content": prompt})
            action_str = call_model(messages, provider)
            messages.append({"role": "assistant", "content": action_str})

            action_dict, display_str = parse_action_string(action_str)
            action_obj = _build_action_obj(action_dict)

            result = await env.step(action_obj)
            obs = result.observation

            reward = result.reward or 0.0
            rewards.append(round(reward, 2))
            done = result.done

            log_step_fn(step=step, action=display_str, reward=reward, done=done, error=None)

            if done:
                break

        # Attach rewards for grading
        env._rewards = rewards
        return env, obs, result.done


def _run_local_episode(
    task_id: str, seed: int, max_steps: int,
    model: str, provider: str, log_step_fn,
):
    """Run episode via direct CodeRedEnvironment import (sync, no Docker)."""
    from server.codered_environment import CodeRedEnvironment
    from server.grader import grade_from_environment

    env = CodeRedEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)
    rewards: list[float] = []
    prev_cum_reward = 0.0
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for step in range(1, max_steps + 1):
        if env._check_done():
            break

        prompt = format_observation(obs)
        messages.append({"role": "user", "content": prompt})
        action_str = call_model(messages, provider)
        messages.append({"role": "assistant", "content": action_str})

        action_dict, display_str = parse_action_string(action_str)
        action_obj = _build_action_obj(action_dict)

        obs = env.step(action_obj)

        current_cum = env.state.cum_reward if hasattr(env.state, "cum_reward") else 0.0
        reward = current_cum - prev_cum_reward
        prev_cum_reward = current_cum
        rewards.append(round(reward, 2))
        done = env._check_done()

        log_step_fn(step=step, action=display_str, reward=reward, done=done, error=None)

        if done:
            break

    env._rewards = rewards
    return env, obs, env._check_done()

# ── CLI entry point ──────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="CodeRedEnv inference")
    parser.add_argument("--task", default=os.getenv("TASK_NAME", "task1"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS_DEFAULT)
    parser.add_argument("--benchmark", default=BENCHMARK)
    parser.add_argument("--provider", choices=["openai", "anthropic", "hf_fallback"])
    args = parser.parse_args()

    try:
        run_episode(
            task_id=args.task, seed=args.seed, max_steps=args.max_steps,
            benchmark=args.benchmark, explicit_provider=args.provider,
        )
    except ValueError as e:
        sys.stderr.write(f"[ERROR] {e}\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
