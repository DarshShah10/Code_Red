"""OpenAI-powered inference agent for CodeRedEnv (OpenEnv spec: inference.py)."""
import os
from typing import Literal, List
import openai

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# OpenAI function definitions matching all action types
FUNCTIONS = [
    {
        "type": "function",
        "name": "dispatch_ambulance",
        "description": "Dispatch an ambulance to a scene node (Phase 1 compat). Must also assign hospital with assign_hospital.",
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
        "description": "Dispatch an ALS ambulance to a pending 911 call (Phase 2). Use triage_call first to decide if ALS is needed.",
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
        "description": "Dispatch a BLS ambulance to a pending 911 call (Phase 2). Use triage_call first to decide if BLS is appropriate.",
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
        "description": "Decide what to do with a pending 911 call (Phase 2). Choices: dispatch_als (ALS needed), dispatch_bls (BLS fine), self_transport, callback (re-contact later), no_dispatch (cancel).",
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
        "description": "Request mutual aid ambulance (use for Tasks 2/3 only).",
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
        "description": "No-op: continue current plan without changes.",
        "parameters": {"type": "object", "properties": {}},
    },
]

SYSTEM_PROMPT = """You are a medical emergency coordination AI. You manage ambulance dispatch, hospital preparation, and patient triage in the city of Prakashnagar.

City layout:
- Hospitals: HOSP_A (AIIMS, full capability), HOSP_B (District Hospital, cardiac/trauma), HOSP_C (Community HC, stabilization only)
- Ambulances: AMB_1, AMB_2 (ALS - advanced), AMB_3, AMB_4, AMB_5 (BLS - basic)
- Priority: CARDIAC > STROKE > TRAUMA > GENERAL

Phase 2 — Dispatch Cascade:
- Incoming 911 calls arrive as pending_calls. You MUST triage each call with triage_call before dispatching.
- dispatch_als commits an ALS unit (AMB_1/AMB_2) to a call. dispatch_bls commits a BLS unit (AMB_3/4/5).
- Ground truth condition is hidden until the ambulance arrives on-scene.
- For Phase 1 tasks (task1): use dispatch_ambulance to go directly to patient nodes.
- For Phase 2 tasks: use triage_call to decide, then dispatch_als or dispatch_bls.

Use the available tools to save patients. Always dispatch ambulances to patient locations, assign them to HOSP_A for cardiac cases, and prepare ORs before patients arrive.

Use maintain_plan only when there is nothing useful to do."""


def format_observation(obs) -> str:
    """Format CodeRedObservation as text for the LLM."""
    lines = [f"=== Step {obs.step} ==="]

    if obs.patients:
        lines.append("\n## Active Patients:")
        for p in obs.patients:
            lines.append(
                f"  {p.patient_id}: {p.condition.value} | "
                f"status={p.status.value} | blood={p.blood_type or '?'} | "
                f"assigned_to={p.assigned_hospital or 'unassigned'} | "
                f"location={p.location_node}"
            )

    if obs.ambulances:
        lines.append("\n## Ambulances:")
        for a in obs.ambulances:
            lines.append(
                f"  {a.id} ({a.equipment.value}): status={a.status.value}"
                + (f" | patient={a.assigned_patient}" if a.assigned_patient else "")
                + (f" | eta={a.eta_minutes}min" if a.eta_minutes > 0 else "")
            )

    if obs.hospitals:
        lines.append("\n## Hospitals:")
        for h in obs.hospitals:
            idle_ors = sum(1 for o in h.operating_rooms if o.status.value == "idle")
            lines.append(
                f"  {h.id}: idle_ors={idle_ors}/{len(h.operating_rooms)} | "
                f"capabilities={', '.join(h.capabilities)}"
            )

    if obs.alerts:
        lines.append("\n## Alerts:")
        for a in obs.alerts[-5:]:
            lines.append(f"  {a}")

    if obs.pending_calls:
        lines.append("\n## Pending 911 Calls:")
        for c in obs.pending_calls:
            lines.append(
                f"  {c.call_id}: {c.category.value} | waiting={c.time_waiting}min | "
                f"severity_est={c.estimated_severity:.1f} | loc={c.location_node}"
                + (f" | patient_will_spawn={c.spawned_patient_id}" if c.spawned_patient_id else "")
            )

    if obs.recent_dispatch_outcomes:
        lines.append("\n## Recent Dispatch Outcomes:")
        for o in obs.recent_dispatch_outcomes[-5:]:
            lines.append(
                f"  {o.call_id}: decision={o.decision}"
                + (f" | true={o.true_condition} (ALS={o.als_needed})" if o.true_condition else " | outcome=pending")
            )

    lines.append(f"\nTime score preview: {obs.time_score_preview:.2f}")
    lines.append(f"Vitals score preview: {obs.vitals_score_preview:.2f}")
    lines.append(f"Patients remaining: {obs.patients_remaining}")
    lines.append(f"Mutual aid remaining: {obs.mutual_aid_remaining}")
    if obs.overcrowding_modifier > 1.0:
        lines.append(f"WARNING: ED overcrowding active (modifier={obs.overcrowding_modifier:.1f}x)")

    return "\n".join(lines)


def parse_agent_response(response) -> dict:
    """Extract action from OpenAI response."""
    import json

    for item in response.output:
        if hasattr(item, "content"):
            for part in item.content:
                if part.type == "function_call":
                    name = part.name
                    try:
                        args = json.loads(part.arguments) if isinstance(part.arguments, str) else part.arguments
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                    return _name_to_action(name, args)

    return {"action_type": "maintain_plan"}


def _name_to_action(name: str, args: dict) -> dict:
    """Map function name to action dict for EnvClient."""
    mapping = {
        "dispatch_ambulance": lambda a: {"action_type": "dispatch_ambulance", **a},
        "dispatch_als": lambda a: {"action_type": "dispatch_als", **a},
        "dispatch_bls": lambda a: {"action_type": "dispatch_bls", **a},
        "triage_call": lambda a: {"action_type": "triage_call", **a},
        "prepare_or": lambda a: {"action_type": "prepare_or", **a},
        "page_specialist": lambda a: {"action_type": "page_specialist", **a},
        "assign_hospital": lambda a: {"action_type": "assign_hospital", **a},
        "preempt_or": lambda a: {"action_type": "preempt_or", **a},
        "allocate_blood": lambda a: {"action_type": "allocate_blood", **a},
        "transfer_blood": lambda a: {"action_type": "transfer_blood", **a},
        "request_mutual_aid": lambda _: {"action_type": "request_mutual_aid"},
        "query_blood_type": lambda a: {"action_type": "query_blood_type", **a},
        "query_or_status": lambda a: {"action_type": "query_or_status", **a},
        "maintain_plan": lambda _: {"action_type": "maintain_plan"},
    }

    if name in mapping:
        return mapping[name](args)
    return {"action_type": "maintain_plan"}


def _build_response_kwargs(model: str, prompt: str) -> dict:
    """Build kwargs for openai.responses.create(), handling gpt-5 reasoning models."""
    kwargs = {
        "model": model,
        "input": prompt,
        "instructions": SYSTEM_PROMPT,
        "tools": FUNCTIONS,
        "tool_choice": "auto",
    }
    # gpt-5 reasoning models only support temperature when reasoning=effort:none
    # For baseline determinism, disable reasoning and set temperature=0
    if model.startswith("gpt-5"):
        kwargs["reasoning"] = {"effort": "none"}
        kwargs["temperature"] = 0.0
    return kwargs


def run_baseline_agent(task_id: Literal["task1", "task2", "task3"], seed: int, api_key: str | None = None, model: str | None = None) -> float:
    """
    Run the OpenAI API baseline agent on a given task and seed.
    Returns the rubric final_score (0.0-1.0).

    Reads OPENAI_API_KEY and OPENAI_MODEL from environment if not passed.
    """
    from codered_env.server.codered_environment import CodeRedEnvironment
    from codered_env.server.grader import grade_from_environment

    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")
    if model is None:
        model = os.environ.get("OPENAI_MODEL", "gpt-5-nano")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set — pass api_key or set in .env")

    openai.api_key = api_key

    env = CodeRedEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)

    for _ in range(60):  # max steps
        prompt = format_observation(obs)

        try:
            response = openai.responses.create(
                **_build_response_kwargs(model, prompt),
            )
        except Exception:
            # On API failure, maintain plan
            obs = env.step({"action_type": "maintain_plan"})
            if getattr(obs, "done", False):
                break
            continue

        action_dict = parse_agent_response(response)
        obs = env.step(action_dict)

        if getattr(obs, "done", False):
            break

    # Grade the episode
    result = grade_from_environment(env)
    return result.final_score


# Make it importable from the package
__all__ = ["run_baseline_agent", "FUNCTIONS", "SYSTEM_PROMPT"]
