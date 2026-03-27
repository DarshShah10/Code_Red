"""OpenAI-powered baseline agent for CodeRedEnv."""
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
        "description": "Dispatch an ambulance to pick up a patient. Must also assign hospital with assign_hospital.",
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

    lines.append(f"\nTime score preview: {obs.time_score_preview:.2f}")
    lines.append(f"Patients remaining: {obs.patients_remaining}")
    lines.append(f"Mutual aid remaining: {obs.mutual_aid_remaining}")

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
