"""Test Phase 2 dispatch: ambulance must carry patient on arrival."""
import pytest
from server.codered_environment import CodeRedEnvironment
from server.models.actions import TriageCall, MaintainPlan


def _run_phase2_episode(task_id="task4", seed=0, max_steps=60):
    """Run a Phase 2 episode, dispatching all pending calls as they appear."""
    env = CodeRedEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)

    for _ in range(max_steps):
        if env._check_done():
            break
        # Dispatch all pending calls that have an available ambulance.
        # Step once with MaintainPlan first so new calls can spawn into _pending_calls.
        obs = env.step(MaintainPlan())
        if env._check_done():
            break
        # Dispatch loop: keep dispatching until no more calls can be handled
        dispatched_this_step = True
        while dispatched_this_step:
            dispatched_this_step = False
            if obs.pending_calls:
                available_amb = next(
                    (a.id for a in env._ambulance_manager.all().values()
                     if a.equipment == "ALS" and a.status == "available"),
                    None
                )
                if available_amb:
                    # Dispatch the oldest pending call
                    call = obs.pending_calls[0]
                    obs = env.step(TriageCall(call_id=call.call_id, decision="dispatch_als", ambulance_id=available_amb))
                    dispatched_this_step = True

    return env


def test_phase2_dispatch_patient_has_ambulance():
    """After dispatching ALS to a call, the spawned patient must have assigned_ambulance set."""
    env = _run_phase2_episode(seed=42)

    # Patients spawned from dispatched calls should have an assigned ambulance
    # (cascade-spawned patients without dispatch are exempt)
    patients_from_dispatched_calls = [
        p for p in env._patients
        if getattr(p, 'dispatch_call_id', None) is not None
        # include all — if it came from a call, it should be linked
    ]
    assert len(patients_from_dispatched_calls) > 0, (
        f"No patients with dispatch_call_id found. "
        f"Patients: {[(p.id, p.status) for p in env._patients]}"
    )
    unlinked = [p for p in patients_from_dispatched_calls if getattr(p, 'assigned_ambulance', None) is None]
    assert len(unlinked) == 0, (
        f"{len(unlinked)} patient(s) from dispatched calls have no assigned ambulance: "
        f"{[(p.id, p.dispatch_call_id) for p in unlinked]}. "
        f"Ambulance-patient linkage is broken in Phase 2 dispatch."
    )


def test_phase2_dispatch_ambulance_carries_patient():
    """Dispatched ambulance must have patient_id set so arrival detection works."""
    env = _run_phase2_episode(seed=42)

    # After dispatch, at least one ambulance should have a patient_id
    found_linked_amb = False
    for amb_id, amb in env._ambulance_manager.all().items():
        pid = getattr(amb, 'patient_id', None)
        if pid is not None:
            if pid.startswith("CALL:"):
                found_linked_amb = True
                break
            patient = env._patient_manager.get(pid)
            if patient and patient.status not in ("treated", "deceased"):
                found_linked_amb = True
                break

    assert found_linked_amb, (
        "No ambulance has patient_id set after Phase 2 dispatch. "
        "dispatch_als/bls must set ambulance.patient_id so arrival detection works."
    )
