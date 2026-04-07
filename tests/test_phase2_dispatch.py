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
        # Step forward so new calls can spawn into _pending_calls
        obs = env.step(MaintainPlan())
        if env._check_done():
            break
        # Dispatch any pending calls that have an available ambulance.
        # Try multiple times per step since an ambulance may become available mid-step.
        dispatched_any = True
        while dispatched_any:
            dispatched_any = False
            # Find the first pending call that has an available ambulance
            for call in list(obs.pending_calls):
                available_amb = next(
                    (a.id for a in env._ambulance_manager.all().values()
                     if a.equipment == "ALS" and a.status == "available"),
                    None
                )
                if available_amb:
                    obs = env.step(TriageCall(call_id=call.call_id, decision="dispatch_als", ambulance_id=available_amb))
                    dispatched_any = True
                    break  # re-check all calls from the start

    return env


def test_phase2_dispatch_patient_has_ambulance():
    """After dispatching ALS to a call, the ambulance that arrives must carry the patient."""
    env = _run_phase2_episode(seed=42)

    # Patients spawned from DISPATCHED calls (countdown expired after ambulance was sent)
    # should have assigned_ambulance set. Patients from un-dispatched calls (no ambulance
    # was ever sent, countdown just expired) have no ambulance — that's correct.
    patients_from_dispatched_calls = [
        p for p in env._patients
        if getattr(p, 'dispatch_call_id', None) is not None
        and getattr(p, 'assigned_ambulance', None) is not None
    ]
    # At least 2 calls were dispatched (CALL_0008, CALL_0016), so at least 2 patients
    # should be linked
    assert len(patients_from_dispatched_calls) >= 2, (
        f"Expected >= 2 patients from dispatched calls with ambulance linkage, got {len(patients_from_dispatched_calls)}. "
        f"All patients: {[(p.id, p.status, getattr(p,'assigned_ambulance',None), getattr(p,'dispatch_call_id',None)) for p in env._patients]}"
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
