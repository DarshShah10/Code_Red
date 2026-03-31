"""Rubric grading system for CodeRedEnv episode evaluation."""
from dataclasses import dataclass
from typing import Any


@dataclass
class RubricResult:
    """Result of grading a CodeRedEnv episode."""
    time_score: float      # 0.0–1.0
    efficiency: float       # 0.0–1.0
    secondary_harm: float   # 0.0–1.0
    prep_ready: float      # 0.0–1.0
    mutual_aid_penalty: float  # 0.0–1.0, subtracted from final_score
    final_score: float    # weighted sum minus mutual_aid_penalty
    breakdown: dict        # human-readable per-axis details
    vitals_score_avg: float = 0.0  # Phase 1: informational only

    def __post_init__(self):
        if self.breakdown is None:
            self.breakdown = {}

    def as_dict(self) -> dict:
        return {
            "time_score": self.time_score,
            "efficiency": self.efficiency,
            "secondary_harm": self.secondary_harm,
            "prep_ready": self.prep_ready,
            "mutual_aid_penalty": self.mutual_aid_penalty,
            "final_score": self.final_score,
            "breakdown": self.breakdown,
            "vitals_score_avg": self.vitals_score_avg,
        }


def grade_episode(episode_log: list[dict]) -> RubricResult:
    """
    Grade a CodeRedEnv episode using the 4-axis rubric.

    episode_log: list of event dicts with keys like:
        - event: "patient_created", "patient_deceased", "treatment_complete"
        - patient_id, condition, is_secondary, reason
        - effective_time, target_time (for treatment_complete)
        - action_prepare_or, action_page_specialist, action_allocate_blood with result: "wasted"
        - mutual_aid_called, mutual_aid_arrived with optimal/actual arrival steps
    """
    # =========================================================================
    # TIME SCORE (40% weight)
    # =========================================================================
    patient_times = {}
    for entry in episode_log:
        if entry.get("event") == "treatment_complete":
            pid = entry.get("patient_id")
            patient_times[pid] = {
                "effective_time": entry.get("effective_time", 0),
                "target_time": entry.get("target_time", 90),
                "treated": True,
            }
        elif entry.get("event") == "patient_deceased":
            pid = entry.get("patient_id")
            patient_times[pid] = {
                "effective_time": 0,
                "target_time": entry.get("target_time", 90),
                "treated": False,
            }
        elif entry.get("event") == "patient_created":
            pid = entry.get("patient_id")
            if pid not in patient_times:
                patient_times[pid] = {"treated": False}

    if patient_times:
        scores = []
        for pid, data in patient_times.items():
            if data["treated"]:
                eff = data["effective_time"]
                tgt = max(data["target_time"], 1)
                score = max(0.0, min(1.0, 1.0 - (eff - tgt) / tgt))
                scores.append(score)
            else:
                scores.append(0.0)
        time_score = sum(scores) / len(scores) if scores else 1.0
    else:
        time_score = 1.0

    # =========================================================================
    # VITALS SCORE AVERAGE (informational — not yet in final_score)
    # =========================================================================
    # Capture vitals at treatment from treatment_complete events
    vitals_scores = []
    for entry in episode_log:
        if entry.get("event") == "treatment_complete":
            vitals_scores.append(entry.get("vitals_at_treatment", 1.0))
    vitals_score_avg = sum(vitals_scores) / len(vitals_scores) if vitals_scores else 1.0

    # =========================================================================
    # EFFICIENCY (20% weight)
    # =========================================================================
    unused_specialist = sum(
        1 for e in episode_log
        if e.get("event") == "action_page_specialist" and e.get("result") == "wasted"
    )
    wasted_or_preps = sum(
        1 for e in episode_log
        if e.get("event") == "action_prepare_or" and e.get("result") == "wasted"
    )
    premature_blood_emerg = sum(
        1 for e in episode_log
        if e.get("event") == "action_allocate_blood" and e.get("result") == "wasted"
    )
    total_penalty = -0.1 * unused_specialist + -0.15 * wasted_or_preps + -0.1 * premature_blood_emerg
    efficiency = max(0.0, min(1.0, 1.0 + total_penalty))

    # =========================================================================
    # SECONDARY HARM (20% weight)
    # =========================================================================
    secondary_deaths = sum(
        1 for e in episode_log
        if e.get("event") == "patient_deceased" and e.get("reason") == "secondary"
    )
    secondary_patients = sum(
        1 for e in episode_log
        if e.get("event") == "patient_created" and e.get("is_secondary", False)
    )
    if secondary_patients > 0:
        secondary_harm = 1.0 - (secondary_deaths / secondary_patients)
    else:
        secondary_harm = 1.0

    # =========================================================================
    # PREP READY (20% weight)
    # =========================================================================
    arrival_scores = []
    for entry in episode_log:
        if entry.get("event") == "patient_arrived_hospital":
            or_ready = entry.get("or_ready", False)
            specialist_available = entry.get("specialist_available", False)
            if or_ready and specialist_available:
                arrival_scores.append(1.0)
            elif or_ready or specialist_available:
                arrival_scores.append(0.5)
            else:
                arrival_scores.append(0.0)
    prep_ready = sum(arrival_scores) / len(arrival_scores) if arrival_scores else 1.0

    # =========================================================================
    # MUTUAL AID PENALTY
    # =========================================================================
    ma_calls = [
        e for e in episode_log
        if e.get("event") == "mutual_aid_called"
    ]
    penalties = []
    for call in ma_calls:
        optimal = call.get("optimal_arrival_step", 0)
        arrival_events = [
            e for e in episode_log
            if e.get("event") == "mutual_aid_arrived"
            and e.get("patient_id") == call.get("patient_id")
        ]
        if arrival_events:
            actual = arrival_events[0].get("actual_arrival_step", optimal)
        else:
            actual = optimal + 10  # assume late if not arrived
        if actual < optimal:
            penalties.append(0.1 * (optimal - actual))
        elif actual > optimal:
            penalties.append(0.2 * (actual - optimal))
    num_calls = len(ma_calls)
    mutual_aid_penalty = sum(penalties) / num_calls if num_calls > 0 else 0.0

    # =========================================================================
    # FINAL SCORE
    # =========================================================================
    raw = (
        0.36 * time_score
        + 0.18 * efficiency
        + 0.18 * secondary_harm
        + 0.18 * prep_ready
        + 0.10 * vitals_score_avg
    )
    final_score = max(0.0, min(1.0, raw - mutual_aid_penalty))

    breakdown = {
        "time_score": time_score,
        "efficiency": efficiency,
        "secondary_harm": secondary_harm,
        "prep_ready": prep_ready,
        "vitals_score_avg": vitals_score_avg,
        "mutual_aid_penalty": mutual_aid_penalty,
        "unused_specialist_pages": unused_specialist,
        "wasted_or_preps": wasted_or_preps,
        "premature_blood_emerg": premature_blood_emerg,
        "secondary_deaths": secondary_deaths,
        "secondary_patients": secondary_patients,
        "mutual_aid_calls": num_calls,
    }

    return RubricResult(
        time_score=round(time_score, 4),
        efficiency=round(efficiency, 4),
        secondary_harm=round(secondary_harm, 4),
        prep_ready=round(prep_ready, 4),
        mutual_aid_penalty=round(mutual_aid_penalty, 4),
        final_score=round(final_score, 4),
        breakdown=breakdown,
        vitals_score_avg=round(vitals_score_avg, 4),
    )


def grade_from_environment(env) -> RubricResult:
    """
    Grade from a CodeRedEnvironment instance with cross-validation.

    Compares the episode log against the environment's patient manager to detect
    silent mismatches between treatment_complete events and patient outcomes.
    Mismatches are penalised proportionally and exposed in the breakdown.
    """
    log = env.get_episode_log()

    # =========================================================================
    # CROSS-VALIDATION: treatment_complete events vs patient manager outcomes
    # =========================================================================
    patients = env._patient_manager.patients

    logged_treated = {
        e["patient_id"]
        for e in log
        if e.get("event") == "treatment_complete"
    }
    patients_saved = {p.id for p in patients if p.outcome == "saved"}
    patients_deceased = {p.id for p in patients if p.outcome == "deceased"}

    # 1. Patients marked as saved in patient manager but no treatment_complete log
    treated_missing_log = patients_saved - logged_treated

    # 2. Patients with treatment_complete event but outcome is "deceased"
    treated_but_deceased = logged_treated & patients_deceased

    # 3. Patients marked as saved but also marked deceased in patient manager
    #    (this is an internal inconsistency in patient_manager)
    saved_but_deceased = patients_saved & patients_deceased

    num_mismatches = (
        len(treated_missing_log)
        + len(treated_but_deceased)
        + len(saved_but_deceased)
    )

    # Expose in breakdown
    cross_validation_penalty = min(1.0, 0.2 * num_mismatches)

    # Grade from the log first
    result = grade_episode(log)

    # Apply cross-validation penalty
    result.final_score = max(0.0, result.final_score - cross_validation_penalty)
    result.breakdown["cross_validation_mismatches"] = num_mismatches
    result.breakdown["cross_validation_penalty"] = round(cross_validation_penalty, 4)
    result.breakdown["treated_missing_log"] = sorted(treated_missing_log)
    result.breakdown["treated_but_deceased"] = sorted(treated_but_deceased)

    return result
