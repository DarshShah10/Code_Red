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
    cascade_score: float = 1.0    # Phase 2: secondary patient management

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
            "cascade_score": self.cascade_score,
        }


def grade_cascade_score(episode_log: list[dict]) -> float:
    """
    Scores how well the agent managed cascade effects.

    Components:
    - Secondary patients saved / total secondary patients (50% weight)
    - Overcrowding events prevented (30% weight): fewer events = better, max 5 baseline
    - News cycle surge handled (20% weight): normalized by count
    """
    secondary_events = [
        e for e in episode_log if e.get("event") == "secondary_patient_spawned"
    ]
    secondary_patient_ids = {e["patient_id"] for e in secondary_events}

    if len(secondary_patient_ids) == 0:
        return 1.0  # No cascades = no penalty

    secondary_saves = [
        e for e in episode_log
        if e.get("event") == "treatment_complete"
        and e.get("patient_id") in secondary_patient_ids
    ]
    # Count deaths: either for secondary patient IDs OR explicitly marked as secondary
    secondary_deaths = [
        e for e in episode_log
        if e.get("event") == "patient_deceased"
        and (e.get("patient_id") in secondary_patient_ids or e.get("reason") == "secondary")
    ]

    num_secondary = len(secondary_patient_ids)
    secondary_saved_count = len(secondary_saves)
    secondary_death_count = len(secondary_deaths)

    secondary_score = 1.0 - (secondary_death_count / num_secondary)

    overcrowding_events = [
        e for e in episode_log if e.get("event") == "overcrowding_started"
    ]
    overcrowding_score = max(0.0, 1.0 - len(overcrowding_events) / 5.0)

    news_cycles = [e for e in episode_log if e.get("event") == "news_cycle"]
    news_score = max(0.0, 1.0 - len(news_cycles) / 10.0)

    cascade_score = 0.5 * secondary_score + 0.3 * overcrowding_score + 0.2 * news_score
    return max(0.0, min(1.0, cascade_score))


def grade_episode(episode_log: list[dict], dispatch_outcomes: list[dict] = None) -> RubricResult:
    """Hybrid Grader: Deep subsystems evaluation with airtight mathematical guards."""

    # Check baseline activity
    patients_created = sum(1 for e in episode_log if e.get("event") == "patient_created")
    if patients_created == 0:
        return RubricResult(0.0, 1.0, 1.0, 1.0, 0.0, 0.0, {"error": "No patients processed"}, 1.0, 1.0)

    # =========================================================================
    # 1. SURVIVAL & SECONDARY HARM
    # We explicitly forgive "hospital_mortality" (RNG) deaths. We only punish preventable deaths.
    # =========================================================================
    preventable_deaths = sum(1 for e in episode_log if e.get("event") == "patient_deceased" and e.get("reason") != "hospital_mortality")
    survival_rate = 1.0 - (preventable_deaths / patients_created)

    # THE ABSOLUTE GUARD: If everyone dies preventably, the run is a 0. No partial credit.
    if survival_rate == 0.0:
        return RubricResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {"error": "100% preventable mortality rate"}, 0.0, 0.0)

    secondary_deaths = sum(1 for e in episode_log if e.get("event") == "patient_deceased" and e.get("reason") == "secondary")
    secondary_patients = sum(1 for e in episode_log if e.get("event") == "patient_created" and e.get("is_secondary"))
    secondary_harm = 1.0 - (secondary_deaths / secondary_patients) if secondary_patients > 0 else 1.0

    # =========================================================================
    # 2. TIME SCORE
    # =========================================================================
    time_scores = []
    for e in episode_log:
        if e.get("event") == "treatment_complete":
            tgt = max(1, e.get("target_time", 90))
            eff = e.get("effective_time", 0)
            time_scores.append(max(0.0, min(1.0, 1.0 - (eff - tgt) / tgt)))
        elif e.get("event") == "patient_deceased" and e.get("reason") != "hospital_mortality":
            time_scores.append(0.0)  # Preventable death = 0 time score

    time_score = sum(time_scores) / len(time_scores) if time_scores else 0.0

    # =========================================================================
    # 3. EFFICIENCY (Uncheatable Ratio)
    # Instead of `-0.1` flat penalties, we use a ratio. Spamming 100 bad actions destroys this score.
    # =========================================================================
    wasted_actions = sum(1 for e in episode_log if e.get("result") == "wasted")
    good_actions = sum(1 for e in episode_log if e.get("result") == "success")
    efficiency = good_actions / (good_actions + wasted_actions) if (good_actions + wasted_actions) > 0 else 1.0

    # =========================================================================
    # 4. PREP READY & PREEMPTION HARM
    # =========================================================================
    arrival_scores = []
    for e in episode_log:
        if e.get("event") == "patient_arrived_hospital":
            if e.get("or_ready") and e.get("specialist_available"):
                arrival_scores.append(1.0)
            elif e.get("or_ready") or e.get("specialist_available"):
                arrival_scores.append(0.5)
            else:
                arrival_scores.append(0.0)

    # Fix the dead-on-street loophole
    if not arrival_scores:
        prep_ready = 0.0
    else:
        prep_ready = sum(arrival_scores) / len(arrival_scores)

    preempted_surgeries = sum(1 for e in episode_log if e.get("event") == "surgery_aborted")
    prep_ready = max(0.0, prep_ready - (0.1 * preempted_surgeries))

    # =========================================================================
    # 5. CASCADE SCORE
    # =========================================================================
    cascade_score = grade_cascade_score(episode_log)

    # =========================================================================
    # 6. VITALS AVERAGE (Informational)
    # =========================================================================
    vitals_scores = [e.get("vitals_at_treatment", 1.0) for e in episode_log if e.get("event") == "treatment_complete"]
    vitals_score_avg = sum(vitals_scores) / len(vitals_scores) if vitals_scores else 0.0

    # =========================================================================
    # DEDUCTIONS (Mutual Aid & Triage - Subtracted directly from Final Score)
    # =========================================================================
    # Mutual Aid: sum penalties (stacking), not averaged!
    ma_penalties = 0.0
    for call in [e for e in episode_log if e.get("event") == "mutual_aid_called"]:
        optimal = call.get("optimal_arrival_step", 0)
        actual_event = next((e for e in episode_log if e.get("event") == "mutual_aid_arrived" and e.get("patient_id") == call.get("patient_id")), None)
        actual = actual_event.get("actual_arrival_step", optimal + 10) if actual_event else optimal + 10
        if actual < optimal:
            ma_penalties += 0.1 * (optimal - actual)
        elif actual > optimal:
            ma_penalties += 0.2 * (actual - optimal)

    # Triage Accuracy (Phase 2)
    triage_penalty = 0.0
    if dispatch_outcomes:
        for o in dispatch_outcomes:
            if o["true_condition"] is not None:
                if o["decision"] == "als" and not o["als_needed"]:
                    triage_penalty += 0.10  # Hoarding ALS
                elif o["decision"] != "als" and o["als_needed"]:
                    triage_penalty += 0.20  # Sent BLS to dying patient

    total_deductions = ma_penalties + triage_penalty

    # =========================================================================
    # FINAL SCORE CALCULATION
    # =========================================================================
    raw = (
        0.30 * time_score
        + 0.20 * efficiency
        + 0.20 * secondary_harm
        + 0.15 * prep_ready
        + 0.15 * cascade_score
    )
    final_score = max(0.0, min(1.0, raw - total_deductions))

    breakdown = {
        "time_score": round(time_score, 4),
        "efficiency": round(efficiency, 4),
        "secondary_harm": round(secondary_harm, 4),
        "prep_ready": round(prep_ready, 4),
        "cascade_score": round(cascade_score, 4),
        "preventable_deaths": preventable_deaths,
        "wasted_actions": wasted_actions,
        "triage_penalty": round(triage_penalty, 4),
        "mutual_aid_penalty": round(ma_penalties, 4)
    }

    return RubricResult(
        time_score=round(time_score, 4),
        efficiency=round(efficiency, 4),
        secondary_harm=round(secondary_harm, 4),
        prep_ready=round(prep_ready, 4),
        mutual_aid_penalty=round(ma_penalties, 4),
        final_score=round(final_score, 4),
        breakdown=breakdown,
        vitals_score_avg=round(vitals_score_avg, 4),
        cascade_score=round(cascade_score, 4),
    )


def grade_from_environment(env) -> RubricResult:
    """Wrapper that passes required state and handles cross-validation."""
    result = grade_episode(env.get_episode_log(), env._dispatch_outcomes_history)

    if result.final_score == 0.0:
        return result  # Don't bother cross-validating a total failure

    # Cross-validation from your original (Ensures logs match patient_manager)
    patients = env._patient_manager.get_all()
    logged_treated = {e["patient_id"] for e in env.get_episode_log() if e.get("event") == "treatment_complete"}
    patients_saved = {p.id for p in patients if p.outcome == "saved"}
    patients_deceased = {p.id for p in patients if p.outcome == "deceased"}

    mismatches = len(patients_saved - logged_treated) + len(logged_treated & patients_deceased) + len(patients_saved & patients_deceased)

    cv_penalty = min(1.0, 0.2 * mismatches)
    icu_penalty = min(1.0, 0.05 * sum(1 for e in env.get_episode_log() if e.get("event") == "icu_boarding"))

    result.final_score = round(max(0.0, min(1.0, result.final_score - cv_penalty - icu_penalty)), 4)
    result.breakdown["cv_penalty"] = cv_penalty
    result.breakdown["icu_penalty"] = icu_penalty

    return result
