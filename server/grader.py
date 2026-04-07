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
        # Empty episode: no patients processed — agent gets 0.0, not 1.0
        time_score = 0.0

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

    # Preemption penalty: each preempted surgery reduces prep_ready
    preempted_surgeries = sum(
        1 for e in episode_log
        if e.get("event") == "surgery_aborted" and e.get("reason") == "or_preempted"
    )
    preemptive_actions = sum(
        1 for e in episode_log
        if e.get("event") == "action_preempt_or"
    )
    prep_ready = max(0.0, prep_ready - 0.1 * preempted_surgeries)

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
    # CASCADE SCORE (Phase 2: secondary patient management)
    # =========================================================================
    cascade_score = grade_cascade_score(episode_log)

    # =========================================================================
    # FINAL SCORE
    # =========================================================================
    raw = (
        0.32 * time_score
        + 0.16 * efficiency
        + 0.16 * secondary_harm
        + 0.16 * prep_ready
        + 0.10 * vitals_score_avg
        + 0.10 * cascade_score
    )
    final_score = max(0.0, min(1.0, raw - mutual_aid_penalty))

    # =========================================================================
    # ANTI-EXPLOIT: empty/no-action episode guard
    # =========================================================================
    patient_events = {
        e["patient_id"]
        for e in episode_log
        if e.get("event") in ("patient_created", "treatment_complete", "patient_deceased")
    }
    if len(patient_events) == 0:
        # No patients processed — agent did nothing, gets 0.0
        final_score = 0.0
        breakdown = {
            "time_score": 0.0,
            "efficiency": 1.0,
            "secondary_harm": 1.0,
            "prep_ready": 1.0,
            "vitals_score_avg": 1.0,
            "cascade_score": 1.0,
            "mutual_aid_penalty": 0.0,
            "anti_exploit": "no_patients_processed",
        }
        return RubricResult(
            time_score=0.0,
            efficiency=1.0,
            secondary_harm=1.0,
            prep_ready=1.0,
            mutual_aid_penalty=0.0,
            final_score=0.0,
            breakdown=breakdown,
            vitals_score_avg=1.0,
            cascade_score=1.0,
        )

    # =========================================================================
    # ANTI-EXPLOIT: E2 Idle strategy detection
    # =========================================================================
    dispatches = [e for e in episode_log if e.get("event") in ("dispatch", "action_dispatch")]
    treatments = [e for e in episode_log if e.get("event") == "treatment_complete"]
    if not dispatches and not treatments and len(patient_times) > 0:
        all_deceased = all(not data.get("treated", False) for data in patient_times.values())
        if all_deceased:
            # Complete inaction — all patients died with no agent action
            time_score = max(0.0, time_score - 0.3)
            breakdown_extra = {"anti_exploit": "complete_inaction"}
        else:
            breakdown_extra = {}
    else:
        breakdown_extra = {}

    # E1: Cascade farming penalty — delayed primary treatment + cascade spawns
    cascade_patients = [e for e in episode_log if e.get("event") == "secondary_patient_spawned"]
    primary_patients = [e for e in episode_log if e.get("event") == "patient_created" and not e.get("is_secondary")]
    if cascade_patients and primary_patients:
        primary_pid = primary_patients[0].get("patient_id")
        for entry in episode_log:
            if (entry.get("event") == "treatment_complete"
                    and entry.get("patient_id") == primary_pid):
                eff = entry.get("effective_time", 0)
                tgt = entry.get("target_time", 90)
                if eff > tgt * 1.5:  # 50% over target
                    farming_penalty = min(0.2, 0.1 * len(cascade_patients))
                    time_score = max(0.0, time_score - farming_penalty)
                    breakdown_extra["cascade_farming_penalty"] = round(farming_penalty, 4)
                break

    breakdown = {
        "time_score": time_score,
        "efficiency": efficiency,
        "secondary_harm": secondary_harm,
        "prep_ready": prep_ready,
        "vitals_score_avg": vitals_score_avg,
        "cascade_score": cascade_score,
        "mutual_aid_penalty": mutual_aid_penalty,
        "unused_specialist_pages": unused_specialist,
        "wasted_or_preps": wasted_or_preps,
        "premature_blood_emerg": premature_blood_emerg,
        "secondary_deaths": secondary_deaths,
        "secondary_patients": secondary_patients,
        "mutual_aid_calls": num_calls,
        "preemptive_actions": preemptive_actions,
        "preempted_surgeries": preempted_surgeries,
    }
    breakdown.update(breakdown_extra)

    # Recompute final score with adjusted time_score
    raw = (
        0.32 * time_score
        + 0.16 * efficiency
        + 0.16 * secondary_harm
        + 0.16 * prep_ready
        + 0.10 * vitals_score_avg
        + 0.10 * cascade_score
    )
    final_score = max(0.0, min(1.0, raw - mutual_aid_penalty))

    return RubricResult(
        time_score=round(time_score, 4),
        efficiency=round(efficiency, 4),
        secondary_harm=round(secondary_harm, 4),
        prep_ready=round(prep_ready, 4),
        mutual_aid_penalty=round(mutual_aid_penalty, 4),
        final_score=round(final_score, 4),
        breakdown=breakdown,
        vitals_score_avg=round(vitals_score_avg, 4),
        cascade_score=round(cascade_score, 4),
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
    patients = env._patient_manager.get_all()

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

    # =========================================================================
    # ICU BOARDING PENALTY (Task 13)
    # =========================================================================
    boarding_events = [
        e for e in log
        if e.get("event") == "icu_boarding"
    ]
    num_boarding = len(boarding_events)
    icu_boarding_penalty = min(1.0, 0.05 * num_boarding)

    # Grade from the log first
    result = grade_episode(log)

    # Apply cross-validation penalty
    result.final_score = max(0.0, result.final_score - cross_validation_penalty)
    result.breakdown["cross_validation_mismatches"] = num_mismatches
    result.breakdown["cross_validation_penalty"] = round(cross_validation_penalty, 4)
    result.breakdown["treated_missing_log"] = sorted(treated_missing_log)
    result.breakdown["treated_but_deceased"] = sorted(treated_but_deceased)

    # Apply ICU boarding penalty
    result.final_score = max(0.0, result.final_score - icu_boarding_penalty)
    result.breakdown["icu_boarding_events"] = num_boarding
    result.breakdown["icu_boarding_penalty"] = round(icu_boarding_penalty, 4)

    return result
