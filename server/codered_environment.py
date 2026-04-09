"""CodeRedEnvironment — Core OpenEnv environment with wired subsystems."""

from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

from server.models import (
    CodeRedAction,
    CodeRedObservation,
    CodeRedState,
)
from server.models.entities import (
    AmbulanceState,
    AmbulanceEquipment,
    AmbulanceStatus,
    BloodBankState,
    EdgeState,
    HospitalState,
    OperatingRoom,
    ORStatus,
    Patient,
    PatientCondition,
    PatientStatus,
    PatientTier,
    RoadNetworkState,
    SpecialistStatus,
)
from server.subsystems.constants import (
    AMBULANCES,
    BLOOD_TYPES,
    CITY_NODES,
    EPISODE_START_HOUR,
    HOSPITALS,
    HOSPITAL_MORTALITY_RATES,
    PATIENT_CONDITION_REQUIREMENTS,
    PATIENT_TARGET_TIMES,
    PROCEDURE_BY_CONDITION,
    SCENE_TIME,
    SPECIALIST_BY_CONDITION,
    TASK_CONFIG,
)

# =============================================================================
# Reward configuration
# =============================================================================
REWARD_CONFIG = {
    # Condition urgency multipliers — cardiac/stroke are the most time-critical
    "urgency_weight": {
        "cardiac": 2.0,
        "stroke":  1.8,
        "trauma":  1.4,
        "general": 1.0,
    },

    # Vitals delta weight (raw delta * urgency * this)
    "vitals_delta_weight": 3.0,

    # Per-step holding penalties — bleed reward for every step of inaction
    "waiting_penalty_per_step":     -0.04,   # waiting / deteriorating / critical
    "dispatched_penalty_per_step":  -0.01,   # ambulance assigned, en-route
    "transporting_penalty_per_step":-0.005,  # patient picked up, moving

    # One-shot milestone rewards (fired on status transition, urgency-scaled)
    "milestone": {
        "dispatched":      +0.30,
        "transporting":    +0.20,
        "in_treatment":    +0.50,
        "treated_on_time": +3.00,   # treated within target window
        "treated_late":    +1.20,   # treated but over-target (fades with lateness)
        "deceased":        -4.00,
    },

    # Extra speed bonus stacked on treated_on_time for being early
    "time_bonus_max": 2.00,

    # Hospital routing (fired immediately on AssignHospital)
    "correct_hospital_bonus":  +0.25,
    "wrong_hospital_penalty":  -0.50,
    "diversion_penalty":       -0.30,
    "nearest_capable_bonus":   +0.15,  # chosen hospital is ≤2 min from optimal

    # ALS / BLS triage (scored when ground truth is revealed)
    "als_correct_dispatch":      +0.20,
    "bls_correct_dispatch":      +0.10,
    "als_undertriage_penalty":   -0.60,  # BLS sent, ALS needed — dangerous
    "als_overtriage_penalty":    -0.15,  # ALS sent, BLS sufficient — wasteful
    "no_dispatch_unnecessary":   +0.05,
    "no_dispatch_urgent_penalty":-0.80,

    # Proactive preparation (OR prep / specialist page that helps an inbound patient)
    "or_prep_used_bonus":          +0.20,
    "specialist_paged_used_bonus": +0.15,

    # Blood management
    "emergency_blood_needed_bonus": +0.05,  # emergency release for critical patient

    # Mutual aid
    "mutual_aid_saves_bonus":   +0.40,
    "mutual_aid_wasted_penalty":-0.25,

    # Wasted / spam
    "wasted_action_penalty":    -0.08,
    "spam_penalty_per_repeat":  -0.15,
    "spam_threshold":           5,

    # Step reward clamp
    "step_reward_clamp": (-3.0, 5.0),
}


class CodeRedEnvironment(Environment):
    """OpenEnv environment for emergency medical coordination."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        from server.subsystems.road_network import RoadNetwork
        from server.subsystems.hospital_system import HospitalSystem
        from server.subsystems.blood_bank import BloodBankSystem
        from server.subsystems.disruption_engine import DisruptionEngine
        from server.subsystems.cascade_engine import CascadeEngine
        from server.subsystems.patient_manager import PatientManager
        from server.subsystems.ambulance_manager import AmbulanceManager
        from server.subsystems.mutual_aid import MutualAidManager

        self._rng: Optional[Any] = None
        self._patients: List[Patient] = []
        self._road_network = RoadNetwork()
        self._hospital_system = HospitalSystem(episode_start_hour=EPISODE_START_HOUR)
        self._blood_bank = BloodBankSystem()
        self._disruption_engine = DisruptionEngine()
        self._patient_manager = PatientManager()
        self._ambulance_manager = AmbulanceManager(AMBULANCES)
        self._mutual_aid_manager: Optional[MutualAidManager] = None
        self._state: CodeRedState = CodeRedState()
        self._alerts: List[str] = []
        self._pending_blood_queries: Dict[str, int] = {}
        self._pending_or_queries: Dict[str, int] = {}
        self._active_disruptions: List[Dict] = []
        self._episode_log: List[dict] = []
        self._active_surgeries: Dict[str, Dict] = {}
        self._prev_vitals: Dict[str, float] = {}
        self._prev_patient_status: Dict[str, str] = {}
        self._pending_calls: List = []
        self._pending_call_countdown: Dict[str, int] = {}
        self._pending_call_to_patient: Dict[str, Dict] = {}
        self._ambulance_pending_patient: Dict[str, Dict] = {}
        self._dispatch_outcomes_history: List = []
        self._cascade_engine = CascadeEngine()
        self._step_count_since_last_call: int = 0
        self._phase2_has_activity: bool = False
        self._action_counts: Dict[str, int] = {}

    # =========================================================================
    # Public API
    # =========================================================================

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: str = "task1",
        **kwargs: Any,
    ) -> CodeRedObservation:
        """Reset the environment for a new episode."""
        import random

        self._rng = random.Random(seed)

        from server.subsystems.road_network import RoadNetwork
        from server.subsystems.hospital_system import HospitalSystem
        from server.subsystems.blood_bank import BloodBankSystem
        from server.subsystems.disruption_engine import DisruptionEngine
        from server.subsystems.cascade_engine import CascadeEngine
        from server.subsystems.patient_manager import PatientManager
        from server.subsystems.ambulance_manager import AmbulanceManager
        from server.subsystems.mutual_aid import MutualAidManager

        self._road_network = RoadNetwork()
        self._hospital_system = HospitalSystem(episode_start_hour=EPISODE_START_HOUR)
        self._blood_bank = BloodBankSystem()
        self._disruption_engine.reset(seed=seed or 0, task_id=task_id)
        self._cascade_engine.reset(seed=seed or 0, episode_config=TASK_CONFIG.get(task_id, {}))
        self._cascade_engine.set_callback(self._cascade_callback)
        self._patient_manager.reset(task_id, self._rng)
        self._patients = self._patient_manager.patients
        self._ambulance_manager = AmbulanceManager(AMBULANCES)
        self._mutual_aid_manager = MutualAidManager(
            available_calls=TASK_CONFIG[task_id]["mutual_aid_calls"],
            seed=seed,
        )

        self._state = CodeRedState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=task_id,
            cum_reward=0.0,
            max_steps=TASK_CONFIG[task_id]["max_steps"],
            mutual_aid_available=TASK_CONFIG[task_id]["mutual_aid_calls"],
            mutual_aid_used=0,
        )

        self._alerts = []
        self._pending_blood_queries = {}
        self._pending_or_queries = {}
        self._active_disruptions = []
        self._episode_log = []
        self._active_surgeries = {}
        self._prev_vitals = {}
        self._prev_patient_status = {}
        self._pending_calls = []
        self._pending_call_countdown = {}
        self._pending_call_to_patient = {}
        self._dispatch_outcomes_history = []
        self._step_count_since_last_call = 0
        self._phase2_has_activity = False
        self._ambulance_pending_patient: Dict[str, Dict] = {}
        self._action_counts: Dict[str, int] = {}

        for p in self._patients:
            target_time = PATIENT_TARGET_TIMES.get(p.condition, 60)
            self._episode_log.append({
                "step": 0,
                "patient_id": p.id,
                "event": "patient_created",
                "condition": p.condition,
                "is_secondary": False,
                "target_time": target_time,
            })

        for p in self._patients:
            self._prev_vitals[p.id] = p.vitals_score
            self._prev_patient_status[p.id] = p.status

        return self._build_observation()

    def step(
        self,
        action: CodeRedAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> CodeRedObservation:
        """Execute one timestep and return the observation."""
        self._state.step_count += 1
        self._alerts = []
        self._action_counts = {}  # reset per-step spam counter

        self._advance_time()
        self._execute_action(action)

        # Track action type for spam detection
        action_key = type(action).__name__
        self._action_counts[action_key] = self._action_counts.get(action_key, 0) + 1

        reward = self._compute_step_reward()
        self._state.cum_reward += reward

        done = self._check_done()
        obs = self._build_observation(done=done)
        return obs

    @property
    def state(self) -> CodeRedState:
        return self._state

    def get_episode_log(self) -> List[dict]:
        return self._episode_log.copy()

    # =========================================================================
    # Time Advancement  (unchanged from original)
    # =========================================================================

    def _advance_time(self) -> None:
        """Advance all systems by 1 minute."""
        active_patients = [p for p in self._patients if p.status not in ("treated", "deceased")]
        overcrowding_modifier = self._cascade_engine.check_overcrowding(len(active_patients))
        self._patient_manager.tick(
            self._patient_manager.get_onset_steps(),
            self._state.step_count,
            overcrowding_modifier=overcrowding_modifier,
        )
        for p in self._patients:
            if p.status == "deceased" and not any(
                log.get("patient_id") == p.id and log.get("event") == "patient_deceased"
                for log in self._episode_log
            ):
                self._episode_log.append({
                    "step": self._state.step_count,
                    "patient_id": p.id,
                    "event": "patient_deceased",
                    "reason": "timeout",
                })
                if p.icu_status == "admitted" and p.assigned_hospital:
                    self._hospital_system.release_icu_bed(p.assigned_hospital)

        self._ambulance_manager.tick()
        for amb_id, amb in self._ambulance_manager.all().items():
            if amb.arrived_with_patient:
                amb.arrived_with_patient = False
                if amb.patient_id:
                    patient = next((pt for pt in self._patients if pt.id == amb.patient_id), None)
                    if patient and patient.status == "transporting":
                        self._do_treatment_arrival(patient, amb_id)
            elif amb.status == "on_scene" and amb.eta_minutes == 0:
                if amb.patient_id:
                    resolved_patient_id = amb.patient_id
                    if resolved_patient_id.startswith("CALL:"):
                        call_id = resolved_patient_id[5:]
                        pending_entry = self._pending_call_to_patient.get(call_id)
                        if pending_entry and pending_entry.get("spawned"):
                            resolved_patient_id = pending_entry["spawned_patient_id"]
                        else:
                            self._force_spawn_at_scene(call_id, amb_id)
                            reloaded = self._pending_call_to_patient.get(call_id, {})
                            if reloaded and reloaded.get("spawned"):
                                resolved_patient_id = reloaded["spawned_patient_id"]
                            else:
                                resolved_patient_id = None

                    if resolved_patient_id:
                        patient = next((pt for pt in self._patients if pt.id == resolved_patient_id), None)
                        if patient and patient.status == "waiting":
                            patient.status = "transporting"
                            patient.assigned_ambulance = amb_id
                            amb.patient_id = resolved_patient_id
                else:
                    pending = self._ambulance_pending_patient.get(amb_id)
                    if pending:
                        call_id = pending["call_id"]
                        entry = self._pending_call_to_patient.get(call_id, {})
                        if not entry.get("spawned"):
                            self._force_spawn_at_scene(call_id, amb_id)
                            reloaded = self._pending_call_to_patient.get(call_id, {})
                            if reloaded and reloaded.get("spawned"):
                                spawned_pid = reloaded["spawned_patient_id"]
                                patient = next((pt for pt in self._patients if pt.id == spawned_pid), None)
                                if patient and patient.status == "waiting":
                                    patient.status = "transporting"
                                    patient.assigned_ambulance = amb_id
                                    amb.patient_id = spawned_pid

        def _ma_arrival_callback(ma_id: str, patient_id: str) -> None:
            patient = self._patient_manager.get(patient_id)
            if patient and patient.status == "transporting":
                self._do_treatment_arrival(patient, ma_id)

        ma_arrivals = self._mutual_aid_manager.tick(
            step_count=self._state.step_count,
            patient_manager=self._patient_manager,
            hospital_system=self._hospital_system,
            arrival_callback=_ma_arrival_callback,
        )
        for event in ma_arrivals:
            if event.get("patient_changed") and event.get("patient_id"):
                for entry in self._episode_log:
                    if (entry.get("event") == "mutual_aid_called"
                            and entry.get("ambulance_id") == event["ambulance_id"]
                            and entry.get("patient_id") != event["patient_id"]):
                        entry["patient_id"] = event["patient_id"]
                        entry["optimal_arrival_step"] = event["actual_arrival_step"]
                        break

            self._episode_log.append({
                "step": self._state.step_count,
                "event": "mutual_aid_arrived",
                "ambulance_id": event["ambulance_id"],
                "patient_id": event.get("patient_id"),
                "actual_arrival_step": event.get("actual_arrival_step", self._state.step_count),
                "had_patient": event.get("had_patient", False),
            })
            if event.get("had_patient"):
                self._alerts.append(
                    f"Mutual aid {event['ambulance_id']} arrived, "
                    f"assigned to {event['patient_id']} → {event.get('hospital_id', '?')}"
                )
            else:
                self._alerts.append(
                    f"Mutual aid {event['ambulance_id']} arrived but no waiting patient — wasted"
                )

        pre_tick_surgery_patients = set(self._active_surgeries.keys())
        self._hospital_system.tick()

        for patient_id in list(pre_tick_surgery_patients):
            if patient_id not in self._active_surgeries:
                continue
            surgery_info = self._active_surgeries[patient_id]
            hosp = self._hospital_system.get(surgery_info["hospital_id"])
            if hosp:
                or_obj = next(
                    (o for o in hosp.operating_rooms if o.index == surgery_info["or_index"]),
                    None,
                )
                if or_obj and or_obj.status == "idle" and or_obj.patient_id is None:
                    patient = self._patient_manager.get(patient_id)
                    if patient and patient.status == "in_treatment":
                        effective_time = self._state.step_count - surgery_info["start_step"]
                        target_time = PATIENT_TARGET_TIMES.get(patient.condition, 60)
                        hosp_id = patient.assigned_hospital
                        mort_rates = HOSPITAL_MORTALITY_RATES.get(hosp_id, {})
                        mort_rate = mort_rates.get(patient.condition, 0.0)
                        survived = self._rng.random() >= mort_rate

                        if survived:
                            self._patient_manager.mark_treated(patient_id, effective_time)
                            patient.outcome = "saved"
                        else:
                            self._patient_manager.mark_deceased(patient_id, reason="hospital_mortality")
                            patient.outcome = "deceased"

                        cascade_enabled = TASK_CONFIG.get(self._state.task_id, {}).get("cascade_enabled", False)
                        if cascade_enabled:
                            self._cascade_engine.on_outcome(
                                patient_id, patient.condition, patient.outcome, self._state.step_count
                            )

                        self._hospital_system.release_icu_bed(hosp_id)
                        self._episode_log.append({
                            "step": self._state.step_count,
                            "patient_id": patient_id,
                            "event": "treatment_complete",
                            "effective_time": effective_time,
                            "target_time": target_time,
                            "vitals_at_treatment": patient.vitals_score,
                            "icu_status": patient.icu_status,
                        })
                        del self._active_surgeries[patient_id]

        for patient in self._patients:
            if (
                patient.status == "in_treatment"
                and patient.assigned_hospital
                and patient.id not in self._active_surgeries
            ):
                hosp = self._hospital_system.get(patient.assigned_hospital)
                if hosp:
                    idle_or = self._hospital_system.get_idle_or(patient.assigned_hospital)
                    if idle_or:
                        procedure_type = PROCEDURE_BY_CONDITION.get(patient.condition, "general")
                        result = self._hospital_system.start_surgery(
                            patient.assigned_hospital,
                            idle_or.index,
                            procedure_type,
                            patient.id,
                            duration_minutes=30,
                        )
                        if result["success"]:
                            self._active_surgeries[patient.id] = {
                                "hospital_id": patient.assigned_hospital,
                                "or_index": idle_or.index,
                                "start_step": self._state.step_count,
                                "procedure_type": procedure_type,
                            }
                            self._episode_log.append({
                                "step": self._state.step_count,
                                "patient_id": patient.id,
                                "event": "treatment_started",
                                "hospital_id": patient.assigned_hospital,
                                "or_index": idle_or.index,
                            })

        self._blood_bank.tick()
        completed = self._blood_bank.flush_completed_crossmatches()
        for entry in completed:
            self._alerts.append(
                f"Crossmatch complete: {entry['units']} units "
                f"{entry['blood_type']} reserved for {entry['patient_id']}"
            )

        self._road_network.tick()
        self._road_network_tick_disruptions()

        for patient_id in list(self._pending_blood_queries.keys()):
            self._pending_blood_queries[patient_id] -= 1
            if self._pending_blood_queries[patient_id] <= 0:
                del self._pending_blood_queries[patient_id]
                self._reveal_blood_type(patient_id)

        for key in list(self._pending_or_queries.keys()):
            self._pending_or_queries[key] -= 1
            if self._pending_or_queries[key] <= 0:
                del self._pending_or_queries[key]

        events = self._disruption_engine.roll_disruptions(
            step=self._state.step_count,
            road_network=self._road_network,
            hospital_system=self._hospital_system,
        )
        for event in events:
            if event["target_type"] == "surge":
                num_secondary = self._rng.randint(1, 2)
                conditions = self._rng.sample(["cardiac", "stroke", "trauma", "general"], k=num_secondary)
                for cond in conditions:
                    sec_patient = self._patient_manager.spawn_secondary(
                        condition=cond,
                        onset_step=self._state.step_count,
                    )
                    self._patients = self._patient_manager.patients
                    self._alerts.append(f"Surge: secondary patient {sec_patient.id} ({cond}) arrived")
                    self._episode_log.append({
                        "step": self._state.step_count,
                        "patient_id": sec_patient.id,
                        "event": "patient_created",
                        "condition": cond,
                        "is_secondary": True,
                        "target_time": PATIENT_TARGET_TIMES.get(cond, 60),
                    })
            else:
                self._alerts.append(
                    f"Disruption: {event['disruption_type']} on {event['target']}"
                )

        self._cascade_engine.tick()

        use_call_queue = TASK_CONFIG.get(self._state.task_id, {}).get("use_call_queue", False)
        if use_call_queue:
            self._step_count_since_last_call += 1
            if self._step_count_since_last_call >= 8:
                self._spawn_dispatch_call()
                self._step_count_since_last_call = 0
            for call_id in list(self._pending_call_countdown.keys()):
                self._pending_call_countdown[call_id] -= 1
                if self._pending_call_countdown[call_id] <= 0:
                    self._spawn_patient_from_call(call_id)

    def _road_network_tick_disruptions(self) -> None:
        active = self._road_network.get_active_disruptions()
        for disp in active:
            if disp["remaining_steps"] <= 0:
                self._road_network.clear_disruption(disp["from_node"], disp["to_node"])
                self._alerts.append(f"Road cleared: {disp['from_node']} <-> {disp['to_node']}")

    def _reveal_blood_type(self, patient_id: str) -> None:
        patient = self._patient_manager.get(patient_id)
        if patient:
            patient.blood_type = self._rng.choice(BLOOD_TYPES)
            self._alerts.append(f"Blood type revealed for {patient_id}: {patient.blood_type}")

    def _do_treatment_arrival(self, patient, ambulance_id: str) -> None:
        if patient.assigned_hospital:
            hosp = self._hospital_system.get(patient.assigned_hospital)
            if hosp:
                procedure_type = PROCEDURE_BY_CONDITION.get(patient.condition, "general")
                idle_or = self._hospital_system.get_idle_or(patient.assigned_hospital)
                specialist_type = SPECIALIST_BY_CONDITION.get(patient.condition, "general_surgeon")
                specialist_available = self._hospital_system.is_specialist_available(
                    patient.assigned_hospital, specialist_type
                )
                self._episode_log.append({
                    "step": self._state.step_count,
                    "patient_id": patient.id,
                    "event": "patient_arrived_hospital",
                    "hospital_id": patient.assigned_hospital,
                    "or_ready": idle_or is not None,
                    "specialist_available": specialist_available,
                })
                if self._hospital_system.can_treat(patient.assigned_hospital, patient.condition):
                    icu_available = self._hospital_system.consume_icu_bed(patient.assigned_hospital)
                    if icu_available:
                        patient.icu_status = "admitted"
                    else:
                        patient.icu_status = "boarding"
                        self._episode_log.append({
                            "step": self._state.step_count,
                            "patient_id": patient.id,
                            "event": "icu_boarding",
                            "hospital_id": patient.assigned_hospital,
                        })
                if idle_or:
                    result = self._hospital_system.start_surgery(
                        patient.assigned_hospital,
                        idle_or.index,
                        procedure_type,
                        patient.id,
                        duration_minutes=30,
                    )
                    if result["success"]:
                        patient.status = "in_treatment"
                        patient.arrival_hospital_step = self._state.step_count
                        self._active_surgeries[patient.id] = {
                            "hospital_id": patient.assigned_hospital,
                            "or_index": idle_or.index,
                            "start_step": self._state.step_count,
                            "procedure_type": procedure_type,
                        }
                        self._episode_log.append({
                            "step": self._state.step_count,
                            "patient_id": patient.id,
                            "event": "treatment_started",
                            "hospital_id": patient.assigned_hospital,
                            "or_index": idle_or.index,
                        })
                        self._ambulance_manager.mark_available(ambulance_id)
                else:
                    patient.status = "in_treatment"

    # =========================================================================
    # Action Execution  (unchanged except _do_assign_hospital — see below)
    # =========================================================================

    def _execute_action(self, action: CodeRedAction) -> None:
        from .models.actions import (
            DispatchAmbulance, PrepareOR, PageSpecialist, AssignHospital,
            PreemptOR, AllocateBlood, TransferBlood, RequestMutualAid,
            QueryBloodType, QueryORStatus, MaintainPlan, TriageCall,
            DispatchALS, DispatchBLS,
        )
        if isinstance(action, DispatchAmbulance):
            self._do_dispatch_ambulance(action)
        elif isinstance(action, DispatchALS):
            self._do_dispatch_als(action)
        elif isinstance(action, DispatchBLS):
            self._do_dispatch_bls(action)
        elif isinstance(action, TriageCall):
            self._do_triage_call(action)
        elif isinstance(action, PrepareOR):
            self._do_prepare_or(action)
        elif isinstance(action, PageSpecialist):
            self._do_page_specialist(action)
        elif isinstance(action, AssignHospital):
            self._do_assign_hospital(action)
        elif isinstance(action, PreemptOR):
            self._do_preempt_or(action)
        elif isinstance(action, AllocateBlood):
            self._do_allocate_blood(action)
        elif isinstance(action, TransferBlood):
            self._do_transfer_blood(action)
        elif isinstance(action, RequestMutualAid):
            self._do_request_mutual_aid(action)
        elif isinstance(action, QueryBloodType):
            self._do_query_blood_type(action)
        elif isinstance(action, QueryORStatus):
            self._do_query_or_status(action)
        elif isinstance(action, MaintainPlan):
            pass

    def _do_dispatch_ambulance(self, action) -> None:
        result = self._ambulance_manager.dispatch(
            action.ambulance_id, action.target_node, self._road_network
        )
        if not result["success"]:
            self._alerts.append(f"Dispatch failed: {result['reason']}")
            self._episode_log.append({
                "step": self._state.step_count,
                "event": "action_dispatch",
                "ambulance_id": action.ambulance_id,
                "result": "wasted",
                "reason": result["reason"],
            })
            return
        waiting_patient = next(
            (p for p in self._patients
             if p.status == "waiting" and p.location_node == action.target_node),
            None,
        )
        if waiting_patient:
            waiting_patient.assigned_ambulance = action.ambulance_id
            waiting_patient.status = "dispatched"
            amb = self._ambulance_manager.get(action.ambulance_id)
            if amb:
                amb.patient_id = waiting_patient.id
            self._episode_log.append({
                "step": self._state.step_count,
                "patient_id": waiting_patient.id,
                "event": "dispatch",
                "ambulance_id": action.ambulance_id,
                "result": "success",
            })
        else:
            self._episode_log.append({
                "step": self._state.step_count,
                "event": "action_dispatch",
                "ambulance_id": action.ambulance_id,
                "target_node": action.target_node,
                "result": "wasted",
            })

    def _do_prepare_or(self, action) -> None:
        result = self._hospital_system.prepare_or(action.hospital_id, action.procedure_type)
        if not result["success"]:
            self._alerts.append(f"PrepareOR failed: {result['reason']}")
            self._episode_log.append({
                "step": self._state.step_count,
                "event": "action_prepare_or",
                "hospital_id": action.hospital_id,
                "procedure_type": action.procedure_type,
                "result": "wasted",
                "reason": result["reason"],
            })
            return
        patient_en_route = any(
            p.status in ("dispatched", "transporting")
            and getattr(p, "assigned_hospital", None) == action.hospital_id
            for p in self._patients
        )
        prep_result = "success" if patient_en_route else "wasted"
        if prep_result == "wasted":
            self._alerts.append(
                f"PrepareOR wasted: no patient en route to {action.hospital_id}"
            )
        self._episode_log.append({
            "step": self._state.step_count,
            "event": "action_prepare_or",
            "hospital_id": action.hospital_id,
            "procedure_type": action.procedure_type,
            "or_index": result["or_index"],
            "result": prep_result,
        })

    def _do_page_specialist(self, action) -> None:
        result = self._hospital_system.page_specialist(action.hospital_id, action.specialist_type)
        if not result["success"]:
            self._alerts.append(f"PageSpecialist failed: {result['reason']}")
            self._episode_log.append({
                "step": self._state.step_count,
                "event": "action_page_specialist",
                "hospital_id": action.hospital_id,
                "specialist_type": action.specialist_type,
                "result": "wasted",
                "reason": result["reason"],
            })
            return
        self._episode_log.append({
            "step": self._state.step_count,
            "event": "action_page_specialist",
            "hospital_id": action.hospital_id,
            "specialist_type": action.specialist_type,
            "result": "success",
        })

    def _do_assign_hospital(self, action) -> None:
        """Assign hospital with immediate routing-quality reward signal."""
        patient = self._patient_manager.get(action.patient_id)
        if patient is None:
            self._alerts.append(f"AssignHospital failed: patient {action.patient_id} not found")
            return

        routing_reward = self._score_hospital_routing(patient, action.hospital_id)

        if not self._hospital_system.can_treat(action.hospital_id, patient.condition):
            self._alerts.append(
                f"AssignHospital: {action.hospital_id} cannot treat "
                f"{patient.condition} (needs {PATIENT_CONDITION_REQUIREMENTS[patient.condition][0]})"
            )
            self._episode_log.append({
                "step": self._state.step_count,
                "event": "action_assign_hospital",
                "patient_id": action.patient_id,
                "hospital_id": action.hospital_id,
                "result": "failed",
                "_routing_reward": routing_reward,
            })
            return

        if self._hospital_system.get(action.hospital_id).on_diversion:
            self._alerts.append(f"AssignHospital: {action.hospital_id} is on diversion")
            self._episode_log.append({
                "step": self._state.step_count,
                "event": "action_assign_hospital",
                "patient_id": action.patient_id,
                "hospital_id": action.hospital_id,
                "result": "diversion",
                "_routing_reward": routing_reward,
            })
            return

        patient.assigned_hospital = action.hospital_id
        self._episode_log.append({
            "step": self._state.step_count,
            "event": "action_assign_hospital",
            "patient_id": action.patient_id,
            "hospital_id": action.hospital_id,
            "result": "success",
            "_routing_reward": routing_reward,
        })

    def _do_preempt_or(self, action) -> None:
        result = self._hospital_system.preempt_or(action.hospital_id, action.or_index)
        if not result["success"]:
            self._alerts.append(f"PreemptOR failed: {result['reason']}")
            self._episode_log.append({
                "step": self._state.step_count,
                "event": "action_preempt_or",
                "hospital_id": action.hospital_id,
                "or_index": action.or_index,
                "result": "failed",
                "reason": result["reason"],
            })
            return
        self._alerts.append(
            f"OR preempted at {action.hospital_id} OR {action.or_index}: "
            f"harm={result['harm']:.2f}, recovery={result['recovery_time']}min"
        )
        self._episode_log.append({
            "step": self._state.step_count,
            "event": "action_preempt_or",
            "hospital_id": action.hospital_id,
            "or_index": action.or_index,
            "result": "success",
            "harm": result["harm"],
            "recovery_time": result["recovery_time"],
        })

    def _do_allocate_blood(self, action) -> None:
        if action.emergency:
            result = self._blood_bank.emergency_release(
                action.hospital_id, action.patient_id, action.blood_type, action.units
            )
            if not result["success"]:
                self._alerts.append(f"AllocateBlood failed: {result['reason']}")
                self._episode_log.append({
                    "step": self._state.step_count,
                    "event": "action_allocate_blood",
                    "hospital_id": action.hospital_id,
                    "patient_id": action.patient_id,
                    "blood_type": action.blood_type,
                    "units": action.units,
                    "result": "wasted",
                    "reason": result["reason"],
                })
                return
            self._alerts.append(
                f"Emergency blood: {result['units']} units {result['blood_type']} "
                f"released for {action.patient_id} at {action.hospital_id}"
            )
            self._episode_log.append({
                "step": self._state.step_count,
                "event": "action_allocate_blood",
                "hospital_id": action.hospital_id,
                "patient_id": action.patient_id,
                "blood_type": action.blood_type,
                "units": action.units,
                "result": "success",
            })
        else:
            result = self._blood_bank.start_crossmatch(
                action.hospital_id, action.patient_id, action.blood_type, action.units
            )
            if not result["success"]:
                self._alerts.append(f"AllocateBlood failed: {result['reason']}")
                self._episode_log.append({
                    "step": self._state.step_count,
                    "event": "action_allocate_blood",
                    "hospital_id": action.hospital_id,
                    "patient_id": action.patient_id,
                    "blood_type": action.blood_type,
                    "units": action.units,
                    "result": "wasted",
                    "reason": result["reason"],
                })
                return
            self._episode_log.append({
                "step": self._state.step_count,
                "event": "action_allocate_blood",
                "hospital_id": action.hospital_id,
                "patient_id": action.patient_id,
                "blood_type": action.blood_type,
                "units": action.units,
                "result": "success",
            })

    def _do_transfer_blood(self, action) -> None:
        result = self._blood_bank.transfer(
            action.from_hospital, action.to_hospital, action.blood_type, action.units
        )
        if not result["success"]:
            self._alerts.append(f"TransferBlood failed: {result['reason']}")
            return
        self._alerts.append(
            f"Blood transfer: {action.units} units {action.blood_type} "
            f"from {action.from_hospital} to {action.to_hospital}"
        )

    def _do_request_mutual_aid(self, action) -> None:
        if self._mutual_aid_manager.get_available() <= 0:
            self._alerts.append("Mutual aid failed: no calls remaining")
            return
        ma_id = self._mutual_aid_manager.request(
            step_count=self._state.step_count,
            road_network=self._road_network,
            patients=self._patients,
        )
        if ma_id is None:
            self._alerts.append("Mutual aid failed: no calls remaining")
            return
        self._state.mutual_aid_available = self._mutual_aid_manager.get_available()
        self._state.mutual_aid_used = self._mutual_aid_manager.get_used()
        pending = self._mutual_aid_manager.get_pending().get(ma_id)
        if pending:
            target_pid = pending.patient_id
            self._alerts.append(
                f"Mutual aid requested: {ma_id} for {target_pid} (ETA step {pending.arrival_step})"
                if target_pid else f"Mutual aid requested: {ma_id} — no waiting patients"
            )
            self._episode_log.append({
                "step": self._state.step_count,
                "event": "mutual_aid_called",
                "ambulance_id": ma_id,
                "patient_id": target_pid,
                "optimal_arrival_step": pending.arrival_step,
            })

    def _do_query_blood_type(self, action) -> None:
        patient = self._patient_manager.get(action.patient_id)
        if patient is None:
            self._alerts.append(f"QueryBloodType failed: patient {action.patient_id} not found")
            return
        self._pending_blood_queries[action.patient_id] = 5

    def _do_query_or_status(self, action) -> None:
        hosp = self._hospital_system.get(action.hospital_id)
        if hosp is None:
            return
        or_obj = next((o for o in hosp.operating_rooms if o.index == action.or_index), None)
        if or_obj:
            self._alerts.append(
                f"OR {action.or_index} at {action.hospital_id}: {or_obj.status} "
                f"(remaining: {or_obj.minutes_remaining})"
            )

    def _do_triage_call(self, action) -> None:
        from server.models.entities import DispatchCategory
        from server.subsystems.constants import DISPATCH_CATEGORY_MAP, ALS_NEEDED_PROB
        call = next((c for c in self._pending_calls if c["call_id"] == action.call_id), None)
        if call is None:
            self._alerts.append(f"TriageCall: call_id {action.call_id} not found")
            return
        decision = action.decision
        if decision in ("dispatch_als", "dispatch_bls"):
            amb = self._ambulance_manager.get(action.ambulance_id)
            if amb is None:
                self._alerts.append(f"Ambulance {action.ambulance_id} not found")
                return
            expected_equip = "ALS" if decision == "dispatch_als" else "BLS"
            if amb.equipment != expected_equip:
                self._alerts.append(f"Wrong equipment")
                return
            if amb.status != "available":
                self._alerts.append(f"Ambulance not available")
                return
            result = self._ambulance_manager.dispatch(action.ambulance_id, call["location_node"], self._road_network)
            if not result["success"]:
                self._alerts.append(f"Dispatch failed")
                return
            self._pending_call_to_patient[action.call_id] = {
                "ambulance_id": action.ambulance_id,
                "location_node": call["location_node"],
                "spawned": False,
                "spawned_patient_id": None,
            }
            self._phase2_has_activity = True
            self._ambulance_pending_patient[action.ambulance_id] = {
                "call_id": action.call_id,
                "location_node": call["location_node"],
            }
            amb = self._ambulance_manager.get(action.ambulance_id)
            if amb:
                amb.patient_id = f"CALL:{action.call_id}"
                amb.target_node = call["location_node"]
            call["assigned_ambulance_id"] = action.ambulance_id
            call["spawned_patient_id"] = None
            category = call["category"]
            cat_val = category.value if hasattr(category, "value") else category
            outcome = {
                "call_id": action.call_id,
                "decision": "als" if decision == "dispatch_als" else "bls",
                "category": cat_val,
                "true_condition": None,
                "als_needed": None,
                "revealed_at_step": None,
            }
            self._dispatch_outcomes_history.append(outcome)
            self._pending_calls = [c for c in self._pending_calls if c["call_id"] != action.call_id]
            if action.call_id in self._pending_call_countdown:
                del self._pending_call_countdown[action.call_id]
            self._alerts.append(f"{'ALS' if decision == 'dispatch_als' else 'BLS'} dispatched to {action.call_id}")
        elif decision == "self_transport":
            category = call["category"]
            cat_val = category.value if hasattr(category, "value") else category
            self._dispatch_outcomes_history.append({
                "call_id": call["call_id"],
                "decision": "self_transport",
                "category": cat_val,
                "true_condition": None,
                "als_needed": None,
                "revealed_at_step": None,
            })
            self._pending_calls = [c for c in self._pending_calls if c["call_id"] != action.call_id]
            self._pending_call_countdown.pop(action.call_id, None)
            self._alerts.append(f"Self-transport advised for {action.call_id}")
        elif decision == "callback":
            call["time_waiting"] = 0
            self._pending_call_countdown[action.call_id] = 15
            self._alerts.append(f"Callback scheduled for {action.call_id}")
        elif decision == "no_dispatch":
            category = call["category"]
            cat_val = category.value if hasattr(category, "value") else category
            self._dispatch_outcomes_history.append({
                "call_id": call["call_id"],
                "decision": "no_dispatch",
                "category": cat_val,
                "true_condition": None,
                "als_needed": None,
                "revealed_at_step": None,
            })
            self._pending_calls = [c for c in self._pending_calls if c["call_id"] != action.call_id]
            self._pending_call_countdown.pop(action.call_id, None)
            self._alerts.append(f"No dispatch for {action.call_id}")

    def _do_dispatch_als(self, action) -> None:
        from .models.actions import TriageCall
        self._do_triage_call(TriageCall(
            call_id=action.call_id, decision="dispatch_als", ambulance_id=action.ambulance_id
        ))

    def _do_dispatch_bls(self, action) -> None:
        from .models.actions import TriageCall
        self._do_triage_call(TriageCall(
            call_id=action.call_id, decision="dispatch_bls", ambulance_id=action.ambulance_id
        ))

    # =========================================================================
    # Termination & Reward
    # =========================================================================

    def _check_done(self) -> bool:
        """Check if episode should terminate."""
        if self._state.step_count >= self._state.max_steps:
            return True
        non_terminal = [p for p in self._patients if p.status not in ("treated", "deceased")]
        if len(non_terminal) == 0:
            use_call_queue = TASK_CONFIG.get(self._state.task_id, {}).get("use_call_queue", False)
            has_pending = (
                self._pending_calls
                or self._pending_call_countdown
                or any(not entry.get("spawned", False) for entry in self._pending_call_to_patient.values())
            )
            if use_call_queue and (not self._phase2_has_activity or has_pending):
                return False
            self._state.all_patients_terminal = True
            return True
        alive = [p for p in self._patients if p.status not in ("treated", "deceased")]
        if alive and all(p.vitals_score <= 0.0 for p in alive):
            self._state.all_patients_terminal = True
            return True
        return False

    def _score_hospital_routing(self, patient, hospital_id: str) -> float:
        """
        Immediate routing quality signal fired when AssignHospital is executed.

        Returns a signed float that is stored in the episode log and consumed
        by _compute_step_reward in the same step.
        """
        cfg = REWARD_CONFIG
        urgency = cfg["urgency_weight"].get(patient.condition, 1.0)
        reward = 0.0

        can_treat = self._hospital_system.can_treat(hospital_id, patient.condition)
        hosp_obj = self._hospital_system.get(hospital_id)
        on_diversion = getattr(hosp_obj, "on_diversion", False) if hosp_obj else False

        if not can_treat:
            reward += cfg["wrong_hospital_penalty"] * urgency
        else:
            reward += cfg["correct_hospital_bonus"] * urgency

        if on_diversion:
            reward += cfg["diversion_penalty"] * urgency

        # Nearest-capable bonus: reward choosing the closest appropriate hospital
        if can_treat and not on_diversion:
            patient_node = getattr(patient, "location_node", None)
            if patient_node:
                best_time = float("inf")
                chosen_time = float("inf")
                for h_id, h in self._hospital_system.all().items():
                    if self._hospital_system.can_treat(h_id, patient.condition) and not h.on_diversion:
                        t = self._road_network.travel_time(patient_node, h.node_id)
                        if t < best_time:
                            best_time = t
                        if h_id == hospital_id:
                            chosen_time = t
                if chosen_time <= best_time + 2:
                    reward += cfg["nearest_capable_bonus"] * urgency

        return reward

    def _compute_step_reward(self) -> float:  # noqa: C901
        """
        Dense, multi-signal reward function.

        Signal hierarchy (high → low impact per step):
          1. Patient outcome milestones  — large one-shot on status transitions
          2. Time-bonus shaping          — proportional credit for treating quickly
          3. Per-step urgency penalties  — bleeds reward while patient waits
          4. Vitals delta                — fine-grained health tracking
          5. Hospital routing            — instant feedback on AssignHospital quality
          6. ALS / BLS triage            — scored when ground truth is revealed
          7. Proactive preparation       — OR prep + specialist page that pays off
          8. Blood management            — emergency release for critical patients
          9. Mutual aid outcomes         — useful vs wasted
         10. Wasted actions & spam       — mild deterrent
        """
        cfg = REWARD_CONFIG
        reward = 0.0

        if self._state.all_patients_terminal:
            return 0.0

        step = self._state.step_count
        step_logs = [e for e in self._episode_log if e.get("step") == step]

        # ── 1, 2, 3, 4: Per-patient signals ──────────────────────────────────
        for patient in self._patient_manager.patients:
            pid = patient.id
            condition = patient.condition
            urgency = cfg["urgency_weight"].get(condition, 1.0)
            curr_status = patient.status
            prev_status = self._prev_patient_status.get(pid, curr_status)
            milestone = cfg["milestone"]

            # 1. Status transition milestones (urgency-scaled)
            if prev_status == "waiting" and curr_status == "dispatched":
                reward += milestone["dispatched"] * urgency

            if prev_status == "dispatched" and curr_status == "transporting":
                reward += milestone["transporting"] * urgency

            if prev_status == "transporting" and curr_status == "in_treatment":
                reward += milestone["in_treatment"] * urgency

            if prev_status not in ("treated", "deceased") and curr_status == "treated":
                target = PATIENT_TARGET_TIMES.get(condition, 60)
                actual = getattr(patient, "treatment_complete_time", None)
                if actual is not None:
                    # 2. Time-bonus shaping
                    ratio = actual / max(target, 1)
                    if ratio <= 1.0:
                        # On time or early: base bonus + speed bonus (max when ratio=0)
                        speed_bonus = cfg["time_bonus_max"] * (1.0 - ratio)
                        reward += (milestone["treated_on_time"] + speed_bonus) * urgency
                    else:
                        # Late: reward fades as lateness grows, hits 0 at 2× overdue
                        lateness_factor = max(0.0, 1.0 - (ratio - 1.0))
                        reward += milestone["treated_late"] * lateness_factor * urgency
                else:
                    reward += milestone["treated_on_time"] * urgency

            if prev_status != "deceased" and curr_status == "deceased":
                reward += milestone["deceased"] * urgency  # large negative

            # 3. Per-step holding penalties — the clock is always ticking
            if curr_status in ("waiting", "deteriorating", "critical"):
                reward += cfg["waiting_penalty_per_step"] * urgency
            elif curr_status == "dispatched":
                reward += cfg["dispatched_penalty_per_step"] * urgency
            elif curr_status == "transporting":
                reward += cfg["transporting_penalty_per_step"] * urgency

            # 4. Vitals delta (urgency-scaled)
            prev_v = self._prev_vitals.get(pid, patient.vitals_score)
            delta = patient.vitals_score - prev_v
            reward += delta * cfg["vitals_delta_weight"] * urgency

        # ── 5. Hospital routing (pulled from log entries written this step) ───
        for entry in step_logs:
            if entry.get("event") == "action_assign_hospital":
                reward += entry.get("_routing_reward", 0.0)

        # ── 6. ALS / BLS triage scoring ──────────────────────────────────────
        for outcome in self._dispatch_outcomes_history:
            if outcome.get("scored"):
                continue
            if outcome.get("true_condition") is None:
                continue
            if outcome.get("revealed_at_step") != step:
                continue

            als_needed = outcome["als_needed"]
            decision = outcome["decision"]

            if decision == "als":
                reward += cfg["als_correct_dispatch"] if als_needed else cfg["als_overtriage_penalty"]
            elif decision == "bls":
                reward += cfg["bls_correct_dispatch"] if not als_needed else cfg["als_undertriage_penalty"]
            elif decision in ("self_transport", "no_dispatch"):
                reward += cfg["no_dispatch_unnecessary"] if not als_needed else cfg["no_dispatch_urgent_penalty"]

            outcome["scored"] = True

        # ── 7. Proactive preparation ──────────────────────────────────────────
        for entry in step_logs:
            if entry.get("event") == "action_prepare_or" and entry.get("result") == "success":
                hosp_id = entry.get("hospital_id")
                if any(
                    p.status in ("dispatched", "transporting", "in_treatment")
                    and getattr(p, "assigned_hospital", None) == hosp_id
                    for p in self._patients
                ):
                    reward += cfg["or_prep_used_bonus"]

            if entry.get("event") == "action_page_specialist" and entry.get("result") == "success":
                hosp_id = entry.get("hospital_id")
                spec_type = entry.get("specialist_type")
                if any(
                    p.status in ("dispatched", "transporting")
                    and getattr(p, "assigned_hospital", None) == hosp_id
                    and SPECIALIST_BY_CONDITION.get(p.condition) == spec_type
                    for p in self._patients
                ):
                    reward += cfg["specialist_paged_used_bonus"]

        # ── 8. Blood management ───────────────────────────────────────────────
        for entry in step_logs:
            if entry.get("event") == "action_allocate_blood" and entry.get("result") == "success":
                pid = entry.get("patient_id")
                if pid:
                    patient = self._patient_manager.get(pid)
                    if patient and patient.vitals_score < 0.4:
                        reward += cfg["emergency_blood_needed_bonus"]

        # ── 9. Mutual aid outcomes ────────────────────────────────────────────
        for entry in step_logs:
            if entry.get("event") == "mutual_aid_arrived":
                reward += (
                    cfg["mutual_aid_saves_bonus"]
                    if entry.get("had_patient")
                    else cfg["mutual_aid_wasted_penalty"]
                )

        # ── 10. Wasted actions & spam ─────────────────────────────────────────
        for entry in step_logs:
            if entry.get("result") == "wasted":
                reward += cfg["wasted_action_penalty"]

        for action_type, count in self._action_counts.items():
            if count > cfg["spam_threshold"]:
                excess = count - cfg["spam_threshold"]
                reward += cfg["spam_penalty_per_repeat"] * excess

        # ── Bootstrap new patients (no fake milestone reward) ─────────────────
        for patient in self._patient_manager.patients:
            if patient.id not in self._prev_vitals:
                self._prev_vitals[patient.id] = patient.vitals_score
                self._prev_patient_status[patient.id] = patient.status

        # ── Snapshot for next step ────────────────────────────────────────────
        self._prev_vitals = {p.id: p.vitals_score for p in self._patient_manager.patients}
        self._prev_patient_status = {p.id: p.status for p in self._patient_manager.patients}

        # ── Clamp ─────────────────────────────────────────────────────────────
        lo, hi = cfg["step_reward_clamp"]
        return max(lo, min(hi, reward))

    def _compute_time_score_preview(self) -> float:
        """
        Informational time-score estimate exposed in the observation [0, 1].
        NOT fed back into step reward to avoid double-counting.
        """
        scores = []
        for p in self._patients:
            target = PATIENT_TARGET_TIMES.get(p.condition, 60)
            if p.status == "treated" and p.treatment_complete_time is not None:
                score = max(0.0, min(1.0, 1.0 - (p.treatment_complete_time - target) / max(target, 1)))
            elif p.status == "deceased":
                score = 0.0
            else:
                elapsed = self._state.step_count - p.onset_step
                projected_total = elapsed + 3 + 5 + 10  # optimistic pipeline
                score = max(0.0, min(1.0, 1.0 - (projected_total - target) / max(target, 1)))
            scores.append(score)
        return sum(scores) / len(scores) if scores else 1.0

    # =========================================================================
    # Observation  (unchanged from original)
    # =========================================================================

    def _build_observation(self, done: bool = False) -> CodeRedObservation:
        self._state.cum_reward = round(self._state.cum_reward, 4)

        def _map_patient_status(s):
            mapping = {
                "waiting": PatientStatus.WAITING,
                "deteriorating": PatientStatus.DETERIORATING,
                "critical": PatientStatus.CRITICAL,
                "dispatched": PatientStatus.DISPATCHED,
                "transporting": PatientStatus.TRANSPORTING,
                "in_treatment": PatientStatus.TREATING,
                "treating": PatientStatus.TREATING,
                "treated": PatientStatus.TREATED,
                "deceased": PatientStatus.DECEASED,
            }
            return mapping.get(s, PatientStatus.WAITING)

        patients = [
            Patient(
                patient_id=p.id,
                condition=PatientCondition(p.condition),
                tier=self._condition_to_tier(p.condition),
                location_node=p.location_node,
                time_since_onset=self._state.step_count - p.onset_step,
                assigned_ambulance=next(
                    (amb_id for amb_id, amb in self._ambulance_manager.all().items()
                     if getattr(amb, "patient_id", None) == p.id),
                    None,
                ),
                assigned_hospital=p.assigned_hospital,
                status=_map_patient_status(p.status),
                vitals_score=p.vitals_score,
                blood_type=p.blood_type,
                treatment_start_time=p.arrival_hospital_step,
                treatment_complete_time=p.treatment_complete_time,
                outcome=p.outcome,
                is_secondary=getattr(p, "is_secondary", False),
                icu_status=p.icu_status,
                dispatch_call_id=getattr(p, "dispatch_call_id", None),
                cascade_trigger_reason=getattr(p, "cascade_trigger_reason", None),
                observed_condition=getattr(p, "observed_condition", None),
            )
            for p in self._patients
        ]

        amb_states = []
        for amb_id, amb in self._ambulance_manager.all().items():
            status_map = {
                "available": AmbulanceStatus.AVAILABLE,
                "en_route": AmbulanceStatus.DISPATCHED,
                "on_scene": AmbulanceStatus.DISPATCHED,
                "returning": AmbulanceStatus.RETURNING,
                "off_duty": AmbulanceStatus.OFFLINE,
            }
            amb_states.append(AmbulanceState(
                id=amb.id,
                node_id=amb.target_node or amb.base_node or "",
                equipment=AmbulanceEquipment(amb.equipment),
                status=status_map.get(amb.status, AmbulanceStatus.AVAILABLE),
                assigned_patient=amb.patient_id,
                route=amb.route,
                eta_minutes=amb.eta_minutes,
                destination_type="patient" if amb.target_node else None,
            ))

        hosp_states = []
        for hosp_id, hosp in self._hospital_system.all().items():
            ors = [
                OperatingRoom(
                    index=o.index,
                    status=ORStatus(o.status),
                    procedure_type=o.procedure_type,
                    minutes_remaining=o.minutes_remaining,
                    patient_id=o.patient_id,
                )
                for o in hosp.operating_rooms
            ]
            specialists = {
                role: SpecialistStatus(
                    available=spec.available,
                    total=spec.total,
                    status=spec.status,
                    minutes_until_available=spec.minutes_until_available,
                )
                for role, spec in hosp.specialists.items()
            }
            hosp_states.append(HospitalState(
                id=hosp.id,
                node_id=hosp.node_id,
                capabilities=hosp.capabilities,
                specialists=specialists,
                operating_rooms=ors,
                icu_beds=hosp.icu_beds,
                blood_stock=hosp.blood_stock,
                on_diversion=hosp.on_diversion,
                or_prep_countdowns=dict(hosp.or_prep_countdowns),
            ))

        blood_states = []
        for bb_id, bb in self._blood_bank.all().items():
            blood_states.append(BloodBankState(
                hospital_id=bb.hospital_id,
                stocks=dict(bb.stocks),
                crossmatch_queue=list(bb.crossmatch_queue),
            ))

        road_state = RoadNetworkState()
        for key, edge in self._road_network.edges.items():
            road_state.edges[key] = EdgeState(
                from_node=edge.from_node,
                to_node=edge.to_node,
                base_time=edge.base_time,
                congestion_multiplier=edge.congestion_multiplier,
                disrupted=edge.disrupted,
                disruption_type=edge.disruption_type,
            )

        from server.models.entities import DispatchCall as DispatchCallModel, DispatchOutcome as DispatchOutcomeModel
        pending_calls_obs = [
            DispatchCallModel(
                call_id=c["call_id"],
                category=c["category"],
                location_node=c["location_node"],
                time_waiting=c["time_waiting"],
                estimated_severity=c["estimated_severity"],
                spawned_patient_id=c["spawned_patient_id"],
            )
            for c in self._pending_calls
        ]
        recent_outcomes = [
            DispatchOutcomeModel(
                call_id=o["call_id"],
                decision=o["decision"],
                category=o["category"],
                true_condition=o["true_condition"],
                als_needed=o["als_needed"] if o["als_needed"] is not None else False,
                revealed_at_step=o["revealed_at_step"],
            )
            for o in self._dispatch_outcomes_history[-5:]
        ]

        active_patients = [p for p in self._patients if p.status not in ("treated", "deceased")]
        overcrowding_modifier = self._cascade_engine.check_overcrowding(len(active_patients))

        return CodeRedObservation(
            done=done,
            step=self._state.step_count + 1,
            patients=patients,
            ambulances=amb_states,
            hospitals=hosp_states,
            blood_banks=blood_states,
            road_network=road_state,
            alerts=list(self._alerts),
            mutual_aid_remaining=self._state.mutual_aid_available,
            time_score_preview=round(self._compute_time_score_preview(), 4),
            vitals_score_preview=round(
                sum(p.vitals_score for p in self._patients if p.status not in ("treated", "deceased"))
                / max(1, sum(1 for p in self._patients if p.status not in ("treated", "deceased"))),
                4,
            ),
            patients_remaining=len([
                p for p in self._patients if p.status not in ("treated", "deceased")
            ]),
            pending_calls=pending_calls_obs,
            recent_dispatch_outcomes=recent_outcomes,
            overcrowding_modifier=overcrowding_modifier,
        )

    # =========================================================================
    # Helpers  (unchanged from original)
    # =========================================================================

    def _condition_to_tier(self, condition: str) -> PatientTier:
        mapping = {
            "cardiac": PatientTier.CRITICAL,
            "stroke":  PatientTier.CRITICAL,
            "trauma":  PatientTier.HIGH,
            "general": PatientTier.MEDIUM,
        }
        return mapping.get(condition, PatientTier.MEDIUM)

    def _cascade_callback(self, event_type: str, **kwargs) -> None:
        if event_type == "spawn_secondary":
            condition = kwargs["condition"]
            reason = kwargs["reason"]
            triggered_at_step = kwargs["triggered_at_step"]
            node = kwargs.get("spawn_node")
            if node is None:
                from server.subsystems.patient_manager import PatientManager
                node = self._rng.choice(PatientManager._SPAWN_NODES)
            patient = self._patient_manager.spawn_secondary(
                condition=condition, triggered_by=None, reason=reason,
                onset_step=triggered_at_step, spawn_node=node,
            )
            self._patients = self._patient_manager.patients
            self._alerts.append(f"CASCADE: Secondary {condition} patient {patient.id} spawned (reason: {reason})")
            self._episode_log.append({
                "step": triggered_at_step, "patient_id": patient.id,
                "event": "secondary_patient_spawned", "condition": condition,
                "is_secondary": True, "reason": reason,
                "target_time": PATIENT_TARGET_TIMES.get(condition, 60),
            })
        elif event_type == "overcrowding_started":
            count = kwargs["active_patient_count"]
            self._alerts.append(f"OVERCROWDING: ED at {count} active patients")
            self._episode_log.append({
                "step": self._state.step_count, "event": "overcrowding_started",
                "active_patient_count": count,
            })
        elif event_type == "news_cycle":
            msg = kwargs["message"]
            steps = kwargs["steps"]
            self._alerts.append(f"NEWS: {msg}")
            self._episode_log.append({
                "step": self._state.step_count, "event": "news_cycle", "message": msg, "steps": steps,
            })

    def _spawn_dispatch_call(self) -> None:
        from server.models.entities import DispatchCategory
        from server.subsystems.constants import DISPATCH_CATEGORY_MAP
        if len(self._pending_calls) >= 5:
            return
        call_id = f"CALL_{self._state.step_count:04d}"
        categories = list(DISPATCH_CATEGORY_MAP.keys())
        weights = [0.20, 0.15, 0.25, 0.20, 0.20]
        category = self._rng.choices(categories, weights=weights)[0]
        from server.subsystems.patient_manager import PatientManager
        location = self._rng.choice(PatientManager._SPAWN_NODES)
        call = {
            "call_id": call_id,
            "category": category,
            "location_node": location,
            "time_waiting": 0,
            "estimated_severity": 0.1,
            "spawned_patient_id": None,
        }
        self._pending_calls.append(call)
        self._pending_call_countdown[call_id] = 20
        self._phase2_has_activity = True
        self._alerts.append(f"CALL: {category.value if hasattr(category, 'value') else category} call {call_id} at {location}")
        self._episode_log.append({
            "step": self._state.step_count,
            "call_id": call_id,
            "event": "call_received",
            "category": category.value if hasattr(category, "value") else category,
            "location": location,
        })

    def _force_spawn_at_scene(self, call_id: str, amb_id: str) -> None:
        from server.subsystems.constants import DISPATCH_CATEGORY_MAP
        from server.subsystems.patient_manager import PatientManager

        call = next((c for c in self._pending_calls if c["call_id"] == call_id), None)
        pending_entry = self._pending_call_to_patient.get(call_id, {})

        if call:
            location = call["location_node"]
            category = call["category"]
            call["assigned_ambulance_id"] = amb_id
        elif pending_entry:
            location = pending_entry.get("location_node", "RAJIV_CHOWK")
            outcome = next((o for o in self._dispatch_outcomes_history if o["call_id"] == call_id), None)
            category = outcome["category"] if outcome else list(DISPATCH_CATEGORY_MAP.keys())[0]
        else:
            return

        spawn_nodes = PatientManager._SPAWN_NODES
        loc = location if location in [n["id"] for n in CITY_NODES] else self._rng.choice(spawn_nodes)
        condition_choices = DISPATCH_CATEGORY_MAP[category]
        conditions, probs = zip(*condition_choices)
        true_condition = self._rng.choices(list(conditions), weights=list(probs))[0]
        patient = self._patient_manager.spawn_secondary(
            condition=true_condition, triggered_by=None, reason="forced_spawn",
            onset_step=self._state.step_count, spawn_node=loc,
        )
        patient.dispatch_call_id = call_id
        patient.observed_condition = true_condition
        patient.assigned_ambulance = amb_id
        self._phase2_has_activity = True

        amb = self._ambulance_manager.get(amb_id)
        if amb:
            amb.patient_id = patient.id

        if not pending_entry.get("spawned"):
            self._pending_call_to_patient[call_id] = {
                "ambulance_id": amb_id,
                "location_node": location or loc,
                "spawned": True,
                "spawned_patient_id": patient.id,
            }

        self._pending_calls = [c for c in self._pending_calls if c["call_id"] != call_id]
        self._pending_call_countdown.pop(call_id, None)
        self._ambulance_pending_patient.pop(amb_id, None)

        self._patients = self._patient_manager.patients
        cat_val = category.value if hasattr(category, "value") else category
        self._alerts.append(f"ON-SCENE: {cat_val} call {call_id} force-spawned for {amb_id}")
        self._episode_log.append({
            "step": self._state.step_count,
            "patient_id": patient.id,
            "event": "patient_created",
            "condition": true_condition,
            "is_secondary": False,
            "call_id": call_id,
            "target_time": PATIENT_TARGET_TIMES.get(true_condition, 60),
            "force_spawn": True,
        })

        outcome = {
            "call_id": call_id,
            "decision": "no_dispatch",
            "category": cat_val,
            "true_condition": true_condition,
            "als_needed": true_condition in ("cardiac", "stroke", "trauma"),
            "revealed_at_step": self._state.step_count,
        }
        found = False
        for entry in self._dispatch_outcomes_history:
            if entry["call_id"] == call_id and entry["true_condition"] is None:
                entry["true_condition"] = true_condition
                entry["als_needed"] = outcome["als_needed"]
                entry["revealed_at_step"] = self._state.step_count
                found = True
                break
        if not found:
            self._dispatch_outcomes_history.append(outcome)

    def _spawn_patient_from_call(self, call_id: str) -> None:
        call = next((c for c in self._pending_calls if c["call_id"] == call_id), None)
        if call is None:
            entry = self._pending_call_to_patient.get(call_id, {})
            amb_id = entry.get("ambulance_id")
            loc = entry.get("location_node", "")
            if loc and amb_id:
                from server.subsystems.constants import DISPATCH_CATEGORY_MAP
                outcome = next((o for o in self._dispatch_outcomes_history if o["call_id"] == call_id), None)
                categories = list(DISPATCH_CATEGORY_MAP.keys())
                category = outcome["category"] if outcome else categories[0]
                conditions, probs = zip(*DISPATCH_CATEGORY_MAP[category])
                true_condition = self._rng.choices(list(conditions), weights=list(probs))[0]
                patient = self._patient_manager.spawn_secondary(
                    condition=true_condition, triggered_by=None, reason="forced_spawn",
                    onset_step=self._state.step_count, spawn_node=loc,
                )
                patient.dispatch_call_id = call_id
                patient.observed_condition = true_condition
                self._phase2_has_activity = True
                patient.assigned_ambulance = amb_id
                amb = self._ambulance_manager.get(amb_id)
                if amb:
                    amb.patient_id = patient.id
                entry["spawned"] = True
                entry["spawned_patient_id"] = patient.id
                self._patients = self._patient_manager.patients
            return

        self._pending_calls = [c for c in self._pending_calls if c["call_id"] != call_id]
        self._pending_call_countdown.pop(call_id, None)

        from server.subsystems.constants import DISPATCH_CATEGORY_MAP
        category = call["category"]
        condition_choices = DISPATCH_CATEGORY_MAP[category]
        conditions, probs = zip(*condition_choices)
        true_condition = self._rng.choices(list(conditions), weights=list(probs))[0]
        patient = self._patient_manager.spawn_secondary(
            condition=true_condition, triggered_by=None, reason="forced_spawn",
            onset_step=self._state.step_count, spawn_node=call["location_node"],
        )
        patient.dispatch_call_id = call_id
        patient.observed_condition = true_condition
        self._phase2_has_activity = True

        call_assigned_amb = call.get("assigned_ambulance_id")
        if call_assigned_amb:
            patient.assigned_ambulance = call_assigned_amb
            amb = self._ambulance_manager.get(call_assigned_amb)
            if amb:
                amb.patient_id = patient.id

        pending_entry = self._pending_call_to_patient.get(call_id)
        if pending_entry:
            pending_entry["spawned"] = True
            pending_entry["spawned_patient_id"] = patient.id
            amb_id = pending_entry.get("ambulance_id")
            if amb_id and self._ambulance_pending_patient.get(amb_id, {}).get("call_id") == call_id:
                del self._ambulance_pending_patient[amb_id]

        call["spawned_patient_id"] = patient.id
        self._patients = self._patient_manager.patients
        cat_val = category.value if hasattr(category, "value") else category
        self._alerts.append(f"ON-SCENE: {cat_val} call {call_id} force-spawned")
        self._episode_log.append({
            "step": self._state.step_count,
            "patient_id": patient.id,
            "event": "patient_created",
            "condition": true_condition,
            "is_secondary": False,
            "call_id": call_id,
            "target_time": PATIENT_TARGET_TIMES.get(true_condition, 60),
            "force_spawn": True,
        })

        outcome = {
            "call_id": call_id,
            "decision": "no_dispatch",
            "category": cat_val,
            "true_condition": true_condition,
            "als_needed": true_condition in ("cardiac", "stroke", "trauma"),
            "revealed_at_step": self._state.step_count,
        }
        found = False
        for entry in self._dispatch_outcomes_history:
            if entry["call_id"] == call_id and entry["true_condition"] is None:
                entry["true_condition"] = true_condition
                entry["als_needed"] = outcome["als_needed"]
                entry["revealed_at_step"] = self._state.step_count
                found = True
                break
        if not found:
            self._dispatch_outcomes_history.append(outcome)