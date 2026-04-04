"""CodeRedEnvironment — Core OpenEnv environment with wired subsystems."""

from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

from .models import (
    CodeRedAction,
    CodeRedObservation,
    CodeRedState,
)
from .models.entities import (
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
from .subsystems.constants import (
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


class CodeRedEnvironment(Environment):
    """OpenEnv environment for emergency medical coordination."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        from .subsystems.road_network import RoadNetwork
        from .subsystems.hospital_system import HospitalSystem
        from .subsystems.blood_bank import BloodBankSystem
        from .subsystems.disruption_engine import DisruptionEngine
        from .subsystems.cascade_engine import CascadeEngine
        from .subsystems.patient_manager import PatientManager
        from .subsystems.ambulance_manager import AmbulanceManager
        from .subsystems.mutual_aid import MutualAidManager

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
        # Track active surgeries for treatment_complete detection
        self._active_surgeries: Dict[str, Dict] = {}  # patient_id -> {hosp_id, or_index, start_step}
        # Prev snapshots for dense reward computation
        self._prev_vitals: Dict[str, float] = {}
        self._prev_patient_status: Dict[str, str] = {}
        self._pending_calls: List = []
        self._pending_call_countdown: Dict[str, int] = {}
        self._dispatch_outcomes_history: List = []
        self._cascade_engine = CascadeEngine()
        self._step_count_since_last_call: int = 0

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

        # Initialize all subsystems
        from .subsystems.road_network import RoadNetwork
        from .subsystems.hospital_system import HospitalSystem
        from .subsystems.blood_bank import BloodBankSystem
        from .subsystems.disruption_engine import DisruptionEngine
        from .subsystems.cascade_engine import CascadeEngine
        from .subsystems.patient_manager import PatientManager
        from .subsystems.ambulance_manager import AmbulanceManager
        from .subsystems.mutual_aid import MutualAidManager

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
        self._dispatch_outcomes_history = []
        self._step_count_since_last_call = 0

        # Log patient creation
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

        # Snapshot vitals and status for reward computation
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

        # Advance all systems by 1 minute
        self._advance_time()

        # Execute action(s)
        self._execute_action(action)

        # Check termination
        done = self._check_done()

        # Compute reward
        reward = self._compute_step_reward()

        self._state.cum_reward += reward

        return self._build_observation()

    @property
    def state(self) -> CodeRedState:
        return self._state

    def get_episode_log(self) -> List[dict]:
        """Return the full episode log for grading."""
        return self._episode_log.copy()

    # =========================================================================
    # Time Advancement
    # =========================================================================

    def _advance_time(self) -> None:
        """Advance all systems by 1 minute."""
        # Patient deterioration + outcome check (with overcrowding modifier)
        active_patients = [p for p in self._patients if p.status not in ("treated", "deceased")]
        overcrowding_modifier = self._cascade_engine.check_overcrowding(len(active_patients))
        self._patient_manager.tick(
            self._patient_manager.get_onset_steps(),
            self._state.step_count,
            overcrowding_modifier=overcrowding_modifier,
        )
        # Check for patient deaths/outcomes
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

        # Ambulances
        self._ambulance_manager.tick()
        for amb_id, amb in self._ambulance_manager.all().items():
            if amb.arrived_with_patient:
                # Task 16: scene time expired — deliver patient to hospital
                amb.arrived_with_patient = False
                if amb.patient_id:
                    patient = next((pt for pt in self._patients if pt.id == amb.patient_id), None)
                    if patient and patient.status == "transporting":
                        self._do_treatment_arrival(patient, amb_id)
            elif amb.status == "on_scene" and amb.eta_minutes == 0:
                # Ambulance arrived at destination
                if amb.patient_id:
                    patient = next((pt for pt in self._patients if pt.id == amb.patient_id), None)
                    if patient and patient.status == "dispatched":
                        # Ambulance just arrived at patient — start scene time countdown
                        patient.status = "transporting"

        # Mutual aid: delegate to subsystem
        def _ma_arrival_callback(ma_id: str, patient_id: str) -> None:
            """Called by MutualAidManager when a MA arrives with a patient."""
            patient = self._patient_manager.get(patient_id)
            if patient and patient.status == "transporting":
                self._do_treatment_arrival(patient, ma_id)

        # Advance MA state and process arrivals
        ma_arrivals = self._mutual_aid_manager.tick(
            step_count=self._state.step_count,
            patient_manager=self._patient_manager,
            hospital_system=self._hospital_system,
            arrival_callback=_ma_arrival_callback,
        )
        for event in ma_arrivals:
            self._episode_log.append({
                "step": self._state.step_count,
                "event": "mutual_aid_arrived",
                "ambulance_id": event["ambulance_id"],
                "patient_id": event.get("patient_id"),
                "actual_arrival_step": event.get("actual_arrival_step", self._state.step_count),
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

        # Hospitals
        # Capture active surgery patient IDs BEFORE tick to detect completions
        pre_tick_surgery_patients = set(self._active_surgeries.keys())

        self._hospital_system.tick()

        # After tick: detect completed surgeries (OR patient_id cleared)
        for patient_id in list(pre_tick_surgery_patients):
            if patient_id not in self._active_surgeries:
                continue  # already handled
            surgery_info = self._active_surgeries[patient_id]
            hosp = self._hospital_system.get(surgery_info["hospital_id"])
            if hosp:
                or_obj = next(
                    (o for o in hosp.operating_rooms if o.index == surgery_info["or_index"]),
                    None,
                )
                if or_obj and or_obj.status == "idle" and or_obj.patient_id is None:
                    # Surgery just completed — roll mortality based on hospital quality
                    patient = self._patient_manager.get(patient_id)
                    if patient and patient.status == "in_treatment":
                        effective_time = self._state.step_count - surgery_info["start_step"]
                        target_time = PATIENT_TARGET_TIMES.get(patient.condition, 60)

                        # Hospital quality variance (Task 12): roll mortality
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

                        # Phase 2: Cascade engine trigger on outcome
                        cascade_enabled = TASK_CONFIG.get(self._state.task_id, {}).get("cascade_enabled", False)
                        if cascade_enabled:
                            outcome_str = patient.outcome
                            self._cascade_engine.on_outcome(
                                patient_id, patient.condition, outcome_str, self._state.step_count
                            )

                        # Release ICU bed on treatment completion (Task 13)
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

        # After hospital tick: check if any waiting patient can now start surgery
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

        # Blood bank
        self._blood_bank.tick()
        completed = self._blood_bank.flush_completed_crossmatches()
        for entry in completed:
            self._alerts.append(
                f"Crossmatch complete: {entry['units']} units "
                f"{entry['blood_type']} reserved for {entry['patient_id']}"
            )

        # Road disruptions tick
        self._road_network.tick()
        self._road_network_tick_disruptions()

        # Pending blood type queries (from QueryBloodType action)
        for patient_id in list(self._pending_blood_queries.keys()):
            self._pending_blood_queries[patient_id] -= 1
            if self._pending_blood_queries[patient_id] <= 0:
                del self._pending_blood_queries[patient_id]
                self._reveal_blood_type(patient_id)

        # Pending OR queries
        for key in list(self._pending_or_queries.keys()):
            self._pending_or_queries[key] -= 1
            if self._pending_or_queries[key] <= 0:
                del self._pending_or_queries[key]

        # Roll new disruptions
        events = self._disruption_engine.roll_disruptions(
            step=self._state.step_count,
            road_network=self._road_network,
            hospital_system=self._hospital_system,
        )
        for event in events:
            if event["target_type"] == "surge":
                # Spawn 1-2 secondary patients on surge
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

        # Cascade engine tick (Phase 2)
        self._cascade_engine.tick()

        # Call queue spawning (Phase 2)
        use_call_queue = TASK_CONFIG.get(self._state.task_id, {}).get("use_call_queue", False)
        if use_call_queue:
            self._step_count_since_last_call += 1
            if self._step_count_since_last_call >= 8:
                self._spawn_dispatch_call()
                self._step_count_since_last_call = 0
            # Tick down pending call countdowns
            for call_id in list(self._pending_call_countdown.keys()):
                self._pending_call_countdown[call_id] -= 1
                if self._pending_call_countdown[call_id] <= 0:
                    self._spawn_patient_from_call(call_id)

    def _road_network_tick_disruptions(self) -> None:
        """Process road network disruption events — check for newly expired disruptions."""
        active = self._road_network.get_active_disruptions()
        for disp in active:
            if disp["remaining_steps"] <= 0:
                self._road_network.clear_disruption(disp["from_node"], disp["to_node"])
                self._alerts.append(f"Road cleared: {disp['from_node']} <-> {disp['to_node']}")

    def _reveal_blood_type(self, patient_id: str) -> None:
        """Reveal a patient's blood type."""
        patient = self._patient_manager.get(patient_id)
        if patient:
            patient.blood_type = self._rng.choice(BLOOD_TYPES)
            self._alerts.append(f"Blood type revealed for {patient_id}: {patient.blood_type}")

    def _do_treatment_arrival(self, patient, ambulance_id: str) -> None:
        """Handle patient arriving at hospital for treatment."""
        if patient.assigned_hospital:
            hosp = self._hospital_system.get(patient.assigned_hospital)
            if hosp:
                # Determine procedure type based on condition
                procedure_type = PROCEDURE_BY_CONDITION.get(patient.condition, "general")

                # Check OR and specialist availability
                idle_or = self._hospital_system.get_idle_or(patient.assigned_hospital)
                # Determine required specialist type
                specialist_type = SPECIALIST_BY_CONDITION.get(patient.condition, "general_surgeon")
                specialist_available = self._hospital_system.is_specialist_available(
                    patient.assigned_hospital, specialist_type
                )

                # Log patient_arrived_hospital event
                self._episode_log.append({
                    "step": self._state.step_count,
                    "patient_id": patient.id,
                    "event": "patient_arrived_hospital",
                    "hospital_id": patient.assigned_hospital,
                    "or_ready": idle_or is not None,
                    "specialist_available": specialist_available,
                })

                # ICU bed constraint (Task 13): consume bed only if hospital can treat this condition
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

                # Start surgery/treatment if OR available
                if idle_or:
                    # Start surgery
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
                        # Track active surgery for treatment_complete detection
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

                        # Mark ambulance as available
                        self._ambulance_manager.mark_available(ambulance_id)
                else:
                    # No OR available, patient waits in treating status
                    patient.status = "in_treatment"

    # =========================================================================
    # Action Execution
    # =========================================================================

    def _execute_action(self, action: CodeRedAction) -> None:
        """Dispatch action to the appropriate handler."""
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
        from .models.actions import DispatchAmbulance
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

        # If patient at target node, assign
        waiting_patient = next(
            (p for p in self._patients
             if p.status == "waiting"
             and p.location_node == action.target_node),
            None,
        )
        if waiting_patient:
            waiting_patient.assigned_ambulance = action.ambulance_id
            waiting_patient.status = "dispatched"
            # Update ambulance with patient assignment
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
            # Dispatched but no patient at location
            self._episode_log.append({
                "step": self._state.step_count,
                "event": "action_dispatch",
                "ambulance_id": action.ambulance_id,
                "target_node": action.target_node,
                "result": "wasted",
            })

    def _do_prepare_or(self, action) -> None:
        from .models.actions import PrepareOR
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
        self._episode_log.append({
            "step": self._state.step_count,
            "event": "action_prepare_or",
            "hospital_id": action.hospital_id,
            "procedure_type": action.procedure_type,
            "or_index": result["or_index"],
            "result": "success",
        })

    def _do_page_specialist(self, action) -> None:
        from .models.actions import PageSpecialist
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
        from .models.actions import AssignHospital
        patient = self._patient_manager.get(action.patient_id)
        if patient is None:
            self._alerts.append(f"AssignHospital failed: patient {action.patient_id} not found")
            return
        if not self._hospital_system.can_treat(action.hospital_id, patient.condition):
            self._alerts.append(
                f"AssignHospital failed: {action.hospital_id} cannot treat "
                f"{patient.condition} (needs {PATIENT_CONDITION_REQUIREMENTS[patient.condition][0]})"
            )
            return
        if self._hospital_system.get(action.hospital_id).on_diversion:
            self._alerts.append(f"AssignHospital failed: {action.hospital_id} is on diversion")
            return
        patient.assigned_hospital = action.hospital_id

    def _do_preempt_or(self, action) -> None:
        from .models.actions import PreemptOR
        result = self._hospital_system.preempt_or(action.hospital_id, action.or_index)
        if not result["success"]:
            self._alerts.append(f"PreemptOR failed: {result['reason']}")
            return
        self._alerts.append(
            f"OR preempted at {action.hospital_id} OR {action.or_index}: "
            f"harm={result['harm']:.2f}, recovery={result['recovery_time']}min"
        )

    def _do_allocate_blood(self, action) -> None:
        from .models.actions import AllocateBlood
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
                f"Emergency blood: {result['units']} units {result['blood_type']} released for {action.patient_id} "
                f"at {action.hospital_id}"
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
        from .models.actions import TransferBlood
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
        from .models.actions import RequestMutualAid
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

        # Sync state for API consumers
        self._state.mutual_aid_available = self._mutual_aid_manager.get_available()
        self._state.mutual_aid_used = self._mutual_aid_manager.get_used()

        # Determine optimal arrival for logging
        pending = self._mutual_aid_manager.get_pending().get(ma_id)
        if pending:
            target_pid = pending.patient_id
            if target_pid:
                patient = self._patient_manager.get(target_pid)
                self._alerts.append(
                    f"Mutual aid requested: {ma_id} for {target_pid} "
                    f"(ETA step {pending.arrival_step})"
                )
            else:
                self._alerts.append(f"Mutual aid requested: {ma_id} — no waiting patients")
            self._episode_log.append({
                "step": self._state.step_count,
                "event": "mutual_aid_called",
                "ambulance_id": ma_id,
                "patient_id": target_pid,
                "optimal_arrival_step": pending.arrival_step,
            })

    def _do_query_blood_type(self, action) -> None:
        from .models.actions import QueryBloodType
        patient = self._patient_manager.get(action.patient_id)
        if patient is None:
            self._alerts.append(f"QueryBloodType failed: patient {action.patient_id} not found")
            return
        # Schedule blood type reveal after 5 minutes
        self._pending_blood_queries[action.patient_id] = 5

    def _do_query_or_status(self, action) -> None:
        from .models.actions import QueryORStatus
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
        """Handle TriageCall action - decide what to do with a pending dispatch call."""
        from .models.entities import DispatchCategory
        from .subsystems.constants import DISPATCH_CATEGORY_MAP, ALS_NEEDED_PROB
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
            outcome = {
                "call_id": call["call_id"],
                "decision": "self_transport",
                "category": cat_val,
                "true_condition": None,
                "als_needed": None,
                "revealed_at_step": None,
            }
            self._dispatch_outcomes_history.append(outcome)
            self._pending_calls = [c for c in self._pending_calls if c["call_id"] != action.call_id]
            if action.call_id in self._pending_call_countdown:
                del self._pending_call_countdown[action.call_id]
            self._alerts.append(f"Self-transport advised for {action.call_id}")
        elif decision == "callback":
            call["time_waiting"] = 0
            self._pending_call_countdown[action.call_id] = 15
            self._alerts.append(f"Callback scheduled for {action.call_id}")
        elif decision == "no_dispatch":
            category = call["category"]
            cat_val = category.value if hasattr(category, "value") else category
            outcome = {
                "call_id": call["call_id"],
                "decision": "no_dispatch",
                "category": cat_val,
                "true_condition": None,
                "als_needed": None,
                "revealed_at_step": None,
            }
            self._dispatch_outcomes_history.append(outcome)
            self._pending_calls = [c for c in self._pending_calls if c["call_id"] != action.call_id]
            if action.call_id in self._pending_call_countdown:
                del self._pending_call_countdown[action.call_id]
            self._alerts.append(f"No dispatch for {action.call_id}")

    def _do_dispatch_als(self, action) -> None:
        """Handle DispatchALS action."""
        from .models.actions import TriageCall
        triage = TriageCall(
            call_id=action.call_id,
            decision="dispatch_als",
            ambulance_id=action.ambulance_id,
        )
        self._do_triage_call(triage)

    def _do_dispatch_bls(self, action) -> None:
        """Handle DispatchBLS action."""
        from .models.actions import TriageCall
        triage = TriageCall(
            call_id=action.call_id,
            decision="dispatch_bls",
            ambulance_id=action.ambulance_id,
        )
        self._do_triage_call(triage)

    # =========================================================================
    # Termination & Reward
    # =========================================================================

    def _check_done(self) -> bool:
        """Check if episode should terminate."""
        if self._state.step_count >= self._state.max_steps:
            return True
        non_terminal = [
            p for p in self._patients
            if p.status not in ("treated", "deceased")
        ]
        if len(non_terminal) == 0:
            self._state.all_patients_terminal = True
            return True
        # Early termination: all non-deceased patients have vitals_score <= 0
        alive = [p for p in self._patients if p.status not in ("treated", "deceased")]
        if alive and all(p.vitals_score <= 0.0 for p in alive):
            self._state.all_patients_terminal = True
            return True
        return False

    def _compute_step_reward(self) -> float:
        """Dense reward: vitals delta shaping + milestone bonuses/penalties."""
        reward = 0.0
        from .subsystems.constants import (
            VITALS_DELTA_WEIGHT, MILESTONE_REWARDS, REWARD_STEP_CLAMP
        )

        # Guard: no reward computation once all patients are terminal
        if self._state.all_patients_terminal:
            return 0.0

        for pid, prev in self._prev_vitals.items():
            curr_patient = self._patient_manager.patients_dict.get(pid)
            if curr_patient is None:
                continue
            # Vitals delta shaping
            reward += (curr_patient.vitals_score - prev) * VITALS_DELTA_WEIGHT

        # Milestone bonuses/penalties
        for pid, prev_status in self._prev_patient_status.items():
            curr_patient = self._patient_manager.patients_dict.get(pid)
            if curr_patient is None:
                continue
            curr_status = curr_patient.status
            if prev_status == "waiting" and curr_status == "dispatched":
                reward += MILESTONE_REWARDS["dispatched"]
            if prev_status == "dispatched" and curr_status == "in_treatment":
                reward += MILESTONE_REWARDS["in_treatment"]
            if prev_status != "treated" and curr_status == "treated":
                reward += MILESTONE_REWARDS["treated"]
            if prev_status != "deceased" and curr_status == "deceased":
                reward += MILESTONE_REWARDS["deceased"]

        # Phase 2: Dispatch classification reward (for task5 with cascade_enabled)
        cascade_enabled = TASK_CONFIG.get(self._state.task_id, {}).get("cascade_enabled", False)
        if cascade_enabled:
            for outcome in self._dispatch_outcomes_history[-10:]:
                if outcome["true_condition"] is None:
                    continue
                als_needed = outcome["als_needed"]
                als_dispatched = outcome["decision"] == "als"
                if als_needed and als_dispatched:
                    reward += 0.005
                elif not als_needed and not als_dispatched:
                    reward += 0.002
                elif als_needed and not als_dispatched:
                    reward -= 0.020
                elif not als_needed and als_dispatched:
                    reward -= 0.005
                if outcome["decision"] in ("self_transport", "callback") and als_needed:
                    reward -= 0.030

        # Bootstrap reward for new patients (secondary/cascade spawns mid-episode)
        # Without this, new patients have no prev_vitals entry → zero delta → agent
        # cannot learn to respond to secondary patients via reward shaping.
        for patient in self._patient_manager.patients:
            if patient.id not in self._prev_vitals:
                self._prev_vitals[patient.id] = patient.vitals_score
                self._prev_patient_status[patient.id] = patient.status
                reward += MILESTONE_REWARDS["dispatched"]  # "you found a new patient" signal

        # Snapshot for next step
        self._prev_vitals = {
            p.id: p.vitals_score for p in self._patient_manager.patients
        }
        self._prev_patient_status = {
            p.id: p.status for p in self._patient_manager.patients
        }

        lo, hi = REWARD_STEP_CLAMP
        return max(lo, min(hi, reward))

    def _compute_time_score_preview(self) -> float:
        """Running estimate of time_score axis."""
        scores = []
        for p in self._patients:
            if p.status == "in_treatment" and p.treatment_complete_time is not None:
                target = PATIENT_TARGET_TIMES.get(p.condition, 60)
                actual = p.treatment_complete_time
                score = max(0.0, min(1.0, 1.0 - (actual - target) / target))
                scores.append(score)
            elif p.status == "deceased":
                scores.append(0.0)
            else:
                target = PATIENT_TARGET_TIMES.get(p.condition, 60)
                projected = self._state.step_count + 5 + 10 + 10
                score = max(0.0, min(1.0, 1.0 - (projected - target) / target))
                scores.append(score)
        return sum(scores) / len(scores) if scores else 1.0

    # =========================================================================
    # Observation
    # =========================================================================

    def _build_observation(self) -> CodeRedObservation:
        """Build the current observation."""
        self._state.cum_reward = round(self._state.cum_reward, 4)

        # Convert subsystem entities to Pydantic models for observation
        # Map subsystem status to Pydantic status
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
                assigned_ambulance=None,  # ambulance assignment tracked by ambulance_manager, not here
                assigned_hospital=p.assigned_hospital,
                status=_map_patient_status(p.status),
                vitals_score=p.vitals_score,
                blood_type=p.blood_type,
                treatment_start_time=p.arrival_hospital_step,
                treatment_complete_time=p.treatment_complete_time,
                outcome=p.outcome,
                is_secondary=getattr(p, "is_secondary", False),
                icu_status=p.icu_status,
                # Phase 2 fields
                dispatch_call_id=getattr(p, "dispatch_call_id", None),
                cascade_trigger_reason=getattr(p, "cascade_trigger_reason", None),
                observed_condition=getattr(p, "observed_condition", None),
            )
            for p in self._patients
        ]

        # Get ambulances from ambulance manager
        amb_states = []
        for amb_id, amb in self._ambulance_manager.all().items():
            # Map subsystem status to Pydantic model status
            status_map = {
                "available": AmbulanceStatus.AVAILABLE,
                "en_route": AmbulanceStatus.DISPATCHED,
                "on_scene": AmbulanceStatus.DISPATCHED,
                "returning": AmbulanceStatus.RETURNING,
                "off_duty": AmbulanceStatus.OFFLINE,
            }
            amb_states.append(AmbulanceState(
                id=amb.id,
                node_id=amb.target_node or amb.base_node or "",  # Last known position
                equipment=AmbulanceEquipment(amb.equipment),
                status=status_map.get(amb.status, AmbulanceStatus.AVAILABLE),
                assigned_patient=amb.patient_id,
                route=amb.route,
                eta_minutes=amb.eta_minutes,
                destination_type="patient" if amb.target_node else None,
            ))

        # Get hospitals from hospital system
        hosp_states = []
        for hosp_id, hosp in self._hospital_system.all().items():
            ors = [
                OperatingRoom(
                    index=o.index,
                    status=ORStatus(o.status),  # subsystem OR uses same lowercase values
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

        # Get blood banks from blood bank system
        blood_states = []
        for bb_id, bb in self._blood_bank.all().items():
            blood_states.append(BloodBankState(
                hospital_id=bb.hospital_id,
                stocks=dict(bb.stocks),
                crossmatch_queue=list(bb.crossmatch_queue),
            ))

        # Convert road network to observation model
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

        # Build pending_calls for Phase 2
        from .models.entities import DispatchCall as DispatchCallModel, DispatchOutcome as DispatchOutcomeModel
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

        # Recent dispatch outcomes (last 5)
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

        # Overcrowding modifier
        active_patients = [p for p in self._patients if p.status not in ("treated", "deceased")]
        overcrowding_modifier = self._cascade_engine.check_overcrowding(len(active_patients))

        return CodeRedObservation(
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
                4
            ),
            patients_remaining=len([
                p for p in self._patients
                if p.status not in ("treated", "deceased")
            ]),
            pending_calls=pending_calls_obs,
            recent_dispatch_outcomes=recent_outcomes,
            overcrowding_modifier=overcrowding_modifier,
        )

    # =========================================================================
    # Helpers
    # =========================================================================

    def _condition_to_tier(self, condition: str) -> PatientTier:
        mapping = {
            "cardiac": PatientTier.CRITICAL,
            "stroke": PatientTier.CRITICAL,
            "trauma": PatientTier.HIGH,
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
                from .subsystems.patient_manager import PatientManager
                spawn_nodes = PatientManager._SPAWN_NODES
                node = self._rng.choice(spawn_nodes)
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
        """Spawn a new dispatch call into the queue."""
        from .models.entities import DispatchCategory
        from .subsystems.constants import DISPATCH_CATEGORY_MAP
        if len(self._pending_calls) >= 5:
            return
        call_id = f"CALL_{self._state.step_count:04d}"
        categories = list(DISPATCH_CATEGORY_MAP.keys())
        weights = [0.20, 0.15, 0.25, 0.20, 0.20]
        category = self._rng.choices(categories, weights=weights)[0]
        from .subsystems.patient_manager import PatientManager
        spawn_nodes = PatientManager._SPAWN_NODES
        location = self._rng.choice(spawn_nodes)
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
        self._alerts.append(f"CALL: {category.value} call {call_id} at {location}")
        self._episode_log.append({
            "step": self._state.step_count,
            "call_id": call_id,
            "event": "call_received",
            "category": category.value,
            "location": location,
        })

    def _spawn_patient_from_call(self, call_id: str) -> None:
        """Force-spawn a patient when a call's countdown expires."""
        call = next((c for c in self._pending_calls if c["call_id"] == call_id), None)
        if call is None:
            return
        self._pending_calls = [c for c in self._pending_calls if c["call_id"] != call_id]
        if call_id in self._pending_call_countdown:
            del self._pending_call_countdown[call_id]
        from .subsystems.constants import DISPATCH_CATEGORY_MAP
        category = call["category"]
        condition_choices = DISPATCH_CATEGORY_MAP[category]
        conditions, probs = zip(*condition_choices)
        true_condition = self._rng.choices(list(conditions), weights=list(probs))[0]
        patient = self._patient_manager.spawn_secondary(
            condition=true_condition,
            triggered_by=None,
            reason="forced_spawn",
            onset_step=self._state.step_count,
            spawn_node=call["location_node"],
        )
        patient.dispatch_call_id = call_id
        patient.observed_condition = true_condition
        call["spawned_patient_id"] = patient.id
        self._patients = self._patient_manager.patients
        self._alerts.append(f"ON-SCENE: {category.value} call {call_id} force-spawned")
        self._episode_log.append({
            "step": self._state.step_count,
            "patient_id": patient.id,
            "event": "patient_created",
            "condition": true_condition,
            "is_secondary": False,
            "call_id": call_id,
            "target_time": PATIENT_TARGET_TIMES.get(true_condition, 60),
        })
        outcome = {
            "call_id": call_id,
            "decision": "no_dispatch",
            "category": category.value,
            "true_condition": true_condition,
            "als_needed": true_condition in ("cardiac", "stroke", "trauma"),
            "revealed_at_step": self._state.step_count,
        }
        # Backfill the existing pending outcome entry from dispatch_als/dispatch_bls.
        # Without this, a dispatch+spawn creates TWO entries: one with null ground truth
        # (from dispatch) and one with correct ground truth (here). The first entry's
        # true_condition stays None forever, poisoning the cascade classification reward.
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
