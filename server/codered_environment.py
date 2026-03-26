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
    HOSPITALS,
    PATIENT_CONDITION_REQUIREMENTS,
    PATIENT_TARGET_TIMES,
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
        from .subsystems.patient_manager import PatientManager
        from .subsystems.ambulance_manager import AmbulanceManager

        self._rng: Optional[Any] = None
        self._patients: List[Patient] = []
        self._road_network = RoadNetwork()
        self._hospital_system = HospitalSystem()
        self._blood_bank = BloodBankSystem()
        self._disruption_engine = DisruptionEngine()
        self._patient_manager = PatientManager()
        self._ambulance_manager = AmbulanceManager(AMBULANCES)
        self._state: CodeRedState = CodeRedState()
        self._alerts: List[str] = []
        self._pending_blood_queries: Dict[str, int] = {}
        self._pending_or_queries: Dict[str, int] = {}
        self._active_disruptions: List[Dict] = []
        self._episode_log: List[dict] = []

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
        from .subsystems.patient_manager import PatientManager
        from .subsystems.ambulance_manager import AmbulanceManager

        self._road_network = RoadNetwork()
        self._hospital_system = HospitalSystem()
        self._blood_bank = BloodBankSystem()
        self._disruption_engine.reset(seed=seed or 0, task_id=task_id)
        self._patient_manager.reset(task_id, self._rng)
        self._patients = self._patient_manager.patients
        self._ambulance_manager = AmbulanceManager(AMBULANCES)

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

        # Log patient creation
        for p in self._patients:
            self._episode_log.append({
                "step": 0,
                "patient_id": p.id,
                "event": "patient_created",
                "condition": p.condition,
                "is_secondary": False,
            })

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
        # Patient deterioration + outcome check
        self._patient_manager.tick()
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
            # Check for treatment completion (patient reaches treating)
            if p.status == "in_treatment" and p.treatment_complete_time is None:
                # This will be set when surgery completes
                pass

        # Ambulances
        self._ambulance_manager.tick()
        for amb_id, amb in self._ambulance_manager.all().items():
            if amb.status == "on_scene" and amb.eta_minutes == 0:
                # Ambulance arrived, check if it was transporting
                if amb.patient_id:
                    patient = next((pt for pt in self._patients if pt.id == amb.patient_id), None)
                    if patient and patient.status == "transporting":
                        self._do_treatment_arrival(patient, amb_id)

        # Hospitals
        self._hospital_system.tick()

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
                self._alerts.append(f"Surge event: additional patients arriving")
            else:
                self._alerts.append(
                    f"Disruption: {event['disruption_type']} on {event['target']}"
                )

    def _road_network_tick_disruptions(self) -> None:
        """Process road network disruption events — check for newly expired disruptions."""
        active = self._road_network.get_active_disruptions()
        for disp in active:
            if disp["remaining_steps"] <= 0:
                self._road_network.clear_disruption(disp["from_node"], disp["to_node"])
                self._alerts.append(f"Road cleared: {disp['from_node']} <-> {disp['to_node']}")

    def _reveal_blood_type(self, patient_id: str) -> None:
        """Reveal a patient's blood type."""
        import random
        patient = self._patient_manager.get(patient_id)
        if patient:
            patient.blood_type = random.choice(BLOOD_TYPES)
            self._alerts.append(f"Blood type revealed for {patient_id}: {patient.blood_type}")

    def _do_treatment_arrival(self, patient, ambulance_id: str) -> None:
        """Handle patient arriving at hospital for treatment."""
        if patient.assigned_hospital:
            hosp = self._hospital_system.get(patient.assigned_hospital)
            if hosp:
                # Start surgery/treatment
                idle_or = self._hospital_system.get_idle_or(patient.assigned_hospital)
                if idle_or:
                    # Determine procedure type based on condition
                    procedure_map = {
                        "cardiac": "cardiac",
                        "stroke": "stroke",
                        "trauma": "trauma",
                        "general": "general",
                    }
                    procedure_type = procedure_map.get(patient.condition, "general")

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
            QueryBloodType, QueryORStatus, MaintainPlan,
        )

        if isinstance(action, DispatchAmbulance):
            self._do_dispatch_ambulance(action)
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
            })

    def _do_prepare_or(self, action) -> None:
        from .models.actions import PrepareOR
        result = self._hospital_system.prepare_or(action.hospital_id, action.procedure_type)
        if not result["success"]:
            self._alerts.append(f"PrepareOR failed: {result['reason']}")
            return
        self._episode_log.append({
            "step": self._state.step_count,
            "event": "action_prepare_or",
            "hospital_id": action.hospital_id,
            "or_index": result["or_index"],
        })

    def _do_page_specialist(self, action) -> None:
        from .models.actions import PageSpecialist
        result = self._hospital_system.page_specialist(action.hospital_id, action.specialist_type)
        if not result["success"]:
            self._alerts.append(f"PageSpecialist failed: {result['reason']}")
            return

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
                return
            self._alerts.append(
                f"Emergency blood: {result['units']} units {result['blood_type']} released for {action.patient_id} "
                f"at {action.hospital_id}"
            )
        else:
            result = self._blood_bank.start_crossmatch(
                action.hospital_id, action.patient_id, action.blood_type, action.units
            )
            if not result["success"]:
                self._alerts.append(f"AllocateBlood failed: {result['reason']}")
                return

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
        if self._state.mutual_aid_available <= 0:
            self._alerts.append("Mutual aid failed: no calls remaining")
            return
        self._state.mutual_aid_available -= 1
        self._state.mutual_aid_used += 1
        import random
        non_hospital = [n["id"] for n in CITY_NODES if n["type"] != "hospital"]
        new_id = f"MUTUAL_{self._state.mutual_aid_used}"
        # Note: Mutual aid ambulance would need to be added to ambulance_manager
        # For now, just track it in episode log
        self._alerts.append(
            f"Mutual aid requested: {new_id} available as reserve"
        )
        self._episode_log.append({
            "step": self._state.step_count,
            "event": "mutual_aid_called",
            "ambulance_id": new_id,
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

    # =========================================================================
    # Routing (delegate to road network subsystem)
    # =========================================================================

    def _compute_route(self, from_node: str, to_node: str) -> List[str]:
        """Compute shortest path using road network subsystem."""
        return self._road_network.shortest_path(from_node, to_node)

    def _route_travel_time(self, route: List[str]) -> int:
        """Compute total travel time for a route using road network subsystem."""
        return self._road_network.route_travel_time(route)

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
        return False

    def _compute_step_reward(self) -> float:
        """Dense reward based on time_score_preview delta."""
        return -0.01

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
                assigned_ambulance=p.assigned_hospital,  # Will be set by ambulance manager
                assigned_hospital=p.assigned_hospital,
                status=_map_patient_status(p.status),
                blood_type=p.blood_type,
                treatment_start_time=p.arrival_hospital_step,
                treatment_complete_time=p.treatment_complete_time,
                outcome=p.outcome,
                is_secondary=False,
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
            patients_remaining=len([
                p for p in self._patients
                if p.status not in ("treated", "deceased")
            ]),
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
