"""CodeRedEnvironment — Core OpenEnv environment."""

from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

from .models import (
    CodeRedAction,
    CodeRedObservation,
    CodeRedState,
)
from .models.entities import (
    AmbulanceEquipment,
    AmbulanceState,
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
    CITY_EDGES,
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
        self._patients: List[Patient] = []
        self._ambulances: Dict[str, AmbulanceState] = {}
        self._hospitals: Dict[str, HospitalState] = {}
        self._blood_banks: Dict[str, BloodBankState] = {}
        self._road_network: RoadNetworkState = RoadNetworkState()
        self._state: CodeRedState = CodeRedState()
        self._alerts: List[str] = []
        self._action_results: Dict[int, str] = {}
        self._rng_seed: Optional[int] = None
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

        self._rng_seed = seed
        rng = random.Random(seed)

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
        self._action_results = {}
        self._episode_log = []

        # Initialize subsystems
        self._init_road_network()
        self._init_hospitals()
        self._init_ambulances()
        self._init_blood_banks()
        self._init_patients(task_id, rng)

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
    # Initialization
    # =========================================================================

    def _init_road_network(self) -> None:
        """Build the road network state from constants."""
        self._road_network = RoadNetworkState()
        for edge in CITY_EDGES:
            key = self._road_network.get_edge_key(edge["from"], edge["to"])
            self._road_network.edges[key] = EdgeState(
                from_node=edge["from"],
                to_node=edge["to"],
                base_time=edge["base_time"],
                congestion_multiplier=1.0,
                disrupted=False,
            )

    def _init_hospitals(self) -> None:
        """Initialize hospital states from constants."""
        self._hospitals = {}
        for hosp in HOSPITALS:
            ors = [
                OperatingRoom(index=i, status=ORStatus.IDLE)
                for i in range(hosp["num_or"])
            ]
            specialists = {
                role: SpecialistStatus(
                    available=data["available"],
                    total=data["total"],
                )
                for role, data in hosp["specialists"].items()
            }
            self._hospitals[hosp["id"]] = HospitalState(
                id=hosp["id"],
                node_id=hosp["node_id"],
                capabilities=hosp["capabilities"],
                specialists=specialists,
                operating_rooms=ors,
                icu_beds=hosp["icu_beds"],
                blood_stock=dict(hosp["blood_stock"]),
            )

    def _init_ambulances(self) -> None:
        """Initialize ambulance fleet from constants."""
        self._ambulances = {}
        for amb in AMBULANCES:
            self._ambulances[amb["id"]] = AmbulanceState(
                id=amb["id"],
                node_id=amb["base_node"],
                equipment=AmbulanceEquipment(amb["equipment"]),
                status=AmbulanceStatus.AVAILABLE,
                route=[],
                eta_minutes=0,
            )

    def _init_blood_banks(self) -> None:
        """Initialize blood banks from hospital states."""
        self._blood_banks = {}
        for hosp_id, hosp in self._hospitals.items():
            self._blood_banks[hosp_id] = BloodBankState(
                hospital_id=hosp_id,
                stocks=dict(hosp.blood_stock),
                crossmatch_queue=[],
            )

    def _init_patients(self, task_id: str, rng) -> None:
        """Generate patients for the task from constants."""
        self._patients = []
        config = TASK_CONFIG[task_id]

        for i, pconfig in enumerate(config["patients"]):
            condition = PatientCondition(pconfig["condition"])
            tier = self._condition_to_tier(condition)

            non_hospital_nodes = [
                n["id"] for n in CITY_NODES if n["type"] != "hospital"
            ]
            location = rng.choice(non_hospital_nodes)

            patient = Patient(
                patient_id=f"P{i+1}",
                condition=condition,
                tier=tier,
                location_node=location,
                time_since_onset=0,
                status=PatientStatus.WAITING,
            )
            self._episode_log.append({
                "step": 0,
                "patient_id": patient.patient_id,
                "event": "patient_created",
                "condition": condition.value,
                "is_secondary": patient.is_secondary,
            })
            self._patients.append(patient)

    # =========================================================================
    # Time Advancement
    # =========================================================================

    def _advance_time(self) -> None:
        """Advance all systems by 1 minute."""
        # Patient deterioration
        for p in self._patients:
            if p.status not in (PatientStatus.TREATED, PatientStatus.DECEASED):
                p.time_since_onset += 1
                self._check_patient_outcome(p)

        # Ambulance movement
        for amb in self._ambulances.values():
            if amb.status in (AmbulanceStatus.DISPATCHED, AmbulanceStatus.TRANSPORTING):
                if amb.eta_minutes > 0:
                    amb.eta_minutes -= 1
                if amb.eta_minutes == 0:
                    self._arrive_at_destination(amb)

        # OR prep countdowns
        for hosp in self._hospitals.values():
            for or_index, mins_left in list(hosp.or_prep_countdowns.items()):
                if mins_left <= 1:
                    del hosp.or_prep_countdowns[or_index]
                    for or_obj in hosp.operating_rooms:
                        if or_obj.index == or_index:
                            or_obj.status = ORStatus.IDLE
                else:
                    hosp.or_prep_countdowns[or_index] = mins_left - 1

            # OR in-use countdowns
            for or_obj in hosp.operating_rooms:
                if or_obj.status == ORStatus.IN_USE and or_obj.minutes_remaining:
                    if or_obj.minutes_remaining <= 1:
                        or_obj.minutes_remaining = None
                        or_obj.status = ORStatus.IDLE
                        or_obj.procedure_type = None
                        or_obj.patient_id = None
                    else:
                        or_obj.minutes_remaining -= 1

        # Blood crossmatch queue
        for hosp_id, bb in self._blood_banks.items():
            for entry in list(bb.crossmatch_queue):
                entry["time_remaining"] -= 1
                if entry["time_remaining"] <= 0:
                    bb.stocks[entry["blood_type"]] -= entry["units"]
                    self._alerts.append(
                        f"Crossmatch complete: {entry['units']} units "
                        f"{entry['blood_type']} reserved for "
                        f"{entry['patient_id']} at {hosp_id}"
                    )

    def _check_patient_outcome(self, patient: Patient) -> None:
        """Check if a patient has exceeded their survival window."""
        target = PATIENT_TARGET_TIMES[patient.condition.value]
        if patient.time_since_onset >= target + 15:
            patient.status = PatientStatus.DECEASED
            patient.outcome = "deceased"
            self._episode_log.append({
                "step": self._state.step_count,
                "patient_id": patient.patient_id,
                "event": "patient_deceased",
                "reason": "timeout",
            })

    def _arrive_at_destination(self, amb: AmbulanceState) -> None:
        """Handle ambulance arriving at its destination."""
        if amb.status == AmbulanceStatus.DISPATCHED:
            patient = next(
                (p for p in self._patients if p.patient_id == amb.assigned_patient),
                None,
            )
            if patient and patient.status == PatientStatus.WAITING:
                patient.status = PatientStatus.TRANSPORTING
                amb.node_id = patient.location_node
                if patient.assigned_hospital:
                    hosp = self._hospitals[patient.assigned_hospital]
                    route = self._compute_route(amb.node_id, hosp.node_id)
                    amb.route = route
                    amb.eta_minutes = self._route_travel_time(route)
                    amb.destination_type = "hospital"
                else:
                    amb.status = AmbulanceStatus.AVAILABLE
                    amb.assigned_patient = None

        elif amb.status == AmbulanceStatus.TRANSPORTING:
            patient = next(
                (p for p in self._patients if p.patient_id == amb.assigned_patient),
                None,
            )
            if patient and patient.assigned_hospital:
                patient.status = PatientStatus.TREATING
                patient.treatment_start_time = self._state.step_count
                amb.status = AmbulanceStatus.AVAILABLE
                amb.assigned_patient = None
                amb.route = []

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
        amb = self._ambulances.get(action.ambulance_id)
        if amb is None:
            self._alerts.append(f"Dispatch failed: ambulance {action.ambulance_id} not found")
            return
        if amb.status != AmbulanceStatus.AVAILABLE:
            self._alerts.append(
                f"Dispatch failed: ambulance {action.ambulance_id} is {amb.status.value}"
            )
            return

        amb.status = AmbulanceStatus.DISPATCHED
        amb.assigned_patient = None
        route = self._compute_route(amb.node_id, action.target_node)
        amb.route = route
        amb.eta_minutes = self._route_travel_time(route)
        amb.destination_type = "patient"

        waiting_patient = next(
            (p for p in self._patients
             if p.status == PatientStatus.WAITING
             and p.location_node == action.target_node),
            None,
        )
        if waiting_patient:
            amb.assigned_patient = waiting_patient.patient_id
            waiting_patient.assigned_ambulance = action.ambulance_id
            waiting_patient.status = PatientStatus.DISPATCHED
            self._episode_log.append({
                "step": self._state.step_count,
                "patient_id": waiting_patient.patient_id,
                "event": "dispatch",
                "ambulance_id": action.ambulance_id,
            })

    def _do_prepare_or(self, action) -> None:
        from .models.actions import PrepareOR
        hosp = self._hospitals.get(action.hospital_id)
        if hosp is None:
            self._alerts.append(f"PrepareOR failed: hospital {action.hospital_id} not found")
            return
        if action.procedure_type not in hosp.capabilities:
            self._alerts.append(
                f"PrepareOR failed: {action.hospital_id} does not have "
                f"{action.procedure_type} capability"
            )
            return
        idle_or = next((or_obj for or_obj in hosp.operating_rooms if or_obj.status == ORStatus.IDLE), None)
        if idle_or is None:
            self._alerts.append(f"PrepareOR failed: no idle OR at {action.hospital_id}")
            return
        idle_or.status = ORStatus.PREP
        hosp.or_prep_countdowns[idle_or.index] = 10
        self._episode_log.append({
            "step": self._state.step_count,
            "event": "action_prepare_or",
            "hospital_id": action.hospital_id,
            "or_index": idle_or.index,
        })

    def _do_page_specialist(self, action) -> None:
        from .models.actions import PageSpecialist
        hosp = self._hospitals.get(action.hospital_id)
        if hosp is None:
            self._alerts.append(f"PageSpecialist failed: hospital {action.hospital_id} not found")
            return
        spec = hosp.specialists.get(action.specialist_type)
        if spec is None or spec.total == 0:
            self._alerts.append(
                f"PageSpecialist failed: no {action.specialist_type} at {action.hospital_id}"
            )
            return
        if spec.available <= 0:
            self._alerts.append(
                f"PageSpecialist failed: no {action.specialist_type} available at {action.hospital_id}"
            )
            return
        spec.available -= 1
        spec.status = "paged"
        spec.minutes_until_available = 8

    def _do_assign_hospital(self, action) -> None:
        from .models.actions import AssignHospital
        patient = next((p for p in self._patients if p.patient_id == action.patient_id), None)
        if patient is None:
            self._alerts.append(f"AssignHospital failed: patient {action.patient_id} not found")
            return
        hosp = self._hospitals.get(action.hospital_id)
        if hosp is None:
            self._alerts.append(f"AssignHospital failed: hospital {action.hospital_id} not found")
            return
        if hosp.on_diversion:
            self._alerts.append(f"AssignHospital failed: {action.hospital_id} is on diversion")
            return
        required = PATIENT_CONDITION_REQUIREMENTS[patient.condition.value][0]
        if required not in hosp.capabilities:
            self._alerts.append(
                f"AssignHospital failed: {action.hospital_id} cannot treat "
                f"{patient.condition.value} (needs {required})"
            )
            return
        patient.assigned_hospital = action.hospital_id

    def _do_preempt_or(self, action) -> None:
        from .models.actions import PreemptOR
        hosp = self._hospitals.get(action.hospital_id)
        if hosp is None:
            self._alerts.append(f"PreemptOR failed: hospital {action.hospital_id} not found")
            return
        or_obj = next(
            (or_o for or_o in hosp.operating_rooms if or_o.index == action.or_index),
            None,
        )
        if or_obj is None:
            self._alerts.append(f"PreemptOR failed: OR {action.or_index} not found at {action.hospital_id}")
            return
        if or_obj.status == ORStatus.IDLE:
            self._alerts.append(f"PreemptOR failed: OR {action.or_index} is already idle")
            return
        harm = (or_obj.minutes_remaining or 0) / 30.0
        harm = min(1.0, harm)
        recovery_time = or_obj.minutes_remaining or 0
        or_obj.status = ORStatus.IDLE
        or_obj.procedure_type = None
        or_obj.minutes_remaining = None
        or_obj.patient_id = None
        self._alerts.append(
            f"OR preempted at {action.hospital_id} OR {action.or_index}: "
            f"harm={harm:.2f}, recovery={recovery_time}min"
        )

    def _do_allocate_blood(self, action) -> None:
        from .models.actions import AllocateBlood
        hosp = self._hospitals.get(action.hospital_id)
        if hosp is None:
            self._alerts.append(f"AllocateBlood failed: hospital {action.hospital_id} not found")
            return
        bb = self._blood_banks.get(action.hospital_id)
        if bb is None:
            return
        if action.emergency:
            if bb.stocks.get("O_NEG", 0) < action.units:
                self._alerts.append(
                    f"AllocateBlood failed: insufficient O_NEG at {action.hospital_id}"
                )
                return
            bb.stocks["O_NEG"] -= action.units
            self._alerts.append(
                f"Emergency blood: {action.units} units O_NEG released for {action.patient_id} "
                f"at {action.hospital_id}"
            )
        else:
            if bb.stocks.get(action.blood_type, 0) < action.units:
                self._alerts.append(
                    f"AllocateBlood failed: insufficient {action.blood_type} at {action.hospital_id}"
                )
                return
            bb.crossmatch_queue.append({
                "patient_id": action.patient_id,
                "blood_type": action.blood_type,
                "units": action.units,
                "time_remaining": 15,
            })

    def _do_transfer_blood(self, action) -> None:
        from .models.actions import TransferBlood
        bb_from = self._blood_banks.get(action.from_hospital)
        bb_to = self._blood_banks.get(action.to_hospital)
        if bb_from is None or bb_to is None:
            self._alerts.append("TransferBlood failed: hospital not found")
            return
        if bb_from.stocks.get(action.blood_type, 0) < action.units:
            self._alerts.append(
                f"TransferBlood failed: insufficient {action.blood_type} at {action.from_hospital}"
            )
            return
        bb_from.stocks[action.blood_type] -= action.units
        bb_to.stocks[action.blood_type] = bb_to.stocks.get(action.blood_type, 0) + action.units
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
        self._ambulances[new_id] = AmbulanceState(
            id=new_id,
            node_id=random.choice(non_hospital),
            equipment=AmbulanceEquipment.BLS,
            status=AmbulanceStatus.AVAILABLE,
        )
        self._alerts.append(
            f"Mutual aid requested: {new_id} arriving in 12 min (available now as reserve)"
        )
        self._episode_log.append({
            "step": self._state.step_count,
            "event": "mutual_aid_called",
            "ambulance_id": new_id,
        })

    def _do_query_blood_type(self, action) -> None:
        from .models.actions import QueryBloodType
        patient = next((p for p in self._patients if p.patient_id == action.patient_id), None)
        if patient is None:
            self._alerts.append(f"QueryBloodType failed: patient {action.patient_id} not found")
            return
        import random
        patient.blood_type = random.choice(BLOOD_TYPES)
        self._alerts.append(
            f"Blood type revealed for {action.patient_id}: {patient.blood_type}"
        )

    def _do_query_or_status(self, action) -> None:
        from .models.actions import QueryORStatus
        hosp = self._hospitals.get(action.hospital_id)
        if hosp is None:
            return
        self._alerts.append(
            f"OR status at {action.hospital_id}: this action costs 1 step"
        )

    # =========================================================================
    # Routing
    # =========================================================================

    def _compute_route(self, from_node: str, to_node: str) -> List[str]:
        """Simple BFS shortest path. Returns list of node IDs including start and end."""
        import heapq
        if from_node == to_node:
            return [from_node]
        pq = [(0, from_node, [from_node])]
        visited = set()
        while pq:
            cost, node, path = heapq.heappop(pq)
            if node in visited:
                continue
            visited.add(node)
            if node == to_node:
                return path
            for edge_key, edge in self._road_network.edges.items():
                if edge.disrupted and edge.disruption_type == "road_closure":
                    continue
                next_node = None
                if edge.from_node == node:
                    next_node = edge.to_node
                elif edge.to_node == node:
                    next_node = edge.from_node
                if next_node and next_node not in visited:
                    new_cost = cost + edge.effective_time()
                    heapq.heappush(pq, (new_cost, next_node, path + [next_node]))
        return [from_node]

    def _route_travel_time(self, route: List[str]) -> int:
        """Compute total travel time for a route."""
        total = 0
        for i in range(len(route) - 1):
            t = self._road_network.get_travel_time(route[i], route[i + 1])
            if t == float("inf"):
                return 999
            total += int(t)
        return total

    # =========================================================================
    # Termination & Reward
    # =========================================================================

    def _check_done(self) -> bool:
        """Check if episode should terminate."""
        if self._state.step_count >= self._state.max_steps:
            return True
        non_terminal = [
            p for p in self._patients
            if p.status not in (PatientStatus.TREATED, PatientStatus.DECEASED)
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
            if p.status == PatientStatus.TREATING and p.treatment_complete_time is not None:
                target = PATIENT_TARGET_TIMES[p.condition.value]
                actual = p.treatment_complete_time
                score = max(0.0, min(1.0, 1.0 - (actual - target) / target))
                scores.append(score)
            elif p.status == PatientStatus.DECEASED:
                scores.append(0.0)
            else:
                target = PATIENT_TARGET_TIMES[p.condition.value]
                projected = p.time_since_onset + 5 + 10 + 10
                score = max(0.0, min(1.0, 1.0 - (projected - target) / target))
                scores.append(score)
        return sum(scores) / len(scores) if scores else 1.0

    # =========================================================================
    # Observation
    # =========================================================================

    def _build_observation(self) -> CodeRedObservation:
        """Build the current observation."""
        self._state.cum_reward = round(self._state.cum_reward, 4)
        return CodeRedObservation(
            step=self._state.step_count + 1,
            patients=list(self._patients),
            ambulances=list(self._ambulances.values()),
            hospitals=list(self._hospitals.values()),
            blood_banks=list(self._blood_banks.values()),
            road_network=self._road_network,
            alerts=list(self._alerts),
            mutual_aid_remaining=self._state.mutual_aid_available,
            time_score_preview=round(self._compute_time_score_preview(), 4),
            patients_remaining=len([
                p for p in self._patients
                if p.status not in (PatientStatus.TREATED, PatientStatus.DECEASED)
            ]),
        )

    # =========================================================================
    # Helpers
    # =========================================================================

    def _condition_to_tier(self, condition: PatientCondition) -> PatientTier:
        mapping = {
            PatientCondition.CARDIAC: PatientTier.CRITICAL,
            PatientCondition.STROKE: PatientTier.CRITICAL,
            PatientCondition.TRAUMA: PatientTier.HIGH,
            PatientCondition.GENERAL: PatientTier.MEDIUM,
        }
        return mapping[condition]
