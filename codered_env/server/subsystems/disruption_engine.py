"""Seeded disruption generator for CodeRedEnv."""

import random
from typing import Dict, List, Literal, Optional

# Disruption type weights per task difficulty
DISRUPTION_TYPES_BY_TASK = {
    "task1": [],
    "task2": ["road_closure", "hospital_diversion", "accident"],
    "task3": ["road_closure", "hospital_diversion", "accident", "equipment_failure", "surge_event"],
    "task4": ["road_closure", "hospital_diversion", "accident"],
    "task5": ["road_closure", "hospital_diversion", "accident", "equipment_failure", "surge_event"],
}

BASE_PROB_PER_TASK = {
    "task1": 0.0,
    "task2": 0.05,
    "task3": 0.15,
    "task4": 0.05,
    "task5": 0.15,
}


class DisruptionEngine:
    """
    Generates disruptions using seeded probability with jitter.

    At reset(seed), a deterministic intensity multiplier k is drawn from
    Uniform(0.7, 1.3). Per-step disruption chance = base_prob * k.
    Disruption target is selected deterministically from seed + step index.
    """

    def __init__(self):
        self._seed: int = 0
        self._task_id: str = "task1"
        self._intensity: float = 1.0
        self._rng: Optional[random.Random] = None
        self._scheduled_disruptions: List[Dict] = []

    def reset(self, seed: int, task_id: str) -> None:
        """Initialize disruption engine for a new episode."""
        self._seed = seed
        self._task_id = task_id
        self._rng = random.Random(seed)

        # Draw intensity multiplier deterministically from seed
        self._intensity = self._rng.uniform(0.7, 1.3)

        # Pre-generate disruption schedule for this episode
        max_steps = {
            "task1": 30, "task2": 45, "task3": 60,
            "task4": 45, "task5": 60,
        }[task_id]
        self._scheduled_disruptions = self._generate_schedule(max_steps)

    def roll_disruptions(
        self,
        step: int,
        road_network,  # RoadNetwork instance
        hospital_system=None,
    ) -> List[Dict]:
        """
        At each step, check if a disruption is scheduled and apply it.
        Returns list of disruption events applied this step.
        """
        events = []
        for sched in self._scheduled_disruptions:
            if sched["step"] == step:
                event = dict(sched)
                target_type = sched["target_type"]

                if target_type == "edge":
                    from_node, to_node = sched["target"].split("->")
                    road_network.set_disruption(
                        from_node, to_node,
                        sched["disruption_type"],
                        sched["duration"],
                    )
                elif target_type == "hospital":
                    if hospital_system:
                        if sched["disruption_type"] == "hospital_diversion":
                            hospital_system.set_diversion(sched["target"], True)
                        elif sched["disruption_type"] == "equipment_failure":
                            or_idx = self._rng.randint(0, 2)
                            hospital_system.set_equipment_failure(
                                sched["target"], or_idx, sched["duration"]
                            )

                events.append(event)

        return events

    def get_optimal_mutual_aid_window(self, call_index: int) -> tuple[int, int]:
        """
        Return (optimal_window_start, optimal_window_end) for mutual aid call.
        """
        first_disruption_step = None
        for sched in self._scheduled_disruptions:
            if sched["disruption_type"] not in ("surge_event",):
                first_disruption_step = sched["step"]
                break

        if first_disruption_step is None:
            defaults = {
                "task2": (15, 25),
                "task3_call_1": (5, 15),
                "task3_call_2": (25, 35),
            }
            key = f"{self._task_id}_call_{call_index}"
            return defaults.get(key, (20, 30))

        start = max(1, first_disruption_step - 8)
        end = first_disruption_step + 2
        return (start, end)

    def _generate_schedule(self, max_steps: int) -> List[Dict]:
        """Generate disruption events from seed and intensity."""
        rng = random.Random(self._seed + 1000)
        base_prob = BASE_PROB_PER_TASK[self._task_id]
        effective_prob = base_prob * self._intensity

        available_types = DISRUPTION_TYPES_BY_TASK[self._task_id]
        if not available_types:
            return []

        from .constants import CITY_EDGES, HOSPITALS

        edges = [f"{e['from']}->{e['to']}" for e in CITY_EDGES]
        hospitals = [h["id"] for h in HOSPITALS]

        schedule = []

        for step_idx in range(1, max_steps + 1):
            if rng.random() < effective_prob:
                disp_type = rng.choice(available_types)

                target_seed = self._seed + step_idx
                target_rng = random.Random(target_seed)

                duration_map = {
                    "road_closure": 999,
                    "hospital_diversion": target_rng.randint(5, 15),
                    "accident": target_rng.randint(10, 20),
                    "equipment_failure": target_rng.randint(10, 30),
                }
                duration = duration_map.get(disp_type, 10)

                if disp_type == "road_closure":
                    edge_idx = target_rng.randint(0, len(edges) - 1)
                    target = edges[edge_idx]
                    target_type = "edge"
                elif disp_type == "hospital_diversion":
                    hosp_idx = target_rng.randint(0, len(hospitals) - 1)
                    target = hospitals[hosp_idx]
                    target_type = "hospital"
                elif disp_type == "equipment_failure":
                    hosp_idx = target_rng.randint(0, len(hospitals) - 1)
                    target = hospitals[hosp_idx]
                    target_type = "hospital"
                elif disp_type == "surge_event":
                    target = "SURGE"
                    target_type = "surge"
                else:
                    edge_idx = target_rng.randint(0, len(edges) - 1)
                    target = edges[edge_idx]
                    target_type = "edge"

                schedule.append({
                    "step": step_idx,
                    "disruption_type": disp_type,
                    "target": target,
                    "target_type": target_type,
                    "duration": duration,
                })

        return schedule
