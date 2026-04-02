"""Event-driven causal cascade engine for interdependent patient outcomes.

Mirrors the architecture of DisruptionEngine: seeded RNG, isolated logic,
testable in isolation, integrated via callback at environment level.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
import random


@dataclass
class CascadeRule:
    """A single cascade rule triggered by a patient outcome."""
    trigger: str  # "patient_deceased", "patient_saved"
    probability: float  # 0.0-1.0
    effect: str  # "spawn_secondary", "news_cycle"
    condition_filter: Optional[str] = None  # e.g. "trauma", "cardiac"
    effect_params: Dict[str, Any] = field(default_factory=dict)


class CascadeEngine:
    """
    Subscribes to patient outcome events and evaluates probabilistic cascade rules.
    """

    def __init__(self) -> None:
        self._rules: List[CascadeRule] = []
        self._rng: Optional[random.Random] = None
        self._overcrowding_active: bool = False
        self._news_cycle_steps_remaining: int = 0
        self._pending_surge_probability: float = 0.0
        self._callback: Optional[Callable] = None

    def reset(self, seed: int, episode_config: Optional[dict] = None) -> None:
        """Initialize engine with seed. Call at environment reset()."""
        self._rng = random.Random(seed)
        self._overcrowding_active = False
        self._news_cycle_steps_remaining = 0
        self._pending_surge_probability = 0.0
        self._load_rules(episode_config or {})

    def set_callback(self, callback: Callable) -> None:
        """Environment provides a callback for spawning patients, raising alerts."""
        self._callback = callback

    def _load_rules(self, config: dict) -> None:
        """Load cascade rules. Config can override probabilities."""
        self._rules = [
            CascadeRule(
                trigger="patient_deceased",
                condition_filter="trauma",
                probability=0.30,
                effect="spawn_secondary",
                effect_params={"secondary_condition": "cardiac", "reason": "psychogenic_cascade"},
            ),
            CascadeRule(
                trigger="patient_deceased",
                condition_filter="cardiac",
                probability=0.20,
                effect="spawn_secondary",
                effect_params={"secondary_condition": "cardiac", "reason": "sympathy_cascade"},
            ),
            CascadeRule(
                trigger="patient_saved",
                condition_filter="cardiac",
                probability=0.15,
                effect="news_cycle",
                effect_params={"steps": 10, "surge_prob_boost": 0.20, "condition": "cardiac"},
            ),
            CascadeRule(
                trigger="patient_saved",
                condition_filter="trauma",
                probability=0.10,
                effect="news_cycle",
                effect_params={"steps": 8, "surge_prob_boost": 0.15, "condition": "trauma"},
            ),
        ]

    def on_outcome(self, patient_id: str, condition: str, outcome: str, step: int) -> None:
        """Called by environment when a patient outcome is determined."""
        if self._rng is None:
            return

        trigger = f"patient_{outcome}"

        for rule in self._rules:
            if rule.trigger != trigger:
                continue
            if rule.condition_filter is not None and rule.condition_filter != condition:
                continue

            if self._rng.random() < rule.probability:
                self._apply_effect(rule.effect, rule.effect_params, step)

    def check_overcrowding(self, active_patient_count: int) -> float:
        """Returns overcrowding modifier (1.0 or 1.2) based on active patient count."""
        overcrowded_threshold = 3
        modifier = 1.2

        if active_patient_count > overcrowded_threshold:
            if not self._overcrowding_active:
                self._overcrowding_active = True
                if self._callback:
                    self._callback("overcrowding_started", active_patient_count=active_patient_count)
            return modifier
        else:
            self._overcrowding_active = False
            return 1.0

    def tick(self) -> None:
        """Advance news cycle timers. Call once per environment step."""
        if self._news_cycle_steps_remaining > 0:
            self._news_cycle_steps_remaining -= 1
        if self._news_cycle_steps_remaining == 0:
            self._pending_surge_probability = max(0.0, self._pending_surge_probability - 0.10)

    def get_surge_probability(self) -> float:
        """Returns current additional surge probability from news cycles."""
        return self._pending_surge_probability

    def _apply_effect(self, effect: str, params: Dict[str, Any], step: int) -> None:
        if effect == "spawn_secondary":
            if self._callback:
                self._callback(
                    "spawn_secondary",
                    condition=params["secondary_condition"],
                    reason=params["reason"],
                    triggered_at_step=step,
                )
        elif effect == "news_cycle":
            self._news_cycle_steps_remaining = params["steps"]
            self._pending_surge_probability += params["surge_prob_boost"]
            if self._callback:
                self._callback(
                    "news_cycle",
                    message=f"News cycle: successful {params.get('condition', 'emergency')} save draws attention",
                    steps=params["steps"],
                )

    # Test helpers (match DisruptionEngine pattern)
    @property
    def overcrowding_modifier(self) -> float:
        return 1.2 if self._overcrowding_active else 1.0

    @property
    def news_cycle_steps_remaining(self) -> int:
        return self._news_cycle_steps_remaining

    @property
    def pending_surge_probability(self) -> float:
        return self._pending_surge_probability
