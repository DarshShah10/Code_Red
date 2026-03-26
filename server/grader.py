"""Rubric grader stub — full implementation in Plan 3 Task 14."""
from dataclasses import dataclass

@dataclass
class RubricResult:
    time_score: float = 1.0
    efficiency: float = 1.0
    secondary_harm: float = 1.0
    prep_ready: float = 1.0
    mutual_aid_penalty: float = 0.0
    final_score: float = 1.0
    breakdown: dict = None

    def __post_init__(self):
        if self.breakdown is None:
            self.breakdown = {}
