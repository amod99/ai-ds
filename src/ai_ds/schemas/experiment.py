"""Experiment results persisted independently of a planner implementation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass(slots=True)
class ExperimentResult:
    experiment_id: int | None
    parent_experiment: int | None
    model: str
    preprocessing: str
    features: list[str]
    cv_folds: int
    metric: str
    metric_direction: Literal["maximize", "minimize"]
    primary_score: float | None
    secondary_scores: dict[str, float] = field(default_factory=dict)
    fold_scores: list[float] = field(default_factory=list)
    runtime_seconds: float = 0.0
    status: Literal["success", "failure", "skipped"] = "success"
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ranking_score(self) -> float | None:
        if self.primary_score is None:
            return None
        return self.primary_score if self.metric_direction == "maximize" else -self.primary_score

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["ranking_score"] = self.ranking_score
        return result
