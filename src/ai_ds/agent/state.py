"""The compact, raw-data-free state provided to an agent planner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ai_ds.problem.solver import ProblemDefinition
from ai_ds.schemas.agent import AgentAction
from ai_ds.schemas.dataset import DatasetProfile
from ai_ds.schemas.experiment import ExperimentResult


@dataclass(slots=True)
class ExperimentRecord:
    action: AgentAction
    result: ExperimentResult
    conclusion: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.to_dict(),
            "result": self.result.to_dict(),
            "conclusion": self.conclusion,
        }


@dataclass(slots=True)
class AgentState:
    profile: DatasetProfile
    problem: ProblemDefinition
    experiments: list[ExperimentRecord] = field(default_factory=list)
    remaining_experiments: int = 0
    remaining_runtime_seconds: float = 0.0

    @property
    def best(self) -> ExperimentRecord | None:
        successful = [
            record for record in self.experiments if record.result.ranking_score is not None
        ]
        if not successful:
            return None
        return max(successful, key=lambda record: record.result.ranking_score or float("-inf"))

    def compact(self) -> dict[str, Any]:
        """A deliberately limited planner view: statistics and history, not raw rows."""

        best = self.best
        return {
            "dataset": {
                "rows": self.profile.rows,
                "columns": self.profile.columns,
                "constant_columns": self.profile.constant_columns,
                "suspicious_id_columns": self.profile.suspicious_id_columns,
                "warnings": self.profile.warnings,
            },
            "problem": self.problem.to_dict(),
            "experiments": [record.to_dict() for record in self.experiments],
            "best_experiment": best.to_dict() if best else None,
            "remaining_experiments": self.remaining_experiments,
            "remaining_runtime_seconds": round(self.remaining_runtime_seconds, 1),
            "available_actions": [
                "train_baseline",
                "train_model",
                "stop",
            ],
            "allowed_models": (
                [
                    "dummy_classifier",
                    "logistic_regression",
                    "random_forest",
                    "hist_gradient_boosting",
                ]
                if self.problem.task_type == "classification"
                else [
                    "dummy_regressor",
                    "linear_regression",
                    "ridge",
                    "random_forest",
                    "hist_gradient_boosting",
                ]
            ),
            "allowed_preprocessing": ["baseline", "frequency", "hash", "target", "log_numeric"],
        }
