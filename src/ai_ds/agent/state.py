"""The compact, raw-data-free state provided to an agent planner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ai_ds.agent.actions import executable_models
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
        return max(
            successful,
            key=lambda record: (
                record.result.ranking_score
                if record.result.ranking_score is not None
                else float("-inf")
            ),
        )

    @staticmethod
    def _compact_experiment(record: ExperimentRecord) -> dict[str, Any]:
        result = record.result
        return {
            "experiment_id": result.experiment_id,
            "parent_experiment": result.parent_experiment,
            "action": record.action.to_dict(),
            "result": {
                "model": result.model,
                "preprocessing": result.preprocessing,
                "metric": result.metric,
                "metric_direction": result.metric_direction,
                "primary_score": result.primary_score,
                "secondary_scores": result.secondary_scores,
                "runtime_seconds": round(result.runtime_seconds, 3),
                "status": result.status,
                "error": result.error,
            },
            "conclusion": record.conclusion,
        }

    def compact(self) -> dict[str, Any]:
        """A deliberately limited planner view: statistics and history, not raw rows."""

        best = self.best
        feature_summaries = []
        for column in self.profile.column_profiles:
            if column.name == self.problem.target_column:
                continue
            feature_summaries.append(
                {
                    "name": column.name,
                    "semantic_type": column.semantic_type,
                    "missing_ratio": round(column.missing_ratio, 4),
                    "unique_count": column.unique_count,
                    "unique_ratio": round(column.unique_ratio, 4),
                    "numeric_summary": column.numeric_summary or None,
                }
            )
        max_feature_summaries = 100
        return {
            "dataset": {
                "rows": self.profile.rows,
                "columns": self.profile.columns,
                "feature_summaries": feature_summaries[:max_feature_summaries],
                "feature_summaries_truncated": len(feature_summaries) > max_feature_summaries,
                "constant_columns": self.profile.constant_columns,
                "suspicious_id_columns": self.profile.suspicious_id_columns,
                "strong_correlations": self.profile.correlations[:20],
                "warnings": self.profile.warnings,
            },
            "problem": self.problem.to_dict(),
            "experiments": [self._compact_experiment(record) for record in self.experiments],
            "best_experiment": self._compact_experiment(best) if best else None,
            "remaining_experiments": self.remaining_experiments,
            "remaining_runtime_seconds": round(self.remaining_runtime_seconds, 1),
            "available_actions": [
                "train_baseline",
                "train_model",
                "stop",
            ],
            "allowed_models": executable_models(self.problem.task_type),
            "allowed_preprocessing": ["baseline", "frequency", "hash", "target", "log_numeric"],
        }
