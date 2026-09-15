"""Conservative target, task, and metric inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

import pandas as pd

from ai_ds.config import RunConfig
from ai_ds.data.leakage import find_target_leakage
from ai_ds.schemas.dataset import DatasetProfile

TaskType = Literal["classification", "regression"]


class TargetAmbiguityError(ValueError):
    """Raised when automatic target inference is too weak to be trustworthy."""


@dataclass(slots=True)
class ProblemDefinition:
    target_column: str
    task_type: TaskType
    primary_metric: str
    secondary_metrics: list[str]
    metric_direction: Literal["maximize", "minimize"]
    problem_confidence: float
    target_cardinality: int
    target_missing_count: int
    class_distribution: dict[str, int] = field(default_factory=dict)
    leakage_risk_columns: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _choose_target(profile: DatasetProfile, config: RunConfig) -> tuple[str, float, list[str]]:
    if config.target_column:
        try:
            profile.get_column(config.target_column)
        except KeyError as exc:
            available = ", ".join(column.name for column in profile.column_profiles)
            raise ValueError(
                f"Target {config.target_column!r} does not exist. Available: {available}"
            ) from exc
        return config.target_column, 1.0, []
    if not profile.target_candidates:
        raise TargetAmbiguityError("No non-constant target candidate was found; pass --target.")
    candidate = profile.target_candidates[0]
    if candidate.score < 0.65:
        hint = ", ".join(item.column for item in profile.target_candidates[:3]) or "none"
        raise TargetAmbiguityError(
            "Target inference is intentionally conservative (confidence "
            f"{candidate.score:.2f}). Pass --target. Possible columns: {hint}."
        )
    return candidate.column, candidate.score, candidate.reasons


def _infer_task(target: pd.Series, configured: TaskType | None) -> TaskType:
    if configured:
        return configured
    observed = target.dropna()
    cardinality = observed.nunique(dropna=True)
    if pd.api.types.is_bool_dtype(observed) or pd.api.types.is_object_dtype(observed):
        return "classification"
    if isinstance(observed.dtype, pd.CategoricalDtype):
        return "classification"
    # Integer-valued, low-cardinality targets are usually class labels. This threshold
    # avoids treating typical continuous regression targets as categorical.
    if cardinality <= 20 and cardinality / max(len(observed), 1) <= 0.20:
        return "classification"
    return "regression"


def _metrics(
    task_type: TaskType, target: pd.Series, configured: str | None
) -> tuple[str, list[str], str]:
    if configured:
        direction = (
            "minimize" if configured in {"rmse", "mae", "mean_squared_error"} else "maximize"
        )
        return configured, [], direction
    if task_type == "regression":
        return "rmse", ["mae", "r2"], "minimize"
    class_count = target.dropna().nunique(dropna=True)
    if class_count == 2:
        return "roc_auc", ["f1", "accuracy"], "maximize"
    return "roc_auc_ovr_weighted", ["f1_weighted", "accuracy"], "maximize"


def solve_problem(
    frame: pd.DataFrame, profile: DatasetProfile, config: RunConfig
) -> ProblemDefinition:
    """Resolve a supervised learning problem or ask the caller for an explicit target."""

    target_name, confidence, reasons = _choose_target(profile, config)
    target = frame[target_name]
    if target.dropna().nunique(dropna=True) < 2:
        raise ValueError(f"Target {target_name!r} has fewer than two observed values")
    task_type = _infer_task(target, config.task_type)
    metric, secondary, direction = _metrics(task_type, target, config.metric)
    warnings = list(profile.warnings)
    if reasons:
        warnings.append(f"Target inferred from: {', '.join(reasons)}")
    target_missing = int(target.isna().sum())
    if target_missing:
        warnings.append(
            f"{target_missing} row(s) with missing target will be excluded from training"
        )
    class_distribution: dict[str, int] = {}
    if task_type == "classification":
        counts = target.dropna().value_counts()
        class_distribution = {str(label): int(count) for label, count in counts.items()}
        smallest = counts.min()
        if smallest / counts.sum() < 0.20:
            warnings.append(
                "Class imbalance detected; compare class_weight='balanced' where supported"
            )
    leakage = find_target_leakage(frame, target_name)
    if leakage:
        warnings.append(f"Obvious leakage risk detected in: {', '.join(leakage)}")
    return ProblemDefinition(
        target_column=target_name,
        task_type=task_type,
        primary_metric=metric,
        secondary_metrics=secondary,
        metric_direction=direction,  # type: ignore[arg-type]
        problem_confidence=confidence,
        target_cardinality=int(target.dropna().nunique(dropna=True)),
        target_missing_count=target_missing,
        class_distribution=class_distribution,
        leakage_risk_columns=leakage,
        warnings=warnings,
    )
