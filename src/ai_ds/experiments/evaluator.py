"""Fixed cross-validation protocol and task-aware metric calculation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate

from ai_ds.problem.solver import ProblemDefinition

_SCORERS = {
    "roc_auc": "roc_auc",
    "roc_auc_ovr_weighted": "roc_auc_ovr_weighted",
    "f1": "f1",
    "f1_weighted": "f1_weighted",
    "accuracy": "accuracy",
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "r2": "r2",
}
_NEGATED = {"rmse", "mae"}


@dataclass(slots=True)
class Evaluation:
    primary_score: float
    secondary_scores: dict[str, float]
    fold_scores: list[float]


def _scoring(problem: ProblemDefinition, target: pd.Series) -> dict[str, object]:
    metrics = [problem.primary_metric, *problem.secondary_metrics]
    missing = [metric for metric in metrics if metric not in _SCORERS]
    if missing:
        raise ValueError(f"Unsupported metric(s): {', '.join(missing)}")
    scorers: dict[str, object] = {metric: _SCORERS[metric] for metric in metrics}
    # sklearn's built-in binary F1 scorer assumes positive label ``1``. Datasets often
    # carry human labels such as "yes"/"no", so choose the deterministic final class.
    if "f1" in scorers:
        labels = sorted(pd.Series(target).dropna().unique().tolist())
        if len(labels) != 2:
            raise ValueError("Binary F1 requires exactly two observed target classes")
        scorers["f1"] = make_scorer(f1_score, pos_label=labels[-1])
    return scorers


def _actual_scores(metric: str, values: np.ndarray) -> np.ndarray:
    return -values if metric in _NEGATED else values


def _splitter(problem: ProblemDefinition, y: pd.Series, folds: int, seed: int):
    if len(y) < folds:
        raise ValueError(f"CV requires at least {folds} rows; dataset has {len(y)}")
    if problem.task_type == "classification":
        smallest_class = int(y.value_counts().min())
        if smallest_class < folds:
            raise ValueError(
                f"CV requires at least {folds} examples in each class; smallest class has {smallest_class}"
            )
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return KFold(n_splits=folds, shuffle=True, random_state=seed)


def evaluate_pipeline(
    pipeline: object,
    features: pd.DataFrame,
    target: pd.Series,
    problem: ProblemDefinition,
    folds: int,
    seed: int,
) -> Evaluation:
    """Evaluate an unfitted pipeline without ever using train scores for selection."""

    scoring = _scoring(problem, target)
    cv = _splitter(problem, target, folds, seed)
    scores = cross_validate(
        pipeline,
        features,
        target,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        return_train_score=False,
        error_score="raise",
    )
    primary_folds = _actual_scores(
        problem.primary_metric, np.asarray(scores[f"test_{problem.primary_metric}"])
    )
    secondary: dict[str, float] = {}
    for metric in problem.secondary_metrics:
        values = _actual_scores(metric, np.asarray(scores[f"test_{metric}"]))
        secondary[metric] = float(np.mean(values))
    return Evaluation(
        primary_score=float(np.mean(primary_folds)),
        secondary_scores=secondary,
        fold_scores=[float(value) for value in primary_folds],
    )
