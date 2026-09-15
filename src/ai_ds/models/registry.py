"""Allowlisted sklearn model constructors; no dynamic imports or code execution."""

from __future__ import annotations

from typing import Any

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge

_CLASSIFIERS = {
    "dummy_classifier",
    "logistic_regression",
    "random_forest",
    "hist_gradient_boosting",
}
_REGRESSORS = {
    "dummy_regressor",
    "linear_regression",
    "ridge",
    "random_forest",
    "hist_gradient_boosting",
}


def supported_models(task_type: str) -> set[str]:
    if task_type == "classification":
        return set(_CLASSIFIERS)
    if task_type == "regression":
        return set(_REGRESSORS)
    raise ValueError(f"Unsupported task type: {task_type}")


def build_model(
    name: str, task_type: str, random_seed: int, class_weight: str | None = None
) -> Any:
    """Construct a V1 model from a fixed registry and safe, bounded defaults."""

    if name not in supported_models(task_type):
        raise ValueError(f"Model {name!r} is not available for {task_type}")
    if task_type == "classification":
        if name == "dummy_classifier":
            return DummyClassifier(strategy="prior")
        if name == "logistic_regression":
            return LogisticRegression(
                max_iter=2_000, random_state=random_seed, class_weight=class_weight, solver="lbfgs"
            )
        if name == "random_forest":
            return RandomForestClassifier(
                n_estimators=250,
                min_samples_leaf=2,
                random_state=random_seed,
                n_jobs=-1,
                class_weight=class_weight,
            )
        if name == "hist_gradient_boosting":
            if class_weight:
                raise ValueError("hist_gradient_boosting does not support class_weight in AI-DS V1")
            return HistGradientBoostingClassifier(max_iter=250, random_state=random_seed)
    else:
        if class_weight:
            raise ValueError("Regression models do not support class_weight")
        if name == "dummy_regressor":
            return DummyRegressor(strategy="mean")
        if name == "linear_regression":
            return LinearRegression()
        if name == "ridge":
            return Ridge(alpha=1.0, random_state=random_seed)
        if name == "random_forest":
            return RandomForestRegressor(
                n_estimators=250, min_samples_leaf=2, random_state=random_seed, n_jobs=-1
            )
        if name == "hist_gradient_boosting":
            return HistGradientBoostingRegressor(max_iter=250, random_state=random_seed)
    raise AssertionError(f"Missing model constructor for {name!r}")
