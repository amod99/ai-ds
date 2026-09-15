from __future__ import annotations

import pandas as pd
import pytest

from ai_ds.config import RunConfig
from ai_ds.data.profiler import profile_dataset
from ai_ds.problem.solver import TargetAmbiguityError, solve_problem


def test_solver_infers_named_binary_classification_target(classification_frame):
    profile = profile_dataset(classification_frame)
    problem = solve_problem(classification_frame, profile, RunConfig())

    assert problem.target_column == "churn"
    assert problem.task_type == "classification"
    assert problem.primary_metric == "roc_auc"
    assert problem.problem_confidence >= 0.65


def test_solver_uses_explicit_regression_target():
    frame = pd.DataFrame({"feature": range(30), "value": [index * 1.5 for index in range(30)]})
    profile = profile_dataset(frame)
    problem = solve_problem(frame, profile, RunConfig(target_column="value"))

    assert problem.task_type == "regression"
    assert problem.primary_metric == "rmse"


def test_solver_refuses_weak_target_inference():
    frame = pd.DataFrame({"a": range(30), "b": range(30, 60)})
    profile = profile_dataset(frame)

    with pytest.raises(TargetAmbiguityError):
        solve_problem(frame, profile, RunConfig())
