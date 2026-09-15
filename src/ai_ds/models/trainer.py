"""Construct full preprocess-then-model pipelines for execution by the evaluator."""

from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline

from ai_ds.models.registry import build_model
from ai_ds.preprocessing.pipeline import PreprocessingConfig, build_preprocessor
from ai_ds.schemas.agent import AgentAction


def build_pipeline(
    features: pd.DataFrame,
    action: AgentAction,
    task_type: str,
    random_seed: int,
) -> Pipeline:
    """Build one deterministic, unfitted sklearn pipeline from a validated action."""

    model_name = action.model
    if action.action == "train_baseline":
        model_name = "dummy_classifier" if task_type == "classification" else "dummy_regressor"
    if not model_name:
        raise ValueError("A train action must resolve to a model")
    scaled = model_name in {"logistic_regression", "linear_regression", "ridge"}
    preprocess_config = PreprocessingConfig.from_action_name(
        action.preprocessing, scale_numeric=scaled
    )
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(features, preprocess_config)),
            ("model", build_model(model_name, task_type, random_seed, action.class_weight)),
        ]
    )
