"""Deterministic execution of validated model-training actions."""

from __future__ import annotations

import time
from dataclasses import dataclass

import pandas as pd

from ai_ds.config import RunConfig
from ai_ds.data.profiler import DatasetProfile
from ai_ds.experiments.evaluator import evaluate_pipeline
from ai_ds.models.registry import supported_models
from ai_ds.models.trainer import build_pipeline
from ai_ds.problem.solver import ProblemDefinition
from ai_ds.schemas.agent import AgentAction
from ai_ds.schemas.experiment import ExperimentResult


class ExperimentNotExecutable(ValueError):
    """Raised for valid planner actions that do not start a model evaluation."""


@dataclass(slots=True)
class PreparedData:
    features: pd.DataFrame
    target: pd.Series
    removed_features: list[str]


class ExperimentEngine:
    """Run deterministic CV experiments independently of any planner or LLM."""

    def __init__(self, config: RunConfig) -> None:
        self.config = config

    def prepare_data(
        self,
        frame: pd.DataFrame,
        profile: DatasetProfile,
        problem: ProblemDefinition,
        action: AgentAction | None = None,
    ) -> PreparedData:
        trainable = frame.loc[frame[problem.target_column].notna()].copy()
        target = trainable.pop(problem.target_column)
        remove = set(profile.constant_columns)
        if self.config.remove_id_columns:
            remove.update(profile.suspicious_id_columns)
        if self.config.remove_leakage_risks:
            remove.update(problem.leakage_risk_columns)
        if action:
            if problem.target_column in action.drop_features:
                raise ValueError("The target column cannot be dropped")
            unknown = sorted(set(action.drop_features) - set(trainable.columns))
            if unknown:
                raise ValueError(f"Cannot drop unknown feature(s): {', '.join(unknown)}")
            remove.update(action.drop_features)
        retained = [column for column in trainable.columns if column not in remove]
        if not retained:
            raise ValueError("No features remain after applying quality and safety filters")
        return PreparedData(
            trainable[retained], target, sorted(remove.intersection(trainable.columns))
        )

    def execute(
        self,
        frame: pd.DataFrame,
        profile: DatasetProfile,
        problem: ProblemDefinition,
        action: AgentAction,
        parent_experiment: int | None = None,
    ) -> ExperimentResult:
        """Run a single validated training experiment and return serializable results."""

        if action.action not in {"train_baseline", "train_model"}:
            raise ExperimentNotExecutable(
                f"Action {action.action!r} does not execute a model experiment"
            )
        model_name = action.model
        if action.action == "train_baseline":
            model_name = (
                "dummy_classifier" if problem.task_type == "classification" else "dummy_regressor"
            )
        if model_name not in supported_models(problem.task_type):
            raise ValueError(f"Model {model_name!r} is invalid for {problem.task_type}")
        prepared = self.prepare_data(frame, profile, problem, action)
        started = time.monotonic()
        pipeline = build_pipeline(
            prepared.features, action, problem.task_type, self.config.random_seed
        )
        evaluation = evaluate_pipeline(
            pipeline,
            prepared.features,
            prepared.target,
            problem,
            self.config.cv_folds,
            self.config.random_seed,
        )
        return ExperimentResult(
            experiment_id=None,
            parent_experiment=parent_experiment,
            model=str(model_name),
            preprocessing=action.preprocessing,
            features=list(prepared.features.columns),
            cv_folds=self.config.cv_folds,
            metric=problem.primary_metric,
            metric_direction=problem.metric_direction,
            primary_score=evaluation.primary_score,
            secondary_scores=evaluation.secondary_scores,
            fold_scores=evaluation.fold_scores,
            runtime_seconds=time.monotonic() - started,
            status="success",
            metadata={"removed_features": prepared.removed_features},
        )

    def fit_full_data(
        self,
        frame: pd.DataFrame,
        profile: DatasetProfile,
        problem: ProblemDefinition,
        action: AgentAction,
    ) -> object:
        """Refit the selected pipeline on all rows with an observed target."""

        prepared = self.prepare_data(frame, profile, problem, action)
        pipeline = build_pipeline(
            prepared.features, action, problem.task_type, self.config.random_seed
        )
        return pipeline.fit(prepared.features, prepared.target)
