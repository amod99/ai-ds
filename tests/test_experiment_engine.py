from __future__ import annotations

from ai_ds.config import RunConfig
from ai_ds.data.profiler import profile_dataset
from ai_ds.experiments.engine import ExperimentEngine
from ai_ds.problem.solver import solve_problem
from ai_ds.schemas.agent import AgentAction


def test_experiment_engine_evaluates_pipeline_without_id_features(classification_frame):
    config = RunConfig(target_column="churn", cv_folds=5)
    profile = profile_dataset(classification_frame)
    problem = solve_problem(classification_frame, profile, config)
    result = ExperimentEngine(config).execute(
        classification_frame,
        profile,
        problem,
        AgentAction(action="train_model", model="logistic_regression"),
    )

    assert result.status == "success"
    assert result.primary_score is not None
    assert len(result.fold_scores) == 5
    assert "customer_id" not in result.features
