from __future__ import annotations

import pytest

from ai_ds.schemas.agent import AgentAction


def test_action_schema_rejects_unknown_fields_and_invalid_models():
    with pytest.raises(ValueError, match="unsupported fields"):
        AgentAction.from_dict(
            {"action": "train_model", "model": "random_forest", "code": "import os"},
            "classification",
        )

    with pytest.raises(ValueError, match="requires a model"):
        AgentAction(action="train_model").validate("classification")


def test_action_schema_allows_constrained_train_action():
    action = AgentAction.from_dict(
        {
            "action": "train_model",
            "model": "hist_gradient_boosting",
            "preprocessing": "frequency",
            "hypothesis": "Nonlinear structure may improve validation performance.",
        },
        "classification",
    )
    assert action.model == "hist_gradient_boosting"
