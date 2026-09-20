"""Action validation and strict structured-output schema generation."""

from __future__ import annotations

from typing import Any

from ai_ds.models.registry import supported_models
from ai_ds.schemas.agent import AgentAction


def validate_action(payload: dict[str, Any], task_type: str) -> AgentAction:
    """Validate untrusted structured planner output against the V1 action schema."""

    action = AgentAction.from_dict(payload, task_type)
    if action.model and action.model not in supported_models(task_type):
        raise ValueError(f"Model {action.model!r} is not available for {task_type}")
    if action.action in {"train_baseline", "stop"} and action.model is not None:
        raise ValueError(f"{action.action} must set model to null")
    return action


def executable_models(task_type: str) -> list[str]:
    """Return non-dummy models an LLM may select with ``train_model``."""

    return sorted(model for model in supported_models(task_type) if not model.startswith("dummy_"))


def experiment_action_json_schema(task_type: str) -> dict[str, Any]:
    """Build the strict JSON Schema sent to the Responses API.

    Structured Outputs guarantees the syntactic shape. ``validate_action`` and the
    planner's state-aware checks remain the semantic trust boundary.
    """

    models = executable_models(task_type)
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "action": {"type": "string", "enum": ["train_baseline", "train_model", "stop"]},
            "model": {"anyOf": [{"type": "string", "enum": models}, {"type": "null"}]},
            "preprocessing": {
                "type": "string",
                "enum": ["baseline", "frequency", "hash", "target", "log_numeric"],
            },
            "features": {"type": "string", "enum": ["all_valid_features"]},
            "drop_features": {"type": "array", "items": {"type": "string"}, "maxItems": 20},
            "class_weight": {"anyOf": [{"type": "string", "enum": ["balanced"]}, {"type": "null"}]},
            "metric": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "hypothesis": {"type": "string", "minLength": 1, "maxLength": 600},
            "reason": {"type": "string", "minLength": 1, "maxLength": 600},
        },
        "required": [
            "action",
            "model",
            "preprocessing",
            "features",
            "drop_features",
            "class_weight",
            "metric",
            "hypothesis",
            "reason",
        ],
    }
