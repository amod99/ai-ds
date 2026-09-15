"""A narrow, validated action vocabulary for planners."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

ActionType = Literal[
    "profile_data",
    "check_target",
    "train_baseline",
    "train_model",
    "transform_feature",
    "drop_feature",
    "change_class_weight",
    "change_metric",
    "compare_experiments",
    "stop",
]

_ALLOWED_ACTIONS = {
    "profile_data",
    "check_target",
    "train_baseline",
    "train_model",
    "transform_feature",
    "drop_feature",
    "change_class_weight",
    "change_metric",
    "compare_experiments",
    "stop",
}
_ALLOWED_PREPROCESSING = {"baseline", "frequency", "hash", "target", "log_numeric"}


@dataclass(slots=True)
class AgentAction:
    action: ActionType
    model: str | None = None
    preprocessing: str = "baseline"
    features: str = "all_valid_features"
    drop_features: list[str] = field(default_factory=list)
    class_weight: str | None = None
    metric: str | None = None
    hypothesis: str = ""
    reason: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any], task_type: str) -> AgentAction:
        """Parse untrusted planner output and reject fields outside the schema."""

        allowed = {
            "action",
            "model",
            "preprocessing",
            "features",
            "drop_features",
            "class_weight",
            "metric",
            "hypothesis",
            "reason",
        }
        unexpected = set(payload) - allowed
        if unexpected:
            raise ValueError(f"Planner action has unsupported fields: {sorted(unexpected)!r}")
        if payload.get("action") not in _ALLOWED_ACTIONS:
            raise ValueError(f"Planner action is not allowlisted: {payload.get('action')!r}")
        action = cls(**payload)
        action.validate(task_type)
        return action

    def validate(self, task_type: str) -> None:
        if self.preprocessing not in _ALLOWED_PREPROCESSING:
            raise ValueError(f"Unsupported preprocessing option: {self.preprocessing!r}")
        if self.action == "train_model" and not self.model:
            raise ValueError("train_model requires a model")
        if self.action == "drop_feature" and not self.drop_features:
            raise ValueError("drop_feature requires at least one feature name")
        if self.class_weight not in (None, "balanced"):
            raise ValueError("class_weight may only be 'balanced'")
        if task_type == "regression" and self.class_weight:
            raise ValueError("class_weight is not valid for regression")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AgentDecision:
    action: AgentAction
    observation: str = ""
    conclusion: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.to_dict(),
            "observation": self.observation,
            "conclusion": self.conclusion,
        }
