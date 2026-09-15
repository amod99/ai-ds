"""Rule-based and callback-backed planners over a constrained action schema."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from ai_ds.agent.actions import validate_action
from ai_ds.agent.state import AgentState
from ai_ds.schemas.agent import AgentAction


class Planner(Protocol):
    def choose(self, state: AgentState) -> AgentAction:
        """Choose one validated action from compact state."""


class RuleBasedPlanner:
    """A deterministic research loop used as a safe V1 planner baseline."""

    def choose(self, state: AgentState) -> AgentAction:
        tried = {
            (item.result.model, item.result.preprocessing, item.action.class_weight)
            for item in state.experiments
        }
        if not state.experiments:
            return AgentAction(
                action="train_baseline",
                hypothesis="A simple prior baseline establishes the minimum useful validation score.",
                reason="Every experiment sequence starts with a reproducible performance floor.",
            )
        if state.problem.task_type == "classification":
            candidates = [
                (
                    "logistic_regression",
                    "baseline",
                    None,
                    "A regularized linear decision boundary is a strong tabular baseline.",
                ),
                (
                    "random_forest",
                    "baseline",
                    None,
                    "Bagged trees can capture interactions missed by a linear model.",
                ),
                (
                    "hist_gradient_boosting",
                    "baseline",
                    None,
                    "Boosted trees test for smooth nonlinear effects.",
                ),
                (
                    "hist_gradient_boosting",
                    "frequency",
                    None,
                    "Frequency encoding may handle high-cardinality categoricals more compactly.",
                ),
                (
                    "random_forest",
                    "target",
                    None,
                    "Fold-fitted target encoding may expose categorical signal without validation leakage.",
                ),
                (
                    "hist_gradient_boosting",
                    "log_numeric",
                    None,
                    "A sign-preserving log transform tests sensitivity to skewed numeric magnitudes.",
                ),
            ]
            if state.problem.class_distribution:
                candidates.insert(
                    1,
                    (
                        "logistic_regression",
                        "baseline",
                        "balanced",
                        "Balanced class weights test whether minority recall improves.",
                    ),
                )
        else:
            candidates = [
                (
                    "linear_regression",
                    "baseline",
                    None,
                    "A linear regression baseline tests for an approximately additive relationship.",
                ),
                (
                    "ridge",
                    "baseline",
                    None,
                    "Regularization may stabilize correlated tabular features.",
                ),
                ("random_forest", "baseline", None, "Bagged trees test nonlinear interactions."),
                (
                    "hist_gradient_boosting",
                    "baseline",
                    None,
                    "Boosted trees often provide a strong nonlinear tabular comparison.",
                ),
                (
                    "hist_gradient_boosting",
                    "frequency",
                    None,
                    "Frequency encoding may improve high-cardinality categorical handling.",
                ),
                (
                    "hist_gradient_boosting",
                    "log_numeric",
                    None,
                    "A sign-preserving log transform tests robustness to skewed numeric predictors.",
                ),
            ]
        previous = state.experiments[-1].result
        previous_score = (
            f"{previous.primary_score:.5f}"
            if previous.primary_score is not None
            else "no validation score"
        )
        for model, preprocessing, class_weight, hypothesis in candidates:
            if (model, preprocessing, class_weight) not in tried:
                return AgentAction(
                    action="train_model",
                    model=model,
                    preprocessing=preprocessing,
                    class_weight=class_weight,
                    hypothesis=hypothesis,
                    reason=(
                        f"The prior experiment scored {previous_score} on "
                        f"{previous.metric}; this is the next non-redundant controlled comparison."
                    ),
                )
        return AgentAction(
            action="stop",
            hypothesis="The bounded candidate catalog has been covered.",
            reason="No additional non-redundant V1 experiment remains.",
        )


class FixedPlanner(RuleBasedPlanner):
    """Alias for the deterministic non-agentic `train` command sequence."""


class LLMPlanner:
    """Adapter for an application-provided structured-output LLM call.

    ``decision_fn`` receives only :meth:`AgentState.compact` output. The caller owns
    provider credentials and transport; this package never grants an LLM tool access.
    """

    def __init__(self, decision_fn: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
        self.decision_fn = decision_fn

    def choose(self, state: AgentState) -> AgentAction:
        payload = self.decision_fn(state.compact())
        if not isinstance(payload, dict):
            raise TypeError("LLM planner must return a JSON object")
        return validate_action(payload, state.problem.task_type)
