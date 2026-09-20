"""Deterministic and OpenAI-backed planners over a constrained action schema."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any, Protocol

from ai_ds.agent.actions import experiment_action_json_schema, validate_action
from ai_ds.agent.prompts import PLANNER_INSTRUCTIONS, planner_input
from ai_ds.agent.state import AgentState
from ai_ds.schemas.agent import AgentAction

DEFAULT_LLM_MODEL = "gpt-5.6-luna"


class PlannerError(RuntimeError):
    """Base error for planner configuration, transport, or response failures."""


class PlannerConfigurationError(PlannerError):
    """The OpenAI planner cannot be initialized from local configuration."""


class PlannerResponseError(PlannerError):
    """The model did not produce a usable action within the retry budget."""


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


class CallbackPlanner:
    """Provider-neutral callback adapter retained for embedding and test harnesses."""

    def __init__(self, decision_fn: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
        self.decision_fn = decision_fn
        self.last_decision_metadata: dict[str, Any] = {}

    def choose(self, state: AgentState) -> AgentAction:
        payload = self.decision_fn(state.compact())
        if not isinstance(payload, dict):
            raise TypeError("Planner callback must return a JSON object")
        action = validate_action(payload, state.problem.task_type)
        self.last_decision_metadata = {"provider": "callback", "payload": payload}
        return action


class LLMPlanner:
    """OpenAI Responses API planner using strict Structured Outputs.

    Only :meth:`AgentState.compact` is sent to the model. The model receives no tools
    and cannot execute code; its JSON decision is schema-validated, semantically checked,
    and then handed to the deterministic experiment engine.
    """

    def __init__(
        self,
        model: str = DEFAULT_LLM_MODEL,
        *,
        reasoning_effort: str | None = "low",
        max_output_tokens: int = 2_000,
        max_retries: int = 2,
        timeout_seconds: float = 60.0,
        client: Any | None = None,
    ) -> None:
        if not model.strip():
            raise PlannerConfigurationError("LLM model name cannot be empty")
        if max_output_tokens < 256:
            raise PlannerConfigurationError("LLM max_output_tokens must be at least 256")
        if max_retries < 0:
            raise PlannerConfigurationError("LLM max_retries cannot be negative")
        if timeout_seconds <= 0:
            raise PlannerConfigurationError("LLM timeout_seconds must be positive")
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.client = client if client is not None else self._create_client()
        self.last_decision_metadata: dict[str, Any] = {}

    def _create_client(self) -> Any:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise PlannerConfigurationError(
                "The OpenAI SDK is not installed. Install the project again with `pip install -e .`."
            ) from exc
        try:
            # SDK retries are disabled because every planner retry must remain visible.
            return OpenAI(timeout=self.timeout_seconds, max_retries=0)
        except Exception as exc:
            raise PlannerConfigurationError(
                "Could not initialize the OpenAI client. Set OPENAI_API_KEY and retry. "
                f"Provider error: {type(exc).__name__}: {exc}"
            ) from exc

    @staticmethod
    def _serializable(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool, list, dict)):
            return value
        if hasattr(value, "model_dump"):
            return value.model_dump()
        return str(value)

    @staticmethod
    def _action_signature(action: AgentAction, task_type: str) -> tuple[Any, ...]:
        model = action.model
        if action.action == "train_baseline":
            model = "dummy_classifier" if task_type == "classification" else "dummy_regressor"
        return (
            model,
            action.preprocessing,
            action.class_weight,
            tuple(sorted(action.drop_features)),
        )

    def _validate_for_state(self, action: AgentAction, state: AgentState) -> None:
        if not state.experiments and action.action != "train_baseline":
            raise ValueError("The first action must be train_baseline")
        if state.experiments and action.action == "train_baseline":
            raise ValueError("The baseline has already been established")
        if action.action == "stop":
            return
        if action.metric not in (None, state.problem.primary_metric):
            raise ValueError(
                f"The fixed protocol metric is {state.problem.primary_metric!r}; "
                "the planner cannot change it"
            )
        if action.class_weight and action.model not in {"logistic_regression", "random_forest"}:
            raise ValueError(
                "class_weight is only supported for logistic_regression or random_forest"
            )
        known_features = {
            column.name
            for column in state.profile.column_profiles
            if column.name != state.problem.target_column
        }
        unknown_drops = sorted(set(action.drop_features) - known_features)
        if unknown_drops:
            raise ValueError(f"Unknown drop_features: {', '.join(unknown_drops)}")
        prior_signatures = {
            self._action_signature(record.action, state.problem.task_type)
            for record in state.experiments
        }
        if self._action_signature(action, state.problem.task_type) in prior_signatures:
            raise ValueError(
                "This model/preprocessing/class-weight/drop combination was already tried"
            )

    def _request(self, state: AgentState, correction: str | None) -> tuple[dict[str, Any], Any]:
        request: dict[str, Any] = {
            "model": self.model,
            "instructions": PLANNER_INSTRUCTIONS,
            "input": planner_input(state.compact(), correction),
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "experiment_action",
                    "strict": True,
                    "schema": experiment_action_json_schema(state.problem.task_type),
                }
            },
            "max_output_tokens": self.max_output_tokens,
            "prompt_cache_key": "ai-ds-experiment-planner-v1",
            "store": False,
        }
        if self.reasoning_effort is not None:
            request["reasoning"] = {"effort": self.reasoning_effort}
        response = self.client.responses.create(**request)
        status = getattr(response, "status", "completed")
        if status != "completed":
            details = self._serializable(getattr(response, "incomplete_details", None))
            raise ValueError(f"OpenAI response status was {status!r}: {details}")
        output_text = getattr(response, "output_text", None)
        if not isinstance(output_text, str) or not output_text.strip():
            raise ValueError("OpenAI response did not contain output_text")
        payload = json.loads(output_text)
        if not isinstance(payload, dict):
            raise TypeError("OpenAI structured output must be a JSON object")
        return payload, response

    def choose(self, state: AgentState) -> AgentAction:
        errors: list[str] = []
        for attempt in range(1, self.max_retries + 2):
            response: Any | None = None
            payload: dict[str, Any] | None = None
            try:
                correction = errors[-1] if errors else None
                payload, response = self._request(state, correction)
                action = validate_action(payload, state.problem.task_type)
                self._validate_for_state(action, state)
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                errors.append(message)
                if attempt > self.max_retries:
                    self.last_decision_metadata = {
                        "provider": "openai",
                        "configured_model": self.model,
                        "attempt_count": attempt,
                        "errors": errors,
                        "last_payload": payload,
                        "response_id": getattr(response, "id", None),
                    }
                    raise PlannerResponseError(
                        f"LLM planner failed after {attempt} attempt(s): {message}"
                    ) from exc
                time.sleep(min(2 ** (attempt - 1), 4))
                continue

            self.last_decision_metadata = {
                "provider": "openai",
                "configured_model": self.model,
                "response_model": getattr(response, "model", self.model),
                "response_id": getattr(response, "id", None),
                "response_status": getattr(response, "status", None),
                "attempt_count": attempt,
                "validation_errors": errors,
                "usage": self._serializable(getattr(response, "usage", None)),
                "payload": payload,
                "store": False,
            }
            return action
        raise AssertionError("Unreachable planner retry state")
