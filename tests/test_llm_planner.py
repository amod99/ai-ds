from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from ai_ds.agent.actions import experiment_action_json_schema
from ai_ds.agent.planner import LLMPlanner, PlannerResponseError
from ai_ds.agent.state import AgentState
from ai_ds.cli import build_parser
from ai_ds.config import RunConfig
from ai_ds.data.profiler import profile_dataset
from ai_ds.problem.solver import solve_problem


def _baseline_payload() -> dict[str, Any]:
    return {
        "action": "train_baseline",
        "model": None,
        "preprocessing": "baseline",
        "features": "all_valid_features",
        "drop_features": [],
        "class_weight": None,
        "metric": None,
        "hypothesis": "A dummy model establishes the validation floor.",
        "reason": "No prior experiment exists.",
    }


@dataclass
class FakeUsage:
    input_tokens: int = 100
    output_tokens: int = 40

    def model_dump(self) -> dict[str, int]:
        return {"input_tokens": self.input_tokens, "output_tokens": self.output_tokens}


class FakeResponse:
    def __init__(self, payload: dict[str, Any] | str, index: int) -> None:
        self.output_text = payload if isinstance(payload, str) else json.dumps(payload)
        self.id = f"resp_{index}"
        self.model = "test-model"
        self.status = "completed"
        self.usage = FakeUsage()


class FakeResponses:
    def __init__(self, payloads: list[dict[str, Any] | str | Exception]) -> None:
        self.payloads = list(payloads)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> FakeResponse:
        self.calls.append(kwargs)
        payload = self.payloads.pop(0)
        if isinstance(payload, Exception):
            raise payload
        return FakeResponse(payload, len(self.calls))


class FakeClient:
    def __init__(self, payloads: list[dict[str, Any] | str | Exception]) -> None:
        self.responses = FakeResponses(payloads)


def _state(classification_frame) -> AgentState:
    config = RunConfig(target_column="churn")
    profile = profile_dataset(classification_frame)
    problem = solve_problem(classification_frame, profile, config)
    return AgentState(
        profile=profile,
        problem=problem,
        remaining_experiments=5,
        remaining_runtime_seconds=300,
    )


def test_llm_planner_calls_responses_api_with_strict_schema(classification_frame):
    client = FakeClient([_baseline_payload()])
    planner = LLMPlanner(model="test-model", client=client, max_retries=0)

    action = planner.choose(_state(classification_frame))

    assert action.action == "train_baseline"
    request = client.responses.calls[0]
    assert request["model"] == "test-model"
    assert request["store"] is False
    assert request["text"]["format"]["type"] == "json_schema"
    assert request["text"]["format"]["strict"] is True
    assert "C000" not in request["input"]  # raw row values never enter compact state
    assert planner.last_decision_metadata["response_id"] == "resp_1"
    assert planner.last_decision_metadata["usage"]["input_tokens"] == 100


def test_llm_planner_retries_semantically_invalid_first_action(classification_frame, monkeypatch):
    invalid_first_action = _baseline_payload() | {
        "action": "train_model",
        "model": "logistic_regression",
    }
    client = FakeClient([invalid_first_action, _baseline_payload()])
    monkeypatch.setattr("ai_ds.agent.planner.time.sleep", lambda _: None)
    planner = LLMPlanner(model="test-model", client=client, max_retries=1)

    action = planner.choose(_state(classification_frame))

    assert action.action == "train_baseline"
    assert len(client.responses.calls) == 2
    assert "previous structured decision failed" in client.responses.calls[1]["input"]
    assert planner.last_decision_metadata["attempt_count"] == 2
    assert planner.last_decision_metadata["validation_errors"]


def test_llm_planner_fails_closed_after_retry_budget(classification_frame):
    client = FakeClient(["not valid json"])
    planner = LLMPlanner(model="test-model", client=client, max_retries=0)

    with pytest.raises(PlannerResponseError, match="failed after 1 attempt"):
        planner.choose(_state(classification_frame))


def test_action_schema_is_strict_and_task_specific():
    schema = experiment_action_json_schema("regression")

    assert schema["additionalProperties"] is False
    model_options = schema["properties"]["model"]["anyOf"][0]["enum"]
    assert "ridge" in model_options
    assert "logistic_regression" not in model_options


def test_run_cli_defaults_to_real_llm_planner():
    args = build_parser().parse_args(["run", "dataset.csv"])

    assert args.planner == "llm"
    assert args.llm_model
