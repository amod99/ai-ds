"""Run lifecycle, budgets, agent loop, final selection, and artifact production."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib

from ai_ds.agent.planner import Planner, PlannerError, RuleBasedPlanner
from ai_ds.agent.state import AgentState, ExperimentRecord
from ai_ds.config import RunConfig
from ai_ds.data.loader import load_csv
from ai_ds.data.profiler import profile_dataset
from ai_ds.experiments.engine import ExperimentEngine
from ai_ds.experiments.store import ExperimentStore
from ai_ds.problem.solver import solve_problem
from ai_ds.reporting.report import render_report
from ai_ds.schemas.experiment import ExperimentResult


@dataclass(slots=True)
class RunSummary:
    run_dir: Path
    report_path: Path
    model_path: Path | None
    best_result: ExperimentResult | None
    stop_reason: str
    experiments: list[ExperimentRecord]


class RunController:
    """Own a single bounded run while keeping planner and execution decoupled."""

    def __init__(self, config: RunConfig, planner: Planner | None = None) -> None:
        self.config = config
        self.planner = planner or RuleBasedPlanner()

    def inspect(self, dataset_path: str | Path):
        """Load and profile a CSV without formulating or training a model."""

        dataset = load_csv(dataset_path)
        return profile_dataset(dataset.frame, dataset.sha256)

    def _new_run_dir(self, dataset_hash: str) -> Path:
        root = Path(self.config.runs_dir).resolve()
        root.mkdir(parents=True, exist_ok=True)
        prefix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        candidate = root / f"{prefix}_{dataset_hash[:10]}"
        suffix = 2
        while candidate.exists():
            candidate = root / f"{prefix}_{dataset_hash[:10]}_{suffix}"
            suffix += 1
        return candidate

    @staticmethod
    def _best(records: list[ExperimentRecord]) -> ExperimentRecord | None:
        successful = [record for record in records if record.result.ranking_score is not None]
        return max(
            successful,
            key=lambda record: (
                record.result.ranking_score
                if record.result.ranking_score is not None
                else float("-inf")
            ),
            default=None,
        )

    def run(self, dataset_path: str | Path) -> RunSummary:
        """Execute profile → formulate → bounded experiments → final report."""

        started = time.monotonic()
        dataset = load_csv(dataset_path)
        profile = profile_dataset(dataset.frame, dataset.sha256)
        problem = solve_problem(dataset.frame, profile, self.config)
        store = ExperimentStore(self._new_run_dir(dataset.sha256))
        engine = ExperimentEngine(self.config)
        records: list[ExperimentRecord] = []
        failures = 0
        no_significant_improvement = 0
        stop_reason = "Unknown"
        model_path: Path | None = None
        final_model: object | None = None
        try:
            store.set_metadata("dataset_path", str(dataset.path))
            store.set_metadata("dataset_hash", dataset.sha256)
            store.set_metadata("config", self.config.serializable())
            store.set_metadata("profile", profile.to_dict())
            store.set_metadata("problem", problem.to_dict())
            store.set_metadata(
                "planner",
                {
                    "type": type(self.planner).__name__,
                    "model": getattr(self.planner, "model", None),
                    "reasoning_effort": getattr(self.planner, "reasoning_effort", None),
                },
            )
            store.record_event({"type": "run_started", "dataset_hash": dataset.sha256})

            while True:
                elapsed = time.monotonic() - started
                if len(records) >= self.config.max_experiments:
                    stop_reason = f"Experiment budget exhausted ({self.config.max_experiments})."
                    break
                if elapsed >= self.config.max_runtime_minutes * 60:
                    stop_reason = (
                        f"Runtime budget exhausted ({self.config.max_runtime_minutes:g} minutes)."
                    )
                    break
                if no_significant_improvement >= self.config.patience:
                    stop_reason = (
                        f"No validation improvement of at least {self.config.minimum_improvement:g} "
                        f"for {self.config.patience} experiments."
                    )
                    break
                if failures >= self.config.max_failures:
                    stop_reason = (
                        f"Execution failure threshold reached ({self.config.max_failures})."
                    )
                    break

                state = AgentState(
                    profile=profile,
                    problem=problem,
                    experiments=records,
                    remaining_experiments=self.config.max_experiments - len(records),
                    remaining_runtime_seconds=max(
                        0, self.config.max_runtime_minutes * 60 - elapsed
                    ),
                )
                try:
                    action = self.planner.choose(state)
                    action.validate(problem.task_type)
                except (PlannerError, TypeError, ValueError) as exc:
                    stop_reason = f"Planner failure: {type(exc).__name__}: {exc}"
                    store.record_event(
                        {
                            "type": "planner_failure",
                            "error": stop_reason,
                            "state": state.compact(),
                            "planner": getattr(self.planner, "last_decision_metadata", {}),
                        }
                    )
                    break
                store.record_event(
                    {
                        "type": "planner_decision",
                        "action": action.to_dict(),
                        "state": state.compact(),
                        "planner": getattr(self.planner, "last_decision_metadata", {}),
                    }
                )
                if action.action == "stop":
                    stop_reason = f"Planner stopped: {action.reason or action.hypothesis or 'no reason supplied'}"
                    store.record_event({"type": "planner_stop", "action": action.to_dict()})
                    break
                if action.action not in {"train_baseline", "train_model"}:
                    stop_reason = (
                        "Planner selected a valid but non-executable action. V1 run mode accepts "
                        "train_baseline, train_model, or stop."
                    )
                    store.record_event(
                        {"type": "unsupported_planner_action", "action": action.to_dict()}
                    )
                    break

                prior_best = self._best(records)
                try:
                    result = engine.execute(
                        dataset.frame,
                        profile,
                        problem,
                        action,
                        parent_experiment=prior_best.result.experiment_id if prior_best else None,
                    )
                except Exception as exc:  # noqa: BLE001 - every executor failure must be recorded
                    failures += 1
                    result = ExperimentResult(
                        experiment_id=None,
                        parent_experiment=prior_best.result.experiment_id if prior_best else None,
                        model=action.model or action.action,
                        preprocessing=action.preprocessing,
                        features=[],
                        cv_folds=self.config.cv_folds,
                        metric=problem.primary_metric,
                        metric_direction=problem.metric_direction,
                        primary_score=None,
                        runtime_seconds=0.0,
                        status="failure",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    conclusion = "Execution failed; the failure is retained in the trajectory."
                else:
                    prior_score = prior_best.result.ranking_score if prior_best else None
                    current_score = result.ranking_score
                    if prior_score is None or (
                        current_score is not None and current_score > prior_score
                    ):
                        improvement = (current_score or 0.0) - (prior_score or 0.0)
                        conclusion = (
                            "New best validation result"
                            if prior_score is None
                            else f"New best validation result (ranking improvement {improvement:.5f})."
                        )
                    else:
                        conclusion = "Did not exceed the current best validation result."
                    if prior_score is None or (
                        current_score is not None
                        and current_score > prior_score + self.config.minimum_improvement
                    ):
                        no_significant_improvement = 0
                    else:
                        no_significant_improvement += 1

                result = store.record_experiment(action, result)
                records.append(
                    ExperimentRecord(action=action, result=result, conclusion=conclusion)
                )

            best = self._best(records)
            if best:
                try:
                    final_model = engine.fit_full_data(dataset.frame, profile, problem, best.action)
                    model_path = store.run_dir / "final_model.joblib"
                    joblib.dump(final_model, model_path)
                    store.record_event(
                        {
                            "type": "final_model_saved",
                            "experiment_id": best.result.experiment_id,
                            "model_path": str(model_path),
                        }
                    )
                except Exception as exc:  # noqa: BLE001 - report a failed final refit in the trajectory
                    store.record_event(
                        {"type": "final_model_failure", "error": f"{type(exc).__name__}: {exc}"}
                    )
                    stop_reason += f" Final-model refit failed: {type(exc).__name__}: {exc}."
            report_path = render_report(
                store.run_dir / "report.md", profile, problem, records, stop_reason, final_model
            )
            store.set_metadata("stop_reason", stop_reason)
            store.set_metadata("best_experiment_id", best.result.experiment_id if best else None)
            store.record_event({"type": "run_completed", "stop_reason": stop_reason})
            return RunSummary(
                run_dir=store.run_dir,
                report_path=report_path,
                model_path=model_path,
                best_result=best.result if best else None,
                stop_reason=stop_reason,
                experiments=records,
            )
        finally:
            store.close()
