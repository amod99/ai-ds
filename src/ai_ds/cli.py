"""Command-line interface for profiling and bounded AI-DS runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ai_ds.agent.planner import FixedPlanner, RuleBasedPlanner
from ai_ds.config import RunConfig
from ai_ds.controller import RunController


def _config_from_args(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        target_column=getattr(args, "target", None),
        task_type=getattr(args, "task", None),
        metric=getattr(args, "metric", None),
        max_experiments=getattr(args, "max_experiments", 15),
        max_runtime_minutes=getattr(args, "max_runtime_minutes", 30.0),
        cv_folds=getattr(args, "cv_folds", 5),
        random_seed=getattr(args, "seed", 42),
        patience=getattr(args, "patience", 4),
        minimum_improvement=getattr(args, "minimum_improvement", 0.002),
        runs_dir=Path(getattr(args, "runs_dir", "runs")),
        remove_id_columns=not getattr(args, "keep_id_columns", False),
        remove_leakage_risks=not getattr(args, "keep_leakage_risks", False),
    )


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("dataset", help="Path to one CSV dataset")
    parser.add_argument(
        "--target", help="Explicit target column; recommended when inference is uncertain"
    )
    parser.add_argument(
        "--task", choices=["classification", "regression"], help="Override task inference"
    )
    parser.add_argument("--metric", help="Override the primary validation metric")
    parser.add_argument("--max-experiments", type=int, default=15)
    parser.add_argument("--max-runtime-minutes", type=float, default=30.0)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--minimum-improvement", type=float, default=0.002)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--keep-id-columns", action="store_true")
    parser.add_argument("--keep-leakage-risks", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-ds", description="Bounded, reproducible experimentation for tabular CSV datasets."
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    profile = subcommands.add_parser("profile", help="Inspect a dataset without fitting a model")
    profile.add_argument("dataset", help="Path to one CSV dataset")
    profile.add_argument(
        "--json", action="store_true", help="Print the full structured profile as JSON"
    )
    train = subcommands.add_parser(
        "train", help="Run the deterministic non-agentic candidate sequence"
    )
    _add_run_arguments(train)
    run = subcommands.add_parser("run", help="Run the bounded rule-based agent loop")
    _add_run_arguments(run)
    return parser


def _print_profile(profile: object, as_json: bool) -> None:
    if as_json:
        print(json.dumps(profile.to_dict(), indent=2, sort_keys=True))
        return
    print(f"Dataset: {profile.rows} rows × {profile.columns} columns")
    print(f"SHA-256: {profile.dataset_hash}")
    print(f"Duplicate rows: {profile.duplicate_rows}")
    print("Likely target candidates:")
    for candidate in profile.target_candidates[:5]:
        reasons = f" ({'; '.join(candidate.reasons)})" if candidate.reasons else ""
        print(f"  - {candidate.column}: {candidate.score:.2f}{reasons}")
    if profile.warnings:
        print("Warnings:")
        for warning in profile.warnings:
            print(f"  - {warning}")


def _print_summary(summary: object) -> None:
    print(f"Run complete: {summary.run_dir}")
    if summary.best_result and summary.best_result.primary_score is not None:
        print(
            f"Best model: {summary.best_result.model} | {summary.best_result.metric} "
            f"= {summary.best_result.primary_score:.5f}"
        )
    else:
        print("No successful experiment was available for final-model selection.")
    print(f"Stop reason: {summary.stop_reason}")
    print(f"Report: {summary.report_path}")
    if summary.model_path:
        print(f"Model: {summary.model_path}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "profile":
            profile = RunController(RunConfig()).inspect(args.dataset)
            _print_profile(profile, args.json)
            return 0
        config = _config_from_args(args)
        planner = FixedPlanner() if args.command == "train" else RuleBasedPlanner()
        summary = RunController(config, planner=planner).run(args.dataset)
        _print_summary(summary)
        return 0
    except (FileNotFoundError, ValueError) as exc:
        print(f"ai-ds: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
