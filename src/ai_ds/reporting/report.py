"""Render an inspectable Markdown research report for one run."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np

from ai_ds.agent.state import ExperimentRecord
from ai_ds.problem.solver import ProblemDefinition
from ai_ds.schemas.dataset import DatasetProfile


def _bullet(items: Iterable[str], empty: str = "None") -> str:
    values = list(items)
    return "\n".join(f"- {item}" for item in values) if values else f"- {empty}"


def important_features(fitted_pipeline: object, limit: int = 12) -> list[tuple[str, float]]:
    """Best-effort feature importance extraction; absence is reported gracefully."""

    try:
        preprocess = fitted_pipeline.named_steps["preprocess"]
        model = fitted_pipeline.named_steps["model"]
        names = list(preprocess.get_feature_names_out())
        if hasattr(model, "feature_importances_"):
            values = np.asarray(model.feature_importances_)
        elif hasattr(model, "coef_"):
            coefficients = np.asarray(model.coef_)
            values = (
                np.mean(np.abs(coefficients), axis=0)
                if coefficients.ndim > 1
                else np.abs(coefficients)
            )
        else:
            return []
        ordered = np.argsort(values)[::-1][:limit]
        return [(str(names[index]), float(values[index])) for index in ordered]
    except (AttributeError, KeyError, TypeError, ValueError):
        return []


def render_report(
    output_path: Path,
    profile: DatasetProfile,
    problem: ProblemDefinition,
    records: list[ExperimentRecord],
    stop_reason: str,
    final_model: object | None = None,
) -> Path:
    """Write one self-contained Markdown account of the run and its trajectory."""

    successful = [record for record in records if record.result.ranking_score is not None]
    best = max(
        successful, key=lambda record: record.result.ranking_score or float("-inf"), default=None
    )
    lines = [
        "# AI-DS Research Report",
        "",
        "## Dataset",
        "",
        f"- Rows: {profile.rows}",
        f"- Columns: {profile.columns}",
        f"- Dataset SHA-256: `{profile.dataset_hash}`",
        f"- Duplicate rows: {profile.duplicate_rows}",
        f"- Constant columns removed: {', '.join(profile.constant_columns) or 'none'}",
        f"- Suspected ID columns removed: {', '.join(profile.suspicious_id_columns) or 'none'}",
        "",
        "## Problem formulation",
        "",
        f"- Target: `{problem.target_column}`",
        f"- Task: {problem.task_type}",
        f"- Primary metric: {problem.primary_metric} ({problem.metric_direction})",
        f"- Confidence: {problem.problem_confidence:.2f}",
        f"- Target cardinality: {problem.target_cardinality}",
        "",
        "### Data-quality and safety warnings",
        "",
        _bullet(problem.warnings),
        "",
        "## Experiment timeline",
        "",
        "| # | Model | Preprocessing | Validation result | Runtime | Status |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for record in records:
        result = record.result
        score = (
            "—" if result.primary_score is None else f"{result.metric} = {result.primary_score:.5f}"
        )
        lines.append(
            f"| {result.experiment_id or '—'} | {result.model} | {result.preprocessing} | "
            f"{score} | {result.runtime_seconds:.2f}s | {result.status} |"
        )
    if not records:
        lines.append("| — | — | — | No experiment executed | — | — |")
    lines.extend(["", "## Experimental reasoning", ""])
    for record in records:
        result = record.result
        lines.extend(
            [
                f"### Experiment {result.experiment_id or '—'}",
                "",
                f"**Hypothesis:** {record.action.hypothesis or 'Not supplied.'}",
                "",
                (
                    f"**Intervention:** `{record.action.action}` using `{result.model}` with "
                    f"`{result.preprocessing}` preprocessing."
                ),
                "",
                f"**Result:** {result.primary_score if result.primary_score is not None else result.error or result.status}",
                "",
                f"**Conclusion:** {record.conclusion or 'Result recorded for comparison.'}",
                "",
            ]
        )
    lines.extend(["## Best model", ""])
    if best:
        lines.extend(
            [
                f"The selected model is **{best.result.model}** with `{best.result.preprocessing}` preprocessing.",
                "",
                f"- Cross-validated {best.result.metric}: **{best.result.primary_score:.5f}**",
                f"- Secondary metrics: {best.result.secondary_scores or 'not configured'}",
                f"- Selected strictly from validation results across {best.result.cv_folds} folds.",
            ]
        )
    else:
        lines.append("No successful experiment was available for final-model selection.")
    lines.extend(["", "## Important features", ""])
    importances = important_features(final_model) if final_model is not None else []
    if importances:
        lines.extend(f"- `{name}`: {value:.5f}" for name, value in importances)
    else:
        lines.append("- Feature importance is unavailable for the selected estimator.")
    lines.extend(["", "## Stop condition", "", f"{stop_reason}", ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
