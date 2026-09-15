"""Deterministic dataset inspection; no model or LLM is involved here."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from ai_ds.data.leakage import find_id_columns
from ai_ds.schemas.dataset import ColumnProfile, DatasetProfile, TargetCandidate

_TARGET_NAME = re.compile(
    r"(target|label|class|outcome|response|churn|default|fraud|diagnosis|price|sales)",
    re.IGNORECASE,
)


def _semantic_type(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    return "categorical"


def _column_profile(name: str, series: pd.Series, total_rows: int) -> ColumnProfile:
    non_null = series.dropna()
    unique_count = int(non_null.nunique(dropna=True))
    semantic_type = _semantic_type(series)
    examples = [str(value) for value in non_null.head(3).tolist()]
    top_values: list[dict[str, Any]] = []
    if semantic_type in {"categorical", "boolean"}:
        counts = non_null.value_counts(dropna=True).head(5)
        top_values = [{"value": str(value), "count": int(count)} for value, count in counts.items()]
    numeric_summary: dict[str, float | None] = {}
    if semantic_type == "numeric" and not non_null.empty:
        numeric_summary = {
            "min": _finite_float(non_null.min()),
            "p25": _finite_float(non_null.quantile(0.25)),
            "median": _finite_float(non_null.median()),
            "p75": _finite_float(non_null.quantile(0.75)),
            "max": _finite_float(non_null.max()),
            "mean": _finite_float(non_null.mean()),
            "std": _finite_float(non_null.std()),
        }
    missing_count = int(series.isna().sum())
    return ColumnProfile(
        name=str(name),
        dtype=str(series.dtype),
        semantic_type=semantic_type,
        missing_count=missing_count,
        missing_ratio=missing_count / total_rows,
        unique_count=unique_count,
        unique_ratio=unique_count / total_rows,
        is_constant=unique_count <= 1,
        examples=examples,
        numeric_summary=numeric_summary,
        top_values=top_values,
    )


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
        return number if np.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _target_candidates(columns: list[ColumnProfile], id_columns: set[str]) -> list[TargetCandidate]:
    candidates: list[TargetCandidate] = []
    for index, column in enumerate(columns):
        if column.is_constant:
            continue
        score = 0.0
        reasons: list[str] = []
        if _TARGET_NAME.search(column.name):
            score += 0.72
            reasons.append("name resembles a target")
        if index == len(columns) - 1:
            score += 0.10
            reasons.append("appears as the final column")
        if column.semantic_type in {"categorical", "boolean"}:
            score += 0.12
            reasons.append("categorical values are plausible labels")
        if column.unique_count <= max(20, int(column.unique_ratio * 0) + 1):
            score += 0.08
            reasons.append("low target-like cardinality")
        if column.name in id_columns:
            score -= 0.80
            reasons.append("looks like an identifier")
        candidates.append(TargetCandidate(column.name, round(max(score, 0.0), 3), reasons))
    return sorted(candidates, key=lambda candidate: candidate.score, reverse=True)


def _correlations(frame: pd.DataFrame) -> list[dict[str, Any]]:
    numeric = frame.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        return []
    matrix = numeric.corr(numeric_only=True)
    pairs: list[dict[str, Any]] = []
    columns = list(matrix.columns)
    for left_index, left in enumerate(columns):
        for right in columns[left_index + 1 :]:
            value = matrix.loc[left, right]
            if pd.notna(value) and abs(float(value)) >= 0.80:
                pairs.append(
                    {"left": str(left), "right": str(right), "correlation": round(float(value), 4)}
                )
    return sorted(pairs, key=lambda pair: abs(pair["correlation"]), reverse=True)[:30]


def profile_dataset(frame: pd.DataFrame, dataset_hash: str = "") -> DatasetProfile:
    """Produce a serializable profile for a non-empty, validated dataframe."""

    profiles = [_column_profile(str(name), frame[name], len(frame)) for name in frame.columns]
    id_columns = find_id_columns(frame)
    constants = [profile.name for profile in profiles if profile.is_constant]
    warnings: list[str] = []
    if constants:
        warnings.append(f"{len(constants)} constant column(s) should be removed before modeling")
    if id_columns:
        warnings.append(f"{len(id_columns)} likely identifier column(s) may cause memorization")
    duplicate_rows = int(frame.duplicated().sum())
    if duplicate_rows:
        warnings.append(f"{duplicate_rows} duplicate row(s) detected")
    missing_columns = sum(profile.missing_count > 0 for profile in profiles)
    if missing_columns:
        warnings.append(f"{missing_columns} column(s) contain missing values")
    return DatasetProfile(
        rows=int(frame.shape[0]),
        columns=int(frame.shape[1]),
        duplicate_rows=duplicate_rows,
        column_profiles=profiles,
        constant_columns=constants,
        suspicious_id_columns=id_columns,
        target_candidates=_target_candidates(profiles, set(id_columns)),
        correlations=_correlations(frame),
        warnings=warnings,
        dataset_hash=dataset_hash,
    )
