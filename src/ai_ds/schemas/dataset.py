"""Dataset profiling schemas with JSON-friendly representations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class ColumnProfile:
    name: str
    dtype: str
    semantic_type: str
    missing_count: int
    missing_ratio: float
    unique_count: int
    unique_ratio: float
    is_constant: bool
    examples: list[str] = field(default_factory=list)
    numeric_summary: dict[str, float | None] = field(default_factory=dict)
    top_values: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TargetCandidate:
    column: str
    score: float
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DatasetProfile:
    rows: int
    columns: int
    duplicate_rows: int
    column_profiles: list[ColumnProfile]
    constant_columns: list[str]
    suspicious_id_columns: list[str]
    target_candidates: list[TargetCandidate]
    correlations: list[dict[str, Any]]
    warnings: list[str]
    dataset_hash: str

    def get_column(self, name: str) -> ColumnProfile:
        for column in self.column_profiles:
            if column.name == name:
                return column
        raise KeyError(f"Column {name!r} was not profiled")

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "columns": self.columns,
            "duplicate_rows": self.duplicate_rows,
            "column_profiles": [column.to_dict() for column in self.column_profiles],
            "constant_columns": self.constant_columns,
            "suspicious_id_columns": self.suspicious_id_columns,
            "target_candidates": [candidate.to_dict() for candidate in self.target_candidates],
            "correlations": self.correlations,
            "warnings": self.warnings,
            "dataset_hash": self.dataset_hash,
        }
