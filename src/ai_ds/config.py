"""Configuration shared by the controller and deterministic experiment engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

TaskType = Literal["classification", "regression"]


@dataclass(slots=True)
class RunConfig:
    """Explicit bounds and protocol settings for one reproducible run."""

    target_column: str | None = None
    task_type: TaskType | None = None
    metric: str | None = None
    max_experiments: int = 15
    max_runtime_minutes: float = 30.0
    cv_folds: int = 5
    random_seed: int = 42
    patience: int = 4
    minimum_improvement: float = 0.002
    max_failures: int = 3
    runs_dir: Path = field(default_factory=lambda: Path("runs"))
    remove_id_columns: bool = True
    remove_leakage_risks: bool = True

    def __post_init__(self) -> None:
        if self.max_experiments < 1:
            raise ValueError("max_experiments must be at least 1")
        if self.max_runtime_minutes <= 0:
            raise ValueError("max_runtime_minutes must be positive")
        if self.cv_folds < 2:
            raise ValueError("cv_folds must be at least 2")
        if self.patience < 1:
            raise ValueError("patience must be at least 1")
        if self.max_failures < 1:
            raise ValueError("max_failures must be at least 1")
        if self.task_type not in (None, "classification", "regression"):
            raise ValueError("task_type must be classification or regression")

    def serializable(self) -> dict[str, object]:
        result = asdict(self)
        result["runs_dir"] = str(self.runs_dir)
        return result
