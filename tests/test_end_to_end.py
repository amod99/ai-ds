from __future__ import annotations

import sqlite3

from ai_ds.config import RunConfig
from ai_ds.controller import RunController


def test_run_creates_reproducible_artifacts(tmp_path, classification_frame):
    dataset = tmp_path / "churn.csv"
    classification_frame.to_csv(dataset, index=False)
    summary = RunController(
        RunConfig(
            target_column="churn",
            cv_folds=5,
            max_experiments=2,
            max_runtime_minutes=2,
            runs_dir=tmp_path / "runs",
        )
    ).run(dataset)

    assert summary.report_path.exists()
    assert (summary.run_dir / "trajectory.jsonl").exists()
    assert (summary.run_dir / "experiments.sqlite").exists()
    assert summary.best_result is not None
    assert summary.model_path and summary.model_path.exists()
    with sqlite3.connect(summary.run_dir / "experiments.sqlite") as connection:
        assert connection.execute("SELECT COUNT(*) FROM experiments").fetchone()[0] == 2
