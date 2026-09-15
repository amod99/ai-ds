"""SQLite persistence for reproducible runs and append-only trajectories."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai_ds.schemas.agent import AgentAction
from ai_ds.schemas.experiment import ExperimentResult


class ExperimentStore:
    """A small local store with an SQLite source of truth and JSONL trace mirror."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.database_path = self.run_dir / "experiments.sqlite"
        self.trajectory_path = self.run_dir / "trajectory.jsonl"
        self.connection = sqlite3.connect(self.database_path)
        self.connection.row_factory = sqlite3.Row
        self._initialize()

    def _initialize(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value_json TEXT NOT NULL);
            CREATE TABLE experiments (
                experiment_id INTEGER PRIMARY KEY AUTOINCREMENT,
                parent_experiment INTEGER,
                created_at TEXT NOT NULL,
                action_json TEXT NOT NULL,
                result_json TEXT NOT NULL
            );
            CREATE TABLE trajectory (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                event_json TEXT NOT NULL
            );
            """
        )
        self.connection.commit()

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()

    def set_metadata(self, key: str, value: Any) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value_json) VALUES (?, ?)",
            (key, json.dumps(value, sort_keys=True, default=str)),
        )
        self.connection.commit()

    def record_event(self, event: dict[str, Any]) -> int:
        timestamp = self._timestamp()
        serialized = json.dumps(event, sort_keys=True, default=str)
        cursor = self.connection.execute(
            "INSERT INTO trajectory(created_at, event_json) VALUES (?, ?)", (timestamp, serialized)
        )
        self.connection.commit()
        with self.trajectory_path.open("a", encoding="utf-8") as trace:
            trace.write(
                json.dumps({"timestamp": timestamp, **event}, sort_keys=True, default=str) + "\n"
            )
        return int(cursor.lastrowid)

    def record_experiment(self, action: AgentAction, result: ExperimentResult) -> ExperimentResult:
        # The ID belongs in the serialized result too, not only in the SQL primary key.
        # Reserve it before serializing so SQLite and JSONL tell the same story.
        cursor = self.connection.execute(
            "INSERT INTO experiments(parent_experiment, created_at, action_json, result_json) VALUES (?, ?, ?, ?)",
            (
                result.parent_experiment,
                self._timestamp(),
                json.dumps(action.to_dict(), sort_keys=True),
                "{}",
            ),
        )
        result.experiment_id = int(cursor.lastrowid)
        self.connection.execute(
            "UPDATE experiments SET result_json = ? WHERE experiment_id = ?",
            (json.dumps(result.to_dict(), sort_keys=True), result.experiment_id),
        )
        self.connection.commit()
        self.record_event(
            {"type": "experiment", "action": action.to_dict(), "result": result.to_dict()}
        )
        return result

    def close(self) -> None:
        self.connection.close()
