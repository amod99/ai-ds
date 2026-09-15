"""CSV loading and content hashing."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(slots=True)
class LoadedDataset:
    path: Path
    frame: pd.DataFrame
    sha256: str


def file_sha256(path: Path) -> str:
    """Return a content hash without keeping the entire source CSV in memory."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_csv(path: str | Path) -> LoadedDataset:
    """Load a single non-empty CSV with clear validation errors."""

    csv_path = Path(path).expanduser().resolve()
    if not csv_path.exists() or not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if csv_path.suffix.lower() != ".csv":
        raise ValueError("AI-DS V1 accepts a CSV file; provide a path ending in .csv")

    try:
        frame = pd.read_csv(csv_path)
    except (UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise ValueError(f"Could not parse {csv_path.name} as a CSV: {exc}") from exc

    if frame.empty:
        raise ValueError("The dataset has no rows")
    if frame.shape[1] < 2:
        raise ValueError("The dataset needs at least two columns: features and a target")
    if not frame.columns.is_unique:
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"Column names must be unique; duplicates include {duplicates!r}")
    return LoadedDataset(path=csv_path, frame=frame, sha256=file_sha256(csv_path))
