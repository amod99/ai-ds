"""Factory for deterministic, cross-validation-safe preprocessing pipelines."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ai_ds.preprocessing.transforms import (
    FrequencyEncoder,
    SignedLog1pTransformer,
    StableFeatureHasher,
    TargetMeanEncoder,
)


@dataclass(frozen=True, slots=True)
class PreprocessingConfig:
    categorical_encoding: str = "onehot"
    scale_numeric: bool = False
    log_numeric: bool = False
    hash_features: int = 32
    target_smoothing: float = 10.0

    @classmethod
    def from_action_name(cls, name: str, scale_numeric: bool = False) -> PreprocessingConfig:
        mapping = {
            "baseline": "onehot",
            "frequency": "frequency",
            "hash": "hash",
            "target": "target",
            "log_numeric": "onehot",
        }
        if name not in mapping:
            raise ValueError(f"Unsupported preprocessing action: {name}")
        return cls(
            categorical_encoding=mapping[name],
            scale_numeric=scale_numeric,
            log_numeric=name == "log_numeric",
        )


def feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Classify columns deterministically using pandas dtypes."""

    numeric = [str(column) for column in frame.select_dtypes(include=["number", "bool"]).columns]
    categorical = [str(column) for column in frame.columns if str(column) not in numeric]
    return numeric, categorical


def _one_hot_encoder() -> OneHotEncoder:
    # ``sparse_output`` is available for the minimum supported sklearn version.
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False)


def build_preprocessor(frame: pd.DataFrame, config: PreprocessingConfig) -> ColumnTransformer:
    """Build an unfitted transformer; callers must place it inside a CV pipeline."""

    numeric_columns, categorical_columns = feature_columns(frame)
    transformers: list[tuple[str, object, list[str]]] = []
    if numeric_columns:
        numeric_steps: list[tuple[str, object]] = [("impute", SimpleImputer(strategy="median"))]
        if config.log_numeric:
            numeric_steps.append(("log", SignedLog1pTransformer()))
        if config.scale_numeric:
            numeric_steps.append(("scale", StandardScaler()))
        transformers.append(("numeric", Pipeline(numeric_steps), numeric_columns))
    if categorical_columns:
        categorical_steps: list[tuple[str, object]] = [
            ("impute", SimpleImputer(strategy="constant", fill_value="__missing__"))
        ]
        if config.categorical_encoding == "onehot":
            categorical_steps.append(("encode", _one_hot_encoder()))
        elif config.categorical_encoding == "frequency":
            categorical_steps.append(("encode", FrequencyEncoder()))
        elif config.categorical_encoding == "target":
            categorical_steps.append(("encode", TargetMeanEncoder(config.target_smoothing)))
        elif config.categorical_encoding == "hash":
            categorical_steps.append(("encode", StableFeatureHasher(config.hash_features)))
        else:
            raise ValueError(f"Unknown categorical encoding: {config.categorical_encoding}")
        transformers.append(("categorical", Pipeline(categorical_steps), categorical_columns))
    if not transformers:
        raise ValueError("No usable feature columns remain after filtering")
    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.0)
