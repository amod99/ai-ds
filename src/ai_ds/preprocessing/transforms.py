"""Small sklearn-compatible transforms used only inside fitted CV pipelines."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def _as_2d_object_array(values: object) -> np.ndarray:
    array = np.asarray(values, dtype=object)
    return array.reshape(-1, 1) if array.ndim == 1 else array


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Map each categorical value to its training-fold frequency."""

    def fit(self, X: object, y: object = None) -> FrequencyEncoder:
        array = _as_2d_object_array(X)
        self.maps_ = []
        self.feature_names_in_ = np.asarray(
            getattr(X, "columns", range(array.shape[1])), dtype=object
        )
        for column in array.T:
            values = pd.Series(column, dtype="object").fillna("__missing__").astype(str)
            self.maps_.append((values.value_counts(dropna=False) / len(values)).to_dict())
        return self

    def transform(self, X: object) -> np.ndarray:
        array = _as_2d_object_array(X)
        if array.shape[1] != len(self.maps_):
            raise ValueError("FrequencyEncoder received a different number of columns")
        result = np.zeros(array.shape, dtype=float)
        for index, column in enumerate(array.T):
            values = pd.Series(column, dtype="object").fillna("__missing__").astype(str)
            result[:, index] = values.map(self.maps_[index]).fillna(0.0).to_numpy(dtype=float)
        return result

    def get_feature_names_out(self, input_features: Iterable[str] | None = None) -> np.ndarray:
        names = input_features if input_features is not None else self.feature_names_in_
        return np.asarray([f"frequency__{name}" for name in names], dtype=object)


class TargetMeanEncoder(BaseEstimator, TransformerMixin):
    """Smoothed categorical target encoding fitted on each training fold.

    When used inside an sklearn ``Pipeline`` passed to cross-validation, ``fit`` is
    called only on that fold's training rows and ``transform`` receives its validation
    rows. Thus validation targets never contribute to their own encodings.
    """

    def __init__(self, smoothing: float = 10.0) -> None:
        self.smoothing = smoothing

    def fit(self, X: object, y: object = None) -> TargetMeanEncoder:
        if y is None:
            raise ValueError("TargetMeanEncoder requires y during fit")
        array = _as_2d_object_array(X)
        target = pd.Series(y).reset_index(drop=True)
        if pd.api.types.is_numeric_dtype(target):
            numeric_target = pd.to_numeric(target, errors="coerce")
        else:
            numeric_target = pd.Series(pd.factorize(target, sort=True)[0], dtype=float)
        self.global_mean_ = float(numeric_target.mean())
        self.maps_ = []
        self.feature_names_in_ = np.asarray(
            getattr(X, "columns", range(array.shape[1])), dtype=object
        )
        for column in array.T:
            categories = pd.Series(column, dtype="object").fillna("__missing__").astype(str)
            stats = pd.DataFrame({"category": categories, "target": numeric_target}).groupby(
                "category"
            )["target"]
            means = stats.mean()
            counts = stats.count()
            smoothed = (means * counts + self.global_mean_ * self.smoothing) / (
                counts + self.smoothing
            )
            self.maps_.append(smoothed.to_dict())
        return self

    def transform(self, X: object) -> np.ndarray:
        array = _as_2d_object_array(X)
        if array.shape[1] != len(self.maps_):
            raise ValueError("TargetMeanEncoder received a different number of columns")
        result = np.zeros(array.shape, dtype=float)
        for index, column in enumerate(array.T):
            values = pd.Series(column, dtype="object").fillna("__missing__").astype(str)
            result[:, index] = (
                values.map(self.maps_[index]).fillna(self.global_mean_).to_numpy(dtype=float)
            )
        return result

    def get_feature_names_out(self, input_features: Iterable[str] | None = None) -> np.ndarray:
        names = input_features if input_features is not None else self.feature_names_in_
        return np.asarray([f"target_mean__{name}" for name in names], dtype=object)


class StableFeatureHasher(BaseEstimator, TransformerMixin):
    """A deterministic, dependency-free categorical hashing transform."""

    def __init__(self, n_features: int = 32) -> None:
        self.n_features = n_features

    def fit(self, X: object, y: object = None) -> StableFeatureHasher:
        if self.n_features < 2:
            raise ValueError("n_features must be at least 2")
        array = _as_2d_object_array(X)
        self.feature_names_in_ = np.asarray(
            getattr(X, "columns", range(array.shape[1])), dtype=object
        )
        return self

    def transform(self, X: object) -> np.ndarray:
        array = _as_2d_object_array(X)
        result = np.zeros((array.shape[0], self.n_features), dtype=float)
        for row_index, row in enumerate(array):
            for column_index, value in enumerate(row):
                token = f"{column_index}={value if pd.notna(value) else '__missing__'}".encode()
                digest = hashlib.blake2b(token, digest_size=8).digest()
                bucket = int.from_bytes(digest, "little") % self.n_features
                result[row_index, bucket] += 1.0
        return result

    def get_feature_names_out(self, input_features: Iterable[str] | None = None) -> np.ndarray:
        return np.asarray([f"hash_{index}" for index in range(self.n_features)], dtype=object)


class SignedLog1pTransformer(BaseEstimator, TransformerMixin):
    """A reversible sign-preserving log transform safe for values below zero."""

    def fit(self, X: object, y: object = None) -> SignedLog1pTransformer:
        # sklearn's fitted-state check requires at least one trailing-underscore
        # attribute for a stateless transformer inside a Pipeline.
        self.n_features_in_ = _as_2d_object_array(X).shape[1]
        return self

    def transform(self, X: object) -> np.ndarray:
        values = np.asarray(X, dtype=float)
        return np.sign(values) * np.log1p(np.abs(values))

    def get_feature_names_out(self, input_features: Iterable[str] | None = None) -> np.ndarray:
        return np.asarray(input_features if input_features is not None else [], dtype=object)
