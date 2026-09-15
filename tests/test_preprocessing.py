from __future__ import annotations

import numpy as np
import pandas as pd

from ai_ds.preprocessing.pipeline import PreprocessingConfig, build_preprocessor
from ai_ds.preprocessing.transforms import TargetMeanEncoder


def test_target_encoder_uses_only_fitted_category_statistics():
    encoder = TargetMeanEncoder(smoothing=0.0).fit([["a"], ["a"], ["b"]], [1, 1, 0])
    encoded = encoder.transform([["a"], ["b"], ["unseen"]])

    assert np.allclose(encoded[:, 0], [1.0, 0.0, 2 / 3])


def test_pipeline_handles_missing_numeric_and_categorical_values():
    features = pd.DataFrame({"number": [1.0, None, 3.0, 4.0], "category": ["a", None, "b", "a"]})
    pipeline = build_preprocessor(features, PreprocessingConfig(categorical_encoding="target"))
    transformed = pipeline.fit_transform(features, [0, 1, 0, 1])

    assert transformed.shape[0] == len(features)
    assert np.isfinite(transformed).all()


def test_log_numeric_pipeline_is_fitted_and_transforms_validation_rows():
    features = pd.DataFrame({"number": [-3.0, -1.0, 1.0, 5.0], "category": ["a", "b", "a", "b"]})
    pipeline = build_preprocessor(features, PreprocessingConfig(log_numeric=True))
    pipeline.fit(features, [0, 1, 0, 1])

    transformed = pipeline.transform(features.iloc[:2])
    assert transformed.shape[0] == 2
