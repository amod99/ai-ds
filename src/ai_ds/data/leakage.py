"""Conservative, explainable checks for IDs and obvious target leakage."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

_ID_NAME = re.compile(r"(^|[_\-\s])(id|uuid|guid|key|identifier)([_\-\s]|$)", re.IGNORECASE)
_LEAK_NAME = re.compile(r"(target|label|outcome|post[_\-]?|future|leak)", re.IGNORECASE)


def find_id_columns(frame: pd.DataFrame) -> list[str]:
    """Flag likely row identifiers; this is a warning, not an irreversible mutation."""

    row_count = len(frame)
    flagged: list[str] = []
    for name in frame.columns:
        series = frame[name]
        non_null = series.dropna()
        if non_null.empty:
            continue
        unique_ratio = non_null.nunique(dropna=True) / row_count
        name_looks_like_id = bool(_ID_NAME.search(str(name)))
        # A unique numeric measurement is commonly a legitimate continuous feature
        # (for example tenure or price). Treat unnamed high-cardinality values as IDs
        # only when they are nonnumeric; numeric IDs need a name-based signal.
        is_nonnumeric = not pd.api.types.is_numeric_dtype(series)
        if (name_looks_like_id and unique_ratio >= 0.80) or (is_nonnumeric and unique_ratio >= 0.995):
            flagged.append(str(name))
    return flagged


def find_target_leakage(frame: pd.DataFrame, target: str) -> list[str]:
    """Return feature names that are trivially equivalent or near-equivalent to target.

    The test intentionally errs on the side of warnings. It does not try to infer
    temporal leakage, which requires domain knowledge.
    """

    if target not in frame.columns:
        return []
    y = frame[target]
    risks: list[str] = []
    for name in frame.columns:
        if name == target:
            continue
        feature = frame[name]
        if feature.equals(y):
            risks.append(str(name))
            continue
        if pd.api.types.is_numeric_dtype(feature) and pd.api.types.is_numeric_dtype(y):
            valid = pd.concat([feature, y], axis=1).dropna()
            if (
                len(valid) >= 3
                and valid.iloc[:, 0].nunique() > 1
                and valid.iloc[:, 1].nunique() > 1
            ):
                correlation = valid.iloc[:, 0].corr(valid.iloc[:, 1])
                if (
                    correlation is not None
                    and np.isfinite(correlation)
                    and abs(correlation) >= 0.999
                ):
                    risks.append(str(name))
                    continue
        if _LEAK_NAME.search(str(name)) and str(name).lower() != str(target).lower():
            risks.append(str(name))
    return risks
