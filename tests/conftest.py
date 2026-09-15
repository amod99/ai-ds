from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def classification_frame() -> pd.DataFrame:
    rows = []
    for index in range(40):
        churn = "yes" if index % 2 else "no"
        rows.append(
            {
                "customer_id": f"C{index:03d}",
                "tenure": index + 1,
                "charge": 35 + (index % 9) * 8 + (20 if churn == "yes" else 0),
                "contract": "monthly" if churn == "yes" else "annual",
                "region": ["north", "south", "east", "west"][index % 4],
                "churn": churn,
            }
        )
    return pd.DataFrame(rows)
