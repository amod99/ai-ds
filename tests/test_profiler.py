from __future__ import annotations

from ai_ds.data.profiler import profile_dataset


def test_profile_reports_quality_signals_and_target_candidates(classification_frame):
    frame = classification_frame.copy()
    frame["constant"] = 1
    frame.loc[0, "charge"] = None
    profile = profile_dataset(frame, "abc123")

    assert profile.rows == 40
    assert profile.dataset_hash == "abc123"
    assert "customer_id" in profile.suspicious_id_columns
    assert "constant" in profile.constant_columns
    assert profile.get_column("charge").missing_count == 1
    assert profile.target_candidates[0].column == "churn"
