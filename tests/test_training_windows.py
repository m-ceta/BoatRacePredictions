from __future__ import annotations

import pandas as pd

from src.models.ranker import with_latest_available_dates


def test_with_latest_available_dates_applies_relative_windows() -> None:
    config = {
        "data": {
            "min_date": "2023-01-01",
            "max_date": "2026-05-24",
            "rolling_years": 3,
        },
        "split": {
            "train_end_date": "2025-12-31",
            "valid_end_date": "2026-05-24",
            "valid_months": 4,
        },
    }

    updated = with_latest_available_dates(config, pd.Timestamp("2026-06-03"))

    assert updated["data"]["max_date"] == "2026-06-03"
    assert updated["data"]["min_date"] == "2023-06-04"
    assert updated["split"]["train_end_date"] == "2026-02-03"
    assert updated["split"]["valid_end_date"] == "2026-06-03"


def test_with_latest_available_dates_preserves_fixed_values_without_relative_windows() -> None:
    config = {
        "data": {
            "min_date": "2023-01-01",
            "max_date": "2026-05-24",
        },
        "split": {
            "train_end_date": "2025-12-31",
            "valid_end_date": "2026-05-24",
        },
    }

    updated = with_latest_available_dates(config, pd.Timestamp("2026-06-03"))

    assert updated["data"]["min_date"] == "2023-01-01"
    assert updated["data"]["max_date"] == "2026-06-03"
    assert updated["split"]["train_end_date"] == "2025-12-31"
    assert updated["split"]["valid_end_date"] == "2026-06-03"
