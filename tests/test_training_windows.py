from __future__ import annotations

import pandas as pd

from src.models.ranker import split_training_frame_random_by_race, with_latest_available_dates


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


def test_with_latest_available_dates_supports_fractional_rolling_years() -> None:
    config = {
        "data": {
            "min_date": "2023-01-01",
            "max_date": "2026-05-24",
            "rolling_years": 3.5,
        },
        "split": {
            "method": "random_by_race",
            "train_ratio": 6,
            "valid_ratio": 1,
            "random_seed": 42,
        },
    }

    updated = with_latest_available_dates(config, pd.Timestamp("2026-06-03"))

    assert updated["data"]["max_date"] == "2026-06-03"
    assert updated["data"]["min_date"] == "2022-12-04"


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


def test_split_training_frame_random_by_race_uses_race_groups() -> None:
    frame = pd.DataFrame(
        {
            "race_id": [f"R{race_no:02d}" for race_no in range(14) for _lane in range(6)],
            "race_date": [pd.Timestamp("2026-01-01")] * 84,
            "lane": list(range(1, 7)) * 14,
        }
    )
    config = {
        "model": {"random_seed": 999},
        "split": {
            "method": "random_by_race",
            "train_ratio": 6,
            "valid_ratio": 1,
            "random_seed": 123,
        },
    }

    train_df, valid_df, test_df = split_training_frame_random_by_race(frame, config)

    train_races = set(train_df["race_id"])
    valid_races = set(valid_df["race_id"])
    assert len(train_races) == 12
    assert len(valid_races) == 2
    assert train_races.isdisjoint(valid_races)
    assert test_df.empty
