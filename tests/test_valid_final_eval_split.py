import pandas as pd

from src.cli import split_valid_for_final_eval


def test_split_valid_for_final_eval_uses_latest_one_month() -> None:
    valid_df = pd.DataFrame(
        {
            "race_id": ["r1", "r2", "r3", "r4"],
            "race_date": pd.to_datetime(["2026-05-31", "2026-06-12", "2026-06-13", "2026-07-12"]),
        }
    )

    tune_df, final_df = split_valid_for_final_eval(valid_df, final_eval_months=1)

    assert tune_df["race_id"].tolist() == ["r1", "r2"]
    assert final_df["race_id"].tolist() == ["r3", "r4"]


def test_split_valid_for_final_eval_keeps_tune_when_valid_is_too_short() -> None:
    valid_df = pd.DataFrame(
        {
            "race_id": ["r1", "r2"],
            "race_date": pd.to_datetime(["2026-07-01", "2026-07-12"]),
        }
    )

    tune_df, final_df = split_valid_for_final_eval(valid_df, final_eval_months=1)

    assert tune_df["race_id"].tolist() == ["r1", "r2"]
    assert final_df.empty
