from __future__ import annotations

from datetime import date

import pandas as pd

from src.features.streaming_builder import (
    HistoryCarryovers,
    IncrementalParquetWriter,
    RESULT_PARQUET_SCHEMA,
    add_racer_history_features_streaming,
)


def test_incremental_parquet_writer_preserves_explicit_string_schema(tmp_path) -> None:
    output_path = tmp_path / "race_results.parquet"
    writer = IncrementalParquetWriter(output_path, schema=RESULT_PARQUET_SCHEMA)

    first_frame = pd.DataFrame(
        [
            {
                "race_id": "20260501_01_01",
                "race_date": date(2026, 5, 1),
                "venue": "桐生",
                "race_no": 1,
                "lane": 1,
                "finish_position": 1,
                "finish_status": None,
                "racer_id": 1234,
                "racer_name": "A",
                "motor_no": 10,
                "boat_no": 20,
                "exhibition_time": 6.75,
                "course": 1,
                "start_timing": 0.13,
                "race_time": 108.2,
                "weather": "晴",
                "wind_direction": "北",
                "wind_speed_m": 2,
                "wave_cm": 3,
                "winning_style": None,
            }
        ]
    )
    second_frame = first_frame.copy()
    second_frame.loc[0, "race_id"] = "20260502_01_01"
    second_frame.loc[0, "race_date"] = date(2026, 5, 2)
    second_frame.loc[0, "winning_style"] = "逃げ"

    writer.write(first_frame)
    writer.write(second_frame)
    writer.close()

    result = pd.read_parquet(output_path)

    assert pd.isna(result.loc[0, "winning_style"])
    assert result.loc[1, "winning_style"] == "逃げ"


def test_incremental_parquet_writer_infers_string_for_all_null_object_column(tmp_path) -> None:
    output_path = tmp_path / "base_bucket.parquet"
    writer = IncrementalParquetWriter(output_path)

    first_frame = pd.DataFrame(
        [
            {
                "race_id": "20260501_01_01",
                "race_date": pd.Timestamp("2026-05-01"),
                "winning_style": None,
                "target_rank": 6,
            }
        ]
    )
    second_frame = pd.DataFrame(
        [
            {
                "race_id": "20260502_01_01",
                "race_date": pd.Timestamp("2026-05-02"),
                "winning_style": "逃げ",
                "target_rank": 6,
            }
        ]
    )

    writer.write(first_frame)
    writer.write(second_frame)
    writer.close()

    result = pd.read_parquet(output_path)

    assert pd.isna(result.loc[0, "winning_style"])
    assert result.loc[1, "winning_style"] == "逃げ"


def test_streaming_history_adds_decision_style_flow_features() -> None:
    carryovers = HistoryCarryovers()
    previous = pd.DataFrame(
        {
            "race_id": [f"R{i}" for i in range(1, 5)],
            "race_date": pd.date_range("2026-01-01", periods=4, freq="D"),
            "race_no": [1, 1, 1, 1],
            "venue": ["01"] * 4,
            "lane": [1] * 4,
            "course": [1] * 4,
            "racer_id": [1001] * 4,
            "motor_no": [10] * 4,
            "boat_no": [20] * 4,
            "finish_position": [1, 1, 1, 1],
            "is_win": [1, 1, 1, 1],
            "is_top2": [1, 1, 1, 1],
            "is_top3": [1, 1, 1, 1],
            "start_timing": [0.12, 0.13, 0.14, 0.15],
            "exhibition_time": [6.70, 6.71, 6.72, 6.73],
            "winning_style": ["逃げ", "逃げ", "差し", "まくり"],
        }
    )
    current = previous.tail(1).copy()
    current["race_id"] = "R5"
    current["race_date"] = pd.Timestamp("2026-01-05")
    current["winning_style"] = "まくり差し"

    add_racer_history_features_streaming(previous, carryovers)
    result = add_racer_history_features_streaming(current, carryovers)

    assert "racer_prev_nige_rate" in result.columns
    assert "flow_prob_nige" in result.columns
    assert result["racer_prev_nige_rate"].iloc[0] > 0.0
    assert result["racer_attack_style_score"].iloc[0] > 0.0
