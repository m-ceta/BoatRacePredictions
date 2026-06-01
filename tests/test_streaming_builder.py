from __future__ import annotations

from datetime import date

import pandas as pd

from src.features.streaming_builder import IncrementalParquetWriter, RESULT_PARQUET_SCHEMA


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
