from __future__ import annotations

import pandas as pd

from src.live import fill_live_measurement_proxies, merge_recent_group_features


def test_merge_recent_group_features_adds_venue_measurement_averages() -> None:
    frame = pd.DataFrame(
        [
            {
                "race_id": "20260618_15_01",
                "race_date": pd.Timestamp("2026-06-18"),
                "venue": "15",
                "lane": 1,
                "racer_id": 1234,
                "motor_no": 11,
                "boat_no": 21,
            }
        ]
    )
    hist = pd.DataFrame(
        [
            {
                "race_date": pd.Timestamp("2026-06-01"),
                "race_no": 1,
                "racer_id": 1234,
                "venue": "15",
                "lane": 1,
                "course": 2,
                "motor_no": 11,
                "boat_no": 21,
                "finish_position": 1,
                "start_timing": 0.12,
                "exhibition_time": 6.71,
                "is_win": 1,
                "is_top3": 1,
            },
            {
                "race_date": pd.Timestamp("2026-06-05"),
                "race_no": 2,
                "racer_id": 1234,
                "venue": "15",
                "lane": 2,
                "course": 3,
                "motor_no": 11,
                "boat_no": 21,
                "finish_position": 3,
                "start_timing": 0.18,
                "exhibition_time": 6.75,
                "is_win": 0,
                "is_top3": 1,
            },
        ]
    )

    merged = merge_recent_group_features(frame, hist)

    assert float(merged.loc[0, "racer_venue_prev_avg_st"]) == 0.15
    assert float(merged.loc[0, "racer_venue_prev_avg_exhibition"]) == 6.73
    assert float(merged.loc[0, "racer_venue_prev_avg_course"]) == 2.5


def test_fill_live_measurement_proxies_uses_venue_then_overall_history() -> None:
    frame = pd.DataFrame(
        [
            {
                "start_timing": pd.NA,
                "course": pd.NA,
                "exhibition_time": pd.NA,
                "racer_venue_prev_avg_st": 0.14,
                "racer_prev_avg_st": 0.17,
                "racer_venue_prev_avg_course": 2.2,
                "racer_prev_avg_course": 2.9,
                "racer_venue_prev_avg_exhibition": 6.72,
                "racer_prev_avg_exhibition": 6.80,
            },
            {
                "start_timing": pd.NA,
                "course": pd.NA,
                "exhibition_time": pd.NA,
                "racer_venue_prev_avg_st": pd.NA,
                "racer_prev_avg_st": 0.16,
                "racer_venue_prev_avg_course": pd.NA,
                "racer_prev_avg_course": 3.1,
                "racer_venue_prev_avg_exhibition": pd.NA,
                "racer_prev_avg_exhibition": 6.77,
            },
        ]
    )

    filled = fill_live_measurement_proxies(frame)

    assert float(filled.loc[0, "start_timing"]) == 0.14
    assert float(filled.loc[0, "course"]) == 2.2
    assert float(filled.loc[0, "exhibition_time"]) == 6.72
    assert float(filled.loc[1, "start_timing"]) == 0.16
    assert float(filled.loc[1, "course"]) == 3.1
    assert float(filled.loc[1, "exhibition_time"]) == 6.77
