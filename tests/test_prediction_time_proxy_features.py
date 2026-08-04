import pandas as pd

from src.models.ranker import apply_prediction_time_measurement_proxies


def test_prediction_time_proxies_preserve_unrebuildable_relative_features():
    frame = pd.DataFrame(
        {
            "race_id": ["r1"] * 6,
            "lane": [1, 2, 3, 4, 5, 6],
            "racer_prev_avg_st_5": [0.16, 0.15, 0.14, 0.13, 0.17, 0.18],
            "venue_course_prev_win_rate_race_rank": [1, 2, 3, 4, 5, 6],
            "venue_course_prev_win_rate_race_mean": [0.1] * 6,
        }
    )

    proxied = apply_prediction_time_measurement_proxies(frame)

    assert "venue_course_prev_win_rate_race_rank" in proxied.columns
    assert "venue_course_prev_win_rate_race_mean" in proxied.columns
    assert "start_timing_race_rank_low" in proxied.columns
