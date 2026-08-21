from __future__ import annotations

import pandas as pd

from src.in_course_probability import (
    attach_in_course_probability_columns,
    build_in_course_probability_training_frame,
    fit_in_course_probability_model,
)


def _race_rows(race_id: str, in_finish: int, race_escape: float, collapse: float) -> list[dict]:
    rows = []
    for lane in range(1, 7):
        rows.append(
            {
                "race_id": race_id,
                "lane": lane,
                "course": lane,
                "finish_position": in_finish if lane == 1 else lane,
                "race_escape_reliability_score": race_escape,
                "race_attack_pressure": 1.0 - race_escape,
                "race_inner_collapse_risk": collapse,
                "race_outer_link_risk": collapse,
                "race_outer_inner_avg_st_gap": 0.01 * lane,
                "pre_race_attack_candidate_course": 4,
                "pre_race_attack_candidate_score": collapse,
                "pre_race_attack_score": collapse if lane >= 2 else 0.0,
                "national_win_rate": 7.0 - lane,
                "national_place_rate": 40.0 - lane,
                "local_win_rate": 6.0 - lane,
                "local_place_rate": 35.0 - lane,
                "racer_prev_win_rate": 0.20 if lane == 1 else 0.10,
                "racer_prev_top3_rate": 0.55 if lane == 1 else 0.35,
                "racer_prev_avg_st_5": 0.15 + lane * 0.01,
                "racer_prev_avg_st_10": 0.15 + lane * 0.01,
                "racer_prev_best_st_30": 0.10 + lane * 0.01,
                "exhibition_time": 6.70 + lane * 0.01,
                "motor_place_rate": 30.0,
                "boat_place_rate": 30.0,
                "venue_course_prev_win_rate": 0.50 if lane == 1 else 0.15,
                "venue_course_prev_top2_rate": 0.70 if lane == 1 else 0.35,
                "venue_course_prev_top3_rate": 0.80 if lane == 1 else 0.45,
                "venue_course_prev_nige_rate": 0.45 if lane == 1 else 0.0,
                "flow_prob_nige": 0.40 if lane == 1 else 0.0,
                "flow_prob_sashi": 0.15,
                "flow_prob_makuri": 0.20 if lane >= 3 else 0.05,
                "flow_prob_makurizashi": 0.15 if lane >= 4 else 0.05,
            }
        )
    return rows


def test_build_in_course_probability_targets() -> None:
    df = pd.DataFrame(
        [
            *_race_rows("r1", in_finish=1, race_escape=0.80, collapse=0.10),
            *_race_rows("r2", in_finish=4, race_escape=0.30, collapse=0.70),
        ]
    )

    features, in_win_target = build_in_course_probability_training_frame(df, target_name="in_win")
    _, in_collapse_target = build_in_course_probability_training_frame(df, target_name="in_collapse")

    assert features["race_id"].tolist() == ["r1", "r2"]
    assert in_win_target.tolist() == [1, 0]
    assert in_collapse_target.tolist() == [0, 1]


def test_attach_in_course_probability_columns() -> None:
    df = pd.DataFrame(
        [
            *_race_rows("r1", in_finish=1, race_escape=0.80, collapse=0.10),
            *_race_rows("r2", in_finish=4, race_escape=0.30, collapse=0.70),
            *_race_rows("r3", in_finish=1, race_escape=0.75, collapse=0.20),
            *_race_rows("r4", in_finish=5, race_escape=0.20, collapse=0.80),
        ]
    )
    in_win = fit_in_course_probability_model(df, target_name="in_win")
    in_collapse = fit_in_course_probability_model(df, target_name="in_collapse")
    trifecta = pd.DataFrame({"race_id": ["r1", "r1", "r2"], "trifecta": ["1-2-3", "1-3-2", "2-1-3"]})

    enriched = attach_in_course_probability_columns(
        trifecta,
        df,
        in_win_payload=in_win,
        in_collapse_payload=in_collapse,
    )

    assert {"in_win_probability", "in_collapse_probability"}.issubset(enriched.columns)
    assert enriched["in_win_probability"].between(0.0, 1.0).all()
    assert enriched["in_collapse_probability"].between(0.0, 1.0).all()
    assert enriched.loc[enriched["race_id"] == "r1", "in_win_probability"].nunique() == 1
