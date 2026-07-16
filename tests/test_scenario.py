from __future__ import annotations

import pandas as pd

from src.features.scenario import classify_result_pattern, scenario_feature_values, scenario_label, score_pre_race_scenarios


def _scenario_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "lane": [1, 2, 3, 4, 5, 6],
            "win_probability_like": [0.55, 0.14, 0.11, 0.08, 0.07, 0.05],
            "win_prob": [0.58, 0.13, 0.10, 0.08, 0.06, 0.05],
            "top2_prob": [0.82, 0.35, 0.28, 0.22, 0.18, 0.12],
            "top3_prob": [0.93, 0.55, 0.45, 0.35, 0.27, 0.18],
            "exact1_prob": [0.58, 0.13, 0.10, 0.08, 0.06, 0.05],
            "flow_prob_nige": [0.78, 0.02, 0.02, 0.01, 0.01, 0.01],
            "flow_prob_sashi": [0.02, 0.42, 0.08, 0.05, 0.03, 0.02],
            "flow_prob_makuri": [0.01, 0.03, 0.10, 0.12, 0.06, 0.04],
            "flow_prob_makurizashi": [0.01, 0.03, 0.08, 0.09, 0.07, 0.05],
            "racer_prev_avg_st_5": [0.12, 0.15, 0.16, 0.17, 0.18, 0.19],
            "exhibition_time": [6.70, 6.74, 6.75, 6.76, 6.78, 6.80],
            "motor_prev_top3_rate": [0.50, 0.38, 0.36, 0.35, 0.33, 0.30],
            "boat_prev_top3_rate": [0.48, 0.36, 0.34, 0.33, 0.31, 0.29],
            "venue_course_prev_win_rate": [0.58, 0.15, 0.10, 0.08, 0.05, 0.04],
            "venue_course_prev_top2_rate": [0.75, 0.34, 0.25, 0.20, 0.15, 0.10],
            "venue_course_prev_top3_rate": [0.86, 0.52, 0.42, 0.32, 0.24, 0.16],
        }
    ).set_index("lane")


def test_pre_race_scenario_detects_in_control() -> None:
    scenario = score_pre_race_scenarios(_scenario_frame())

    assert scenario_label(scenario) == "S0_C1_ESCAPE_IN_WIN"


def test_pre_race_scenario_detects_course4_attack() -> None:
    frame = _scenario_frame()
    frame.loc[1, ["win_probability_like", "win_prob", "top2_prob", "flow_prob_nige"]] = [0.25, 0.25, 0.42, 0.18]
    frame.loc[4, ["win_probability_like", "win_prob", "top2_prob", "top3_prob"]] = [0.28, 0.28, 0.58, 0.75]
    frame.loc[4, ["flow_prob_makuri", "flow_prob_makurizashi"]] = [0.82, 0.62]
    frame.loc[4, ["racer_prev_avg_st_5", "exhibition_time"]] = [0.10, 6.65]
    frame.loc[4, ["motor_prev_top3_rate", "boat_prev_top3_rate", "venue_course_prev_top3_rate"]] = [0.55, 0.52, 0.62]

    scenario = score_pre_race_scenarios(frame)

    assert scenario_label(scenario) == "S4_C4_WIN_KADO"


def test_pre_race_scenario_detects_outside_attack_and_exports_compatibility_features() -> None:
    frame = _scenario_frame()
    frame.loc[1, ["win_probability_like", "win_prob", "top2_prob", "flow_prob_nige"]] = [0.25, 0.25, 0.42, 0.18]
    frame.loc[5, ["win_probability_like", "win_prob", "top2_prob", "top3_prob"]] = [0.32, 0.32, 0.62, 0.82]
    frame.loc[5, ["flow_prob_makurizashi", "racer_prev_avg_st_5", "exhibition_time"]] = [0.90, 0.09, 6.63]
    frame.loc[5, ["venue_course_prev_top3_rate", "motor_prev_top3_rate", "boat_prev_top3_rate"]] = [0.65, 0.52, 0.49]

    scenario = score_pre_race_scenarios(frame)
    features = scenario_feature_values(scenario)

    assert scenario_label(scenario) == "S5_OUTER_WIN_OR_CHAIN"
    assert "scenario_s5_outer_win_or_chain_score" in features
    assert "scenario_s5_score" in features
    assert "scenario_s7_score" in features
    assert features["scenario_s7_score"] == 0.0


def test_chaos_label_is_not_used_as_generic_other() -> None:
    scenario = {
        "escape_strength": 0.32,
        "inner_collapse_risk": 0.55,
        "attack_pressure": 0.36,
        "sashi_pressure_2": 0.10,
        "s2_makuri_pressure": 0.12,
        "s3_attack_pressure": 0.18,
        "s4_course_attack_pressure": 0.14,
        "s5_outside_attack_pressure": 0.10,
        "makuri_pressure_3_4": 0.18,
        "makurizashi_pressure": 0.15,
        "outer_sweep_risk": 0.18,
        "venue_escape_top2_rate": 0.60,
        "venue_outer_top3_rate": 0.30,
        "second_attack_pressure": 0.34,
        "chaos_pressure": 0.50,
    }

    assert scenario_label(scenario) != "S6_MIXED_OTHER"


def test_chaos_label_is_used_for_high_chaos_pressure() -> None:
    scenario = {
        "escape_strength": 0.25,
        "inner_collapse_risk": 0.70,
        "attack_pressure": 0.45,
        "sashi_pressure_2": 0.10,
        "s2_makuri_pressure": 0.12,
        "s3_attack_pressure": 0.18,
        "s4_course_attack_pressure": 0.16,
        "s5_outside_attack_pressure": 0.15,
        "makuri_pressure_3_4": 0.18,
        "makurizashi_pressure": 0.16,
        "outer_sweep_risk": 0.40,
        "venue_escape_top2_rate": 0.45,
        "venue_outer_top3_rate": 0.35,
        "second_attack_pressure": 0.42,
        "chaos_pressure": 0.68,
    }

    assert scenario_label(scenario) == "S6_MIXED_OTHER"


def test_classify_result_pattern_builds_objective_composite_label() -> None:
    race = pd.DataFrame(
        {
            "lane": [1, 2, 3, 4, 5, 6],
            "course": [1, 2, 3, 4, 5, 6],
            "finish_position": [2, 1, 4, 3, 5, 6],
            "finish_status": ["", "", "", "", "", ""],
            "winning_style": ["差し"] * 6,
        }
    )

    result = classify_result_pattern(race)

    assert result["scenario"] == "S1_C2_SASHI_IN_REMAIN"
    assert result["result_scenario"] == "C2__SASHI__IN_REMAIN__INNER_CENTER"
    assert result["winner_course"] == 2
    assert result["lane1_status"] == "IN_REMAIN"
