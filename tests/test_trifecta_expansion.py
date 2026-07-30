from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.builder import add_race_relative_features
from src.models.ranker import (
    _build_trifecta_feature_frame_legacy,
    apply_prediction_time_measurement_proxies,
    blend_conservative_rerank_scores,
    build_phase3_second_feature_frame,
    build_phase3_third_feature_frame,
    build_trifecta_feature_frame,
    enumerate_trifecta_probabilities_from_scores,
    get_dynamic_rerank_weight_for_race,
    get_rerank_top_n,
    infer_feature_columns,
    predict_race_order,
    predict_trifecta_probabilities,
    restrict_trifecta_candidates_for_rerank,
    select_rerank_candidate_mask,
    select_rerank_candidate_mask_from_v1,
    with_rerank_top_n,
)
from src.odds.expected_value import attach_expected_value_columns


class DummyModel:
    def __init__(self, predictions: list[float]) -> None:
        self.predictions = np.asarray(predictions, dtype=float)

    def predict(self, _: object) -> np.ndarray:
        return self.predictions.copy()


def make_future_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "race_id": ["R1"] * 6,
            "lane": [1, 2, 3, 4, 5, 6],
            "base_feature": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
        }
    )


def make_models() -> dict[str, DummyModel]:
    predictions = [3.0, 2.0, 1.0, 0.0, -1.0, -2.0]
    return {
        "catboost": DummyModel(predictions),
        "lightgbm": DummyModel(predictions),
    }


def test_race_group_of_six_is_preserved() -> None:
    future_df = make_future_df()
    ranked = predict_race_order(make_models(), ["base_feature"], future_df, {"catboost": 0.5, "lightgbm": 0.5})
    assert ranked.groupby("race_id").size().to_dict() == {"R1": 6}


def test_trifecta_generates_120_combinations() -> None:
    future_df = make_future_df()
    ranked = predict_race_order(make_models(), ["base_feature"], future_df, {"catboost": 0.5, "lightgbm": 0.5})
    trifecta = enumerate_trifecta_probabilities_from_scores(ranked)
    assert len(trifecta) == 120
    assert trifecta["trifecta"].nunique() == 120


def test_trifecta_probabilities_sum_to_one_per_race() -> None:
    future_df = make_future_df()
    trifecta = predict_trifecta_probabilities(
        models=make_models(),
        feature_columns=["base_feature"],
        future_df=future_df,
        ensemble_weights={"catboost": 0.5, "lightgbm": 0.5},
        trifecta_calibrator=None,
    )
    grouped = trifecta.groupby("race_id")["probability"].sum().round(12)
    assert grouped.to_dict() == {"R1": 1.0}


def test_predict_trifecta_without_odds_does_not_fail() -> None:
    future_df = make_future_df()
    trifecta = predict_trifecta_probabilities(
        models=make_models(),
        feature_columns=["base_feature"],
        future_df=future_df,
        ensemble_weights={"catboost": 0.5, "lightgbm": 0.5},
        trifecta_calibrator=None,
        odds_df=None,
    )
    assert "probability_v1" in trifecta.columns
    assert "probability_v2" in trifecta.columns
    assert "race_upset_score" in trifecta.columns
    assert "race_upset_label" in trifecta.columns
    assert "race_scenario_id" in trifecta.columns
    assert "race_scenario_name" in trifecta.columns
    assert "race_scenario_description" in trifecta.columns
    assert "trifecta_darkhorse_score" in trifecta.columns
    assert "is_darkhorse_candidate" in trifecta.columns
    assert "ticket_hint" in trifecta.columns
    assert "odds" not in trifecta.columns


def test_dynamic_rerank_weight_rule_selects_subset_weight() -> None:
    race_df = make_future_df()
    race_df["win_probability_like"] = [0.45, 0.20, 0.14, 0.09, 0.07, 0.05]
    model = {
        "model_type": "lgbm_ranker",
        "conservative_v1_weight": 0.9,
        "dynamic_rerank_weight": {
            "enabled": True,
            "default_weight": 0.9,
            "thresholds": {
                "attack_pressure_high": 0.0,
                "inner_collapse_risk_high": 2.0,
                "race_upset_score_high": 2.0,
                "probability_flatness_high": 2.0,
                "escape_strength_high": 2.0,
                "inner_collapse_risk_mid": 0.0,
            },
            "rules": [{"subset": "attack_or_collapse", "weight": 0.7}],
        },
    }

    weight, subset, enabled = get_dynamic_rerank_weight_for_race(model, race_df)

    assert enabled
    assert subset == "attack_or_collapse"
    assert weight == 0.7


def test_expected_value_calculation_is_correct() -> None:
    trifecta = pd.DataFrame(
        {
            "race_id": ["R1", "R1"],
            "trifecta": ["1-2-3", "1-3-2"],
            "probability": [0.1, 0.05],
            "odds": [15.0, 8.0],
        }
    )
    enriched = attach_expected_value_columns(trifecta)
    first = enriched.loc[enriched["trifecta"] == "1-2-3"].iloc[0]
    assert first["expected_value"] == 1.5
    assert first["is_value_bet"]
    assert 0.0 <= first["stake_fraction"] <= 0.02


def test_future_information_columns_do_not_enter_features() -> None:
    df = pd.DataFrame(
        {
            "race_id": ["R1"],
            "race_date": [pd.Timestamp("2026-01-01")],
            "lane": [1],
            "base_feature": [0.5],
            "finish_position": [1],
            "is_win": [1],
            "is_top3": [1],
            "winning_style": ["逃げ"],
        }
    )
    feature_columns = infer_feature_columns(df)
    assert "base_feature" in feature_columns
    assert "finish_position" not in feature_columns
    assert "is_win" not in feature_columns
    assert "is_top3" not in feature_columns
    assert "winning_style" not in feature_columns


def test_high_correlation_drop_columns_do_not_enter_features() -> None:
    df = pd.DataFrame(
        {
            "race_id": ["R1"],
            "race_date": [pd.Timestamp("2026-01-01")],
            "lane": [1],
            "base_feature": [0.5],
            "racer_prev_avg_st": [0.15],
            "start_timing": [0.15],
            "race_attack_pressure": [0.2],
            "pre_race_attack_candidate_score": [0.2],
        }
    )

    feature_columns = infer_feature_columns(df)

    assert "base_feature" in feature_columns
    assert "start_timing" in feature_columns
    assert "pre_race_attack_candidate_score" in feature_columns
    assert "racer_prev_avg_st" not in feature_columns
    assert "race_attack_pressure" not in feature_columns


def test_medium_correlation_drop_columns_do_not_enter_features() -> None:
    df = pd.DataFrame(
        {
            "race_id": ["R1"],
            "race_date": [pd.Timestamp("2026-01-01")],
            "lane": [1],
            "venue_course_prev_win_rate": [0.5],
            "venue_course_prev_win_rate_race_diff_mean": [0.1],
            "venue_course_prev_top2_rate": [0.7],
            "venue_course_prev_top2_rate_race_diff_mean": [0.2],
            "start_timing_race_rank": [1],
            "start_timing_race_rank_low": [6],
        }
    )

    feature_columns = infer_feature_columns(df)

    assert "venue_course_prev_win_rate" not in feature_columns
    assert "venue_course_prev_top2_rate" not in feature_columns
    assert "start_timing_race_rank" not in feature_columns
    assert "venue_course_prev_win_rate_race_diff_mean" in feature_columns
    assert "venue_course_prev_top2_rate_race_diff_mean" in feature_columns
    assert "start_timing_race_rank_low" in feature_columns


def test_evaluation_measurement_proxies_replace_start_and_exhibition_but_keep_course() -> None:
    frame = pd.DataFrame(
        {
            "start_timing": [0.01, 0.02],
            "exhibition_time": [6.50, 6.51],
            "course": [4, 5],
            "racer_venue_prev_avg_st": [0.14, pd.NA],
            "racer_prev_avg_st": [0.17, 0.16],
            "racer_venue_prev_avg_exhibition": [6.72, pd.NA],
            "racer_prev_avg_exhibition": [6.80, 6.77],
        }
    )

    proxied = apply_prediction_time_measurement_proxies(frame)

    assert proxied["start_timing"].tolist() == [0.14, 0.16]
    assert proxied["exhibition_time"].tolist() == [6.72, 6.77]
    assert proxied["course"].tolist() == [4, 5]


def test_evaluation_measurement_proxies_rebuild_relative_features() -> None:
    frame = pd.DataFrame(
        {
            "race_id": ["R1", "R1"],
            "lane": [1, 2],
            "start_timing": [0.01, 0.02],
            "exhibition_time": [6.50, 6.51],
            "course": [1, 2],
            "racer_venue_prev_avg_st": [0.18, 0.12],
            "racer_prev_avg_st": [0.17, 0.16],
            "racer_venue_prev_avg_exhibition": [6.80, 6.70],
            "racer_prev_avg_exhibition": [6.85, 6.75],
            "national_win_rate": [7.0, 5.0],
            "motor_place_rate": [40.0, 35.0],
            "venue_course_prev_win_rate": [0.55, 0.20],
            "venue_course_prev_top3_rate": [0.80, 0.55],
        }
    )
    with_actual_relative = add_race_relative_features(frame)
    assert with_actual_relative.loc[0, "exhibition_time_race_rank_low"] == 1

    proxied = apply_prediction_time_measurement_proxies(with_actual_relative)

    assert proxied["start_timing"].tolist() == [0.18, 0.12]
    assert proxied["exhibition_time"].tolist() == [6.80, 6.70]
    assert proxied.loc[0, "exhibition_time_race_rank_low"] == 2
    assert proxied.loc[1, "exhibition_time_race_rank_low"] == 1
    assert "race_escape_reliability_score" in proxied.columns
    assert "race_attack_pressure" in proxied.columns
    assert "race_inner_collapse_risk" in proxied.columns
    assert "race_outer_link_risk" in proxied.columns


def test_proxy_st_structure_features_are_rebuilt_from_prediction_time_st() -> None:
    frame = pd.DataFrame(
        {
            "race_id": ["R1"] * 6,
            "lane": [1, 2, 3, 4, 5, 6],
            "start_timing": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
            "exhibition_time": [6.70, 6.71, 6.72, 6.73, 6.74, 6.75],
            "course": [1, 2, 3, 4, 5, 6],
            "racer_venue_prev_avg_st": [0.18, 0.17, 0.16, 0.15, 0.14, 0.13],
            "racer_prev_avg_st": [0.19, 0.18, 0.17, 0.16, 0.15, 0.14],
            "racer_prev_avg_st_5": [0.16, 0.19, 0.15, 0.18, 0.13, 0.15],
            "racer_prev_avg_st_10": [0.17, 0.16, 0.18, 0.14, 0.16, 0.12],
            "racer_venue_prev_avg_exhibition": [6.80, 6.79, 6.78, 6.77, 6.76, 6.75],
            "racer_prev_avg_exhibition": [6.81, 6.80, 6.79, 6.78, 6.77, 6.76],
        }
    )
    with_actual_relative = add_race_relative_features(frame)

    proxied = apply_prediction_time_measurement_proxies(with_actual_relative)

    assert np.isclose(proxied.loc[0, "race_inner3_avg_st"], (0.18 + 0.17 + 0.16) / 3)
    assert np.isclose(proxied.loc[0, "race_outer3_avg_st"], (0.15 + 0.14 + 0.13) / 3)
    assert np.isclose(proxied.loc[0, "race_outer_inner_avg_st_gap"], 0.14 - 0.17)
    assert np.isclose(proxied.loc[0, "lane_st_gap_inner3_avg"], 0.18 - 0.17)
    assert np.isclose(proxied.loc[5, "lane_st_gap_outer3_avg"], 0.13 - 0.14)
    assert np.isclose(proxied.loc[0, "racer_prev_avg_st_window_best"], 0.16)
    assert np.isclose(proxied.loc[0, "racer_prev_avg_st_window_worst"], 0.19)
    assert np.isclose(proxied.loc[0, "start_timing_gap_st_window_best"], 0.18 - 0.16)


def test_select_rerank_candidate_mask_from_v1_does_not_force_actual_inclusion() -> None:
    v1_df = pd.DataFrame(
        {
            "race_id": ["R1"] * 4,
            "trifecta": ["1-2-3", "1-3-2", "2-1-3", "3-1-2"],
            "raw_probability_v1": [0.4, 0.3, 0.2, 0.1],
            "is_actual": [False, False, False, True],
        }
    )

    mask = select_rerank_candidate_mask_from_v1(v1_df, top_n=2)

    assert mask.tolist() == [True, True, False, False]


def test_restrict_trifecta_candidates_for_rerank_does_not_force_actual_inclusion() -> None:
    trifecta_df = pd.DataFrame(
        {
            "race_id": ["R1"] * 4,
            "trifecta": ["1-2-3", "1-3-2", "2-1-3", "3-1-2"],
            "raw_probability_v1": [0.4, 0.3, 0.2, 0.1],
            "probability_v1": [0.4, 0.3, 0.2, 0.1],
            "probability_v2": [0.4, 0.3, 0.2, 0.1],
            "is_actual": [False, False, False, True],
        }
    )

    restricted = restrict_trifecta_candidates_for_rerank(trifecta_df, top_n=2)

    assert restricted["trifecta"].tolist() == ["1-2-3", "1-3-2"]
    assert np.isclose(restricted["probability_v1"].sum(), 1.0)
    assert np.isclose(restricted["probability_v2"].sum(), 1.0)


def test_conservative_rerank_blend_keeps_higher_scores_better() -> None:
    base = np.asarray([0.9, 0.8, 0.7, 0.6], dtype=float)
    update = np.asarray([0.9, 0.7, 0.8, 0.6], dtype=float)

    blended = blend_conservative_rerank_scores(
        base,
        update,
        conservative_weight=0.95,
        rank_penalty_strength=0.0,
    )

    assert int(np.argmax(blended)) == 0
    assert blended[0] > blended[-1]


def test_rerank_blend_uses_update_direction_when_not_conservative() -> None:
    base = np.asarray([0.9, 0.8, 0.7, 0.6], dtype=float)
    update = np.asarray([0.1, 0.2, 0.9, 0.3], dtype=float)

    blended = blend_conservative_rerank_scores(
        base,
        update,
        conservative_weight=0.0,
        rank_penalty_strength=0.0,
    )

    assert int(np.argmax(blended)) == 2


def test_rerank_top_n_prefers_model_value_over_fallback() -> None:
    model = {"model_type": "lgbm_ranker"}

    updated = with_rerank_top_n(model, 16)

    assert get_rerank_top_n(updated, 10) == 16
    assert get_rerank_top_n(model, 10) == 10


def test_phase3_scenario_features_are_added_to_rerank_frames() -> None:
    race_df = pd.DataFrame(
        {
            "race_id": ["R1"] * 6,
            "lane": [1, 2, 3, 4, 5, 6],
            "win_probability_like": [0.42, 0.20, 0.16, 0.10, 0.07, 0.05],
            "win_prob": [0.46, 0.18, 0.14, 0.10, 0.07, 0.05],
            "top2_prob": [0.72, 0.42, 0.35, 0.25, 0.17, 0.12],
            "top3_prob": [0.88, 0.65, 0.55, 0.40, 0.29, 0.21],
            "exact1_prob": [0.46, 0.18, 0.14, 0.10, 0.07, 0.05],
            "exact2_prob": [0.26, 0.24, 0.21, 0.15, 0.10, 0.07],
            "exact3_prob": [0.16, 0.23, 0.20, 0.15, 0.12, 0.09],
            "flow_prob_nige": [0.70, 0.03, 0.03, 0.02, 0.01, 0.01],
            "flow_prob_sashi": [0.05, 0.45, 0.12, 0.09, 0.05, 0.03],
            "flow_prob_makuri": [0.02, 0.06, 0.38, 0.30, 0.12, 0.07],
            "flow_prob_makurizashi": [0.02, 0.07, 0.24, 0.32, 0.20, 0.12],
            "racer_prev_avg_st_5": [0.12, 0.14, 0.11, 0.13, 0.16, 0.17],
            "exhibition_time": [6.70, 6.74, 6.68, 6.73, 6.78, 6.80],
            "motor_prev_top3_rate": [0.48, 0.40, 0.46, 0.44, 0.37, 0.35],
            "boat_prev_top3_rate": [0.45, 0.38, 0.43, 0.42, 0.35, 0.34],
        }
    )
    v1 = pd.DataFrame(
        {
            "trifecta": ["1-2-3", "3-1-4"],
            "raw_probability": [0.5, 0.2],
        }
    )
    v2 = pd.DataFrame(
        {
            "trifecta": ["1-2-3", "3-1-4"],
            "raw_probability_v2": [0.45, 0.25],
        }
    )

    trifecta_features = build_trifecta_feature_frame(race_df, v1, v2)
    second_features = build_phase3_second_feature_frame(race_df, 1)
    third_features = build_phase3_third_feature_frame(race_df, 1, 2)

    expected = {
        "escape_strength",
        "attack_lane",
        "attack_pressure",
        "escape_line_fit",
        "scenario_mismatch_penalty",
        "scenario_s0_score",
        "scenario_s7_score",
        "scenario_id_numeric",
    }
    assert expected.issubset(trifecta_features.columns)
    assert expected.issubset(second_features.columns)
    assert expected.issubset(third_features.columns)
    assert trifecta_features.loc[0, "escape_line_fit"] > 0.0


def test_vectorized_phase3_trifecta_features_match_legacy() -> None:
    race_df = pd.DataFrame(
        {
            "race_id": ["R1"] * 6,
            "lane": [1, 2, 3, 4, 5, 6],
            "win_probability_like": [0.42, 0.20, 0.16, 0.10, 0.07, 0.05],
            "win_prob": [0.46, 0.18, 0.14, 0.10, 0.07, 0.05],
            "top2_prob": [0.72, 0.42, 0.35, 0.25, 0.17, 0.12],
            "top3_prob": [0.88, 0.65, 0.55, 0.40, 0.29, 0.21],
            "exact1_prob": [0.46, 0.18, 0.14, 0.10, 0.07, 0.05],
            "exact2_prob": [0.26, 0.24, 0.21, 0.15, 0.10, 0.07],
            "exact3_prob": [0.16, 0.23, 0.20, 0.15, 0.12, 0.09],
            "flow_prob_nige": [0.70, 0.03, 0.03, 0.02, 0.01, 0.01],
            "flow_prob_sashi": [0.05, 0.45, 0.12, 0.09, 0.05, 0.03],
            "flow_prob_makuri": [0.02, 0.06, 0.38, 0.30, 0.12, 0.07],
            "flow_prob_makurizashi": [0.02, 0.07, 0.24, 0.32, 0.20, 0.12],
            "racer_prev_avg_st_5": [0.12, 0.14, 0.11, 0.13, 0.16, 0.17],
            "exhibition_time": [6.70, 6.74, 6.68, 6.73, 6.78, 6.80],
            "motor_prev_top3_rate": [0.48, 0.40, 0.46, 0.44, 0.37, 0.35],
            "boat_prev_top3_rate": [0.45, 0.38, 0.43, 0.42, 0.35, 0.34],
            "venue_course_prev_win_rate": [0.58, 0.15, 0.10, 0.08, 0.05, 0.04],
            "venue_course_prev_top2_rate": [0.75, 0.34, 0.25, 0.20, 0.15, 0.10],
            "venue_course_prev_top3_rate": [0.86, 0.52, 0.42, 0.32, 0.24, 0.16],
        }
    )
    v1 = pd.DataFrame(
        {
            "trifecta": ["1-2-3", "3-1-4", "4-5-1", "6-4-2"],
            "raw_probability": [0.5, 0.2, 0.15, 0.05],
        }
    )
    v2 = pd.DataFrame(
        {
            "trifecta": ["1-2-3", "3-1-4", "4-5-1", "6-4-2"],
            "raw_probability_v2": [0.45, 0.25, 0.12, 0.08],
        }
    )

    legacy = _build_trifecta_feature_frame_legacy(race_df, v1, v2)
    vectorized = build_trifecta_feature_frame(race_df, v1, v2)

    pd.testing.assert_index_equal(vectorized.columns, legacy.columns)
    pd.testing.assert_frame_equal(vectorized, legacy)


def test_scenario_candidates_expand_v1_top_n_without_actual_result() -> None:
    race_df = pd.DataFrame(
        {
            "race_id": ["R1"] * 6,
            "lane": [1, 2, 3, 4, 5, 6],
            "win_probability_like": [0.38, 0.18, 0.17, 0.14, 0.08, 0.05],
            "win_prob": [0.40, 0.16, 0.16, 0.15, 0.08, 0.05],
            "top2_prob": [0.60, 0.33, 0.35, 0.39, 0.20, 0.13],
            "top3_prob": [0.72, 0.48, 0.52, 0.58, 0.39, 0.25],
            "flow_prob_nige": [0.20, 0.01, 0.01, 0.01, 0.01, 0.01],
            "flow_prob_sashi": [0.01, 0.10, 0.05, 0.04, 0.02, 0.01],
            "flow_prob_makuri": [0.01, 0.03, 0.25, 0.75, 0.15, 0.05],
            "flow_prob_makurizashi": [0.01, 0.03, 0.20, 0.65, 0.18, 0.08],
        }
    )
    v1 = pd.DataFrame(
        {
            "trifecta": ["1-2-3", "1-3-2", "4-1-3"],
            "raw_probability_v1": [0.5, 0.4, 0.01],
        }
    )

    selected = select_rerank_candidate_mask(
        v1,
        race_df,
        top_n=1,
        scenario_top_n=1,
    )

    assert selected.sum() == 2
    assert selected.iloc[0]
    assert selected.iloc[2]
