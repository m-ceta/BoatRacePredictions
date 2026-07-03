from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.ranker import (
    blend_conservative_rerank_scores,
    enumerate_trifecta_probabilities_from_scores,
    infer_feature_columns,
    predict_race_order,
    predict_trifecta_probabilities,
    restrict_trifecta_candidates_for_rerank,
    select_rerank_candidate_mask_from_v1,
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
    assert "odds" not in trifecta.columns


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
