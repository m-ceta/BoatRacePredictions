from __future__ import annotations

import pandas as pd

import src.cli as cli
import src.models.ranker as ranker


class ActualPreferringRerankModel:
    def predict(self, features: pd.DataFrame):
        mask = (
            (features["first_lane"] == 6)
            & (features["second_lane"] == 5)
            & (features["third_lane"] == 4)
        )
        return mask.astype(float).to_numpy()


def test_train_trifecta_v2_phase3_rerank_does_not_force_actual_candidate(monkeypatch) -> None:
    ranked = pd.DataFrame(
        {
            "race_id": ["R1"] * 6,
            "lane": [1, 2, 3, 4, 5, 6],
            "win_probability_like": [0.40, 0.25, 0.15, 0.10, 0.06, 0.04],
            "finish_position": [4, 5, 6, 3, 2, 1],
        }
    )

    def fake_build_trifecta_feature_frame(
        race_df: pd.DataFrame,
        v1_df: pd.DataFrame,
        v2_df: pd.DataFrame,
    ) -> pd.DataFrame:
        rows: list[dict[str, int]] = []
        for trifecta in v1_df["trifecta"].astype(str).tolist():
            first, second, third = [int(token) for token in trifecta.split("-")]
            rows.append(
                {
                    "first_lane": first,
                    "second_lane": second,
                    "third_lane": third,
                }
            )
        return pd.DataFrame(rows)

    monkeypatch.setattr(ranker, "build_trifecta_feature_frame", fake_build_trifecta_feature_frame)
    monkeypatch.setattr(ranker, "get_conservative_rerank_weight", lambda model: 0.0)
    monkeypatch.setattr(ranker, "get_rank_penalty_strength", lambda model: 0.0)

    trifecta = ranker.build_trifecta_prediction_frame(
        ranked,
        trifecta_calibrator=None,
        use_v2=True,
        trifecta_v2_model=ActualPreferringRerankModel(),
        rerank_top_n=2,
    )

    assert "6-5-4" not in trifecta["trifecta"].tolist()
    assert trifecta.iloc[0]["trifecta"] != "6-5-4"


def test_eval_trifecta_full_uses_requested_rerank_top_n_for_each_chunk(monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "race_id": ["R1", "R1", "R2", "R2"],
            "race_date": [
                pd.Timestamp("2026-05-01"),
                pd.Timestamp("2026-05-01"),
                pd.Timestamp("2026-06-01"),
                pd.Timestamp("2026-06-01"),
            ],
        }
    )

    calls: list[dict[str, object]] = []

    def fake_evaluate_trifecta(*args, **kwargs):
        chunk = args[3]
        calls.append(
            {
                "rerank_top_n": kwargs["rerank_top_n"],
                "use_v2": kwargs["use_v2"],
                "race_ids": tuple(chunk["race_id"].drop_duplicates().tolist()),
            }
        )
        return {
            "race_count": float(chunk["race_id"].nunique()),
            "top1_hit_rate": 0.0,
            "top3_hit_rate": 0.0,
            "top5_hit_rate": 0.0,
        }

    monkeypatch.setattr(cli, "evaluate_trifecta", fake_evaluate_trifecta)

    result = cli.evaluate_trifecta_in_chunks(
        frame,
        models={},
        weights={},
        feature_columns=[],
        categorical_columns=[],
        classifier_models={},
        flow_model=None,
        flow_classes=None,
        staged_models={},
        trifecta_v3_model=None,
        v1_calibrator=None,
        v3_calibrator=None,
        rerank_top_n=7,
    )

    assert len(calls) == 4
    assert {call["rerank_top_n"] for call in calls} == {7}
    assert sum(call["use_v2"] is False for call in calls) == 2
    assert sum(call["use_v2"] is True for call in calls) == 2
    assert result["v1"]["valid_calibrated"]["race_count"] == 2.0
    assert result["phase3"]["valid_calibrated"]["race_count"] == 2.0
