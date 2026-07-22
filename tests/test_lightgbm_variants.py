from __future__ import annotations

import pandas as pd
import pytest

from src.models import ranker


def test_lightgbm_variants_are_disabled_by_default() -> None:
    assert ranker.get_enabled_lightgbm_variants({}) == []


def test_lightgbm_variant_config_validates_names() -> None:
    config = {
        "models": {
            "lightgbm_variants": {
                "enabled": True,
                "variants": [
                    {"name": "lightgbm_top1", "params": {"num_leaves": 15}},
                ],
            }
        }
    }

    variants = ranker.get_enabled_lightgbm_variants(config)

    assert variants == [{"name": "lightgbm_top1", "params": {"num_leaves": 15}}]


def test_lightgbm_variant_config_rejects_reserved_name() -> None:
    config = {
        "models": {
            "lightgbm_variants": {
                "enabled": True,
                "variants": [
                    {"name": "lightgbm", "params": {}},
                ],
            }
        }
    }

    with pytest.raises(ValueError, match="reserved"):
        ranker.get_enabled_lightgbm_variants(config)


def test_ensemble_weight_optimization_includes_lightgbm_variants(monkeypatch) -> None:
    valid_df = pd.DataFrame(
        {
            "race_id": ["R1", "R1", "R2", "R2"],
            "lane": [1, 2, 1, 2],
            "finish_position": [2, 1, 2, 1],
        }
    )

    def fake_score_frame(model, model_type, df, feature_columns, categorical_columns):
        if model_type == "lightgbm_top1":
            scores = [0.1, 0.9, 0.1, 0.9]
        elif model_type == "lightgbm":
            scores = [0.4, 0.6, 0.4, 0.6]
        else:
            scores = [0.6, 0.4, 0.6, 0.4]
        frame = valid_df.copy()
        frame["score_probability_like"] = scores
        return frame

    monkeypatch.setattr(ranker, "score_frame", fake_score_frame)

    weights = ranker.optimize_ensemble_weights(
        {"catboost": object(), "lightgbm": object(), "lightgbm_top1": object()},
        valid_df,
        feature_columns=[],
        categorical_columns=[],
    )

    assert weights["lightgbm_top1"] > 0.0
    assert weights["validation_top1_accuracy"] == 1.0
