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


def test_ensemble_weight_optimization_supports_rank_legacy_objective(monkeypatch) -> None:
    valid_df = pd.DataFrame(
        {
            "race_id": ["R1", "R1", "R2", "R2"],
            "lane": [1, 2, 1, 2],
            "finish_position": [2, 1, 2, 1],
        }
    )

    def fake_score_frame(model, model_type, df, feature_columns, categorical_columns):
        scores = [0.1, 0.9, 0.1, 0.9] if model_type == "lightgbm_legacy_top1_top3" else [0.9, 0.1, 0.9, 0.1]
        frame = valid_df.copy()
        frame["score_probability_like"] = scores
        return frame

    monkeypatch.setattr(ranker, "score_frame", fake_score_frame)

    weights = ranker.optimize_ensemble_weights(
        {"lightgbm": object(), "lightgbm_legacy_top1_top3": object()},
        valid_df,
        feature_columns=[],
        categorical_columns=[],
        config={"models": {"ensemble": {"objective": "rank_legacy", "grid_step": 0.1}}},
    )

    assert weights["lightgbm_legacy_top1_top3"] == 1.0
    assert weights["validation_objective_name"] == "rank_legacy"
    assert weights["validation_top1_accuracy"] == 1.0


def test_ensemble_weight_optimization_uses_fast_trifecta_objective(monkeypatch) -> None:
    valid_df = pd.DataFrame(
        {
            "race_id": ["R1"] * 6,
            "lane": [1, 2, 3, 4, 5, 6],
            "finish_position": [1, 2, 3, 4, 5, 6],
        }
    )

    def fake_score_frame(model, model_type, df, feature_columns, categorical_columns):
        if model_type == "lightgbm_top1":
            scores = [0.70, 0.12, 0.10, 0.04, 0.03, 0.01]
        elif model_type == "lightgbm_stable_top6":
            scores = [0.40, 0.22, 0.18, 0.10, 0.06, 0.04]
        elif model_type == "lightgbm":
            scores = [0.30, 0.25, 0.20, 0.10, 0.08, 0.07]
        else:
            scores = [0.10, 0.15, 0.20, 0.25, 0.16, 0.14]
        frame = valid_df.copy()
        frame["score_probability_like"] = scores
        return frame

    monkeypatch.setattr(ranker, "score_frame", fake_score_frame)
    config = {
        "models": {
            "ensemble": {
                "grid_step": 0.10,
                "objective": "trifecta_fast",
                "parallel_workers": 1,
            }
        }
    }

    weights = ranker.optimize_ensemble_weights(
        {
            "catboost": object(),
            "lightgbm": object(),
            "lightgbm_top1": object(),
            "lightgbm_stable_top6": object(),
        },
        valid_df,
        feature_columns=[],
        categorical_columns=[],
        config=config,
    )

    assert weights["validation_candidate_count"] == 286.0
    assert weights["validation_objective_name"] == "trifecta_fast"
    assert weights["validation_fast_trifecta_races"] == 1.0
    assert weights["validation_top12_hit_rate"] == 1.0
    assert weights["validation_top5_hit_rate"] == 1.0
    assert weights["validation_log_loss"] > 0.0


def test_trifecta_v1_model_metrics_include_ensemble_and_individual_models(monkeypatch) -> None:
    valid_df = pd.DataFrame({"race_id": ["R1"], "lane": [1], "finish_position": [1]})
    calls: list[str] = []

    def fake_evaluate_trifecta(
        models,
        weights,
        calibrator,
        df,
        feature_columns,
        categorical_columns,
        **kwargs,
    ):
        model_name = "ensemble" if len(models) > 1 else next(iter(models))
        calls.append(model_name)
        return {"race_count": 1.0, "model_name": model_name, "calibrated": float(calibrator is not None)}

    def fake_fit_trifecta_calibrator(models, weights, valid_df, feature_columns, categorical_columns):
        return object()

    monkeypatch.setattr(ranker, "evaluate_trifecta", fake_evaluate_trifecta)
    monkeypatch.setattr(ranker, "fit_trifecta_calibrator", fake_fit_trifecta_calibrator)

    metrics = ranker.evaluate_trifecta_v1_model_metrics(
        {"catboost": object(), "lightgbm": object()},
        {"catboost": 0.4, "lightgbm": 0.6},
        object(),
        valid_df,
        pd.DataFrame(),
        feature_columns=[],
        categorical_columns=[],
    )

    assert set(metrics) == {"ensemble", "catboost", "lightgbm"}
    assert metrics["ensemble"]["valid_raw"]["model_name"] == "ensemble"
    assert metrics["catboost"]["valid_calibrated"]["model_name"] == "catboost"
    assert metrics["lightgbm"]["valid_calibrated"]["model_name"] == "lightgbm"
    assert calls.count("ensemble") == 4
