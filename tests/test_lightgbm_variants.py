from __future__ import annotations

import pandas as pd
import pytest

from src.models import ranker


def test_lightgbm_variants_are_disabled_by_default() -> None:
    assert ranker.get_enabled_lightgbm_variants({}) == []


def test_catboost_is_enabled_by_default_and_can_be_disabled() -> None:
    assert ranker.is_catboost_enabled({}) is True
    assert ranker.is_catboost_enabled({"models": {"catboost": {"enabled": False}}}) is False


def test_lightgbm_regression_variants_are_disabled_by_default() -> None:
    assert ranker.get_enabled_lightgbm_regression_variants({}) == []


def test_lightgbm_seed_ensemble_settings_deduplicate_seeds() -> None:
    config = {
        "models": {
            "lightgbm_seed_ensemble": {
                "enabled": True,
                "seeds": [42, 42, 202],
                "parallel_workers": 0,
                "num_threads_per_seed": 0,
            }
        }
    }

    settings = ranker.get_lightgbm_seed_ensemble_settings(config)

    assert settings["enabled"] is True
    assert settings["seeds"] == [42, 202]
    assert settings["parallel_workers"] == 1
    assert settings["num_threads_per_seed"] == 1


def test_lightgbm_seed_ensemble_requires_multiple_seeds() -> None:
    config = {
        "models": {
            "lightgbm_seed_ensemble": {
                "enabled": True,
                "seeds": [42],
            }
        }
    }

    settings = ranker.get_lightgbm_seed_ensemble_settings(config)

    assert settings["enabled"] is False


def test_ensemble_settings_normalizes_model_metrics_parallel_workers() -> None:
    config = {
        "models": {
            "ensemble": {
                "parallel_workers": 8,
                "model_metrics_parallel_workers": 0,
            }
        }
    }

    settings = ranker.get_ensemble_settings(config)

    assert settings["parallel_workers"] == 8
    assert settings["model_metrics_parallel_workers"] == 1


def test_xgboost_and_random_forest_regression_variants_are_disabled_by_default() -> None:
    assert ranker.get_enabled_xgboost_regression_variants({}) == []
    assert ranker.get_enabled_random_forest_regression_variants({}) == []
    assert ranker.get_enabled_ridge_regression_variants({}) == []
    assert ranker.get_enabled_neural_regression_variants({}) == []


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

    assert variants == [{"name": "lightgbm_top1", "feature_set": "full", "params": {"num_leaves": 15}}]


def test_upset_variant_config_applies_defaults() -> None:
    config = {
        "models": {
            "lightgbm_variants": {
                "enabled": True,
                "variants": [
                    {
                        "name": "lightgbm_upset_variant",
                        "upset_training": {"enabled": True, "history_years": 8.0},
                    }
                ],
            }
        }
    }

    variant = ranker.get_enabled_lightgbm_variants(config)[0]

    assert variant["upset_training"]["enabled"] is True
    assert variant["upset_training"]["history_years"] == 8.0
    assert variant["upset_training"]["recent_years"] == 3.5
    assert variant["upset_training"]["control_ratio"] == 3


def test_value_recovery_variant_config_applies_defaults() -> None:
    config = {
        "models": {
            "lightgbm_variants": {
                "enabled": True,
                "variants": [
                    {
                        "name": "lightgbm_value_recovery_variant",
                        "value_recovery_training": {
                            "enabled": True,
                            "payout_weight_10000_30000": 2.0,
                        },
                    }
                ],
            }
        }
    }

    variant = ranker.get_enabled_lightgbm_variants(config)[0]

    assert variant["value_recovery_training"]["enabled"] is True
    assert variant["value_recovery_training"]["payout_weight_10000_30000"] == 2.0
    assert variant["value_recovery_training"]["payout_weight_under_1000"] == 0.7
    assert variant["value_recovery_training"]["payout_weight_50000_over"] == 0.6


def test_value_recovery_variant_training_frame_weights_payout_bands() -> None:
    rows = []
    payouts = {
        "P0": 800.0,
        "P1": 3000.0,
        "P2": 8000.0,
        "P3": 20000.0,
        "P4": 40000.0,
        "P5": 60000.0,
    }
    for race_id, payout in payouts.items():
        for lane in range(1, 7):
            rows.append({"race_id": race_id, "lane": lane, "trifecta_payout": payout})
    train_df = pd.DataFrame(rows)

    weighted = ranker.build_value_recovery_variant_training_frame(
        train_df,
        ranker.DEFAULT_VALUE_RECOVERY_TRAINING_SETTINGS,
    )
    race_weights = weighted.drop_duplicates("race_id").set_index("race_id")[
        ranker.VALUE_RECOVERY_TRAINING_WEIGHT_COLUMN
    ]

    assert race_weights.loc["P0"] == pytest.approx(0.7)
    assert race_weights.loc["P1"] == pytest.approx(1.0)
    assert race_weights.loc["P2"] == pytest.approx(1.3)
    assert race_weights.loc["P3"] == pytest.approx(1.8)
    assert race_weights.loc["P4"] == pytest.approx(1.2)
    assert race_weights.loc["P5"] == pytest.approx(0.6)
    assert weighted.groupby("race_id")[ranker.VALUE_RECOVERY_TRAINING_WEIGHT_COLUMN].nunique().eq(1).all()


def test_select_upset_history_races_adds_matched_controls_and_weights() -> None:
    rows = []
    race_specs = [
        ("H10", "2020-01-10", 12000.0),
        ("H50", "2020-01-11", 60000.0),
        ("C1", "2020-01-12", 1000.0),
        ("C2", "2020-01-13", 2000.0),
        ("C3", "2020-01-14", 3000.0),
        ("C4", "2020-01-15", 4000.0),
        ("OLD", "2017-01-10", 120000.0),
    ]
    for race_id, race_date, payout in race_specs:
        for lane in range(1, 7):
            rows.append(
                {
                    "race_id": race_id,
                    "race_date": pd.Timestamp(race_date),
                    "venue": "01",
                    "lane": lane,
                    "trifecta_payout": payout,
                }
            )
    historical_df = pd.DataFrame(rows)
    settings = {
        **ranker.DEFAULT_UPSET_TRAINING_SETTINGS,
        "control_ratio": 1,
        "random_seed": 7,
    }

    selected = ranker.select_upset_history_races(historical_df, pd.Timestamp("2026-07-28"), settings)
    race_rows = selected.drop_duplicates("race_id").set_index("race_id")

    assert {"H10", "H50", "OLD"}.issubset(race_rows.index)
    assert len(race_rows.index.intersection({"C1", "C2", "C3", "C4"})) == 2
    assert race_rows.loc["H10", ranker.UPSET_TRAINING_WEIGHT_COLUMN] == pytest.approx(1.4)
    assert race_rows.loc["H50", ranker.UPSET_TRAINING_WEIGHT_COLUMN] == pytest.approx(2.1)
    assert race_rows.loc["OLD", ranker.UPSET_TRAINING_WEIGHT_COLUMN] == pytest.approx(1.5)
    assert selected.groupby("race_id")[ranker.UPSET_TRAINING_WEIGHT_COLUMN].nunique().eq(1).all()


def test_legacy_feature_set_excludes_late_added_features() -> None:
    columns = [
        "lane",
        "racer_prev_win_rate",
        "racer_prev_win_rate_race_rank",
        "venue_course_prev_win_rate",
        "venue_course_prev_win_rate_race_rank",
        "national_win_rate_race_mean",
        "racer_prev_win_rate_gap_inner",
        "race_attack_pressure",
        "lane_outer_link_fit",
        "pre_race_attack_score",
    ]

    selected = ranker.select_feature_columns_for_set(columns, "legacy_20260712")

    assert selected == [
        "lane",
        "racer_prev_win_rate",
        "racer_prev_win_rate_race_rank",
    ]


def test_lightgbm_regression_variant_config_validates_finish_position_target() -> None:
    config = {
        "models": {
            "lightgbm_regression_variants": {
                "enabled": True,
                "variants": [
                    {
                        "name": "lightgbm_reg_finish_position",
                        "target": "finish_position",
                        "score_transform": "negative",
                        "params": {"num_leaves": 31},
                    },
                ],
            }
        }
    }

    variants = ranker.get_enabled_lightgbm_regression_variants(config)

    assert variants == [
        {
            "name": "lightgbm_reg_finish_position",
            "target": "finish_position",
            "score_transform": "negative",
            "params": {"num_leaves": 31},
        }
    ]


def test_xgboost_regression_variant_config_validates_finish_position_target() -> None:
    config = {
        "models": {
            "xgboost_regression_variants": {
                "enabled": True,
                "variants": [
                    {
                        "name": "xgboost_reg_finish_position",
                        "target": "finish_position",
                        "score_transform": "negative",
                        "params": {"max_depth": 3},
                    },
                ],
            }
        }
    }

    variants = ranker.get_enabled_xgboost_regression_variants(config)

    assert variants == [
        {
            "name": "xgboost_reg_finish_position",
            "target": "finish_position",
            "score_transform": "negative",
            "params": {"max_depth": 3},
        }
    ]


def test_neural_value_recovery_variant_config_applies_defaults() -> None:
    config = {
        "models": {
            "neural_regression_variants": {
                "enabled": True,
                "variants": [
                    {
                        "name": "mlp_reg_value_recovery",
                        "model_type": "mlp",
                        "target": "finish_position",
                        "value_recovery_training": {
                            "enabled": True,
                            "payout_weight_10000_30000": 2.0,
                        },
                    },
                ],
            }
        }
    }

    variant = ranker.get_enabled_neural_regression_variants(config)[0]

    assert variant["name"] == "mlp_reg_value_recovery"
    assert variant["value_recovery_training"]["enabled"] is True
    assert variant["value_recovery_training"]["payout_weight_10000_30000"] == 2.0
    assert variant["value_recovery_training"]["payout_weight_under_1000"] == 0.7


def test_random_forest_regression_variant_config_validates_finish_position_target() -> None:
    config = {
        "models": {
            "random_forest_regression_variants": {
                "enabled": True,
                "variants": [
                    {
                        "name": "random_forest_reg_finish_position",
                        "target": "finish_position",
                        "score_transform": "negative",
                        "params": {"max_depth": 10},
                    },
                ],
            }
        }
    }

    variants = ranker.get_enabled_random_forest_regression_variants(config)

    assert variants == [
        {
            "name": "random_forest_reg_finish_position",
            "target": "finish_position",
            "score_transform": "negative",
            "params": {"max_depth": 10},
        }
    ]


def test_ridge_regression_variant_config_validates_finish_position_target() -> None:
    config = {
        "models": {
            "ridge_regression_variants": {
                "enabled": True,
                "variants": [
                    {
                        "name": "ridge_reg_finish_position",
                        "target": "finish_position",
                        "score_transform": "negative",
                        "params": {"alpha": 10.0},
                    },
                ],
            }
        }
    }

    variants = ranker.get_enabled_ridge_regression_variants(config)

    assert variants == [
        {
            "name": "ridge_reg_finish_position",
            "target": "finish_position",
            "score_transform": "negative",
            "params": {"alpha": 10.0},
        }
    ]


def test_neural_regression_variant_config_validates_supported_models() -> None:
    config = {
        "models": {
            "neural_regression_variants": {
                "enabled": True,
                "variants": [
                    {
                        "name": "mlp_reg_finish_position",
                        "model_type": "mlp",
                        "target": "finish_position",
                        "score_transform": "negative",
                        "params": {"epochs": 2, "device": "auto", "device_id": 0, "predict_device": "cpu"},
                    },
                    {
                        "name": "tabnet_reg_finish_position",
                        "model_type": "tabnet",
                        "target": "finish_position",
                        "score_transform": "negative",
                        "params": {"max_epochs": 2, "device": "cpu", "predict_device": "cpu"},
                    },
                ],
            }
        }
    }

    variants = ranker.get_enabled_neural_regression_variants(config)

    assert variants == [
        {
            "name": "mlp_reg_finish_position",
            "model_type": "mlp",
            "target": "finish_position",
            "score_transform": "negative",
            "params": {"epochs": 2, "device": "auto", "device_id": 0, "predict_device": "cpu"},
        },
        {
            "name": "tabnet_reg_finish_position",
            "model_type": "tabnet",
            "target": "finish_position",
            "score_transform": "negative",
            "params": {"max_epochs": 2, "device": "cpu", "predict_device": "cpu"},
        },
    ]


def test_variant_config_skips_disabled_individual_variant() -> None:
    config = {
        "models": {
            "neural_regression_variants": {
                "enabled": True,
                "variants": [
                    {
                        "name": "mlp_reg_finish_position",
                        "model_type": "mlp",
                        "target": "finish_position",
                        "score_transform": "negative",
                        "params": {},
                    },
                    {
                        "name": "tabnet_reg_finish_position",
                        "enabled": False,
                        "model_type": "tabnet",
                        "target": "finish_position",
                        "score_transform": "negative",
                        "params": {},
                    },
                ],
            }
        }
    }

    variants = ranker.get_enabled_neural_regression_variants(config)

    assert [variant["name"] for variant in variants] == ["mlp_reg_finish_position"]


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


def test_lightgbm_regression_variant_config_rejects_non_regression_name() -> None:
    config = {
        "models": {
            "lightgbm_regression_variants": {
                "enabled": True,
                "variants": [
                    {"name": "lightgbm_top1", "target": "finish_position", "params": {}},
                ],
            }
        }
    }

    with pytest.raises(ValueError, match="Invalid"):
        ranker.get_enabled_lightgbm_regression_variants(config)


def test_lightgbm_regression_score_uses_lower_predicted_finish_as_better() -> None:
    class DummyRegressor:
        def predict(self, frame):
            return [3.0, 1.0]

    df = pd.DataFrame(
        {
            "race_id": ["R1", "R1"],
            "lane": [1, 2],
            "finish_position": [2, 1],
        }
    )

    scored = ranker.score_frame(
        DummyRegressor(),
        "lightgbm_reg_finish_position",
        df,
        feature_columns=[],
        categorical_columns=[],
    )

    assert scored.loc[scored["lane"] == 2, "pred_rank"].iloc[0] == 1
    assert scored.loc[scored["lane"] == 2, "score_probability_like"].iloc[0] > scored.loc[
        scored["lane"] == 1,
        "score_probability_like",
    ].iloc[0]


def test_random_forest_regression_score_uses_lower_predicted_finish_as_better() -> None:
    class DummyRegressor:
        def predict(self, frame):
            return [3.0, 1.0]

    df = pd.DataFrame(
        {
            "race_id": ["R1", "R1"],
            "lane": [1, 2],
            "finish_position": [2, 1],
        }
    )

    scored = ranker.score_frame(
        DummyRegressor(),
        "random_forest_reg_finish_position",
        df,
        feature_columns=[],
        categorical_columns=[],
    )

    assert scored.loc[scored["lane"] == 2, "pred_rank"].iloc[0] == 1


def test_ridge_regression_score_uses_lower_predicted_finish_as_better() -> None:
    class DummyRegressor:
        def predict(self, frame):
            return [3.0, 1.0]

    df = pd.DataFrame(
        {
            "race_id": ["R1", "R1"],
            "lane": [1, 2],
            "finish_position": [2, 1],
        }
    )

    scored = ranker.score_frame(
        DummyRegressor(),
        "ridge_reg_finish_position",
        df,
        feature_columns=[],
        categorical_columns=[],
    )

    assert scored.loc[scored["lane"] == 2, "pred_rank"].iloc[0] == 1


def test_neural_regression_score_uses_lower_predicted_finish_as_better() -> None:
    class DummyRegressor:
        def predict(self, frame):
            return [3.0, 1.0]

    df = pd.DataFrame(
        {
            "race_id": ["R1", "R1"],
            "lane": [1, 2],
            "finish_position": [2, 1],
        }
    )

    scored = ranker.score_frame(
        DummyRegressor(),
        "mlp_reg_finish_position",
        df,
        feature_columns=[],
        categorical_columns=[],
    )

    assert scored.loc[scored["lane"] == 2, "pred_rank"].iloc[0] == 1


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


def test_ensemble_weight_optimization_uses_value_balanced_objective(monkeypatch) -> None:
    valid_df = pd.DataFrame(
        {
            "race_id": ["R1"] * 6,
            "lane": [1, 2, 3, 4, 5, 6],
            "finish_position": [1, 2, 3, 4, 5, 6],
            "trifecta_payout": [12000.0] * 6,
        }
    )

    def fake_score_frame(model, model_type, df, feature_columns, categorical_columns):
        frame = valid_df.copy()
        frame["score_probability_like"] = [10.0, 5.0, 4.0, 0.0, -1.0, -2.0]
        return frame

    monkeypatch.setattr(ranker, "score_frame", fake_score_frame)
    config = {
        "models": {
            "ensemble": {
                "grid_step": 0.50,
                "objective": "trifecta_value_balanced",
                "parallel_workers": 1,
                "min_purchase_rate": 1.0,
                "top12_payout_score_cap": 12000.0,
                "value_rule": {"high": "top3", "middle": "top3", "low": "skip"},
            }
        }
    }

    weights = ranker.optimize_ensemble_weights(
        {
            "lightgbm": object(),
            "lightgbm_value_recovery_variant": object(),
        },
        valid_df,
        feature_columns=[],
        categorical_columns=[],
        config=config,
    )

    assert weights["validation_objective_name"] == "trifecta_value_balanced"
    assert weights["validation_fast_trifecta_races"] == 1.0
    assert weights["validation_value_rule_purchase_rate"] == 1.0
    assert weights["validation_value_rule_hit_rate"] == 1.0
    assert weights["validation_value_rule_recovery_rate"] == pytest.approx(40.0)
    assert weights["validation_normalized_recovery_score"] == 1.0
    assert weights["validation_top12_payout_capture_mean"] == pytest.approx(12000.0)
    assert weights["validation_normalized_top12_payout_capture_score"] == 1.0


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
