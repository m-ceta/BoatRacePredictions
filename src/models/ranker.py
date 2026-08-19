from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import gc
import hashlib
import itertools
import json
import os
import shutil
from pathlib import Path
from typing import Any, Callable
import re

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from catboost import CatBoostError, CatBoostRanker, Pool
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover - exercised only when optional dependency is missing
    xgb = None

from src.evaluation.metrics import compute_trifecta_metrics, compute_trifecta_rerank_metrics
from src.features.builder import add_race_relative_features, drop_race_relative_features
from src.features.scenario import (
    SCENARIO_NAMES,
    SCENARIO_SHORT_TO_LABEL,
    classify_result_pattern,
    scenario_feature_values,
    scenario_description,
    scenario_display_name,
    scenario_label,
    scenario_line_features,
    scenario_numeric_id,
    scenario_scores,
    score_pre_race_scenarios,
)
from src.models.classifiers import (
    evaluate_classifier_models,
    load_classifier_models,
    predict_classifier_probabilities,
    save_classifier_models,
    train_classifiers,
)
from src.models.flow import (
    evaluate_flow_model,
    load_flow_model,
    predict_flow_probabilities,
    save_flow_model,
)
from src.models.staged import (
    evaluate_staged_models,
    load_staged_models,
    predict_staged_probabilities,
    save_staged_models,
)
from src.models.training_device import catboost_training_kwargs, train_lightgbm_with_optional_gpu
from src.odds.expected_value import (
    BUY_EXPECTED_VALUE_THRESHOLD,
    BUY_MIN_ODDS,
    attach_expected_value_columns,
)
from src.top12_confidence import fit_top12_probability_adjustment_table


DEFAULT_DROP_COLUMNS = {
    "race_id",
    "race_date",
    "race_title",
    "racer_name",
    "current_meet_results",
    "finish_position",
    "finish_status",
    "target_rank",
    "is_win",
    "is_top2",
    "is_top3",
    "winning_style",
    "trifecta_payout",
    "decision_style_nige_win",
    "decision_style_sashi_win",
    "decision_style_makuri_win",
    "decision_style_makurizashi_win",
    "decision_style_nuki_win",
}

DEFAULT_ARTIFACT_PATHS = {
    "catboost_model_path": "artifacts/catboost_ranker.cbm",
    "lightgbm_model_path": "artifacts/lightgbm_ranker.txt",
    "xgboost_model_path": "artifacts/xgboost_ranker.json",
    "random_forest_model_path": "artifacts/random_forest_regressor.joblib",
    "ridge_model_path": "artifacts/ridge_regressor.joblib",
    "neural_model_path": "artifacts/neural_regressor.joblib",
    "features_path": "artifacts/feature_columns.json",
    "ensemble_weights_path": "artifacts/ensemble_weights.json",
    "trifecta_calibrator_path": "artifacts/trifecta_isotonic.joblib",
    "probability_adjustment_path": "artifacts/probability_adjustment.json",
    "metrics_path": "artifacts/metrics.json",
    "classifier_dir": "artifacts/classifiers",
    "train_checkpoint_path": "artifacts/train_checkpoint.json",
}


RESERVED_MODEL_NAMES = {"catboost", "lightgbm", "xgboost", "lightgbm_seed_ensemble"}
LIGHTGBM_SEED_ENSEMBLE_MODEL_NAME = "lightgbm_seed_ensemble"
LIGHTGBM_VARIANT_NAME_RE = re.compile(r"^lightgbm_[A-Za-z0-9_]+$")
XGBOOST_VARIANT_NAME_RE = re.compile(r"^xgboost_[A-Za-z0-9_]+$")
RANDOM_FOREST_VARIANT_NAME_RE = re.compile(r"^random_forest_[A-Za-z0-9_]+$")
RIDGE_VARIANT_NAME_RE = re.compile(r"^ridge_[A-Za-z0-9_]+$")
NEURAL_VARIANT_NAME_RE = re.compile(r"^(mlp|tabnet)_[A-Za-z0-9_]+$")
TRIFECTA_V2_FEATURE_VERSION = 2

DEFAULT_CATBOOST_SETTINGS = {
    "enabled": True,
    "params": {},
}

DEFAULT_PHASE3_SETTINGS = {
    "label_weights": {
        "exact_full": 1.0,
        "exact_top2": 0.75,
        "exact_first_with_two": 0.6,
        "exact_first": 0.5,
        "top2_overlap": 0.35,
        "top3_overlap": 0.2,
    },
    "regularization": {
        "num_leaves": 31,
        "min_data_in_leaf": 30,
        "feature_fraction": 0.75,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 0.0,
        "lambda_l2": 2.0,
        "num_boost_round": 120,
    },
    "rerank": {
        "default_conservative_weight": 0.9,
        "weight_grid": [0.5, 0.7, 0.8, 0.9, 0.95],
        "top_n_grid": [12, 16, 20, 24],
        "rank_penalty_strength_grid": [0.0, 0.01, 0.02, 0.03],
        "default_rank_penalty_strength": 0.02,
        "rank_penalty_start": 5,
        "scenario_candidate_top_n": 12,
        "coarse_eval_max_races": 750,
        "coarse_penalty_grid": [0.0],
        "fine_top_k": 5,
        "log_loss_max_delta_vs_v1": 0.03,
        "objective_top12_weight": 0.40,
        "objective_top5_weight": 0.15,
        "objective_top3_weight": 0.15,
        "objective_top1_weight": 0.15,
        "objective_mrr_weight": 0.15,
        "objective_log_loss_weight": 0.10,
        "objective_log_loss_excess_penalty": 1.0,
    },
    "calibration": {
        "window_days_options": [30, 60, 90],
        "default_window_days": 60,
    },
    "evaluation": {
        "scenario_min_races": 100,
    },
    "dynamic_rerank_weight": {
        "enabled": False,
        "optimize": False,
        "min_subset_races": 500,
        "top12_min_improvement": 0.0,
        "log_loss_max_delta": 0.03,
        "weight_grid": [0.7, 0.8, 0.9, 0.95],
        "quantile_high": 0.7,
        "quantile_mid": 0.5,
    },
}


PHASE3_SCENARIO_NAMES = {
    "S0": "イン主導・逃げ展開",
    "S1": "2コース差し展開",
    "S2": "2コースまくり展開",
    "S3": "3コース攻め展開",
    "S4": "センターまくり差し展開",
    "S5": "カドまくり展開",
    "S6": "外攻め展開",
    "S7": "攻め連鎖・混戦展開",
}


PHASE3_SCENARIO_NAMES = SCENARIO_NAMES


DEFAULT_LIGHTGBM_VARIANT_SETTINGS = {
    "enabled": False,
    "parallel_workers": 2,
    "num_threads_per_variant": 2,
    "variants": [
        {
            "name": "lightgbm_stable_top6",
            "params": {
                "num_leaves": 63,
                "min_data_in_leaf": 80,
                "feature_fraction": 0.65,
                "lambda_l2": 5.0,
            },
        },
    ],
}

DEFAULT_UPSET_TRAINING_SETTINGS = {
    "enabled": False,
    "history_years": 10.0,
    "recent_years": 3.5,
    "middle_cutoff_years": 7.0,
    "payout_threshold": 10000.0,
    "control_ratio": 3,
    "recent_history_weight": 0.7,
    "older_history_weight": 0.3,
    "payout_weight_10000": 2.0,
    "payout_weight_50000": 3.0,
    "payout_weight_100000": 5.0,
    "random_seed": 42,
}

UPSET_TRAINING_WEIGHT_COLUMN = "_upset_training_weight"

DEFAULT_VALUE_RECOVERY_TRAINING_SETTINGS = {
    "enabled": False,
    "payout_weight_under_1000": 0.7,
    "payout_weight_1000_5000": 1.0,
    "payout_weight_5000_10000": 1.3,
    "payout_weight_10000_30000": 1.8,
    "payout_weight_30000_50000": 1.2,
    "payout_weight_50000_over": 0.6,
}

VALUE_RECOVERY_TRAINING_WEIGHT_COLUMN = "_value_recovery_training_weight"


DEFAULT_LIGHTGBM_SEED_ENSEMBLE_SETTINGS = {
    "enabled": False,
    "seeds": [42, 202, 777],
    "parallel_workers": 2,
    "num_threads_per_seed": 2,
    "params": {},
}


DEFAULT_LIGHTGBM_REGRESSION_VARIANT_SETTINGS = {
    "enabled": False,
    "parallel_workers": 1,
    "num_threads_per_variant": 2,
    "variants": [
        {
            "name": "lightgbm_reg_finish_position",
            "target": "finish_position",
            "score_transform": "negative",
            "params": {
                "objective": "regression",
                "metric": "rmse",
                "num_leaves": 31,
                "min_data_in_leaf": 100,
                "learning_rate": 0.03,
                "feature_fraction": 0.70,
                "bagging_fraction": 0.80,
                "bagging_freq": 1,
                "lambda_l2": 8.0,
            },
        },
    ],
}


DEFAULT_XGBOOST_VARIANT_SETTINGS = {
    "enabled": False,
    "parallel_workers": 1,
    "num_threads_per_variant": 2,
    "variants": [
        {
            "name": "xgboost_pairwise_conservative",
            "params": {
                "objective": "rank:pairwise",
                "eval_metric": "ndcg@12",
                "max_depth": 3,
                "eta": 0.025,
                "min_child_weight": 8.0,
                "subsample": 0.75,
                "colsample_bytree": 0.70,
                "lambda": 8.0,
                "alpha": 1.0,
                "gamma": 0.5,
            },
        },
    ],
}


DEFAULT_XGBOOST_REGRESSION_VARIANT_SETTINGS = {
    "enabled": False,
    "parallel_workers": 1,
    "num_threads_per_variant": 2,
    "variants": [
        {
            "name": "xgboost_reg_finish_position",
            "target": "finish_position",
            "score_transform": "negative",
            "params": {
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "max_depth": 3,
                "eta": 0.025,
                "min_child_weight": 10.0,
                "subsample": 0.75,
                "colsample_bytree": 0.70,
                "lambda": 8.0,
                "alpha": 1.0,
                "gamma": 0.5,
            },
        },
    ],
}


DEFAULT_RANDOM_FOREST_REGRESSION_VARIANT_SETTINGS = {
    "enabled": False,
    "parallel_workers": 1,
    "num_threads_per_variant": 2,
    "variants": [
        {
            "name": "random_forest_reg_finish_position",
            "target": "finish_position",
            "score_transform": "negative",
            "params": {
                "n_estimators": 80,
                "max_depth": 10,
                "min_samples_leaf": 50,
                "max_features": 0.65,
                "bootstrap": True,
                "max_samples": 0.35,
                "random_state": 42,
            },
        },
    ],
}


DEFAULT_RIDGE_REGRESSION_VARIANT_SETTINGS = {
    "enabled": False,
    "parallel_workers": 1,
    "num_threads_per_variant": 1,
    "variants": [
        {
            "name": "ridge_reg_finish_position",
            "target": "finish_position",
            "score_transform": "negative",
            "params": {
                "alpha": 10.0,
            },
        },
    ],
}


DEFAULT_NEURAL_REGRESSION_VARIANT_SETTINGS = {
    "enabled": False,
    "parallel_workers": 1,
    "variants": [
        {
            "name": "mlp_reg_finish_position",
            "model_type": "mlp",
            "target": "finish_position",
            "score_transform": "negative",
            "params": {
                "hidden_units": [256, 128, 64, 32],
                "embedding_dim": 8,
                "dropout": 0.10,
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "epochs": 20,
                "batch_size": 4096,
                "patience": 5,
                "torch_num_threads": 2,
            },
        },
        {
            "name": "tabnet_reg_finish_position",
            "model_type": "tabnet",
            "target": "finish_position",
            "score_transform": "negative",
            "params": {
                "n_d": 16,
                "n_a": 16,
                "n_steps": 4,
                "gamma": 1.5,
                "lambda_sparse": 0.0001,
                "learning_rate": 0.001,
                "max_epochs": 30,
                "batch_size": 4096,
                "virtual_batch_size": 512,
                "patience": 5,
            },
        },
    ],
}


DEFAULT_ENSEMBLE_SETTINGS = {
    "parallel_workers": 1,
    "model_metrics_parallel_workers": 1,
    "max_eval_races": 0,
    "grid_step": 0.10,
    "max_model_weight": 1.0,
    "min_nonzero_weight": 0.0,
    "objective": "trifecta_top3_balanced",
    "objective_top12_weight": 0.05,
    "objective_top5_weight": 0.10,
    "objective_top3_weight": 0.45,
    "objective_boat_top1_weight": 0.35,
    "objective_top1_weight": 0.05,
    "objective_top3_overlap_weight": 0.00,
    "objective_recovery_weight": 0.00,
    "objective_value_hit_weight": 0.00,
    "objective_top12_payout_weight": 0.00,
    "objective_log_loss_weight": 0.00,
    "recovery_score_cap": 0.80,
    "top12_payout_score_cap": 8000.0,
    "min_purchase_rate": 0.10,
    "purchase_rate_penalty_weight": 0.20,
    "value_confidence_calibration": {
        "enabled": True,
    },
    "value_rule": {
        "high": "top3",
        "middle": "top3",
        "low": "skip",
    },
    "cross_validation": {
        "enabled": False,
        "folds": 1,
        "method": "race_id_hash",
    },
}


ENSEMBLE_OBJECTIVES = {
    "rank_legacy",
    "trifecta_fast",
    "trifecta_top3_balanced",
    "trifecta_top12_balanced",
    "trifecta_top12_simple",
    "trifecta_value_balanced",
}


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def get_lightgbm_variant_settings(config: dict | None) -> dict[str, Any]:
    configured = ((config or {}).get("models", {}) or {}).get("lightgbm_variants", {})
    settings = {
        **DEFAULT_LIGHTGBM_VARIANT_SETTINGS,
        **(configured or {}),
    }
    settings["variants"] = list((configured or {}).get("variants", DEFAULT_LIGHTGBM_VARIANT_SETTINGS["variants"]))
    settings["parallel_workers"] = max(int(settings.get("parallel_workers", 1)), 1)
    settings["num_threads_per_variant"] = max(int(settings.get("num_threads_per_variant", 2)), 1)
    return settings


def get_lightgbm_seed_ensemble_settings(config: dict | None) -> dict[str, Any]:
    configured = ((config or {}).get("models", {}) or {}).get("lightgbm_seed_ensemble", {})
    settings = {
        **DEFAULT_LIGHTGBM_SEED_ENSEMBLE_SETTINGS,
        **(configured or {}),
    }
    seeds: list[int] = []
    for seed in settings.get("seeds", DEFAULT_LIGHTGBM_SEED_ENSEMBLE_SETTINGS["seeds"]):
        seed_value = int(seed)
        if seed_value not in seeds:
            seeds.append(seed_value)
    if len(seeds) < 2:
        settings["enabled"] = False
    settings["seeds"] = seeds
    settings["parallel_workers"] = max(int(settings.get("parallel_workers", 1)), 1)
    settings["num_threads_per_seed"] = max(int(settings.get("num_threads_per_seed", 2)), 1)
    settings["params"] = dict((configured or {}).get("params", DEFAULT_LIGHTGBM_SEED_ENSEMBLE_SETTINGS["params"]))
    return settings


def is_lightgbm_seed_ensemble_enabled(config: dict | None) -> bool:
    return bool(get_lightgbm_seed_ensemble_settings(config).get("enabled", False))


def get_lightgbm_regression_variant_settings(config: dict | None) -> dict[str, Any]:
    configured = ((config or {}).get("models", {}) or {}).get("lightgbm_regression_variants", {})
    settings = {
        **DEFAULT_LIGHTGBM_REGRESSION_VARIANT_SETTINGS,
        **(configured or {}),
    }
    settings["variants"] = list(
        (configured or {}).get("variants", DEFAULT_LIGHTGBM_REGRESSION_VARIANT_SETTINGS["variants"])
    )
    settings["parallel_workers"] = max(int(settings.get("parallel_workers", 1)), 1)
    settings["num_threads_per_variant"] = max(int(settings.get("num_threads_per_variant", 2)), 1)
    return settings


def get_catboost_settings(config: dict | None) -> dict[str, Any]:
    configured = ((config or {}).get("models", {}) or {}).get("catboost", {})
    settings = {
        **DEFAULT_CATBOOST_SETTINGS,
        **(configured or {}),
    }
    settings["params"] = dict((configured or {}).get("params", DEFAULT_CATBOOST_SETTINGS["params"]))
    settings["enabled"] = bool(settings.get("enabled", True))
    return settings


def is_catboost_enabled(config: dict | None) -> bool:
    return bool(get_catboost_settings(config).get("enabled", True))


def get_xgboost_variant_settings(config: dict | None) -> dict[str, Any]:
    configured = ((config or {}).get("models", {}) or {}).get("xgboost_variants", {})
    settings = {
        **DEFAULT_XGBOOST_VARIANT_SETTINGS,
        **(configured or {}),
    }
    settings["variants"] = list((configured or {}).get("variants", DEFAULT_XGBOOST_VARIANT_SETTINGS["variants"]))
    settings["parallel_workers"] = max(int(settings.get("parallel_workers", 1)), 1)
    settings["num_threads_per_variant"] = max(int(settings.get("num_threads_per_variant", 2)), 1)
    return settings


def get_xgboost_regression_variant_settings(config: dict | None) -> dict[str, Any]:
    configured = ((config or {}).get("models", {}) or {}).get("xgboost_regression_variants", {})
    settings = {
        **DEFAULT_XGBOOST_REGRESSION_VARIANT_SETTINGS,
        **(configured or {}),
    }
    settings["variants"] = list(
        (configured or {}).get("variants", DEFAULT_XGBOOST_REGRESSION_VARIANT_SETTINGS["variants"])
    )
    settings["parallel_workers"] = max(int(settings.get("parallel_workers", 1)), 1)
    settings["num_threads_per_variant"] = max(int(settings.get("num_threads_per_variant", 2)), 1)
    return settings


def get_random_forest_regression_variant_settings(config: dict | None) -> dict[str, Any]:
    configured = ((config or {}).get("models", {}) or {}).get("random_forest_regression_variants", {})
    settings = {
        **DEFAULT_RANDOM_FOREST_REGRESSION_VARIANT_SETTINGS,
        **(configured or {}),
    }
    settings["variants"] = list(
        (configured or {}).get("variants", DEFAULT_RANDOM_FOREST_REGRESSION_VARIANT_SETTINGS["variants"])
    )
    settings["parallel_workers"] = max(int(settings.get("parallel_workers", 1)), 1)
    settings["num_threads_per_variant"] = max(int(settings.get("num_threads_per_variant", 2)), 1)
    return settings


def get_ridge_regression_variant_settings(config: dict | None) -> dict[str, Any]:
    configured = ((config or {}).get("models", {}) or {}).get("ridge_regression_variants", {})
    settings = {
        **DEFAULT_RIDGE_REGRESSION_VARIANT_SETTINGS,
        **(configured or {}),
    }
    settings["variants"] = list((configured or {}).get("variants", DEFAULT_RIDGE_REGRESSION_VARIANT_SETTINGS["variants"]))
    settings["parallel_workers"] = max(int(settings.get("parallel_workers", 1)), 1)
    settings["num_threads_per_variant"] = max(int(settings.get("num_threads_per_variant", 1)), 1)
    return settings


def get_neural_regression_variant_settings(config: dict | None) -> dict[str, Any]:
    configured = ((config or {}).get("models", {}) or {}).get("neural_regression_variants", {})
    settings = {
        **DEFAULT_NEURAL_REGRESSION_VARIANT_SETTINGS,
        **(configured or {}),
    }
    settings["variants"] = list(
        (configured or {}).get("variants", DEFAULT_NEURAL_REGRESSION_VARIANT_SETTINGS["variants"])
    )
    settings["parallel_workers"] = max(int(settings.get("parallel_workers", 1)), 1)
    return settings


def get_ensemble_settings(config: dict | None) -> dict[str, Any]:
    configured = ((config or {}).get("models", {}) or {}).get("ensemble", {})
    settings = {
        **DEFAULT_ENSEMBLE_SETTINGS,
        **(configured or {}),
    }
    settings["parallel_workers"] = max(int(settings.get("parallel_workers", 1)), 1)
    settings["model_metrics_parallel_workers"] = max(int(settings.get("model_metrics_parallel_workers", 1)), 1)
    settings["max_eval_races"] = max(int(settings.get("max_eval_races", 0)), 0)
    settings["grid_step"] = float(settings.get("grid_step", 0.10))
    if settings["grid_step"] <= 0 or settings["grid_step"] > 1:
        raise ValueError("models.ensemble.grid_step must be in the range (0, 1].")
    settings["max_model_weight"] = float(settings.get("max_model_weight", 1.0))
    if settings["max_model_weight"] <= 0 or settings["max_model_weight"] > 1:
        raise ValueError("models.ensemble.max_model_weight must be in the range (0, 1].")
    settings["min_nonzero_weight"] = float(settings.get("min_nonzero_weight", 0.0))
    if settings["min_nonzero_weight"] < 0 or settings["min_nonzero_weight"] > 1:
        raise ValueError("models.ensemble.min_nonzero_weight must be in the range [0, 1].")
    settings["objective"] = str(settings.get("objective", "trifecta_fast")).strip()
    if settings["objective"] not in ENSEMBLE_OBJECTIVES:
        raise ValueError(f"models.ensemble.objective must be one of {sorted(ENSEMBLE_OBJECTIVES)}.")
    settings["value_rule"] = {
        **DEFAULT_ENSEMBLE_SETTINGS["value_rule"],
        **dict(settings.get("value_rule", {}) or {}),
    }
    settings["value_confidence_calibration"] = {
        **DEFAULT_ENSEMBLE_SETTINGS["value_confidence_calibration"],
        **dict(settings.get("value_confidence_calibration", {}) or {}),
    }
    settings["value_confidence_calibration"]["enabled"] = bool(
        settings["value_confidence_calibration"].get("enabled", True)
    )
    cv_configured = dict((configured or {}).get("cross_validation", {}) or {})
    cv_default = DEFAULT_ENSEMBLE_SETTINGS["cross_validation"]
    cross_validation = {**cv_default, **cv_configured}
    cross_validation["enabled"] = bool(cross_validation.get("enabled", False))
    cross_validation["folds"] = max(int(cross_validation.get("folds", 1)), 1)
    cross_validation["method"] = str(cross_validation.get("method", "race_id_hash")).strip()
    if cross_validation["method"] != "race_id_hash":
        raise ValueError("models.ensemble.cross_validation.method must be race_id_hash.")
    if not cross_validation["enabled"]:
        cross_validation["folds"] = 1
    settings["cross_validation"] = cross_validation
    return settings


def get_enabled_lightgbm_variants(config: dict | None) -> list[dict[str, Any]]:
    settings = get_lightgbm_variant_settings(config)
    if not bool(settings.get("enabled", False)):
        return []
    variants = [dict(variant) for variant in settings.get("variants", []) if bool(variant.get("enabled", True))]
    seen: set[str] = set()
    for variant in variants:
        name = str(variant.get("name", "")).strip()
        if not name:
            raise ValueError("LightGBM variant name must not be empty.")
        if name in RESERVED_MODEL_NAMES:
            raise ValueError(f"LightGBM variant name conflicts with reserved model name: {name}")
        if not LIGHTGBM_VARIANT_NAME_RE.match(name):
            raise ValueError(f"Invalid LightGBM variant name: {name}")
        if name in seen:
            raise ValueError(f"Duplicate LightGBM variant name: {name}")
        seen.add(name)
        variant["name"] = name
        variant["feature_set"] = str(variant.get("feature_set", "full")).strip() or "full"
        variant["params"] = dict(variant.get("params", {}) or {})
        if "upset_training" in variant:
            upset_training = {
                **DEFAULT_UPSET_TRAINING_SETTINGS,
                **dict(variant.get("upset_training", {}) or {}),
            }
            upset_training["enabled"] = bool(upset_training.get("enabled", False))
            upset_training["history_years"] = float(upset_training.get("history_years", 10.0))
            upset_training["recent_years"] = float(upset_training.get("recent_years", 3.5))
            upset_training["middle_cutoff_years"] = float(upset_training.get("middle_cutoff_years", 7.0))
            upset_training["payout_threshold"] = float(upset_training.get("payout_threshold", 10000.0))
            upset_training["control_ratio"] = max(int(upset_training.get("control_ratio", 3)), 0)
            if upset_training["history_years"] <= upset_training["recent_years"]:
                raise ValueError("upset_training.history_years must be greater than recent_years.")
            if not upset_training["recent_years"] < upset_training["middle_cutoff_years"] <= upset_training["history_years"]:
                raise ValueError(
                    "upset_training.middle_cutoff_years must be greater than recent_years "
                    "and no greater than history_years."
                )
            if upset_training["payout_threshold"] <= 0:
                raise ValueError("upset_training.payout_threshold must be positive.")
            variant["upset_training"] = upset_training
        if "value_recovery_training" in variant:
            value_recovery_training = {
                **DEFAULT_VALUE_RECOVERY_TRAINING_SETTINGS,
                **dict(variant.get("value_recovery_training", {}) or {}),
            }
            value_recovery_training["enabled"] = bool(value_recovery_training.get("enabled", False))
            for key in DEFAULT_VALUE_RECOVERY_TRAINING_SETTINGS:
                if key == "enabled":
                    continue
                value_recovery_training[key] = float(value_recovery_training.get(key, 1.0))
                if value_recovery_training[key] < 0:
                    raise ValueError(f"value_recovery_training.{key} must be non-negative.")
            variant["value_recovery_training"] = value_recovery_training
    return variants


def get_enabled_lightgbm_regression_variants(config: dict | None) -> list[dict[str, Any]]:
    settings = get_lightgbm_regression_variant_settings(config)
    if not bool(settings.get("enabled", False)):
        return []
    variants = [dict(variant) for variant in settings.get("variants", []) if bool(variant.get("enabled", True))]
    seen: set[str] = set()
    for variant in variants:
        name = str(variant.get("name", "")).strip()
        if not name:
            raise ValueError("LightGBM regression variant name must not be empty.")
        if name in RESERVED_MODEL_NAMES:
            raise ValueError(f"LightGBM regression variant name conflicts with reserved model name: {name}")
        if not LIGHTGBM_VARIANT_NAME_RE.match(name) or not name.startswith("lightgbm_reg_"):
            raise ValueError(f"Invalid LightGBM regression variant name: {name}")
        if name in seen:
            raise ValueError(f"Duplicate LightGBM regression variant name: {name}")
        target = str(variant.get("target", "finish_position")).strip()
        if target not in {"finish_position"}:
            raise ValueError(f"Unsupported LightGBM regression variant target: {target}")
        score_transform = str(variant.get("score_transform", "negative")).strip()
        if score_transform not in {"negative"}:
            raise ValueError(f"Unsupported LightGBM regression score_transform: {score_transform}")
        seen.add(name)
        variant["name"] = name
        variant["target"] = target
        variant["score_transform"] = score_transform
        variant["params"] = dict(variant.get("params", {}) or {})
    return variants


def get_enabled_xgboost_variants(config: dict | None) -> list[dict[str, Any]]:
    settings = get_xgboost_variant_settings(config)
    if not bool(settings.get("enabled", False)):
        return []
    variants = [dict(variant) for variant in settings.get("variants", []) if bool(variant.get("enabled", True))]
    seen: set[str] = set()
    for variant in variants:
        name = str(variant.get("name", "")).strip()
        if not name:
            raise ValueError("XGBoost variant name must not be empty.")
        if name in RESERVED_MODEL_NAMES:
            raise ValueError(f"XGBoost variant name conflicts with reserved model name: {name}")
        if not XGBOOST_VARIANT_NAME_RE.match(name):
            raise ValueError(f"Invalid XGBoost variant name: {name}")
        if name in seen:
            raise ValueError(f"Duplicate XGBoost variant name: {name}")
        seen.add(name)
        variant["name"] = name
        variant["params"] = dict(variant.get("params", {}) or {})
    return variants


def get_enabled_xgboost_regression_variants(config: dict | None) -> list[dict[str, Any]]:
    settings = get_xgboost_regression_variant_settings(config)
    if not bool(settings.get("enabled", False)):
        return []
    variants = [dict(variant) for variant in settings.get("variants", []) if bool(variant.get("enabled", True))]
    seen: set[str] = set()
    for variant in variants:
        name = str(variant.get("name", "")).strip()
        if not name:
            raise ValueError("XGBoost regression variant name must not be empty.")
        if name in RESERVED_MODEL_NAMES:
            raise ValueError(f"XGBoost regression variant name conflicts with reserved model name: {name}")
        if not XGBOOST_VARIANT_NAME_RE.match(name) or not name.startswith("xgboost_reg_"):
            raise ValueError(f"Invalid XGBoost regression variant name: {name}")
        if name in seen:
            raise ValueError(f"Duplicate XGBoost regression variant name: {name}")
        target = str(variant.get("target", "finish_position")).strip()
        if target not in {"finish_position"}:
            raise ValueError(f"Unsupported XGBoost regression variant target: {target}")
        score_transform = str(variant.get("score_transform", "negative")).strip()
        if score_transform not in {"negative"}:
            raise ValueError(f"Unsupported XGBoost regression score_transform: {score_transform}")
        seen.add(name)
        variant["name"] = name
        variant["target"] = target
        variant["score_transform"] = score_transform
        variant["params"] = dict(variant.get("params", {}) or {})
    return variants


def get_enabled_random_forest_regression_variants(config: dict | None) -> list[dict[str, Any]]:
    settings = get_random_forest_regression_variant_settings(config)
    if not bool(settings.get("enabled", False)):
        return []
    variants = [dict(variant) for variant in settings.get("variants", []) if bool(variant.get("enabled", True))]
    seen: set[str] = set()
    for variant in variants:
        name = str(variant.get("name", "")).strip()
        if not name:
            raise ValueError("Random forest regression variant name must not be empty.")
        if name in RESERVED_MODEL_NAMES:
            raise ValueError(f"Random forest regression variant name conflicts with reserved model name: {name}")
        if not RANDOM_FOREST_VARIANT_NAME_RE.match(name) or not name.startswith("random_forest_reg_"):
            raise ValueError(f"Invalid random forest regression variant name: {name}")
        if name in seen:
            raise ValueError(f"Duplicate random forest regression variant name: {name}")
        target = str(variant.get("target", "finish_position")).strip()
        if target not in {"finish_position"}:
            raise ValueError(f"Unsupported random forest regression variant target: {target}")
        score_transform = str(variant.get("score_transform", "negative")).strip()
        if score_transform not in {"negative"}:
            raise ValueError(f"Unsupported random forest regression score_transform: {score_transform}")
        seen.add(name)
        variant["name"] = name
        variant["target"] = target
        variant["score_transform"] = score_transform
        variant["params"] = dict(variant.get("params", {}) or {})
    return variants


def get_enabled_ridge_regression_variants(config: dict | None) -> list[dict[str, Any]]:
    settings = get_ridge_regression_variant_settings(config)
    if not bool(settings.get("enabled", False)):
        return []
    variants = [dict(variant) for variant in settings.get("variants", []) if bool(variant.get("enabled", True))]
    seen: set[str] = set()
    for variant in variants:
        name = str(variant.get("name", "")).strip()
        if not name:
            raise ValueError("Ridge regression variant name must not be empty.")
        if name in RESERVED_MODEL_NAMES:
            raise ValueError(f"Ridge regression variant name conflicts with reserved model name: {name}")
        if not RIDGE_VARIANT_NAME_RE.match(name) or not name.startswith("ridge_reg_"):
            raise ValueError(f"Invalid ridge regression variant name: {name}")
        if name in seen:
            raise ValueError(f"Duplicate ridge regression variant name: {name}")
        target = str(variant.get("target", "finish_position")).strip()
        if target not in {"finish_position"}:
            raise ValueError(f"Unsupported ridge regression variant target: {target}")
        score_transform = str(variant.get("score_transform", "negative")).strip()
        if score_transform not in {"negative"}:
            raise ValueError(f"Unsupported ridge regression score_transform: {score_transform}")
        seen.add(name)
        variant["name"] = name
        variant["target"] = target
        variant["score_transform"] = score_transform
        variant["params"] = dict(variant.get("params", {}) or {})
    return variants


def get_enabled_neural_regression_variants(config: dict | None) -> list[dict[str, Any]]:
    settings = get_neural_regression_variant_settings(config)
    if not bool(settings.get("enabled", False)):
        return []
    variants = [dict(variant) for variant in settings.get("variants", []) if bool(variant.get("enabled", True))]
    seen: set[str] = set()
    for variant in variants:
        name = str(variant.get("name", "")).strip()
        if not name:
            raise ValueError("Neural regression variant name must not be empty.")
        if name in RESERVED_MODEL_NAMES:
            raise ValueError(f"Neural regression variant name conflicts with reserved model name: {name}")
        if not NEURAL_VARIANT_NAME_RE.match(name):
            raise ValueError(f"Invalid neural regression variant name: {name}")
        if name in seen:
            raise ValueError(f"Duplicate neural regression variant name: {name}")
        model_type = str(variant.get("model_type", "mlp")).strip()
        if model_type not in {"mlp", "tabnet"}:
            raise ValueError(f"Unsupported neural regression model_type: {model_type}")
        target = str(variant.get("target", "finish_position")).strip()
        if target not in {"finish_position"}:
            raise ValueError(f"Unsupported neural regression variant target: {target}")
        score_transform = str(variant.get("score_transform", "negative")).strip()
        if score_transform not in {"negative"}:
            raise ValueError(f"Unsupported neural regression score_transform: {score_transform}")
        seen.add(name)
        variant["name"] = name
        variant["model_type"] = model_type
        variant["target"] = target
        variant["score_transform"] = score_transform
        variant["params"] = dict(variant.get("params", {}) or {})
        if "upset_training" in variant:
            upset_training = {
                **DEFAULT_UPSET_TRAINING_SETTINGS,
                **dict(variant.get("upset_training", {}) or {}),
            }
            upset_training["enabled"] = bool(upset_training.get("enabled", False))
            upset_training["history_years"] = float(upset_training.get("history_years", 10.0))
            upset_training["recent_years"] = float(upset_training.get("recent_years", 3.5))
            upset_training["middle_cutoff_years"] = float(upset_training.get("middle_cutoff_years", 7.0))
            upset_training["payout_threshold"] = float(upset_training.get("payout_threshold", 10000.0))
            upset_training["control_ratio"] = max(int(upset_training.get("control_ratio", 3)), 0)
            if upset_training["history_years"] <= upset_training["recent_years"]:
                raise ValueError("upset_training.history_years must be greater than recent_years.")
            if not upset_training["recent_years"] < upset_training["middle_cutoff_years"] <= upset_training["history_years"]:
                raise ValueError(
                    "upset_training.middle_cutoff_years must be greater than recent_years "
                    "and no greater than history_years."
                )
            if upset_training["payout_threshold"] <= 0:
                raise ValueError("upset_training.payout_threshold must be positive.")
            variant["upset_training"] = upset_training
        if "value_recovery_training" in variant:
            value_recovery_training = {
                **DEFAULT_VALUE_RECOVERY_TRAINING_SETTINGS,
                **dict(variant.get("value_recovery_training", {}) or {}),
            }
            value_recovery_training["enabled"] = bool(value_recovery_training.get("enabled", False))
            for key in DEFAULT_VALUE_RECOVERY_TRAINING_SETTINGS:
                if key == "enabled":
                    continue
                value_recovery_training[key] = float(value_recovery_training.get(key, 1.0))
                if value_recovery_training[key] < 0:
                    raise ValueError(f"value_recovery_training.{key} must be non-negative.")
            variant["value_recovery_training"] = value_recovery_training
    return variants


def is_lightgbm_model_name(model_name: str) -> bool:
    return model_name == "lightgbm" or model_name.startswith("lightgbm_")


def is_lightgbm_seed_ensemble_model_name(model_name: str) -> bool:
    return model_name == LIGHTGBM_SEED_ENSEMBLE_MODEL_NAME


def is_lightgbm_regression_model_name(model_name: str) -> bool:
    return model_name.startswith("lightgbm_reg_")


def is_xgboost_model_name(model_name: str) -> bool:
    return model_name == "xgboost" or model_name.startswith("xgboost_")


def is_xgboost_regression_model_name(model_name: str) -> bool:
    return model_name.startswith("xgboost_reg_")


def is_random_forest_model_name(model_name: str) -> bool:
    return model_name.startswith("random_forest_")


def is_random_forest_regression_model_name(model_name: str) -> bool:
    return model_name.startswith("random_forest_reg_")


def is_ridge_model_name(model_name: str) -> bool:
    return model_name.startswith("ridge_")


def is_ridge_regression_model_name(model_name: str) -> bool:
    return model_name.startswith("ridge_reg_")


def is_neural_model_name(model_name: str) -> bool:
    return model_name.startswith("mlp_") or model_name.startswith("tabnet_")


def is_neural_regression_model_name(model_name: str) -> bool:
    return model_name.startswith("mlp_reg_") or model_name.startswith("tabnet_reg_")


def lightgbm_variant_model_path(base_path: Path, variant_name: str) -> Path:
    if variant_name == "lightgbm":
        return base_path
    suffix = variant_name.removeprefix("lightgbm_")
    return base_path.with_name(f"{base_path.stem}_{suffix}{base_path.suffix}")


def lightgbm_seed_model_path(base_path: Path, seed: int) -> Path:
    return base_path.with_name(f"{base_path.stem}_seed_{int(seed)}{base_path.suffix}")


def xgboost_variant_model_path(base_path: Path, variant_name: str) -> Path:
    if variant_name == "xgboost":
        return base_path
    suffix = variant_name.removeprefix("xgboost_")
    return base_path.with_name(f"{base_path.stem}_{suffix}{base_path.suffix}")


def random_forest_variant_model_path(base_path: Path, variant_name: str) -> Path:
    suffix = variant_name.removeprefix("random_forest_")
    return base_path.with_name(f"{base_path.stem}_{suffix}{base_path.suffix}")


def ridge_variant_model_path(base_path: Path, variant_name: str) -> Path:
    suffix = variant_name.removeprefix("ridge_")
    return base_path.with_name(f"{base_path.stem}_{suffix}{base_path.suffix}")


def neural_variant_model_path(base_path: Path, variant_name: str) -> Path:
    return base_path.with_name(f"{base_path.stem}_{variant_name}{base_path.suffix}")


def get_artifact_paths(config: dict) -> dict[str, Path]:
    artifacts = config.get("artifacts", {})
    if not artifacts:
        artifacts = (config.get("models", {}) or {}).get("artifacts", {})
    if not artifacts:
        artifacts = (config.get("model", {}) or {}).get("artifacts", {})
    paths = {
        name: Path(artifacts.get(name, default_path))
        for name, default_path in DEFAULT_ARTIFACT_PATHS.items()
    }
    metrics_path = (
        config.get("metrics_path")
        or (config.get("models", {}) or {}).get("metrics_path")
        or (config.get("model", {}) or {}).get("metrics_path")
    )
    if metrics_path:
        paths["metrics_path"] = Path(metrics_path)
    return paths


def cleanup_processed_intermediate_dirs(config: dict) -> list[Path]:
    data_config = config.get("data", {})
    processed_dir_value = data_config.get("processed_dir")
    if processed_dir_value:
        processed_dir = Path(processed_dir_value)
    else:
        training_table = data_config.get("training_table")
        processed_dir = Path(training_table).parent if training_table else Path("data/processed")

    removed: list[Path] = []
    for folder_name in ("base_buckets", "history_months"):
        target = processed_dir / folder_name
        if target.exists() and target.is_dir():
            shutil.rmtree(target)
            removed.append(target)
    return removed


def get_phase3_settings(config: dict | None = None) -> dict[str, Any]:
    phase3 = (config or {}).get("phase3", {})
    settings = json.loads(json.dumps(DEFAULT_PHASE3_SETTINGS))
    for section, defaults in DEFAULT_PHASE3_SETTINGS.items():
        incoming = phase3.get(section, {})
        if isinstance(defaults, dict) and isinstance(incoming, dict):
            settings[section].update(incoming)
    return settings


def get_default_rerank_top_n(config: dict | None = None) -> int:
    top_n_grid = get_phase3_settings(config)["rerank"].get("top_n_grid", [10])
    if not top_n_grid:
        return 10
    return int(top_n_grid[0])


def infer_latest_available_race_date(training_table: pd.DataFrame) -> pd.Timestamp | None:
    if "race_date" not in training_table.columns or training_table.empty:
        return None
    race_dates = pd.to_datetime(training_table["race_date"], errors="coerce")
    if race_dates.dropna().empty:
        return None
    return pd.Timestamp(race_dates.max()).normalize()


def _parse_positive_window_int(value: Any) -> int | None:
    if value in (None, "", 0, "0"):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _parse_positive_window_float(value: Any) -> float | None:
    if value in (None, "", 0, "0"):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _years_to_date_offset(years: float) -> pd.DateOffset:
    # Use months so fractional year windows such as 3.5 years are deterministic.
    return pd.DateOffset(months=max(1, int(round(years * 12))))


def collect_garbage() -> None:
    gc.collect()


def _is_random_by_race_split(config: dict) -> bool:
    method = str(config.get("split", {}).get("method", "")).strip().lower()
    return method in {"random_by_race", "race_random", "random"}


def split_training_frame_random_by_race(
    frame: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        return frame.copy(), frame.copy(), frame.copy()
    if "race_id" not in frame.columns:
        raise ValueError("random_by_race split requires race_id column.")

    split_config = config.get("split", {})
    train_ratio = float(split_config.get("train_ratio", 6) or 6)
    valid_ratio = float(split_config.get("valid_ratio", 1) or 1)
    if train_ratio <= 0 or valid_ratio <= 0:
        raise ValueError("split.train_ratio and split.valid_ratio must be positive.")

    random_seed = int(split_config.get("random_seed", config.get("model", {}).get("random_seed", 42)) or 42)
    race_ids = pd.Series(frame["race_id"].dropna().astype(str).unique())
    if race_ids.empty:
        return frame.copy(), frame.iloc[0:0].copy(), frame.iloc[0:0].copy()

    shuffled = race_ids.to_numpy(copy=True)
    np.random.default_rng(random_seed).shuffle(shuffled)
    valid_fraction = valid_ratio / (train_ratio + valid_ratio)
    valid_count = int(round(len(shuffled) * valid_fraction))
    if len(shuffled) > 1:
        valid_count = min(max(valid_count, 1), len(shuffled) - 1)
    else:
        valid_count = 0

    valid_ids = set(shuffled[:valid_count].tolist())
    race_id_text = frame["race_id"].astype(str)
    valid_mask = race_id_text.isin(valid_ids)
    train_df = frame.loc[~valid_mask].copy()
    valid_df = frame.loc[valid_mask].copy()
    test_df = frame.iloc[0:0].copy()
    return train_df, valid_df, test_df


def load_training_splits_from_parquet(
    training_table_path: Path,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    latest_race_dates = pd.read_parquet(training_table_path, columns=["race_date"])
    synced_config = with_latest_available_dates(config, infer_latest_available_race_date(latest_race_dates))
    del latest_race_dates
    collect_garbage()

    data_config = synced_config.get("data", {})
    min_date = pd.Timestamp(data_config.get("min_date")) if data_config.get("min_date") else None
    max_date = pd.Timestamp(data_config.get("max_date")) if data_config.get("max_date") else None
    if _is_random_by_race_split(synced_config):
        filters = []
        if min_date is not None:
            filters.append(("race_date", ">=", min_date))
        if max_date is not None:
            filters.append(("race_date", "<=", max_date))
        frame = pd.read_parquet(training_table_path, filters=filters or None)
        frame = prepare_training_table(frame, synced_config)
        train_df, valid_df, test_df = split_training_frame_random_by_race(frame, synced_config)
        del frame
        collect_garbage()
        return train_df, valid_df, test_df, synced_config

    train_end = pd.Timestamp(synced_config["split"]["train_end_date"])
    valid_end = pd.Timestamp(synced_config["split"]["valid_end_date"])

    train_start = min_date if min_date is not None else pd.Timestamp("1900-01-01")
    train_stop = min(train_end, max_date) if max_date is not None else train_end
    valid_start = max(train_end + pd.Timedelta(days=1), min_date) if min_date is not None else train_end + pd.Timedelta(days=1)
    valid_stop = min(valid_end, max_date) if max_date is not None else valid_end
    test_start = max(valid_end + pd.Timedelta(days=1), min_date) if min_date is not None else valid_end + pd.Timedelta(days=1)
    test_stop = max_date

    train_df = pd.read_parquet(
        training_table_path,
        filters=[("race_date", ">=", train_start), ("race_date", "<=", train_stop)],
    )
    valid_df = (
        pd.read_parquet(
            training_table_path,
            filters=[("race_date", ">=", valid_start), ("race_date", "<=", valid_stop)],
        )
        if valid_start <= valid_stop
        else pd.DataFrame()
    )
    if test_stop is not None and test_start <= test_stop:
        test_df = pd.read_parquet(
            training_table_path,
            filters=[("race_date", ">=", test_start), ("race_date", "<=", test_stop)],
        )
    else:
        test_df = pd.DataFrame()

    train_df = prepare_training_table(train_df, synced_config)
    valid_df = prepare_training_table(valid_df, synced_config) if not valid_df.empty else valid_df
    test_df = prepare_training_table(test_df, synced_config) if not test_df.empty else test_df
    collect_garbage()
    return train_df, valid_df, test_df, synced_config


def with_latest_available_dates(config: dict, latest_race_date: pd.Timestamp | str | None) -> dict:
    if latest_race_date is None:
        return config

    synced = json.loads(json.dumps(config))
    latest_ts = pd.Timestamp(latest_race_date).normalize()
    latest_str = latest_ts.strftime("%Y-%m-%d")

    synced.setdefault("data", {})
    synced["data"]["max_date"] = latest_str

    rolling_years = _parse_positive_window_float(synced["data"].get("rolling_years"))
    if rolling_years is not None:
        min_ts = (latest_ts - _years_to_date_offset(rolling_years) + pd.Timedelta(days=1)).normalize()
        synced["data"]["min_date"] = min_ts.strftime("%Y-%m-%d")

    synced.setdefault("split", {})
    synced["split"]["valid_end_date"] = latest_str
    valid_months = _parse_positive_window_int(synced["split"].get("valid_months"))
    if valid_months is not None:
        train_end_ts = (latest_ts - pd.DateOffset(months=valid_months)).normalize()
        min_date = synced["data"].get("min_date")
        if min_date:
            train_end_ts = max(train_end_ts, pd.Timestamp(min_date))
        synced["split"]["train_end_date"] = train_end_ts.strftime("%Y-%m-%d")
    return synced


def is_trifecta_v2_bundle(model: Any) -> bool:
    return isinstance(model, dict) and "model_type" in model


def save_trifecta_v2_model_artifact(model: Any, path: Path) -> None:
    if model is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = model
    if is_trifecta_v2_bundle(model) and isinstance(model.get("booster"), lgb.Booster):
        payload = dict(model)
        payload["booster_model_str"] = model["booster"].model_to_string()
        payload.pop("booster", None)
        if "phase3_second_model" in payload and isinstance(payload.get("phase3_second_model"), lgb.Booster):
            payload["phase3_second_model_str"] = payload["phase3_second_model"].model_to_string()
            payload.pop("phase3_second_model", None)
        if "phase3_third_model" in payload and isinstance(payload.get("phase3_third_model"), lgb.Booster):
            payload["phase3_third_model_str"] = payload["phase3_third_model"].model_to_string()
            payload.pop("phase3_third_model", None)
        if "phase3_pattern_model" in payload and isinstance(payload.get("phase3_pattern_model"), lgb.Booster):
            payload["phase3_pattern_model_str"] = payload["phase3_pattern_model"].model_to_string()
            payload.pop("phase3_pattern_model", None)
    joblib.dump(payload, path)


def load_trifecta_v2_model_artifact_payload(path: Path) -> Any | None:
    if not path.exists():
        return None
    payload = joblib.load(path)
    if is_trifecta_v2_bundle(payload):
        restored = dict(payload)
        if "booster_model_str" in restored and "booster" not in restored:
            restored["booster"] = lgb.Booster(model_str=restored.pop("booster_model_str"))
        if "phase3_second_model_str" in restored and "phase3_second_model" not in restored:
            restored["phase3_second_model"] = lgb.Booster(model_str=restored.pop("phase3_second_model_str"))
        if "phase3_third_model_str" in restored and "phase3_third_model" not in restored:
            restored["phase3_third_model"] = lgb.Booster(model_str=restored.pop("phase3_third_model_str"))
        if "phase3_pattern_model_str" in restored and "phase3_pattern_model" not in restored:
            restored["phase3_pattern_model"] = lgb.Booster(model_str=restored.pop("phase3_pattern_model_str"))
        return restored
    return payload


def predict_trifecta_v2_scores(model: Any, features: pd.DataFrame) -> np.ndarray:
    if model is None:
        return np.zeros(len(features), dtype=float)
    if is_trifecta_v2_bundle(model):
        model_type = model.get("model_type")
        required = model.get("feature_names", list(features.columns))
        aligned = features.reindex(columns=required, fill_value=0.0)
        if model_type == "lgbm_ranker":
            return np.asarray(model["booster"].predict(aligned), dtype=float)
        if model_type == "lgbm_binary":
            return np.asarray(model["booster"].predict(aligned), dtype=float)
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(features)[:, 1], dtype=float)
    if hasattr(model, "predict"):
        return np.asarray(model.predict(features), dtype=float)
    raise TypeError("Unsupported trifecta v2 model artifact")


def get_conservative_rerank_weight(model: Any) -> float:
    if is_trifecta_v2_bundle(model):
        return float(
            model.get(
                "conservative_v1_weight",
                DEFAULT_PHASE3_SETTINGS["rerank"]["default_conservative_weight"],
            )
        )
    return float(DEFAULT_PHASE3_SETTINGS["rerank"]["default_conservative_weight"])


def get_rank_penalty_strength(model: Any) -> float:
    if is_trifecta_v2_bundle(model):
        return float(
            model.get(
                "rank_penalty_strength",
                DEFAULT_PHASE3_SETTINGS["rerank"]["default_rank_penalty_strength"],
            )
        )
    return float(DEFAULT_PHASE3_SETTINGS["rerank"]["default_rank_penalty_strength"])


def get_rank_penalty_start(model: Any) -> int:
    if is_trifecta_v2_bundle(model):
        return int(
            model.get(
                "rank_penalty_start",
                DEFAULT_PHASE3_SETTINGS["rerank"]["rank_penalty_start"],
            )
        )
    return int(DEFAULT_PHASE3_SETTINGS["rerank"]["rank_penalty_start"])


def get_rerank_top_n(model: Any, fallback: int | None = None) -> int | None:
    if is_trifecta_v2_bundle(model) and model.get("rerank_top_n") is not None:
        return int(model["rerank_top_n"])
    return None if fallback is None else int(fallback)


def get_scenario_candidate_top_n(model: Any) -> int:
    if is_trifecta_v2_bundle(model):
        return max(int(model.get("scenario_candidate_top_n", 0) or 0), 0)
    return 0


def get_dynamic_rerank_weight_metadata(model: Any) -> dict[str, Any]:
    if is_trifecta_v2_bundle(model) and isinstance(model.get("dynamic_rerank_weight"), dict):
        return dict(model["dynamic_rerank_weight"])
    return {"enabled": False, "rules": [], "default_weight": get_conservative_rerank_weight(model)}


def with_conservative_rerank_weight(model: Any, weight: float) -> Any:
    if not is_trifecta_v2_bundle(model):
        return model
    updated = dict(model)
    updated["conservative_v1_weight"] = float(weight)
    return updated


def with_dynamic_rerank_weight_metadata(model: Any, metadata: dict[str, Any] | None) -> Any:
    if not is_trifecta_v2_bundle(model):
        return model
    updated = dict(model)
    updated["dynamic_rerank_weight"] = metadata or {
        "enabled": False,
        "rules": [],
        "default_weight": get_conservative_rerank_weight(model),
    }
    return updated


def with_rank_penalty_settings(model: Any, strength: float, start_rank: int) -> Any:
    if not is_trifecta_v2_bundle(model):
        return model
    updated = dict(model)
    updated["rank_penalty_strength"] = float(strength)
    updated["rank_penalty_start"] = int(start_rank)
    return updated


def with_rerank_top_n(model: Any, top_n: int) -> Any:
    if not is_trifecta_v2_bundle(model):
        return model
    updated = dict(model)
    updated["rerank_top_n"] = int(top_n)
    return updated


def with_calibration_window_days(model: Any, window_days: int) -> Any:
    if not is_trifecta_v2_bundle(model):
        return model
    updated = dict(model)
    updated["calibration_window_days"] = int(window_days)
    return updated


def with_phase3_optimization_metadata(
    model: Any,
    rerank_optimization: dict[str, Any] | None = None,
    calibration_optimization: dict[str, Any] | None = None,
) -> Any:
    if not is_trifecta_v2_bundle(model):
        return model
    updated = dict(model)
    updated["phase3_optimization"] = {
        "rerank": rerank_optimization or {},
        "calibration": calibration_optimization or {},
    }
    return updated


def blend_conservative_rerank_scores(
    v1_scores: np.ndarray,
    rerank_scores: np.ndarray,
    conservative_weight: float,
    rank_penalty_strength: float = 0.0,
    rank_penalty_start: int = 5,
) -> np.ndarray:
    base = np.asarray(v1_scores, dtype=float)
    update = np.asarray(rerank_scores, dtype=float)
    if len(base) == 0:
        return update
    conservative_weight = float(np.clip(conservative_weight, 0.0, 1.0))
    if np.allclose(update, update[0]):
        return base
    base_rank_order = pd.Series(base).rank(ascending=False, method="first").to_numpy(dtype=float)
    denom = float(max(len(base) - 1, 1))
    base_rank = pd.Series(base).rank(ascending=False, method="average").to_numpy(dtype=float)
    update_rank = pd.Series(update).rank(ascending=False, method="average").to_numpy(dtype=float)
    base_rank_score = 1.0 - ((base_rank - 1.0) / denom)
    update_rank_score = 1.0 - ((update_rank - 1.0) / denom)
    rank_penalty_strength = float(max(rank_penalty_strength, 0.0))
    if rank_penalty_strength > 0.0 and len(base_rank_order) > 0:
        penalty_denom = float(max(len(base_rank_order) - int(rank_penalty_start), 1))
        overflow = np.clip((base_rank_order - float(rank_penalty_start)) / penalty_denom, 0.0, 1.0)
        update_rank_score = np.clip(update_rank_score - (overflow * rank_penalty_strength), 0.0, 1.0)
    return conservative_weight * base_rank_score + (1.0 - conservative_weight) * update_rank_score


@dataclass(frozen=True)
class FastRerankRacePayload:
    race_id: str
    raw_v1: np.ndarray
    base_rank_score: np.ndarray
    update_rank_score: np.ndarray
    rank_penalty_overflow: np.ndarray
    actual_index: int
    flat_update: bool


def _normalize_fast_scores(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    total = float(values.sum())
    if total <= 0 or len(values) == 0:
        return np.full(len(values), 1.0 / max(len(values), 1), dtype=float)
    return values / total


def _stable_desc_rank_position(values: np.ndarray, actual_index: int) -> int | None:
    if actual_index < 0 or actual_index >= len(values):
        return None
    order = np.argsort(-np.asarray(values, dtype=float), kind="mergesort")
    positions = np.empty(len(order), dtype=int)
    positions[order] = np.arange(len(order), dtype=int)
    return int(positions[int(actual_index)])


def _fast_rank_components(
    raw_v1: np.ndarray,
    rerank_scores: np.ndarray,
    rank_penalty_start: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    base = np.asarray(raw_v1, dtype=float)
    update = np.asarray(rerank_scores, dtype=float)
    if len(base) == 0:
        empty = np.asarray([], dtype=float)
        return empty, empty, empty, True
    flat_update = bool(np.allclose(update, update[0])) if len(update) else True
    denom = float(max(len(base) - 1, 1))
    base_rank_order = pd.Series(base).rank(ascending=False, method="first").to_numpy(dtype=float)
    base_rank = pd.Series(base).rank(ascending=False, method="average").to_numpy(dtype=float)
    update_rank = pd.Series(update).rank(ascending=False, method="average").to_numpy(dtype=float)
    base_rank_score = 1.0 - ((base_rank - 1.0) / denom)
    update_rank_score = 1.0 - ((update_rank - 1.0) / denom)
    penalty_denom = float(max(len(base_rank_order) - int(rank_penalty_start), 1))
    overflow = np.clip((base_rank_order - float(rank_penalty_start)) / penalty_denom, 0.0, 1.0)
    return base_rank_score, update_rank_score, overflow, flat_update


def _fast_rerank_scores(
    payload: FastRerankRacePayload,
    conservative_weight: float,
    rank_penalty_strength: float,
) -> np.ndarray:
    if payload.flat_update:
        return np.asarray(payload.raw_v1, dtype=float)
    weight = float(np.clip(conservative_weight, 0.0, 1.0))
    penalty = float(max(rank_penalty_strength, 0.0))
    update_score = np.clip(
        payload.update_rank_score - (payload.rank_penalty_overflow * penalty),
        0.0,
        1.0,
    )
    return weight * payload.base_rank_score + (1.0 - weight) * update_score


def _evaluate_fast_rerank_payloads(
    payloads: list[FastRerankRacePayload],
    conservative_weight: float | None = None,
    rank_penalty_strength: float = 0.0,
    use_v2: bool = False,
) -> dict[str, Any]:
    race_count = len(payloads)
    top_hits = {1: 0, 3: 0, 5: 0, 10: 0, 12: 0}
    covered = 0
    rerank_top1 = 0
    rerank_mrr = 0.0
    baseline_mrr = 0.0
    mean_rank_improvement = 0.0
    log_losses: list[float] = []
    brier_scores: list[float] = []
    actual_probabilities: list[float] = []
    top_probabilities: list[float] = []

    for payload in payloads:
        if use_v2:
            scores = _fast_rerank_scores(
                payload,
                float(conservative_weight if conservative_weight is not None else 1.0),
                rank_penalty_strength,
            )
        else:
            scores = np.asarray(payload.raw_v1, dtype=float)
        probs = _normalize_fast_scores(scores)
        actual_rank = _stable_desc_rank_position(probs, payload.actual_index)
        if actual_rank is None:
            continue

        covered += 1
        actual_probability = max(float(probs[payload.actual_index]), 1e-15)
        labels = np.zeros(len(probs), dtype=float)
        labels[payload.actual_index] = 1.0
        for top_k in top_hits:
            top_hits[top_k] += int(actual_rank < top_k)
        log_losses.append(-np.log(actual_probability))
        brier_scores.append(float(np.mean((probs - labels) ** 2)))
        actual_probabilities.append(actual_probability)
        top_probabilities.append(float(probs[np.argsort(-probs, kind="mergesort")[0]]))

        baseline_probs = _normalize_fast_scores(payload.raw_v1)
        baseline_rank = _stable_desc_rank_position(baseline_probs, payload.actual_index)
        if baseline_rank is not None:
            rerank_pos = actual_rank + 1
            baseline_pos = baseline_rank + 1
            rerank_top1 += int(rerank_pos == 1)
            rerank_mrr += 1.0 / rerank_pos
            baseline_mrr += 1.0 / baseline_pos
            mean_rank_improvement += baseline_pos - rerank_pos

    metrics: dict[str, Any] = {
        "race_count": float(race_count),
        "covered_races": float(covered),
        "candidate_coverage_rate": covered / race_count if race_count else 0.0,
        "top1_hit_rate": top_hits[1] / race_count if race_count else 0.0,
        "top3_hit_rate": top_hits[3] / race_count if race_count else 0.0,
        "top5_hit_rate": top_hits[5] / race_count if race_count else 0.0,
        "top10_hit_rate": top_hits[10] / race_count if race_count else 0.0,
        "top12_hit_rate": top_hits[12] / race_count if race_count else 0.0,
        "log_loss": float(np.mean(log_losses)) if log_losses else 0.0,
        "brier_score": float(np.mean(brier_scores)) if brier_scores else 0.0,
        "mean_actual_probability": float(np.mean(actual_probabilities)) if actual_probabilities else 0.0,
        "mean_top_probability": float(np.mean(top_probabilities)) if top_probabilities else 0.0,
    }
    metrics["rerank_metrics"] = {
        "coverage_races": float(covered),
        "rerank_top1_hit_rate": rerank_top1 / covered if covered else 0.0,
        "rerank_mrr": rerank_mrr / covered if covered else 0.0,
        "baseline_mrr": baseline_mrr / covered if covered else 0.0,
        "mean_rank_improvement": mean_rank_improvement / covered if covered else 0.0,
    }
    return metrics


def _fit_isotonic_from_raw(raw_values: np.ndarray, labels: np.ndarray) -> IsotonicRegression:
    raw = np.asarray(raw_values, dtype=float)
    y = np.asarray(labels, dtype=float)
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    if len(raw) == 0 or len(y) == 0:
        calibrator.fit(np.asarray([0.0, 1.0]), np.asarray([0.0, 1.0]))
        return calibrator
    calibrator.fit(raw, y)
    return calibrator


def _fast_calibrated_probabilities(
    raw_values: np.ndarray,
    race_ids: np.ndarray,
    calibrator: IsotonicRegression,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(raw_values) == 0:
        empty = np.asarray([], dtype=float)
        return empty, np.asarray([], dtype=int), np.asarray([0], dtype=int)
    race_codes, _ = pd.factorize(race_ids, sort=False)
    order = np.argsort(race_codes, kind="mergesort")
    sorted_codes = race_codes[order]
    sorted_raw = np.asarray(raw_values, dtype=float)[order]
    calibrated = np.asarray(calibrator.predict(sorted_raw), dtype=float)
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_codes)) + 1]
    ends = np.r_[starts[1:], len(sorted_codes)]
    sums = np.add.reduceat(calibrated, starts)
    lengths = ends - starts
    normalized = np.empty_like(calibrated, dtype=float)
    for start, end, total, length in zip(starts, ends, sums, lengths, strict=True):
        if total > 0:
            normalized[start:end] = calibrated[start:end] / float(total)
        else:
            normalized[start:end] = 1.0 / float(max(length, 1))
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return normalized[inverse], race_codes, starts


def _fast_calibrated_metrics_from_frame(
    trifecta_frame: pd.DataFrame,
    raw_col: str,
    calibrator: IsotonicRegression,
    baseline_raw_col: str | None = None,
) -> dict[str, Any]:
    if trifecta_frame.empty or raw_col not in trifecta_frame.columns or "is_actual" not in trifecta_frame.columns:
        return {}
    race_ids = trifecta_frame["race_id"].astype(str).to_numpy()
    race_codes, race_labels = pd.factorize(race_ids, sort=False)
    probabilities, _, _ = _fast_calibrated_probabilities(
        trifecta_frame[raw_col].to_numpy(dtype=float),
        race_ids,
        calibrator,
    )
    baseline_probabilities: np.ndarray | None = None
    if baseline_raw_col is not None and baseline_raw_col in trifecta_frame.columns:
        baseline_probabilities, _, _ = _fast_calibrated_probabilities(
            trifecta_frame[baseline_raw_col].to_numpy(dtype=float),
            race_ids,
            calibrator,
        )
    actual = trifecta_frame["is_actual"].astype(bool).to_numpy()
    top_hits = {1: 0, 3: 0, 5: 0, 10: 0, 12: 0}
    covered = 0
    rerank_top1 = 0
    rerank_mrr = 0.0
    baseline_mrr = 0.0
    mean_rank_improvement = 0.0
    log_losses: list[float] = []
    brier_scores: list[float] = []
    actual_probabilities: list[float] = []
    top_probabilities: list[float] = []

    for code in range(len(race_labels)):
        indices = np.flatnonzero(race_codes == code)
        actual_positions = np.flatnonzero(actual[indices])
        if len(actual_positions) != 1:
            continue
        covered += 1
        probs = probabilities[indices]
        actual_idx = int(actual_positions[0])
        actual_probability = max(float(probs[actual_idx]), 1e-15)
        order = np.argsort(-probs, kind="mergesort")
        ranked_positions = np.empty(len(order), dtype=int)
        ranked_positions[order] = np.arange(len(order), dtype=int)
        actual_rank = int(ranked_positions[actual_idx])
        labels = np.zeros(len(indices), dtype=float)
        labels[actual_idx] = 1.0
        for top_k in top_hits:
            top_hits[top_k] += int(actual_rank < top_k)
        log_losses.append(-np.log(actual_probability))
        brier_scores.append(float(np.mean((probs - labels) ** 2)))
        actual_probabilities.append(actual_probability)
        top_probabilities.append(float(probs[order[0]]))

        if baseline_probabilities is not None:
            baseline_probs = baseline_probabilities[indices]
            baseline_order = np.argsort(-baseline_probs, kind="mergesort")
            baseline_positions = np.empty(len(baseline_order), dtype=int)
            baseline_positions[baseline_order] = np.arange(len(baseline_order), dtype=int)
            rerank_pos = actual_rank + 1
            baseline_pos = int(baseline_positions[actual_idx]) + 1
            rerank_top1 += int(rerank_pos == 1)
            rerank_mrr += 1.0 / rerank_pos
            baseline_mrr += 1.0 / baseline_pos
            mean_rank_improvement += baseline_pos - rerank_pos

    race_count = int(len(race_labels))
    metrics: dict[str, Any] = {
        "race_count": float(race_count),
        "covered_races": float(covered),
        "candidate_coverage_rate": covered / race_count if race_count else 0.0,
        "top1_hit_rate": top_hits[1] / race_count if race_count else 0.0,
        "top3_hit_rate": top_hits[3] / race_count if race_count else 0.0,
        "top5_hit_rate": top_hits[5] / race_count if race_count else 0.0,
        "top10_hit_rate": top_hits[10] / race_count if race_count else 0.0,
        "top12_hit_rate": top_hits[12] / race_count if race_count else 0.0,
        "log_loss": float(np.mean(log_losses)) if log_losses else 0.0,
        "brier_score": float(np.mean(brier_scores)) if brier_scores else 0.0,
        "mean_actual_probability": float(np.mean(actual_probabilities)) if actual_probabilities else 0.0,
        "mean_top_probability": float(np.mean(top_probabilities)) if top_probabilities else 0.0,
    }
    if baseline_probabilities is not None:
        metrics["rerank_metrics"] = {
            "coverage_races": float(covered),
            "rerank_top1_hit_rate": rerank_top1 / covered if covered else 0.0,
            "rerank_mrr": rerank_mrr / covered if covered else 0.0,
            "baseline_mrr": baseline_mrr / covered if covered else 0.0,
            "mean_rank_improvement": mean_rank_improvement / covered if covered else 0.0,
        }
    return metrics


def _fast_matrix_metrics(prob_matrix: np.ndarray, actual_indices: np.ndarray) -> dict[str, Any]:
    probs = np.asarray(prob_matrix, dtype=float)
    actual = np.asarray(actual_indices, dtype=int)
    race_count = int(len(actual))
    if race_count == 0 or probs.size == 0:
        return {}
    valid_mask = (actual >= 0) & (actual < probs.shape[1])
    covered = int(valid_mask.sum())
    top_hits = {1: 0, 3: 0, 5: 0, 10: 0, 12: 0}
    boat_top1_hit_rate = 0.0
    if covered:
        valid_probs = probs[valid_mask]
        valid_actual = actual[valid_mask]
        order = np.argsort(-valid_probs, axis=1, kind="mergesort")
        positions = np.empty_like(order)
        positions[np.arange(len(order))[:, None], order] = np.arange(order.shape[1], dtype=int)
        actual_ranks = positions[np.arange(len(valid_actual)), valid_actual]
        actual_probabilities = np.clip(valid_probs[np.arange(len(valid_actual)), valid_actual], 1e-15, 1.0)
        for top_k in top_hits:
            top_hits[top_k] = int(np.sum(actual_ranks < top_k))
        predicted_winners = TRIFECTA_FAST_PERMUTATIONS[order[:, 0], 0]
        actual_winners = TRIFECTA_FAST_PERMUTATIONS[valid_actual, 0]
        boat_top1_hit_rate = float(np.mean(predicted_winners == actual_winners))
        brier_scores = (
            np.sum(valid_probs * valid_probs, axis=1)
            - (2.0 * valid_probs[np.arange(len(valid_actual)), valid_actual])
            + 1.0
        ) / float(valid_probs.shape[1])
        log_loss_value = float(np.mean(-np.log(actual_probabilities)))
        brier_value = float(np.mean(brier_scores))
        mean_actual = float(np.mean(actual_probabilities))
        mean_top = float(np.mean(valid_probs[np.arange(len(order)), order[:, 0]]))
    else:
        log_loss_value = 0.0
        brier_value = 0.0
        mean_actual = 0.0
        mean_top = 0.0

    return {
        "race_count": float(race_count),
        "covered_races": float(covered),
        "candidate_coverage_rate": covered / race_count if race_count else 0.0,
        "boat_top1_hit_rate": boat_top1_hit_rate,
        "top1_hit_rate": top_hits[1] / race_count if race_count else 0.0,
        "top3_hit_rate": top_hits[3] / race_count if race_count else 0.0,
        "top5_hit_rate": top_hits[5] / race_count if race_count else 0.0,
        "top10_hit_rate": top_hits[10] / race_count if race_count else 0.0,
        "top12_hit_rate": top_hits[12] / race_count if race_count else 0.0,
        "log_loss": log_loss_value,
        "brier_score": brier_value,
        "mean_actual_probability": mean_actual,
        "mean_top_probability": mean_top,
    }


def _evaluate_fast_v1_trifecta_metrics_from_ranked(
    ranked: pd.DataFrame,
    calibrator: IsotonicRegression | None,
    weights: dict[str, Any],
) -> dict[str, Any] | None:
    if ranked.empty or not {"race_id", "lane", "finish_position", "win_probability_like"}.issubset(ranked.columns):
        return None

    lane_prob_rows: list[np.ndarray] = []
    actual_indices: list[int] = []
    scenario_labels: list[str] = []
    entry_course_subset_labels: list[str] = []
    race_ids: list[str] = []
    trifecta_payouts: list[float] = []
    for race_id, race_df in ranked.groupby("race_id", sort=False):
        if len(race_df) != 6:
            continue
        race_df = race_df.sort_values("lane").reset_index(drop=True)
        lanes = pd.to_numeric(race_df["lane"], errors="coerce")
        finishes = pd.to_numeric(race_df["finish_position"], errors="coerce")
        lane_probs = pd.to_numeric(race_df["win_probability_like"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if lanes.isna().any() or finishes.isna().any() or len(set(lanes.astype(int).tolist())) != 6:
            continue
        prob_sum = float(lane_probs.sum())
        if prob_sum <= 0:
            continue
        lane_probs = lane_probs / prob_sum
        lane_to_position = {int(lane): position for position, lane in enumerate(lanes.astype(int).tolist())}
        ordered_actual = race_df.assign(_finish=finishes).sort_values("_finish").head(3)
        actual_tuple = tuple(lane_to_position.get(int(lane), -1) for lane in ordered_actual["lane"].tolist())
        actual_index = TRIFECTA_FAST_PERMUTATION_INDEX.get(actual_tuple, -1)
        lane_prob_rows.append(lane_probs)
        actual_indices.append(int(actual_index))
        scenario_labels.append(_phase3_scenario_label(_phase3_scenario_context(race_df.set_index("lane"))))
        entry_course_subset_labels.append(_entry_course_subset_label(race_df))
        race_ids.append(str(race_id))
        if "trifecta_payout" in race_df.columns:
            payout_values = pd.to_numeric(race_df["trifecta_payout"], errors="coerce").dropna()
            trifecta_payouts.append(float(payout_values.iloc[0]) if not payout_values.empty else float("nan"))
        else:
            trifecta_payouts.append(float("nan"))
    if not lane_prob_rows:
        return None

    lane_probs_matrix = np.vstack(lane_prob_rows)
    first = TRIFECTA_FAST_PERMUTATIONS[:, 0]
    second = TRIFECTA_FAST_PERMUTATIONS[:, 1]
    third = TRIFECTA_FAST_PERMUTATIONS[:, 2]
    p1 = lane_probs_matrix[:, first]
    p2_base = lane_probs_matrix[:, second]
    p3_base = lane_probs_matrix[:, third]
    denom2 = np.clip(1.0 - p1, 1e-12, None)
    denom3 = np.clip(1.0 - p1 - p2_base, 1e-12, None)
    raw_matrix = p1 * (p2_base / denom2) * (p3_base / denom3)
    if calibrator is not None:
        calibrated = np.asarray(calibrator.predict(raw_matrix.reshape(-1)), dtype=float).reshape(raw_matrix.shape)
        row_sums = calibrated.sum(axis=1, keepdims=True)
        prob_matrix = np.divide(
            calibrated,
            row_sums,
            out=np.full_like(calibrated, 1.0 / calibrated.shape[1]),
            where=row_sums > 0,
        )
    else:
        row_sums = raw_matrix.sum(axis=1, keepdims=True)
        prob_matrix = np.divide(
            raw_matrix,
            row_sums,
            out=np.full_like(raw_matrix, 1.0 / raw_matrix.shape[1]),
            where=row_sums > 0,
        )

    actual_array = np.asarray(actual_indices, dtype=int)
    scenario_array = np.asarray(scenario_labels, dtype=object)
    metrics = _fast_matrix_metrics(prob_matrix, actual_array)
    if not metrics:
        return None
    metrics["entry_course_subset_metrics"] = _fast_entry_course_subset_metrics(
        prob_matrix,
        actual_array,
        entry_course_subset_labels,
    )
    top12_confidence_metrics = _fast_top12_confidence_metrics(prob_matrix, actual_array)
    if top12_confidence_metrics:
        metrics["top12_confidence_metrics"] = top12_confidence_metrics
    top3_confidence_metrics = _fast_top3_confidence_metrics(prob_matrix, actual_array)
    if top3_confidence_metrics:
        metrics["top3_confidence_metrics"] = top3_confidence_metrics
    boat_top1_confidence_metrics = _fast_boat_top1_confidence_metrics(
        prob_matrix,
        actual_array,
        np.asarray(trifecta_payouts, dtype=float),
    )
    if boat_top1_confidence_metrics:
        metrics["boat_top1_confidence_metrics"] = boat_top1_confidence_metrics
    top3_x_boat_top1_confidence_metrics = _fast_top3_x_boat_top1_confidence_metrics(
        prob_matrix,
        actual_array,
        np.asarray(trifecta_payouts, dtype=float),
    )
    if top3_x_boat_top1_confidence_metrics:
        metrics["top3_x_boat_top1_confidence_metrics"] = top3_x_boat_top1_confidence_metrics
    payout_band_metrics = _fast_payout_band_metrics(
        prob_matrix,
        actual_array,
        np.asarray(trifecta_payouts, dtype=float),
    )
    if payout_band_metrics:
        metrics["payout_band_metrics"] = payout_band_metrics
    uniform_recovery_metrics = _fast_uniform_ticket_recovery_metrics(
        prob_matrix,
        actual_array,
        np.asarray(trifecta_payouts, dtype=float),
    )
    if uniform_recovery_metrics:
        metrics["uniform_ticket_recovery_metrics"] = uniform_recovery_metrics
    confidence_recovery_metrics = _fast_top12_confidence_recovery_metrics(
        prob_matrix,
        actual_array,
        np.asarray(trifecta_payouts, dtype=float),
    )
    if confidence_recovery_metrics:
        metrics["top12_confidence_recovery_metrics"] = confidence_recovery_metrics
    confidence_strategy_recovery_metrics = _fast_top12_confidence_strategy_recovery_metrics(
        prob_matrix,
        actual_array,
        np.asarray(trifecta_payouts, dtype=float),
    )
    if confidence_strategy_recovery_metrics:
        metrics["top12_confidence_strategy_recovery_metrics"] = confidence_strategy_recovery_metrics
    variable_ticket_recovery_metrics = _fast_variable_ticket_recovery_metrics(
        prob_matrix,
        actual_array,
        np.asarray(trifecta_payouts, dtype=float),
        value_rule=dict(weights.get("value_rule", {}) or {}),
    )
    if variable_ticket_recovery_metrics:
        metrics["variable_ticket_recovery_metrics"] = variable_ticket_recovery_metrics

    scenario_min_races = int(weights.get("scenario_metric_min_races", DEFAULT_PHASE3_SETTINGS["evaluation"]["scenario_min_races"]))
    scenario_metrics: dict[str, Any] = {}
    scenario_groups: dict[str, np.ndarray] = {}
    for scenario_label in sorted(set(scenario_labels)):
        mask = scenario_array == scenario_label
        scenario_result = _fast_matrix_metrics(prob_matrix[mask], actual_array[mask])
        scenario_result["scenario_id"] = float(scenario_numeric_id(scenario_label))
        scenario_result["scenario_min_races"] = float(scenario_min_races)
        scenario_result["is_small_sample"] = float(int(mask.sum()) < scenario_min_races)
        scenario_metrics[scenario_label] = scenario_result
        grouped_label = scenario_label if int(mask.sum()) >= scenario_min_races else "__small_sample__"
        scenario_groups[grouped_label] = mask if grouped_label not in scenario_groups else (scenario_groups[grouped_label] | mask)
    metrics["scenario_metrics"] = scenario_metrics

    grouped_metrics: dict[str, Any] = {}
    for scenario_label, mask in sorted(scenario_groups.items()):
        scenario_result = _fast_matrix_metrics(prob_matrix[mask], actual_array[mask])
        scenario_result["scenario_min_races"] = float(scenario_min_races)
        scenario_result["is_small_sample_group"] = float(scenario_label == "__small_sample__")
        grouped_metrics[scenario_label] = scenario_result
    metrics["scenario_metrics_grouped"] = grouped_metrics
    metrics["evaluation_mode"] = "fast_numpy_v1"
    metrics["fast_eval_races"] = float(len(race_ids))
    return metrics


def _entry_course_subset_label(race_df: pd.DataFrame) -> str:
    if not {"lane", "course"}.issubset(race_df.columns):
        return "course_unknown"
    lanes = pd.to_numeric(race_df["lane"], errors="coerce")
    courses = pd.to_numeric(race_df["course"], errors="coerce")
    if lanes.isna().any() or courses.isna().any() or len(race_df) != 6:
        return "course_unknown"
    lane_values = lanes.astype(int).to_numpy()
    course_values = courses.astype(int).to_numpy()
    if len(set(lane_values.tolist())) != 6 or len(set(course_values.tolist())) != 6:
        return "course_unknown"
    return "lane_course_match" if bool(np.all(lane_values == course_values)) else "lane_course_mismatch"


def _entry_course_subset_by_race(df: pd.DataFrame) -> dict[str, str]:
    if df.empty or "race_id" not in df.columns:
        return {}
    return {
        str(race_id): _entry_course_subset_label(race_df)
        for race_id, race_df in df.groupby("race_id", sort=False)
    }


def _fast_entry_course_subset_metrics(
    prob_matrix: np.ndarray,
    actual_indices: np.ndarray,
    subset_labels: list[str],
) -> dict[str, Any]:
    if len(subset_labels) != len(actual_indices):
        return {}
    subset_array = np.asarray(subset_labels, dtype=object)
    metrics: dict[str, Any] = {}
    for subset_label in ("lane_course_match", "lane_course_mismatch", "course_unknown"):
        mask = subset_array == subset_label
        if not mask.any():
            continue
        metrics[subset_label] = _fast_matrix_metrics(prob_matrix[mask], actual_indices[mask])
    return metrics


def _fast_top12_confidence_metrics(
    prob_matrix: np.ndarray,
    actual_indices: np.ndarray,
) -> dict[str, Any]:
    if prob_matrix.size == 0 or len(actual_indices) == 0:
        return {}
    valid_mask = (
        (actual_indices >= 0)
        & (actual_indices < prob_matrix.shape[1])
    )
    if not bool(valid_mask.any()):
        return {}

    valid_probs = prob_matrix[valid_mask]
    valid_actual = actual_indices[valid_mask].astype(int)
    order = np.argsort(-valid_probs, axis=1, kind="mergesort")
    positions = np.empty_like(order)
    positions[np.arange(len(order))[:, None], order] = np.arange(order.shape[1], dtype=int)
    actual_ranks = positions[np.arange(len(valid_actual)), valid_actual]
    sorted_probs = np.take_along_axis(valid_probs, order, axis=1)
    top12_mass = sorted_probs[:, :12].sum(axis=1)
    top5_mass = sorted_probs[:, :5].sum(axis=1)
    top1_top2_gap = sorted_probs[:, 0] - sorted_probs[:, 1]
    top12_margin = sorted_probs[:, 11] - sorted_probs[:, 12]
    clipped = np.clip(sorted_probs, 1e-12, 1.0)
    entropy = -np.sum(clipped * np.log(clipped), axis=1)
    max_entropy = float(np.log(sorted_probs.shape[1])) if sorted_probs.shape[1] > 1 else 1.0
    concentration = 1.0 - np.clip(entropy / max_entropy, 0.0, 1.0)
    score = (
        0.45 * np.clip((top12_mass - 0.10) / 0.45, 0.0, 1.0)
        + 0.20 * np.clip((top5_mass - 0.04) / 0.25, 0.0, 1.0)
        + 0.15 * np.clip(top1_top2_gap / 0.06, 0.0, 1.0)
        + 0.10 * np.clip(top12_margin / 0.01, 0.0, 1.0)
        + 0.10 * np.clip(concentration / 0.25, 0.0, 1.0)
    ) * 100.0
    labels = np.where(score >= 75.0, "high", np.where(score >= 60.0, "middle", "low"))
    top12_hits = (actual_ranks < 12).astype(float)

    metrics: dict[str, Any] = {}
    for label in ("high", "middle", "low"):
        mask = labels == label
        if not bool(mask.any()):
            continue
        metrics[label] = {
            "race_count": float(mask.sum()),
            "race_rate": float(mask.sum() / len(valid_actual)) if len(valid_actual) else 0.0,
            "top12_hit_rate": float(np.mean(top12_hits[mask])),
            "mean_score": float(np.mean(score[mask])),
            "mean_top12_probability_mass": float(np.mean(top12_mass[mask])),
            "mean_top5_probability_mass": float(np.mean(top5_mass[mask])),
        }
    return metrics


def _fast_top3_confidence_metrics(
    prob_matrix: np.ndarray,
    actual_indices: np.ndarray,
) -> dict[str, Any]:
    if prob_matrix.size == 0 or len(actual_indices) == 0:
        return {}
    valid_mask = (
        (actual_indices >= 0)
        & (actual_indices < prob_matrix.shape[1])
    )
    if not bool(valid_mask.any()):
        return {}

    valid_probs = prob_matrix[valid_mask]
    valid_actual = actual_indices[valid_mask].astype(int)
    order = np.argsort(-valid_probs, axis=1, kind="mergesort")
    positions = np.empty_like(order)
    positions[np.arange(len(order))[:, None], order] = np.arange(order.shape[1], dtype=int)
    actual_ranks = positions[np.arange(len(valid_actual)), valid_actual]
    sorted_probs = np.take_along_axis(valid_probs, order, axis=1)
    top3_mass = sorted_probs[:, :3].sum(axis=1)
    top1_probability = sorted_probs[:, 0]
    top3_margin = sorted_probs[:, 2] - sorted_probs[:, 3] if sorted_probs.shape[1] > 3 else np.zeros(len(sorted_probs))
    scores = _top3_confidence_scores_from_sorted_probs(sorted_probs)
    labels = np.where(scores >= 75.0, "high", np.where(scores >= 60.0, "middle", "low"))
    top3_hits = (actual_ranks < 3).astype(float)
    predicted_winners = TRIFECTA_FAST_PERMUTATIONS[order[:, 0], 0]
    actual_winners = TRIFECTA_FAST_PERMUTATIONS[valid_actual, 0]
    boat_top1_hits = (predicted_winners == actual_winners).astype(float)

    metrics: dict[str, Any] = {}
    for label in ("high", "middle", "low"):
        mask = labels == label
        if not bool(mask.any()):
            continue
        metrics[label] = {
            "race_count": float(mask.sum()),
            "race_rate": float(mask.sum() / len(valid_actual)) if len(valid_actual) else 0.0,
            "boat_top1_hit_rate": float(np.mean(boat_top1_hits[mask])),
            "top3_hit_rate": float(np.mean(top3_hits[mask])),
            "mean_score": float(np.mean(scores[mask])),
            "mean_top3_probability_mass": float(np.mean(top3_mass[mask])),
            "mean_top1_probability": float(np.mean(top1_probability[mask])),
            "mean_top3_probability_margin": float(np.mean(top3_margin[mask])),
        }
    return metrics


def _fast_boat_top1_confidence_metrics(
    prob_matrix: np.ndarray,
    actual_indices: np.ndarray,
    trifecta_payouts: np.ndarray,
    stake_per_ticket: float = 100.0,
) -> dict[str, Any]:
    if prob_matrix.size == 0 or len(actual_indices) == 0:
        return {}
    valid_mask = (
        (actual_indices >= 0)
        & (actual_indices < prob_matrix.shape[1])
    )
    if not bool(valid_mask.any()):
        return {}

    valid_probs = prob_matrix[valid_mask]
    valid_actual = actual_indices[valid_mask].astype(int)
    valid_payouts = (
        trifecta_payouts[valid_mask].astype(float)
        if len(trifecta_payouts) == len(actual_indices)
        else np.full(len(valid_actual), np.nan, dtype=float)
    )
    order = np.argsort(-valid_probs, axis=1, kind="mergesort")
    positions = np.empty_like(order)
    positions[np.arange(len(order))[:, None], order] = np.arange(order.shape[1], dtype=int)
    actual_ranks = positions[np.arange(len(valid_actual)), valid_actual]
    sorted_probs = np.take_along_axis(valid_probs, order, axis=1)

    first_boats = TRIFECTA_FAST_PERMUTATIONS[: valid_probs.shape[1], 0].astype(int)
    boat_prob_matrix = np.zeros((len(valid_probs), 6), dtype=float)
    for boat in range(1, 7):
        boat_prob_matrix[:, boat - 1] = valid_probs[:, first_boats == boat].sum(axis=1)
    predicted_boats = np.argmax(boat_prob_matrix, axis=1) + 1
    sorted_boat_probs = np.sort(boat_prob_matrix, axis=1)[:, ::-1]
    predicted_probabilities = sorted_boat_probs[:, 0]
    gaps = sorted_boat_probs[:, 0] - sorted_boat_probs[:, 1]
    top3_first_boats = first_boats[order[:, : min(3, order.shape[1])]]
    top3_same_first_boat_rate = np.mean(top3_first_boats == predicted_boats[:, None], axis=1)
    scores = _boat_top1_confidence_scores_from_boat_probs(
        boat_prob_matrix,
        top3_same_first_boat_rate=top3_same_first_boat_rate,
    )
    labels = np.where(scores >= 75.0, "high", np.where(scores >= 60.0, "middle", "low"))
    actual_winners = TRIFECTA_FAST_PERMUTATIONS[valid_actual, 0].astype(int)
    boat_top1_hits = predicted_boats == actual_winners
    top3_hits = actual_ranks < 3
    top12_hits = actual_ranks < 12
    valid_payout_mask = np.isfinite(valid_payouts) & (valid_payouts > 0.0)
    top3_ticket_count = min(3, valid_probs.shape[1])
    top3_stake_per_race = float(top3_ticket_count) * float(stake_per_ticket)

    metrics: dict[str, Any] = {}
    for label in ("high", "middle", "low"):
        mask = labels == label
        if not bool(mask.any()):
            continue
        payout_mask = mask & valid_payout_mask
        total_stake = float(payout_mask.sum()) * top3_stake_per_race
        total_return = float(np.sum(np.where(top3_hits[payout_mask], valid_payouts[payout_mask], 0.0)))
        hit_payouts = valid_payouts[payout_mask & top3_hits]
        metrics[label] = {
            "race_count": float(mask.sum()),
            "race_rate": float(mask.sum() / len(valid_actual)) if len(valid_actual) else 0.0,
            "boat_top1_hit_rate": float(np.mean(boat_top1_hits[mask])),
            "top3_hit_rate": float(np.mean(top3_hits[mask])),
            "top12_hit_rate": float(np.mean(top12_hits[mask])),
            "top3_total_stake": total_stake,
            "top3_total_return": total_return,
            "top3_recovery_rate": total_return / total_stake if total_stake else 0.0,
            "mean_score": float(np.mean(scores[mask])),
            "mean_predicted_first_boat_probability": float(np.mean(predicted_probabilities[mask])),
            "mean_predicted_first_boat_gap": float(np.mean(gaps[mask])),
            "mean_payout_hit": float(np.mean(hit_payouts)) if len(hit_payouts) else 0.0,
        }
    return metrics


def _fast_top3_x_boat_top1_confidence_metrics(
    prob_matrix: np.ndarray,
    actual_indices: np.ndarray,
    trifecta_payouts: np.ndarray,
    stake_per_ticket: float = 100.0,
) -> dict[str, Any]:
    if prob_matrix.size == 0 or len(actual_indices) == 0:
        return {}
    valid_mask = (
        (actual_indices >= 0)
        & (actual_indices < prob_matrix.shape[1])
    )
    if not bool(valid_mask.any()):
        return {}

    valid_probs = prob_matrix[valid_mask]
    valid_actual = actual_indices[valid_mask].astype(int)
    valid_payouts = (
        trifecta_payouts[valid_mask].astype(float)
        if len(trifecta_payouts) == len(actual_indices)
        else np.full(len(valid_actual), np.nan, dtype=float)
    )
    order = np.argsort(-valid_probs, axis=1, kind="mergesort")
    positions = np.empty_like(order)
    positions[np.arange(len(order))[:, None], order] = np.arange(order.shape[1], dtype=int)
    actual_ranks = positions[np.arange(len(valid_actual)), valid_actual]
    sorted_probs = np.take_along_axis(valid_probs, order, axis=1)
    top3_scores = _top3_confidence_scores_from_sorted_probs(sorted_probs)
    top3_labels = np.where(top3_scores >= 75.0, "high", np.where(top3_scores >= 60.0, "middle", "low"))

    first_boats = TRIFECTA_FAST_PERMUTATIONS[: valid_probs.shape[1], 0].astype(int)
    boat_prob_matrix = np.zeros((len(valid_probs), 6), dtype=float)
    for boat in range(1, 7):
        boat_prob_matrix[:, boat - 1] = valid_probs[:, first_boats == boat].sum(axis=1)
    predicted_boats = np.argmax(boat_prob_matrix, axis=1) + 1
    sorted_boat_probs = np.sort(boat_prob_matrix, axis=1)[:, ::-1]
    predicted_probabilities = sorted_boat_probs[:, 0]
    gaps = sorted_boat_probs[:, 0] - sorted_boat_probs[:, 1]
    top3_first_boats = first_boats[order[:, : min(3, order.shape[1])]]
    top3_same_first_boat_rate = np.mean(top3_first_boats == predicted_boats[:, None], axis=1)
    boat_scores = _boat_top1_confidence_scores_from_boat_probs(
        boat_prob_matrix,
        top3_same_first_boat_rate=top3_same_first_boat_rate,
    )
    boat_labels = np.where(boat_scores >= 75.0, "high", np.where(boat_scores >= 60.0, "middle", "low"))

    actual_winners = TRIFECTA_FAST_PERMUTATIONS[valid_actual, 0].astype(int)
    boat_top1_hits = predicted_boats == actual_winners
    top3_hits = actual_ranks < 3
    top12_hits = actual_ranks < 12
    valid_payout_mask = np.isfinite(valid_payouts) & (valid_payouts > 0.0)
    top3_ticket_count = min(3, valid_probs.shape[1])
    top3_stake_per_race = float(top3_ticket_count) * float(stake_per_ticket)

    result: dict[str, Any] = {}
    for top3_label in ("high", "middle", "low"):
        row: dict[str, Any] = {}
        top3_mask = top3_labels == top3_label
        if not bool(top3_mask.any()):
            continue
        for boat_label in ("high", "middle", "low"):
            mask = top3_mask & (boat_labels == boat_label)
            if not bool(mask.any()):
                continue
            payout_mask = mask & valid_payout_mask
            total_stake = float(payout_mask.sum()) * top3_stake_per_race
            total_return = float(np.sum(np.where(top3_hits[payout_mask], valid_payouts[payout_mask], 0.0)))
            hit_payouts = valid_payouts[payout_mask & top3_hits]
            row[boat_label] = {
                "race_count": float(mask.sum()),
                "race_rate": float(mask.sum() / len(valid_actual)) if len(valid_actual) else 0.0,
                "boat_top1_hit_rate": float(np.mean(boat_top1_hits[mask])),
                "top3_hit_rate": float(np.mean(top3_hits[mask])),
                "top12_hit_rate": float(np.mean(top12_hits[mask])),
                "top3_total_stake": total_stake,
                "top3_total_return": total_return,
                "top3_recovery_rate": total_return / total_stake if total_stake else 0.0,
                "mean_top3_confidence_score": float(np.mean(top3_scores[mask])),
                "mean_boat_top1_confidence_score": float(np.mean(boat_scores[mask])),
                "mean_predicted_first_boat_probability": float(np.mean(predicted_probabilities[mask])),
                "mean_predicted_first_boat_gap": float(np.mean(gaps[mask])),
                "mean_payout_hit": float(np.mean(hit_payouts)) if len(hit_payouts) else 0.0,
            }
        if row:
            result[top3_label] = row
    return result


def _fast_payout_band_metrics(
    prob_matrix: np.ndarray,
    actual_indices: np.ndarray,
    trifecta_payouts: np.ndarray,
) -> dict[str, Any]:
    if prob_matrix.size == 0 or len(actual_indices) == 0 or len(trifecta_payouts) != len(actual_indices):
        return {}
    valid_mask = (
        (actual_indices >= 0)
        & (actual_indices < prob_matrix.shape[1])
        & np.isfinite(trifecta_payouts)
        & (trifecta_payouts > 0.0)
    )
    if not bool(valid_mask.any()):
        return {}

    valid_probs = prob_matrix[valid_mask]
    valid_actual = actual_indices[valid_mask].astype(int)
    valid_payouts = trifecta_payouts[valid_mask].astype(float)
    order = np.argsort(-valid_probs, axis=1, kind="mergesort")
    positions = np.empty_like(order)
    positions[np.arange(len(order))[:, None], order] = np.arange(order.shape[1], dtype=int)
    actual_ranks = positions[np.arange(len(valid_actual)), valid_actual]
    actual_probabilities = np.clip(valid_probs[np.arange(len(valid_actual)), valid_actual], 1e-15, 1.0)

    band_keys = np.asarray([_payout_band_key_for_metrics(float(payout)) for payout in valid_payouts], dtype=object)
    metrics: dict[str, Any] = {}
    for band_key in ("lt_10000", "gte_10000_lt_50000", "gte_50000_lt_100000", "gte_100000"):
        mask = band_keys == band_key
        if not bool(mask.any()):
            continue
        metrics[band_key] = {
            "label": _payout_band_label_for_metrics(band_key),
            "race_count": float(mask.sum()),
            "race_rate": float(mask.sum() / len(valid_actual)) if len(valid_actual) else 0.0,
            "top1_hit_rate": float(np.mean(actual_ranks[mask] < 1)),
            "top3_hit_rate": float(np.mean(actual_ranks[mask] < 3)),
            "top5_hit_rate": float(np.mean(actual_ranks[mask] < 5)),
            "top10_hit_rate": float(np.mean(actual_ranks[mask] < 10)),
            "top12_hit_rate": float(np.mean(actual_ranks[mask] < 12)),
            "log_loss": float(np.mean(-np.log(actual_probabilities[mask]))),
            "mean_payout": float(np.mean(valid_payouts[mask])),
        }
    return metrics


def _fast_uniform_ticket_recovery_metrics(
    prob_matrix: np.ndarray,
    actual_indices: np.ndarray,
    trifecta_payouts: np.ndarray,
    top_ns: tuple[int, ...] = (1, 3, 5, 8, 12),
    bottom_ns: tuple[int, ...] = (8, 6),
    stake_per_ticket: float = 100.0,
) -> dict[str, Any]:
    if prob_matrix.size == 0 or len(actual_indices) == 0 or len(trifecta_payouts) != len(actual_indices):
        return {}
    valid_mask = (
        (actual_indices >= 0)
        & (actual_indices < prob_matrix.shape[1])
        & np.isfinite(trifecta_payouts)
        & (trifecta_payouts > 0.0)
    )
    if not bool(valid_mask.any()):
        return {}

    valid_probs = prob_matrix[valid_mask]
    valid_actual = actual_indices[valid_mask].astype(int)
    valid_payouts = trifecta_payouts[valid_mask].astype(float)
    order = np.argsort(-valid_probs, axis=1, kind="mergesort")
    positions = np.empty_like(order)
    positions[np.arange(len(order))[:, None], order] = np.arange(order.shape[1], dtype=int)
    actual_ranks = positions[np.arange(len(valid_actual)), valid_actual]

    metrics: dict[str, Any] = {}
    for top_n in top_ns:
        ticket_count = min(int(top_n), valid_probs.shape[1])
        hits = actual_ranks < ticket_count
        total_stake = float(len(valid_actual) * ticket_count * float(stake_per_ticket))
        total_return = float(np.sum(np.where(hits, valid_payouts, 0.0)))
        hit_payouts = valid_payouts[hits]
        metrics[f"top{top_n}"] = {
            "race_count": float(len(valid_actual)),
            "race_rate": 1.0,
            "ticket_count": float(ticket_count),
            "hit_rate": float(np.mean(hits)),
            "total_stake": total_stake,
            "total_return": total_return,
            "recovery_rate": total_return / total_stake if total_stake else 0.0,
            "mean_payout_all": float(np.mean(valid_payouts)),
            "mean_payout_hit": float(np.mean(hit_payouts)) if len(hit_payouts) else 0.0,
        }
    for bottom_n in bottom_ns:
        ticket_count = min(int(bottom_n), valid_probs.shape[1])
        candidate_pool = min(12, valid_probs.shape[1])
        bottom_start = max(candidate_pool - ticket_count, 0)
        hits = (actual_ranks >= bottom_start) & (actual_ranks < candidate_pool)
        total_stake = float(len(valid_actual) * ticket_count * float(stake_per_ticket))
        total_return = float(np.sum(np.where(hits, valid_payouts, 0.0)))
        hit_payouts = valid_payouts[hits]
        metrics[f"bottom{bottom_n}"] = {
            "race_count": float(len(valid_actual)),
            "race_rate": 1.0,
            "ticket_count": float(ticket_count),
            "hit_rate": float(np.mean(hits)),
            "total_stake": total_stake,
            "total_return": total_return,
            "recovery_rate": total_return / total_stake if total_stake else 0.0,
            "mean_payout_all": float(np.mean(valid_payouts)),
            "mean_payout_hit": float(np.mean(hit_payouts)) if len(hit_payouts) else 0.0,
        }
    return metrics


def _fast_top12_confidence_recovery_metrics(
    prob_matrix: np.ndarray,
    actual_indices: np.ndarray,
    trifecta_payouts: np.ndarray,
    stake_per_ticket: float = 100.0,
) -> dict[str, Any]:
    if prob_matrix.size == 0 or len(actual_indices) == 0 or len(trifecta_payouts) != len(actual_indices):
        return {}
    valid_mask = (
        (actual_indices >= 0)
        & (actual_indices < prob_matrix.shape[1])
        & np.isfinite(trifecta_payouts)
        & (trifecta_payouts > 0.0)
    )
    if not bool(valid_mask.any()):
        return {}

    valid_probs = prob_matrix[valid_mask]
    valid_actual = actual_indices[valid_mask].astype(int)
    valid_payouts = trifecta_payouts[valid_mask].astype(float)
    order = np.argsort(-valid_probs, axis=1, kind="mergesort")
    positions = np.empty_like(order)
    positions[np.arange(len(order))[:, None], order] = np.arange(order.shape[1], dtype=int)
    actual_ranks = positions[np.arange(len(valid_actual)), valid_actual]
    sorted_probs = np.take_along_axis(valid_probs, order, axis=1)
    scores = _top12_confidence_scores_from_sorted_probs(sorted_probs)
    labels = np.where(scores >= 75.0, "high", np.where(scores >= 60.0, "middle", "low"))
    top12_hits = actual_ranks < 12
    ticket_count = min(12, valid_probs.shape[1])
    stake_per_race = float(ticket_count) * float(stake_per_ticket)

    metrics: dict[str, Any] = {}
    for label in ("high", "middle", "low"):
        mask = labels == label
        if not bool(mask.any()):
            continue
        hit_payouts = valid_payouts[mask & top12_hits]
        total_stake = float(mask.sum()) * stake_per_race
        total_return = float(np.sum(np.where(top12_hits[mask], valid_payouts[mask], 0.0)))
        metrics[label] = {
            "race_count": float(mask.sum()),
            "race_rate": float(mask.sum() / len(valid_actual)) if len(valid_actual) else 0.0,
            "ticket_count": float(ticket_count),
            "top12_hit_rate": float(np.mean(top12_hits[mask])),
            "hit_rate": float(np.mean(top12_hits[mask])),
            "total_stake": total_stake,
            "total_return": total_return,
            "recovery_rate": total_return / total_stake if total_stake else 0.0,
            "mean_score": float(np.mean(scores[mask])),
            "mean_payout_all": float(np.mean(valid_payouts[mask])),
            "mean_payout_hit": float(np.mean(hit_payouts)) if len(hit_payouts) else 0.0,
        }
    return metrics


def _fast_variable_ticket_recovery_metrics(
    prob_matrix: np.ndarray,
    actual_indices: np.ndarray,
    trifecta_payouts: np.ndarray,
    value_rule: dict[str, Any] | None = None,
    stake_per_ticket: float = 100.0,
) -> dict[str, Any]:
    if prob_matrix.size == 0 or len(actual_indices) == 0 or len(trifecta_payouts) != len(actual_indices):
        return {}
    valid_mask = (
        (actual_indices >= 0)
        & (actual_indices < prob_matrix.shape[1])
        & np.isfinite(trifecta_payouts)
        & (trifecta_payouts > 0.0)
    )
    if not bool(valid_mask.any()):
        return {}

    valid_probs = prob_matrix[valid_mask]
    valid_actual = actual_indices[valid_mask].astype(int)
    valid_payouts = trifecta_payouts[valid_mask].astype(float)
    order = np.argsort(-valid_probs, axis=1, kind="mergesort")
    positions = np.empty_like(order)
    positions[np.arange(len(order))[:, None], order] = np.arange(order.shape[1], dtype=int)
    actual_ranks = positions[np.arange(len(valid_actual)), valid_actual]
    sorted_probs = np.take_along_axis(valid_probs, order, axis=1)
    scores = _top3_confidence_scores_from_sorted_probs(sorted_probs)
    labels = np.where(scores >= 75.0, "high", np.where(scores >= 60.0, "middle", "low"))
    rule = {
        "high": "top3",
        "middle": "top3",
        "low": "skip",
        **dict(value_rule or {}),
    }
    ticket_counts = np.zeros(len(valid_actual), dtype=int)
    decisions = np.full(len(valid_actual), "skip", dtype=object)
    for label in ("high", "middle", "low"):
        decision = str(rule.get(label, "skip")).strip().lower()
        mask = labels == label
        decisions[mask] = decision
        if decision.startswith("top"):
            try:
                ticket_count = int(decision.removeprefix("top"))
            except ValueError:
                ticket_count = 0
            ticket_counts[mask] = max(ticket_count, 0)
    ticket_counts = np.minimum(ticket_counts, valid_probs.shape[1])
    hits = (ticket_counts > 0) & (actual_ranks < ticket_counts)
    stakes = ticket_counts.astype(float) * float(stake_per_ticket)
    returns = np.where(hits, valid_payouts, 0.0)
    decision_order = tuple(dict.fromkeys(["skip", *[str(rule.get(label, "skip")).strip().lower() for label in ("high", "middle", "low")]]))

    return {
        "summary": _fast_summarize_variable_ticket_records(
            ticket_counts=ticket_counts,
            hits=hits,
            stakes=stakes,
            returns=returns,
            payouts=valid_payouts,
            scores=scores,
        ),
        "by_decision": {
            decision: _fast_summarize_variable_ticket_records(
                ticket_counts=ticket_counts[mask],
                hits=hits[mask],
                stakes=stakes[mask],
                returns=returns[mask],
                payouts=valid_payouts[mask],
                scores=scores[mask],
                total_races=len(valid_actual),
            )
            for decision in decision_order
            if bool((mask := decisions == decision).any())
        },
        "rule": rule,
        "confidence_type": "top3",
    }


def _fast_top12_confidence_strategy_recovery_metrics(
    prob_matrix: np.ndarray,
    actual_indices: np.ndarray,
    trifecta_payouts: np.ndarray,
    top_ns: tuple[int, ...] = (1, 3, 5, 8, 12),
    bottom_ns: tuple[int, ...] = (8, 6),
    stake_per_ticket: float = 100.0,
) -> dict[str, Any]:
    if prob_matrix.size == 0 or len(actual_indices) == 0 or len(trifecta_payouts) != len(actual_indices):
        return {}
    valid_mask = (
        (actual_indices >= 0)
        & (actual_indices < prob_matrix.shape[1])
        & np.isfinite(trifecta_payouts)
        & (trifecta_payouts > 0.0)
    )
    if not bool(valid_mask.any()):
        return {}

    valid_probs = prob_matrix[valid_mask]
    valid_actual = actual_indices[valid_mask].astype(int)
    valid_payouts = trifecta_payouts[valid_mask].astype(float)
    order = np.argsort(-valid_probs, axis=1, kind="mergesort")
    positions = np.empty_like(order)
    positions[np.arange(len(order))[:, None], order] = np.arange(order.shape[1], dtype=int)
    actual_ranks = positions[np.arange(len(valid_actual)), valid_actual]
    sorted_probs = np.take_along_axis(valid_probs, order, axis=1)
    top12_mass = sorted_probs[:, :12].sum(axis=1)
    top5_mass = sorted_probs[:, :5].sum(axis=1)
    top1_top2_gap = sorted_probs[:, 0] - sorted_probs[:, 1]
    top12_margin = sorted_probs[:, 11] - sorted_probs[:, 12]
    clipped = np.clip(sorted_probs, 1e-12, 1.0)
    entropy = -np.sum(clipped * np.log(clipped), axis=1)
    max_entropy = float(np.log(sorted_probs.shape[1])) if sorted_probs.shape[1] > 1 else 1.0
    concentration = 1.0 - np.clip(entropy / max_entropy, 0.0, 1.0)
    scores = (
        0.45 * np.clip((top12_mass - 0.10) / 0.45, 0.0, 1.0)
        + 0.20 * np.clip((top5_mass - 0.04) / 0.25, 0.0, 1.0)
        + 0.15 * np.clip(top1_top2_gap / 0.06, 0.0, 1.0)
        + 0.10 * np.clip(top12_margin / 0.01, 0.0, 1.0)
        + 0.10 * np.clip(concentration / 0.25, 0.0, 1.0)
    ) * 100.0
    labels = np.where(scores >= 75.0, "high", np.where(scores >= 60.0, "middle", "low"))

    metrics: dict[str, Any] = {}
    for label in ("high", "middle", "low"):
        label_mask = labels == label
        if not bool(label_mask.any()):
            continue
        label_metrics: dict[str, Any] = {}
        for top_n in top_ns:
            ticket_count = min(int(top_n), valid_probs.shape[1])
            hits = actual_ranks < ticket_count
            label_metrics[f"top{top_n}"] = _fast_summarize_ticket_recovery_arrays(
                hits=hits[label_mask],
                payouts=valid_payouts[label_mask],
                scores=scores[label_mask],
                ticket_count=ticket_count,
                total_races=len(valid_actual),
                stake_per_ticket=stake_per_ticket,
            )
        for bottom_n in bottom_ns:
            ticket_count = min(int(bottom_n), valid_probs.shape[1])
            candidate_pool = min(12, valid_probs.shape[1])
            bottom_start = max(candidate_pool - ticket_count, 0)
            hits = (actual_ranks >= bottom_start) & (actual_ranks < candidate_pool)
            label_metrics[f"bottom{bottom_n}"] = _fast_summarize_ticket_recovery_arrays(
                hits=hits[label_mask],
                payouts=valid_payouts[label_mask],
                scores=scores[label_mask],
                ticket_count=ticket_count,
                total_races=len(valid_actual),
                stake_per_ticket=stake_per_ticket,
            )
        metrics[label] = label_metrics
    return metrics


def _fast_summarize_ticket_recovery_arrays(
    hits: np.ndarray,
    payouts: np.ndarray,
    scores: np.ndarray,
    ticket_count: int,
    total_races: int,
    stake_per_ticket: float = 100.0,
) -> dict[str, float]:
    if len(hits) == 0:
        return {
            "race_count": 0.0,
            "race_rate": 0.0,
            "hit_rate": 0.0,
            "total_stake": 0.0,
            "total_return": 0.0,
            "recovery_rate": 0.0,
            "mean_payout_all": 0.0,
            "mean_payout_hit": 0.0,
            "mean_score": 0.0,
            "ticket_count": float(ticket_count),
        }
    hit_mask = hits.astype(bool)
    total_stake = float(len(hits) * int(ticket_count) * float(stake_per_ticket))
    total_return = float(np.sum(np.where(hit_mask, payouts, 0.0)))
    hit_payouts = payouts[hit_mask]
    return {
        "race_count": float(len(hits)),
        "race_rate": float(len(hits) / total_races) if total_races else 0.0,
        "hit_rate": float(np.mean(hit_mask)),
        "total_stake": total_stake,
        "total_return": total_return,
        "recovery_rate": total_return / total_stake if total_stake else 0.0,
        "mean_payout_all": float(np.mean(payouts)) if len(payouts) else 0.0,
        "mean_payout_hit": float(np.mean(hit_payouts)) if len(hit_payouts) else 0.0,
        "mean_score": float(np.mean(scores)) if len(scores) else 0.0,
        "ticket_count": float(ticket_count),
    }


def _fast_summarize_variable_ticket_records(
    ticket_counts: np.ndarray,
    hits: np.ndarray,
    stakes: np.ndarray,
    returns: np.ndarray,
    payouts: np.ndarray,
    scores: np.ndarray,
    total_races: int | None = None,
) -> dict[str, float]:
    if len(ticket_counts) == 0:
        return {
            "race_count": 0.0,
            "race_rate": 0.0,
            "purchased_race_count": 0.0,
            "purchase_rate": 0.0,
            "average_ticket_count": 0.0,
            "average_ticket_count_purchased": 0.0,
            "hit_rate": 0.0,
            "overall_hit_rate": 0.0,
            "total_stake": 0.0,
            "total_return": 0.0,
            "recovery_rate": 0.0,
            "mean_payout_all": 0.0,
            "mean_payout_hit": 0.0,
            "mean_score": 0.0,
        }
    race_count = int(len(ticket_counts))
    purchase_mask = ticket_counts > 0
    purchased_race_count = int(purchase_mask.sum())
    total_stake = float(np.sum(stakes))
    total_return = float(np.sum(returns))
    hit_payouts = payouts[hits]
    return {
        "race_count": float(race_count),
        "race_rate": float(race_count / total_races) if total_races else 1.0,
        "purchased_race_count": float(purchased_race_count),
        "purchase_rate": float(purchased_race_count / race_count) if race_count else 0.0,
        "average_ticket_count": float(np.mean(ticket_counts)) if race_count else 0.0,
        "average_ticket_count_purchased": float(np.mean(ticket_counts[purchase_mask])) if purchased_race_count else 0.0,
        "hit_rate": float(np.mean(hits[purchase_mask])) if purchased_race_count else 0.0,
        "overall_hit_rate": float(np.mean(hits)) if race_count else 0.0,
        "total_stake": total_stake,
        "total_return": total_return,
        "recovery_rate": total_return / total_stake if total_stake else 0.0,
        "mean_payout_all": float(np.mean(payouts)) if race_count else 0.0,
        "mean_payout_hit": float(np.mean(hit_payouts)) if len(hit_payouts) else 0.0,
        "mean_score": float(np.mean(scores)) if race_count else 0.0,
    }


def _payout_band_key_for_metrics(payout: float) -> str:
    if payout < 10000.0:
        return "lt_10000"
    if payout < 50000.0:
        return "gte_10000_lt_50000"
    if payout < 100000.0:
        return "gte_50000_lt_100000"
    return "gte_100000"


def _payout_band_label_for_metrics(key: str) -> str:
    labels = {
        "lt_10000": "under_10000",
        "gte_10000_lt_50000": "10000_to_49999",
        "gte_50000_lt_100000": "50000_to_99999",
        "gte_100000": "100000_or_more",
    }
    return labels.get(key, "unknown")


def _fast_v1_raw_and_labels_from_ranked(ranked: pd.DataFrame) -> tuple[np.ndarray, np.ndarray] | None:
    if ranked.empty or not {"race_id", "lane", "finish_position", "win_probability_like"}.issubset(ranked.columns):
        return None
    raw_rows: list[np.ndarray] = []
    actual_indices: list[int] = []
    for _, race_df in ranked.groupby("race_id", sort=False):
        if len(race_df) != 6:
            continue
        race_df = race_df.sort_values("lane").reset_index(drop=True)
        lanes = pd.to_numeric(race_df["lane"], errors="coerce")
        finishes = pd.to_numeric(race_df["finish_position"], errors="coerce")
        lane_probs = pd.to_numeric(race_df["win_probability_like"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if lanes.isna().any() or finishes.isna().any() or len(set(lanes.astype(int).tolist())) != 6:
            continue
        prob_sum = float(lane_probs.sum())
        if prob_sum <= 0:
            continue
        lane_probs = lane_probs / prob_sum
        lane_to_position = {int(lane): position for position, lane in enumerate(lanes.astype(int).tolist())}
        ordered_actual = race_df.assign(_finish=finishes).sort_values("_finish").head(3)
        actual_tuple = tuple(lane_to_position.get(int(lane), -1) for lane in ordered_actual["lane"].tolist())
        actual_index = TRIFECTA_FAST_PERMUTATION_INDEX.get(actual_tuple, -1)
        raw_rows.append(lane_probs)
        actual_indices.append(int(actual_index))
    if not raw_rows:
        return None
    lane_probs_matrix = np.vstack(raw_rows)
    first = TRIFECTA_FAST_PERMUTATIONS[:, 0]
    second = TRIFECTA_FAST_PERMUTATIONS[:, 1]
    third = TRIFECTA_FAST_PERMUTATIONS[:, 2]
    p1 = lane_probs_matrix[:, first]
    p2_base = lane_probs_matrix[:, second]
    p3_base = lane_probs_matrix[:, third]
    denom2 = np.clip(1.0 - p1, 1e-12, None)
    denom3 = np.clip(1.0 - p1 - p2_base, 1e-12, None)
    raw_matrix = p1 * (p2_base / denom2) * (p3_base / denom3)
    labels = np.zeros(raw_matrix.shape, dtype=float)
    actual = np.asarray(actual_indices, dtype=int)
    valid_mask = (actual >= 0) & (actual < raw_matrix.shape[1])
    if valid_mask.any():
        labels[np.flatnonzero(valid_mask), actual[valid_mask]] = 1.0
    return raw_matrix.reshape(-1), labels.reshape(-1)


def fit_trifecta_calibrator_fast_from_ranked(ranked: pd.DataFrame) -> IsotonicRegression | None:
    payload = _fast_v1_raw_and_labels_from_ranked(ranked)
    if payload is None:
        return None
    raw, labels = payload
    return _fit_isotonic_from_raw(raw, labels)


def build_fast_rerank_payloads_from_ranked(
    ranked: pd.DataFrame,
    weights: dict[str, float],
    trifecta_v2_model: Any,
    top_n: int,
) -> list[FastRerankRacePayload]:
    if ranked.empty or "lane" not in ranked.columns or "finish_position" not in ranked.columns:
        return []
    if not is_trifecta_v2_bundle(trifecta_v2_model) or "booster" not in trifecta_v2_model:
        return []

    payloads: list[FastRerankRacePayload] = []
    v2_v1_weight = float(weights.get("trifecta_v2_v1_weight", 0.9))
    scenario_top_n = get_scenario_candidate_top_n(trifecta_v2_model)
    rank_penalty_start = get_rank_penalty_start(trifecta_v2_model)

    for race_id, race_df in ranked.groupby("race_id", sort=False):
        actual_order = actual_trifecta_order(race_df)
        v1 = enumerate_trifecta_probabilities_from_scores(race_df).rename(
            columns={"raw_probability": "raw_probability_v1"}
        )
        if v1.empty:
            continue
        v2 = enumerate_trifecta_probabilities_v2(race_df).rename(columns={"raw_probability": "raw_probability_v2"})
        candidate_mask = select_rerank_candidate_mask(
            v1,
            race_df,
            top_n=int(top_n),
            scenario_top_n=scenario_top_n,
        )
        selected_v1 = v1.loc[candidate_mask].reset_index(drop=True)
        selected_v2 = v2.loc[candidate_mask].reset_index(drop=True)
        if selected_v1.empty:
            payloads.append(
                FastRerankRacePayload(
                    race_id=str(race_id),
                    raw_v1=np.asarray([], dtype=float),
                    base_rank_score=np.asarray([], dtype=float),
                    update_rank_score=np.asarray([], dtype=float),
                    rank_penalty_overflow=np.asarray([], dtype=float),
                    actual_index=-1,
                    flat_update=True,
                )
            )
            continue

        selected_v2["raw_probability_v2"] = blend_trifecta_raw_probabilities(
            selected_v1["raw_probability_v1"].to_numpy(dtype=float),
            selected_v2["raw_probability_v2"].to_numpy(dtype=float),
            v2_v1_weight,
        )
        features = build_trifecta_feature_frame(
            race_df,
            selected_v1,
            selected_v2,
            scenario_model_bundle=trifecta_v2_model,
        )
        rerank_scores = predict_trifecta_v2_scores(trifecta_v2_model, features)
        if is_trifecta_v2_bundle(trifecta_v2_model) and trifecta_v2_model.get("phase") == "phase3_conditional":
            rerank_scores = apply_phase3_conditional_scores(
                race_df=race_df,
                trifecta_df=selected_v2,
                base_scores=rerank_scores,
                model_bundle=trifecta_v2_model,
            )
        raw_v1 = selected_v1["raw_probability_v1"].to_numpy(dtype=float)
        base_rank_score, update_rank_score, overflow, flat_update = _fast_rank_components(
            raw_v1,
            rerank_scores,
            rank_penalty_start,
        )
        actual_index = -1
        if actual_order is not None:
            actual_label = "-".join(str(value) for value in actual_order)
            matches = np.flatnonzero(selected_v1["trifecta"].astype(str).to_numpy() == actual_label)
            if len(matches) == 1:
                actual_index = int(matches[0])
        payloads.append(
            FastRerankRacePayload(
                race_id=str(race_id),
                raw_v1=raw_v1,
                base_rank_score=base_rank_score,
                update_rank_score=update_rank_score,
                rank_penalty_overflow=overflow,
                actual_index=actual_index,
                flat_update=flat_update,
            )
        )
    return payloads


def build_fast_rerank_ranked_frame(
    models: dict[str, Any],
    weights: dict[str, float],
    valid_df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    classifier_models: dict[str, lgb.Booster] | None = None,
    flow_model: lgb.Booster | None = None,
    flow_classes: list[str] | None = None,
    staged_models: dict[str, lgb.Booster] | None = None,
) -> pd.DataFrame | None:
    if not models or valid_df.empty or not {"race_id", "lane", "finish_position"}.issubset(valid_df.columns):
        return None
    eval_df = apply_prediction_time_measurement_proxies(valid_df)
    return build_weighted_lane_probabilities(
        models,
        weights,
        eval_df,
        feature_columns,
        categorical_columns,
        classifier_models=classifier_models,
        flow_model=flow_model,
        flow_classes=flow_classes,
        staged_models=staged_models,
    )


def build_disabled_dynamic_rerank_weight_metadata(
    config: dict | None,
    default_weight: float,
) -> dict[str, Any]:
    settings = get_phase3_settings(config)["dynamic_rerank_weight"]
    return {
        "enabled": bool(settings.get("enabled", False)),
        "optimized": False,
        "default_weight": float(default_weight),
        "rules": [],
        "thresholds": {},
        "diagnostics": {},
    }


def build_dynamic_rerank_race_features(
    race_df: pd.DataFrame,
    model_bundle: Any | None = None,
) -> dict[str, float]:
    if race_df.empty:
        return {
            "escape_strength": 0.0,
            "inner_collapse_risk": 0.0,
            "attack_pressure": 0.0,
            "race_upset_score": 0.0,
            "probability_flatness": 0.0,
        }
    lane_frame = race_df.set_index("lane").copy()
    scenario = apply_phase3_pattern_model_to_scenario(_phase3_scenario_context(lane_frame), race_df, model_bundle)
    scenario_score_map = _phase3_scenario_scores(scenario)
    probability_source = (
        race_df["win_probability_like"]
        if "win_probability_like" in race_df.columns
        else pd.Series(0.0, index=race_df.index)
    )
    ordered_probs = (
        pd.to_numeric(probability_source, errors="coerce").fillna(0.0).sort_values(ascending=False).to_numpy(dtype=float)
    )
    top_prob = float(ordered_probs[0]) if len(ordered_probs) > 0 else 0.0
    second_prob = float(ordered_probs[1]) if len(ordered_probs) > 1 else 0.0
    top10_entropy = _normalized_entropy(ordered_probs[:10])
    probability_flatness = _clip01(
        0.55 * top10_entropy
        + 0.25 * (1.0 - top_prob)
        + 0.20 * (1.0 - max(top_prob - second_prob, 0.0))
    )
    outer_scenario_pressure = max(
        scenario_score_map.get("S3", 0.0),
        scenario_score_map.get("S4", 0.0),
        scenario_score_map.get("S5", 0.0),
        scenario_score_map.get("S6", 0.0),
        scenario_score_map.get("S7", 0.0),
    )
    race_upset_score = _clip01(
        0.28 * float(scenario.get("inner_collapse_risk", 0.0))
        + 0.24 * float(scenario.get("attack_pressure", 0.0))
        + 0.20 * float(scenario.get("outer_sweep_risk", 0.0))
        + 0.18 * outer_scenario_pressure
        + 0.10 * probability_flatness
    )
    return {
        "escape_strength": float(scenario.get("escape_strength", 0.0)),
        "inner_collapse_risk": float(scenario.get("inner_collapse_risk", 0.0)),
        "attack_pressure": float(scenario.get("attack_pressure", 0.0)),
        "race_upset_score": race_upset_score,
        "probability_flatness": probability_flatness,
    }


def build_dynamic_rerank_thresholds(
    feature_frame: pd.DataFrame,
    settings: dict[str, Any],
) -> dict[str, float]:
    if feature_frame.empty:
        return {}
    high_q = float(settings.get("quantile_high", 0.7))
    mid_q = float(settings.get("quantile_mid", 0.5))
    return {
        "escape_strength_high": float(feature_frame["escape_strength"].quantile(high_q)),
        "inner_collapse_risk_high": float(feature_frame["inner_collapse_risk"].quantile(high_q)),
        "inner_collapse_risk_mid": float(feature_frame["inner_collapse_risk"].quantile(mid_q)),
        "attack_pressure_high": float(feature_frame["attack_pressure"].quantile(high_q)),
        "race_upset_score_high": float(feature_frame["race_upset_score"].quantile(high_q)),
        "probability_flatness_high": float(feature_frame["probability_flatness"].quantile(high_q)),
    }


def classify_dynamic_rerank_subset(
    features: dict[str, float],
    thresholds: dict[str, float],
) -> str:
    if not thresholds:
        return "neutral"
    attack = float(features.get("attack_pressure", 0.0))
    collapse = float(features.get("inner_collapse_risk", 0.0))
    upset = float(features.get("race_upset_score", 0.0))
    flat = float(features.get("probability_flatness", 0.0))
    escape = float(features.get("escape_strength", 0.0))
    if attack >= float(thresholds.get("attack_pressure_high", 1.0)) or collapse >= float(
        thresholds.get("inner_collapse_risk_high", 1.0)
    ):
        return "attack_or_collapse"
    if upset >= float(thresholds.get("race_upset_score_high", 1.0)) or flat >= float(
        thresholds.get("probability_flatness_high", 1.0)
    ):
        return "upset_or_flat"
    if escape >= float(thresholds.get("escape_strength_high", 1.0)) and collapse <= float(
        thresholds.get("inner_collapse_risk_mid", 0.0)
    ):
        return "stable_escape"
    return "neutral"


def build_dynamic_rerank_subset_frame(
    ranked_df: pd.DataFrame,
    model_bundle: Any | None,
    settings: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for race_id, race_df in ranked_df.groupby("race_id", sort=False):
        features = build_dynamic_rerank_race_features(race_df, model_bundle)
        rows.append({"race_id": str(race_id), **features})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    thresholds = build_dynamic_rerank_thresholds(frame, settings)
    frame["dynamic_rerank_subset"] = [
        classify_dynamic_rerank_subset(row, thresholds) for row in frame.to_dict(orient="records")
    ]
    frame.attrs["dynamic_rerank_thresholds"] = thresholds
    return frame


def _dynamic_rerank_candidate_diagnostic(
    subset_name: str,
    candidate_weight: float,
    candidate_metrics: dict[str, Any],
    baseline_top12: float,
    baseline_log_loss: float,
    log_loss_max_delta: float,
    default_weight: float,
) -> dict[str, Any]:
    top12_hit_rate = float(candidate_metrics.get("top12_hit_rate", 0.0))
    log_loss = float(candidate_metrics.get("log_loss", 0.0))
    rerank_metrics = candidate_metrics.get("rerank_metrics", {})
    rerank_mrr = float(rerank_metrics.get("rerank_mrr", 0.0)) if isinstance(rerank_metrics, dict) else 0.0
    log_loss_delta = log_loss - baseline_log_loss
    return {
        "subset": str(subset_name),
        "weight": float(candidate_weight),
        "is_default_weight": bool(np.isclose(float(candidate_weight), float(default_weight))),
        "race_count": float(candidate_metrics.get("race_count", 0.0)),
        "covered_races": float(candidate_metrics.get("covered_races", 0.0)),
        "top1_hit_rate": float(candidate_metrics.get("top1_hit_rate", 0.0)),
        "top3_hit_rate": float(candidate_metrics.get("top3_hit_rate", 0.0)),
        "top5_hit_rate": float(candidate_metrics.get("top5_hit_rate", 0.0)),
        "top10_hit_rate": float(candidate_metrics.get("top10_hit_rate", 0.0)),
        "top12_hit_rate": top12_hit_rate,
        "top12_delta": top12_hit_rate - baseline_top12,
        "log_loss": log_loss,
        "log_loss_delta": log_loss_delta,
        "log_loss_within_guard": bool(log_loss <= baseline_log_loss + log_loss_max_delta),
        "brier_score": float(candidate_metrics.get("brier_score", 0.0)),
        "rerank_mrr": rerank_mrr,
    }


def get_dynamic_rerank_weight_for_race(
    model: Any,
    race_df: pd.DataFrame,
) -> tuple[float, str, bool]:
    default_weight = get_conservative_rerank_weight(model)
    metadata = get_dynamic_rerank_weight_metadata(model)
    if not bool(metadata.get("enabled", False)):
        return default_weight, "disabled", False
    thresholds = metadata.get("thresholds", {})
    rules = metadata.get("rules", [])
    if not isinstance(thresholds, dict) or not isinstance(rules, list):
        return default_weight, "neutral", False
    features = build_dynamic_rerank_race_features(race_df, model)
    subset = classify_dynamic_rerank_subset(features, thresholds)
    for rule in rules:
        if isinstance(rule, dict) and rule.get("subset") == subset:
            return float(rule.get("weight", default_weight)), subset, True
    return float(metadata.get("default_weight", default_weight)), subset, True


def train_ranker(
    training_table: pd.DataFrame,
    config: dict,
) -> tuple[
    dict[str, Any],
    list[str],
    dict[str, Any],
    IsotonicRegression,
    dict[str, lgb.Booster],
    lgb.Booster | None,
    list[str] | None,
    dict[str, lgb.Booster],
    Any | None,
]:
    config = with_latest_available_dates(config, infer_latest_available_race_date(training_table))
    training_table = prepare_training_table(training_table, config)

    if _is_random_by_race_split(config):
        train_df, valid_df, test_df = split_training_frame_random_by_race(training_table, config)
    else:
        train_end = pd.Timestamp(config["split"]["train_end_date"])
        valid_end = pd.Timestamp(config["split"]["valid_end_date"])

        train_df = training_table[training_table["race_date"] <= train_end].copy()
        valid_df = training_table[
            (training_table["race_date"] > train_end) & (training_table["race_date"] <= valid_end)
        ].copy()
        test_df = training_table[training_table["race_date"] > valid_end].copy()

    train_df = apply_prediction_time_measurement_proxies(train_df)
    valid_df = apply_prediction_time_measurement_proxies(valid_df) if not valid_df.empty else valid_df
    test_df = apply_prediction_time_measurement_proxies(test_df) if not test_df.empty else test_df

    schema_frames = [train_df.head(200)]
    if not valid_df.empty:
        schema_frames.append(valid_df.head(200))
    if not test_df.empty:
        schema_frames.append(test_df.head(200))
    schema_df = pd.concat(schema_frames, ignore_index=True)
    feature_columns = infer_feature_columns(schema_df)
    categorical_columns = infer_categorical_columns(schema_df, feature_columns)
    del schema_df, schema_frames

    catboost_model = None
    if is_catboost_enabled(config):
        catboost_model = train_catboost(train_df, valid_df, feature_columns, categorical_columns, config)
        collect_garbage()
    lightgbm_model = train_lightgbm(train_df, valid_df, feature_columns, categorical_columns, config)
    collect_garbage()
    lightgbm_seed_ensemble_models = train_lightgbm_seed_ensemble(
        train_df,
        valid_df,
        feature_columns,
        categorical_columns,
        config,
        lightgbm_model,
    )
    collect_garbage()
    lightgbm_variant_models = train_lightgbm_variants(train_df, valid_df, feature_columns, categorical_columns, config)
    collect_garbage()
    lightgbm_regression_variant_models = train_lightgbm_regression_variants(
        train_df,
        valid_df,
        feature_columns,
        categorical_columns,
        config,
    )
    collect_garbage()
    xgboost_variant_models = train_xgboost_variants(train_df, valid_df, feature_columns, categorical_columns, config)
    collect_garbage()
    xgboost_regression_variant_models = train_xgboost_regression_variants(
        train_df,
        valid_df,
        feature_columns,
        categorical_columns,
        config,
    )
    collect_garbage()
    random_forest_regression_variant_models = train_random_forest_regression_variants(
        train_df,
        valid_df,
        feature_columns,
        categorical_columns,
        config,
    )
    collect_garbage()
    ridge_regression_variant_models = train_ridge_regression_variants(
        train_df,
        valid_df,
        feature_columns,
        categorical_columns,
        config,
    )
    collect_garbage()
    neural_regression_variant_models = train_neural_regression_variants(
        train_df,
        valid_df,
        feature_columns,
        categorical_columns,
        config,
    )
    collect_garbage()
    classifier_models = train_classifiers(train_df, valid_df, feature_columns, categorical_columns, config)
    collect_garbage()
    flow_model = None
    flow_classes = None
    staged_models: dict[str, lgb.Booster] = {}

    models = {
        **({"catboost": catboost_model} if catboost_model is not None else {}),
        "lightgbm": lightgbm_model,
        **lightgbm_seed_ensemble_models,
        **lightgbm_variant_models,
        **lightgbm_regression_variant_models,
        **xgboost_variant_models,
        **xgboost_regression_variant_models,
        **random_forest_regression_variant_models,
        **ridge_regression_variant_models,
        **neural_regression_variant_models,
    }
    ensemble_weights = optimize_ensemble_weights(
        models,
        valid_df,
        feature_columns,
        categorical_columns,
        config=config,
    )
    ensemble_weights["scenario_metric_min_races"] = int(
        get_phase3_settings(config)["evaluation"].get("scenario_min_races", 100)
    )
    trifecta_calibrator = fit_trifecta_calibrator(
        models,
        ensemble_weights,
        valid_df,
        feature_columns,
        categorical_columns,
    )
    collect_garbage()

    ranker_metrics = {
        **(
            {
                "catboost": evaluate_model_bundle(
                    catboost_model,
                    "catboost",
                    train_df,
                    valid_df,
                    test_df,
                    feature_columns,
                    categorical_columns,
                )
            }
            if catboost_model is not None
            else {}
        ),
        "lightgbm": evaluate_model_bundle(
            lightgbm_model,
            "lightgbm",
            train_df,
            valid_df,
            test_df,
            feature_columns,
            categorical_columns,
        ),
        **{
            name: evaluate_model_bundle(
                model,
                name,
                train_df,
                valid_df,
                test_df,
                feature_columns,
                categorical_columns,
            )
            for name, model in lightgbm_seed_ensemble_models.items()
        },
        **{
            name: evaluate_model_bundle(
                model,
                name,
                train_df,
                valid_df,
                test_df,
                feature_columns,
                categorical_columns,
            )
            for name, model in lightgbm_variant_models.items()
        },
        **{
            name: evaluate_model_bundle(
                model,
                name,
                train_df,
                valid_df,
                test_df,
                feature_columns,
                categorical_columns,
            )
            for name, model in lightgbm_regression_variant_models.items()
        },
        **{
            name: evaluate_model_bundle(
                model,
                name,
                train_df,
                valid_df,
                test_df,
                feature_columns,
                categorical_columns,
            )
            for name, model in xgboost_variant_models.items()
        },
        **{
            name: evaluate_model_bundle(
                model,
                name,
                train_df,
                valid_df,
                test_df,
                feature_columns,
                categorical_columns,
            )
            for name, model in xgboost_regression_variant_models.items()
        },
        **{
            name: evaluate_model_bundle(
                model,
                name,
                train_df,
                valid_df,
                test_df,
                feature_columns,
                categorical_columns,
            )
            for name, model in random_forest_regression_variant_models.items()
        },
        **{
            name: evaluate_model_bundle(
                model,
                name,
                train_df,
                valid_df,
                test_df,
                feature_columns,
                categorical_columns,
            )
            for name, model in ridge_regression_variant_models.items()
        },
        **{
            name: evaluate_model_bundle(
                model,
                name,
                train_df,
                valid_df,
                test_df,
                feature_columns,
                categorical_columns,
            )
            for name, model in neural_regression_variant_models.items()
        },
        "ensemble": evaluate_ensemble(
            models,
            ensemble_weights,
            train_df,
            valid_df,
            test_df,
            feature_columns,
            categorical_columns,
        ),
    }
    trifecta_v1_model_metrics = evaluate_trifecta_v1_model_metrics(
        models,
        ensemble_weights,
        trifecta_calibrator,
        valid_df,
        test_df,
        feature_columns,
        categorical_columns,
        classifier_models=classifier_models,
        flow_model=flow_model,
        flow_classes=flow_classes,
        staged_models=staged_models,
        config=config,
    )
    trifecta_v1_metrics = trifecta_v1_model_metrics["ensemble"]
    train_eval_df = train_df
    valid_eval_df = valid_df
    test_eval_df = test_df
    classifier_metrics = evaluate_classifier_models(
        classifier_models,
        train_eval_df,
        valid_eval_df,
        test_eval_df,
        feature_columns,
        categorical_columns,
    )
    flow_metrics = evaluate_flow_model(
        flow_model,
        flow_classes,
        train_eval_df,
        valid_eval_df,
        test_eval_df,
        feature_columns,
        categorical_columns,
    )
    staged_metrics = evaluate_staged_models(
        staged_models,
        train_eval_df,
        valid_eval_df,
        test_eval_df,
        feature_columns,
        categorical_columns,
    )

    metrics = {
        **ranker_metrics,
        "ensemble_weights": ensemble_weights,
        "trifecta": trifecta_v1_metrics,
        "ranker_metrics": ranker_metrics,
        "trifecta_v1_metrics": trifecta_v1_metrics,
        "trifecta_v1_model_metrics": trifecta_v1_model_metrics,
        "classifier_metrics": classifier_metrics,
        "flow_model_metrics": flow_metrics,
        "staged_model_metrics": staged_metrics,
    }
    return (
        models,
        feature_columns,
        metrics,
        trifecta_calibrator,
        classifier_models,
        flow_model,
        flow_classes,
        staged_models,
        None,
    )


def train_ranker_from_splits(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: dict,
    progress_callback: Callable[[str], None] | None = None,
    resume: bool = False,
    reset_train_checkpoint: bool = False,
    skip_evaluation: bool = False,
    skip_variant_evaluation: bool = False,
) -> tuple[
    dict[str, Any],
    list[str],
    dict[str, Any],
    IsotonicRegression,
    dict[str, lgb.Booster],
    lgb.Booster | None,
    list[str] | None,
    dict[str, lgb.Booster],
    Any | None,
]:
    def progress(message: str) -> None:
        if progress_callback is not None:
            progress_callback(message)

    def skipped_metrics(name: str) -> dict[str, Any]:
        return {"status": "skipped_by_request", "model": name}

    def evaluate_model_or_skip(model: Any, name: str) -> dict[str, Any]:
        if skip_evaluation:
            progress(f"skipping {name} evaluation: --skip-evaluation")
            return skipped_metrics(name)
        if skip_variant_evaluation:
            progress(f"skipping {name} evaluation: --skip-variant-evaluation")
            return {"status": "skipped_by_variant_evaluation_request", "model": name}
        return evaluate_model_bundle(
            model,
            name,
            train_df,
            valid_df,
            test_df,
            feature_columns,
            categorical_columns,
        )

    progress("applying prediction-time measurement proxies")
    train_df = apply_prediction_time_measurement_proxies(train_df)
    valid_df = apply_prediction_time_measurement_proxies(valid_df) if not valid_df.empty else valid_df
    test_df = apply_prediction_time_measurement_proxies(test_df) if not test_df.empty else test_df

    progress("inferring feature columns")
    schema_frames = [train_df.head(200)]
    if not valid_df.empty:
        schema_frames.append(valid_df.head(200))
    if not test_df.empty:
        schema_frames.append(test_df.head(200))
    schema_df = pd.concat(schema_frames, ignore_index=True)
    feature_columns = infer_feature_columns(schema_df)
    categorical_columns = infer_categorical_columns(schema_df, feature_columns)
    del schema_df, schema_frames
    collect_garbage()
    progress(f"inferred features: numeric_and_categorical={len(feature_columns)}, categorical={len(categorical_columns)}")

    artifacts = get_artifact_paths(config)
    checkpoint_path = artifacts["train_checkpoint_path"]
    signature = train_checkpoint_signature(config, train_df, valid_df, test_df)
    if reset_train_checkpoint and checkpoint_path.exists():
        progress(f"resetting train checkpoint: {checkpoint_path}")
        checkpoint_path.unlink()
    checkpoint = load_train_checkpoint(checkpoint_path, signature) if resume else {
        "signature": signature,
        "completed": {},
        "metrics": {},
    }
    save_train_checkpoint(checkpoint_path, checkpoint)
    artifacts["features_path"].parent.mkdir(parents=True, exist_ok=True)
    artifacts["features_path"].write_text(json.dumps(feature_columns, ensure_ascii=False, indent=2), encoding="utf-8")
    checkpoint_metrics = checkpoint.setdefault("metrics", {})

    catboost_model = None
    catboost_metrics: dict[str, Any] = {}
    if not is_catboost_enabled(config):
        progress("skipping catboost ranker: disabled")
    elif resume and train_stage_completed(checkpoint, "catboost", [artifacts["catboost_model_path"]]) and "catboost" in checkpoint_metrics:
        progress("skipping catboost ranker: train checkpoint completed")
        catboost_model = CatBoostRanker()
        catboost_model.load_model(str(artifacts["catboost_model_path"]))
        catboost_metrics = checkpoint_metrics.get("catboost", {})
    else:
        progress("training catboost ranker")
        catboost_model = train_catboost(train_df, valid_df, feature_columns, categorical_columns, config)
        artifacts["catboost_model_path"].parent.mkdir(parents=True, exist_ok=True)
        catboost_model.save_model(artifacts["catboost_model_path"])
        progress("evaluating catboost ranker")
        catboost_metrics = evaluate_model_or_skip(catboost_model, "catboost")
        mark_train_stage_completed(checkpoint_path, checkpoint, "catboost", catboost_metrics)
    collect_garbage()

    if resume and train_stage_completed(checkpoint, "lightgbm", [artifacts["lightgbm_model_path"]]) and "lightgbm" in checkpoint_metrics:
        progress("skipping lightgbm ranker: train checkpoint completed")
        lightgbm_model = lgb.Booster(model_file=str(artifacts["lightgbm_model_path"]))
        lightgbm_metrics = checkpoint_metrics.get("lightgbm", {})
    else:
        progress("training lightgbm ranker")
        lightgbm_model = train_lightgbm(train_df, valid_df, feature_columns, categorical_columns, config)
        artifacts["lightgbm_model_path"].parent.mkdir(parents=True, exist_ok=True)
        lightgbm_model.save_model(str(artifacts["lightgbm_model_path"]))
        progress("evaluating lightgbm ranker")
        lightgbm_metrics = evaluate_model_or_skip(lightgbm_model, "lightgbm")
        mark_train_stage_completed(checkpoint_path, checkpoint, "lightgbm", lightgbm_metrics)
    collect_garbage()

    seed_ensemble_paths = enabled_lightgbm_seed_ensemble_paths(config, artifacts["lightgbm_model_path"])
    if (
        resume
        and train_stage_completed(checkpoint, "lightgbm_seed_ensemble", seed_ensemble_paths)
        and "lightgbm_seed_ensemble" in checkpoint_metrics
    ):
        progress("skipping lightgbm seed ensemble: train checkpoint completed")
        lightgbm_seed_ensemble_models = {
            LIGHTGBM_SEED_ENSEMBLE_MODEL_NAME: load_lightgbm_seed_ensemble(
                config,
                artifacts["lightgbm_model_path"],
                lightgbm_model,
            )
        }
        lightgbm_seed_ensemble_metrics = checkpoint_metrics.get("lightgbm_seed_ensemble", {})
    else:
        lightgbm_seed_ensemble_models = train_lightgbm_seed_ensemble(
            train_df,
            valid_df,
            feature_columns,
            categorical_columns,
            config,
            lightgbm_model,
            progress_callback=progress,
        )
        if lightgbm_seed_ensemble_models:
            save_lightgbm_seed_ensemble(
                lightgbm_seed_ensemble_models[LIGHTGBM_SEED_ENSEMBLE_MODEL_NAME],
                artifacts["lightgbm_model_path"],
            )
            lightgbm_seed_ensemble_metrics = {
                name: evaluate_model_or_skip(model, name)
                for name, model in lightgbm_seed_ensemble_models.items()
            }
        else:
            lightgbm_seed_ensemble_metrics = {}
        mark_train_stage_completed(
            checkpoint_path,
            checkpoint,
            "lightgbm_seed_ensemble",
            lightgbm_seed_ensemble_metrics,
        )
    collect_garbage()

    variant_paths = enabled_lightgbm_variant_paths(config, artifacts["lightgbm_model_path"])
    if resume and train_stage_completed(checkpoint, "lightgbm_variants", variant_paths) and "lightgbm_variants" in checkpoint_metrics:
        progress("skipping lightgbm variants: train checkpoint completed")
        lightgbm_variant_models = {
            str(variant["name"]): lgb.Booster(
                model_file=str(lightgbm_variant_model_path(artifacts["lightgbm_model_path"], str(variant["name"])))
            )
            for variant in get_enabled_lightgbm_variants(config)
        }
        lightgbm_variant_metrics = checkpoint_metrics.get("lightgbm_variants", {})
    else:
        lightgbm_variant_models = train_lightgbm_variants(
            train_df,
            valid_df,
            feature_columns,
            categorical_columns,
            config,
            progress_callback=progress,
        )
        save_lightgbm_variants(lightgbm_variant_models, artifacts["lightgbm_model_path"])
        lightgbm_variant_metrics = {
            name: evaluate_model_or_skip(model, name)
            for name, model in lightgbm_variant_models.items()
        }
        mark_train_stage_completed(checkpoint_path, checkpoint, "lightgbm_variants", lightgbm_variant_metrics)
    collect_garbage()

    regression_variant_paths = enabled_lightgbm_regression_variant_paths(config, artifacts["lightgbm_model_path"])
    if (
        resume
        and train_stage_completed(checkpoint, "lightgbm_regression_variants", regression_variant_paths)
        and "lightgbm_regression_variants" in checkpoint_metrics
    ):
        progress("skipping lightgbm regression variants: train checkpoint completed")
        lightgbm_regression_variant_models = {
            str(variant["name"]): lgb.Booster(
                model_file=str(lightgbm_variant_model_path(artifacts["lightgbm_model_path"], str(variant["name"])))
            )
            for variant in get_enabled_lightgbm_regression_variants(config)
        }
        lightgbm_regression_variant_metrics = checkpoint_metrics.get("lightgbm_regression_variants", {})
    else:
        lightgbm_regression_variant_models = train_lightgbm_regression_variants(
            train_df,
            valid_df,
            feature_columns,
            categorical_columns,
            config,
            progress_callback=progress,
        )
        save_lightgbm_variants(lightgbm_regression_variant_models, artifacts["lightgbm_model_path"])
        lightgbm_regression_variant_metrics = {
            name: evaluate_model_or_skip(model, name)
            for name, model in lightgbm_regression_variant_models.items()
        }
        mark_train_stage_completed(
            checkpoint_path,
            checkpoint,
            "lightgbm_regression_variants",
            lightgbm_regression_variant_metrics,
        )
    collect_garbage()

    xgboost_variant_paths = enabled_xgboost_variant_paths(config, artifacts["xgboost_model_path"])
    if (
        resume
        and train_stage_completed(checkpoint, "xgboost_variants", xgboost_variant_paths)
        and "xgboost_variants" in checkpoint_metrics
    ):
        progress("skipping xgboost variants: train checkpoint completed")
        xgb_module = require_xgboost()
        xgboost_variant_models = {}
        for variant in get_enabled_xgboost_variants(config):
            name = str(variant["name"])
            model = xgb_module.Booster()
            model.load_model(str(xgboost_variant_model_path(artifacts["xgboost_model_path"], name)))
            xgboost_variant_models[name] = model
        xgboost_variant_metrics = checkpoint_metrics.get("xgboost_variants", {})
    else:
        xgboost_variant_models = train_xgboost_variants(
            train_df,
            valid_df,
            feature_columns,
            categorical_columns,
            config,
            progress_callback=progress,
        )
        save_xgboost_variants(xgboost_variant_models, artifacts["xgboost_model_path"])
        xgboost_variant_metrics = {
            name: evaluate_model_or_skip(model, name)
            for name, model in xgboost_variant_models.items()
        }
        mark_train_stage_completed(checkpoint_path, checkpoint, "xgboost_variants", xgboost_variant_metrics)
    collect_garbage()

    xgboost_regression_variant_paths = enabled_xgboost_regression_variant_paths(config, artifacts["xgboost_model_path"])
    if (
        resume
        and train_stage_completed(checkpoint, "xgboost_regression_variants", xgboost_regression_variant_paths)
        and "xgboost_regression_variants" in checkpoint_metrics
    ):
        progress("skipping xgboost regression variants: train checkpoint completed")
        xgb_module = require_xgboost()
        xgboost_regression_variant_models = {}
        for variant in get_enabled_xgboost_regression_variants(config):
            name = str(variant["name"])
            model = xgb_module.Booster()
            model.load_model(str(xgboost_variant_model_path(artifacts["xgboost_model_path"], name)))
            xgboost_regression_variant_models[name] = model
        xgboost_regression_variant_metrics = checkpoint_metrics.get("xgboost_regression_variants", {})
    else:
        xgboost_regression_variant_models = train_xgboost_regression_variants(
            train_df,
            valid_df,
            feature_columns,
            categorical_columns,
            config,
            progress_callback=progress,
        )
        save_xgboost_variants(xgboost_regression_variant_models, artifacts["xgboost_model_path"])
        xgboost_regression_variant_metrics = {
            name: evaluate_model_or_skip(model, name)
            for name, model in xgboost_regression_variant_models.items()
        }
        mark_train_stage_completed(
            checkpoint_path,
            checkpoint,
            "xgboost_regression_variants",
            xgboost_regression_variant_metrics,
        )
    collect_garbage()

    random_forest_regression_variant_paths = enabled_random_forest_regression_variant_paths(
        config,
        artifacts["random_forest_model_path"],
    )
    if (
        resume
        and train_stage_completed(
            checkpoint,
            "random_forest_regression_variants",
            random_forest_regression_variant_paths,
        )
        and "random_forest_regression_variants" in checkpoint_metrics
    ):
        progress("skipping random forest regression variants: train checkpoint completed")
        random_forest_regression_variant_models = {
            str(variant["name"]): joblib.load(
                random_forest_variant_model_path(artifacts["random_forest_model_path"], str(variant["name"]))
            )
            for variant in get_enabled_random_forest_regression_variants(config)
        }
        random_forest_regression_variant_metrics = checkpoint_metrics.get("random_forest_regression_variants", {})
    else:
        random_forest_regression_variant_models = train_random_forest_regression_variants(
            train_df,
            valid_df,
            feature_columns,
            categorical_columns,
            config,
            progress_callback=progress,
        )
        save_random_forest_variants(random_forest_regression_variant_models, artifacts["random_forest_model_path"])
        random_forest_regression_variant_metrics = {
            name: evaluate_model_or_skip(model, name)
            for name, model in random_forest_regression_variant_models.items()
        }
        mark_train_stage_completed(
            checkpoint_path,
            checkpoint,
            "random_forest_regression_variants",
            random_forest_regression_variant_metrics,
        )
    collect_garbage()

    ridge_regression_variant_paths = enabled_ridge_regression_variant_paths(
        config,
        artifacts["ridge_model_path"],
    )
    if (
        resume
        and train_stage_completed(checkpoint, "ridge_regression_variants", ridge_regression_variant_paths)
        and "ridge_regression_variants" in checkpoint_metrics
    ):
        progress("skipping ridge regression variants: train checkpoint completed")
        ridge_regression_variant_models = {
            str(variant["name"]): joblib.load(
                ridge_variant_model_path(artifacts["ridge_model_path"], str(variant["name"]))
            )
            for variant in get_enabled_ridge_regression_variants(config)
        }
        ridge_regression_variant_metrics = checkpoint_metrics.get("ridge_regression_variants", {})
    else:
        ridge_regression_variant_models = train_ridge_regression_variants(
            train_df,
            valid_df,
            feature_columns,
            categorical_columns,
            config,
            progress_callback=progress,
        )
        save_ridge_variants(ridge_regression_variant_models, artifacts["ridge_model_path"])
        ridge_regression_variant_metrics = {
            name: evaluate_model_or_skip(model, name)
            for name, model in ridge_regression_variant_models.items()
        }
        mark_train_stage_completed(
            checkpoint_path,
            checkpoint,
            "ridge_regression_variants",
            ridge_regression_variant_metrics,
        )
    collect_garbage()

    neural_regression_variant_paths = enabled_neural_regression_variant_paths(
        config,
        artifacts["neural_model_path"],
    )
    if (
        resume
        and train_stage_completed(checkpoint, "neural_regression_variants", neural_regression_variant_paths)
        and "neural_regression_variants" in checkpoint_metrics
    ):
        progress("skipping neural regression variants: train checkpoint completed")
        neural_regression_variant_models = {
            str(variant["name"]): joblib.load(
                neural_variant_model_path(artifacts["neural_model_path"], str(variant["name"]))
            )
            for variant in get_enabled_neural_regression_variants(config)
        }
        neural_regression_variant_metrics = checkpoint_metrics.get("neural_regression_variants", {})
    else:
        neural_regression_variant_models = train_neural_regression_variants(
            train_df,
            valid_df,
            feature_columns,
            categorical_columns,
            config,
            progress_callback=progress,
        )
        save_neural_variants(neural_regression_variant_models, artifacts["neural_model_path"])
        neural_regression_variant_metrics = {
            name: evaluate_model_or_skip(model, name)
            for name, model in neural_regression_variant_models.items()
        }
        mark_train_stage_completed(
            checkpoint_path,
            checkpoint,
            "neural_regression_variants",
            neural_regression_variant_metrics,
        )
    collect_garbage()

    models = {
        **({"catboost": catboost_model} if catboost_model is not None else {}),
        "lightgbm": lightgbm_model,
        **lightgbm_seed_ensemble_models,
        **lightgbm_variant_models,
        **lightgbm_regression_variant_models,
        **xgboost_variant_models,
        **xgboost_regression_variant_models,
        **random_forest_regression_variant_models,
        **ridge_regression_variant_models,
        **neural_regression_variant_models,
    }
    if resume and train_stage_completed(checkpoint, "ensemble", [artifacts["ensemble_weights_path"]]) and "ensemble" in checkpoint_metrics:
        progress("skipping ensemble optimization: train checkpoint completed")
        ensemble_weights = load_ensemble_weights(artifacts["ensemble_weights_path"])
        ensemble_metrics = checkpoint_metrics.get("ensemble", {})
    else:
        progress("optimizing ensemble weights")
        ensemble_weights = optimize_ensemble_weights(
            models,
            valid_df,
            feature_columns,
            categorical_columns,
            config=config,
            progress_callback=progress,
        )
        ensemble_weights["scenario_metric_min_races"] = int(
            get_phase3_settings(config)["evaluation"].get("scenario_min_races", 100)
        )
        artifacts["ensemble_weights_path"].parent.mkdir(parents=True, exist_ok=True)
        artifacts["ensemble_weights_path"].write_text(
            json.dumps(ensemble_weights, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if skip_evaluation:
            progress("skipping ensemble evaluation: --skip-evaluation")
            ensemble_metrics = skipped_metrics("ensemble")
        else:
            progress("evaluating ensemble")
            ensemble_metrics = evaluate_ensemble(
                models,
                ensemble_weights,
                train_df,
                valid_df,
                test_df,
                feature_columns,
                categorical_columns,
            )
        mark_train_stage_completed(checkpoint_path, checkpoint, "ensemble", ensemble_metrics)
    collect_garbage()

    train_eval_df = train_df
    valid_eval_df = valid_df
    test_eval_df = test_df
    if resume and train_stage_completed(checkpoint, "classifiers", [artifacts["classifier_dir"]]) and "classifiers" in checkpoint_metrics:
        progress("skipping classifier models: train checkpoint completed")
        classifier_models = load_classifier_models(artifacts["classifier_dir"])
        classifier_metrics = checkpoint_metrics.get("classifiers", {})
    else:
        progress("training classifier models")
        classifier_models = train_classifiers(train_df, valid_df, feature_columns, categorical_columns, config)
        save_classifier_models(classifier_models, artifacts["classifier_dir"])
        if skip_evaluation:
            progress("skipping classifier evaluation: --skip-evaluation")
            classifier_metrics = {"status": "skipped_by_request"}
        else:
            progress("evaluating classifier models")
            classifier_metrics = evaluate_classifier_models(
                classifier_models,
                train_eval_df,
                valid_eval_df,
                test_eval_df,
                feature_columns,
                categorical_columns,
            )
        mark_train_stage_completed(checkpoint_path, checkpoint, "classifiers", classifier_metrics)
    collect_garbage()

    flow_model = None
    flow_classes = None
    staged_models: dict[str, lgb.Booster] = {}
    flow_metrics = {"status": "skipped_by_base_train"}
    staged_metrics = {"status": "skipped_by_base_train"}
    progress("skipping flow and staged models in base train")

    if resume and train_stage_completed(checkpoint, "trifecta_v1_calibrator", [artifacts["trifecta_calibrator_path"]]):
        progress("skipping trifecta v1 calibrator: train checkpoint completed")
        trifecta_calibrator = load_trifecta_calibrator(artifacts["trifecta_calibrator_path"])
    else:
        progress("fitting trifecta v1 calibrator")
        trifecta_calibrator = fit_trifecta_calibrator(
            models,
            ensemble_weights,
            valid_df,
            feature_columns,
            categorical_columns,
        )
        artifacts["trifecta_calibrator_path"].parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(trifecta_calibrator, artifacts["trifecta_calibrator_path"])
        mark_train_stage_completed(checkpoint_path, checkpoint, "trifecta_v1_calibrator", {"status": "completed"})
    collect_garbage()

    probability_adjustment_table: dict[str, Any] | None = None
    if skip_evaluation:
        progress("skipping probability adjustment table: --skip-evaluation")
        probability_adjustment_table = fit_top12_probability_adjustment_table(pd.DataFrame())
        artifacts["probability_adjustment_path"].parent.mkdir(parents=True, exist_ok=True)
        artifacts["probability_adjustment_path"].write_text(
            json.dumps(probability_adjustment_table, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        mark_train_stage_completed(
            checkpoint_path,
            checkpoint,
            "probability_adjustment",
            {"status": "skipped_by_request", "rule_count": 0.0},
        )
    elif resume and train_stage_completed(
        checkpoint,
        "probability_adjustment",
        [artifacts["probability_adjustment_path"]],
    ):
        progress("skipping probability adjustment table: train checkpoint completed")
        probability_adjustment_table = json.loads(
            artifacts["probability_adjustment_path"].read_text(encoding="utf-8")
        )
    else:
        progress("fitting probability adjustment table")
        probability_adjustment_table = fit_probability_adjustment_table(
            models,
            ensemble_weights,
            trifecta_calibrator,
            valid_df,
            feature_columns,
            categorical_columns,
            classifier_models=classifier_models,
        )
        artifacts["probability_adjustment_path"].parent.mkdir(parents=True, exist_ok=True)
        artifacts["probability_adjustment_path"].write_text(
            json.dumps(probability_adjustment_table, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        mark_train_stage_completed(
            checkpoint_path,
            checkpoint,
            "probability_adjustment",
            {
                "status": "completed",
                "rule_count": float(len(probability_adjustment_table.get("rules", []))),
            },
        )
    collect_garbage()

    if skip_evaluation:
        progress("skipping trifecta v1 metrics by model: --skip-evaluation")
        trifecta_v1_metrics = {"status": "skipped_by_request"}
        trifecta_v1_model_metrics = {"ensemble": trifecta_v1_metrics}
        mark_train_stage_completed(
            checkpoint_path,
            checkpoint,
            "trifecta_v1_model_metrics",
            trifecta_v1_model_metrics,
        )
        mark_train_stage_completed(checkpoint_path, checkpoint, "trifecta_v1_metrics", trifecta_v1_metrics)
    elif (
        resume
        and train_stage_completed(checkpoint, "trifecta_v1_model_metrics")
        and "trifecta_v1_model_metrics" in checkpoint_metrics
    ):
        progress("skipping trifecta v1 model metrics: train checkpoint completed")
        trifecta_v1_model_metrics = checkpoint_metrics.get("trifecta_v1_model_metrics", {})
        trifecta_v1_metrics = trifecta_v1_model_metrics.get("ensemble", {})
    else:
        progress("evaluating trifecta v1 metrics by model")
        trifecta_v1_model_metrics = evaluate_trifecta_v1_model_metrics(
            models,
            ensemble_weights,
            trifecta_calibrator,
            valid_df,
            test_df,
            feature_columns,
            categorical_columns,
            classifier_models=classifier_models,
            flow_model=flow_model,
            flow_classes=flow_classes,
            staged_models=staged_models,
            config=config,
            progress_callback=progress,
            skip_individual_models=skip_variant_evaluation,
        )
        trifecta_v1_metrics = trifecta_v1_model_metrics["ensemble"]
        mark_train_stage_completed(
            checkpoint_path,
            checkpoint,
            "trifecta_v1_model_metrics",
            trifecta_v1_model_metrics,
        )
        mark_train_stage_completed(checkpoint_path, checkpoint, "trifecta_v1_metrics", trifecta_v1_metrics)
    collect_garbage()

    ranker_metrics = {
        **({"catboost": catboost_metrics} if catboost_model is not None else {}),
        "lightgbm": lightgbm_metrics,
        **lightgbm_seed_ensemble_metrics,
        **lightgbm_variant_metrics,
        **lightgbm_regression_variant_metrics,
        **xgboost_variant_metrics,
        **xgboost_regression_variant_metrics,
        **random_forest_regression_variant_metrics,
        **ridge_regression_variant_metrics,
        **neural_regression_variant_metrics,
        "ensemble": ensemble_metrics,
    }
    metrics = {
        **ranker_metrics,
        "ensemble_weights": ensemble_weights,
        "trifecta": trifecta_v1_metrics,
        "ranker_metrics": ranker_metrics,
        "trifecta_v1_metrics": trifecta_v1_metrics,
        "trifecta_v1_model_metrics": trifecta_v1_model_metrics,
        "probability_adjustment": probability_adjustment_table or {"status": "skipped"},
        "classifier_metrics": classifier_metrics,
        "flow_model_metrics": flow_metrics,
        "staged_model_metrics": staged_metrics,
    }
    progress("base model training complete")
    return (
        models,
        feature_columns,
        metrics,
        trifecta_calibrator,
        classifier_models,
        flow_model,
        flow_classes,
        staged_models,
        None,
    )


def prepare_training_table(training_table: pd.DataFrame, config: dict) -> pd.DataFrame:
    frame = training_table.copy()
    frame["race_date"] = pd.to_datetime(frame["race_date"])
    if "is_top2" not in frame.columns and "finish_position" in frame.columns:
        frame["is_top2"] = (pd.to_numeric(frame["finish_position"], errors="coerce") <= 2).astype(int)

    data_config = config.get("data", {})
    min_date = data_config.get("min_date")
    max_date = data_config.get("max_date")
    if min_date:
        frame = frame[frame["race_date"] >= pd.Timestamp(min_date)].copy()
    if max_date:
        frame = frame[frame["race_date"] <= pd.Timestamp(max_date)].copy()
    return frame


def apply_prediction_time_measurement_proxies(df: pd.DataFrame) -> pd.DataFrame:
    """Replace result-time measurements with prediction-time proxies.

    Training, evaluation, calibration, and optimization all use the same kind of
    pre-race proxy values that live prediction has. The actual course is
    intentionally kept unchanged by request.
    """
    frame = df.copy()
    frame["start_timing"] = _first_available_numeric_series(
        frame,
        [
            "racer_venue_prev_avg_st",
            "racer_prev_avg_st",
            "racer_prev_avg_st_5",
            "racer_prev_avg_st_10",
            "racer_lane_prev_avg_st",
            "racer_venue_lane_prev_avg_st",
        ],
    )
    frame["exhibition_time"] = _first_available_numeric_series(
        frame,
        [
            "racer_venue_prev_avg_exhibition",
            "racer_prev_avg_exhibition",
        ],
    )
    if "race_id" in frame.columns and "lane" in frame.columns:
        frame = add_race_relative_features(drop_race_relative_features(frame, preserve_missing_sources=True))
    return frame


def _first_available_numeric_series(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    values = pd.Series(pd.NA, index=df.index, dtype="Float64")
    for column in columns:
        if column not in df.columns:
            continue
        values = values.fillna(pd.to_numeric(df[column], errors="coerce"))
    return values


HIGH_CORRELATION_DROP_COLUMNS = frozenset(
    {
        "racer_prev_avg_exhibition_race_rank",
        "racer_prev_avg_st_race_diff_mean_safe",
        "exhibition_time_race_diff_mean_safe",
        "pre_race_attack_candidate_lane",
        "attack_candidate_outer_count",
        "current_meet_win_count",
        "pre_race_attack_score_gap_candidate",
        "race_attack_pressure",
        "racer_lane_prev_avg_st_race_diff_mean_safe",
        "racer_prev_avg_exhibition",
        "racer_prev_avg_exhibition_race_diff_mean",
        "racer_prev_avg_st",
        "racer_prev_avg_st_10_race_diff_mean_safe",
        "racer_prev_avg_st_5_race_diff_mean_safe",
        "racer_prev_avg_st_gap_inner",
        "racer_prev_avg_st_gap_outer",
        "racer_prev_avg_st_race_diff_best",
        "racer_venue_lane_prev_avg_st_race_diff_mean_safe",
        "start_timing_race_diff_mean_safe",
        "racer_prev_avg_st_race_rank_low",
        "racer_prev_avg_st_race_zscore",
        "racer_prev_avg_st_race_rank",
        "racer_prev_avg_st_race_diff_mean",
        "start_timing_race_mean",
    }
)


MEDIUM_CORRELATION_DROP_COLUMNS = frozenset(
    {
        "exhibition_time_race_rank",
        "start_timing_race_std",
        "start_timing_race_rank",
        "racer_prev_avg_st_5_race_rank",
        "venue_course_prev_win_rate",
        "racer_prev_avg_st_10_race_rank",
        "venue_lane_prev_win_rate",
        "racer_lane_prev_avg_st_race_rank",
        "venue_course_prev_top2_rate_race_zscore",
        "venue_lane_prev_top2_rate_race_zscore",
        "racer_venue_lane_prev_avg_st_race_rank",
        "venue_course_prev_top2_rate",
        "venue_lane_prev_top2_rate",
        "venue_lane_prev_top3_rate_race_diff_mean",
        "venue_course_prev_top3_rate_race_diff_mean",
        "venue_lane_prev_top3_rate",
        "venue_course_prev_top3_rate",
    }
)


REDUNDANT_CORRELATION_DROP_COLUMNS = HIGH_CORRELATION_DROP_COLUMNS | MEDIUM_CORRELATION_DROP_COLUMNS


def infer_feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        column
        for column in df.columns
        if column not in DEFAULT_DROP_COLUMNS and column not in REDUNDANT_CORRELATION_DROP_COLUMNS
    ]


def infer_categorical_columns(df: pd.DataFrame, feature_columns: list[str]) -> list[str]:
    explicit_categorical = {
        "venue",
        "leg_type",
        "bet_type",
        "class_name",
        "branch",
        "weather",
        "wind_direction",
    }
    categorical_columns: list[str] = []
    for column in feature_columns:
        if column in explicit_categorical:
            categorical_columns.append(column)
            continue

        dtype_name = str(df[column].dtype)
        if dtype_name not in {"object", "string"}:
            continue

        non_null = df[column].dropna()
        if non_null.empty:
            continue

        numeric_probe = pd.to_numeric(non_null, errors="coerce")
        if numeric_probe.notna().all():
            continue
        categorical_columns.append(column)

    return categorical_columns


LEGACY_20260712_EXCLUDE_PREFIXES = (
    "pre_race_attack_",
    "race_attack_",
    "race_escape_",
    "race_inner_",
    "race_outer_",
    "lane_attack_",
    "lane_escape_",
    "lane_outer_",
    "attack_candidate_",
    "distance_from_attack_candidate",
    "is_attack_candidate",
    "is_inside_of_attack_candidate",
    "is_outside_of_attack_candidate",
    "venue_course_prev_",
    "venue_lane_prev_",
    "racer_course_prev_",
)


LEGACY_20260712_EXCLUDE_SUFFIXES = (
    "_race_rank_low",
    "_race_diff_best",
    "_race_diff_mean_safe",
    "_race_mean",
    "_race_std",
    "_race_zscore",
    "_gap_inner",
    "_gap_outer",
    "_gap_inner_mean",
    "_gap_outer_mean",
)


def select_feature_columns_for_set(feature_columns: list[str], feature_set: str | None) -> list[str]:
    name = str(feature_set or "full").strip()
    if name in {"", "full"}:
        return list(feature_columns)
    if name != "legacy_20260712":
        raise ValueError(f"Unsupported feature_set: {name}")
    selected = [
        column
        for column in feature_columns
        if not column.startswith(LEGACY_20260712_EXCLUDE_PREFIXES)
        and not column.endswith(LEGACY_20260712_EXCLUDE_SUFFIXES)
    ]
    if not selected:
        raise ValueError("legacy_20260712 feature_set selected no features.")
    return selected


def lightgbm_prediction_feature_columns(model: lgb.Booster, default_feature_columns: list[str]) -> list[str]:
    if not hasattr(model, "feature_name"):
        return default_feature_columns
    names = [str(name) for name in model.feature_name()]
    if names and names != ["Column_0"] and all(name in default_feature_columns for name in names):
        return names
    return default_feature_columns


def predict_lightgbm_seed_ensemble_raw(
    model_bundle: dict[str, Any],
    df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
) -> np.ndarray:
    boosters = model_bundle.get("models", {}) if isinstance(model_bundle, dict) else {}
    if not boosters:
        raise ValueError("LightGBM seed ensemble contains no models.")
    raw_parts: list[np.ndarray] = []
    for booster in boosters.values():
        model_feature_columns = lightgbm_prediction_feature_columns(booster, feature_columns)
        model_categorical_columns = [column for column in categorical_columns if column in model_feature_columns]
        frame = build_lightgbm_frame(df, model_feature_columns, model_categorical_columns)
        raw_parts.append(np.asarray(booster.predict(frame), dtype=float))
    return np.mean(np.vstack(raw_parts), axis=0)


def train_catboost(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    config: dict,
) -> CatBoostRanker:
    train_pool = build_catboost_pool(train_df, feature_columns, categorical_columns)
    valid_pool = build_catboost_pool(valid_df, feature_columns, categorical_columns)
    gpu_kwargs = catboost_training_kwargs(config)
    catboost_params = dict(get_catboost_settings(config).get("params", {}) or {})
    eval_metric = catboost_params.pop("eval_metric", config["model"]["eval_metric"])
    if gpu_kwargs and str(eval_metric).upper() == "NDCG":
        print("CatBoost GPU training detected; omitting eval_metric=NDCG to avoid CPU-side metric evaluation.")
        eval_metric = None

    model_kwargs = dict(
        iterations=catboost_params.pop("iterations", config["model"]["iterations"]),
        learning_rate=catboost_params.pop("learning_rate", config["model"]["learning_rate"]),
        depth=catboost_params.pop("depth", config["model"]["depth"]),
        loss_function=catboost_params.pop("loss_function", config["model"]["loss_function"]),
        random_seed=catboost_params.pop("random_seed", config["model"]["random_seed"]),
        verbose=100,
    )
    if eval_metric is not None:
        model_kwargs["eval_metric"] = eval_metric
    model_kwargs.update(catboost_params)
    model_kwargs.update(gpu_kwargs)

    model = CatBoostRanker(**model_kwargs)
    try:
        try:
            model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
        except CatBoostError as exc:
            if not gpu_kwargs:
                raise
            print(f"CatBoost GPU training failed; falling back to CPU. Reason: {exc}")
            fallback_kwargs = dict(model_kwargs)
            fallback_kwargs.pop("task_type", None)
            fallback_kwargs.pop("devices", None)
            fallback_kwargs["eval_metric"] = config["model"]["eval_metric"]
            model = CatBoostRanker(**fallback_kwargs)
            model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
        return model
    finally:
        del train_pool, valid_pool
        collect_garbage()


def build_catboost_pool(
    df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
) -> Pool:
    df = sort_for_grouping(df)
    data = df[feature_columns].copy()
    for column in feature_columns:
        if column in categorical_columns:
            data[column] = data[column].fillna("NA").astype(str)
        else:
            data[column] = pd.to_numeric(data[column], errors="coerce").astype(float)

    group_id, _ = pd.factorize(df["race_id"])
    cat_indices = [feature_columns.index(column) for column in categorical_columns]
    return Pool(
        data=data,
        label=df["target_rank"].astype(float),
        group_id=group_id,
        cat_features=cat_indices,
    )


def train_lightgbm(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    config: dict,
    *,
    param_overrides: dict[str, Any] | None = None,
    num_threads: int | None = None,
    sample_weight_column: str | None = None,
) -> lgb.Booster:
    train_lgb = build_lightgbm_frame(train_df, feature_columns, categorical_columns)
    valid_lgb = build_lightgbm_frame(valid_df, feature_columns, categorical_columns)

    train_group = train_df.groupby("race_id").size().sort_index().tolist()
    valid_group = valid_df.groupby("race_id").size().sort_index().tolist()

    train_dataset = lgb.Dataset(
        train_lgb,
        label=sort_for_grouping(train_df)["target_rank"].astype(float),
        weight=(
            pd.to_numeric(sort_for_grouping(train_df)[sample_weight_column], errors="coerce").fillna(1.0)
            if sample_weight_column and sample_weight_column in train_df.columns
            else None
        ),
        group=train_group,
        categorical_feature=[c for c in categorical_columns if c in train_lgb.columns],
        free_raw_data=False,
    )
    valid_dataset = lgb.Dataset(
        valid_lgb,
        label=sort_for_grouping(valid_df)["target_rank"].astype(float),
        group=valid_group,
        categorical_feature=[c for c in categorical_columns if c in valid_lgb.columns],
        reference=train_dataset,
        free_raw_data=False,
    )

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "learning_rate": config["model"]["learning_rate"],
        "num_leaves": 63,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbosity": -1,
        "seed": config["model"]["random_seed"],
        "ndcg_eval_at": [1, 3, 6],
    }
    if param_overrides:
        params.update(param_overrides)
    if num_threads is not None and int(num_threads) > 0:
        params["num_threads"] = int(num_threads)
    try:
        return train_lightgbm_with_optional_gpu(
            params,
            train_dataset,
            config,
            num_boost_round=config["model"]["iterations"],
            valid_sets=[valid_dataset],
            valid_names=["valid"],
            callbacks=[lgb.log_evaluation(100)],
        )
    finally:
        del train_lgb, valid_lgb, train_dataset, valid_dataset, train_group, valid_group
        collect_garbage()


def train_lightgbm_variants(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    config: dict,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, lgb.Booster]:
    variants = get_enabled_lightgbm_variants(config)
    if not variants:
        return {}
    settings = get_lightgbm_variant_settings(config)
    workers = min(int(settings.get("parallel_workers", 1)), len(variants))
    num_threads = int(settings.get("num_threads_per_variant", 2))
    _emit_progress(
        progress_callback,
        f"training lightgbm variants: workers={workers}, variants={len(variants)}",
    )

    def train_one(variant: dict[str, Any]) -> tuple[str, lgb.Booster]:
        name = str(variant["name"])
        variant_feature_columns = select_feature_columns_for_set(
            feature_columns,
            str(variant.get("feature_set", "full")),
        )
        variant_categorical_columns = [column for column in categorical_columns if column in variant_feature_columns]
        _emit_progress(
            progress_callback,
            f"training lightgbm variant: {name}, feature_set={variant.get('feature_set', 'full')}, "
            f"features={len(variant_feature_columns)}",
        )
        variant_train_df = train_df
        sample_weight_column = None
        upset_settings = dict(variant.get("upset_training", {}) or {})
        if bool(upset_settings.get("enabled", False)):
            variant_train_df = build_upset_variant_training_frame(
                train_df,
                config,
                upset_settings,
                progress_callback=progress_callback,
            )
            sample_weight_column = UPSET_TRAINING_WEIGHT_COLUMN
        value_recovery_settings = dict(variant.get("value_recovery_training", {}) or {})
        if bool(value_recovery_settings.get("enabled", False)):
            variant_train_df = build_value_recovery_variant_training_frame(
                variant_train_df,
                value_recovery_settings,
                base_weight_column=sample_weight_column,
                progress_callback=progress_callback,
            )
            sample_weight_column = VALUE_RECOVERY_TRAINING_WEIGHT_COLUMN
        model = train_lightgbm(
            variant_train_df,
            valid_df,
            variant_feature_columns,
            variant_categorical_columns,
            config,
            param_overrides=dict(variant.get("params", {}) or {}),
            num_threads=num_threads,
            sample_weight_column=sample_weight_column,
        )
        if variant_train_df is not train_df:
            del variant_train_df
            collect_garbage()
        _emit_progress(progress_callback, f"completed lightgbm variant: {name}")
        return name, model

    if workers <= 1 or len(variants) <= 1:
        return dict(train_one(variant) for variant in variants)

    trained: dict[str, lgb.Booster] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_name = {executor.submit(train_one, variant): str(variant["name"]) for variant in variants}
        for future in as_completed(future_to_name):
            name, model = future.result()
            trained[name] = model
    return trained


def build_value_recovery_variant_training_frame(
    train_df: pd.DataFrame,
    settings: dict[str, Any],
    *,
    base_weight_column: str | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    if train_df.empty:
        result = train_df.copy()
        result[VALUE_RECOVERY_TRAINING_WEIGHT_COLUMN] = pd.Series(dtype=float)
        return result
    if "trifecta_payout" not in train_df.columns:
        raise ValueError("value_recovery_training requires trifecta_payout column.")

    result = train_df.copy()
    payouts = pd.to_numeric(result["trifecta_payout"], errors="coerce").to_numpy(dtype=float)
    weights = np.select(
        [
            np.isfinite(payouts) & (payouts < 1000.0),
            np.isfinite(payouts) & (payouts < 5000.0),
            np.isfinite(payouts) & (payouts < 10000.0),
            np.isfinite(payouts) & (payouts < 30000.0),
            np.isfinite(payouts) & (payouts < 50000.0),
            np.isfinite(payouts) & (payouts >= 50000.0),
        ],
        [
            float(settings.get("payout_weight_under_1000", 0.7)),
            float(settings.get("payout_weight_1000_5000", 1.0)),
            float(settings.get("payout_weight_5000_10000", 1.3)),
            float(settings.get("payout_weight_10000_30000", 1.8)),
            float(settings.get("payout_weight_30000_50000", 1.2)),
            float(settings.get("payout_weight_50000_over", 0.6)),
        ],
        default=1.0,
    )
    final_weights = weights.astype(float)
    if base_weight_column and base_weight_column in result.columns:
        base_weights = pd.to_numeric(result[base_weight_column], errors="coerce").fillna(1.0).to_numpy(dtype=float)
        final_weights = final_weights * base_weights
    result[VALUE_RECOVERY_TRAINING_WEIGHT_COLUMN] = final_weights
    race_count = int(result["race_id"].nunique()) if "race_id" in result.columns else int(len(result))
    _emit_progress(
        progress_callback,
        "value recovery variant training data: "
        f"races={race_count}, mean_weight={float(np.mean(final_weights)):.4g}, "
        f"high_value_races={int(result.loc[payouts >= 10000.0, 'race_id'].nunique()) if 'race_id' in result.columns else int(np.sum(payouts >= 10000.0))}",
    )
    return result


def build_upset_variant_training_frame(
    recent_train_df: pd.DataFrame,
    config: dict,
    settings: dict[str, Any],
    progress_callback: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    if recent_train_df.empty:
        return recent_train_df.copy()
    training_table_path = Path(config.get("data", {}).get("training_table", ""))
    if not training_table_path.exists():
        raise FileNotFoundError(f"Upset variant training table not found: {training_table_path}")

    latest_date = pd.to_datetime(recent_train_df["race_date"], errors="coerce").max().normalize()
    recent_years = float(settings.get("recent_years", 3.5))
    history_years = float(settings.get("history_years", 10.0))
    history_start = (latest_date - _years_to_date_offset(history_years) + pd.Timedelta(days=1)).normalize()
    history_end = (latest_date - _years_to_date_offset(recent_years)).normalize()
    historical_df = pd.read_parquet(
        training_table_path,
        filters=[("race_date", ">=", history_start), ("race_date", "<=", history_end)],
    )
    historical_config = json.loads(json.dumps(config))
    historical_config.setdefault("data", {})["min_date"] = history_start.strftime("%Y-%m-%d")
    historical_config["data"]["max_date"] = history_end.strftime("%Y-%m-%d")
    historical_df = prepare_training_table(historical_df, historical_config)
    historical_df = apply_prediction_time_measurement_proxies(historical_df)
    sampled_history = select_upset_history_races(historical_df, latest_date, settings)
    del historical_df
    collect_garbage()

    recent = recent_train_df.copy()
    recent[UPSET_TRAINING_WEIGHT_COLUMN] = 1.0
    combined = pd.concat([recent, sampled_history], ignore_index=True, sort=False)
    _emit_progress(
        progress_callback,
        "upset variant training data: "
        f"recent_races={recent['race_id'].nunique()}, historical_races={sampled_history['race_id'].nunique()}, "
        f"high_payout_historical_races={sampled_history.loc[sampled_history['trifecta_payout'] >= float(settings.get('payout_threshold', 10000.0)), 'race_id'].nunique()}",
    )
    return combined


def select_upset_history_races(
    historical_df: pd.DataFrame,
    latest_date: pd.Timestamp,
    settings: dict[str, Any],
) -> pd.DataFrame:
    if historical_df.empty:
        result = historical_df.copy()
        result[UPSET_TRAINING_WEIGHT_COLUMN] = pd.Series(dtype=float)
        return result
    required = {"race_id", "race_date", "trifecta_payout"}
    missing = required - set(historical_df.columns)
    if missing:
        raise ValueError(f"Upset training data is missing columns: {sorted(missing)}")

    race_columns = ["race_id", "race_date", "trifecta_payout"]
    if "venue" in historical_df.columns:
        race_columns.append("venue")
    races = historical_df[race_columns].drop_duplicates("race_id").copy()
    races["race_date"] = pd.to_datetime(races["race_date"], errors="coerce")
    races["trifecta_payout"] = pd.to_numeric(races["trifecta_payout"], errors="coerce")
    races = races.dropna(subset=["race_date", "trifecta_payout"])
    threshold = float(settings.get("payout_threshold", 10000.0))
    high = races[races["trifecta_payout"] >= threshold].copy()
    controls = races[races["trifecta_payout"] < threshold].copy()
    if high.empty:
        result = historical_df.iloc[0:0].copy()
        result[UPSET_TRAINING_WEIGHT_COLUMN] = pd.Series(dtype=float)
        return result

    high["_month"] = high["race_date"].dt.to_period("M").astype(str)
    controls["_month"] = controls["race_date"].dt.to_period("M").astype(str)
    strata = ["_month"]
    if "venue" in high.columns:
        strata.append("venue")
    ratio = int(settings.get("control_ratio", 3))
    seed = int(settings.get("random_seed", 42))
    selected_control_ids: list[str] = []
    if ratio > 0:
        high_counts = high.groupby(strata, dropna=False).size()
        for key, high_count in high_counts.items():
            key_tuple = key if isinstance(key, tuple) else (key,)
            mask = pd.Series(True, index=controls.index)
            for column, value in zip(strata, key_tuple):
                mask &= controls[column].eq(value)
            candidates = controls.loc[mask, "race_id"]
            take = min(len(candidates), int(high_count) * ratio)
            if take:
                stratum_seed = (
                    seed + int(hashlib.sha256(str(key_tuple).encode("utf-8")).hexdigest()[:8], 16)
                ) % (2**32 - 1)
                selected_control_ids.extend(candidates.sample(n=take, random_state=stratum_seed).astype(str).tolist())

    selected_ids = set(high["race_id"].astype(str)) | set(selected_control_ids)
    selected = historical_df[historical_df["race_id"].astype(str).isin(selected_ids)].copy()
    payout = pd.to_numeric(selected["trifecta_payout"], errors="coerce").fillna(0.0)
    race_date = pd.to_datetime(selected["race_date"], errors="coerce")
    recent_history_cutoff = latest_date - _years_to_date_offset(float(settings.get("middle_cutoff_years", 7.0)))
    time_weight = np.where(
        race_date >= recent_history_cutoff,
        float(settings.get("recent_history_weight", 0.7)),
        float(settings.get("older_history_weight", 0.3)),
    )
    payout_weight = np.select(
        [payout >= 100000.0, payout >= 50000.0, payout >= threshold],
        [
            float(settings.get("payout_weight_100000", 5.0)),
            float(settings.get("payout_weight_50000", 3.0)),
            float(settings.get("payout_weight_10000", 2.0)),
        ],
        default=1.0,
    )
    selected[UPSET_TRAINING_WEIGHT_COLUMN] = np.asarray(time_weight, dtype=float) * np.asarray(
        payout_weight, dtype=float
    )
    return selected


def train_lightgbm_seed_ensemble(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    config: dict,
    base_model: lgb.Booster,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    settings = get_lightgbm_seed_ensemble_settings(config)
    if not bool(settings.get("enabled", False)):
        return {}

    seeds = [int(seed) for seed in settings.get("seeds", [])]
    base_seed = int(config.get("model", {}).get("random_seed", seeds[0] if seeds else 42))
    extra_seeds = [seed for seed in seeds if seed != base_seed]
    if not extra_seeds:
        return {}

    workers = min(int(settings.get("parallel_workers", 1)), len(extra_seeds))
    num_threads = int(settings.get("num_threads_per_seed", 2))
    _emit_progress(
        progress_callback,
        f"training lightgbm seed ensemble: workers={workers}, seeds={seeds}",
    )

    def train_one(seed: int) -> tuple[int, lgb.Booster]:
        _emit_progress(progress_callback, f"training lightgbm seed model: seed={seed}")
        params = dict(settings.get("params", {}) or {})
        params.update(
            {
                "seed": seed,
                "bagging_seed": seed,
                "feature_fraction_seed": seed,
                "data_random_seed": seed,
                "drop_seed": seed,
            }
        )
        model = train_lightgbm(
            train_df,
            valid_df,
            feature_columns,
            categorical_columns,
            config,
            param_overrides=params,
            num_threads=num_threads,
        )
        _emit_progress(progress_callback, f"completed lightgbm seed model: seed={seed}")
        return seed, model

    trained: dict[int, lgb.Booster] = {base_seed: base_model}
    if workers <= 1 or len(extra_seeds) <= 1:
        trained.update(dict(train_one(seed) for seed in extra_seeds))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_seed = {executor.submit(train_one, seed): seed for seed in extra_seeds}
            for future in as_completed(future_to_seed):
                seed, model = future.result()
                trained[seed] = model

    ordered_models = {str(seed): trained[seed] for seed in seeds if seed in trained}
    if len(ordered_models) < 2:
        return {}
    return {
        LIGHTGBM_SEED_ENSEMBLE_MODEL_NAME: {
            "type": LIGHTGBM_SEED_ENSEMBLE_MODEL_NAME,
            "base_seed": base_seed,
            "seeds": seeds,
            "models": ordered_models,
        }
    }


def train_lightgbm_regression(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    config: dict,
    *,
    target: str = "finish_position",
    param_overrides: dict[str, Any] | None = None,
    num_threads: int | None = None,
) -> lgb.Booster:
    if target not in train_df.columns or target not in valid_df.columns:
        raise ValueError(f"LightGBM regression target column is missing: {target}")
    train_lgb = build_lightgbm_frame(train_df, feature_columns, categorical_columns)
    valid_lgb = build_lightgbm_frame(valid_df, feature_columns, categorical_columns)
    train_dataset = lgb.Dataset(
        train_lgb,
        label=pd.to_numeric(sort_for_grouping(train_df)[target], errors="coerce").astype(float),
        categorical_feature=[c for c in categorical_columns if c in train_lgb.columns],
        free_raw_data=False,
    )
    valid_dataset = lgb.Dataset(
        valid_lgb,
        label=pd.to_numeric(sort_for_grouping(valid_df)[target], errors="coerce").astype(float),
        categorical_feature=[c for c in categorical_columns if c in valid_lgb.columns],
        reference=train_dataset,
        free_raw_data=False,
    )
    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": config["model"]["learning_rate"],
        "num_leaves": 31,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.70,
        "bagging_fraction": 0.80,
        "bagging_freq": 1,
        "lambda_l2": 8.0,
        "verbosity": -1,
        "seed": config["model"]["random_seed"],
    }
    if param_overrides:
        params.update(param_overrides)
    if num_threads is not None and int(num_threads) > 0:
        params["num_threads"] = int(num_threads)
    try:
        return train_lightgbm_with_optional_gpu(
            params,
            train_dataset,
            config,
            num_boost_round=config["model"]["iterations"],
            valid_sets=[valid_dataset],
            valid_names=["valid"],
            callbacks=[lgb.log_evaluation(100)],
        )
    finally:
        del train_lgb, valid_lgb, train_dataset, valid_dataset
        collect_garbage()


def train_lightgbm_regression_variants(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    config: dict,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, lgb.Booster]:
    variants = get_enabled_lightgbm_regression_variants(config)
    if not variants:
        return {}
    settings = get_lightgbm_regression_variant_settings(config)
    workers = min(int(settings.get("parallel_workers", 1)), len(variants))
    num_threads = int(settings.get("num_threads_per_variant", 2))
    _emit_progress(
        progress_callback,
        f"training lightgbm regression variants: workers={workers}, variants={len(variants)}",
    )

    def train_one(variant: dict[str, Any]) -> tuple[str, lgb.Booster]:
        name = str(variant["name"])
        _emit_progress(progress_callback, f"training lightgbm regression variant: {name}")
        model = train_lightgbm_regression(
            train_df,
            valid_df,
            feature_columns,
            categorical_columns,
            config,
            target=str(variant.get("target", "finish_position")),
            param_overrides=dict(variant.get("params", {}) or {}),
            num_threads=num_threads,
        )
        _emit_progress(progress_callback, f"completed lightgbm regression variant: {name}")
        return name, model

    if workers <= 1 or len(variants) <= 1:
        return dict(train_one(variant) for variant in variants)

    trained: dict[str, lgb.Booster] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_name = {executor.submit(train_one, variant): str(variant["name"]) for variant in variants}
        for future in as_completed(future_to_name):
            name, model = future.result()
            trained[name] = model
    return trained


def require_xgboost() -> Any:
    if xgb is None:
        raise ImportError(
            "xgboost is required when models.xgboost_variants.enabled=true. "
            "Install dependencies with: python -m pip install -r requirements.txt"
        )
    return xgb


def build_xgboost_frame(
    df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
) -> pd.DataFrame:
    df = sort_for_grouping(df)
    data = df[feature_columns].copy()
    for column in feature_columns:
        if column in categorical_columns:
            values = data[column].fillna("NA").astype(str)
            data[column] = (
                pd.util.hash_pandas_object(values, index=False).to_numpy(dtype=np.uint64) % np.uint64(2_000_000_000)
            ).astype("float32")
        else:
            data[column] = pd.to_numeric(data[column], errors="coerce").astype("float32")
    return data


def require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise ImportError(
            "torch is required when models.neural_regression_variants includes model_type=mlp. "
            "Install dependencies with: python -m pip install -e \".[nn]\""
        ) from exc
    return torch


def require_tabnet_regressor() -> Any:
    try:
        from pytorch_tabnet.tab_model import TabNetRegressor
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise ImportError(
            "pytorch-tabnet is required when models.neural_regression_variants includes model_type=tabnet. "
            "Install dependencies with: python -m pip install -e \".[nn]\""
        ) from exc
    return TabNetRegressor


def resolve_torch_device(torch: Any, params: dict[str, Any], *, key: str = "device", default: str = "auto") -> Any:
    device_name = str(params.get(key, default)).strip().lower()
    device_id = int(params.get("device_id", 0))
    if device_name in {"auto", ""}:
        resolved = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
    elif device_name in {"cuda", "gpu"}:
        if not torch.cuda.is_available():
            raise ValueError("Neural regression variant requested CUDA, but torch.cuda.is_available() is false.")
        resolved = f"cuda:{device_id}"
    elif device_name.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise ValueError("Neural regression variant requested CUDA, but torch.cuda.is_available() is false.")
        resolved = device_name
    elif device_name == "cpu":
        resolved = "cpu"
    else:
        raise ValueError(f"Unsupported neural device: {params.get(key)}")
    return torch.device(resolved)


def resolve_tabnet_device_name(params: dict[str, Any], *, key: str = "device", default: str = "auto") -> str:
    device_name = str(params.get(key, default)).strip().lower()
    if device_name in {"", "auto"}:
        return "auto"
    if device_name == "cpu":
        return "cpu"
    if device_name in {"cuda", "gpu"} or device_name.startswith("cuda:"):
        return "cuda"
    raise ValueError(f"Unsupported TabNet device: {params.get(key)}")


class TabularNeuralPreprocessor:
    def __init__(self, categorical_columns: list[str]) -> None:
        self.requested_categorical_columns = list(categorical_columns)
        self.feature_columns: list[str] = []
        self.numeric_columns: list[str] = []
        self.categorical_columns: list[str] = []
        self.numeric_medians: dict[str, float] = {}
        self.numeric_means: dict[str, float] = {}
        self.numeric_scales: dict[str, float] = {}
        self.category_maps: dict[str, dict[str, int]] = {}

    def fit(self, frame: pd.DataFrame, feature_columns: list[str]) -> "TabularNeuralPreprocessor":
        self.feature_columns = list(feature_columns)
        categorical_set = set(self.requested_categorical_columns)
        self.categorical_columns = [column for column in self.feature_columns if column in categorical_set]
        self.numeric_columns = [column for column in self.feature_columns if column not in categorical_set]

        for column in self.numeric_columns:
            values = pd.to_numeric(frame[column], errors="coerce").astype(float)
            median = float(values.median(skipna=True)) if values.notna().any() else 0.0
            filled = values.fillna(median)
            mean = float(filled.mean())
            scale = float(filled.std())
            if not np.isfinite(scale) or scale <= 1e-12:
                scale = 1.0
            self.numeric_medians[column] = median
            self.numeric_means[column] = mean
            self.numeric_scales[column] = scale

        for column in self.categorical_columns:
            values = frame[column].fillna("NA").astype(str)
            uniques = pd.Index(values.unique()).sort_values()
            self.category_maps[column] = {str(value): index + 1 for index, value in enumerate(uniques)}
        return self

    def transform_numeric(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.numeric_columns:
            return np.zeros((len(frame), 0), dtype=np.float32)
        parts = []
        for column in self.numeric_columns:
            values = pd.to_numeric(frame[column], errors="coerce").astype(float)
            filled = values.fillna(self.numeric_medians[column])
            scaled = (filled - self.numeric_means[column]) / self.numeric_scales[column]
            parts.append(scaled.to_numpy(dtype=np.float32))
        return np.column_stack(parts).astype(np.float32)

    def transform_categorical(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.categorical_columns:
            return np.zeros((len(frame), 0), dtype=np.int64)
        parts = []
        for column in self.categorical_columns:
            values = frame[column].fillna("NA").astype(str)
            mapping = self.category_maps[column]
            parts.append(values.map(mapping).fillna(0).to_numpy(dtype=np.int64))
        return np.column_stack(parts).astype(np.int64)

    def categorical_cardinalities(self) -> list[int]:
        return [len(self.category_maps[column]) + 1 for column in self.categorical_columns]

    def transform_tabnet(self, frame: pd.DataFrame) -> np.ndarray:
        numeric = self.transform_numeric(frame)
        categorical = self.transform_categorical(frame).astype(np.float32)
        if categorical.size == 0:
            return numeric.astype(np.float32)
        if numeric.size == 0:
            return categorical.astype(np.float32)
        return np.column_stack([numeric, categorical]).astype(np.float32)


class TorchEmbeddingMLPRegressor:
    def __init__(
        self,
        *,
        categorical_columns: list[str],
        params: dict[str, Any],
        random_seed: int,
    ) -> None:
        self.params = dict(params)
        self.random_seed = int(random_seed)
        self.preprocessor = TabularNeuralPreprocessor(categorical_columns)
        self.state_dict: dict[str, Any] | None = None
        self.input_size = 0
        self.embedding_dims: list[int] = []
        self._module: Any | None = None

    def _build_module(self) -> Any:
        torch = require_torch()
        nn = torch.nn
        numeric_size = len(self.preprocessor.numeric_columns)
        cardinalities = self.preprocessor.categorical_cardinalities()
        embedding_dim = int(self.params.get("embedding_dim", 8))
        embedding_dims = [min(max(2, embedding_dim), max(2, cardinality)) for cardinality in cardinalities]
        hidden_units = [int(value) for value in self.params.get("hidden_units", [256, 128, 64, 32])]
        dropout = float(self.params.get("dropout", 0.10))

        class Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embeddings = nn.ModuleList(
                    [nn.Embedding(cardinality, dim) for cardinality, dim in zip(cardinalities, embedding_dims)]
                )
                layers: list[Any] = []
                in_features = numeric_size + sum(embedding_dims)
                for units in hidden_units:
                    layers.append(nn.Linear(in_features, units))
                    layers.append(nn.ReLU())
                    if dropout > 0:
                        layers.append(nn.Dropout(dropout))
                    in_features = units
                layers.append(nn.Linear(in_features, 1))
                self.net = nn.Sequential(*layers)

            def forward(self, numeric: Any, categorical: Any) -> Any:
                if self.embeddings:
                    embedded = [embedding(categorical[:, index]) for index, embedding in enumerate(self.embeddings)]
                    x = torch.cat([numeric, *embedded], dim=1) if numeric.shape[1] else torch.cat(embedded, dim=1)
                else:
                    x = numeric
                return self.net(x).squeeze(1)

        return Module()

    def fit(
        self,
        train_frame: pd.DataFrame,
        y_train: pd.Series,
        feature_columns: list[str],
        valid_frame: pd.DataFrame,
        y_valid: pd.Series,
        sample_weight: pd.Series | np.ndarray | None = None,
    ) -> "TorchEmbeddingMLPRegressor":
        torch = require_torch()
        torch.manual_seed(self.random_seed)
        torch.set_num_threads(max(int(self.params.get("torch_num_threads", 2)), 1))
        device = resolve_torch_device(torch, self.params)
        self.preprocessor.fit(train_frame, feature_columns)
        train_numeric = torch.tensor(self.preprocessor.transform_numeric(train_frame), dtype=torch.float32, device=device)
        train_categorical = torch.tensor(self.preprocessor.transform_categorical(train_frame), dtype=torch.long, device=device)
        train_target = torch.tensor(
            pd.to_numeric(y_train, errors="coerce").to_numpy(dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
        valid_numeric = torch.tensor(self.preprocessor.transform_numeric(valid_frame), dtype=torch.float32, device=device)
        valid_categorical = torch.tensor(self.preprocessor.transform_categorical(valid_frame), dtype=torch.long, device=device)
        valid_target = torch.tensor(
            pd.to_numeric(y_valid, errors="coerce").to_numpy(dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
        train_weight = None
        if sample_weight is not None:
            weight_values = pd.to_numeric(pd.Series(sample_weight), errors="coerce").fillna(1.0).to_numpy(dtype=np.float32)
            if len(weight_values) != len(train_frame):
                raise ValueError("Neural regression sample_weight length does not match train_frame.")
            train_weight = torch.tensor(weight_values, dtype=torch.float32, device=device)

        self._module = self._build_module().to(device)
        optimizer = torch.optim.AdamW(
            self._module.parameters(),
            lr=float(self.params.get("learning_rate", 0.001)),
            weight_decay=float(self.params.get("weight_decay", 0.0001)),
        )
        train_loss_fn = torch.nn.MSELoss(reduction="none")
        valid_loss_fn = torch.nn.MSELoss()
        batch_size = max(int(self.params.get("batch_size", 4096)), 1)
        epochs = max(int(self.params.get("epochs", 20)), 1)
        patience = max(int(self.params.get("patience", 5)), 1)
        best_loss = float("inf")
        best_state = None
        stale_epochs = 0

        for _epoch in range(epochs):
            permutation = torch.randperm(train_numeric.shape[0])
            self._module.train()
            for start in range(0, len(permutation), batch_size):
                batch = permutation[start : start + batch_size]
                optimizer.zero_grad()
                prediction = self._module(train_numeric[batch], train_categorical[batch])
                loss_values = train_loss_fn(prediction, train_target[batch])
                if train_weight is not None:
                    batch_weight = train_weight[batch]
                    weight_sum = torch.clamp(batch_weight.sum(), min=1e-6)
                    loss = (loss_values * batch_weight).sum() / weight_sum
                else:
                    loss = loss_values.mean()
                loss.backward()
                optimizer.step()

            self._module.eval()
            with torch.no_grad():
                valid_prediction = self._module(valid_numeric, valid_categorical)
                valid_loss = float(valid_loss_fn(valid_prediction, valid_target).item())
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_state = {key: value.detach().cpu().clone() for key, value in self._module.state_dict().items()}
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    break

        if best_state is not None:
            self._module.load_state_dict(best_state)
            self.state_dict = best_state
        else:
            self.state_dict = {key: value.detach().cpu().clone() for key, value in self._module.state_dict().items()}
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        torch = require_torch()
        device = resolve_torch_device(torch, self.params, key="predict_device", default="cpu")
        if self._module is None:
            self._module = self._build_module()
            if self.state_dict is not None:
                self._module.load_state_dict(self.state_dict, strict=True)
        self._module = self._module.to(device)
        numeric = torch.tensor(self.preprocessor.transform_numeric(frame), dtype=torch.float32, device=device)
        categorical = torch.tensor(self.preprocessor.transform_categorical(frame), dtype=torch.long, device=device)
        self._module.eval()
        with torch.no_grad():
            return self._module(numeric, categorical).detach().cpu().numpy().astype(float)

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_module"] = None
        return state


class TabNetFinishPositionRegressor:
    def __init__(
        self,
        *,
        categorical_columns: list[str],
        params: dict[str, Any],
        random_seed: int,
    ) -> None:
        self.params = dict(params)
        self.random_seed = int(random_seed)
        self.preprocessor = TabularNeuralPreprocessor(categorical_columns)
        self.model: Any | None = None

    def fit(
        self,
        train_frame: pd.DataFrame,
        y_train: pd.Series,
        feature_columns: list[str],
        valid_frame: pd.DataFrame,
        y_valid: pd.Series,
        sample_weight: pd.Series | np.ndarray | None = None,
    ) -> "TabNetFinishPositionRegressor":
        TabNetRegressor = require_tabnet_regressor()
        self.preprocessor.fit(train_frame, feature_columns)
        train_x = self.preprocessor.transform_tabnet(train_frame)
        valid_x = self.preprocessor.transform_tabnet(valid_frame)
        train_y = pd.to_numeric(y_train, errors="coerce").to_numpy(dtype=np.float32).reshape(-1, 1)
        valid_y = pd.to_numeric(y_valid, errors="coerce").to_numpy(dtype=np.float32).reshape(-1, 1)
        numeric_count = len(self.preprocessor.numeric_columns)
        cat_dims = self.preprocessor.categorical_cardinalities()
        cat_idxs = list(range(numeric_count, numeric_count + len(cat_dims)))
        model_params = {
            "n_d": int(self.params.get("n_d", 16)),
            "n_a": int(self.params.get("n_a", 16)),
            "n_steps": int(self.params.get("n_steps", 4)),
            "gamma": float(self.params.get("gamma", 1.5)),
            "lambda_sparse": float(self.params.get("lambda_sparse", 0.0001)),
            "seed": self.random_seed,
            "verbose": 0,
            "cat_idxs": cat_idxs,
            "cat_dims": cat_dims,
            "cat_emb_dim": int(self.params.get("cat_emb_dim", 8)),
            "optimizer_params": {"lr": float(self.params.get("learning_rate", 0.001))},
            "device_name": resolve_tabnet_device_name(self.params),
        }
        self.model = TabNetRegressor(**model_params)
        self.model.fit(
            train_x,
            train_y,
            eval_set=[(valid_x, valid_y)],
            max_epochs=int(self.params.get("max_epochs", 30)),
            patience=int(self.params.get("patience", 5)),
            batch_size=int(self.params.get("batch_size", 4096)),
            virtual_batch_size=int(self.params.get("virtual_batch_size", 512)),
            drop_last=False,
        )
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("TabNet model is not fitted.")
        self._move_model_to_predict_device()
        return np.asarray(self.model.predict(self.preprocessor.transform_tabnet(frame))).reshape(-1).astype(float)

    def _move_model_to_predict_device(self) -> None:
        device_name = resolve_tabnet_device_name(self.params, key="predict_device", default="cpu")
        if self.model is None:
            return
        try:
            torch = require_torch()
            torch_device = torch.device("cuda" if device_name == "cuda" else "cpu")
        except Exception:
            return
        if hasattr(self.model, "device_name"):
            self.model.device_name = device_name
        if hasattr(self.model, "device"):
            self.model.device = torch_device
        network = getattr(self.model, "network", None)
        if network is not None and hasattr(network, "to"):
            network.to(torch_device)


def train_xgboost_ranker(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    config: dict,
    *,
    param_overrides: dict[str, Any] | None = None,
    num_threads: int | None = None,
) -> Any:
    xgb_module = require_xgboost()
    train_sorted = sort_for_grouping(train_df)
    valid_sorted = sort_for_grouping(valid_df)
    train_xgb = build_xgboost_frame(train_sorted, feature_columns, categorical_columns)
    valid_xgb = build_xgboost_frame(valid_sorted, feature_columns, categorical_columns)
    train_group = train_sorted.groupby("race_id", sort=False).size().tolist()
    valid_group = valid_sorted.groupby("race_id", sort=False).size().tolist()
    train_dataset = xgb_module.DMatrix(
        train_xgb,
        label=pd.to_numeric(train_sorted["target_rank"], errors="coerce").astype(float),
    )
    valid_dataset = xgb_module.DMatrix(
        valid_xgb,
        label=pd.to_numeric(valid_sorted["target_rank"], errors="coerce").astype(float),
    )
    train_dataset.set_group(train_group)
    valid_dataset.set_group(valid_group)
    params = {
        "objective": "rank:ndcg",
        "eval_metric": "ndcg@6",
        "eta": config["model"]["learning_rate"],
        "max_depth": 4,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "lambda": 3.0,
        "alpha": 0.0,
        "seed": config["model"]["random_seed"],
        "verbosity": 1,
        "tree_method": "hist",
    }
    if param_overrides:
        params.update(param_overrides)
    if num_threads is not None and int(num_threads) > 0:
        params["nthread"] = int(num_threads)
    try:
        return xgb_module.train(
            params,
            train_dataset,
            num_boost_round=int(config["model"]["iterations"]),
            evals=[(valid_dataset, "valid")],
            verbose_eval=100,
        )
    finally:
        del train_sorted, valid_sorted, train_xgb, valid_xgb, train_dataset, valid_dataset
        collect_garbage()


def train_xgboost_regression(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    config: dict,
    *,
    target: str = "finish_position",
    param_overrides: dict[str, Any] | None = None,
    num_threads: int | None = None,
) -> Any:
    if target not in train_df.columns or target not in valid_df.columns:
        raise ValueError(f"XGBoost regression target column is missing: {target}")
    xgb_module = require_xgboost()
    train_sorted = sort_for_grouping(train_df)
    valid_sorted = sort_for_grouping(valid_df)
    train_xgb = build_xgboost_frame(train_sorted, feature_columns, categorical_columns)
    valid_xgb = build_xgboost_frame(valid_sorted, feature_columns, categorical_columns)
    train_dataset = xgb_module.DMatrix(
        train_xgb,
        label=pd.to_numeric(train_sorted[target], errors="coerce").astype(float),
    )
    valid_dataset = xgb_module.DMatrix(
        valid_xgb,
        label=pd.to_numeric(valid_sorted[target], errors="coerce").astype(float),
    )
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": config["model"]["learning_rate"],
        "max_depth": 3,
        "min_child_weight": 10.0,
        "subsample": 0.75,
        "colsample_bytree": 0.70,
        "lambda": 8.0,
        "alpha": 1.0,
        "gamma": 0.5,
        "seed": config["model"]["random_seed"],
        "verbosity": 1,
        "tree_method": "hist",
    }
    if param_overrides:
        params.update(param_overrides)
    if num_threads is not None and int(num_threads) > 0:
        params["nthread"] = int(num_threads)
    try:
        return xgb_module.train(
            params,
            train_dataset,
            num_boost_round=int(config["model"]["iterations"]),
            evals=[(valid_dataset, "valid")],
            verbose_eval=100,
        )
    finally:
        del train_sorted, valid_sorted, train_xgb, valid_xgb, train_dataset, valid_dataset
        collect_garbage()


def train_xgboost_variants(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    config: dict,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    variants = get_enabled_xgboost_variants(config)
    if not variants:
        return {}
    settings = get_xgboost_variant_settings(config)
    workers = min(int(settings.get("parallel_workers", 1)), len(variants))
    num_threads = int(settings.get("num_threads_per_variant", 2))
    _emit_progress(
        progress_callback,
        f"training xgboost variants: workers={workers}, variants={len(variants)}",
    )

    def train_one(variant: dict[str, Any]) -> tuple[str, Any]:
        name = str(variant["name"])
        _emit_progress(progress_callback, f"training xgboost variant: {name}")
        model = train_xgboost_ranker(
            train_df,
            valid_df,
            feature_columns,
            categorical_columns,
            config,
            param_overrides=dict(variant.get("params", {}) or {}),
            num_threads=num_threads,
        )
        _emit_progress(progress_callback, f"completed xgboost variant: {name}")
        return name, model

    if workers <= 1 or len(variants) <= 1:
        return dict(train_one(variant) for variant in variants)

    trained: dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_name = {executor.submit(train_one, variant): str(variant["name"]) for variant in variants}
        for future in as_completed(future_to_name):
            name, model = future.result()
            trained[name] = model
    return trained


def train_xgboost_regression_variants(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    config: dict,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    variants = get_enabled_xgboost_regression_variants(config)
    if not variants:
        return {}
    settings = get_xgboost_regression_variant_settings(config)
    workers = min(int(settings.get("parallel_workers", 1)), len(variants))
    num_threads = int(settings.get("num_threads_per_variant", 2))
    _emit_progress(
        progress_callback,
        f"training xgboost regression variants: workers={workers}, variants={len(variants)}",
    )

    def train_one(variant: dict[str, Any]) -> tuple[str, Any]:
        name = str(variant["name"])
        _emit_progress(progress_callback, f"training xgboost regression variant: {name}")
        model = train_xgboost_regression(
            train_df,
            valid_df,
            feature_columns,
            categorical_columns,
            config,
            target=str(variant.get("target", "finish_position")),
            param_overrides=dict(variant.get("params", {}) or {}),
            num_threads=num_threads,
        )
        _emit_progress(progress_callback, f"completed xgboost regression variant: {name}")
        return name, model

    if workers <= 1 or len(variants) <= 1:
        return dict(train_one(variant) for variant in variants)

    trained: dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_name = {executor.submit(train_one, variant): str(variant["name"]) for variant in variants}
        for future in as_completed(future_to_name):
            name, model = future.result()
            trained[name] = model
    return trained


def train_random_forest_regression(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    config: dict,
    *,
    target: str = "finish_position",
    param_overrides: dict[str, Any] | None = None,
    num_threads: int | None = None,
) -> RandomForestRegressor:
    if target not in train_df.columns:
        raise ValueError(f"Random forest regression target column is missing: {target}")
    train_sorted = sort_for_grouping(train_df)
    train_frame = build_xgboost_frame(train_sorted, feature_columns, categorical_columns)
    params: dict[str, Any] = {
        "n_estimators": 80,
        "max_depth": 10,
        "min_samples_leaf": 50,
        "max_features": 0.65,
        "bootstrap": True,
        "max_samples": 0.35,
        "random_state": config["model"]["random_seed"],
        "n_jobs": int(num_threads) if num_threads is not None and int(num_threads) > 0 else 1,
    }
    if param_overrides:
        params.update(param_overrides)
    model = RandomForestRegressor(**params)
    try:
        model.fit(
            train_frame,
            pd.to_numeric(train_sorted[target], errors="coerce").astype(float),
        )
        return model
    finally:
        del train_sorted, train_frame
        collect_garbage()


def train_random_forest_regression_variants(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    config: dict,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, RandomForestRegressor]:
    variants = get_enabled_random_forest_regression_variants(config)
    if not variants:
        return {}
    settings = get_random_forest_regression_variant_settings(config)
    workers = min(int(settings.get("parallel_workers", 1)), len(variants))
    num_threads = int(settings.get("num_threads_per_variant", 2))
    _emit_progress(
        progress_callback,
        f"training random forest regression variants: workers={workers}, variants={len(variants)}",
    )

    def train_one(variant: dict[str, Any]) -> tuple[str, RandomForestRegressor]:
        name = str(variant["name"])
        _emit_progress(progress_callback, f"training random forest regression variant: {name}")
        model = train_random_forest_regression(
            train_df,
            valid_df,
            feature_columns,
            categorical_columns,
            config,
            target=str(variant.get("target", "finish_position")),
            param_overrides=dict(variant.get("params", {}) or {}),
            num_threads=num_threads,
        )
        _emit_progress(progress_callback, f"completed random forest regression variant: {name}")
        return name, model

    if workers <= 1 or len(variants) <= 1:
        return dict(train_one(variant) for variant in variants)

    trained: dict[str, RandomForestRegressor] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_name = {executor.submit(train_one, variant): str(variant["name"]) for variant in variants}
        for future in as_completed(future_to_name):
            name, model = future.result()
            trained[name] = model
    return trained


def train_ridge_regression(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    config: dict,
    *,
    target: str = "finish_position",
    param_overrides: dict[str, Any] | None = None,
    num_threads: int | None = None,
) -> Any:
    if target not in train_df.columns:
        raise ValueError(f"Ridge regression target column is missing: {target}")
    train_sorted = sort_for_grouping(train_df)
    train_frame = build_xgboost_frame(train_sorted, feature_columns, categorical_columns)
    params: dict[str, Any] = {
        "alpha": 10.0,
        "random_state": config["model"]["random_seed"],
    }
    if param_overrides:
        params.update(param_overrides)
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(with_mean=False),
        Ridge(**params),
    )
    try:
        model.fit(
            train_frame,
            pd.to_numeric(train_sorted[target], errors="coerce").astype(float),
        )
        return model
    finally:
        del train_sorted, train_frame
        collect_garbage()


def train_ridge_regression_variants(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    config: dict,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    variants = get_enabled_ridge_regression_variants(config)
    if not variants:
        return {}
    settings = get_ridge_regression_variant_settings(config)
    workers = min(int(settings.get("parallel_workers", 1)), len(variants))
    num_threads = int(settings.get("num_threads_per_variant", 1))
    _emit_progress(
        progress_callback,
        f"training ridge regression variants: workers={workers}, variants={len(variants)}",
    )

    def train_one(variant: dict[str, Any]) -> tuple[str, Any]:
        name = str(variant["name"])
        _emit_progress(progress_callback, f"training ridge regression variant: {name}")
        model = train_ridge_regression(
            train_df,
            valid_df,
            feature_columns,
            categorical_columns,
            config,
            target=str(variant.get("target", "finish_position")),
            param_overrides=dict(variant.get("params", {}) or {}),
            num_threads=num_threads,
        )
        _emit_progress(progress_callback, f"completed ridge regression variant: {name}")
        return name, model

    if workers <= 1 or len(variants) <= 1:
        return dict(train_one(variant) for variant in variants)

    trained: dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_name = {executor.submit(train_one, variant): str(variant["name"]) for variant in variants}
        for future in as_completed(future_to_name):
            name, model = future.result()
            trained[name] = model
    return trained


def train_neural_regression(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    config: dict,
    *,
    model_type: str,
    target: str = "finish_position",
    param_overrides: dict[str, Any] | None = None,
    sample_weight_column: str | None = None,
) -> Any:
    if target not in train_df.columns or target not in valid_df.columns:
        raise ValueError(f"Neural regression target column is missing: {target}")
    train_sorted = sort_for_grouping(train_df)
    valid_sorted = sort_for_grouping(valid_df)
    params = dict(param_overrides or {})
    random_seed = int(params.pop("random_seed", config["model"]["random_seed"]))
    categorical_for_model = [column for column in categorical_columns if column in feature_columns]
    if model_type == "mlp":
        model = TorchEmbeddingMLPRegressor(
            categorical_columns=categorical_for_model,
            params=params,
            random_seed=random_seed,
        )
    elif model_type == "tabnet":
        model = TabNetFinishPositionRegressor(
            categorical_columns=categorical_for_model,
            params=params,
            random_seed=random_seed,
        )
    else:
        raise ValueError(f"Unsupported neural regression model_type: {model_type}")
    try:
        sample_weight = (
            pd.to_numeric(train_sorted[sample_weight_column], errors="coerce").fillna(1.0)
            if sample_weight_column and sample_weight_column in train_sorted.columns
            else None
        )
        model.fit(
            train_sorted[feature_columns],
            train_sorted[target],
            feature_columns,
            valid_sorted[feature_columns],
            valid_sorted[target],
            sample_weight=sample_weight,
        )
        return model
    finally:
        del train_sorted, valid_sorted
        collect_garbage()


def train_neural_regression_variants(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    config: dict,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    variants = get_enabled_neural_regression_variants(config)
    if not variants:
        return {}
    settings = get_neural_regression_variant_settings(config)
    workers = min(int(settings.get("parallel_workers", 1)), len(variants))
    _emit_progress(
        progress_callback,
        f"training neural regression variants: workers={workers}, variants={len(variants)}",
    )

    def train_one(variant: dict[str, Any]) -> tuple[str, Any]:
        name = str(variant["name"])
        model_type = str(variant.get("model_type", "mlp"))
        _emit_progress(progress_callback, f"training neural regression variant: {name}, model_type={model_type}")
        variant_train_df = train_df
        sample_weight_column = None
        upset_settings = dict(variant.get("upset_training", {}) or {})
        if bool(upset_settings.get("enabled", False)):
            if model_type != "mlp":
                raise ValueError("upset_training is supported only for neural model_type=mlp.")
            variant_train_df = build_upset_variant_training_frame(
                train_df,
                config,
                upset_settings,
                progress_callback=progress_callback,
            )
            sample_weight_column = UPSET_TRAINING_WEIGHT_COLUMN
        value_recovery_settings = dict(variant.get("value_recovery_training", {}) or {})
        if bool(value_recovery_settings.get("enabled", False)):
            if model_type != "mlp":
                raise ValueError("value_recovery_training is supported only for neural model_type=mlp.")
            variant_train_df = build_value_recovery_variant_training_frame(
                variant_train_df,
                value_recovery_settings,
                base_weight_column=sample_weight_column,
                progress_callback=progress_callback,
            )
            sample_weight_column = VALUE_RECOVERY_TRAINING_WEIGHT_COLUMN
        model = train_neural_regression(
            variant_train_df,
            valid_df,
            feature_columns,
            categorical_columns,
            config,
            model_type=model_type,
            target=str(variant.get("target", "finish_position")),
            param_overrides=dict(variant.get("params", {}) or {}),
            sample_weight_column=sample_weight_column,
        )
        if variant_train_df is not train_df:
            del variant_train_df
            collect_garbage()
        _emit_progress(progress_callback, f"completed neural regression variant: {name}")
        return name, model

    if workers <= 1 or len(variants) <= 1:
        return dict(train_one(variant) for variant in variants)

    trained: dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_name = {executor.submit(train_one, variant): str(variant["name"]) for variant in variants}
        for future in as_completed(future_to_name):
            name, model = future.result()
            trained[name] = model
    return trained


def build_lightgbm_frame(
    df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
) -> pd.DataFrame:
    df = sort_for_grouping(df)
    data = df[feature_columns].copy()
    for column in feature_columns:
        if column in categorical_columns:
            data[column] = data[column].fillna("NA").astype("category")
        else:
            data[column] = pd.to_numeric(data[column], errors="coerce").astype("float32")
    return data


def evaluate_model_bundle(
    model: Any,
    model_type: str,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
) -> dict[str, dict[str, float]]:
    return {
        "train": evaluate_race_order(model, model_type, train_df, feature_columns, categorical_columns),
        "valid": evaluate_race_order(model, model_type, valid_df, feature_columns, categorical_columns),
        "test": evaluate_race_order(model, model_type, test_df, feature_columns, categorical_columns),
    }


def evaluate_ensemble(
    models: dict[str, Any],
    weights: dict[str, float],
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
) -> dict[str, dict[str, float]]:
    return {
        "train": evaluate_ensemble_scores(models, weights, train_df, feature_columns, categorical_columns),
        "valid": evaluate_ensemble_scores(models, weights, valid_df, feature_columns, categorical_columns),
        "test": evaluate_ensemble_scores(models, weights, test_df, feature_columns, categorical_columns),
    }


def evaluate_race_order(
    model: Any,
    model_type: str,
    df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
) -> dict[str, float]:
    if df.empty:
        return {}

    df = apply_prediction_time_measurement_proxies(df)
    scored = score_frame(model, model_type, df, feature_columns, categorical_columns)
    return summarize_rank_metrics(scored)


def evaluate_ensemble_scores(
    models: dict[str, Any],
    weights: dict[str, float],
    df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
) -> dict[str, float]:
    if df.empty:
        return {}

    df = apply_prediction_time_measurement_proxies(df)
    base = sort_for_grouping(df)[["race_id", "lane", "finish_position"]].copy()
    combined = np.zeros(len(base), dtype=float)
    for model_type, model in models.items():
        scored = score_frame(model, model_type, df, feature_columns, categorical_columns)
        combined += weights.get(model_type, 0.0) * scored["score_probability_like"].to_numpy()

    base["score"] = combined
    base["pred_rank"] = base.groupby("race_id")["score"].rank(ascending=False, method="first")
    return summarize_rank_metrics(base)


def score_frame(
    model: Any,
    model_type: str,
    df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
) -> pd.DataFrame:
    df = sort_for_grouping(df)
    if model_type == "catboost":
        pool = build_catboost_pool(df, feature_columns, categorical_columns)
        raw_scores = model.predict(pool)
    elif is_lightgbm_seed_ensemble_model_name(model_type):
        raw_scores = predict_lightgbm_seed_ensemble_raw(model, df, feature_columns, categorical_columns)
    elif is_lightgbm_model_name(model_type):
        model_feature_columns = lightgbm_prediction_feature_columns(model, feature_columns)
        model_categorical_columns = [column for column in categorical_columns if column in model_feature_columns]
        frame = build_lightgbm_frame(df, model_feature_columns, model_categorical_columns)
        raw_scores = model.predict(frame)
        if is_lightgbm_regression_model_name(model_type):
            raw_scores = -np.asarray(raw_scores, dtype=float)
    elif is_xgboost_model_name(model_type):
        xgb_module = require_xgboost()
        frame = build_xgboost_frame(df, feature_columns, categorical_columns)
        raw_scores = model.predict(xgb_module.DMatrix(frame))
        if is_xgboost_regression_model_name(model_type):
            raw_scores = -np.asarray(raw_scores, dtype=float)
    elif is_random_forest_model_name(model_type):
        frame = build_xgboost_frame(df, feature_columns, categorical_columns)
        raw_scores = model.predict(frame)
        if is_random_forest_regression_model_name(model_type):
            raw_scores = -np.asarray(raw_scores, dtype=float)
    elif is_ridge_model_name(model_type):
        frame = build_xgboost_frame(df, feature_columns, categorical_columns)
        raw_scores = model.predict(frame)
        if is_ridge_regression_model_name(model_type):
            raw_scores = -np.asarray(raw_scores, dtype=float)
    elif is_neural_model_name(model_type):
        raw_scores = model.predict(df[feature_columns])
        if is_neural_regression_model_name(model_type):
            raw_scores = -np.asarray(raw_scores, dtype=float)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    scored = df[["race_id", "lane"]].copy()
    if "finish_position" in df.columns:
        scored["finish_position"] = df["finish_position"].to_numpy()
    if "trifecta_payout" in df.columns:
        scored["trifecta_payout"] = df["trifecta_payout"].to_numpy()
    scored["score_raw"] = raw_scores
    scored["score_probability_like"] = scored.groupby("race_id")["score_raw"].transform(_softmax)
    scored["pred_rank"] = scored.groupby("race_id")["score_probability_like"].rank(
        ascending=False,
        method="first",
    )
    scored["score"] = scored["score_probability_like"]
    return scored


def summarize_rank_metrics(scored: pd.DataFrame) -> dict[str, float]:
    if "finish_position" not in scored.columns or scored.empty:
        return {}

    exact_top1 = (
        scored.loc[scored["pred_rank"] == 1, ["race_id", "lane"]]
        .merge(
            scored.loc[scored["finish_position"] == 1, ["race_id", "lane"]],
            on=["race_id", "lane"],
            how="inner",
        )
        .shape[0]
    )
    race_count = scored["race_id"].nunique()

    top3_overlap = []
    for _, race_df in scored.groupby("race_id"):
        pred_top3 = set(race_df.nsmallest(3, "pred_rank")["lane"].tolist())
        actual_top3 = set(race_df.nsmallest(3, "finish_position")["lane"].tolist())
        top3_overlap.append(len(pred_top3 & actual_top3) / 3.0)

    return {
        "race_count": float(race_count),
        "top1_accuracy": exact_top1 / race_count if race_count else 0.0,
        "avg_top3_overlap": float(np.mean(top3_overlap)) if top3_overlap else 0.0,
    }


def save_artifacts(
    models: dict[str, Any],
    feature_columns: list[str],
    metrics: dict[str, Any],
    trifecta_calibrator: IsotonicRegression,
    catboost_model_path: Path,
    lightgbm_model_path: Path,
    features_path: Path,
    ensemble_weights_path: Path,
    trifecta_calibrator_path: Path,
    metrics_path: Path,
    classifier_models: dict[str, lgb.Booster] | None = None,
    classifier_output_dir: Path | None = None,
    flow_model: lgb.Booster | None = None,
    flow_classes: list[str] | None = None,
    flow_model_path: Path | None = None,
    flow_meta_path: Path | None = None,
    staged_models: dict[str, lgb.Booster] | None = None,
    staged_output_dir: Path | None = None,
    trifecta_v2_model: Any | None = None,
    trifecta_v2_model_path: Path | None = None,
    trifecta_v2_calibrator: IsotonicRegression | None = None,
    trifecta_v2_calibrator_path: Path | None = None,
    trifecta_v3_calibrator: IsotonicRegression | None = None,
    trifecta_v3_calibrator_path: Path | None = None,
    xgboost_model_path: Path | None = None,
    random_forest_model_path: Path | None = None,
    ridge_model_path: Path | None = None,
    neural_model_path: Path | None = None,
) -> None:
    catboost_model_path.parent.mkdir(parents=True, exist_ok=True)
    lightgbm_model_path.parent.mkdir(parents=True, exist_ok=True)
    features_path.parent.mkdir(parents=True, exist_ok=True)
    ensemble_weights_path.parent.mkdir(parents=True, exist_ok=True)
    trifecta_calibrator_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    if "catboost" in models:
        models["catboost"].save_model(catboost_model_path)
    models["lightgbm"].save_model(str(lightgbm_model_path))
    for model_name, model in models.items():
        if is_lightgbm_seed_ensemble_model_name(model_name):
            save_lightgbm_seed_ensemble(model, lightgbm_model_path)
            continue
        if model_name in RESERVED_MODEL_NAMES or not is_lightgbm_model_name(model_name):
            continue
        path = lightgbm_variant_model_path(lightgbm_model_path, model_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(path))
    if xgboost_model_path is not None:
        for model_name, model in models.items():
            if not is_xgboost_model_name(model_name):
                continue
            path = xgboost_variant_model_path(xgboost_model_path, model_name)
            path.parent.mkdir(parents=True, exist_ok=True)
            model.save_model(str(path))
    if random_forest_model_path is not None:
        save_random_forest_variants(
            {name: model for name, model in models.items() if is_random_forest_model_name(name)},
            random_forest_model_path,
        )
    if ridge_model_path is not None:
        save_ridge_variants(
            {name: model for name, model in models.items() if is_ridge_model_name(name)},
            ridge_model_path,
        )
    if neural_model_path is not None:
        save_neural_variants(
            {name: model for name, model in models.items() if is_neural_model_name(name)},
            neural_model_path,
        )
    features_path.write_text(json.dumps(feature_columns, ensure_ascii=False, indent=2), encoding="utf-8")
    ensemble_weights_path.write_text(
        json.dumps(metrics["ensemble_weights"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    joblib.dump(trifecta_calibrator, trifecta_calibrator_path)
    if trifecta_v2_calibrator is not None and trifecta_v2_calibrator_path is not None:
        trifecta_v2_calibrator_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(trifecta_v2_calibrator, trifecta_v2_calibrator_path)
    if trifecta_v3_calibrator is not None and trifecta_v3_calibrator_path is not None:
        trifecta_v3_calibrator_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(trifecta_v3_calibrator, trifecta_v3_calibrator_path)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    if classifier_models and classifier_output_dir is not None:
        save_classifier_models(classifier_models, classifier_output_dir)
    if flow_model_path is not None and flow_meta_path is not None:
        save_flow_model(flow_model, flow_classes, flow_model_path, flow_meta_path)
    if staged_models and staged_output_dir is not None:
        save_staged_models(staged_models, staged_output_dir)
    if trifecta_v2_model is not None and trifecta_v2_model_path is not None:
        trifecta_v2_model_path.parent.mkdir(parents=True, exist_ok=True)
        save_trifecta_v2_model_artifact(trifecta_v2_model, trifecta_v2_model_path)


def load_feature_columns(path: Path) -> list[str]:
    return json.loads(path.read_text(encoding="utf-8"))


def predict_race_order(
    models: dict[str, Any],
    feature_columns: list[str],
    future_df: pd.DataFrame,
    ensemble_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    future_df = sort_for_grouping(future_df)
    missing = [column for column in feature_columns if column not in future_df.columns]
    if missing:
        raise ValueError(f"Missing feature columns for prediction: {missing}")

    categorical_columns = infer_categorical_columns(future_df, feature_columns)
    base = future_df.copy()
    combined = np.zeros(len(base), dtype=float)
    weights = ensemble_weights or {name: 1.0 / len(models) for name in models}

    for model_type, model in models.items():
        if model_type == "catboost":
            pool = build_catboost_pool_for_inference(base, feature_columns, categorical_columns)
            raw_scores = model.predict(pool)
        elif is_lightgbm_seed_ensemble_model_name(model_type):
            raw_scores = predict_lightgbm_seed_ensemble_raw(model, base, feature_columns, categorical_columns)
        elif is_lightgbm_model_name(model_type):
            model_feature_columns = lightgbm_prediction_feature_columns(model, feature_columns)
            model_categorical_columns = [column for column in categorical_columns if column in model_feature_columns]
            frame = build_lightgbm_frame(base, model_feature_columns, model_categorical_columns)
            raw_scores = model.predict(frame)
            if is_lightgbm_regression_model_name(model_type):
                raw_scores = -np.asarray(raw_scores, dtype=float)
        elif is_xgboost_model_name(model_type):
            xgb_module = require_xgboost()
            frame = build_xgboost_frame(base, feature_columns, categorical_columns)
            raw_scores = model.predict(xgb_module.DMatrix(frame))
            if is_xgboost_regression_model_name(model_type):
                raw_scores = -np.asarray(raw_scores, dtype=float)
        elif is_random_forest_model_name(model_type):
            frame = build_xgboost_frame(base, feature_columns, categorical_columns)
            raw_scores = model.predict(frame)
            if is_random_forest_regression_model_name(model_type):
                raw_scores = -np.asarray(raw_scores, dtype=float)
        elif is_ridge_model_name(model_type):
            frame = build_xgboost_frame(base, feature_columns, categorical_columns)
            raw_scores = model.predict(frame)
            if is_ridge_regression_model_name(model_type):
                raw_scores = -np.asarray(raw_scores, dtype=float)
        elif is_neural_model_name(model_type):
            raw_scores = model.predict(base[feature_columns])
            if is_neural_regression_model_name(model_type):
                raw_scores = -np.asarray(raw_scores, dtype=float)
        else:
            continue

        score_series = pd.Series(raw_scores, index=base.index)
        combined += weights.get(model_type, 0.0) * score_series.groupby(base["race_id"]).transform(_softmax).to_numpy()

    base["score"] = combined
    base["predicted_rank"] = base.groupby("race_id")["score"].rank(ascending=False, method="first")
    base["win_probability_like"] = base.groupby("race_id")["score"].transform(_softmax)
    return base.sort_values(["race_id", "predicted_rank", "lane"]).reset_index(drop=True)


def predict_trifecta_probabilities(
    models: dict[str, Any],
    feature_columns: list[str],
    future_df: pd.DataFrame,
    ensemble_weights: dict[str, float] | None = None,
    trifecta_calibrator: IsotonicRegression | None = None,
    classifier_models: dict[str, lgb.Booster] | None = None,
    flow_model: lgb.Booster | None = None,
    flow_classes: list[str] | None = None,
    staged_models: dict[str, lgb.Booster] | None = None,
    trifecta_v2_model: Any | None = None,
    odds_df: pd.DataFrame | None = None,
    use_v2: bool = False,
    rerank_top_n: int | None = None,
) -> pd.DataFrame:
    categorical_columns = infer_categorical_columns(future_df, feature_columns)
    ranked = build_weighted_lane_probabilities(
        models,
        ensemble_weights or {name: 1.0 / len(models) for name in models},
        future_df,
        feature_columns,
        categorical_columns,
        classifier_models=classifier_models,
        flow_model=flow_model,
        flow_classes=flow_classes,
        staged_models=staged_models,
    )
    trifecta = build_trifecta_prediction_frame(
        ranked,
        trifecta_calibrator=trifecta_calibrator,
        use_v2=use_v2,
        odds_df=odds_df,
        trifecta_v2_v1_weight=float((ensemble_weights or {}).get("trifecta_v2_v1_weight", 0.9)),
        trifecta_v2_model=trifecta_v2_model,
        rerank_top_n=rerank_top_n,
    )
    return trifecta


def build_trifecta_prediction_frame(
    ranked: pd.DataFrame,
    trifecta_calibrator: IsotonicRegression | None = None,
    use_v2: bool = False,
    odds_df: pd.DataFrame | None = None,
    trifecta_v2_v1_weight: float = 0.9,
    trifecta_v2_model: Any | None = None,
    rerank_top_n: int | None = None,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for _, race_df in ranked.groupby("race_id", sort=False):
        v1 = enumerate_trifecta_probabilities_from_scores(race_df)
        prob_v1 = normalize_trifecta_probabilities(v1["raw_probability"].to_numpy(dtype=float), trifecta_calibrator)
        v1 = v1.rename(columns={"raw_probability": "raw_probability_v1"})
        v1["probability_v1"] = prob_v1
        if not use_v2:
            merged = v1[["race_id", "trifecta", "raw_probability_v1", "probability_v1"]].copy()
            merged["raw_probability_v2"] = merged["raw_probability_v1"]
            merged["probability_v2"] = merged["probability_v1"]
            if "is_actual" in v1.columns:
                merged["is_actual"] = v1["is_actual"].to_numpy()
            merged["probability"] = merged["probability_v1"]
            if "trifecta_payout" in race_df.columns:
                payout_values = pd.to_numeric(race_df["trifecta_payout"], errors="coerce").dropna()
                if not payout_values.empty:
                    merged["trifecta_payout"] = float(payout_values.iloc[0])
            merged = attach_race_upset_and_darkhorse_scores(merged, race_df, scenario_model_bundle=None)
            rows.append(merged)
            continue

        v2 = enumerate_trifecta_probabilities_v2(race_df)
        v2 = v2.rename(columns={"raw_probability": "raw_probability_v2"})
        candidate_mask = None
        if rerank_top_n is not None and rerank_top_n > 0:
            candidate_mask = select_rerank_candidate_mask(
                v1,
                race_df,
                top_n=rerank_top_n,
                scenario_top_n=get_scenario_candidate_top_n(trifecta_v2_model),
            )
            if use_v2:
                v1 = v1.loc[candidate_mask].reset_index(drop=True)
                v2 = v2.loc[candidate_mask].reset_index(drop=True)
        v2["raw_probability_v2"] = blend_trifecta_raw_probabilities(
            v1["raw_probability_v1"].to_numpy(dtype=float),
            v2["raw_probability_v2"].to_numpy(dtype=float),
            trifecta_v2_v1_weight,
        )
        if trifecta_v2_model is not None:
            v2_features = build_trifecta_feature_frame(race_df, v1, v2, scenario_model_bundle=trifecta_v2_model)
            rerank_scores = predict_trifecta_v2_scores(trifecta_v2_model, v2_features)
            if is_trifecta_v2_bundle(trifecta_v2_model) and trifecta_v2_model.get("phase") == "phase3_conditional":
                rerank_scores = apply_phase3_conditional_scores(
                    race_df=race_df,
                    trifecta_df=v2,
                    base_scores=rerank_scores,
                    model_bundle=trifecta_v2_model,
                )
            dynamic_weight, dynamic_subset, dynamic_enabled = get_dynamic_rerank_weight_for_race(
                trifecta_v2_model,
                race_df,
            )
            v2["raw_probability_v2"] = blend_conservative_rerank_scores(
                v1["raw_probability_v1"].to_numpy(dtype=float),
                rerank_scores,
                dynamic_weight,
                get_rank_penalty_strength(trifecta_v2_model),
                get_rank_penalty_start(trifecta_v2_model),
            )
        else:
            dynamic_weight = get_conservative_rerank_weight(trifecta_v2_model)
            dynamic_subset = "disabled"
            dynamic_enabled = False
        prob_v2 = normalize_trifecta_probabilities(v2["raw_probability_v2"].to_numpy(dtype=float), trifecta_calibrator)
        v2["probability_v2"] = prob_v2

        merged = v1[["race_id", "trifecta", "raw_probability_v1", "probability_v1"]].merge(
            v2[["trifecta", "raw_probability_v2", "probability_v2"]],
            on="trifecta",
            how="left",
            validate="one_to_one",
        )
        if "is_actual" in v1.columns:
            merged["is_actual"] = v1["is_actual"].to_numpy()
        if "trifecta_payout" in race_df.columns:
            payout_values = pd.to_numeric(race_df["trifecta_payout"], errors="coerce").dropna()
            if not payout_values.empty:
                merged["trifecta_payout"] = float(payout_values.iloc[0])
        if rerank_top_n is not None and rerank_top_n > 0 and not use_v2:
            merged = merged.loc[candidate_mask].copy().reset_index(drop=True)
            for probability_col in ("probability_v1", "probability_v2"):
                denom = float(merged[probability_col].sum())
                if denom > 0:
                    merged[probability_col] = merged[probability_col] / denom
        merged["probability"] = merged["probability_v2"] if use_v2 else merged["probability_v1"]
        merged = attach_race_upset_and_darkhorse_scores(merged, race_df, scenario_model_bundle=trifecta_v2_model)
        merged["dynamic_rerank_subset"] = dynamic_subset
        merged["dynamic_rerank_weight"] = float(dynamic_weight)
        merged["dynamic_rerank_enabled"] = bool(dynamic_enabled)
        rows.append(merged.sort_values("probability", ascending=False).reset_index(drop=True))

    if not rows:
        return pd.DataFrame(
            columns=[
                "race_id",
                "trifecta",
                "raw_probability_v1",
                "probability_v1",
                "raw_probability_v2",
                "probability_v2",
                "probability",
                "race_upset_score",
                "race_upset_label",
                "race_probability_flatness",
                "race_scenario_id",
                "race_scenario_name",
                "race_scenario_description",
                "scenario_line_fit_score",
                "trifecta_darkhorse_score",
                "is_darkhorse_candidate",
                "ticket_priority_score",
                "ticket_hint",
                "trifecta_payout",
                "dynamic_rerank_subset",
                "dynamic_rerank_weight",
                "dynamic_rerank_enabled",
            ]
        )

    trifecta = pd.concat(rows, ignore_index=True)
    if odds_df is not None:
        trifecta = merge_odds_into_trifecta(trifecta, odds_df)
        trifecta = attach_expected_value_columns(trifecta, probability_col="probability", odds_col="odds")
        trifecta = attach_darkhorse_odds_context(trifecta)
    return trifecta.sort_values(["race_id", "probability"], ascending=[True, False]).reset_index(drop=True)


def attach_race_upset_and_darkhorse_scores(
    trifecta_df: pd.DataFrame,
    race_df: pd.DataFrame,
    scenario_model_bundle: Any | None = None,
) -> pd.DataFrame:
    frame = trifecta_df.copy()
    if frame.empty:
        return frame

    scenario = apply_phase3_pattern_model_to_scenario(
        _phase3_scenario_context(race_df.set_index("lane")),
        race_df,
        scenario_model_bundle,
    )
    scenario_scores = _phase3_scenario_scores(scenario)
    probabilities = pd.to_numeric(frame["probability"], errors="coerce").fillna(0.0)
    ordered_probs = probabilities.sort_values(ascending=False).to_numpy(dtype=float)
    top_prob = float(ordered_probs[0]) if len(ordered_probs) else 0.0
    second_prob = float(ordered_probs[1]) if len(ordered_probs) > 1 else 0.0
    top10_mass = float(ordered_probs[:10].sum()) if len(ordered_probs) else 0.0
    top10_entropy = _normalized_entropy(ordered_probs[:10])

    lane_frame = race_df.set_index("lane")
    lane1_win = _row_numeric(lane_frame.loc[1], "venue_course_prev_win_rate", "venue_lane_prev_win_rate") if 1 in lane_frame.index else 0.0
    lane1_top2 = _row_numeric(lane_frame.loc[1], "venue_course_prev_top2_rate", "venue_lane_prev_top2_rate") if 1 in lane_frame.index else 0.0
    weak_escape = _clip01(1.0 - float(scenario.get("escape_strength", 0.0)))
    collapse = float(scenario.get("inner_collapse_risk", 0.0))
    attack = max(
        float(scenario.get("attack_pressure", 0.0)),
        float(scenario.get("outer_sweep_risk", 0.0)),
        float(scenario.get("s7_chain_pressure", 0.0)),
    )
    scenario_upset = max(
        scenario_scores.get("S3", 0.0),
        scenario_scores.get("S4", 0.0),
        scenario_scores.get("S5", 0.0),
        scenario_scores.get("S6", 0.0),
        scenario_scores.get("S7", 0.0),
    )
    probability_flatness = _clip01(0.55 * top10_entropy + 0.25 * (1.0 - top_prob) + 0.20 * (1.0 - max(top_prob - second_prob, 0.0)))
    lane1_venue_weakness = _clip01(1.0 - max(lane1_win, 0.6 * lane1_top2))
    race_upset_score = _clip01(
        0.24 * weak_escape
        + 0.22 * collapse
        + 0.18 * attack
        + 0.16 * scenario_upset
        + 0.10 * lane1_venue_weakness
        + 0.10 * probability_flatness
    )

    frame["race_upset_score"] = race_upset_score
    frame["race_upset_label"] = label_race_upset(race_upset_score)
    frame["race_probability_flatness"] = probability_flatness
    scenario_id = _phase3_scenario_label(scenario)
    frame["race_scenario_id"] = scenario_id
    frame["race_scenario_name"] = scenario_display_name(scenario_id)
    frame["race_scenario_description"] = scenario_description(scenario_id)

    darkhorse_scores: list[float] = []
    line_fit_scores: list[float] = []
    for trifecta, probability in zip(frame["trifecta"].astype(str), probabilities):
        first, second, third = [int(value) for value in trifecta.split("-")]
        line_features = _phase3_line_features(scenario, first, second, third)
        line_fit = _clip01(
            line_features["makuri_line_fit"]
            + line_features["makurizashi_line_fit"]
            + line_features["outer_follow_fit"]
            + line_features["attack_line_fit"]
            + 0.5 * line_features["sashi_line_fit"]
            - line_features["scenario_mismatch_penalty"]
        )
        outer_mix = _clip01(
            0.34 * float(first >= 4)
            + 0.24 * float(second >= 4)
            + 0.16 * float(third >= 4)
            + 0.16 * float(first != 1)
            + 0.10 * float(1 in (second, third))
        )
        low_model_probability = _clip01(1.0 - float(probability) / max(top_prob, 1e-12))
        darkhorse_score = _clip01(
            0.34 * race_upset_score
            + 0.26 * outer_mix
            + 0.22 * line_fit
            + 0.18 * low_model_probability
        )
        line_fit_scores.append(line_fit)
        darkhorse_scores.append(darkhorse_score)

    frame["scenario_line_fit_score"] = line_fit_scores
    frame["trifecta_darkhorse_score"] = darkhorse_scores
    frame["is_darkhorse_candidate"] = frame["trifecta_darkhorse_score"] >= 0.58
    frame["ticket_priority_score"] = _ticket_priority_score(frame)
    frame["ticket_hint"] = frame.apply(_ticket_hint_from_row, axis=1)
    return frame


def attach_darkhorse_odds_context(trifecta_df: pd.DataFrame) -> pd.DataFrame:
    frame = trifecta_df.copy()
    if frame.empty or "odds" not in frame.columns:
        return frame
    odds = pd.to_numeric(frame["odds"], errors="coerce")
    odds_score = ((odds - 10.0) / 40.0).clip(0.0, 1.0).fillna(0.0)
    expected_value = pd.to_numeric(frame.get("expected_value"), errors="coerce").fillna(0.0)
    value_score = ((expected_value - 0.8) / 0.7).clip(0.0, 1.0)
    base_score = pd.to_numeric(frame.get("trifecta_darkhorse_score"), errors="coerce").fillna(0.0)
    frame["trifecta_darkhorse_score"] = (0.70 * base_score + 0.18 * odds_score + 0.12 * value_score).clip(0.0, 1.0)
    frame["is_darkhorse_candidate"] = frame["trifecta_darkhorse_score"] >= 0.58
    frame["ticket_priority_score"] = _ticket_priority_score(frame)
    frame["ticket_hint"] = frame.apply(_ticket_hint_from_row, axis=1)
    return frame


def label_race_upset(score: float) -> str:
    if score >= 0.72:
        return "波乱"
    if score >= 0.55:
        return "荒れ気配"
    if score >= 0.38:
        return "標準"
    return "堅め"


def _normalized_entropy(values: np.ndarray) -> float:
    probs = np.asarray(values, dtype=float)
    total = float(probs.sum())
    if total <= 0 or len(probs) <= 1:
        return 0.0
    probs = probs / total
    entropy = -float(np.sum(probs * np.log(np.clip(probs, 1e-12, None))))
    return _clip01(entropy / float(np.log(len(probs))))


def _ticket_priority_score(frame: pd.DataFrame) -> pd.Series:
    probability = pd.to_numeric(frame.get("probability"), errors="coerce").fillna(0.0)
    darkhorse = pd.to_numeric(frame.get("trifecta_darkhorse_score"), errors="coerce").fillna(0.0)
    line_fit = pd.to_numeric(frame.get("scenario_line_fit_score"), errors="coerce").fillna(0.0)
    if "expected_value" in frame.columns:
        expected_value = pd.to_numeric(frame["expected_value"], errors="coerce").fillna(0.0)
    else:
        expected_value = pd.Series(0.0, index=frame.index)
    probability_score = probability.groupby(frame["race_id"]).rank(pct=True, ascending=True).fillna(0.0)
    value_score = ((expected_value - 0.8) / 0.7).clip(0.0, 1.0)
    return (0.42 * probability_score + 0.28 * darkhorse + 0.18 * line_fit + 0.12 * value_score).clip(0.0, 1.0)


def _ticket_hint_from_row(row: pd.Series) -> str:
    darkhorse = float(row.get("trifecta_darkhorse_score", 0.0) or 0.0)
    priority = float(row.get("ticket_priority_score", 0.0) or 0.0)
    probability = float(row.get("probability", 0.0) or 0.0)
    if darkhorse >= 0.70 and priority >= 0.45:
        return "妙味穴"
    if darkhorse >= 0.58:
        return "穴候補"
    if probability >= 0.08 or priority >= 0.72:
        return "本線"
    return "押さえ"


def normalize_trifecta_probabilities(
    raw_probabilities: np.ndarray,
    trifecta_calibrator: IsotonicRegression | None = None,
) -> np.ndarray:
    probs = np.asarray(raw_probabilities, dtype=float)
    if trifecta_calibrator is not None:
        probs = trifecta_calibrator.predict(probs)
    prob_sum = probs.sum()
    if prob_sum <= 0:
        return np.full_like(probs, 1.0 / len(probs))
    return probs / prob_sum


def restrict_trifecta_candidates_for_rerank(trifecta_df: pd.DataFrame, top_n: int = 24) -> pd.DataFrame:
    if trifecta_df.empty:
        return trifecta_df.copy()
    ordered = trifecta_df["raw_probability_v1"].rank(ascending=False, method="first") <= int(top_n)
    restricted = trifecta_df.loc[ordered].copy()
    for probability_col in ("probability_v1", "probability_v2"):
        if probability_col not in restricted.columns:
            continue
        denom = float(restricted[probability_col].sum())
        if denom > 0:
            restricted[probability_col] = restricted[probability_col] / denom
    return restricted.reset_index(drop=True)


def select_rerank_candidate_mask_from_v1(v1_df: pd.DataFrame, top_n: int = 24) -> pd.Series:
    if v1_df.empty:
        return pd.Series(dtype=bool)
    ordered = v1_df["raw_probability_v1"].rank(ascending=False, method="first") <= int(top_n)
    return ordered.astype(bool)


def select_rerank_candidate_mask(
    v1_df: pd.DataFrame,
    race_df: pd.DataFrame,
    top_n: int,
    scenario_top_n: int = 0,
) -> pd.Series:
    selected = select_rerank_candidate_mask_from_v1(v1_df, top_n=top_n)
    if scenario_top_n <= 0 or v1_df.empty or race_df.empty:
        return selected

    scenario = _phase3_scenario_context(race_df.set_index("lane"))
    scenario_scores: list[float] = []
    for trifecta in v1_df["trifecta"].astype(str):
        first, second, third = [int(value) for value in trifecta.split("-")]
        features = _phase3_line_features(scenario, first, second, third)
        score = (
            features["escape_line_fit"]
            + features["sashi_line_fit"]
            + features["makuri_line_fit"]
            + features["makurizashi_line_fit"]
            + features["outer_follow_fit"]
            + features["attack_line_fit"]
            - features["scenario_mismatch_penalty"]
        )
        scenario_scores.append(float(score))
    scenario_selected = (
        pd.Series(scenario_scores, index=v1_df.index).rank(ascending=False, method="first")
        <= int(scenario_top_n)
    )
    return (selected | scenario_selected).astype(bool)


def merge_odds_into_trifecta(trifecta_df: pd.DataFrame, odds_df: pd.DataFrame | None) -> pd.DataFrame:
    if odds_df is None or odds_df.empty:
        return trifecta_df.copy()
    if not {"race_id", "trifecta"}.issubset(odds_df.columns):
        return trifecta_df.copy()

    odds_frame = odds_df.copy()
    if "odds" not in odds_frame.columns:
        numeric_candidates = [c for c in odds_frame.columns if c not in {"race_id", "trifecta"}]
        if not numeric_candidates:
            return trifecta_df.copy()
        odds_frame = odds_frame.rename(columns={numeric_candidates[0]: "odds"})

    return trifecta_df.merge(
        odds_frame[["race_id", "trifecta", "odds"]],
        on=["race_id", "trifecta"],
        how="left",
    )


def build_catboost_pool_for_inference(
    df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
) -> Pool:
    df = sort_for_grouping(df)
    data = df[feature_columns].copy()
    for column in feature_columns:
        if column in categorical_columns:
            data[column] = data[column].fillna("NA").astype(str)
        else:
            data[column] = pd.to_numeric(data[column], errors="coerce").astype(float)

    group_id, _ = pd.factorize(df["race_id"])
    cat_indices = [feature_columns.index(column) for column in categorical_columns]
    return Pool(data=data, group_id=group_id, cat_features=cat_indices)


def load_models(config: dict) -> dict[str, Any]:
    artifacts = get_artifact_paths(config)
    lightgbm_model = lgb.Booster(model_file=str(artifacts["lightgbm_model_path"]))
    models: dict[str, Any] = {
        "lightgbm": lightgbm_model,
    }
    if is_catboost_enabled(config):
        catboost_model = CatBoostRanker()
        catboost_model.load_model(str(artifacts["catboost_model_path"]))
        models["catboost"] = catboost_model
    if is_lightgbm_seed_ensemble_enabled(config):
        models[LIGHTGBM_SEED_ENSEMBLE_MODEL_NAME] = load_lightgbm_seed_ensemble(
            config,
            artifacts["lightgbm_model_path"],
            lightgbm_model,
        )
    for variant in get_enabled_lightgbm_variants(config):
        name = str(variant["name"])
        path = lightgbm_variant_model_path(artifacts["lightgbm_model_path"], name)
        if not path.exists():
            raise FileNotFoundError(f"Configured LightGBM variant artifact not found: {path}")
        models[name] = lgb.Booster(model_file=str(path))
    for variant in get_enabled_lightgbm_regression_variants(config):
        name = str(variant["name"])
        path = lightgbm_variant_model_path(artifacts["lightgbm_model_path"], name)
        if not path.exists():
            raise FileNotFoundError(f"Configured LightGBM regression variant artifact not found: {path}")
        models[name] = lgb.Booster(model_file=str(path))
    if get_enabled_xgboost_variants(config):
        xgb_module = require_xgboost()
        for variant in get_enabled_xgboost_variants(config):
            name = str(variant["name"])
            path = xgboost_variant_model_path(artifacts["xgboost_model_path"], name)
            if not path.exists():
                raise FileNotFoundError(f"Configured XGBoost variant artifact not found: {path}")
            model = xgb_module.Booster()
            model.load_model(str(path))
            models[name] = model
    if get_enabled_xgboost_regression_variants(config):
        xgb_module = require_xgboost()
        for variant in get_enabled_xgboost_regression_variants(config):
            name = str(variant["name"])
            path = xgboost_variant_model_path(artifacts["xgboost_model_path"], name)
            if not path.exists():
                raise FileNotFoundError(f"Configured XGBoost regression variant artifact not found: {path}")
            model = xgb_module.Booster()
            model.load_model(str(path))
            models[name] = model
    for variant in get_enabled_random_forest_regression_variants(config):
        name = str(variant["name"])
        path = random_forest_variant_model_path(artifacts["random_forest_model_path"], name)
        if not path.exists():
            raise FileNotFoundError(f"Configured random forest regression variant artifact not found: {path}")
        models[name] = joblib.load(path)
    for variant in get_enabled_ridge_regression_variants(config):
        name = str(variant["name"])
        path = ridge_variant_model_path(artifacts["ridge_model_path"], name)
        if not path.exists():
            raise FileNotFoundError(f"Configured ridge regression variant artifact not found: {path}")
        models[name] = joblib.load(path)
    for variant in get_enabled_neural_regression_variants(config):
        name = str(variant["name"])
        path = neural_variant_model_path(artifacts["neural_model_path"], name)
        if not path.exists():
            raise FileNotFoundError(f"Configured neural regression variant artifact not found: {path}")
        models[name] = joblib.load(path)
    return models


def load_classifier_artifacts(config: dict) -> dict[str, lgb.Booster]:
    artifacts = get_artifact_paths(config)
    return load_classifier_models(artifacts["classifier_dir"])


def load_flow_artifacts(config: dict) -> tuple[lgb.Booster | None, list[str] | None]:
    artifacts = get_artifact_paths(config)
    return load_flow_model(artifacts["flow_model_path"], artifacts["flow_meta_path"])


def load_staged_model_artifacts(config: dict) -> dict[str, lgb.Booster]:
    artifacts = get_artifact_paths(config)
    return load_staged_models(artifacts["staged_dir"])


def load_trifecta_v2_model_artifact(config: dict) -> Any | None:
    artifacts = get_artifact_paths(config)
    path = artifacts["trifecta_v2_model_path"]
    return load_trifecta_v2_model_artifact_payload(path)


def load_ensemble_weights(path: Path) -> dict[str, float]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_probability_adjustment_table(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_trifecta_calibrator(path: Path) -> IsotonicRegression:
    return joblib.load(path)


def load_optional_trifecta_calibrator(path: Path) -> IsotonicRegression | None:
    if not path.exists():
        return None
    return joblib.load(path)


def save_trifecta_calibrator_artifact(calibrator: IsotonicRegression | None, path: Path) -> None:
    if calibrator is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrator, path)


def train_checkpoint_signature(
    config: dict,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> str:
    payload = {
        "config": config,
        "train_races": race_count_from_frame(train_df),
        "valid_races": race_count_from_frame(valid_df),
        "test_races": race_count_from_frame(test_df),
        "train_min_date": _frame_date_bound(train_df, "min"),
        "train_max_date": _frame_date_bound(train_df, "max"),
        "valid_min_date": _frame_date_bound(valid_df, "min"),
        "valid_max_date": _frame_date_bound(valid_df, "max"),
        "test_min_date": _frame_date_bound(test_df, "min"),
        "test_max_date": _frame_date_bound(test_df, "max"),
    }
    raw = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def race_count_from_frame(df: pd.DataFrame) -> int:
    return int(df["race_id"].nunique()) if "race_id" in df.columns and not df.empty else 0


def _frame_date_bound(df: pd.DataFrame, bound: str) -> str | None:
    if df.empty or "race_date" not in df.columns:
        return None
    dates = pd.to_datetime(df["race_date"], errors="coerce").dropna()
    if dates.empty:
        return None
    value = dates.min() if bound == "min" else dates.max()
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def load_train_checkpoint(path: Path, signature: str) -> dict[str, Any]:
    if not path.exists():
        return {"signature": signature, "completed": {}, "metrics": {}}
    try:
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"signature": signature, "completed": {}, "metrics": {}}
    if checkpoint.get("signature") != signature:
        return {"signature": signature, "completed": {}, "metrics": {}}
    checkpoint.setdefault("completed", {})
    checkpoint.setdefault("metrics", {})
    return checkpoint


def save_train_checkpoint(path: Path, checkpoint: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8")


def mark_train_stage_completed(
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    stage: str,
    metrics: dict[str, Any] | None = None,
) -> None:
    checkpoint.setdefault("completed", {})[stage] = True
    if metrics is not None:
        checkpoint.setdefault("metrics", {})[stage] = metrics
    save_train_checkpoint(checkpoint_path, checkpoint)


def train_stage_completed(
    checkpoint: dict[str, Any],
    stage: str,
    required_paths: list[Path] | None = None,
) -> bool:
    if not bool(checkpoint.get("completed", {}).get(stage, False)):
        return False
    return all(path.exists() for path in (required_paths or []))


def save_lightgbm_variants(models: dict[str, lgb.Booster], lightgbm_model_path: Path) -> None:
    for model_name, model in models.items():
        path = lightgbm_variant_model_path(lightgbm_model_path, model_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(path))


def save_lightgbm_seed_ensemble(model_bundle: dict[str, Any], lightgbm_model_path: Path) -> None:
    base_seed = int(model_bundle.get("base_seed", -1))
    for seed_text, model in (model_bundle.get("models", {}) or {}).items():
        seed = int(seed_text)
        if seed == base_seed:
            continue
        path = lightgbm_seed_model_path(lightgbm_model_path, seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(path))


def load_lightgbm_seed_ensemble(
    config: dict,
    lightgbm_model_path: Path,
    base_model: lgb.Booster,
) -> dict[str, Any]:
    settings = get_lightgbm_seed_ensemble_settings(config)
    seeds = [int(seed) for seed in settings.get("seeds", [])]
    base_seed = int(config.get("model", {}).get("random_seed", seeds[0] if seeds else 42))
    models: dict[str, lgb.Booster] = {}
    for seed in seeds:
        if seed == base_seed:
            models[str(seed)] = base_model
            continue
        path = lightgbm_seed_model_path(lightgbm_model_path, seed)
        if not path.exists():
            raise FileNotFoundError(f"Configured LightGBM seed ensemble artifact not found: {path}")
        models[str(seed)] = lgb.Booster(model_file=str(path))
    if len(models) < 2:
        raise ValueError("LightGBM seed ensemble requires at least two available seed models.")
    return {
        "type": LIGHTGBM_SEED_ENSEMBLE_MODEL_NAME,
        "base_seed": base_seed,
        "seeds": seeds,
        "models": models,
    }


def save_xgboost_variants(models: dict[str, Any], xgboost_model_path: Path) -> None:
    for model_name, model in models.items():
        path = xgboost_variant_model_path(xgboost_model_path, model_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(path))


def save_random_forest_variants(models: dict[str, Any], random_forest_model_path: Path) -> None:
    for model_name, model in models.items():
        path = random_forest_variant_model_path(random_forest_model_path, model_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)


def save_ridge_variants(models: dict[str, Any], ridge_model_path: Path) -> None:
    for model_name, model in models.items():
        path = ridge_variant_model_path(ridge_model_path, model_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)


def save_neural_variants(models: dict[str, Any], neural_model_path: Path) -> None:
    for model_name, model in models.items():
        path = neural_variant_model_path(neural_model_path, model_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)


def enabled_lightgbm_variant_paths(config: dict, lightgbm_model_path: Path) -> list[Path]:
    return [
        lightgbm_variant_model_path(lightgbm_model_path, str(variant["name"]))
        for variant in get_enabled_lightgbm_variants(config)
    ]


def enabled_lightgbm_seed_ensemble_paths(config: dict, lightgbm_model_path: Path) -> list[Path]:
    settings = get_lightgbm_seed_ensemble_settings(config)
    if not bool(settings.get("enabled", False)):
        return []
    base_seed = int((config or {}).get("model", {}).get("random_seed", 42))
    return [
        lightgbm_seed_model_path(lightgbm_model_path, int(seed))
        for seed in settings.get("seeds", [])
        if int(seed) != base_seed
    ]


def enabled_lightgbm_regression_variant_paths(config: dict, lightgbm_model_path: Path) -> list[Path]:
    return [
        lightgbm_variant_model_path(lightgbm_model_path, str(variant["name"]))
        for variant in get_enabled_lightgbm_regression_variants(config)
    ]


def enabled_xgboost_variant_paths(config: dict, xgboost_model_path: Path) -> list[Path]:
    return [
        xgboost_variant_model_path(xgboost_model_path, str(variant["name"]))
        for variant in get_enabled_xgboost_variants(config)
    ]


def enabled_xgboost_regression_variant_paths(config: dict, xgboost_model_path: Path) -> list[Path]:
    return [
        xgboost_variant_model_path(xgboost_model_path, str(variant["name"]))
        for variant in get_enabled_xgboost_regression_variants(config)
    ]


def enabled_random_forest_regression_variant_paths(config: dict, random_forest_model_path: Path) -> list[Path]:
    return [
        random_forest_variant_model_path(random_forest_model_path, str(variant["name"]))
        for variant in get_enabled_random_forest_regression_variants(config)
    ]


def enabled_ridge_regression_variant_paths(config: dict, ridge_model_path: Path) -> list[Path]:
    return [
        ridge_variant_model_path(ridge_model_path, str(variant["name"]))
        for variant in get_enabled_ridge_regression_variants(config)
    ]


def enabled_neural_regression_variant_paths(config: dict, neural_model_path: Path) -> list[Path]:
    return [
        neural_variant_model_path(neural_model_path, str(variant["name"]))
        for variant in get_enabled_neural_regression_variants(config)
    ]


def simplex_weight_vectors(
    model_count: int,
    steps: int = 20,
    max_model_weight: float = 1.0,
    min_nonzero_weight: float = 0.0,
) -> list[tuple[float, ...]]:
    if model_count <= 0:
        return []
    if model_count == 1:
        return [(1.0,)]
    max_weight_step = int(np.floor(float(max_model_weight) * float(steps) + 1e-9))
    min_nonzero_step = int(np.ceil(float(min_nonzero_weight) * float(steps) - 1e-9))
    if max_weight_step <= 0:
        return []
    vectors: list[tuple[float, ...]] = []

    def build(prefix: list[int], remaining: int, slots: int) -> None:
        if slots == 1:
            if remaining <= max_weight_step and (remaining == 0 or remaining >= min_nonzero_step):
                vectors.append(tuple([*prefix, remaining]))
            return
        for value in range(min(remaining, max_weight_step) + 1):
            if value > 0 and value < min_nonzero_step:
                continue
            build([*prefix, value], remaining - value, slots - 1)

    build([], int(steps), model_count)
    return [tuple(float(value) / float(steps) for value in vector) for vector in vectors]


def _stable_race_fold(race_id: Any, folds: int) -> int:
    digest = hashlib.blake2b(str(race_id).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) % max(int(folds), 1)


def _split_base_by_race_hash(base: pd.DataFrame, folds: int) -> list[pd.DataFrame]:
    if folds <= 1 or base.empty or "race_id" not in base.columns:
        return [base]
    race_ids = base["race_id"].drop_duplicates().tolist()
    fold_ids = {race_id: _stable_race_fold(race_id, folds) for race_id in race_ids}
    parts: list[pd.DataFrame] = []
    for fold in range(folds):
        selected = {race_id for race_id, fold_id in fold_ids.items() if fold_id == fold}
        if not selected:
            continue
        parts.append(base[base["race_id"].isin(selected)].copy())
    return parts or [base]


def _weighted_average_metrics(metrics_by_fold: list[dict[str, float]]) -> dict[str, float]:
    if not metrics_by_fold:
        return {}
    total_races = sum(float(metrics.get("race_count", 0.0)) for metrics in metrics_by_fold)
    averaged: dict[str, float] = {"race_count": float(total_races)}
    keys = set().union(*(metrics.keys() for metrics in metrics_by_fold))
    for key in keys:
        if key == "race_count":
            continue
        numerator = sum(
            float(metrics.get(key, 0.0)) * float(metrics.get("race_count", 0.0))
            for metrics in metrics_by_fold
        )
        averaged[key] = numerator / total_races if total_races > 0 else 0.0
    return averaged


TRIFECTA_FAST_PERMUTATIONS = np.asarray(list(itertools.permutations(range(6), 3)), dtype=np.int16)
TRIFECTA_FAST_PERMUTATION_INDEX = {
    tuple(int(value) for value in permutation): idx
    for idx, permutation in enumerate(TRIFECTA_FAST_PERMUTATIONS.tolist())
}


def build_fast_trifecta_eval_context(base: pd.DataFrame) -> dict[str, Any]:
    if base.empty or not {"race_id", "lane", "finish_position"}.issubset(base.columns):
        return {"race_count": 0}
    row_groups: list[np.ndarray] = []
    actual_indices: list[int] = []
    trifecta_payouts: list[float] = []
    race_upset_scores: list[float] = []
    for _, race_df in base.groupby("race_id", sort=False):
        if len(race_df) != 6:
            continue
        lanes = pd.to_numeric(race_df["lane"], errors="coerce").astype("Int64")
        finishes = pd.to_numeric(race_df["finish_position"], errors="coerce")
        if lanes.isna().any() or finishes.isna().any():
            continue
        lane_values = [int(value) for value in lanes.tolist()]
        if len(set(lane_values)) != 6:
            continue
        ordered_actual = race_df.assign(_finish=finishes).sort_values("_finish").head(3)
        if len(ordered_actual) < 3:
            continue
        lane_to_position = {lane: position for position, lane in enumerate(lane_values)}
        actual_tuple = tuple(lane_to_position.get(int(lane), -1) for lane in ordered_actual["lane"].tolist())
        actual_index = TRIFECTA_FAST_PERMUTATION_INDEX.get(actual_tuple)
        if actual_index is None:
            continue
        row_groups.append(race_df.index.to_numpy(dtype=np.int64))
        actual_indices.append(int(actual_index))
        if "trifecta_payout" in race_df.columns:
            payout_values = pd.to_numeric(race_df["trifecta_payout"], errors="coerce").dropna()
            trifecta_payouts.append(float(payout_values.iloc[0]) if not payout_values.empty else float("nan"))
        if "race_upset_score" in race_df.columns:
            upset_values = pd.to_numeric(race_df["race_upset_score"], errors="coerce").dropna()
            race_upset_scores.append(float(upset_values.iloc[0]) if not upset_values.empty else 0.0)
    if not row_groups:
        return {"race_count": 0}
    context = {
        "race_count": len(row_groups),
        "row_indices": np.vstack(row_groups),
        "actual_indices": np.asarray(actual_indices, dtype=np.int16),
    }
    if len(trifecta_payouts) == len(actual_indices):
        context["trifecta_payouts"] = np.asarray(trifecta_payouts, dtype=float)
    if len(race_upset_scores) == len(actual_indices):
        context["race_upset_scores"] = np.asarray(race_upset_scores, dtype=float)
    return context


def _softmax_rows(values: np.ndarray) -> np.ndarray:
    shifted = values - np.nanmax(values, axis=1, keepdims=True)
    exps = np.exp(shifted)
    denom = exps.sum(axis=1, keepdims=True)
    return np.divide(exps, denom, out=np.full_like(exps, 1.0 / values.shape[1]), where=denom > 0)


def _fast_trifecta_probs_from_lane_scores(lane_scores: np.ndarray) -> np.ndarray:
    lane_probs = _softmax_rows(lane_scores)
    first = TRIFECTA_FAST_PERMUTATIONS[:, 0]
    second = TRIFECTA_FAST_PERMUTATIONS[:, 1]
    third = TRIFECTA_FAST_PERMUTATIONS[:, 2]
    p1 = lane_probs[:, first]
    p2_base = lane_probs[:, second]
    p3_base = lane_probs[:, third]
    denom2 = np.clip(1.0 - p1, 1e-12, None)
    denom3 = np.clip(1.0 - p1 - p2_base, 1e-12, None)
    return p1 * (p2_base / denom2) * (p3_base / denom3)


def _fit_fast_confidence_calibration(
    raw_probs: np.ndarray,
    actual_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    if raw_probs.size == 0 or len(actual_indices) == 0:
        return None
    labels = np.zeros(raw_probs.shape, dtype=float)
    valid_mask = (actual_indices >= 0) & (actual_indices < raw_probs.shape[1])
    if not bool(valid_mask.any()):
        return None
    labels[np.flatnonzero(valid_mask), actual_indices[valid_mask].astype(int)] = 1.0
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(raw_probs.reshape(-1), labels.reshape(-1))
    return (
        np.asarray(calibrator.X_thresholds_, dtype=float),
        np.asarray(calibrator.y_thresholds_, dtype=float),
    )


def _apply_fast_confidence_calibration(
    raw_probs: np.ndarray,
    calibration: tuple[np.ndarray, np.ndarray] | None,
) -> np.ndarray:
    if calibration is None:
        row_sums = raw_probs.sum(axis=1, keepdims=True)
        return np.divide(
            raw_probs,
            row_sums,
            out=np.full_like(raw_probs, 1.0 / raw_probs.shape[1]),
            where=row_sums > 0,
        )
    x_thresholds, y_thresholds = calibration
    calibrated = np.interp(raw_probs.reshape(-1), x_thresholds, y_thresholds).reshape(raw_probs.shape)
    row_sums = calibrated.sum(axis=1, keepdims=True)
    return np.divide(
        calibrated,
        row_sums,
        out=np.full_like(calibrated, 1.0 / calibrated.shape[1]),
        where=row_sums > 0,
    )


def _top12_confidence_scores_from_sorted_probs(sorted_probs: np.ndarray) -> np.ndarray:
    if sorted_probs.size == 0:
        return np.asarray([], dtype=float)
    top12_end = min(12, sorted_probs.shape[1])
    top5_end = min(5, sorted_probs.shape[1])
    top12_mass = sorted_probs[:, :top12_end].sum(axis=1)
    top5_mass = sorted_probs[:, :top5_end].sum(axis=1)
    top1_top2_gap = sorted_probs[:, 0] - sorted_probs[:, 1] if sorted_probs.shape[1] > 1 else sorted_probs[:, 0]
    top12_margin = (
        sorted_probs[:, 11] - sorted_probs[:, 12]
        if sorted_probs.shape[1] > 12
        else np.zeros(len(sorted_probs), dtype=float)
    )
    clipped = np.clip(sorted_probs, 1e-12, 1.0)
    entropy = -np.sum(clipped * np.log(clipped), axis=1)
    max_entropy = float(np.log(sorted_probs.shape[1])) if sorted_probs.shape[1] > 1 else 1.0
    concentration = 1.0 - np.clip(entropy / max_entropy, 0.0, 1.0)
    return (
        0.45 * np.clip((top12_mass - 0.10) / 0.45, 0.0, 1.0)
        + 0.20 * np.clip((top5_mass - 0.04) / 0.25, 0.0, 1.0)
        + 0.15 * np.clip(top1_top2_gap / 0.06, 0.0, 1.0)
        + 0.10 * np.clip(top12_margin / 0.01, 0.0, 1.0)
        + 0.10 * np.clip(concentration / 0.25, 0.0, 1.0)
    ) * 100.0


def _top3_confidence_scores_from_sorted_probs(
    sorted_probs: np.ndarray,
    race_upset_scores: np.ndarray | None = None,
) -> np.ndarray:
    if sorted_probs.size == 0:
        return np.asarray([], dtype=float)
    top3_end = min(3, sorted_probs.shape[1])
    top3_mass = sorted_probs[:, :top3_end].sum(axis=1)
    top1_probability = sorted_probs[:, 0]
    top1_top2_gap = sorted_probs[:, 0] - sorted_probs[:, 1] if sorted_probs.shape[1] > 1 else sorted_probs[:, 0]
    top3_margin = (
        sorted_probs[:, 2] - sorted_probs[:, 3]
        if sorted_probs.shape[1] > 3
        else np.zeros(len(sorted_probs), dtype=float)
    )
    clipped = np.clip(sorted_probs, 1e-12, 1.0)
    entropy = -np.sum(clipped * np.log(clipped), axis=1)
    max_entropy = float(np.log(sorted_probs.shape[1])) if sorted_probs.shape[1] > 1 else 1.0
    concentration = 1.0 - np.clip(entropy / max_entropy, 0.0, 1.0)
    raw_scores = (
        0.35 * np.clip((top3_mass - 0.03) / 0.16, 0.0, 1.0)
        + 0.20 * np.clip((top1_probability - 0.01) / 0.08, 0.0, 1.0)
        + 0.15 * np.clip(top1_top2_gap / 0.035, 0.0, 1.0)
        + 0.15 * np.clip(top3_margin / 0.012, 0.0, 1.0)
        + 0.15 * np.clip(concentration / 0.25, 0.0, 1.0)
    )
    if race_upset_scores is not None and len(race_upset_scores) == len(sorted_probs):
        upset_scores = np.nan_to_num(np.asarray(race_upset_scores, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        raw_scores = raw_scores - 0.08 * np.clip((upset_scores - 0.75) / 0.25, 0.0, 1.0)
    return np.clip(raw_scores, 0.0, 1.0) * 100.0


def _boat_top1_confidence_scores_from_boat_probs(
    boat_prob_matrix: np.ndarray,
    top3_same_first_boat_rate: np.ndarray | None = None,
    race_upset_scores: np.ndarray | None = None,
) -> np.ndarray:
    if boat_prob_matrix.size == 0:
        return np.asarray([], dtype=float)
    normalized = np.nan_to_num(np.asarray(boat_prob_matrix, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    row_sums = normalized.sum(axis=1, keepdims=True)
    normalized = np.divide(
        normalized,
        row_sums,
        out=np.full_like(normalized, 1.0 / normalized.shape[1]),
        where=row_sums > 0,
    )
    sorted_boat_probs = np.sort(normalized, axis=1)[:, ::-1]
    predicted_probabilities = sorted_boat_probs[:, 0]
    gaps = sorted_boat_probs[:, 0] - sorted_boat_probs[:, 1] if sorted_boat_probs.shape[1] > 1 else sorted_boat_probs[:, 0]
    clipped = np.clip(normalized, 1e-12, 1.0)
    entropy = -np.sum(clipped * np.log(clipped), axis=1)
    max_entropy = float(np.log(normalized.shape[1])) if normalized.shape[1] > 1 else 1.0
    concentration = 1.0 - np.clip(entropy / max_entropy, 0.0, 1.0)
    if top3_same_first_boat_rate is None or len(top3_same_first_boat_rate) != len(normalized):
        same_first_rate = np.zeros(len(normalized), dtype=float)
    else:
        same_first_rate = np.clip(np.nan_to_num(top3_same_first_boat_rate, nan=0.0), 0.0, 1.0)
    raw_scores = (
        0.45 * np.clip((predicted_probabilities - (1.0 / 6.0)) / 0.45, 0.0, 1.0)
        + 0.25 * np.clip(gaps / 0.25, 0.0, 1.0)
        + 0.15 * same_first_rate
        + 0.15 * np.clip(concentration / 0.45, 0.0, 1.0)
    )
    if race_upset_scores is not None and len(race_upset_scores) == len(normalized):
        upset_scores = np.nan_to_num(np.asarray(race_upset_scores, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        raw_scores = raw_scores - 0.08 * np.clip((upset_scores - 0.75) / 0.25, 0.0, 1.0)
    return np.clip(raw_scores, 0.0, 1.0) * 100.0


def _fast_value_rule_metrics(
    actual_rank: np.ndarray,
    sorted_probs: np.ndarray,
    trifecta_payouts: np.ndarray | None,
    settings: dict[str, Any],
    race_upset_scores: np.ndarray | None = None,
    stake_per_ticket: float = 100.0,
) -> dict[str, float]:
    if trifecta_payouts is None or len(trifecta_payouts) != len(actual_rank):
        min_purchase_rate = float(settings.get("min_purchase_rate", 0.10))
        purchase_penalty_weight = float(settings.get("purchase_rate_penalty_weight", 0.20))
        return {
            "value_rule_recovery_rate": 0.0,
            "value_rule_hit_rate": 0.0,
            "value_rule_purchase_rate": 0.0,
            "value_rule_bought_races": 0.0,
            "value_rule_total_stake": 0.0,
            "value_rule_total_return": 0.0,
            "normalized_recovery_score": 0.0,
            "value_rule_coverage_factor": 0.0,
            "purchase_rate_penalty": min_purchase_rate * purchase_penalty_weight,
        }
    valid_payout_mask = np.isfinite(trifecta_payouts) & (trifecta_payouts > 0.0)
    if not bool(valid_payout_mask.any()):
        min_purchase_rate = float(settings.get("min_purchase_rate", 0.10))
        purchase_penalty_weight = float(settings.get("purchase_rate_penalty_weight", 0.20))
        return {
            "value_rule_recovery_rate": 0.0,
            "value_rule_hit_rate": 0.0,
            "value_rule_purchase_rate": 0.0,
            "value_rule_bought_races": 0.0,
            "value_rule_total_stake": 0.0,
            "value_rule_total_return": 0.0,
            "normalized_recovery_score": 0.0,
            "value_rule_coverage_factor": 0.0,
            "purchase_rate_penalty": min_purchase_rate * purchase_penalty_weight,
        }

    valid_actual_rank = actual_rank[valid_payout_mask]
    valid_sorted_probs = sorted_probs[valid_payout_mask]
    valid_payouts = trifecta_payouts[valid_payout_mask].astype(float)
    valid_upset_scores = (
        np.asarray(race_upset_scores, dtype=float)[valid_payout_mask]
        if race_upset_scores is not None and len(race_upset_scores) == len(actual_rank)
        else None
    )
    scores = _top3_confidence_scores_from_sorted_probs(valid_sorted_probs, race_upset_scores=valid_upset_scores)
    labels = np.where(scores >= 75.0, "high", np.where(scores >= 60.0, "middle", "low"))
    rule = dict(settings.get("value_rule", {}) or {})
    ticket_counts = np.zeros(len(valid_actual_rank), dtype=int)
    for label in ("high", "middle", "low"):
        decision = str(rule.get(label, "skip")).strip().lower()
        if decision.startswith("top"):
            try:
                ticket_count = int(decision.removeprefix("top"))
            except ValueError:
                ticket_count = 0
            ticket_counts[labels == label] = max(ticket_count, 0)
    ticket_counts = np.minimum(ticket_counts, sorted_probs.shape[1])
    bought_mask = ticket_counts > 0
    bought_count = int(bought_mask.sum())
    purchase_rate = float(bought_count / len(valid_actual_rank)) if len(valid_actual_rank) else 0.0
    if bought_count == 0:
        recovery_rate = 0.0
        hit_rate = 0.0
        total_stake = 0.0
        total_return = 0.0
    else:
        hits = bought_mask & (valid_actual_rank < ticket_counts)
        total_stake = float(np.sum(ticket_counts.astype(float) * float(stake_per_ticket)))
        total_return = float(np.sum(np.where(hits, valid_payouts, 0.0)))
        recovery_rate = total_return / total_stake if total_stake else 0.0
        hit_rate = float(np.mean(hits[bought_mask]))

    recovery_score_cap = max(float(settings.get("recovery_score_cap", 0.80)), 1e-9)
    min_purchase_rate = float(settings.get("min_purchase_rate", 0.10))
    purchase_penalty_weight = float(settings.get("purchase_rate_penalty_weight", 0.20))
    coverage_factor = float(np.clip(purchase_rate / max(min_purchase_rate, 1e-9), 0.0, 1.0))
    normalized_recovery_score = float(np.clip(recovery_rate / recovery_score_cap, 0.0, 1.0) * coverage_factor)
    purchase_rate_penalty = max(min_purchase_rate - purchase_rate, 0.0) * purchase_penalty_weight
    return {
        "value_rule_recovery_rate": float(recovery_rate),
        "value_rule_hit_rate": float(hit_rate * coverage_factor),
        "value_rule_raw_hit_rate": float(hit_rate),
        "value_rule_purchase_rate": purchase_rate,
        "value_rule_bought_races": float(bought_count),
        "value_rule_total_stake": float(total_stake),
        "value_rule_total_return": float(total_return),
        "normalized_recovery_score": normalized_recovery_score,
        "value_rule_coverage_factor": coverage_factor,
        "purchase_rate_penalty": float(purchase_rate_penalty),
    }


def _fast_top12_payout_capture_metrics(
    actual_rank: np.ndarray,
    trifecta_payouts: np.ndarray | None,
    settings: dict[str, Any],
) -> dict[str, float]:
    if trifecta_payouts is None or len(trifecta_payouts) != len(actual_rank):
        return {
            "top12_payout_capture_mean": 0.0,
            "top12_payout_capture_hit_mean": 0.0,
            "top12_payout_capture_total": 0.0,
            "normalized_top12_payout_capture_score": 0.0,
        }
    valid_mask = np.isfinite(trifecta_payouts) & (trifecta_payouts > 0.0)
    if not bool(valid_mask.any()):
        return {
            "top12_payout_capture_mean": 0.0,
            "top12_payout_capture_hit_mean": 0.0,
            "top12_payout_capture_total": 0.0,
            "normalized_top12_payout_capture_score": 0.0,
        }
    valid_ranks = actual_rank[valid_mask]
    valid_payouts = trifecta_payouts[valid_mask].astype(float)
    top12_hits = valid_ranks < 12
    captured = np.where(top12_hits, valid_payouts, 0.0)
    mean_capture = float(np.mean(captured))
    score_cap = max(float(settings.get("top12_payout_score_cap", 8000.0)), 1e-9)
    return {
        "top12_payout_capture_mean": mean_capture,
        "top12_payout_capture_hit_mean": float(np.mean(valid_payouts[top12_hits])) if bool(top12_hits.any()) else 0.0,
        "top12_payout_capture_total": float(np.sum(captured)),
        "normalized_top12_payout_capture_score": float(np.clip(mean_capture / score_cap, 0.0, 1.0)),
    }


def evaluate_fast_trifecta_ensemble_candidate(
    score_arrays: dict[str, np.ndarray],
    model_names: list[str],
    weight_values: tuple[float, ...],
    context: dict[str, Any],
    settings: dict[str, Any],
) -> tuple[float, dict[str, float], dict[str, float]]:
    row_indices = context["row_indices"]
    actual_indices = context["actual_indices"]
    combined = np.zeros(row_indices.shape, dtype=float)
    candidate_weights = dict(zip(model_names, weight_values, strict=True))
    for model_name, model_weight in candidate_weights.items():
        if model_weight == 0:
            continue
        combined += float(model_weight) * score_arrays[model_name][row_indices]
    raw_trifecta_probs = _fast_trifecta_probs_from_lane_scores(combined)
    calibrated_trifecta_probs = _apply_fast_confidence_calibration(
        raw_trifecta_probs,
        context.get("confidence_calibration"),
    )
    actual_probs = np.clip(
        calibrated_trifecta_probs[np.arange(len(actual_indices)), actual_indices],
        1e-12,
        1.0,
    )

    ranked_indices = np.argsort(-raw_trifecta_probs, axis=1)
    actual_ranks = np.empty_like(ranked_indices)
    actual_ranks[np.arange(len(actual_indices))[:, None], ranked_indices] = np.arange(ranked_indices.shape[1])
    actual_rank = actual_ranks[np.arange(len(actual_indices)), actual_indices]
    best_permutations = TRIFECTA_FAST_PERMUTATIONS[ranked_indices[:, 0]]
    actual_permutations = TRIFECTA_FAST_PERMUTATIONS[actual_indices]
    predicted_top3_members = np.zeros((len(actual_indices), 6), dtype=bool)
    actual_top3_members = np.zeros((len(actual_indices), 6), dtype=bool)
    race_positions = np.arange(len(actual_indices))[:, None]
    predicted_top3_members[race_positions, best_permutations] = True
    actual_top3_members[race_positions, actual_permutations] = True
    top3_overlap = np.logical_and(predicted_top3_members, actual_top3_members).sum(axis=1) / 3.0
    top1_hit = actual_rank == 0
    top3_hit = actual_rank < 3
    top5_hit = actual_rank < 5
    top12_hit = actual_rank < 12
    boat_top1_hit = best_permutations[:, 0] == actual_permutations[:, 0]
    log_loss = -float(np.mean(np.log(actual_probs)))
    top1_hit_rate = float(np.mean(top1_hit))
    top3_hit_rate = float(np.mean(top3_hit))
    top12_hit_rate = float(np.mean(top12_hit))
    top5_hit_rate = float(np.mean(top5_hit))
    boat_top1_hit_rate = float(np.mean(boat_top1_hit))
    avg_top3_overlap = float(np.mean(top3_overlap))
    normalized_log_loss = log_loss / float(np.log(120.0))
    objective_name = str(settings.get("objective", "trifecta_top12_balanced"))
    sorted_probs = np.take_along_axis(calibrated_trifecta_probs, ranked_indices, axis=1)
    value_metrics = _fast_value_rule_metrics(
        actual_rank=actual_rank,
        sorted_probs=sorted_probs,
        trifecta_payouts=context.get("trifecta_payouts"),
        settings=settings,
        race_upset_scores=context.get("race_upset_scores"),
    )
    top12_payout_metrics = _fast_top12_payout_capture_metrics(
        actual_rank=actual_rank,
        trifecta_payouts=context.get("trifecta_payouts"),
        settings=settings,
    )
    if objective_name in {"trifecta_fast", "trifecta_top12_simple"}:
        objective = (
            float(settings.get("objective_top12_weight", 0.60)) * top12_hit_rate
            + float(settings.get("objective_top5_weight", 0.30)) * top5_hit_rate
            - float(settings.get("objective_log_loss_weight", 0.10)) * normalized_log_loss
        )
    elif objective_name == "trifecta_value_balanced":
        objective = (
            float(settings.get("objective_top12_weight", 0.32)) * top12_hit_rate
            + float(settings.get("objective_top5_weight", 0.18)) * top5_hit_rate
            + float(settings.get("objective_top3_weight", 0.28)) * top3_hit_rate
            + float(settings.get("objective_top1_weight", 0.10)) * top1_hit_rate
            + float(settings.get("objective_recovery_weight", 0.08))
            * float(value_metrics.get("normalized_recovery_score", 0.0))
            + float(settings.get("objective_value_hit_weight", 0.02))
            * float(value_metrics.get("value_rule_hit_rate", 0.0))
            + float(settings.get("objective_top12_payout_weight", 0.02))
            * float(top12_payout_metrics.get("normalized_top12_payout_capture_score", 0.0))
            - float(settings.get("objective_log_loss_weight", 0.05)) * normalized_log_loss
            - float(value_metrics.get("purchase_rate_penalty", 0.0))
        )
    elif objective_name == "trifecta_top3_balanced":
        objective = (
            float(settings.get("objective_boat_top1_weight", 0.35)) * boat_top1_hit_rate
            + float(settings.get("objective_top3_weight", 0.45)) * top3_hit_rate
            + float(settings.get("objective_top5_weight", 0.10)) * top5_hit_rate
            + float(settings.get("objective_top12_weight", 0.05)) * top12_hit_rate
            + float(settings.get("objective_top1_weight", 0.05)) * top1_hit_rate
            + float(settings.get("objective_top3_overlap_weight", 0.0)) * avg_top3_overlap
            - float(settings.get("objective_log_loss_weight", 0.0)) * normalized_log_loss
        )
    else:
        objective = (
            float(settings.get("objective_top12_weight", 0.35)) * top12_hit_rate
            + float(settings.get("objective_top5_weight", 0.25)) * top5_hit_rate
            + float(settings.get("objective_top3_weight", 0.15)) * top3_hit_rate
            + float(settings.get("objective_top1_weight", 0.10)) * top1_hit_rate
            + float(settings.get("objective_top3_overlap_weight", 0.10)) * avg_top3_overlap
            - float(settings.get("objective_log_loss_weight", 0.05)) * normalized_log_loss
        )
    metrics = {
        "boat_top1_hit_rate": boat_top1_hit_rate,
        "top1_hit_rate": top1_hit_rate,
        "top3_hit_rate": top3_hit_rate,
        "top12_hit_rate": top12_hit_rate,
        "top5_hit_rate": top5_hit_rate,
        "avg_top3_overlap": avg_top3_overlap,
        "log_loss": log_loss,
        "normalized_log_loss": normalized_log_loss,
        "race_count": float(len(actual_indices)),
        **value_metrics,
        **top12_payout_metrics,
    }
    return float(objective), {name: float(weight) for name, weight in candidate_weights.items()}, metrics


def sample_races_evenly(df: pd.DataFrame, max_races: int) -> pd.DataFrame:
    if df.empty or max_races <= 0 or "race_id" not in df.columns:
        return df.copy()
    races = df[["race_id"]].drop_duplicates().reset_index(drop=True)
    if len(races) <= max_races:
        return df.copy()
    indices = np.linspace(0, len(races) - 1, num=max_races, dtype=int)
    selected_ids = set(races.iloc[np.unique(indices)]["race_id"].tolist())
    return df[df["race_id"].isin(selected_ids)].copy()


def optimize_ensemble_weights(
    models: dict[str, Any],
    valid_df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    config: dict | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, float]:
    settings = get_ensemble_settings(config)
    valid_df = sample_races_evenly(valid_df, int(settings.get("max_eval_races", 0)))
    valid_df = apply_prediction_time_measurement_proxies(valid_df)
    model_names = list(models.keys())
    eval_races = int(valid_df["race_id"].nunique()) if "race_id" in valid_df.columns else len(valid_df)
    objective_name = str(settings.get("objective", "trifecta_fast"))
    grid_step = float(settings.get("grid_step", 0.10))
    grid_steps = max(int(round(1.0 / grid_step)), 1)
    max_model_weight = float(settings.get("max_model_weight", 1.0))
    min_nonzero_weight = float(settings.get("min_nonzero_weight", 0.0))
    candidate_vectors = simplex_weight_vectors(
        len(model_names),
        steps=grid_steps,
        max_model_weight=max_model_weight,
        min_nonzero_weight=min_nonzero_weight,
    )
    if not candidate_vectors:
        raise ValueError(
            "No ensemble weight candidates generated. "
            "Relax models.ensemble.max_model_weight or models.ensemble.min_nonzero_weight."
        )
    cross_validation = dict(settings.get("cross_validation", {}) or {})
    cv_folds_requested = int(cross_validation.get("folds", 1)) if bool(cross_validation.get("enabled", False)) else 1
    workers = min(int(settings.get("parallel_workers", 1)), len(candidate_vectors))
    _emit_progress(
        progress_callback,
        "ensemble weight search: "
        f"models={len(model_names)}, races={eval_races}, candidates={len(candidate_vectors)}, "
        f"workers={workers}, objective={objective_name}, grid_step={grid_step:.4g}, "
        f"max_model_weight={max_model_weight:.4g}, min_nonzero_weight={min_nonzero_weight:.4g}, "
        f"cv_folds={cv_folds_requested}",
    )
    score_by_model = {
        model_name: score_frame(model, model_name, valid_df, feature_columns, categorical_columns)
        for model_name, model in models.items()
    }
    score_arrays = {
        model_name: score_by_model[model_name]["score_probability_like"].to_numpy(dtype=float)
        for model_name in model_names
    }

    best_weights = {model_name: 1.0 / len(model_names) for model_name in model_names}
    best_metrics: dict[str, float] | None = None
    best_objective = float("-inf")
    base_columns = ["race_id", "lane", "finish_position"]
    if "trifecta_payout" in score_by_model[model_names[0]].columns:
        base_columns.append("trifecta_payout")
    if "race_upset_score" in score_by_model[model_names[0]].columns:
        base_columns.append("race_upset_score")
    base = score_by_model[model_names[0]][base_columns].copy()
    fold_bases = _split_base_by_race_hash(base, cv_folds_requested)
    fast_contexts = [build_fast_trifecta_eval_context(fold_base) for fold_base in fold_bases]
    use_fast_trifecta = (
        objective_name in {
            "trifecta_fast",
            "trifecta_top3_balanced",
            "trifecta_top12_balanced",
            "trifecta_top12_simple",
            "trifecta_value_balanced",
        }
        and all(int(fast_context.get("race_count", 0)) > 0 for fast_context in fast_contexts)
    )
    cv_folds_used = len(fold_bases)
    fast_eval_races = sum(int(fast_context.get("race_count", 0)) for fast_context in fast_contexts)
    if (
        objective_name == "trifecta_value_balanced"
        and bool((settings.get("value_confidence_calibration", {}) or {}).get("enabled", True))
        and all(int(fast_context.get("race_count", 0)) > 0 for fast_context in fast_contexts)
    ):
        equal_weight = 1.0 / len(model_names) if model_names else 0.0
        for fast_context in fast_contexts:
            row_indices = fast_context["row_indices"]
            reference_scores = np.zeros(row_indices.shape, dtype=float)
            for model_name in model_names:
                reference_scores += equal_weight * score_arrays[model_name][row_indices]
            reference_raw_probs = _fast_trifecta_probs_from_lane_scores(reference_scores)
            fast_context["confidence_calibration"] = _fit_fast_confidence_calibration(
                reference_raw_probs,
                fast_context["actual_indices"],
            )

    def evaluate_weight_values(weight_values: tuple[float, ...]) -> tuple[float, dict[str, float], dict[str, float]]:
        objectives: list[float] = []
        metrics_by_fold: list[dict[str, float]] = []
        if use_fast_trifecta:
            for fast_context in fast_contexts:
                objective, candidate_weights, metrics = evaluate_fast_trifecta_ensemble_candidate(
                    score_arrays,
                    model_names,
                    weight_values,
                    fast_context,
                    settings,
                )
                objectives.append(float(objective))
                metrics_by_fold.append(metrics)
        else:
            candidate_weights = dict(zip(model_names, weight_values, strict=True))
            for fold_base in fold_bases:
                row_indices = fold_base.index.to_numpy(dtype=np.int64)
                scored = fold_base.copy()
                scored["score"] = np.zeros(len(fold_base), dtype=float)
                for model_name, model_weight in candidate_weights.items():
                    scored["score"] += float(model_weight) * score_arrays[model_name][row_indices]
                scored["pred_rank"] = scored.groupby("race_id")["score"].rank(ascending=False, method="first")
                metrics = summarize_rank_metrics(scored)
                objective = metrics["top1_accuracy"] + 0.1 * metrics["avg_top3_overlap"]
                objectives.append(float(objective))
                metrics_by_fold.append(metrics)
        averaged_metrics = _weighted_average_metrics(metrics_by_fold)
        averaged_objective = float(np.mean(objectives)) if objectives else float("-inf")
        averaged_metrics["cv_fold_count"] = float(cv_folds_used)
        return averaged_objective, {name: float(weight) for name, weight in zip(model_names, weight_values, strict=True)}, averaged_metrics

    if workers <= 1 or len(candidate_vectors) <= 1:
        evaluated = (evaluate_weight_values(weight_values) for weight_values in candidate_vectors)
        for objective, candidate_weights, metrics in evaluated:
            if objective > best_objective:
                best_objective = objective
                best_weights = candidate_weights
                best_metrics = metrics
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(evaluate_weight_values, weight_values) for weight_values in candidate_vectors]
            for future in as_completed(futures):
                objective, candidate_weights, metrics = future.result()
                if objective > best_objective:
                    best_objective = objective
                    best_weights = candidate_weights
                    best_metrics = metrics

    _emit_progress(
        progress_callback,
        f"ensemble weight search completed: objective={best_objective:.6g}",
    )
    best_metrics = best_metrics or {}

    return {
        **best_weights,
        "validation_objective": best_objective,
        "validation_top1_accuracy": float(best_metrics.get("top1_accuracy", best_metrics.get("top1_hit_rate", 0.0))),
        "validation_avg_top3_overlap": float(best_metrics.get("avg_top3_overlap", 0.0)),
        "validation_boat_top1_hit_rate": float(best_metrics.get("boat_top1_hit_rate", 0.0)),
        "validation_top1_hit_rate": float(best_metrics.get("top1_hit_rate", 0.0)),
        "validation_top3_hit_rate": float(best_metrics.get("top3_hit_rate", 0.0)),
        "validation_top5_hit_rate": float(best_metrics.get("top5_hit_rate", 0.0)),
        "validation_top12_hit_rate": float(best_metrics.get("top12_hit_rate", 0.0)),
        "validation_log_loss": float(best_metrics.get("log_loss", 0.0)),
        "validation_normalized_log_loss": float(best_metrics.get("normalized_log_loss", 0.0)),
        "validation_value_rule_recovery_rate": float(best_metrics.get("value_rule_recovery_rate", 0.0)),
        "validation_value_rule_hit_rate": float(best_metrics.get("value_rule_hit_rate", 0.0)),
        "validation_value_rule_purchase_rate": float(best_metrics.get("value_rule_purchase_rate", 0.0)),
        "validation_value_rule_bought_races": float(best_metrics.get("value_rule_bought_races", 0.0)),
        "validation_normalized_recovery_score": float(best_metrics.get("normalized_recovery_score", 0.0)),
        "validation_purchase_rate_penalty": float(best_metrics.get("purchase_rate_penalty", 0.0)),
        "validation_top12_payout_capture_mean": float(best_metrics.get("top12_payout_capture_mean", 0.0)),
        "validation_top12_payout_capture_hit_mean": float(best_metrics.get("top12_payout_capture_hit_mean", 0.0)),
        "validation_top12_payout_capture_total": float(best_metrics.get("top12_payout_capture_total", 0.0)),
        "validation_normalized_top12_payout_capture_score": float(
            best_metrics.get("normalized_top12_payout_capture_score", 0.0)
        ),
        "validation_eval_races": float(eval_races),
        "validation_candidate_count": float(len(candidate_vectors)),
        "validation_parallel_workers": float(workers),
        "validation_grid_step": float(grid_step),
        "validation_max_model_weight": float(max_model_weight),
        "validation_min_nonzero_weight": float(min_nonzero_weight),
        "validation_cv_folds": float(cv_folds_used),
        "validation_objective_name": objective_name,
        "validation_fast_trifecta_races": float(fast_eval_races),
        "value_rule": dict(settings.get("value_rule", {}) or {}),
    }


def fit_trifecta_calibrator(
    models: dict[str, Any],
    weights: dict[str, float],
    valid_df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
) -> IsotonicRegression:
    valid_df = apply_prediction_time_measurement_proxies(valid_df)
    race_probs = build_weighted_lane_probabilities(
        models,
        weights,
        valid_df,
        feature_columns,
        categorical_columns,
    )
    fast_calibrator = fit_trifecta_calibrator_fast_from_ranked(race_probs)
    if fast_calibrator is not None:
        return fast_calibrator
    trifecta = build_trifecta_prediction_frame(race_probs, trifecta_calibrator=None, use_v2=False)
    if trifecta.empty or "is_actual" not in trifecta.columns:
        calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        calibrator.fit(np.asarray([0.0, 1.0]), np.asarray([0.0, 1.0]))
        return calibrator

    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(
        trifecta["raw_probability_v1"].to_numpy(dtype=float),
        trifecta["is_actual"].astype(int).to_numpy(dtype=float),
    )
    return calibrator


def fit_probability_adjustment_table(
    models: dict[str, Any],
    weights: dict[str, float],
    trifecta_calibrator: IsotonicRegression | None,
    valid_df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    classifier_models: dict[str, lgb.Booster] | None = None,
) -> dict[str, Any]:
    calibration_df = apply_prediction_time_measurement_proxies(valid_df)
    if calibration_df.empty:
        return fit_top12_probability_adjustment_table(pd.DataFrame())
    race_probs = build_weighted_lane_probabilities(
        models,
        weights,
        calibration_df,
        feature_columns,
        categorical_columns,
        classifier_models=classifier_models,
    )
    trifecta = build_trifecta_prediction_frame(
        race_probs,
        trifecta_calibrator=trifecta_calibrator,
        use_v2=False,
    )
    return fit_top12_probability_adjustment_table(trifecta, probability_col="probability", actual_col="is_actual")


def fit_model_trifecta_calibrator(
    models: dict[str, Any],
    weights: dict[str, float],
    valid_df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    classifier_models: dict[str, lgb.Booster] | None = None,
    flow_model: lgb.Booster | None = None,
    flow_classes: list[str] | None = None,
    staged_models: dict[str, lgb.Booster] | None = None,
    trifecta_v2_model: Any | None = None,
    use_v2: bool = False,
    rerank_top_n: int | None = None,
    calibration_window_days: int | None = None,
) -> IsotonicRegression:
    calibration_df = valid_df
    if calibration_window_days is not None and not valid_df.empty:
        last_date = pd.to_datetime(valid_df["race_date"]).max()
        window_start = last_date - pd.Timedelta(days=int(calibration_window_days) - 1)
        calibration_df = valid_df[pd.to_datetime(valid_df["race_date"]) >= window_start].copy()
        if calibration_df.empty:
            calibration_df = valid_df
    calibration_df = apply_prediction_time_measurement_proxies(calibration_df)
    race_probs = build_weighted_lane_probabilities(
        models,
        weights,
        calibration_df,
        feature_columns,
        categorical_columns,
        classifier_models=classifier_models,
        flow_model=flow_model,
        flow_classes=flow_classes,
        staged_models=staged_models,
    )
    trifecta = build_trifecta_prediction_frame(
        race_probs,
        trifecta_calibrator=None,
        use_v2=use_v2,
        trifecta_v2_v1_weight=float(weights.get("trifecta_v2_v1_weight", 0.9)),
        trifecta_v2_model=trifecta_v2_model,
        rerank_top_n=rerank_top_n,
    )
    raw_col = "raw_probability_v2" if use_v2 else "raw_probability_v1"
    if trifecta.empty or "is_actual" not in trifecta.columns or raw_col not in trifecta.columns:
        calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        calibrator.fit(np.asarray([0.0, 1.0]), np.asarray([0.0, 1.0]))
        return calibrator
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(
        trifecta[raw_col].to_numpy(dtype=float),
        trifecta["is_actual"].astype(int).to_numpy(dtype=float),
    )
    return calibrator


def single_model_ensemble_weights(model_name: str, config: dict | None = None) -> dict[str, float]:
    weights = {model_name: 1.0}
    weights["scenario_metric_min_races"] = float(
        get_phase3_settings(config)["evaluation"].get("scenario_min_races", 100)
    )
    return weights


def evaluate_trifecta_v1_metrics(
    models: dict[str, Any],
    weights: dict[str, float],
    calibrator: IsotonicRegression | None,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    *,
    classifier_models: dict[str, lgb.Booster] | None = None,
    flow_model: lgb.Booster | None = None,
    flow_classes: list[str] | None = None,
    staged_models: dict[str, lgb.Booster] | None = None,
) -> dict[str, Any]:
    return {
        "valid_raw": evaluate_trifecta(
            models,
            weights,
            None,
            valid_df,
            feature_columns,
            categorical_columns,
            classifier_models=classifier_models,
            flow_model=flow_model,
            flow_classes=flow_classes,
            staged_models=staged_models,
            use_v2=False,
        ),
        "valid_calibrated": evaluate_trifecta(
            models,
            weights,
            calibrator,
            valid_df,
            feature_columns,
            categorical_columns,
            classifier_models=classifier_models,
            flow_model=flow_model,
            flow_classes=flow_classes,
            staged_models=staged_models,
            use_v2=False,
        ),
        "test_raw": evaluate_trifecta(
            models,
            weights,
            None,
            test_df,
            feature_columns,
            categorical_columns,
            classifier_models=classifier_models,
            flow_model=flow_model,
            flow_classes=flow_classes,
            staged_models=staged_models,
            use_v2=False,
        ),
        "test_calibrated": evaluate_trifecta(
            models,
            weights,
            calibrator,
            test_df,
            feature_columns,
            categorical_columns,
            classifier_models=classifier_models,
            flow_model=flow_model,
            flow_classes=flow_classes,
            staged_models=staged_models,
            use_v2=False,
        ),
    }


def evaluate_trifecta_v1_model_metrics(
    models: dict[str, Any],
    ensemble_weights: dict[str, float],
    ensemble_calibrator: IsotonicRegression | None,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    *,
    classifier_models: dict[str, lgb.Booster] | None = None,
    flow_model: lgb.Booster | None = None,
    flow_classes: list[str] | None = None,
    staged_models: dict[str, lgb.Booster] | None = None,
    config: dict | None = None,
    progress_callback: Callable[[str], None] | None = None,
    skip_individual_models: bool = False,
) -> dict[str, Any]:
    settings = get_ensemble_settings(config)
    metrics: dict[str, Any] = {
        "ensemble": evaluate_trifecta_v1_metrics(
            models,
            ensemble_weights,
            ensemble_calibrator,
            valid_df,
            test_df,
            feature_columns,
            categorical_columns,
            classifier_models=classifier_models,
            flow_model=flow_model,
            flow_classes=flow_classes,
            staged_models=staged_models,
        )
    }
    if skip_individual_models:
        _emit_progress(progress_callback, "skipping individual trifecta v1 model metrics: --skip-variant-evaluation")
        return metrics

    model_items = list(models.items())
    workers = min(int(settings.get("model_metrics_parallel_workers", 1)), len(model_items)) if model_items else 1
    _emit_progress(progress_callback, f"trifecta v1 model metrics workers: {workers}")

    def evaluate_one(model_name: str, model: Any) -> tuple[str, dict[str, Any]]:
        _emit_progress(progress_callback, f"evaluating trifecta v1 metrics for model: {model_name}")
        single_models = {model_name: model}
        single_weights = single_model_ensemble_weights(model_name, config)
        single_calibrator = fit_trifecta_calibrator(
            single_models,
            single_weights,
            valid_df,
            feature_columns,
            categorical_columns,
        )
        return (
            model_name,
            evaluate_trifecta_v1_metrics(
                single_models,
                single_weights,
                single_calibrator,
                valid_df,
                test_df,
                feature_columns,
                categorical_columns,
                classifier_models=classifier_models,
                flow_model=flow_model,
                flow_classes=flow_classes,
                staged_models=staged_models,
            ),
        )

    if workers <= 1 or len(model_items) <= 1:
        for model_name, model in model_items:
            name, result = evaluate_one(model_name, model)
            metrics[name] = result
    else:
        parallel_results: dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(evaluate_one, model_name, model): model_name
                for model_name, model in model_items
            }
            for future in as_completed(futures):
                name, result = future.result()
                parallel_results[name] = result
        for model_name, _ in model_items:
            metrics[model_name] = parallel_results[model_name]
    return metrics


def _metric_value(metrics: dict[str, Any], key: str) -> float:
    return float(metrics.get(key, 0.0) or 0.0)


def _rerank_metric_value(metrics: dict[str, Any], key: str) -> float:
    rerank_metrics = metrics.get("rerank_metrics", {})
    if not isinstance(rerank_metrics, dict):
        return 0.0
    return float(rerank_metrics.get(key, 0.0) or 0.0)


def _phase3_tuning_objective(
    metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    settings: dict[str, Any],
) -> tuple[float, float, float, bool]:
    baseline_log_loss = _metric_value(baseline_metrics, "log_loss")
    log_loss = _metric_value(metrics, "log_loss")
    max_delta = float(settings.get("log_loss_max_delta_vs_v1", 0.03))
    allowed_log_loss = baseline_log_loss + max_delta
    log_loss_excess = max(log_loss - allowed_log_loss, 0.0)
    objective = (
        float(settings.get("objective_top12_weight", 0.40)) * _metric_value(metrics, "top12_hit_rate")
        + float(settings.get("objective_top5_weight", 0.15)) * _metric_value(metrics, "top5_hit_rate")
        + float(settings.get("objective_top3_weight", 0.15)) * _metric_value(metrics, "top3_hit_rate")
        + float(settings.get("objective_top1_weight", 0.15)) * _metric_value(metrics, "top1_hit_rate")
        + float(settings.get("objective_mrr_weight", 0.15)) * _rerank_metric_value(metrics, "rerank_mrr")
        - float(settings.get("objective_log_loss_weight", 0.10)) * log_loss
        - float(settings.get("objective_log_loss_excess_penalty", 1.0)) * log_loss_excess
    )
    return float(objective), float(allowed_log_loss), float(log_loss_excess), bool(log_loss_excess <= 0.0)


def _rerank_checkpoint_key(top_n: int, conservative_weight: float, rank_penalty_strength: float) -> str:
    return f"{int(top_n)}|{float(conservative_weight):.12g}|{float(rank_penalty_strength):.12g}"


def _rerank_checkpoint_grid(
    top_n_grid: list[Any],
    weight_grid: list[Any],
    penalty_grid: list[Any],
) -> dict[str, list[float]]:
    return {
        "top_n_grid": [float(value) for value in top_n_grid],
        "weight_grid": [float(value) for value in weight_grid],
        "rank_penalty_strength_grid": [float(value) for value in penalty_grid],
    }


def _empty_rerank_checkpoint(
    *,
    grid: dict[str, list[float]],
    valid_races: int,
    total_count: int,
) -> dict[str, Any]:
    return {
        "version": 1,
        "status": "running",
        "grid": grid,
        "valid_races": int(valid_races),
        "total_count": int(total_count),
        "completed_count": 0,
        "completed": {},
        "best": None,
    }


def _load_rerank_optimization_checkpoint(
    checkpoint_path: Path | None,
    *,
    grid: dict[str, list[float]],
    valid_races: int,
    total_count: int,
) -> dict[str, Any]:
    if checkpoint_path is None or not checkpoint_path.exists():
        return _empty_rerank_checkpoint(grid=grid, valid_races=valid_races, total_count=total_count)
    try:
        state = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _empty_rerank_checkpoint(grid=grid, valid_races=valid_races, total_count=total_count)
    if state.get("version") != 1 or state.get("grid") != grid or int(state.get("valid_races", -1)) != int(valid_races):
        return _empty_rerank_checkpoint(grid=grid, valid_races=valid_races, total_count=total_count)
    completed = state.get("completed")
    if not isinstance(completed, dict):
        state["completed"] = {}
    state["total_count"] = int(total_count)
    state["completed_count"] = len(state["completed"])
    return state


def _save_rerank_optimization_checkpoint(checkpoint_path: Path | None, state: dict[str, Any]) -> None:
    if checkpoint_path is None:
        return
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    state["completed_count"] = len(state.get("completed", {}))
    temp_path = checkpoint_path.with_name(f"{checkpoint_path.name}.tmp")
    temp_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(checkpoint_path)


def _emit_progress(progress_callback: Callable[[str], None] | None, message: str) -> None:
    if progress_callback is not None:
        progress_callback(message)


def _resolve_rerank_optimization_workers(workers: int | None) -> int:
    if workers is None:
        return 1
    if int(workers) <= 0:
        return max((os.cpu_count() or 1) - 1, 1)
    return max(int(workers), 1)


def _rerank_result_with_candidates(
    best: dict[str, Any],
    completed: dict[str, Any],
) -> dict[str, Any]:
    result = dict(best)
    result["ranked_candidates"] = sorted(
        (dict(record) for record in completed.values()),
        key=lambda record: float(record.get("objective", float("-inf"))),
        reverse=True,
    )
    return result


def _sample_races_for_rerank_search(df: pd.DataFrame, max_races: int) -> pd.DataFrame:
    if df.empty or max_races <= 0 or "race_id" not in df.columns:
        return df.copy()
    races = df[["race_id"]].drop_duplicates().reset_index(drop=True)
    if len(races) <= max_races:
        return df.copy()
    indices = np.linspace(0, len(races) - 1, num=max_races, dtype=int)
    selected_ids = set(races.iloc[np.unique(indices)]["race_id"].tolist())
    return df[df["race_id"].isin(selected_ids)].copy()


def _stage_checkpoint_path(checkpoint_path: Path | None, suffix: str) -> Path | None:
    if checkpoint_path is None:
        return None
    return checkpoint_path.with_name(f"{checkpoint_path.stem}_{suffix}{checkpoint_path.suffix}")


def optimize_rerank_inference_settings_two_stage(
    models: dict[str, Any],
    weights: dict[str, float],
    valid_df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    classifier_models: dict[str, lgb.Booster] | None = None,
    flow_model: lgb.Booster | None = None,
    flow_classes: list[str] | None = None,
    staged_models: dict[str, lgb.Booster] | None = None,
    trifecta_v2_model: Any | None = None,
    config: dict | None = None,
    checkpoint_path: Path | None = None,
    progress_callback: Callable[[str], None] | None = None,
    workers: int | None = None,
) -> dict[str, Any]:
    settings = get_phase3_settings(config)["rerank"]
    top_n_grid = list(settings["top_n_grid"])
    weight_grid = list(settings["weight_grid"])
    penalty_grid = list(settings["rank_penalty_strength_grid"])
    fine_top_k = max(int(settings.get("fine_top_k", 5)), 1)
    full_grid_count = len(top_n_grid) * len(weight_grid) * len(penalty_grid)
    if full_grid_count <= fine_top_k:
        _emit_progress(
            progress_callback,
            "rerank direct full search: "
            f"races={int(valid_df['race_id'].nunique()) if 'race_id' in valid_df.columns else len(valid_df)}, "
            f"candidates={full_grid_count}",
        )
        result = optimize_rerank_inference_settings(
            models,
            weights,
            valid_df,
            feature_columns,
            categorical_columns,
            classifier_models=classifier_models,
            flow_model=flow_model,
            flow_classes=flow_classes,
            staged_models=staged_models,
            trifecta_v2_model=trifecta_v2_model,
            top_n_candidates=top_n_grid,
            conservative_weights=weight_grid,
            rank_penalty_strengths=penalty_grid,
            config=config,
            checkpoint_path=checkpoint_path,
            progress_callback=progress_callback,
            workers=workers,
        )
        result = dict(result)
        result["search_mode"] = "direct_full_grid"
        result["coarse_races"] = 0
        result["fine_races"] = int(valid_df["race_id"].nunique()) if "race_id" in valid_df.columns else int(len(valid_df))
        result["full_grid_candidates"] = int(full_grid_count)
        return result

    coarse_df = _sample_races_for_rerank_search(
        valid_df,
        int(settings.get("coarse_eval_max_races", 750)),
    )
    _emit_progress(
        progress_callback,
        "rerank coarse search: "
        f"races={int(coarse_df['race_id'].nunique()) if 'race_id' in coarse_df.columns else len(coarse_df)}",
    )
    coarse = optimize_rerank_inference_settings(
        models,
        weights,
        coarse_df,
        feature_columns,
        categorical_columns,
        classifier_models=classifier_models,
        flow_model=flow_model,
        flow_classes=flow_classes,
        staged_models=staged_models,
        trifecta_v2_model=trifecta_v2_model,
        top_n_candidates=top_n_grid,
        conservative_weights=weight_grid,
        rank_penalty_strengths=list(settings.get("coarse_penalty_grid", [0.0])),
        config=config,
        checkpoint_path=_stage_checkpoint_path(checkpoint_path, "coarse"),
        progress_callback=progress_callback,
        workers=workers,
    )
    shortlisted = list(coarse.get("ranked_candidates", []))[:fine_top_k]
    if not shortlisted:
        return coarse

    grouped_weights: dict[int, list[float]] = {}
    for record in shortlisted:
        top_n = int(record["top_n"])
        weight = float(record["conservative_weight"])
        grouped_weights.setdefault(top_n, [])
        if weight not in grouped_weights[top_n]:
            grouped_weights[top_n].append(weight)

    _emit_progress(
        progress_callback,
        f"rerank fine search: races={int(valid_df['race_id'].nunique())}, shortlisted={len(shortlisted)}",
    )
    fine_results: list[dict[str, Any]] = []
    for top_n, fine_weights in grouped_weights.items():
        fine_results.append(
            optimize_rerank_inference_settings(
                models,
                weights,
                valid_df,
                feature_columns,
                categorical_columns,
                classifier_models=classifier_models,
                flow_model=flow_model,
                flow_classes=flow_classes,
                staged_models=staged_models,
                trifecta_v2_model=trifecta_v2_model,
                top_n_candidates=[top_n],
                conservative_weights=fine_weights,
                rank_penalty_strengths=list(settings["rank_penalty_strength_grid"]),
                config=config,
                checkpoint_path=_stage_checkpoint_path(checkpoint_path, f"fine_{top_n}"),
                progress_callback=progress_callback,
                workers=workers,
            )
        )
    best = max(fine_results, key=lambda result: float(result.get("objective", float("-inf"))))
    result = dict(best)
    result["search_mode"] = "two_stage"
    result["coarse_races"] = int(coarse_df["race_id"].nunique())
    result["fine_races"] = int(valid_df["race_id"].nunique())
    result["coarse_shortlist"] = shortlisted
    return result


def optimize_rerank_inference_settings(
    models: dict[str, Any],
    weights: dict[str, float],
    valid_df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    classifier_models: dict[str, lgb.Booster] | None = None,
    flow_model: lgb.Booster | None = None,
    flow_classes: list[str] | None = None,
    staged_models: dict[str, lgb.Booster] | None = None,
    trifecta_v2_model: Any | None = None,
    top_n_candidates: list[int] | None = None,
    conservative_weights: list[float] | None = None,
    rank_penalty_strengths: list[float] | None = None,
    config: dict | None = None,
    checkpoint_path: Path | None = None,
    progress_callback: Callable[[str], None] | None = None,
    workers: int | None = None,
) -> dict[str, Any]:
    rerank_settings = get_phase3_settings(config)["rerank"]
    if valid_df.empty or trifecta_v2_model is None or not is_trifecta_v2_bundle(trifecta_v2_model):
        default_top_n = get_default_rerank_top_n(config)
        _emit_progress(progress_callback, "rerank optimization skipped")
        return {
            "best_top_n": float(default_top_n),
            "best_conservative_weight": get_conservative_rerank_weight(trifecta_v2_model),
            "best_rank_penalty_strength": get_rank_penalty_strength(trifecta_v2_model),
            "objective": 0.0,
            "status": "skipped",
        }

    top_n_grid = top_n_candidates or list(rerank_settings["top_n_grid"])
    weight_grid = conservative_weights or list(rerank_settings["weight_grid"])
    penalty_grid = rank_penalty_strengths or list(rerank_settings["rank_penalty_strength_grid"])
    total_count = len(top_n_grid) * len(weight_grid) * len(penalty_grid)
    checkpoint_grid = _rerank_checkpoint_grid(top_n_grid, weight_grid, penalty_grid)
    valid_races = int(valid_df["race_id"].nunique()) if "race_id" in valid_df.columns else int(len(valid_df))
    checkpoint = _load_rerank_optimization_checkpoint(
        checkpoint_path,
        grid=checkpoint_grid,
        valid_races=valid_races,
        total_count=total_count,
    )
    if checkpoint.get("status") == "completed" and checkpoint.get("best"):
        _emit_progress(
            progress_callback,
            f"rerank optimization already completed: {checkpoint.get('completed_count', total_count)}/{total_count}",
        )
        return _rerank_result_with_candidates(checkpoint["best"], checkpoint["completed"])

    best: dict[str, float] = dict(checkpoint["best"]) if checkpoint.get("best") else {
        "best_top_n": float(top_n_grid[0]),
        "best_conservative_weight": float(weight_grid[0]),
        "best_rank_penalty_strength": float(penalty_grid[0]),
        "objective": float("-inf"),
    }
    completed: dict[str, Any] = checkpoint["completed"]
    worker_count = _resolve_rerank_optimization_workers(workers)
    _emit_progress(
        progress_callback,
        f"rerank optimization resumed: {len(completed)}/{total_count} completed",
    )
    _emit_progress(progress_callback, f"rerank optimization workers: {worker_count}")
    baseline_by_top_n: dict[int, dict[str, Any]] = {}
    fast_ranked: pd.DataFrame | None = None
    fast_payloads_by_top_n: dict[int, list[FastRerankRacePayload]] = {}
    try:
        fast_ranked = build_fast_rerank_ranked_frame(
            models,
            weights,
            valid_df,
            feature_columns,
            categorical_columns,
            classifier_models=classifier_models,
            flow_model=flow_model,
            flow_classes=flow_classes,
            staged_models=staged_models,
        )
        if fast_ranked is not None:
            _emit_progress(
                progress_callback,
                f"rerank optimization fast evaluator prepared: races={int(fast_ranked['race_id'].nunique())}",
            )
    except Exception as exc:
        fast_ranked = None
        _emit_progress(progress_callback, f"rerank optimization fast evaluator disabled: {type(exc).__name__}: {exc}")

    def get_fast_payloads(top_n_int: int) -> list[FastRerankRacePayload]:
        if fast_ranked is None:
            return []
        if top_n_int not in fast_payloads_by_top_n:
            try:
                fast_payloads_by_top_n[top_n_int] = build_fast_rerank_payloads_from_ranked(
                    fast_ranked,
                    weights,
                    trifecta_v2_model,
                    top_n_int,
                )
                _emit_progress(
                    progress_callback,
                    (
                        f"rerank optimization fast payloads: "
                        f"top_n={top_n_int}, races={len(fast_payloads_by_top_n[top_n_int])}"
                    ),
                )
            except Exception as exc:
                fast_payloads_by_top_n[top_n_int] = []
                _emit_progress(
                    progress_callback,
                    f"rerank optimization fast payloads disabled top_n={top_n_int}: {type(exc).__name__}: {exc}",
                )
        return fast_payloads_by_top_n[top_n_int]

    def evaluate_candidate(
        top_n_int: int,
        conservative_weight: float,
        rank_penalty_strength: float,
    ) -> tuple[str, dict[str, Any], dict[str, float]]:
        checkpoint_key = _rerank_checkpoint_key(top_n_int, conservative_weight, rank_penalty_strength)
        fast_payloads = get_fast_payloads(top_n_int)
        if fast_payloads:
            metrics = _evaluate_fast_rerank_payloads(
                fast_payloads,
                conservative_weight=conservative_weight,
                rank_penalty_strength=rank_penalty_strength,
                use_v2=True,
            )
        else:
            candidate_model = with_conservative_rerank_weight(trifecta_v2_model, conservative_weight)
            candidate_model = with_rank_penalty_settings(
                candidate_model,
                rank_penalty_strength,
                get_rank_penalty_start(trifecta_v2_model),
            )
            calibrator = fit_model_trifecta_calibrator(
                models,
                weights,
                valid_df,
                feature_columns,
                categorical_columns,
                classifier_models=classifier_models,
                flow_model=flow_model,
                flow_classes=flow_classes,
                staged_models=staged_models,
                trifecta_v2_model=candidate_model,
                use_v2=True,
                rerank_top_n=top_n_int,
            )
            metrics = evaluate_trifecta(
                models,
                weights,
                calibrator,
                valid_df,
                feature_columns,
                categorical_columns,
                classifier_models=classifier_models,
                flow_model=flow_model,
                flow_classes=flow_classes,
                staged_models=staged_models,
                trifecta_v2_model=candidate_model,
                use_v2=True,
                rerank_top_n=top_n_int,
            )
        baseline_metrics = baseline_by_top_n[top_n_int]
        objective, allowed_log_loss, log_loss_excess, within_guard = _phase3_tuning_objective(
            metrics,
            baseline_metrics,
            rerank_settings,
        )
        completed_record = {
            "top_n": float(top_n_int),
            "conservative_weight": float(conservative_weight),
            "rank_penalty_strength": float(rank_penalty_strength),
            "objective": float(objective),
            "v1_log_loss": float(baseline_metrics.get("log_loss", 0.0)),
            "allowed_log_loss": float(allowed_log_loss),
            "log_loss_excess": float(log_loss_excess),
            "within_log_loss_guard": bool(within_guard),
            "top1_hit_rate": float(metrics.get("top1_hit_rate", 0.0)),
            "top3_hit_rate": float(metrics.get("top3_hit_rate", 0.0)),
            "top5_hit_rate": float(metrics.get("top5_hit_rate", 0.0)),
            "top10_hit_rate": float(metrics.get("top10_hit_rate", 0.0)),
            "top12_hit_rate": float(metrics.get("top12_hit_rate", 0.0)),
            "log_loss": float(metrics.get("log_loss", 0.0)),
            "rerank_mrr": _rerank_metric_value(metrics, "rerank_mrr"),
        }
        best_record = {
            "best_top_n": float(top_n_int),
            "best_conservative_weight": float(conservative_weight),
            "best_rank_penalty_strength": float(rank_penalty_strength),
            "objective": float(objective),
            "v1_log_loss": float(baseline_metrics.get("log_loss", 0.0)),
            "allowed_log_loss": float(allowed_log_loss),
            "log_loss_excess": float(log_loss_excess),
            "within_log_loss_guard": float(within_guard),
            "top1_hit_rate": float(metrics.get("top1_hit_rate", 0.0)),
            "top3_hit_rate": float(metrics.get("top3_hit_rate", 0.0)),
            "top5_hit_rate": float(metrics.get("top5_hit_rate", 0.0)),
            "top10_hit_rate": float(metrics.get("top10_hit_rate", 0.0)),
            "top12_hit_rate": float(metrics.get("top12_hit_rate", 0.0)),
            "log_loss": float(metrics.get("log_loss", 0.0)),
            "rerank_mrr": _rerank_metric_value(metrics, "rerank_mrr"),
        }
        return checkpoint_key, completed_record, best_record

    def save_candidate_result(checkpoint_key: str, completed_record: dict[str, Any], best_record: dict[str, float]) -> None:
        nonlocal best
        completed[checkpoint_key] = completed_record
        objective = float(completed_record["objective"])
        if objective > best["objective"]:
            best = best_record
        checkpoint["best"] = best
        checkpoint["status"] = "completed" if len(completed) >= total_count else "running"
        _save_rerank_optimization_checkpoint(checkpoint_path, checkpoint)
        _emit_progress(
            progress_callback,
            (
                f"rerank optimization saved {len(completed)}/{total_count}: "
                f"objective={objective:.6g}, best={float(best['objective']):.6g}"
            ),
        )

    for top_n in top_n_grid:
        top_n_int = int(top_n)
        top_n_keys = [
            _rerank_checkpoint_key(top_n_int, conservative_weight, rank_penalty_strength)
            for conservative_weight in weight_grid
            for rank_penalty_strength in penalty_grid
        ]
        if all(key in completed for key in top_n_keys):
            _emit_progress(progress_callback, f"rerank optimization skip top_n={top_n_int}: already completed")
            continue
        _emit_progress(progress_callback, f"rerank optimization baseline top_n={top_n_int}")
        fast_payloads = get_fast_payloads(top_n_int)
        if fast_payloads:
            baseline_by_top_n[top_n_int] = _evaluate_fast_rerank_payloads(fast_payloads, use_v2=False)
        else:
            baseline_calibrator = fit_model_trifecta_calibrator(
                models,
                weights,
                valid_df,
                feature_columns,
                categorical_columns,
                classifier_models=classifier_models,
                flow_model=flow_model,
                flow_classes=flow_classes,
                staged_models=staged_models,
                trifecta_v2_model=trifecta_v2_model,
                use_v2=False,
                rerank_top_n=top_n_int,
            )
            baseline_by_top_n[top_n_int] = evaluate_trifecta(
                models,
                weights,
                baseline_calibrator,
                valid_df,
                feature_columns,
                categorical_columns,
                classifier_models=classifier_models,
                flow_model=flow_model,
                flow_classes=flow_classes,
                staged_models=staged_models,
                trifecta_v2_model=trifecta_v2_model,
                use_v2=False,
                rerank_top_n=top_n_int,
            )
        pending_candidates = [
            (float(conservative_weight), float(rank_penalty_strength))
            for conservative_weight in weight_grid
            for rank_penalty_strength in penalty_grid
            if _rerank_checkpoint_key(top_n_int, conservative_weight, rank_penalty_strength) not in completed
        ]
        if worker_count <= 1 or len(pending_candidates) <= 1:
            for conservative_weight, rank_penalty_strength in pending_candidates:
                _emit_progress(
                    progress_callback,
                    (
                        f"rerank optimization {len(completed) + 1}/{total_count}: "
                        f"top_n={top_n_int}, weight={float(conservative_weight):.4g}, "
                        f"penalty={float(rank_penalty_strength):.4g}"
                    ),
                )
                checkpoint_key, completed_record, best_record = evaluate_candidate(
                    top_n_int,
                    conservative_weight,
                    rank_penalty_strength,
                )
                save_candidate_result(checkpoint_key, completed_record, best_record)
        else:
            top_n_workers = min(worker_count, len(pending_candidates))
            _emit_progress(
                progress_callback,
                f"rerank optimization parallel top_n={top_n_int}: workers={top_n_workers}, candidates={len(pending_candidates)}",
            )
            with ThreadPoolExecutor(max_workers=top_n_workers) as executor:
                future_to_candidate = {
                    executor.submit(evaluate_candidate, top_n_int, conservative_weight, rank_penalty_strength): (
                        conservative_weight,
                        rank_penalty_strength,
                    )
                    for conservative_weight, rank_penalty_strength in pending_candidates
                }
                for future in as_completed(future_to_candidate):
                    conservative_weight, rank_penalty_strength = future_to_candidate[future]
                    _emit_progress(
                        progress_callback,
                        (
                            f"rerank optimization completed candidate: "
                            f"top_n={top_n_int}, weight={float(conservative_weight):.4g}, "
                            f"penalty={float(rank_penalty_strength):.4g}"
                        ),
                    )
                    checkpoint_key, completed_record, best_record = future.result()
                    save_candidate_result(checkpoint_key, completed_record, best_record)
    checkpoint["best"] = best
    checkpoint["status"] = "completed"
    _save_rerank_optimization_checkpoint(checkpoint_path, checkpoint)
    _emit_progress(
        progress_callback,
        (
            "rerank optimization completed: "
            f"best_top_n={int(best['best_top_n'])}, "
            f"best_weight={float(best['best_conservative_weight']):.4g}, "
            f"best_penalty={float(best['best_rank_penalty_strength']):.4g}, "
            f"objective={float(best['objective']):.6g}"
        ),
    )
    return _rerank_result_with_candidates(best, completed)


def _build_raw_calibration_trifecta_frame(
    ranked: pd.DataFrame,
    weights: dict[str, float],
    trifecta_v2_model: Any | None,
    rerank_top_n: int | None,
    source_df: pd.DataFrame,
) -> pd.DataFrame:
    trifecta = build_trifecta_prediction_frame(
        ranked,
        trifecta_calibrator=None,
        use_v2=True,
        trifecta_v2_v1_weight=float(weights.get("trifecta_v2_v1_weight", 0.9)),
        trifecta_v2_model=trifecta_v2_model,
        rerank_top_n=rerank_top_n,
    )
    if trifecta.empty or "race_date" in trifecta.columns:
        return trifecta
    race_dates = source_df[["race_id", "race_date"]].drop_duplicates().copy()
    race_dates["race_id"] = race_dates["race_id"].astype(str)
    trifecta["race_id"] = trifecta["race_id"].astype(str)
    return trifecta.merge(race_dates, on="race_id", how="left")


def _optimize_phase3_calibration_window_fast(
    models: dict[str, Any],
    weights: dict[str, float],
    calibration_source: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    classifier_models: dict[str, lgb.Booster] | None = None,
    flow_model: lgb.Booster | None = None,
    flow_classes: list[str] | None = None,
    staged_models: dict[str, lgb.Booster] | None = None,
    trifecta_v2_model: Any | None = None,
    rerank_top_n: int | None = None,
    window_days_options: list[int] | None = None,
    phase3_settings: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if not models or calibration_source.empty or eval_df.empty:
        return None
    phase3_settings = phase3_settings or DEFAULT_PHASE3_SETTINGS
    try:
        calibration_ranked = build_fast_rerank_ranked_frame(
            models,
            weights,
            calibration_source,
            feature_columns,
            categorical_columns,
            classifier_models=classifier_models,
            flow_model=flow_model,
            flow_classes=flow_classes,
            staged_models=staged_models,
        )
        eval_ranked = build_fast_rerank_ranked_frame(
            models,
            weights,
            eval_df,
            feature_columns,
            categorical_columns,
            classifier_models=classifier_models,
            flow_model=flow_model,
            flow_classes=flow_classes,
            staged_models=staged_models,
        )
        if calibration_ranked is None or eval_ranked is None:
            return None
        calibration_trifecta = _build_raw_calibration_trifecta_frame(
            calibration_ranked,
            weights,
            trifecta_v2_model,
            rerank_top_n,
            calibration_source,
        )
        eval_trifecta = _build_raw_calibration_trifecta_frame(
            eval_ranked,
            weights,
            trifecta_v2_model,
            rerank_top_n,
            eval_df,
        )
    except Exception:
        return None
    required_columns = {"race_id", "race_date", "raw_probability_v1", "raw_probability_v2", "is_actual"}
    if (
        calibration_trifecta.empty
        or eval_trifecta.empty
        or not required_columns.issubset(calibration_trifecta.columns)
        or not required_columns.issubset(eval_trifecta.columns)
    ):
        return None

    baseline_calibrator = _fit_isotonic_from_raw(
        calibration_trifecta["raw_probability_v1"].to_numpy(dtype=float),
        calibration_trifecta["is_actual"].astype(int).to_numpy(dtype=float),
    )
    baseline_metrics = _fast_calibrated_metrics_from_frame(
        eval_trifecta,
        raw_col="raw_probability_v1",
        calibrator=baseline_calibrator,
    )
    if not baseline_metrics:
        return None

    calibration_dates = pd.to_datetime(calibration_trifecta["race_date"], errors="coerce")
    if calibration_dates.isna().all():
        return None
    last_date = calibration_dates.max()
    best = {
        "best_window_days": float(phase3_settings["calibration"]["default_window_days"]),
        "objective": float("-inf"),
    }
    options = window_days_options or list(phase3_settings["calibration"]["window_days_options"])
    for window_days in options:
        window_start = last_date - pd.Timedelta(days=int(window_days) - 1)
        window_mask = calibration_dates >= window_start
        if not bool(window_mask.any()):
            window_mask = pd.Series(True, index=calibration_trifecta.index)
        train_frame = calibration_trifecta.loc[window_mask].copy()
        calibrator = _fit_isotonic_from_raw(
            train_frame["raw_probability_v2"].to_numpy(dtype=float),
            train_frame["is_actual"].astype(int).to_numpy(dtype=float),
        )
        metrics = _fast_calibrated_metrics_from_frame(
            eval_trifecta,
            raw_col="raw_probability_v2",
            calibrator=calibrator,
            baseline_raw_col="raw_probability_v1",
        )
        objective, allowed_log_loss, log_loss_excess, within_guard = _phase3_tuning_objective(
            metrics,
            baseline_metrics,
            phase3_settings["rerank"],
        )
        if objective > best["objective"]:
            best = {
                "best_window_days": float(window_days),
                "objective": float(objective),
                "v1_log_loss": float(baseline_metrics.get("log_loss", 0.0)),
                "allowed_log_loss": float(allowed_log_loss),
                "log_loss_excess": float(log_loss_excess),
                "within_log_loss_guard": float(within_guard),
                "top1_hit_rate": float(metrics.get("top1_hit_rate", 0.0)),
                "top3_hit_rate": float(metrics.get("top3_hit_rate", 0.0)),
                "top5_hit_rate": float(metrics.get("top5_hit_rate", 0.0)),
                "top12_hit_rate": float(metrics.get("top12_hit_rate", 0.0)),
                "log_loss": float(metrics.get("log_loss", 0.0)),
                "rerank_mrr": _rerank_metric_value(metrics, "rerank_mrr"),
                "evaluation_mode": "fast_numpy",
            }
    return best


def optimize_phase3_calibration_window(
    models: dict[str, Any],
    weights: dict[str, float],
    valid_df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    classifier_models: dict[str, lgb.Booster] | None = None,
    flow_model: lgb.Booster | None = None,
    flow_classes: list[str] | None = None,
    staged_models: dict[str, lgb.Booster] | None = None,
    trifecta_v2_model: Any | None = None,
    rerank_top_n: int | None = None,
    window_days_options: list[int] | None = None,
    config: dict | None = None,
) -> dict[str, Any]:
    phase3_settings = get_phase3_settings(config)
    if valid_df.empty:
        return {"best_window_days": float(phase3_settings["calibration"]["default_window_days"]), "objective": 0.0}
    unique_dates = sorted(pd.to_datetime(valid_df["race_date"]).dropna().dt.normalize().unique().tolist())
    if len(unique_dates) < 4:
        default_window = int(phase3_settings["calibration"]["default_window_days"])
        return {"best_window_days": float(default_window), "objective": 0.0}
    split_index = max(int(len(unique_dates) * 0.7), 1)
    split_date = pd.Timestamp(unique_dates[min(split_index - 1, len(unique_dates) - 1)])
    calibration_source = valid_df[pd.to_datetime(valid_df["race_date"]) <= split_date].copy()
    eval_df = valid_df[pd.to_datetime(valid_df["race_date"]) > split_date].copy()
    if calibration_source.empty or eval_df.empty:
        default_window = int(phase3_settings["calibration"]["default_window_days"])
        return {"best_window_days": float(default_window), "objective": 0.0}
    fast_result = _optimize_phase3_calibration_window_fast(
        models,
        weights,
        calibration_source,
        eval_df,
        feature_columns,
        categorical_columns,
        classifier_models=classifier_models,
        flow_model=flow_model,
        flow_classes=flow_classes,
        staged_models=staged_models,
        trifecta_v2_model=trifecta_v2_model,
        rerank_top_n=rerank_top_n,
        window_days_options=window_days_options,
        phase3_settings=phase3_settings,
    )
    if fast_result is not None:
        return fast_result
    baseline_calibrator = fit_model_trifecta_calibrator(
        models,
        weights,
        calibration_source,
        feature_columns,
        categorical_columns,
        classifier_models=classifier_models,
        flow_model=flow_model,
        flow_classes=flow_classes,
        staged_models=staged_models,
        trifecta_v2_model=trifecta_v2_model,
        use_v2=False,
        rerank_top_n=rerank_top_n,
    )
    baseline_metrics = evaluate_trifecta(
        models,
        weights,
        baseline_calibrator,
        eval_df,
        feature_columns,
        categorical_columns,
        classifier_models=classifier_models,
        flow_model=flow_model,
        flow_classes=flow_classes,
        staged_models=staged_models,
        trifecta_v2_model=trifecta_v2_model,
        use_v2=False,
        rerank_top_n=rerank_top_n,
    )
    best = {
        "best_window_days": float(phase3_settings["calibration"]["default_window_days"]),
        "objective": float("-inf"),
    }
    for window_days in (window_days_options or list(phase3_settings["calibration"]["window_days_options"])):
        calibrator = fit_model_trifecta_calibrator(
            models,
            weights,
            calibration_source,
            feature_columns,
            categorical_columns,
            classifier_models=classifier_models,
            flow_model=flow_model,
            flow_classes=flow_classes,
            staged_models=staged_models,
            trifecta_v2_model=trifecta_v2_model,
            use_v2=True,
            rerank_top_n=rerank_top_n,
            calibration_window_days=int(window_days),
        )
        metrics = evaluate_trifecta(
            models,
            weights,
            calibrator,
            eval_df,
            feature_columns,
            categorical_columns,
            classifier_models=classifier_models,
            flow_model=flow_model,
            flow_classes=flow_classes,
            staged_models=staged_models,
            trifecta_v2_model=trifecta_v2_model,
            use_v2=True,
            rerank_top_n=rerank_top_n,
        )
        objective, allowed_log_loss, log_loss_excess, within_guard = _phase3_tuning_objective(
            metrics,
            baseline_metrics,
            phase3_settings["rerank"],
        )
        if objective > best["objective"]:
            best = {
                "best_window_days": float(window_days),
                "objective": float(objective),
                "v1_log_loss": float(baseline_metrics.get("log_loss", 0.0)),
                "allowed_log_loss": float(allowed_log_loss),
                "log_loss_excess": float(log_loss_excess),
                "within_log_loss_guard": float(within_guard),
                "top1_hit_rate": float(metrics.get("top1_hit_rate", 0.0)),
                "top3_hit_rate": float(metrics.get("top3_hit_rate", 0.0)),
                "top5_hit_rate": float(metrics.get("top5_hit_rate", 0.0)),
                "top12_hit_rate": float(metrics.get("top12_hit_rate", 0.0)),
                "log_loss": float(metrics.get("log_loss", 0.0)),
                "rerank_mrr": _rerank_metric_value(metrics, "rerank_mrr"),
            }
    return best


def evaluate_trifecta(
    models: dict[str, Any],
    weights: dict[str, float],
    calibrator: IsotonicRegression | None,
    df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    classifier_models: dict[str, lgb.Booster] | None = None,
    flow_model: lgb.Booster | None = None,
    flow_classes: list[str] | None = None,
    staged_models: dict[str, lgb.Booster] | None = None,
    trifecta_v2_model: Any | None = None,
    odds_df: pd.DataFrame | None = None,
    use_v2: bool = False,
    rerank_top_n: int | None = None,
) -> dict[str, float | dict[str, float]]:
    if df.empty:
        return {}

    df = apply_prediction_time_measurement_proxies(df)
    race_probs = build_weighted_lane_probabilities(
        models,
        weights,
        df,
        feature_columns,
        categorical_columns,
        classifier_models=classifier_models,
        flow_model=flow_model,
        flow_classes=flow_classes,
        staged_models=staged_models,
    )
    if not use_v2 and rerank_top_n is None and odds_df is None:
        fast_metrics = _evaluate_fast_v1_trifecta_metrics_from_ranked(race_probs, calibrator, weights)
        if fast_metrics is not None:
            return fast_metrics
    trifecta = build_trifecta_prediction_frame(
        race_probs,
        trifecta_calibrator=calibrator,
        use_v2=use_v2,
        odds_df=odds_df,
        trifecta_v2_v1_weight=float(weights.get("trifecta_v2_v1_weight", 0.9)),
        trifecta_v2_model=trifecta_v2_model,
        rerank_top_n=rerank_top_n,
    )
    probability_col = "probability_v2" if use_v2 else "probability_v1"
    trifecta["probability"] = trifecta[probability_col]
    if odds_df is not None and {"odds", "expected_value"}.issubset(trifecta.columns):
        expected_value = pd.to_numeric(trifecta["expected_value"], errors="coerce").fillna(0.0)
        odds = pd.to_numeric(trifecta["odds"], errors="coerce").fillna(0.0)
        trifecta["buy_decision"] = np.where(
            (expected_value >= BUY_EXPECTED_VALUE_THRESHOLD) & (odds >= BUY_MIN_ODDS),
            "買い",
            "見送り",
        )
    metrics = compute_trifecta_metrics(trifecta, probability_col="probability")
    if rerank_top_n is not None:
        metrics["rerank_metrics"] = compute_trifecta_rerank_metrics(
            trifecta,
            probability_col="probability",
            baseline_col="probability_v1",
        )
    entry_course_subset_by_race = _entry_course_subset_by_race(race_probs)
    entry_course_subset_metrics: dict[str, Any] = {}
    for subset_label in ("lane_course_match", "lane_course_mismatch", "course_unknown"):
        race_ids = {
            race_id for race_id, label in entry_course_subset_by_race.items() if label == subset_label
        }
        if not race_ids:
            continue
        subset_frame = trifecta[trifecta["race_id"].astype(str).isin(race_ids)].copy()
        subset_result = compute_trifecta_metrics(subset_frame, probability_col="probability")
        if rerank_top_n is not None:
            subset_result["rerank_metrics"] = compute_trifecta_rerank_metrics(
                subset_frame,
                probability_col="probability",
                baseline_col="probability_v1",
            )
        entry_course_subset_metrics[subset_label] = subset_result
    metrics["entry_course_subset_metrics"] = entry_course_subset_metrics
    scenario_by_race = {
        str(race_id): _phase3_scenario_label(_phase3_scenario_context(race_df.set_index("lane")))
        for race_id, race_df in race_probs.groupby("race_id", sort=False)
    }
    scenario_min_races = int(weights.get("scenario_metric_min_races", DEFAULT_PHASE3_SETTINGS["evaluation"]["scenario_min_races"]))
    scenario_metrics: dict[str, Any] = {}
    scenario_groups: dict[str, set[str]] = {}
    for scenario_label in sorted(set(scenario_by_race.values())):
        race_ids = {
            race_id for race_id, label in scenario_by_race.items() if label == scenario_label
        }
        scenario_frame = trifecta[trifecta["race_id"].astype(str).isin(race_ids)].copy()
        scenario_result = compute_trifecta_metrics(scenario_frame, probability_col="probability")
        if rerank_top_n is not None:
            scenario_result["rerank_metrics"] = compute_trifecta_rerank_metrics(
                scenario_frame,
                probability_col="probability",
                baseline_col="probability_v1",
            )
        scenario_result["scenario_id"] = float(scenario_numeric_id(scenario_label))
        scenario_result["scenario_min_races"] = float(scenario_min_races)
        scenario_result["is_small_sample"] = float(len(race_ids) < scenario_min_races)
        scenario_metrics[scenario_label] = scenario_result
        grouped_label = scenario_label if len(race_ids) >= scenario_min_races else "__small_sample__"
        scenario_groups.setdefault(grouped_label, set()).update(race_ids)
    metrics["scenario_metrics"] = scenario_metrics
    grouped_metrics: dict[str, Any] = {}
    for scenario_label, race_ids in sorted(scenario_groups.items()):
        scenario_frame = trifecta[trifecta["race_id"].astype(str).isin(race_ids)].copy()
        scenario_result = compute_trifecta_metrics(scenario_frame, probability_col="probability")
        if rerank_top_n is not None:
            scenario_result["rerank_metrics"] = compute_trifecta_rerank_metrics(
                scenario_frame,
                probability_col="probability",
                baseline_col="probability_v1",
            )
        scenario_result["scenario_min_races"] = float(scenario_min_races)
        scenario_result["is_small_sample_group"] = float(scenario_label == "__small_sample__")
        grouped_metrics[scenario_label] = scenario_result
    metrics["scenario_metrics_grouped"] = grouped_metrics
    return metrics


def optimize_dynamic_rerank_weights(
    models: dict[str, Any],
    weights: dict[str, float],
    valid_df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    classifier_models: dict[str, lgb.Booster] | None = None,
    flow_model: lgb.Booster | None = None,
    flow_classes: list[str] | None = None,
    staged_models: dict[str, lgb.Booster] | None = None,
    trifecta_v2_model: Any | None = None,
    rerank_top_n: int | None = None,
    config: dict | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    default_weight = get_conservative_rerank_weight(trifecta_v2_model)
    settings = get_phase3_settings(config)["dynamic_rerank_weight"]
    metadata = build_disabled_dynamic_rerank_weight_metadata(config, default_weight)
    if not bool(settings.get("optimize", False)):
        return metadata
    if valid_df.empty or trifecta_v2_model is None or not is_trifecta_v2_bundle(trifecta_v2_model):
        return metadata

    _emit_progress(progress_callback, "dynamic rerank weight: building subset profile")
    eval_df = apply_prediction_time_measurement_proxies(valid_df)
    ranked = build_weighted_lane_probabilities(
        models,
        weights,
        eval_df,
        feature_columns,
        categorical_columns,
        classifier_models=classifier_models,
        flow_model=flow_model,
        flow_classes=flow_classes,
        staged_models=staged_models,
    )
    subset_frame = build_dynamic_rerank_subset_frame(ranked, trifecta_v2_model, settings)
    if subset_frame.empty:
        return metadata

    thresholds = dict(subset_frame.attrs.get("dynamic_rerank_thresholds", {}))
    metadata["thresholds"] = thresholds
    subset_by_race = subset_frame.set_index("race_id")["dynamic_rerank_subset"].to_dict()
    subset_counts = subset_frame["dynamic_rerank_subset"].value_counts().to_dict()
    min_subset_races = int(settings.get("min_subset_races", 500))
    candidate_weights = [float(value) for value in settings.get("weight_grid", [default_weight])]
    if default_weight not in candidate_weights:
        candidate_weights.append(float(default_weight))
    candidate_weights = sorted(set(candidate_weights))
    top12_min_improvement = float(settings.get("top12_min_improvement", 0.0))
    log_loss_max_delta = float(settings.get("log_loss_max_delta", 0.03))
    fast_payload_by_race: dict[str, FastRerankRacePayload] = {}
    if rerank_top_n is not None and rerank_top_n > 0:
        try:
            fast_payloads = build_fast_rerank_payloads_from_ranked(
                ranked,
                weights,
                trifecta_v2_model,
                int(rerank_top_n),
            )
            fast_payload_by_race = {payload.race_id: payload for payload in fast_payloads}
            _emit_progress(
                progress_callback,
                f"dynamic rerank weight: fast evaluator payloads={len(fast_payload_by_race)}",
            )
        except Exception as exc:
            fast_payload_by_race = {}
            _emit_progress(
                progress_callback,
                f"dynamic rerank weight: fast evaluator disabled: {type(exc).__name__}: {exc}",
            )

    default_model = with_dynamic_rerank_weight_metadata(
        with_conservative_rerank_weight(trifecta_v2_model, default_weight),
        {"enabled": False, "rules": [], "default_weight": default_weight},
    )
    diagnostics: dict[str, Any] = {
        "subset_counts": {str(key): int(value) for key, value in subset_counts.items()},
        "thresholds": thresholds,
        "candidate_weights": [float(value) for value in candidate_weights],
        "min_subset_races": int(min_subset_races),
        "top12_min_improvement": float(top12_min_improvement),
        "log_loss_max_delta": float(log_loss_max_delta),
        "rules": {},
    }
    rules: list[dict[str, Any]] = []

    for subset_name in ("stable_escape", "attack_or_collapse", "upset_or_flat", "neutral"):
        race_ids = [race_id for race_id, subset in subset_by_race.items() if subset == subset_name]
        subset_race_count = len(race_ids)
        if subset_race_count < min_subset_races:
            diagnostics["rules"][subset_name] = {
                "race_count": int(subset_race_count),
                "status": "skipped_small_subset",
                "candidate_results": [],
            }
            continue
        subset_df = valid_df[valid_df["race_id"].astype(str).isin(race_ids)].copy()
        subset_payloads = [fast_payload_by_race[race_id] for race_id in race_ids if race_id in fast_payload_by_race]
        if subset_payloads:
            baseline_metrics = _evaluate_fast_rerank_payloads(
                subset_payloads,
                conservative_weight=default_weight,
                rank_penalty_strength=get_rank_penalty_strength(trifecta_v2_model),
                use_v2=True,
            )
        else:
            baseline_metrics = evaluate_trifecta(
                models,
                weights,
                None,
                subset_df,
                feature_columns,
                categorical_columns,
                classifier_models=classifier_models,
                flow_model=flow_model,
                flow_classes=flow_classes,
                staged_models=staged_models,
                trifecta_v2_model=default_model,
                use_v2=True,
                rerank_top_n=rerank_top_n,
            )
        baseline_top12 = float(baseline_metrics.get("top12_hit_rate", 0.0))
        baseline_log_loss = float(baseline_metrics.get("log_loss", 0.0))
        best_weight = float(default_weight)
        best_metrics = dict(baseline_metrics)
        candidate_results: list[dict[str, Any]] = []
        for candidate_weight in candidate_weights:
            if np.isclose(candidate_weight, default_weight):
                candidate_metrics = dict(baseline_metrics)
            elif subset_payloads:
                candidate_metrics = _evaluate_fast_rerank_payloads(
                    subset_payloads,
                    conservative_weight=candidate_weight,
                    rank_penalty_strength=get_rank_penalty_strength(trifecta_v2_model),
                    use_v2=True,
                )
            else:
                candidate_model = with_dynamic_rerank_weight_metadata(
                    with_conservative_rerank_weight(trifecta_v2_model, candidate_weight),
                    {"enabled": False, "rules": [], "default_weight": candidate_weight},
                )
                candidate_metrics = evaluate_trifecta(
                    models,
                    weights,
                    None,
                    subset_df,
                    feature_columns,
                    categorical_columns,
                    classifier_models=classifier_models,
                    flow_model=flow_model,
                    flow_classes=flow_classes,
                    staged_models=staged_models,
                    trifecta_v2_model=candidate_model,
                    use_v2=True,
                    rerank_top_n=rerank_top_n,
                )
            candidate_results.append(
                _dynamic_rerank_candidate_diagnostic(
                    subset_name,
                    candidate_weight,
                    candidate_metrics,
                    baseline_top12,
                    baseline_log_loss,
                    log_loss_max_delta,
                    default_weight,
                )
            )
            if (
                float(candidate_metrics.get("top12_hit_rate", 0.0)) > float(best_metrics.get("top12_hit_rate", 0.0))
                and float(candidate_metrics.get("log_loss", 0.0)) <= baseline_log_loss + log_loss_max_delta
            ):
                best_weight = float(candidate_weight)
                best_metrics = dict(candidate_metrics)
        top12_improvement = float(best_metrics.get("top12_hit_rate", 0.0)) - baseline_top12
        accepted = top12_improvement >= top12_min_improvement and float(best_metrics.get("log_loss", 0.0)) <= (
            baseline_log_loss + log_loss_max_delta
        )
        diagnostics["rules"][subset_name] = {
            "race_count": int(subset_race_count),
            "status": "evaluated",
            "baseline_weight": float(default_weight),
            "best_weight": float(best_weight),
            "accepted": bool(accepted),
            "baseline_top12_hit_rate": baseline_top12,
            "best_top12_hit_rate": float(best_metrics.get("top12_hit_rate", 0.0)),
            "top12_improvement": top12_improvement,
            "baseline_log_loss": baseline_log_loss,
            "best_log_loss": float(best_metrics.get("log_loss", 0.0)),
            "best_log_loss_delta": float(best_metrics.get("log_loss", 0.0)) - baseline_log_loss,
            "candidate_results": candidate_results,
        }
        if accepted:
            rules.append(
                {
                    "subset": subset_name,
                    "weight": float(best_weight),
                    "race_count": int(subset_race_count),
                    "top12_hit_rate": float(best_metrics.get("top12_hit_rate", 0.0)),
                    "top12_improvement": top12_improvement,
                    "log_loss": float(best_metrics.get("log_loss", 0.0)),
                }
            )

    metadata.update(
        {
            "enabled": bool(settings.get("enabled", False)),
            "optimized": True,
            "default_weight": float(default_weight),
            "rules": rules,
            "diagnostics": diagnostics,
        }
    )
    _emit_progress(
        progress_callback,
        f"dynamic rerank weight: optimized rules={len(rules)}, enabled={bool(metadata['enabled'])}",
    )
    return metadata


def build_weighted_lane_probabilities(
    models: dict[str, Any],
    weights: dict[str, float],
    df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    classifier_models: dict[str, lgb.Booster] | None = None,
    flow_model: lgb.Booster | None = None,
    flow_classes: list[str] | None = None,
    staged_models: dict[str, lgb.Booster] | None = None,
) -> pd.DataFrame:
    ranked = predict_race_order(models, feature_columns, df, weights)
    ranked["lane_probability"] = ranked["win_probability_like"]

    if classifier_models:
        ranked = predict_classifier_probabilities(
            classifier_models,
            ranked,
            feature_columns,
            categorical_columns,
        )
    if flow_model is not None and flow_classes:
        ranked = predict_flow_probabilities(
            flow_model,
            flow_classes,
            ranked,
            feature_columns,
            categorical_columns,
        )
    if staged_models:
        ranked = predict_staged_probabilities(
            staged_models,
            ranked,
            feature_columns,
            categorical_columns,
        )
    return ranked


def enumerate_trifecta_probabilities(race_df: pd.DataFrame) -> pd.DataFrame:
    return enumerate_trifecta_probabilities_from_column(race_df, probability_column="lane_probability")


def enumerate_trifecta_probabilities_from_scores(race_df: pd.DataFrame) -> pd.DataFrame:
    return enumerate_trifecta_probabilities_from_column(race_df, probability_column="win_probability_like")


def enumerate_trifecta_probabilities_from_column(
    race_df: pd.DataFrame,
    probability_column: str,
) -> pd.DataFrame:
    race_df = race_df.sort_values("lane").reset_index(drop=True)
    lane_to_prob = {int(row.lane): float(max(getattr(row, probability_column), 1e-12)) for row in race_df.itertuples()}
    actual_order = actual_trifecta_order(race_df)
    rows: list[dict[str, Any]] = []

    for trifecta in itertools.permutations(sorted(lane_to_prob.keys()), 3):
        row = {
            "race_id": race_df["race_id"].iloc[0],
            "trifecta": "-".join(str(x) for x in trifecta),
            "raw_probability": plackett_luce_probability(lane_to_prob, trifecta),
        }
        if actual_order is not None:
            row["is_actual"] = trifecta == actual_order
        rows.append(row)
    return pd.DataFrame(rows)


def enumerate_trifecta_probabilities_v2(race_df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"win_prob", "top2_prob", "top3_prob"}
    if not required_columns.issubset(race_df.columns):
        return enumerate_trifecta_probabilities_from_scores(race_df)

    race_df = race_df.sort_values("lane").reset_index(drop=True)
    rank_series = pd.to_numeric(race_df["win_probability_like"], errors="coerce").fillna(0.0)
    win_series = pd.to_numeric(race_df["win_prob"], errors="coerce").where(lambda s: s.notna(), rank_series)
    top2_series = pd.to_numeric(race_df["top2_prob"], errors="coerce").where(lambda s: s.notna(), win_series)
    top3_series = pd.to_numeric(race_df["top3_prob"], errors="coerce").where(lambda s: s.notna(), top2_series)

    rank_prob = rank_series.to_numpy(dtype=float)
    win_prob = win_series.to_numpy(dtype=float)
    top2_prob = top2_series.to_numpy(dtype=float)
    top3_prob = top3_series.to_numpy(dtype=float)

    exact_first = np.clip(win_prob, 1e-9, None)
    exact_second = np.clip(top2_prob - win_prob, 1e-9, None)
    exact_third = np.clip(top3_prob - top2_prob, 1e-9, None)

    first_weights = {
        int(lane): float(0.7 * exact_first[idx] + 0.3 * max(rank_prob[idx], 1e-9))
        for idx, lane in enumerate(race_df["lane"].astype(int).tolist())
    }
    second_weights = {
        int(lane): float(0.7 * exact_second[idx] + 0.3 * max(rank_prob[idx], 1e-9))
        for idx, lane in enumerate(race_df["lane"].astype(int).tolist())
    }
    third_weights = {
        int(lane): float(0.7 * exact_third[idx] + 0.3 * max(rank_prob[idx], 1e-9))
        for idx, lane in enumerate(race_df["lane"].astype(int).tolist())
    }

    actual_order = actual_trifecta_order(race_df)
    rows: list[dict[str, Any]] = []
    for trifecta in itertools.permutations(sorted(first_weights.keys()), 3):
        row = {
            "race_id": race_df["race_id"].iloc[0],
            "trifecta": "-".join(str(x) for x in trifecta),
            "raw_probability": conditional_position_probability(
                first_weights,
                second_weights,
                third_weights,
                trifecta,
            ),
        }
        if actual_order is not None:
            row["is_actual"] = trifecta == actual_order
        rows.append(row)
    return pd.DataFrame(rows)


def blend_trifecta_raw_probabilities(
    raw_v1: np.ndarray,
    raw_v2: np.ndarray,
    v1_weight: float,
) -> np.ndarray:
    weight = float(min(max(v1_weight, 0.0), 1.0))
    return weight * np.asarray(raw_v1, dtype=float) + (1.0 - weight) * np.asarray(raw_v2, dtype=float)


def optimize_trifecta_v2_blend_weight(
    models: dict[str, Any],
    weights: dict[str, float],
    valid_df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    classifier_models: dict[str, lgb.Booster] | None = None,
    flow_model: lgb.Booster | None = None,
    flow_classes: list[str] | None = None,
    staged_models: dict[str, lgb.Booster] | None = None,
) -> float:
    if valid_df.empty or not classifier_models:
        return 1.0

    valid_df = apply_prediction_time_measurement_proxies(valid_df)
    ranked = build_weighted_lane_probabilities(
        models,
        weights,
        valid_df,
        feature_columns,
        categorical_columns,
        classifier_models=classifier_models,
        flow_model=flow_model,
        flow_classes=flow_classes,
        staged_models=staged_models,
    )

    race_payloads: list[tuple[np.ndarray, np.ndarray, int]] = []
    for _, race_df in ranked.groupby("race_id", sort=False):
        v1 = enumerate_trifecta_probabilities_from_scores(race_df)
        v2 = enumerate_trifecta_probabilities_v2(race_df)
        actual_positions = np.flatnonzero(v1["is_actual"].to_numpy(dtype=bool)) if "is_actual" in v1.columns else np.array([], dtype=int)
        if len(actual_positions) != 1:
            continue
        race_payloads.append(
            (
                v1["raw_probability"].to_numpy(dtype=float),
                v2["raw_probability"].to_numpy(dtype=float),
                int(actual_positions[0]),
            )
        )

    if not race_payloads:
        return 1.0

    best_weight = 1.0
    best_objective = float("-inf")
    for weight in np.linspace(0.5, 1.0, 11):
        top1 = 0
        top12 = 0
        log_losses: list[float] = []
        for raw_v1, raw_v2, actual_idx in race_payloads:
            blended = blend_trifecta_raw_probabilities(raw_v1, raw_v2, float(weight))
            prob_sum = blended.sum()
            probs = blended / prob_sum if prob_sum > 0 else np.full_like(blended, 1.0 / len(blended))
            order = np.argsort(-probs)
            rank = int(np.where(order == actual_idx)[0][0])
            actual_prob = max(float(probs[actual_idx]), 1e-15)
            top1 += int(rank == 0)
            top12 += int(rank < 12)
            log_losses.append(-np.log(actual_prob))

        race_count = len(race_payloads)
        objective = (top1 / race_count) + 0.1 * (top12 / race_count) - 0.05 * float(np.mean(log_losses))
        if objective > best_objective:
            best_objective = objective
            best_weight = float(weight)

    return best_weight


def train_trifecta_v2_model(
    train_df: pd.DataFrame,
    models: dict[str, Any],
    ensemble_weights: dict[str, float],
    feature_columns: list[str],
    categorical_columns: list[str],
    classifier_models: dict[str, lgb.Booster] | None = None,
    flow_model: lgb.Booster | None = None,
    flow_classes: list[str] | None = None,
    staged_models: dict[str, lgb.Booster] | None = None,
    max_races: int = 0,
    top_n_candidates: int = 24,
    config: dict | None = None,
) -> Any | None:
    if train_df.empty or not classifier_models:
        return None

    race_ids = train_df["race_id"].drop_duplicates().tolist()
    if max_races and max_races > 0 and len(race_ids) > max_races:
        race_ids = race_ids[-max_races:]
        train_df = train_df[train_df["race_id"].isin(race_ids)].copy()

    ranked = build_weighted_lane_probabilities(
        models,
        ensemble_weights,
        train_df,
        feature_columns,
        categorical_columns,
        classifier_models=classifier_models,
        flow_model=flow_model,
        flow_classes=flow_classes,
        staged_models=staged_models,
    )

    phase3_settings = get_phase3_settings(config)
    feature_rows: list[pd.DataFrame] = []
    labels: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    groups: list[int] = []
    for _, race_df in ranked.groupby("race_id", sort=False):
        v1 = enumerate_trifecta_probabilities_from_scores(race_df)
        v2 = enumerate_trifecta_probabilities_v2(race_df).rename(columns={"raw_probability": "raw_probability_v2"})
        actual_order = actual_trifecta_order(race_df)
        if actual_order is None:
            continue
        candidate_mask = select_trifecta_training_candidates(v1, top_n=top_n_candidates, actual_order=actual_order)
        selected_v1 = v1.loc[candidate_mask].reset_index(drop=True)
        selected_v2 = v2.loc[candidate_mask].reset_index(drop=True)
        target_frame = build_trifecta_training_targets(
            selected_v1["trifecta"],
            actual_order,
            label_weights=phase3_settings["label_weights"],
        )
        if target_frame.empty:
            continue
        feature_rows.append(build_trifecta_feature_frame(race_df, selected_v1, selected_v2))
        labels.append(target_frame["relevance"].to_numpy(dtype=int))
        weights.append(target_frame["sample_weight"].to_numpy(dtype=float))
        groups.append(len(target_frame))

    if not feature_rows:
        return None

    x_train = pd.concat(feature_rows, ignore_index=True)
    y_train = np.concatenate(labels)
    sample_weight = np.concatenate(weights) if weights else np.ones(len(x_train), dtype=float)
    if y_train.max(initial=0) <= 0:
        return None

    train_set = lgb.Dataset(
        x_train,
        label=y_train,
        weight=sample_weight,
        group=groups,
        free_raw_data=False,
    )
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbosity": -1,
        "seed": 42,
        "label_gain": list(range(int(y_train.max()) + 2)),
    }
    booster = train_lightgbm_with_optional_gpu(
        params,
        train_set,
        config,
        num_boost_round=200,
        callbacks=[lgb.log_evaluation(100)],
    )
    return {
        "model_type": "lgbm_ranker",
        "phase": "phase2_reranker",
        "feature_version": TRIFECTA_V2_FEATURE_VERSION,
        "feature_names": list(x_train.columns),
        "booster": booster,
        "conservative_v1_weight": float(phase3_settings["rerank"]["default_conservative_weight"]),
        "rank_penalty_strength": float(phase3_settings["rerank"]["default_rank_penalty_strength"]),
        "rank_penalty_start": int(phase3_settings["rerank"]["rank_penalty_start"]),
        "rerank_top_n": get_default_rerank_top_n(config),
    }


def select_trifecta_training_candidates(
    v1_df: pd.DataFrame,
    top_n: int = 24,
    actual_order: tuple[int, int, int] | None = None,
) -> pd.Series:
    ordered = v1_df["raw_probability"].rank(ascending=False, method="first") <= int(top_n)
    actual = v1_df["is_actual"].astype(bool) if "is_actual" in v1_df.columns else pd.Series(False, index=v1_df.index)
    if actual_order is None:
        return ordered | actual
    first_actual, second_actual, third_actual = actual_order
    lanes = v1_df["trifecta"].str.split("-", expand=True).astype(int)
    first_match = lanes[0] == first_actual
    top2_overlap = (lanes[[0, 1]].isin([first_actual, second_actual]).sum(axis=1) >= 2)
    top3_overlap = (lanes[[0, 1, 2]].isin([first_actual, second_actual, third_actual]).sum(axis=1) >= 2)
    return ordered | actual | first_match | top2_overlap | top3_overlap


def build_trifecta_training_targets(
    trifectas: pd.Series,
    actual_order: tuple[int, int, int],
    label_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    weights = dict(DEFAULT_PHASE3_SETTINGS["label_weights"])
    if label_weights:
        weights.update(label_weights)
    rows: list[dict[str, float]] = []
    actual_top2 = set(actual_order[:2])
    actual_top3 = set(actual_order)
    for trifecta in trifectas.astype(str).tolist():
        first, second, third = [int(x) for x in trifecta.split("-")]
        top2_count = int(len({first, second} & actual_top2))
        top3_count = int(len({first, second, third} & actual_top3))
        exact_first = int(first == actual_order[0])
        exact_top2 = int(first == actual_order[0] and second == actual_order[1])
        exact_full = int((first, second, third) == actual_order)
        if exact_full:
            relevance = 5
            sample_weight = float(weights["exact_full"])
        elif exact_top2:
            relevance = 4
            sample_weight = float(weights["exact_top2"])
        elif exact_first and top3_count >= 2:
            relevance = 3
            sample_weight = float(weights["exact_first_with_two"])
        elif exact_first:
            relevance = 2
            sample_weight = float(weights["exact_first"])
        elif top2_count >= 2:
            relevance = 1
            sample_weight = float(weights["top2_overlap"])
        elif top3_count >= 2:
            relevance = 1
            sample_weight = float(weights["top3_overlap"])
        else:
            relevance = 0
            sample_weight = 0.15
        sample_weight += 0.03 * top3_count
        rows.append(
            {
                "trifecta": trifecta,
                "relevance": relevance,
                "sample_weight": sample_weight,
                "top2_overlap_count": top2_count,
                "top3_overlap_count": top3_count,
                "exact_first_match": exact_first,
                "exact_top2_match": exact_top2,
                "exact_full_match": exact_full,
            }
        )
    return pd.DataFrame(rows)


def train_phase3_conditional_trifecta_model(
    train_df: pd.DataFrame,
    models: dict[str, Any],
    ensemble_weights: dict[str, float],
    feature_columns: list[str],
    categorical_columns: list[str],
    classifier_models: dict[str, lgb.Booster] | None = None,
    flow_model: lgb.Booster | None = None,
    flow_classes: list[str] | None = None,
    staged_models: dict[str, lgb.Booster] | None = None,
    base_model: Any | None = None,
    max_races: int = 0,
    config: dict | None = None,
) -> Any | None:
    if base_model is None or train_df.empty or not classifier_models:
        return base_model

    race_ids = train_df["race_id"].drop_duplicates().tolist()
    if max_races and max_races > 0 and len(race_ids) > max_races:
        race_ids = race_ids[-max_races:]
        train_df = train_df[train_df["race_id"].isin(race_ids)].copy()

    ranked = build_weighted_lane_probabilities(
        models,
        ensemble_weights,
        train_df,
        feature_columns,
        categorical_columns,
        classifier_models=classifier_models,
        flow_model=flow_model,
        flow_classes=flow_classes,
        staged_models=staged_models,
    )
    pattern_model, pattern_feature_names, pattern_classes = train_phase3_pattern_model(ranked, config=config)
    pattern_bundle: dict[str, Any] | None = None
    if pattern_model is not None:
        pattern_bundle = {
            "model_type": "phase3_pattern_only",
            "phase3_pattern_model": pattern_model,
            "phase3_pattern_feature_names": pattern_feature_names,
            "phase3_pattern_classes": pattern_classes,
        }

    second_rows: list[pd.DataFrame] = []
    second_labels: list[np.ndarray] = []
    third_rows: list[pd.DataFrame] = []
    third_labels: list[np.ndarray] = []
    for _, race_df in ranked.groupby("race_id", sort=False):
        actual_order = actual_trifecta_order(race_df)
        if actual_order is None:
            continue
        first_ranked = (
            race_df.sort_values("win_probability_like", ascending=False)["lane"]
            .astype(int)
            .head(3)
            .tolist()
        )
        first_candidates = list(dict.fromkeys([*first_ranked, int(actual_order[0])]))
        for first_lane in first_candidates:
            second_target = next((lane for lane in actual_order if lane != first_lane), None)
            second_frame = build_phase3_second_feature_frame(
                race_df,
                first_lane,
                scenario_model_bundle=pattern_bundle,
            )
            if second_target is not None and not second_frame.empty:
                second_rows.append(second_frame)
                second_labels.append(
                    (second_frame["second_lane"].astype(int) == int(second_target)).astype(int).to_numpy()
                )

            second_ranked = (
                second_frame.sort_values("second_top2_prob", ascending=False)["second_lane"]
                .astype(int)
                .head(3)
                .tolist()
                if not second_frame.empty
                else []
            )
            second_candidates = list(dict.fromkeys([*second_ranked, int(second_target)]))
            for second_lane in second_candidates:
                third_target = next(
                    (lane for lane in actual_order if lane not in {first_lane, second_lane}),
                    None,
                )
                third_frame = build_phase3_third_feature_frame(
                    race_df,
                    first_lane,
                    second_lane,
                    scenario_model_bundle=pattern_bundle,
                )
                if third_target is not None and not third_frame.empty:
                    third_rows.append(third_frame)
                    third_labels.append(
                        (third_frame["third_lane"].astype(int) == int(third_target)).astype(int).to_numpy()
                    )

    if not second_rows or not third_rows:
        return base_model

    x_second = pd.concat(second_rows, ignore_index=True)
    y_second = np.concatenate(second_labels)
    x_third = pd.concat(third_rows, ignore_index=True)
    y_third = np.concatenate(third_labels)
    if y_second.sum() == 0 or y_third.sum() == 0:
        return base_model

    second_set = lgb.Dataset(x_second, label=y_second, free_raw_data=False)
    third_set = lgb.Dataset(x_third, label=y_third, free_raw_data=False)
    phase3_settings = get_phase3_settings(config)
    regularization = phase3_settings["regularization"]
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": int(regularization["num_leaves"]),
        "min_data_in_leaf": int(regularization["min_data_in_leaf"]),
        "feature_fraction": float(regularization["feature_fraction"]),
        "bagging_fraction": float(regularization["bagging_fraction"]),
        "bagging_freq": int(regularization["bagging_freq"]),
        "lambda_l1": float(regularization["lambda_l1"]),
        "lambda_l2": float(regularization["lambda_l2"]),
        "verbosity": -1,
        "seed": 42,
    }
    boost_round = int(regularization["num_boost_round"])
    second_model = train_lightgbm_with_optional_gpu(
        params,
        second_set,
        config,
        num_boost_round=boost_round,
        callbacks=[lgb.log_evaluation(100)],
    )
    third_model = train_lightgbm_with_optional_gpu(
        params,
        third_set,
        config,
        num_boost_round=boost_round,
        callbacks=[lgb.log_evaluation(100)],
    )

    bundle = dict(base_model) if is_trifecta_v2_bundle(base_model) else {
        "model_type": "lgbm_ranker",
        "phase": "phase2_reranker",
        "feature_names": [],
        "booster": base_model,
    }
    bundle["phase"] = "phase3_conditional"
    bundle["phase3_second_model"] = second_model
    bundle["phase3_second_feature_names"] = list(x_second.columns)
    bundle["phase3_third_model"] = third_model
    bundle["phase3_third_feature_names"] = list(x_third.columns)
    if pattern_model is not None:
        bundle["phase3_pattern_model"] = pattern_model
        bundle["phase3_pattern_feature_names"] = pattern_feature_names
        bundle["phase3_pattern_classes"] = pattern_classes
    bundle["conservative_v1_weight"] = float(phase3_settings["rerank"]["default_conservative_weight"])
    bundle["rank_penalty_strength"] = float(phase3_settings["rerank"]["default_rank_penalty_strength"])
    bundle["rank_penalty_start"] = int(phase3_settings["rerank"]["rank_penalty_start"])
    bundle["rerank_top_n"] = get_default_rerank_top_n(config)
    bundle["scenario_candidate_top_n"] = int(
        phase3_settings["rerank"].get("scenario_candidate_top_n", 0)
    )
    return bundle


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _row_numeric(row: pd.Series, *columns: str, default: float = 0.0) -> float:
    for column in columns:
        value = pd.to_numeric(row.get(column), errors="coerce")
        if pd.notna(value):
            return float(value)
    return float(default)


def _race_scale(values: pd.Series, lower_is_better: bool = False) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() < 2:
        return pd.Series(0.0, index=values.index)
    min_value = float(numeric.min())
    max_value = float(numeric.max())
    if np.isclose(max_value, min_value):
        return pd.Series(0.0, index=values.index)
    scaled = (numeric - min_value) / (max_value - min_value)
    if lower_is_better:
        scaled = 1.0 - scaled
    return scaled.fillna(0.0).astype(float)


def _phase3_scenario_context(lane_frame: pd.DataFrame) -> dict[str, float]:
    if lane_frame.empty:
        return {
            "escape_strength": 0.0,
            "inner_collapse_risk": 0.0,
            "sashi_pressure_2": 0.0,
            "makuri_pressure_3_4": 0.0,
            "makurizashi_pressure": 0.0,
            "outer_sweep_risk": 0.0,
            "attack_lane": 0.0,
            "attack_pressure": 0.0,
            "venue_escape_win_rate": 0.0,
            "venue_escape_top2_rate": 0.0,
            "venue_outer_top3_rate": 0.0,
            "s2_makuri_pressure": 0.0,
            "s3_attack_pressure": 0.0,
            "s4_makurizashi_pressure": 0.0,
            "s5_kado_makuri_pressure": 0.0,
            "s6_outer_attack_pressure": 0.0,
            "s7_chain_pressure": 0.0,
        }

    frame = lane_frame.copy()
    frame["rank_prob_ctx"] = frame.apply(lambda row: _row_numeric(row, "win_probability_like"), axis=1)
    frame["top2_prob_ctx"] = frame.apply(lambda row: _row_numeric(row, "top2_prob"), axis=1)
    frame["top3_prob_ctx"] = frame.apply(lambda row: _row_numeric(row, "top3_prob"), axis=1)
    frame["exact1_prob_ctx"] = frame.apply(lambda row: _row_numeric(row, "exact1_prob", "win_prob"), axis=1)
    frame["nige_prob_ctx"] = frame.apply(lambda row: _row_numeric(row, "flow_prob_nige"), axis=1)
    frame["sashi_prob_ctx"] = frame.apply(lambda row: _row_numeric(row, "flow_prob_sashi"), axis=1)
    frame["makuri_prob_ctx"] = frame.apply(lambda row: _row_numeric(row, "flow_prob_makuri"), axis=1)
    frame["makurizashi_prob_ctx"] = frame.apply(lambda row: _row_numeric(row, "flow_prob_makurizashi"), axis=1)
    frame["venue_course_win_ctx"] = frame.apply(
        lambda row: _row_numeric(row, "venue_course_prev_win_rate", "venue_lane_prev_win_rate"),
        axis=1,
    )
    frame["venue_course_top2_ctx"] = frame.apply(
        lambda row: _row_numeric(row, "venue_course_prev_top2_rate", "venue_lane_prev_top2_rate"),
        axis=1,
    )
    frame["venue_course_top3_ctx"] = frame.apply(
        lambda row: _row_numeric(row, "venue_course_prev_top3_rate", "venue_lane_prev_top3_rate"),
        axis=1,
    )
    frame["machine_ctx"] = frame.apply(
        lambda row: _row_numeric(row, "motor_top2_rate_hist", "motor_prev_top3_rate", "motor_place_rate")
        + _row_numeric(row, "boat_top2_rate_hist", "boat_prev_top3_rate", "boat_place_rate"),
        axis=1,
    )
    frame["st_ctx"] = frame.apply(
        lambda row: _row_numeric(row, "avg_st_last5", "racer_prev_avg_st_5", "racer_prev_avg_st"),
        axis=1,
    )
    frame["exhibition_ctx"] = frame.apply(lambda row: _row_numeric(row, "exhibition_time"), axis=1)
    frame["st_adv_ctx"] = _race_scale(frame["st_ctx"], lower_is_better=True)
    frame["exhibition_adv_ctx"] = _race_scale(frame["exhibition_ctx"], lower_is_better=True)
    frame["machine_adv_ctx"] = _race_scale(frame["machine_ctx"])

    attack_scores: dict[int, float] = {}
    for lane_value, row in frame.iterrows():
        lane = int(lane_value)
        lane_bias = 0.0
        if lane == 2:
            lane_bias = 0.08
        elif lane in (3, 4):
            lane_bias = 0.12
        elif lane >= 5:
            lane_bias = 0.05
        attack_scores[lane] = _clip01(
            0.22 * float(row["rank_prob_ctx"])
            + 0.18 * float(row["top2_prob_ctx"])
            + 0.16 * float(row["sashi_prob_ctx"])
            + 0.20 * max(float(row["makuri_prob_ctx"]), float(row["makurizashi_prob_ctx"]))
            + 0.06 * float(row["st_adv_ctx"])
            + 0.04 * float(row["exhibition_adv_ctx"])
            + 0.03 * float(row["machine_adv_ctx"])
            + 0.07 * float(row["venue_course_top2_ctx"])
            + 0.04 * float(row["venue_course_top3_ctx"])
            + lane_bias
        )

    outer_attack_scores = {lane: score for lane, score in attack_scores.items() if lane != 1}
    attack_lane = max(outer_attack_scores, key=outer_attack_scores.get) if outer_attack_scores else 0
    attack_pressure = float(outer_attack_scores.get(attack_lane, 0.0))
    lane1 = frame.loc[1] if 1 in frame.index else None
    if lane1 is None:
        escape_strength = 0.0
        lane1_top2 = 0.0
    else:
        lane1_top2 = float(lane1["top2_prob_ctx"])
        lane1_venue_win = float(lane1["venue_course_win_ctx"])
        lane1_venue_top2 = float(lane1["venue_course_top2_ctx"])
        escape_base = _clip01(
            0.27 * float(lane1["exact1_prob_ctx"])
            + 0.20 * float(lane1["rank_prob_ctx"])
            + 0.19 * float(lane1["nige_prob_ctx"])
            + 0.07 * float(lane1["st_adv_ctx"])
            + 0.05 * float(lane1["exhibition_adv_ctx"])
            + 0.07 * float(lane1["machine_adv_ctx"])
            + 0.09 * lane1_venue_win
            + 0.06 * lane1_venue_top2
        )
        escape_strength = _clip01(escape_base * (1.0 - 0.35 * attack_pressure))

    lane2 = frame.loc[2] if 2 in frame.index else None
    if lane2 is None:
        sashi_pressure_2 = 0.0
        s2_makuri_pressure = 0.0
    else:
        sashi_pressure_2 = _clip01(
            0.38 * float(lane2["sashi_prob_ctx"])
            + 0.24 * float(lane2["rank_prob_ctx"])
            + 0.18 * float(lane2["top2_prob_ctx"])
            + 0.10 * float(lane2["st_adv_ctx"])
            + 0.05 * float(lane2["exhibition_adv_ctx"])
            + 0.05 * float(lane2["machine_adv_ctx"])
        )
        s2_makuri_pressure = _clip01(
            0.34 * float(lane2["makuri_prob_ctx"])
            + 0.20 * float(lane2["rank_prob_ctx"])
            + 0.16 * float(lane2["top2_prob_ctx"])
            + 0.12 * float(lane2["st_adv_ctx"])
            + 0.08 * float(lane2["exhibition_adv_ctx"])
            + 0.05 * float(lane2["machine_adv_ctx"])
            + 0.05 * float(lane2["venue_course_top2_ctx"])
        )

    makuri_pressure_3_4 = max(
        (
            _clip01(
                0.34 * float(frame.loc[lane, "makuri_prob_ctx"])
                + 0.20 * float(frame.loc[lane, "makurizashi_prob_ctx"])
                + 0.20 * float(frame.loc[lane, "rank_prob_ctx"])
                + 0.12 * float(frame.loc[lane, "top2_prob_ctx"])
                + 0.08 * float(frame.loc[lane, "st_adv_ctx"])
                + 0.06 * float(frame.loc[lane, "exhibition_adv_ctx"])
            )
            for lane in (3, 4)
            if lane in frame.index
        ),
        default=0.0,
    )
    s3_attack_pressure = _clip01(
        0.42 * float(frame.loc[3, "makuri_prob_ctx"])
        + 0.18 * float(frame.loc[3, "makurizashi_prob_ctx"])
        + 0.16 * float(frame.loc[3, "rank_prob_ctx"])
        + 0.10 * float(frame.loc[3, "st_adv_ctx"])
        + 0.06 * float(frame.loc[3, "exhibition_adv_ctx"])
        + 0.08 * float(frame.loc[3, "venue_course_top2_ctx"])
    ) if 3 in frame.index else 0.0
    s5_kado_makuri_pressure = _clip01(
        0.46 * float(frame.loc[4, "makuri_prob_ctx"])
        + 0.16 * float(frame.loc[4, "rank_prob_ctx"])
        + 0.12 * float(frame.loc[4, "top2_prob_ctx"])
        + 0.10 * float(frame.loc[4, "st_adv_ctx"])
        + 0.06 * float(frame.loc[4, "exhibition_adv_ctx"])
        + 0.10 * float(frame.loc[4, "venue_course_top3_ctx"])
    ) if 4 in frame.index else 0.0
    makurizashi_pressure = max(
        (
            _clip01(
                0.42 * float(frame.loc[lane, "makurizashi_prob_ctx"])
                + 0.18 * float(frame.loc[lane, "makuri_prob_ctx"])
                + 0.18 * float(frame.loc[lane, "rank_prob_ctx"])
                + 0.12 * float(frame.loc[lane, "top3_prob_ctx"])
                + 0.10 * float(frame.loc[lane, "exhibition_adv_ctx"])
            )
            for lane in (3, 4, 5)
            if lane in frame.index
        ),
        default=0.0,
    )
    s4_makurizashi_pressure = max(
        (
            _clip01(
                0.44 * float(frame.loc[lane, "makurizashi_prob_ctx"])
                + 0.16 * float(frame.loc[lane, "makuri_prob_ctx"])
                + 0.16 * float(frame.loc[lane, "rank_prob_ctx"])
                + 0.10 * float(frame.loc[lane, "top3_prob_ctx"])
                + 0.08 * float(frame.loc[lane, "exhibition_adv_ctx"])
                + 0.06 * float(frame.loc[lane, "venue_course_top3_ctx"])
            )
            for lane in (3, 4)
            if lane in frame.index
        ),
        default=0.0,
    )
    s6_outer_attack_pressure = max(
        (
            _clip01(
                0.34 * float(frame.loc[lane, "makurizashi_prob_ctx"])
                + 0.20 * float(frame.loc[lane, "makuri_prob_ctx"])
                + 0.14 * float(frame.loc[lane, "rank_prob_ctx"])
                + 0.12 * float(frame.loc[lane, "top3_prob_ctx"])
                + 0.08 * float(frame.loc[lane, "exhibition_adv_ctx"])
                + 0.12 * float(frame.loc[lane, "venue_course_top3_ctx"])
            )
            for lane in (5, 6)
            if lane in frame.index
        ),
        default=0.0,
    )
    outer_sweep_risk = _clip01(
        0.30 * max((attack_scores.get(lane, 0.0) for lane in (4, 5, 6)), default=0.0)
        + 0.30 * makurizashi_pressure
        + 0.25 * max((float(frame.loc[lane, "top3_prob_ctx"]) for lane in (4, 5, 6) if lane in frame.index), default=0.0)
        + 0.15 * max(
            (float(frame.loc[lane, "venue_course_top3_ctx"]) for lane in (4, 5, 6) if lane in frame.index),
            default=0.0,
        )
    )
    lane1_venue_top2 = float(frame.loc[1, "venue_course_top2_ctx"]) if 1 in frame.index else 0.0
    inner_collapse_risk = _clip01(
        (1.0 - escape_strength) * (0.42 + 0.32 * attack_pressure)
        + 0.18 * (1.0 - lane1_top2)
        + 0.08 * (1.0 - lane1_venue_top2)
    )
    venue_escape_win_rate = float(frame.loc[1, "venue_course_win_ctx"]) if 1 in frame.index else 0.0
    venue_outer_top3_rate = max(
        (float(frame.loc[lane, "venue_course_top3_ctx"]) for lane in (4, 5, 6) if lane in frame.index),
        default=0.0,
    )
    attack_values = [float(score) for lane, score in attack_scores.items() if lane != 1]
    attack_spread = float(np.mean(sorted(attack_values, reverse=True)[:2])) if len(attack_values) >= 2 else attack_pressure
    s7_chain_pressure = _clip01(
        0.45 * inner_collapse_risk
        + 0.25 * attack_spread
        + 0.15 * max(s3_attack_pressure, s4_makurizashi_pressure, s5_kado_makuri_pressure)
        + 0.15 * max(s6_outer_attack_pressure, outer_sweep_risk)
    )

    return {
        "escape_strength": escape_strength,
        "inner_collapse_risk": inner_collapse_risk,
        "sashi_pressure_2": sashi_pressure_2,
        "makuri_pressure_3_4": makuri_pressure_3_4,
        "makurizashi_pressure": makurizashi_pressure,
        "outer_sweep_risk": outer_sweep_risk,
        "attack_lane": float(attack_lane),
        "attack_pressure": attack_pressure,
        "venue_escape_win_rate": venue_escape_win_rate,
        "venue_escape_top2_rate": lane1_venue_top2,
        "venue_outer_top3_rate": venue_outer_top3_rate,
        "s2_makuri_pressure": s2_makuri_pressure,
        "s3_attack_pressure": s3_attack_pressure,
        "s4_makurizashi_pressure": s4_makurizashi_pressure,
        "s5_kado_makuri_pressure": s5_kado_makuri_pressure,
        "s6_outer_attack_pressure": s6_outer_attack_pressure,
        "s7_chain_pressure": s7_chain_pressure,
    }


def _phase3_scenario_scores(scenario: dict[str, float]) -> dict[str, float]:
    escape = float(scenario.get("escape_strength", 0.0))
    collapse = float(scenario.get("inner_collapse_risk", 0.0))
    attack = float(scenario.get("attack_pressure", 0.0))
    return {
        "S0": _clip01(0.72 * escape + 0.18 * (1.0 - collapse) + 0.10 * (1.0 - attack)),
        "S1": _clip01(
            0.52 * float(scenario.get("sashi_pressure_2", 0.0))
            + 0.24 * escape
            + 0.14 * float(scenario.get("venue_escape_top2_rate", 0.0))
            + 0.10 * (1.0 - collapse)
        ),
        "S2": _clip01(
            0.58 * float(scenario.get("s2_makuri_pressure", 0.0))
            + 0.18 * attack
            + 0.14 * collapse
            + 0.10 * (1.0 - escape)
        ),
        "S3": _clip01(
            0.58 * float(scenario.get("s3_attack_pressure", 0.0))
            + 0.18 * float(scenario.get("makuri_pressure_3_4", 0.0))
            + 0.14 * collapse
            + 0.10 * attack
        ),
        "S4": _clip01(
            0.58 * float(scenario.get("s4_makurizashi_pressure", 0.0))
            + 0.18 * float(scenario.get("makurizashi_pressure", 0.0))
            + 0.12 * attack
            + 0.12 * collapse
        ),
        "S5": _clip01(
            0.62 * float(scenario.get("s5_kado_makuri_pressure", 0.0))
            + 0.16 * float(scenario.get("outer_sweep_risk", 0.0))
            + 0.12 * collapse
            + 0.10 * float(scenario.get("venue_outer_top3_rate", 0.0))
        ),
        "S6": _clip01(
            0.60 * float(scenario.get("s6_outer_attack_pressure", 0.0))
            + 0.18 * float(scenario.get("outer_sweep_risk", 0.0))
            + 0.12 * collapse
            + 0.10 * (1.0 - escape)
        ),
        "S7": _clip01(
            0.62 * float(scenario.get("s7_chain_pressure", 0.0))
            + 0.18 * collapse
            + 0.12 * attack
            + 0.08 * (1.0 - escape)
        ),
    }


def _phase3_scenario_label(scenario: dict[str, float]) -> str:
    candidates = _phase3_scenario_scores(scenario)
    label, strength = max(candidates.items(), key=lambda item: item[1])
    if strength < 0.25:
        return "S7"
    return label


def _phase3_scenario_feature_values(scenario: dict[str, float]) -> dict[str, float]:
    label = _phase3_scenario_label(scenario)
    scores = _phase3_scenario_scores(scenario)
    features = {f"scenario_{key.lower()}_score": value for key, value in scores.items()}
    features["scenario_id_numeric"] = float(int(label[1:]))
    return features


def _phase3_line_features(
    scenario: dict[str, float],
    first_lane: int,
    second_lane: int | None = None,
    third_lane: int | None = None,
) -> dict[str, float]:
    lanes = [lane for lane in (first_lane, second_lane, third_lane) if lane is not None]
    attack_lane = int(scenario.get("attack_lane", 0))
    attack_in_line = attack_lane in lanes
    escape_line_fit = float(first_lane == 1) * scenario["escape_strength"]
    sashi_line_fit = float(first_lane == 2 and (second_lane in (1, 3, 4))) * scenario["sashi_pressure_2"]
    makuri_line_fit = float(first_lane in (3, 4)) * scenario["makuri_pressure_3_4"]
    makurizashi_line_fit = float(first_lane in (3, 4, 5) and (second_lane is None or second_lane <= 4)) * scenario["makurizashi_pressure"]
    outer_follow_fit = float(any(lane >= 4 for lane in lanes[1:])) * scenario["outer_sweep_risk"]
    attack_line_fit = float(first_lane == attack_lane or (second_lane == attack_lane and first_lane == 1)) * scenario["attack_pressure"]
    scenario_mismatch_penalty = _clip01(
        float(scenario["escape_strength"] > 0.45 and first_lane != 1) * scenario["escape_strength"]
        + float(scenario["inner_collapse_risk"] > 0.45 and first_lane == 1) * scenario["inner_collapse_risk"]
        + float(scenario["attack_pressure"] > 0.45 and not attack_in_line) * scenario["attack_pressure"]
    )
    return {
        "escape_line_fit": escape_line_fit,
        "sashi_line_fit": sashi_line_fit,
        "makuri_line_fit": makuri_line_fit,
        "makurizashi_line_fit": makurizashi_line_fit,
        "outer_follow_fit": outer_follow_fit,
        "attack_line_fit": attack_line_fit,
        "scenario_mismatch_penalty": scenario_mismatch_penalty,
    }


def _phase3_scenario_context(lane_frame: pd.DataFrame) -> dict[str, float]:
    return score_pre_race_scenarios(lane_frame)


def _phase3_scenario_scores(scenario: dict[str, float]) -> dict[str, float]:
    return scenario_scores(scenario)


def _phase3_scenario_label(scenario: dict[str, float]) -> str:
    return scenario_label(scenario)


def _phase3_scenario_feature_values(scenario: dict[str, float]) -> dict[str, float]:
    return scenario_feature_values(scenario)


def _phase3_line_features(
    scenario: dict[str, float],
    first_lane: int,
    second_lane: int | None = None,
    third_lane: int | None = None,
) -> dict[str, float]:
    return scenario_line_features(scenario, first_lane, second_lane, third_lane)


def build_phase3_pattern_feature_frame(race_df: pd.DataFrame) -> pd.DataFrame:
    lane_frame = race_df.set_index("lane").copy()
    scenario = score_pre_race_scenarios(lane_frame)
    scores = scenario_scores(scenario)
    row: dict[str, float] = {
        "escape_strength": float(scenario.get("escape_strength", 0.0)),
        "inner_collapse_risk": float(scenario.get("inner_collapse_risk", 0.0)),
        "sashi_pressure_2": float(scenario.get("sashi_pressure_2", 0.0)),
        "s2_makuri_pressure": float(scenario.get("s2_makuri_pressure", 0.0)),
        "s3_attack_pressure": float(scenario.get("s3_attack_pressure", 0.0)),
        "s4_course_attack_pressure": float(scenario.get("s4_course_attack_pressure", 0.0)),
        "s5_outside_attack_pressure": float(scenario.get("s5_outside_attack_pressure", 0.0)),
        "makuri_pressure_3_4": float(scenario.get("makuri_pressure_3_4", 0.0)),
        "makurizashi_pressure": float(scenario.get("makurizashi_pressure", 0.0)),
        "outer_sweep_risk": float(scenario.get("outer_sweep_risk", 0.0)),
        "attack_lane": float(scenario.get("attack_lane", 0.0)),
        "attack_pressure": float(scenario.get("attack_pressure", 0.0)),
        "second_attack_pressure": float(scenario.get("second_attack_pressure", 0.0)),
        "attack_score_margin": float(scenario.get("attack_score_margin", 0.0)),
        "chaos_pressure": float(scenario.get("chaos_pressure", 0.0)),
        "venue_escape_win_rate": float(scenario.get("venue_escape_win_rate", 0.0)),
        "venue_escape_top2_rate": float(scenario.get("venue_escape_top2_rate", 0.0)),
        "venue_outer_top3_rate": float(scenario.get("venue_outer_top3_rate", 0.0)),
    }
    for short_id, score in scores.items():
        row[f"rule_{short_id.lower()}_score"] = float(score)

    for lane in range(1, 7):
        if lane in lane_frame.index:
            lane_row = lane_frame.loc[lane]
            row[f"lane{lane}_win_prob"] = _row_numeric(lane_row, "win_probability_like", "win_prob")
            row[f"lane{lane}_top2_prob"] = _row_numeric(lane_row, "top2_prob")
            row[f"lane{lane}_top3_prob"] = _row_numeric(lane_row, "top3_prob")
            row[f"lane{lane}_exact1_prob"] = _row_numeric(lane_row, "exact1_prob", "win_prob")
            row[f"lane{lane}_nige_prob"] = _row_numeric(lane_row, "flow_prob_nige")
            row[f"lane{lane}_sashi_prob"] = _row_numeric(lane_row, "flow_prob_sashi")
            row[f"lane{lane}_makuri_prob"] = _row_numeric(lane_row, "flow_prob_makuri")
            row[f"lane{lane}_makurizashi_prob"] = _row_numeric(lane_row, "flow_prob_makurizashi")
            row[f"lane{lane}_venue_course_win_rate"] = _row_numeric(
                lane_row, "venue_course_prev_win_rate", "venue_lane_prev_win_rate"
            )
            row[f"lane{lane}_venue_course_top3_rate"] = _row_numeric(
                lane_row, "venue_course_prev_top3_rate", "venue_lane_prev_top3_rate"
            )
        else:
            for suffix in (
                "win_prob",
                "top2_prob",
                "top3_prob",
                "exact1_prob",
                "nige_prob",
                "sashi_prob",
                "makuri_prob",
                "makurizashi_prob",
                "venue_course_win_rate",
                "venue_course_top3_rate",
            ):
                row[f"lane{lane}_{suffix}"] = 0.0
    return pd.DataFrame([row]).fillna(0.0).astype(float)


def train_phase3_pattern_model(
    ranked: pd.DataFrame,
    config: dict | None = None,
) -> tuple[lgb.Booster | None, list[str], list[str]]:
    feature_rows: list[pd.DataFrame] = []
    labels: list[int] = []
    class_labels = [SCENARIO_SHORT_TO_LABEL[f"S{i}"] for i in range(7)]
    class_to_index = {label: index for index, label in enumerate(class_labels)}
    for _, race_df in ranked.groupby("race_id", sort=False):
        result = classify_result_pattern(race_df)
        label = str(result.get("scenario", "S6_MIXED_OTHER"))
        if label not in class_to_index:
            continue
        feature_rows.append(build_phase3_pattern_feature_frame(race_df))
        labels.append(class_to_index[label])
    if len(feature_rows) < 20 or len(set(labels)) < 2:
        return None, [], class_labels

    x_train = pd.concat(feature_rows, ignore_index=True)
    y_train = np.asarray(labels, dtype=int)
    train_set = lgb.Dataset(x_train, label=y_train, free_raw_data=False)
    phase3_settings = get_phase3_settings(config)
    regularization = phase3_settings["regularization"]
    params = {
        "objective": "multiclass",
        "metric": "multi_logloss",
        "num_class": len(class_labels),
        "learning_rate": 0.05,
        "num_leaves": int(regularization["num_leaves"]),
        "min_data_in_leaf": max(int(regularization["min_data_in_leaf"]), 10),
        "feature_fraction": float(regularization["feature_fraction"]),
        "bagging_fraction": float(regularization["bagging_fraction"]),
        "bagging_freq": int(regularization["bagging_freq"]),
        "lambda_l1": float(regularization["lambda_l1"]),
        "lambda_l2": float(regularization["lambda_l2"]),
        "verbosity": -1,
        "seed": 42,
    }
    booster = train_lightgbm_with_optional_gpu(
        params,
        train_set,
        config,
        num_boost_round=max(80, min(int(regularization["num_boost_round"]), 200)),
        callbacks=[lgb.log_evaluation(100)],
    )
    return booster, list(x_train.columns), class_labels


def apply_phase3_pattern_model_to_scenario(
    scenario: dict[str, float],
    race_df: pd.DataFrame,
    model_bundle: Any | None,
) -> dict[str, float]:
    if not is_trifecta_v2_bundle(model_bundle):
        return scenario
    pattern_model = model_bundle.get("phase3_pattern_model")
    feature_names = model_bundle.get("phase3_pattern_feature_names", [])
    class_labels = model_bundle.get("phase3_pattern_classes", [])
    if pattern_model is None or not feature_names or not class_labels:
        return scenario

    features = build_phase3_pattern_feature_frame(race_df).reindex(columns=feature_names, fill_value=0.0)
    probabilities = np.asarray(pattern_model.predict(features), dtype=float)
    if probabilities.ndim == 2:
        probabilities = probabilities[0]
    if probabilities.size != len(class_labels):
        return scenario

    updated = dict(scenario)
    for label, probability in zip(class_labels, probabilities):
        numeric_id = scenario_numeric_id(label)
        updated[f"pattern_model_s{numeric_id}_probability"] = float(probability)
        if 0 <= numeric_id <= 6:
            updated[f"model_s{numeric_id}_score"] = float(probability)
    updated["pattern_model_available"] = 1.0
    return updated


def _lane_numeric_array(lane_frame: pd.DataFrame, column: str) -> np.ndarray:
    lane_index = pd.Index(range(1, 7), name=lane_frame.index.name)
    if column not in lane_frame.columns:
        return np.zeros(6, dtype=float)
    return (
        pd.to_numeric(lane_frame[column], errors="coerce")
        .reindex(lane_index)
        .fillna(0.0)
        .to_numpy(dtype=float)
    )


def _trifecta_probability_values(source: pd.DataFrame, trifectas: np.ndarray, probability_col: str) -> np.ndarray:
    probability_by_trifecta = (
        source[["trifecta", probability_col]]
        .drop_duplicates("trifecta", keep="last")
        .set_index("trifecta")[probability_col]
    )
    mapped = pd.Series(trifectas).map(probability_by_trifecta)
    return pd.to_numeric(mapped, errors="coerce").fillna(0.0).to_numpy(dtype=float)


def _phase3_line_feature_arrays(
    scenario: dict[str, float],
    first: np.ndarray,
    second: np.ndarray,
    third: np.ndarray,
) -> dict[str, np.ndarray]:
    attack_lane = int(scenario.get("attack_lane", 0))
    attack_in_line = (first == attack_lane) | (second == attack_lane) | (third == attack_lane)
    escape_strength = float(scenario.get("escape_strength", 0.0))
    sashi_pressure = float(scenario.get("sashi_pressure_2", 0.0))
    makuri_pressure = max(
        float(scenario.get("s2_makuri_pressure", 0.0)),
        float(scenario.get("s3_attack_pressure", 0.0)),
        float(scenario.get("s4_course_attack_pressure", 0.0)),
    )
    makurizashi_pressure = float(scenario.get("makurizashi_pressure", 0.0))
    outer_follow_pressure = max(
        float(scenario.get("outer_sweep_risk", 0.0)),
        float(scenario.get("s5_outside_attack_pressure", 0.0)),
    )
    attack_pressure = float(scenario.get("attack_pressure", 0.0))
    inner_collapse_risk = float(scenario.get("inner_collapse_risk", 0.0))
    inner_line = (first == 1) & np.isin(second, [2, 3])
    scenario_mismatch_penalty = np.clip(
        ((escape_strength > 0.45) & (first != 1)).astype(float) * escape_strength
        + ((inner_collapse_risk > 0.45) & inner_line).astype(float) * inner_collapse_risk
        + ((attack_pressure > 0.45) & (~attack_in_line)).astype(float) * attack_pressure,
        0.0,
        1.0,
    )
    return {
        "escape_line_fit": (first == 1).astype(float) * escape_strength,
        "sashi_line_fit": ((first == 2) & np.isin(second, [1, 3, 4])).astype(float) * sashi_pressure,
        "makuri_line_fit": np.isin(first, [2, 3, 4]).astype(float) * makuri_pressure,
        "makurizashi_line_fit": (np.isin(first, [3, 4, 5, 6]) & (second <= 4)).astype(float)
        * makurizashi_pressure,
        "outer_follow_fit": ((second >= 4) | (third >= 4)).astype(float) * outer_follow_pressure,
        "attack_line_fit": ((first == attack_lane) | ((second == attack_lane) & np.isin(first, [1, 2]))).astype(float)
        * attack_pressure,
        "scenario_mismatch_penalty": scenario_mismatch_penalty,
    }


def _build_trifecta_feature_frame_legacy(
    race_df: pd.DataFrame,
    v1_df: pd.DataFrame,
    v2_df: pd.DataFrame,
    scenario_model_bundle: Any | None = None,
) -> pd.DataFrame:
    v1_col = "raw_probability_v1" if "raw_probability_v1" in v1_df.columns else "raw_probability"
    v2_col = "raw_probability_v2" if "raw_probability_v2" in v2_df.columns else "raw_probability"
    lane_frame = race_df.set_index("lane").copy()
    top2_exact = np.clip(
        pd.to_numeric(lane_frame["top2_prob"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        - pd.to_numeric(lane_frame["win_prob"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
        1e-9,
        None,
    )
    top3_exact = np.clip(
        pd.to_numeric(lane_frame["top3_prob"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        - pd.to_numeric(lane_frame["top2_prob"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
        1e-9,
        None,
    )
    lane_frame["exact_second_prob"] = top2_exact
    lane_frame["exact_third_prob"] = top3_exact
    scenario = apply_phase3_pattern_model_to_scenario(_phase3_scenario_context(lane_frame), race_df, scenario_model_bundle)
    scenario_features = _phase3_scenario_feature_values(scenario)

    v1_map = dict(zip(v1_df["trifecta"], v1_df[v1_col]))
    v2_map = dict(zip(v2_df["trifecta"], v2_df[v2_col]))
    rows: list[dict[str, float]] = []
    for trifecta in v1_df["trifecta"].tolist():
        first, second, third = [int(x) for x in trifecta.split("-")]
        first_row = lane_frame.loc[first]
        second_row = lane_frame.loc[second]
        third_row = lane_frame.loc[third]
        line_features = _phase3_line_features(scenario, first, second, third)
        rows.append(
            {
                "raw_probability_v1": float(v1_map[trifecta]),
                "raw_probability_v2": float(v2_map.get(trifecta, 0.0)),
                "first_win_prob": float(pd.to_numeric(first_row.get("win_prob"), errors="coerce") or 0.0),
                "second_exact_prob": float(second_row["exact_second_prob"]),
                "third_exact_prob": float(third_row["exact_third_prob"]),
                "first_rank_prob": float(pd.to_numeric(first_row.get("win_probability_like"), errors="coerce") or 0.0),
                "second_rank_prob": float(pd.to_numeric(second_row.get("win_probability_like"), errors="coerce") or 0.0),
                "third_rank_prob": float(pd.to_numeric(third_row.get("win_probability_like"), errors="coerce") or 0.0),
                "first_exact1_prob": float(pd.to_numeric(first_row.get("exact1_prob"), errors="coerce") or 0.0),
                "second_exact2_prob": float(pd.to_numeric(second_row.get("exact2_prob"), errors="coerce") or 0.0),
                "third_exact3_prob": float(pd.to_numeric(third_row.get("exact3_prob"), errors="coerce") or 0.0),
                "first_flow_nige": float(pd.to_numeric(first_row.get("flow_prob_nige"), errors="coerce") or 0.0),
                "second_flow_sashi": float(pd.to_numeric(second_row.get("flow_prob_sashi"), errors="coerce") or 0.0),
                "third_flow_makurizashi": float(pd.to_numeric(third_row.get("flow_prob_makurizashi"), errors="coerce") or 0.0),
                "first_venue_course_win_rate": float(pd.to_numeric(first_row.get("venue_course_prev_win_rate"), errors="coerce") or 0.0),
                "first_venue_course_top2_rate": float(pd.to_numeric(first_row.get("venue_course_prev_top2_rate"), errors="coerce") or 0.0),
                "first_venue_course_top3_rate": float(pd.to_numeric(first_row.get("venue_course_prev_top3_rate"), errors="coerce") or 0.0),
                "second_venue_course_top2_rate": float(pd.to_numeric(second_row.get("venue_course_prev_top2_rate"), errors="coerce") or 0.0),
                "second_venue_course_top3_rate": float(pd.to_numeric(second_row.get("venue_course_prev_top3_rate"), errors="coerce") or 0.0),
                "third_venue_course_top3_rate": float(pd.to_numeric(third_row.get("venue_course_prev_top3_rate"), errors="coerce") or 0.0),
                "first_flow_makuri": float(pd.to_numeric(first_row.get("flow_prob_makuri"), errors="coerce") or 0.0),
                "first_flow_nuki": float(pd.to_numeric(first_row.get("flow_prob_nuki"), errors="coerce") or 0.0),
                "first_lane": float(first),
                "second_lane": float(second),
                "third_lane": float(third),
                "lane_sum": float(first + second + third),
                "inner_bias": float((first <= 2) + (second <= 3) + (third <= 4)),
                "rank_gap_12": float(
                    (pd.to_numeric(first_row.get("win_probability_like"), errors="coerce") or 0.0)
                    - (pd.to_numeric(second_row.get("win_probability_like"), errors="coerce") or 0.0)
                ),
                "rank_gap_23": float(
                    (pd.to_numeric(second_row.get("win_probability_like"), errors="coerce") or 0.0)
                    - (pd.to_numeric(third_row.get("win_probability_like"), errors="coerce") or 0.0)
                ),
                "lane_gap_12": float(abs(first - second)),
                "lane_gap_23": float(abs(second - third)),
                "lane_gap_13": float(abs(first - third)),
                "prob_sum_top3": float(
                    (pd.to_numeric(first_row.get("win_probability_like"), errors="coerce") or 0.0)
                    + (pd.to_numeric(second_row.get("win_probability_like"), errors="coerce") or 0.0)
                    + (pd.to_numeric(third_row.get("win_probability_like"), errors="coerce") or 0.0)
                ),
                "top2_prob_sum": float(
                    (pd.to_numeric(first_row.get("top2_prob"), errors="coerce") or 0.0)
                    + (pd.to_numeric(second_row.get("top2_prob"), errors="coerce") or 0.0)
                    + (pd.to_numeric(third_row.get("top2_prob"), errors="coerce") or 0.0)
                ),
                "top3_prob_sum": float(
                    (pd.to_numeric(first_row.get("top3_prob"), errors="coerce") or 0.0)
                    + (pd.to_numeric(second_row.get("top3_prob"), errors="coerce") or 0.0)
                    + (pd.to_numeric(third_row.get("top3_prob"), errors="coerce") or 0.0)
                ),
                "exact_sum_top3": float(
                    (pd.to_numeric(first_row.get("exact1_prob"), errors="coerce") or 0.0)
                    + (pd.to_numeric(second_row.get("exact2_prob"), errors="coerce") or 0.0)
                    + (pd.to_numeric(third_row.get("exact3_prob"), errors="coerce") or 0.0)
                ),
                "nige_alignment": float((first == 1) * (pd.to_numeric(first_row.get("flow_prob_nige"), errors="coerce") or 0.0)),
                "sashi_alignment": float((second > 1) * (pd.to_numeric(first_row.get("flow_prob_sashi"), errors="coerce") or 0.0)),
                "makuri_alignment": float((first >= 3) * (pd.to_numeric(first_row.get("flow_prob_makuri"), errors="coerce") or 0.0)),
                "makurizashi_alignment": float((first >= 3) * (pd.to_numeric(first_row.get("flow_prob_makurizashi"), errors="coerce") or 0.0)),
                "outer_attack_bias": float((first >= 4) + (second >= 4)),
                "escape_strength": scenario["escape_strength"],
                "inner_collapse_risk": scenario["inner_collapse_risk"],
                "sashi_pressure_2": scenario["sashi_pressure_2"],
                "makuri_pressure_3_4": scenario["makuri_pressure_3_4"],
                "makurizashi_pressure": scenario["makurizashi_pressure"],
                "outer_sweep_risk": scenario["outer_sweep_risk"],
                "attack_lane": scenario["attack_lane"],
                "attack_pressure": scenario["attack_pressure"],
                "venue_escape_win_rate": scenario["venue_escape_win_rate"],
                "venue_escape_top2_rate": scenario["venue_escape_top2_rate"],
                "venue_outer_top3_rate": scenario["venue_outer_top3_rate"],
                **scenario_features,
                **line_features,
            }
        )
    frame = pd.DataFrame(rows)
    for col in frame.columns:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0).astype(float)
    return frame


def build_trifecta_feature_frame(
    race_df: pd.DataFrame,
    v1_df: pd.DataFrame,
    v2_df: pd.DataFrame,
    scenario_model_bundle: Any | None = None,
) -> pd.DataFrame:
    v1_col = "raw_probability_v1" if "raw_probability_v1" in v1_df.columns else "raw_probability"
    v2_col = "raw_probability_v2" if "raw_probability_v2" in v2_df.columns else "raw_probability"
    lane_frame = race_df.set_index("lane").copy()
    scenario = apply_phase3_pattern_model_to_scenario(_phase3_scenario_context(lane_frame), race_df, scenario_model_bundle)
    scenario_features = _phase3_scenario_feature_values(scenario)

    trifectas = v1_df["trifecta"].astype(str).to_numpy()
    lanes = v1_df["trifecta"].astype(str).str.split("-", expand=True).astype(int).to_numpy(dtype=np.int16)
    first = lanes[:, 0].astype(np.int16)
    second = lanes[:, 1].astype(np.int16)
    third = lanes[:, 2].astype(np.int16)
    first_idx = first - 1
    second_idx = second - 1
    third_idx = third - 1

    win = _lane_numeric_array(lane_frame, "win_prob")
    rank = _lane_numeric_array(lane_frame, "win_probability_like")
    top2 = _lane_numeric_array(lane_frame, "top2_prob")
    top3 = _lane_numeric_array(lane_frame, "top3_prob")
    exact1 = _lane_numeric_array(lane_frame, "exact1_prob")
    exact2 = _lane_numeric_array(lane_frame, "exact2_prob")
    exact3 = _lane_numeric_array(lane_frame, "exact3_prob")
    flow_nige = _lane_numeric_array(lane_frame, "flow_prob_nige")
    flow_sashi = _lane_numeric_array(lane_frame, "flow_prob_sashi")
    flow_makuri = _lane_numeric_array(lane_frame, "flow_prob_makuri")
    flow_makurizashi = _lane_numeric_array(lane_frame, "flow_prob_makurizashi")
    flow_nuki = _lane_numeric_array(lane_frame, "flow_prob_nuki")
    venue_win = _lane_numeric_array(lane_frame, "venue_course_prev_win_rate")
    venue_top2 = _lane_numeric_array(lane_frame, "venue_course_prev_top2_rate")
    venue_top3 = _lane_numeric_array(lane_frame, "venue_course_prev_top3_rate")
    exact_second = np.clip(top2 - win, 1e-9, None)
    exact_third = np.clip(top3 - top2, 1e-9, None)

    row_count = len(trifectas)
    data: dict[str, Any] = {
        "raw_probability_v1": _trifecta_probability_values(v1_df, trifectas, v1_col),
        "raw_probability_v2": _trifecta_probability_values(v2_df, trifectas, v2_col),
        "first_win_prob": win[first_idx],
        "second_exact_prob": exact_second[second_idx],
        "third_exact_prob": exact_third[third_idx],
        "first_rank_prob": rank[first_idx],
        "second_rank_prob": rank[second_idx],
        "third_rank_prob": rank[third_idx],
        "first_exact1_prob": exact1[first_idx],
        "second_exact2_prob": exact2[second_idx],
        "third_exact3_prob": exact3[third_idx],
        "first_flow_nige": flow_nige[first_idx],
        "second_flow_sashi": flow_sashi[second_idx],
        "third_flow_makurizashi": flow_makurizashi[third_idx],
        "first_venue_course_win_rate": venue_win[first_idx],
        "first_venue_course_top2_rate": venue_top2[first_idx],
        "first_venue_course_top3_rate": venue_top3[first_idx],
        "second_venue_course_top2_rate": venue_top2[second_idx],
        "second_venue_course_top3_rate": venue_top3[second_idx],
        "third_venue_course_top3_rate": venue_top3[third_idx],
        "first_flow_makuri": flow_makuri[first_idx],
        "first_flow_nuki": flow_nuki[first_idx],
        "first_lane": first.astype(float),
        "second_lane": second.astype(float),
        "third_lane": third.astype(float),
        "lane_sum": (first + second + third).astype(float),
        "inner_bias": ((first <= 2).astype(int) + (second <= 3).astype(int) + (third <= 4).astype(int)).astype(float),
        "rank_gap_12": rank[first_idx] - rank[second_idx],
        "rank_gap_23": rank[second_idx] - rank[third_idx],
        "lane_gap_12": np.abs(first - second).astype(float),
        "lane_gap_23": np.abs(second - third).astype(float),
        "lane_gap_13": np.abs(first - third).astype(float),
        "prob_sum_top3": rank[first_idx] + rank[second_idx] + rank[third_idx],
        "top2_prob_sum": top2[first_idx] + top2[second_idx] + top2[third_idx],
        "top3_prob_sum": top3[first_idx] + top3[second_idx] + top3[third_idx],
        "exact_sum_top3": exact1[first_idx] + exact2[second_idx] + exact3[third_idx],
        "nige_alignment": (first == 1).astype(float) * flow_nige[first_idx],
        "sashi_alignment": (second > 1).astype(float) * flow_sashi[first_idx],
        "makuri_alignment": (first >= 3).astype(float) * flow_makuri[first_idx],
        "makurizashi_alignment": (first >= 3).astype(float) * flow_makurizashi[first_idx],
        "outer_attack_bias": ((first >= 4).astype(int) + (second >= 4).astype(int)).astype(float),
        "escape_strength": np.full(row_count, float(scenario["escape_strength"])),
        "inner_collapse_risk": np.full(row_count, float(scenario["inner_collapse_risk"])),
        "sashi_pressure_2": np.full(row_count, float(scenario["sashi_pressure_2"])),
        "makuri_pressure_3_4": np.full(row_count, float(scenario["makuri_pressure_3_4"])),
        "makurizashi_pressure": np.full(row_count, float(scenario["makurizashi_pressure"])),
        "outer_sweep_risk": np.full(row_count, float(scenario["outer_sweep_risk"])),
        "attack_lane": np.full(row_count, float(scenario["attack_lane"])),
        "attack_pressure": np.full(row_count, float(scenario["attack_pressure"])),
        "venue_escape_win_rate": np.full(row_count, float(scenario["venue_escape_win_rate"])),
        "venue_escape_top2_rate": np.full(row_count, float(scenario["venue_escape_top2_rate"])),
        "venue_outer_top3_rate": np.full(row_count, float(scenario["venue_outer_top3_rate"])),
    }
    for key, value in scenario_features.items():
        data[key] = np.full(row_count, float(value))
    data.update(_phase3_line_feature_arrays(scenario, first, second, third))

    frame = pd.DataFrame(data)
    for col in frame.columns:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0).astype(float)
    return frame


def build_phase3_second_feature_frame(
    race_df: pd.DataFrame,
    first_lane: int,
    scenario_model_bundle: Any | None = None,
) -> pd.DataFrame:
    lane_frame = race_df.set_index("lane").copy()
    if first_lane not in lane_frame.index:
        return pd.DataFrame()
    first_row = lane_frame.loc[first_lane]
    scenario = apply_phase3_pattern_model_to_scenario(_phase3_scenario_context(lane_frame), race_df, scenario_model_bundle)
    scenario_features = _phase3_scenario_feature_values(scenario)
    rows: list[dict[str, float]] = []
    for second_lane in [int(lane) for lane in lane_frame.index if int(lane) != int(first_lane)]:
        second_row = lane_frame.loc[second_lane]
        line_features = _phase3_line_features(scenario, int(first_lane), int(second_lane))
        first_rank_prob = float(pd.to_numeric(first_row.get("win_probability_like"), errors="coerce") or 0.0)
        second_rank_prob = float(pd.to_numeric(second_row.get("win_probability_like"), errors="coerce") or 0.0)
        first_top2_prob = float(pd.to_numeric(first_row.get("top2_prob"), errors="coerce") or 0.0)
        second_top2_prob = float(pd.to_numeric(second_row.get("top2_prob"), errors="coerce") or 0.0)
        first_top3_prob = float(pd.to_numeric(first_row.get("top3_prob"), errors="coerce") or 0.0)
        second_top3_prob = float(pd.to_numeric(second_row.get("top3_prob"), errors="coerce") or 0.0)
        first_st5 = float(pd.to_numeric(first_row.get("avg_st_last5"), errors="coerce") or 0.0)
        second_st5 = float(pd.to_numeric(second_row.get("avg_st_last5"), errors="coerce") or 0.0)
        first_st10 = float(pd.to_numeric(first_row.get("avg_st_last10"), errors="coerce") or 0.0)
        second_st10 = float(pd.to_numeric(second_row.get("avg_st_last10"), errors="coerce") or 0.0)
        first_finish5 = float(pd.to_numeric(first_row.get("avg_finish_last5"), errors="coerce") or 0.0)
        second_finish5 = float(pd.to_numeric(second_row.get("avg_finish_last5"), errors="coerce") or 0.0)
        first_finish10 = float(pd.to_numeric(first_row.get("avg_finish_last10"), errors="coerce") or 0.0)
        second_finish10 = float(pd.to_numeric(second_row.get("avg_finish_last10"), errors="coerce") or 0.0)
        first_course_top1 = float(pd.to_numeric(first_row.get("course_top1_rate_hist"), errors="coerce") or 0.0)
        second_course_top1 = float(pd.to_numeric(second_row.get("course_top1_rate_hist"), errors="coerce") or 0.0)
        second_course_top3 = float(pd.to_numeric(second_row.get("course_top3_rate_hist"), errors="coerce") or 0.0)
        second_local_win = float(pd.to_numeric(second_row.get("local_win_rate"), errors="coerce") or 0.0)
        second_national_win = float(pd.to_numeric(second_row.get("national_win_rate"), errors="coerce") or 0.0)
        second_motor = float(pd.to_numeric(second_row.get("motor_top2_rate_hist"), errors="coerce") or 0.0)
        second_boat = float(pd.to_numeric(second_row.get("boat_top2_rate_hist"), errors="coerce") or 0.0)
        first_venue_win = float(pd.to_numeric(first_row.get("venue_course_prev_win_rate"), errors="coerce") or 0.0)
        first_venue_top2 = float(pd.to_numeric(first_row.get("venue_course_prev_top2_rate"), errors="coerce") or 0.0)
        second_venue_top2 = float(pd.to_numeric(second_row.get("venue_course_prev_top2_rate"), errors="coerce") or 0.0)
        second_venue_top3 = float(pd.to_numeric(second_row.get("venue_course_prev_top3_rate"), errors="coerce") or 0.0)
        rows.append(
            {
                "first_lane": float(first_lane),
                "second_lane": float(second_lane),
                "lane_gap_12": float(abs(int(first_lane) - int(second_lane))),
                "first_exact1_prob": float(pd.to_numeric(first_row.get("exact1_prob"), errors="coerce") or 0.0),
                "first_win_prob": float(pd.to_numeric(first_row.get("win_prob"), errors="coerce") or 0.0),
                "first_rank_prob": first_rank_prob,
                "second_exact2_prob": float(pd.to_numeric(second_row.get("exact2_prob"), errors="coerce") or 0.0),
                "second_top2_prob": second_top2_prob,
                "second_rank_prob": second_rank_prob,
                "first_flow_nige": float(pd.to_numeric(first_row.get("flow_prob_nige"), errors="coerce") or 0.0),
                "first_flow_makuri": float(pd.to_numeric(first_row.get("flow_prob_makuri"), errors="coerce") or 0.0),
                "first_flow_sashi": float(pd.to_numeric(first_row.get("flow_prob_sashi"), errors="coerce") or 0.0),
                "rank_gap_12": first_rank_prob - second_rank_prob,
                "top2_gap_12": first_top2_prob - second_top2_prob,
                "top3_gap_12": first_top3_prob - second_top3_prob,
                "st_gap_5": second_st5 - first_st5,
                "st_gap_10": second_st10 - first_st10,
                "finish_gap_5": first_finish5 - second_finish5,
                "finish_gap_10": first_finish10 - second_finish10,
                "second_course_top1_rate": second_course_top1,
                "second_course_top3_rate": second_course_top3,
                "first_course_top1_rate": first_course_top1,
                "course_top1_gap_12": first_course_top1 - second_course_top1,
                "second_local_win_rate": second_local_win,
                "second_national_win_rate": second_national_win,
                "second_motor_top2_rate": second_motor,
                "second_boat_top2_rate": second_boat,
                "second_machine_strength": second_motor + second_boat,
                "first_venue_course_win_rate": first_venue_win,
                "first_venue_course_top2_rate": first_venue_top2,
                "second_venue_course_top2_rate": second_venue_top2,
                "second_venue_course_top3_rate": second_venue_top3,
                "venue_course_top2_gap_12": first_venue_top2 - second_venue_top2,
                "inside_follow_alignment": float((int(first_lane) <= 2 and int(second_lane) <= 4) * second_course_top3),
                "outside_chase_alignment": float((int(first_lane) >= 3 and int(second_lane) >= 4) * second_top2_prob),
                "escape_strength": scenario["escape_strength"],
                "inner_collapse_risk": scenario["inner_collapse_risk"],
                "sashi_pressure_2": scenario["sashi_pressure_2"],
                "makuri_pressure_3_4": scenario["makuri_pressure_3_4"],
                "makurizashi_pressure": scenario["makurizashi_pressure"],
                "outer_sweep_risk": scenario["outer_sweep_risk"],
                "attack_lane": scenario["attack_lane"],
                "attack_pressure": scenario["attack_pressure"],
                "venue_escape_win_rate": scenario["venue_escape_win_rate"],
                "venue_escape_top2_rate": scenario["venue_escape_top2_rate"],
                "venue_outer_top3_rate": scenario["venue_outer_top3_rate"],
                **scenario_features,
                **line_features,
            }
        )
    return pd.DataFrame(rows)


def build_phase3_third_feature_frame(
    race_df: pd.DataFrame,
    first_lane: int,
    second_lane: int,
    scenario_model_bundle: Any | None = None,
) -> pd.DataFrame:
    lane_frame = race_df.set_index("lane").copy()
    if first_lane not in lane_frame.index or second_lane not in lane_frame.index:
        return pd.DataFrame()
    first_row = lane_frame.loc[first_lane]
    second_row = lane_frame.loc[second_lane]
    scenario = apply_phase3_pattern_model_to_scenario(_phase3_scenario_context(lane_frame), race_df, scenario_model_bundle)
    scenario_features = _phase3_scenario_feature_values(scenario)
    rows: list[dict[str, float]] = []
    excluded = {int(first_lane), int(second_lane)}
    for third_lane in [int(lane) for lane in lane_frame.index if int(lane) not in excluded]:
        third_row = lane_frame.loc[third_lane]
        line_features = _phase3_line_features(scenario, int(first_lane), int(second_lane), int(third_lane))
        first_rank_prob = float(pd.to_numeric(first_row.get("win_probability_like"), errors="coerce") or 0.0)
        second_rank_prob = float(pd.to_numeric(second_row.get("win_probability_like"), errors="coerce") or 0.0)
        third_rank_prob = float(pd.to_numeric(third_row.get("win_probability_like"), errors="coerce") or 0.0)
        second_top2_prob = float(pd.to_numeric(second_row.get("top2_prob"), errors="coerce") or 0.0)
        third_top3_prob = float(pd.to_numeric(third_row.get("top3_prob"), errors="coerce") or 0.0)
        first_top3_prob = float(pd.to_numeric(first_row.get("top3_prob"), errors="coerce") or 0.0)
        second_top3_prob = float(pd.to_numeric(second_row.get("top3_prob"), errors="coerce") or 0.0)
        third_st5 = float(pd.to_numeric(third_row.get("avg_st_last5"), errors="coerce") or 0.0)
        third_st10 = float(pd.to_numeric(third_row.get("avg_st_last10"), errors="coerce") or 0.0)
        second_st5 = float(pd.to_numeric(second_row.get("avg_st_last5"), errors="coerce") or 0.0)
        third_finish5 = float(pd.to_numeric(third_row.get("avg_finish_last5"), errors="coerce") or 0.0)
        third_finish10 = float(pd.to_numeric(third_row.get("avg_finish_last10"), errors="coerce") or 0.0)
        third_course_top3 = float(pd.to_numeric(third_row.get("course_top3_rate_hist"), errors="coerce") or 0.0)
        third_local_win = float(pd.to_numeric(third_row.get("local_win_rate"), errors="coerce") or 0.0)
        third_national_win = float(pd.to_numeric(third_row.get("national_win_rate"), errors="coerce") or 0.0)
        third_motor = float(pd.to_numeric(third_row.get("motor_top2_rate_hist"), errors="coerce") or 0.0)
        third_boat = float(pd.to_numeric(third_row.get("boat_top2_rate_hist"), errors="coerce") or 0.0)
        first_venue_win = float(pd.to_numeric(first_row.get("venue_course_prev_win_rate"), errors="coerce") or 0.0)
        second_venue_top2 = float(pd.to_numeric(second_row.get("venue_course_prev_top2_rate"), errors="coerce") or 0.0)
        third_venue_top3 = float(pd.to_numeric(third_row.get("venue_course_prev_top3_rate"), errors="coerce") or 0.0)
        rows.append(
            {
                "first_lane": float(first_lane),
                "second_lane": float(second_lane),
                "third_lane": float(third_lane),
                "lane_gap_12": float(abs(int(first_lane) - int(second_lane))),
                "lane_gap_23": float(abs(int(second_lane) - int(third_lane))),
                "lane_gap_13": float(abs(int(first_lane) - int(third_lane))),
                "first_exact1_prob": float(pd.to_numeric(first_row.get("exact1_prob"), errors="coerce") or 0.0),
                "second_exact2_prob": float(pd.to_numeric(second_row.get("exact2_prob"), errors="coerce") or 0.0),
                "third_exact3_prob": float(pd.to_numeric(third_row.get("exact3_prob"), errors="coerce") or 0.0),
                "third_top3_prob": third_top3_prob,
                "third_rank_prob": third_rank_prob,
                "first_flow_nige": float(pd.to_numeric(first_row.get("flow_prob_nige"), errors="coerce") or 0.0),
                "second_flow_sashi": float(pd.to_numeric(second_row.get("flow_prob_sashi"), errors="coerce") or 0.0),
                "third_flow_makurizashi": float(pd.to_numeric(third_row.get("flow_prob_makurizashi"), errors="coerce") or 0.0),
                "rank_gap_23": second_rank_prob - third_rank_prob,
                "rank_gap_13": first_rank_prob - third_rank_prob,
                "top3_gap_23": second_top3_prob - third_top3_prob,
                "top3_gap_13": first_top3_prob - third_top3_prob,
                "st_gap_23_5": third_st5 - second_st5,
                "st_10_third": third_st10,
                "finish_last5_third": third_finish5,
                "finish_last10_third": third_finish10,
                "third_course_top3_rate": third_course_top3,
                "third_local_win_rate": third_local_win,
                "third_national_win_rate": third_national_win,
                "third_motor_top2_rate": third_motor,
                "third_boat_top2_rate": third_boat,
                "third_machine_strength": third_motor + third_boat,
                "first_venue_course_win_rate": first_venue_win,
                "second_venue_course_top2_rate": second_venue_top2,
                "third_venue_course_top3_rate": third_venue_top3,
                "venue_course_top3_gap_23": second_venue_top2 - third_venue_top3,
                "top3_mass_first_second": first_top3_prob + second_top3_prob,
                "third_residual_top3": max(third_top3_prob - second_top2_prob, 0.0),
                "third_inside_scrap_alignment": float((int(third_lane) <= 4) * third_course_top3),
                "third_outer_scrap_alignment": float((int(third_lane) >= 4) * third_top3_prob),
                "escape_strength": scenario["escape_strength"],
                "inner_collapse_risk": scenario["inner_collapse_risk"],
                "sashi_pressure_2": scenario["sashi_pressure_2"],
                "makuri_pressure_3_4": scenario["makuri_pressure_3_4"],
                "makurizashi_pressure": scenario["makurizashi_pressure"],
                "outer_sweep_risk": scenario["outer_sweep_risk"],
                "attack_lane": scenario["attack_lane"],
                "attack_pressure": scenario["attack_pressure"],
                "venue_escape_win_rate": scenario["venue_escape_win_rate"],
                "venue_escape_top2_rate": scenario["venue_escape_top2_rate"],
                "venue_outer_top3_rate": scenario["venue_outer_top3_rate"],
                **scenario_features,
                **line_features,
            }
        )
    return pd.DataFrame(rows)


def apply_phase3_conditional_scores(
    race_df: pd.DataFrame,
    trifecta_df: pd.DataFrame,
    base_scores: np.ndarray,
    model_bundle: dict[str, Any],
) -> np.ndarray:
    second_model = model_bundle.get("phase3_second_model")
    third_model = model_bundle.get("phase3_third_model")
    if second_model is None or third_model is None:
        return base_scores

    adjusted = np.asarray(base_scores, dtype=float).copy()
    second_cache: dict[int, pd.DataFrame] = {}
    third_cache: dict[tuple[int, int], pd.DataFrame] = {}
    second_names = model_bundle.get("phase3_second_feature_names", [])
    third_names = model_bundle.get("phase3_third_feature_names", [])

    for idx, trifecta in enumerate(trifecta_df["trifecta"].astype(str).tolist()):
        first_lane, second_lane, third_lane = [int(x) for x in trifecta.split("-")]
        if first_lane not in second_cache:
            second_cache[first_lane] = build_phase3_second_feature_frame(
                race_df,
                first_lane,
                scenario_model_bundle=model_bundle,
            )
        second_frame = second_cache[first_lane]
        second_prob = 1.0
        if not second_frame.empty:
            second_match = second_frame[second_frame["second_lane"] == float(second_lane)]
            if not second_match.empty:
                second_prob = float(
                    second_model.predict(second_match.reindex(columns=second_names, fill_value=0.0))[0]
                )

        pair_key = (first_lane, second_lane)
        if pair_key not in third_cache:
            third_cache[pair_key] = build_phase3_third_feature_frame(
                race_df,
                first_lane,
                second_lane,
                scenario_model_bundle=model_bundle,
            )
        third_frame = third_cache[pair_key]
        third_prob = 1.0
        if not third_frame.empty:
            third_match = third_frame[third_frame["third_lane"] == float(third_lane)]
            if not third_match.empty:
                third_prob = float(
                    third_model.predict(third_match.reindex(columns=third_names, fill_value=0.0))[0]
                )

        adjusted[idx] = float(adjusted[idx]) * max(second_prob, 1e-9) * max(third_prob, 1e-9)
    return adjusted


def actual_trifecta_order(race_df: pd.DataFrame) -> tuple[int, int, int] | None:
    if "finish_position" not in race_df.columns:
        return None
    ordered = race_df.sort_values("finish_position").head(3)
    if len(ordered) < 3:
        return None
    return tuple(ordered["lane"].astype(int).tolist())


def plackett_luce_probability(lane_to_prob: dict[int, float], trifecta: tuple[int, int, int]) -> float:
    remaining = dict(lane_to_prob)
    probability = 1.0
    for lane in trifecta:
        denom = sum(remaining.values())
        if denom <= 0:
            return 0.0
        probability *= remaining[lane] / denom
        remaining.pop(lane, None)
    return probability


def conditional_position_probability(
    first_weights: dict[int, float],
    second_weights: dict[int, float],
    third_weights: dict[int, float],
    trifecta: tuple[int, int, int],
) -> float:
    first, second, third = trifecta
    p1 = _conditional_pick_probability(first_weights, first, excluded=())
    p2 = _conditional_pick_probability(second_weights, second, excluded=(first,))
    p3 = _conditional_pick_probability(third_weights, third, excluded=(first, second))
    return p1 * p2 * p3


def _conditional_pick_probability(
    weights: dict[int, float],
    lane: int,
    excluded: tuple[int, ...],
) -> float:
    remaining = {key: value for key, value in weights.items() if key not in excluded}
    denom = sum(remaining.values())
    if denom <= 0 or lane not in remaining:
        return 0.0
    return remaining[lane] / denom


def _softmax(values: pd.Series | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    shifted = array - np.nanmax(array)
    exps = np.exp(shifted)
    denom = exps.sum()
    if denom == 0:
        return np.full_like(exps, 1.0 / len(exps))
    return exps / denom


def sort_for_grouping(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["race_id", "lane"]).reset_index(drop=True)
