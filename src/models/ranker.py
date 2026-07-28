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
from src.odds.expected_value import attach_expected_value_columns


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
}

DEFAULT_ARTIFACT_PATHS = {
    "catboost_model_path": "artifacts/catboost_ranker.cbm",
    "lightgbm_model_path": "artifacts/lightgbm_ranker.txt",
    "xgboost_model_path": "artifacts/xgboost_ranker.json",
    "features_path": "artifacts/feature_columns.json",
    "ensemble_weights_path": "artifacts/ensemble_weights.json",
    "trifecta_calibrator_path": "artifacts/trifecta_isotonic.joblib",
    "metrics_path": "artifacts/metrics.json",
    "classifier_dir": "artifacts/classifiers",
    "train_checkpoint_path": "artifacts/train_checkpoint.json",
}


RESERVED_MODEL_NAMES = {"catboost", "lightgbm", "xgboost"}
LIGHTGBM_VARIANT_NAME_RE = re.compile(r"^lightgbm_[A-Za-z0-9_]+$")
XGBOOST_VARIANT_NAME_RE = re.compile(r"^xgboost_[A-Za-z0-9_]+$")
TRIFECTA_V2_FEATURE_VERSION = 2

DEFAULT_CATBOOST_SETTINGS = {
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


DEFAULT_ENSEMBLE_SETTINGS = {
    "parallel_workers": 1,
    "max_eval_races": 0,
    "grid_step": 0.10,
    "objective": "trifecta_top12_balanced",
    "objective_top12_weight": 0.35,
    "objective_top5_weight": 0.25,
    "objective_top3_weight": 0.15,
    "objective_top1_weight": 0.10,
    "objective_top3_overlap_weight": 0.10,
    "objective_log_loss_weight": 0.05,
}


ENSEMBLE_OBJECTIVES = {
    "rank_legacy",
    "trifecta_fast",
    "trifecta_top12_balanced",
    "trifecta_top12_simple",
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
    return settings


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


def get_ensemble_settings(config: dict | None) -> dict[str, Any]:
    configured = ((config or {}).get("models", {}) or {}).get("ensemble", {})
    settings = {
        **DEFAULT_ENSEMBLE_SETTINGS,
        **(configured or {}),
    }
    settings["parallel_workers"] = max(int(settings.get("parallel_workers", 1)), 1)
    settings["max_eval_races"] = max(int(settings.get("max_eval_races", 0)), 0)
    settings["grid_step"] = float(settings.get("grid_step", 0.10))
    if settings["grid_step"] <= 0 or settings["grid_step"] > 1:
        raise ValueError("models.ensemble.grid_step must be in the range (0, 1].")
    settings["objective"] = str(settings.get("objective", "trifecta_fast")).strip()
    if settings["objective"] not in ENSEMBLE_OBJECTIVES:
        raise ValueError(f"models.ensemble.objective must be one of {sorted(ENSEMBLE_OBJECTIVES)}.")
    return settings


def get_enabled_lightgbm_variants(config: dict | None) -> list[dict[str, Any]]:
    settings = get_lightgbm_variant_settings(config)
    if not bool(settings.get("enabled", False)):
        return []
    variants = [dict(variant) for variant in settings.get("variants", [])]
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
        variant["params"] = dict(variant.get("params", {}) or {})
    return variants


def get_enabled_lightgbm_regression_variants(config: dict | None) -> list[dict[str, Any]]:
    settings = get_lightgbm_regression_variant_settings(config)
    if not bool(settings.get("enabled", False)):
        return []
    variants = [dict(variant) for variant in settings.get("variants", [])]
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
    variants = [dict(variant) for variant in settings.get("variants", [])]
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


def is_lightgbm_model_name(model_name: str) -> bool:
    return model_name == "lightgbm" or model_name.startswith("lightgbm_")


def is_lightgbm_regression_model_name(model_name: str) -> bool:
    return model_name.startswith("lightgbm_reg_")


def is_xgboost_model_name(model_name: str) -> bool:
    return model_name == "xgboost" or model_name.startswith("xgboost_")


def lightgbm_variant_model_path(base_path: Path, variant_name: str) -> Path:
    if variant_name == "lightgbm":
        return base_path
    suffix = variant_name.removeprefix("lightgbm_")
    return base_path.with_name(f"{base_path.stem}_{suffix}{base_path.suffix}")


def xgboost_variant_model_path(base_path: Path, variant_name: str) -> Path:
    if variant_name == "xgboost":
        return base_path
    suffix = variant_name.removeprefix("xgboost_")
    return base_path.with_name(f"{base_path.stem}_{suffix}{base_path.suffix}")


def get_artifact_paths(config: dict) -> dict[str, Path]:
    artifacts = config.get("artifacts", {})
    return {
        name: Path(artifacts.get(name, default_path))
        for name, default_path in DEFAULT_ARTIFACT_PATHS.items()
    }


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
    weights: dict[str, float],
) -> dict[str, Any] | None:
    if ranked.empty or not {"race_id", "lane", "finish_position", "win_probability_like"}.issubset(ranked.columns):
        return None

    lane_prob_rows: list[np.ndarray] = []
    actual_indices: list[int] = []
    scenario_labels: list[str] = []
    race_ids: list[str] = []
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
        race_ids.append(str(race_id))
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

    catboost_model = train_catboost(train_df, valid_df, feature_columns, categorical_columns, config)
    collect_garbage()
    lightgbm_model = train_lightgbm(train_df, valid_df, feature_columns, categorical_columns, config)
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
    classifier_models = train_classifiers(train_df, valid_df, feature_columns, categorical_columns, config)
    collect_garbage()
    flow_model = None
    flow_classes = None
    staged_models: dict[str, lgb.Booster] = {}

    models = {
        "catboost": catboost_model,
        "lightgbm": lightgbm_model,
        **lightgbm_variant_models,
        **lightgbm_regression_variant_models,
        **xgboost_variant_models,
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
        "catboost": evaluate_model_bundle(
            catboost_model,
            "catboost",
            train_df,
            valid_df,
            test_df,
            feature_columns,
            categorical_columns,
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

    if resume and train_stage_completed(checkpoint, "catboost", [artifacts["catboost_model_path"]]) and "catboost" in checkpoint_metrics:
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
        catboost_metrics = evaluate_model_bundle(
            catboost_model,
            "catboost",
            train_df,
            valid_df,
            test_df,
            feature_columns,
            categorical_columns,
        )
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
        lightgbm_metrics = evaluate_model_bundle(
            lightgbm_model,
            "lightgbm",
            train_df,
            valid_df,
            test_df,
            feature_columns,
            categorical_columns,
        )
        mark_train_stage_completed(checkpoint_path, checkpoint, "lightgbm", lightgbm_metrics)
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
        }
        mark_train_stage_completed(checkpoint_path, checkpoint, "xgboost_variants", xgboost_variant_metrics)
    collect_garbage()

    models = {
        "catboost": catboost_model,
        "lightgbm": lightgbm_model,
        **lightgbm_variant_models,
        **lightgbm_regression_variant_models,
        **xgboost_variant_models,
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

    if (
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
        "catboost": catboost_metrics,
        "lightgbm": lightgbm_metrics,
        **lightgbm_variant_metrics,
        **lightgbm_regression_variant_metrics,
        **xgboost_variant_metrics,
        "ensemble": ensemble_metrics,
    }
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
        frame = add_race_relative_features(drop_race_relative_features(frame))
    return frame


def _first_available_numeric_series(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    values = pd.Series(pd.NA, index=df.index, dtype="Float64")
    for column in columns:
        if column not in df.columns:
            continue
        values = values.fillna(pd.to_numeric(df[column], errors="coerce"))
    return values


def infer_feature_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if column not in DEFAULT_DROP_COLUMNS]


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
) -> lgb.Booster:
    train_lgb = build_lightgbm_frame(train_df, feature_columns, categorical_columns)
    valid_lgb = build_lightgbm_frame(valid_df, feature_columns, categorical_columns)

    train_group = train_df.groupby("race_id").size().sort_index().tolist()
    valid_group = valid_df.groupby("race_id").size().sort_index().tolist()

    train_dataset = lgb.Dataset(
        train_lgb,
        label=sort_for_grouping(train_df)["target_rank"].astype(float),
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
        _emit_progress(progress_callback, f"training lightgbm variant: {name}")
        model = train_lightgbm(
            train_df,
            valid_df,
            feature_columns,
            categorical_columns,
            config,
            param_overrides=dict(variant.get("params", {}) or {}),
            num_threads=num_threads,
        )
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
    elif is_lightgbm_model_name(model_type):
        frame = build_lightgbm_frame(df, feature_columns, categorical_columns)
        raw_scores = model.predict(frame)
        if is_lightgbm_regression_model_name(model_type):
            raw_scores = -np.asarray(raw_scores, dtype=float)
    elif is_xgboost_model_name(model_type):
        xgb_module = require_xgboost()
        frame = build_xgboost_frame(df, feature_columns, categorical_columns)
        raw_scores = model.predict(xgb_module.DMatrix(frame))
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    scored = df[["race_id", "lane"]].copy()
    if "finish_position" in df.columns:
        scored["finish_position"] = df["finish_position"].to_numpy()
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
) -> None:
    catboost_model_path.parent.mkdir(parents=True, exist_ok=True)
    lightgbm_model_path.parent.mkdir(parents=True, exist_ok=True)
    features_path.parent.mkdir(parents=True, exist_ok=True)
    ensemble_weights_path.parent.mkdir(parents=True, exist_ok=True)
    trifecta_calibrator_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    models["catboost"].save_model(catboost_model_path)
    models["lightgbm"].save_model(str(lightgbm_model_path))
    for model_name, model in models.items():
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
        elif is_lightgbm_model_name(model_type):
            frame = build_lightgbm_frame(base, feature_columns, categorical_columns)
            raw_scores = model.predict(frame)
            if is_lightgbm_regression_model_name(model_type):
                raw_scores = -np.asarray(raw_scores, dtype=float)
        elif is_xgboost_model_name(model_type):
            xgb_module = require_xgboost()
            frame = build_xgboost_frame(base, feature_columns, categorical_columns)
            raw_scores = model.predict(xgb_module.DMatrix(frame))
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
    catboost_model = CatBoostRanker()
    catboost_model.load_model(str(artifacts["catboost_model_path"]))
    lightgbm_model = lgb.Booster(model_file=str(artifacts["lightgbm_model_path"]))
    models: dict[str, Any] = {
        "catboost": catboost_model,
        "lightgbm": lightgbm_model,
    }
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


def save_xgboost_variants(models: dict[str, Any], xgboost_model_path: Path) -> None:
    for model_name, model in models.items():
        path = xgboost_variant_model_path(xgboost_model_path, model_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(path))


def enabled_lightgbm_variant_paths(config: dict, lightgbm_model_path: Path) -> list[Path]:
    return [
        lightgbm_variant_model_path(lightgbm_model_path, str(variant["name"]))
        for variant in get_enabled_lightgbm_variants(config)
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


def simplex_weight_vectors(model_count: int, steps: int = 20) -> list[tuple[float, ...]]:
    if model_count <= 0:
        return []
    if model_count == 1:
        return [(1.0,)]
    vectors: list[tuple[float, ...]] = []

    def build(prefix: list[int], remaining: int, slots: int) -> None:
        if slots == 1:
            vectors.append(tuple([*prefix, remaining]))
            return
        for value in range(remaining + 1):
            build([*prefix, value], remaining - value, slots - 1)

    build([], int(steps), model_count)
    return [tuple(float(value) / float(steps) for value in vector) for vector in vectors]


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
    if not row_groups:
        return {"race_count": 0}
    return {
        "race_count": len(row_groups),
        "row_indices": np.vstack(row_groups),
        "actual_indices": np.asarray(actual_indices, dtype=np.int16),
    }


def _softmax_rows(values: np.ndarray) -> np.ndarray:
    shifted = values - np.nanmax(values, axis=1, keepdims=True)
    exps = np.exp(shifted)
    denom = exps.sum(axis=1, keepdims=True)
    return np.divide(exps, denom, out=np.full_like(exps, 1.0 / values.shape[1]), where=denom > 0)


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
    lane_probs = _softmax_rows(combined)

    first = TRIFECTA_FAST_PERMUTATIONS[:, 0]
    second = TRIFECTA_FAST_PERMUTATIONS[:, 1]
    third = TRIFECTA_FAST_PERMUTATIONS[:, 2]
    p1 = lane_probs[:, first]
    p2_base = lane_probs[:, second]
    p3_base = lane_probs[:, third]
    denom2 = np.clip(1.0 - p1, 1e-12, None)
    denom3 = np.clip(1.0 - p1 - p2_base, 1e-12, None)
    trifecta_probs = p1 * (p2_base / denom2) * (p3_base / denom3)
    actual_probs = np.clip(trifecta_probs[np.arange(len(actual_indices)), actual_indices], 1e-12, 1.0)

    ranked_indices = np.argsort(-trifecta_probs, axis=1)
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
    log_loss = -float(np.mean(np.log(actual_probs)))
    top1_hit_rate = float(np.mean(top1_hit))
    top3_hit_rate = float(np.mean(top3_hit))
    top12_hit_rate = float(np.mean(top12_hit))
    top5_hit_rate = float(np.mean(top5_hit))
    avg_top3_overlap = float(np.mean(top3_overlap))
    normalized_log_loss = log_loss / float(np.log(120.0))
    objective_name = str(settings.get("objective", "trifecta_top12_balanced"))
    if objective_name in {"trifecta_fast", "trifecta_top12_simple"}:
        objective = (
            float(settings.get("objective_top12_weight", 0.60)) * top12_hit_rate
            + float(settings.get("objective_top5_weight", 0.30)) * top5_hit_rate
            - float(settings.get("objective_log_loss_weight", 0.10)) * normalized_log_loss
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
        "top1_hit_rate": top1_hit_rate,
        "top3_hit_rate": top3_hit_rate,
        "top12_hit_rate": top12_hit_rate,
        "top5_hit_rate": top5_hit_rate,
        "avg_top3_overlap": avg_top3_overlap,
        "log_loss": log_loss,
        "normalized_log_loss": normalized_log_loss,
        "race_count": float(len(actual_indices)),
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
    candidate_vectors = simplex_weight_vectors(len(model_names), steps=grid_steps)
    workers = min(int(settings.get("parallel_workers", 1)), len(candidate_vectors))
    _emit_progress(
        progress_callback,
        "ensemble weight search: "
        f"models={len(model_names)}, races={eval_races}, candidates={len(candidate_vectors)}, "
        f"workers={workers}, objective={objective_name}, grid_step={grid_step:.4g}",
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
    base = score_by_model[model_names[0]][["race_id", "lane", "finish_position"]].copy()
    fast_context = build_fast_trifecta_eval_context(base)
    use_fast_trifecta = (
        objective_name in {"trifecta_fast", "trifecta_top12_balanced", "trifecta_top12_simple"}
        and int(fast_context.get("race_count", 0)) > 0
    )

    def evaluate_weight_values(weight_values: tuple[float, ...]) -> tuple[float, dict[str, float], dict[str, float]]:
        if use_fast_trifecta:
            return evaluate_fast_trifecta_ensemble_candidate(
                score_arrays,
                model_names,
                weight_values,
                fast_context,
                settings,
            )
        candidate_weights = dict(zip(model_names, weight_values, strict=True))
        scored = base.copy()
        scored["score"] = np.zeros(len(base), dtype=float)
        for model_name, model_weight in candidate_weights.items():
            scored["score"] += float(model_weight) * score_arrays[model_name]
        scored["pred_rank"] = scored.groupby("race_id")["score"].rank(ascending=False, method="first")
        metrics = summarize_rank_metrics(scored)
        objective = metrics["top1_accuracy"] + 0.1 * metrics["avg_top3_overlap"]
        return float(objective), {name: float(weight) for name, weight in candidate_weights.items()}, metrics

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
        "validation_top1_hit_rate": float(best_metrics.get("top1_hit_rate", 0.0)),
        "validation_top3_hit_rate": float(best_metrics.get("top3_hit_rate", 0.0)),
        "validation_top5_hit_rate": float(best_metrics.get("top5_hit_rate", 0.0)),
        "validation_top12_hit_rate": float(best_metrics.get("top12_hit_rate", 0.0)),
        "validation_log_loss": float(best_metrics.get("log_loss", 0.0)),
        "validation_normalized_log_loss": float(best_metrics.get("normalized_log_loss", 0.0)),
        "validation_eval_races": float(eval_races),
        "validation_candidate_count": float(len(candidate_vectors)),
        "validation_parallel_workers": float(workers),
        "validation_grid_step": float(grid_step),
        "validation_objective_name": objective_name,
        "validation_fast_trifecta_races": float(fast_context.get("race_count", 0)),
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
) -> dict[str, Any]:
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
    for model_name, model in models.items():
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
        metrics[model_name] = evaluate_trifecta_v1_metrics(
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
        )
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
    metrics = compute_trifecta_metrics(trifecta, probability_col="probability")
    if rerank_top_n is not None:
        metrics["rerank_metrics"] = compute_trifecta_rerank_metrics(
            trifecta,
            probability_col="probability",
            baseline_col="probability_v1",
        )
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
