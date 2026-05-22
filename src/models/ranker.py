from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from catboost import CatBoostRanker, Pool
from sklearn.isotonic import IsotonicRegression

from src.evaluation.metrics import compute_trifecta_metrics, compute_trifecta_rerank_metrics
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
    train_flow_model,
)
from src.models.staged import (
    evaluate_staged_models,
    load_staged_models,
    predict_staged_probabilities,
    save_staged_models,
    train_staged_models,
)
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
    "features_path": "artifacts/feature_columns.json",
    "ensemble_weights_path": "artifacts/ensemble_weights.json",
    "trifecta_calibrator_path": "artifacts/trifecta_isotonic.joblib",
    "trifecta_v2_calibrator_path": "artifacts/trifecta_v2_isotonic.joblib",
    "trifecta_v3_calibrator_path": "artifacts/trifecta_v3_isotonic.joblib",
    "metrics_path": "artifacts/metrics.json",
    "classifier_dir": "artifacts/classifiers",
    "flow_model_path": "artifacts/flow_lightgbm.txt",
    "flow_meta_path": "artifacts/flow_classes.json",
    "trifecta_v2_model_path": "artifacts/trifecta_v2_model.joblib",
    "staged_dir": "artifacts/staged",
}


TRIFECTA_V2_FEATURE_VERSION = 2

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
        "default_conservative_weight": 0.95,
        "weight_grid": [0.95, 0.97, 0.98, 0.99],
        "top_n_grid": [10, 16, 24, 32],
        "rank_penalty_strength_grid": [0.0, 0.01, 0.02, 0.03],
        "default_rank_penalty_strength": 0.02,
        "rank_penalty_start": 5,
    },
    "calibration": {
        "window_days_options": [30, 60, 90],
        "default_window_days": 60,
    },
}


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def get_artifact_paths(config: dict) -> dict[str, Path]:
    artifacts = config.get("artifacts", {})
    return {
        name: Path(artifacts.get(name, default_path))
        for name, default_path in DEFAULT_ARTIFACT_PATHS.items()
    }


def get_phase3_settings(config: dict | None = None) -> dict[str, Any]:
    phase3 = (config or {}).get("phase3", {})
    settings = json.loads(json.dumps(DEFAULT_PHASE3_SETTINGS))
    for section, defaults in DEFAULT_PHASE3_SETTINGS.items():
        incoming = phase3.get(section, {})
        if isinstance(defaults, dict) and isinstance(incoming, dict):
            settings[section].update(incoming)
    return settings


def is_trifecta_v2_bundle(model: Any) -> bool:
    return isinstance(model, dict) and "model_type" in model


def save_trifecta_v2_model_artifact(model: Any, path: Path) -> None:
    if model is None:
        return
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


def with_conservative_rerank_weight(model: Any, weight: float) -> Any:
    if not is_trifecta_v2_bundle(model):
        return model
    updated = dict(model)
    updated["conservative_v1_weight"] = float(weight)
    return updated


def with_rank_penalty_settings(model: Any, strength: float, start_rank: int) -> Any:
    if not is_trifecta_v2_bundle(model):
        return model
    updated = dict(model)
    updated["rank_penalty_strength"] = float(strength)
    updated["rank_penalty_start"] = int(start_rank)
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
    base_rank = pd.Series(base).rank(ascending=False, method="average", pct=True).to_numpy(dtype=float)
    update_rank = pd.Series(update).rank(ascending=False, method="average", pct=True).to_numpy(dtype=float)
    rank_penalty_strength = float(max(rank_penalty_strength, 0.0))
    if rank_penalty_strength > 0.0 and len(base_rank_order) > 0:
        denom = float(max(len(base_rank_order) - int(rank_penalty_start), 1))
        overflow = np.clip((base_rank_order - float(rank_penalty_start)) / denom, 0.0, 1.0)
        update_rank = np.clip(update_rank - (overflow * rank_penalty_strength), 0.0, 1.0)
    return conservative_weight * base_rank + (1.0 - conservative_weight) * update_rank


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
    training_table = prepare_training_table(training_table, config)

    train_end = pd.Timestamp(config["split"]["train_end_date"])
    valid_end = pd.Timestamp(config["split"]["valid_end_date"])

    train_df = training_table[training_table["race_date"] <= train_end].copy()
    valid_df = training_table[
        (training_table["race_date"] > train_end) & (training_table["race_date"] <= valid_end)
    ].copy()
    test_df = training_table[training_table["race_date"] > valid_end].copy()

    feature_columns = infer_feature_columns(training_table)
    categorical_columns = infer_categorical_columns(training_table, feature_columns)

    catboost_model = train_catboost(train_df, valid_df, feature_columns, categorical_columns, config)
    lightgbm_model = train_lightgbm(train_df, valid_df, feature_columns, categorical_columns, config)
    classifier_models = train_classifiers(train_df, valid_df, feature_columns, categorical_columns, config)
    flow_model, flow_classes = train_flow_model(
        train_df,
        valid_df,
        feature_columns,
        categorical_columns,
        config,
    )
    staged_models = train_staged_models(train_df, valid_df, feature_columns, categorical_columns, config)

    models = {
        "catboost": catboost_model,
        "lightgbm": lightgbm_model,
    }
    ensemble_weights = optimize_ensemble_weights(
        models,
        valid_df,
        feature_columns,
        categorical_columns,
    )
    trifecta_v2_v1_weight = optimize_trifecta_v2_blend_weight(
        models,
        ensemble_weights,
        valid_df,
        feature_columns,
        categorical_columns,
        classifier_models=classifier_models,
        flow_model=flow_model,
        flow_classes=flow_classes,
        staged_models=staged_models,
    )
    ensemble_weights["trifecta_v2_v1_weight"] = trifecta_v2_v1_weight
    trifecta_v2_model = train_trifecta_v2_model(
        train_df,
        models=models,
        ensemble_weights=ensemble_weights,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        classifier_models=classifier_models,
        flow_model=flow_model,
        flow_classes=flow_classes,
        staged_models=staged_models,
        config=config,
    )
    trifecta_phase3_model = train_phase3_conditional_trifecta_model(
        train_df,
        models=models,
        ensemble_weights=ensemble_weights,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        classifier_models=classifier_models,
        flow_model=flow_model,
        flow_classes=flow_classes,
        staged_models=staged_models,
        base_model=trifecta_v2_model,
        config=config,
    )
    trifecta_calibrator = fit_trifecta_calibrator(
        models,
        ensemble_weights,
        valid_df,
        feature_columns,
        categorical_columns,
    )

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
    trifecta_v1_metrics = {
        "valid_raw": evaluate_trifecta(
            models,
            ensemble_weights,
            None,
            valid_df,
            feature_columns,
            categorical_columns,
            classifier_models=classifier_models,
            flow_model=flow_model,
            flow_classes=flow_classes,
            staged_models=staged_models,
            trifecta_v2_model=trifecta_v2_model,
            use_v2=False,
        ),
        "valid_calibrated": evaluate_trifecta(
            models,
            ensemble_weights,
            trifecta_calibrator,
            valid_df,
            feature_columns,
            categorical_columns,
            classifier_models=classifier_models,
            flow_model=flow_model,
            flow_classes=flow_classes,
            staged_models=staged_models,
            trifecta_v2_model=trifecta_v2_model,
            use_v2=False,
        ),
        "test_raw": evaluate_trifecta(
            models,
            ensemble_weights,
            None,
            test_df,
            feature_columns,
            categorical_columns,
            classifier_models=classifier_models,
            flow_model=flow_model,
            flow_classes=flow_classes,
            staged_models=staged_models,
            trifecta_v2_model=trifecta_v2_model,
            use_v2=False,
        ),
        "test_calibrated": evaluate_trifecta(
            models,
            ensemble_weights,
            trifecta_calibrator,
            test_df,
            feature_columns,
            categorical_columns,
            classifier_models=classifier_models,
            flow_model=flow_model,
            flow_classes=flow_classes,
            staged_models=staged_models,
            trifecta_v2_model=trifecta_v2_model,
            use_v2=False,
        ),
    }
    trifecta_v2_metrics = {
        "valid_raw": evaluate_trifecta(
            models,
            ensemble_weights,
            None,
            valid_df,
            feature_columns,
            categorical_columns,
            classifier_models=classifier_models,
            flow_model=flow_model,
            flow_classes=flow_classes,
            staged_models=staged_models,
            trifecta_v2_model=trifecta_v2_model,
            use_v2=True,
        ),
        "valid_calibrated": evaluate_trifecta(
            models,
            ensemble_weights,
            trifecta_calibrator,
            valid_df,
            feature_columns,
            categorical_columns,
            classifier_models=classifier_models,
            flow_model=flow_model,
            flow_classes=flow_classes,
            staged_models=staged_models,
            trifecta_v2_model=trifecta_v2_model,
            use_v2=True,
        ),
        "test_raw": evaluate_trifecta(
            models,
            ensemble_weights,
            None,
            test_df,
            feature_columns,
            categorical_columns,
            classifier_models=classifier_models,
            flow_model=flow_model,
            flow_classes=flow_classes,
            staged_models=staged_models,
            trifecta_v2_model=trifecta_v2_model,
            use_v2=True,
        ),
        "test_calibrated": evaluate_trifecta(
            models,
            ensemble_weights,
            trifecta_calibrator,
            test_df,
            feature_columns,
            categorical_columns,
            classifier_models=classifier_models,
            flow_model=flow_model,
            flow_classes=flow_classes,
            staged_models=staged_models,
            trifecta_v2_model=trifecta_v2_model,
            use_v2=True,
        ),
    }
    trifecta_v3_metrics = {
        "valid_raw": evaluate_trifecta(
            models,
            ensemble_weights,
            None,
            valid_df,
            feature_columns,
            categorical_columns,
            classifier_models=classifier_models,
            flow_model=flow_model,
            flow_classes=flow_classes,
            staged_models=staged_models,
            trifecta_v2_model=trifecta_phase3_model,
            use_v2=True,
        ),
        "valid_calibrated": evaluate_trifecta(
            models,
            ensemble_weights,
            trifecta_calibrator,
            valid_df,
            feature_columns,
            categorical_columns,
            classifier_models=classifier_models,
            flow_model=flow_model,
            flow_classes=flow_classes,
            staged_models=staged_models,
            trifecta_v2_model=trifecta_phase3_model,
            use_v2=True,
        ),
        "test_raw": evaluate_trifecta(
            models,
            ensemble_weights,
            None,
            test_df,
            feature_columns,
            categorical_columns,
            classifier_models=classifier_models,
            flow_model=flow_model,
            flow_classes=flow_classes,
            staged_models=staged_models,
            trifecta_v2_model=trifecta_phase3_model,
            use_v2=True,
        ),
        "test_calibrated": evaluate_trifecta(
            models,
            ensemble_weights,
            trifecta_calibrator,
            test_df,
            feature_columns,
            categorical_columns,
            classifier_models=classifier_models,
            flow_model=flow_model,
            flow_classes=flow_classes,
            staged_models=staged_models,
            trifecta_v2_model=trifecta_phase3_model,
            use_v2=True,
        ),
    }
    classifier_metrics = evaluate_classifier_models(
        classifier_models,
        train_df,
        valid_df,
        test_df,
        feature_columns,
        categorical_columns,
    )
    flow_metrics = evaluate_flow_model(
        flow_model,
        flow_classes,
        train_df,
        valid_df,
        test_df,
        feature_columns,
        categorical_columns,
    )
    staged_metrics = evaluate_staged_models(
        staged_models,
        train_df,
        valid_df,
        test_df,
        feature_columns,
        categorical_columns,
    )

    metrics = {
        **ranker_metrics,
        "ensemble_weights": ensemble_weights,
        "trifecta": trifecta_v1_metrics,
        "ranker_metrics": ranker_metrics,
        "trifecta_v1_metrics": trifecta_v1_metrics,
        "trifecta_v2_metrics": trifecta_v2_metrics,
        "trifecta_v3_metrics": trifecta_v3_metrics,
        "classifier_metrics": classifier_metrics,
        "flow_model_metrics": flow_metrics,
        "staged_model_metrics": staged_metrics,
        "expected_value_backtest_metrics": trifecta_v3_metrics.get("test_calibrated", {}).get(
            "expected_value_backtest",
            {},
        ),
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
        trifecta_phase3_model,
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

    model = CatBoostRanker(
        iterations=config["model"]["iterations"],
        learning_rate=config["model"]["learning_rate"],
        depth=config["model"]["depth"],
        loss_function=config["model"]["loss_function"],
        eval_metric=config["model"]["eval_metric"],
        random_seed=config["model"]["random_seed"],
        verbose=100,
    )
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    return model


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
    return lgb.train(
        params,
        train_dataset,
        num_boost_round=config["model"]["iterations"],
        valid_sets=[valid_dataset],
        valid_names=["valid"],
        callbacks=[lgb.log_evaluation(100)],
    )


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
    elif model_type == "lightgbm":
        frame = build_lightgbm_frame(df, feature_columns, categorical_columns)
        raw_scores = model.predict(frame)
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
) -> None:
    catboost_model_path.parent.mkdir(parents=True, exist_ok=True)
    lightgbm_model_path.parent.mkdir(parents=True, exist_ok=True)
    features_path.parent.mkdir(parents=True, exist_ok=True)
    ensemble_weights_path.parent.mkdir(parents=True, exist_ok=True)
    trifecta_calibrator_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    models["catboost"].save_model(catboost_model_path)
    models["lightgbm"].save_model(str(lightgbm_model_path))
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
        elif model_type == "lightgbm":
            frame = build_lightgbm_frame(base, feature_columns, categorical_columns)
            raw_scores = model.predict(frame)
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
    use_v2: bool = True,
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
    use_v2: bool = True,
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

        v2 = enumerate_trifecta_probabilities_v2(race_df)
        v2 = v2.rename(columns={"raw_probability": "raw_probability_v2"})
        candidate_mask = None
        if rerank_top_n is not None and rerank_top_n > 0:
            candidate_mask = select_rerank_candidate_mask_from_v1(v1, top_n=rerank_top_n)
            if use_v2:
                v1 = v1.loc[candidate_mask].reset_index(drop=True)
                v2 = v2.loc[candidate_mask].reset_index(drop=True)
        v2["raw_probability_v2"] = blend_trifecta_raw_probabilities(
            v1["raw_probability_v1"].to_numpy(dtype=float),
            v2["raw_probability_v2"].to_numpy(dtype=float),
            trifecta_v2_v1_weight,
        )
        if trifecta_v2_model is not None:
            v2_features = build_trifecta_feature_frame(race_df, v1, v2)
            rerank_scores = predict_trifecta_v2_scores(trifecta_v2_model, v2_features)
            if is_trifecta_v2_bundle(trifecta_v2_model) and trifecta_v2_model.get("phase") == "phase3_conditional":
                rerank_scores = apply_phase3_conditional_scores(
                    race_df=race_df,
                    trifecta_df=v2,
                    base_scores=rerank_scores,
                    model_bundle=trifecta_v2_model,
                )
            v2["raw_probability_v2"] = blend_conservative_rerank_scores(
                v1["raw_probability_v1"].to_numpy(dtype=float),
                rerank_scores,
                get_conservative_rerank_weight(trifecta_v2_model),
                get_rank_penalty_strength(trifecta_v2_model),
                get_rank_penalty_start(trifecta_v2_model),
            )
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
            merged = restrict_trifecta_candidates_for_rerank(merged, top_n=rerank_top_n)
        merged["probability"] = merged["probability_v2"] if use_v2 else merged["probability_v1"]
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
            ]
        )

    trifecta = pd.concat(rows, ignore_index=True)
    if odds_df is not None:
        trifecta = merge_odds_into_trifecta(trifecta, odds_df)
        trifecta = attach_expected_value_columns(trifecta, probability_col="probability", odds_col="odds")
    return trifecta.sort_values(["race_id", "probability"], ascending=[True, False]).reset_index(drop=True)


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
    actual = trifecta_df["is_actual"].astype(bool) if "is_actual" in trifecta_df.columns else pd.Series(False, index=trifecta_df.index)
    restricted = trifecta_df.loc[ordered | actual].copy()
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
    actual = v1_df["is_actual"].astype(bool) if "is_actual" in v1_df.columns else pd.Series(False, index=v1_df.index)
    return (ordered | actual).astype(bool)


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
    return {
        "catboost": catboost_model,
        "lightgbm": lightgbm_model,
    }


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


def optimize_ensemble_weights(
    models: dict[str, Any],
    valid_df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
) -> dict[str, float]:
    cat_scores = score_frame(models["catboost"], "catboost", valid_df, feature_columns, categorical_columns)
    lgb_scores = score_frame(models["lightgbm"], "lightgbm", valid_df, feature_columns, categorical_columns)

    best_weight = 0.5
    best_metrics: dict[str, float] | None = None
    best_objective = float("-inf")
    base = cat_scores[["race_id", "lane", "finish_position"]].copy()

    for cat_weight in np.linspace(0.0, 1.0, 21):
        lgb_weight = 1.0 - cat_weight
        scored = base.copy()
        scored["score"] = (
            cat_weight * cat_scores["score_probability_like"].to_numpy()
            + lgb_weight * lgb_scores["score_probability_like"].to_numpy()
        )
        scored["pred_rank"] = scored.groupby("race_id")["score"].rank(ascending=False, method="first")
        metrics = summarize_rank_metrics(scored)
        objective = metrics["top1_accuracy"] + 0.1 * metrics["avg_top3_overlap"]
        if objective > best_objective:
            best_objective = objective
            best_weight = float(cat_weight)
            best_metrics = metrics

    return {
        "catboost": best_weight,
        "lightgbm": 1.0 - best_weight,
        "validation_objective": best_objective,
        "validation_top1_accuracy": 0.0 if best_metrics is None else best_metrics["top1_accuracy"],
        "validation_avg_top3_overlap": 0.0 if best_metrics is None else best_metrics["avg_top3_overlap"],
    }


def fit_trifecta_calibrator(
    models: dict[str, Any],
    weights: dict[str, float],
    valid_df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
) -> IsotonicRegression:
    race_probs = build_weighted_lane_probabilities(
        models,
        weights,
        valid_df,
        feature_columns,
        categorical_columns,
    )
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
    use_v2: bool = True,
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
) -> dict[str, float]:
    if valid_df.empty or trifecta_v2_model is None or not is_trifecta_v2_bundle(trifecta_v2_model):
        return {
            "best_top_n": 10.0,
            "best_conservative_weight": get_conservative_rerank_weight(trifecta_v2_model),
            "best_rank_penalty_strength": get_rank_penalty_strength(trifecta_v2_model),
            "objective": 0.0,
        }

    top_n_grid = top_n_candidates or list(DEFAULT_PHASE3_SETTINGS["rerank"]["top_n_grid"])
    weight_grid = conservative_weights or list(DEFAULT_PHASE3_SETTINGS["rerank"]["weight_grid"])
    penalty_grid = rank_penalty_strengths or list(DEFAULT_PHASE3_SETTINGS["rerank"]["rank_penalty_strength_grid"])
    best: dict[str, float] = {
        "best_top_n": float(top_n_grid[0]),
        "best_conservative_weight": float(weight_grid[0]),
        "best_rank_penalty_strength": float(penalty_grid[0]),
        "objective": float("-inf"),
    }
    for top_n in top_n_grid:
        for conservative_weight in weight_grid:
            for rank_penalty_strength in penalty_grid:
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
                    rerank_top_n=int(top_n),
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
                    rerank_top_n=int(top_n),
                )
                objective = (
                    0.65 * float(metrics.get("top5_hit_rate", 0.0))
                    + 0.20 * float(metrics.get("top3_hit_rate", 0.0))
                    + 0.10 * float(metrics.get("top1_hit_rate", 0.0))
                    - 0.015 * float(metrics.get("log_loss", 0.0))
                )
                if objective > best["objective"]:
                    best = {
                        "best_top_n": float(top_n),
                        "best_conservative_weight": float(conservative_weight),
                        "best_rank_penalty_strength": float(rank_penalty_strength),
                        "objective": float(objective),
                        "top1_hit_rate": float(metrics.get("top1_hit_rate", 0.0)),
                        "top3_hit_rate": float(metrics.get("top3_hit_rate", 0.0)),
                        "top5_hit_rate": float(metrics.get("top5_hit_rate", 0.0)),
                        "top10_hit_rate": float(metrics.get("top10_hit_rate", 0.0)),
                        "log_loss": float(metrics.get("log_loss", 0.0)),
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
) -> dict[str, float]:
    if valid_df.empty:
        return {"best_window_days": float(DEFAULT_PHASE3_SETTINGS["calibration"]["default_window_days"]), "objective": 0.0}
    unique_dates = sorted(pd.to_datetime(valid_df["race_date"]).dropna().dt.normalize().unique().tolist())
    if len(unique_dates) < 4:
        default_window = int(DEFAULT_PHASE3_SETTINGS["calibration"]["default_window_days"])
        return {"best_window_days": float(default_window), "objective": 0.0}
    split_index = max(int(len(unique_dates) * 0.7), 1)
    split_date = pd.Timestamp(unique_dates[min(split_index - 1, len(unique_dates) - 1)])
    calibration_source = valid_df[pd.to_datetime(valid_df["race_date"]) <= split_date].copy()
    eval_df = valid_df[pd.to_datetime(valid_df["race_date"]) > split_date].copy()
    if calibration_source.empty or eval_df.empty:
        default_window = int(DEFAULT_PHASE3_SETTINGS["calibration"]["default_window_days"])
        return {"best_window_days": float(default_window), "objective": 0.0}
    best = {
        "best_window_days": float(DEFAULT_PHASE3_SETTINGS["calibration"]["default_window_days"]),
        "objective": float("-inf"),
    }
    for window_days in (window_days_options or list(DEFAULT_PHASE3_SETTINGS["calibration"]["window_days_options"])):
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
        objective = (
            0.75 * float(metrics.get("top5_hit_rate", 0.0))
            + 0.15 * float(metrics.get("top3_hit_rate", 0.0))
            - 0.02 * float(metrics.get("log_loss", 0.0))
        )
        if objective > best["objective"]:
            best = {
                "best_window_days": float(window_days),
                "objective": float(objective),
                "top3_hit_rate": float(metrics.get("top3_hit_rate", 0.0)),
                "top5_hit_rate": float(metrics.get("top5_hit_rate", 0.0)),
                "log_loss": float(metrics.get("log_loss", 0.0)),
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
    return metrics


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
) -> float:
    if valid_df.empty or not classifier_models:
        return 1.0

    ranked = build_weighted_lane_probabilities(
        models,
        weights,
        valid_df,
        feature_columns,
        categorical_columns,
        classifier_models=classifier_models,
        flow_model=flow_model,
        flow_classes=flow_classes,
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
        top5 = 0
        log_losses: list[float] = []
        for raw_v1, raw_v2, actual_idx in race_payloads:
            blended = blend_trifecta_raw_probabilities(raw_v1, raw_v2, float(weight))
            prob_sum = blended.sum()
            probs = blended / prob_sum if prob_sum > 0 else np.full_like(blended, 1.0 / len(blended))
            order = np.argsort(-probs)
            rank = int(np.where(order == actual_idx)[0][0])
            actual_prob = max(float(probs[actual_idx]), 1e-15)
            top1 += int(rank == 0)
            top5 += int(rank < 5)
            log_losses.append(-np.log(actual_prob))

        race_count = len(race_payloads)
        objective = (top1 / race_count) + 0.1 * (top5 / race_count) - 0.05 * float(np.mean(log_losses))
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
    max_races: int = 1000,
    top_n_candidates: int = 24,
    config: dict | None = None,
) -> Any | None:
    if train_df.empty or not classifier_models:
        return None

    race_ids = train_df["race_id"].drop_duplicates().tolist()
    if len(race_ids) > max_races:
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
    booster = lgb.train(params, train_set, num_boost_round=200, callbacks=[lgb.log_evaluation(100)])
    return {
        "model_type": "lgbm_ranker",
        "phase": "phase2_reranker",
        "feature_version": TRIFECTA_V2_FEATURE_VERSION,
        "feature_names": list(x_train.columns),
        "booster": booster,
        "conservative_v1_weight": float(phase3_settings["rerank"]["default_conservative_weight"]),
        "rank_penalty_strength": float(phase3_settings["rerank"]["default_rank_penalty_strength"]),
        "rank_penalty_start": int(phase3_settings["rerank"]["rank_penalty_start"]),
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
    max_races: int = 1000,
    config: dict | None = None,
) -> Any | None:
    if base_model is None or train_df.empty or not classifier_models:
        return base_model

    race_ids = train_df["race_id"].drop_duplicates().tolist()
    if len(race_ids) > max_races:
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

    second_rows: list[pd.DataFrame] = []
    second_labels: list[np.ndarray] = []
    third_rows: list[pd.DataFrame] = []
    third_labels: list[np.ndarray] = []
    for _, race_df in ranked.groupby("race_id", sort=False):
        actual_order = actual_trifecta_order(race_df)
        if actual_order is None:
            continue
        second_frame = build_phase3_second_feature_frame(race_df, actual_order[0])
        if not second_frame.empty:
            second_rows.append(second_frame)
            second_labels.append((second_frame["second_lane"].astype(int) == int(actual_order[1])).astype(int).to_numpy())

        third_frame = build_phase3_third_feature_frame(race_df, actual_order[0], actual_order[1])
        if not third_frame.empty:
            third_rows.append(third_frame)
            third_labels.append((third_frame["third_lane"].astype(int) == int(actual_order[2])).astype(int).to_numpy())

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
    second_model = lgb.train(params, second_set, num_boost_round=boost_round, callbacks=[lgb.log_evaluation(100)])
    third_model = lgb.train(params, third_set, num_boost_round=boost_round, callbacks=[lgb.log_evaluation(100)])

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
    bundle["conservative_v1_weight"] = float(phase3_settings["rerank"]["default_conservative_weight"])
    bundle["rank_penalty_strength"] = float(phase3_settings["rerank"]["default_rank_penalty_strength"])
    bundle["rank_penalty_start"] = int(phase3_settings["rerank"]["rank_penalty_start"])
    return bundle


def build_trifecta_feature_frame(
    race_df: pd.DataFrame,
    v1_df: pd.DataFrame,
    v2_df: pd.DataFrame,
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

    v1_map = dict(zip(v1_df["trifecta"], v1_df[v1_col]))
    v2_map = dict(zip(v2_df["trifecta"], v2_df[v2_col]))
    rows: list[dict[str, float]] = []
    for trifecta in v1_df["trifecta"].tolist():
        first, second, third = [int(x) for x in trifecta.split("-")]
        first_row = lane_frame.loc[first]
        second_row = lane_frame.loc[second]
        third_row = lane_frame.loc[third]
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
            }
        )
    frame = pd.DataFrame(rows)
    for col in frame.columns:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0).astype(float)
    return frame


def build_phase3_second_feature_frame(race_df: pd.DataFrame, first_lane: int) -> pd.DataFrame:
    lane_frame = race_df.set_index("lane").copy()
    if first_lane not in lane_frame.index:
        return pd.DataFrame()
    first_row = lane_frame.loc[first_lane]
    rows: list[dict[str, float]] = []
    for second_lane in [int(lane) for lane in lane_frame.index if int(lane) != int(first_lane)]:
        second_row = lane_frame.loc[second_lane]
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
                "inside_follow_alignment": float((int(first_lane) <= 2 and int(second_lane) <= 4) * second_course_top3),
                "outside_chase_alignment": float((int(first_lane) >= 3 and int(second_lane) >= 4) * second_top2_prob),
            }
        )
    return pd.DataFrame(rows)


def build_phase3_third_feature_frame(race_df: pd.DataFrame, first_lane: int, second_lane: int) -> pd.DataFrame:
    lane_frame = race_df.set_index("lane").copy()
    if first_lane not in lane_frame.index or second_lane not in lane_frame.index:
        return pd.DataFrame()
    first_row = lane_frame.loc[first_lane]
    second_row = lane_frame.loc[second_lane]
    rows: list[dict[str, float]] = []
    excluded = {int(first_lane), int(second_lane)}
    for third_lane in [int(lane) for lane in lane_frame.index if int(lane) not in excluded]:
        third_row = lane_frame.loc[third_lane]
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
                "top3_mass_first_second": first_top3_prob + second_top3_prob,
                "third_residual_top3": max(third_top3_prob - second_top2_prob, 0.0),
                "third_inside_scrap_alignment": float((int(third_lane) <= 4) * third_course_top3),
                "third_outer_scrap_alignment": float((int(third_lane) >= 4) * third_top3_prob),
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
            second_cache[first_lane] = build_phase3_second_feature_frame(race_df, first_lane)
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
            third_cache[pair_key] = build_phase3_third_feature_frame(race_df, first_lane, second_lane)
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
