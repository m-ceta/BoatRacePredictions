from __future__ import annotations

from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_binary_classification_metrics


CLASSIFIER_TARGETS = {
    "is_win": "win_prob",
    "is_top2": "top2_prob",
    "is_top3": "top3_prob",
}


def ensure_classifier_targets(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    if "is_top2" not in frame.columns and "finish_position" in frame.columns:
        frame["is_top2"] = (pd.to_numeric(frame["finish_position"], errors="coerce") <= 2).astype(int)
    return frame


def train_classifiers(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    config: dict[str, Any],
) -> dict[str, lgb.Booster]:
    train_df = ensure_classifier_targets(train_df)
    valid_df = ensure_classifier_targets(valid_df)

    models: dict[str, lgb.Booster] = {}
    for target in CLASSIFIER_TARGETS:
        if target not in train_df.columns or target not in valid_df.columns:
            continue
        models[target] = train_binary_classifier(
            train_df,
            valid_df,
            target,
            feature_columns,
            categorical_columns,
            config,
        )
    return models


def train_binary_classifier(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    target: str,
    feature_columns: list[str],
    categorical_columns: list[str],
    config: dict[str, Any],
) -> lgb.Booster:
    train_lgb = build_lightgbm_frame(train_df, feature_columns, categorical_columns)
    valid_lgb = build_lightgbm_frame(valid_df, feature_columns, categorical_columns)

    train_dataset = lgb.Dataset(
        train_lgb,
        label=pd.to_numeric(train_df[target], errors="coerce").fillna(0).astype(int),
        categorical_feature=[c for c in categorical_columns if c in train_lgb.columns],
        free_raw_data=False,
    )
    valid_dataset = lgb.Dataset(
        valid_lgb,
        label=pd.to_numeric(valid_df[target], errors="coerce").fillna(0).astype(int),
        categorical_feature=[c for c in categorical_columns if c in valid_lgb.columns],
        reference=train_dataset,
        free_raw_data=False,
    )

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": config["model"]["learning_rate"],
        "num_leaves": 63,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbosity": -1,
        "seed": config["model"]["random_seed"],
    }
    return lgb.train(
        params,
        train_dataset,
        num_boost_round=config["model"]["iterations"],
        valid_sets=[valid_dataset],
        valid_names=[f"valid_{target}"],
        callbacks=[lgb.log_evaluation(100)],
    )


def predict_classifier_probabilities(
    models: dict[str, lgb.Booster],
    df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
) -> pd.DataFrame:
    if not models:
        return df.copy()

    frame = build_lightgbm_frame(df, feature_columns, categorical_columns)
    enriched = df.copy()
    for target, output_col in CLASSIFIER_TARGETS.items():
        model = models.get(target)
        if model is None:
            continue
        enriched[output_col] = model.predict(frame)
    return enriched


def evaluate_classifier_models(
    models: dict[str, lgb.Booster],
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
) -> dict[str, dict[str, dict[str, float]]]:
    if not models:
        return {"status": "skipped"}

    train_df = ensure_classifier_targets(train_df)
    valid_df = ensure_classifier_targets(valid_df)
    test_df = ensure_classifier_targets(test_df)
    results: dict[str, dict[str, dict[str, float]]] = {}

    for target, output_col in CLASSIFIER_TARGETS.items():
        model = models.get(target)
        if model is None:
            continue
        results[target] = {
            "train": evaluate_binary_classifier(model, train_df, target, feature_columns, categorical_columns),
            "valid": evaluate_binary_classifier(model, valid_df, target, feature_columns, categorical_columns),
            "test": evaluate_binary_classifier(model, test_df, target, feature_columns, categorical_columns),
            "output_column": {"name": output_col},
        }
    return results


def evaluate_binary_classifier(
    model: lgb.Booster,
    df: pd.DataFrame,
    target: str,
    feature_columns: list[str],
    categorical_columns: list[str],
) -> dict[str, float]:
    if df.empty or target not in df.columns:
        return {}
    frame = build_lightgbm_frame(df, feature_columns, categorical_columns)
    probabilities = model.predict(frame)
    return compute_binary_classification_metrics(df[target], probabilities)


def save_classifier_models(models: dict[str, lgb.Booster], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for target, model in models.items():
        model.save_model(str(output_dir / f"{target}_lightgbm.txt"))


def load_classifier_models(output_dir: Path) -> dict[str, lgb.Booster]:
    models: dict[str, lgb.Booster] = {}
    for target in CLASSIFIER_TARGETS:
        path = output_dir / f"{target}_lightgbm.txt"
        if path.exists():
            models[target] = lgb.Booster(model_file=str(path))
    return models


def build_lightgbm_frame(
    df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
) -> pd.DataFrame:
    data = df.sort_values(["race_id", "lane"]).reset_index(drop=True)[feature_columns].copy()
    for column in feature_columns:
        if column in categorical_columns:
            data[column] = data[column].fillna("NA").astype("category")
        else:
            data[column] = pd.to_numeric(data[column], errors="coerce").astype("float32")
    return data
