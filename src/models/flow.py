from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_multiclass_classification_metrics
from src.models.training_device import train_lightgbm_with_optional_gpu


FLOW_STYLE_MAP = {
    "逃げ": "nige",
    "差し": "sashi",
    "まくり": "makuri",
    "まくり差し": "makurizashi",
    "抜き": "nuki",
    "恵まれ": "megumare",
}

FLOW_CLASSES = ["nige", "sashi", "makuri", "makurizashi", "nuki", "megumare", "other"]


def normalize_winning_style(value: str | None) -> str | None:
    if value is None or pd.isna(value):
        return None
    cleaned = str(value).strip()
    if not cleaned:
        return None
    return FLOW_STYLE_MAP.get(cleaned, "other")


def train_flow_model(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    config: dict[str, Any],
    min_samples: int = 100,
) -> tuple[lgb.Booster | None, list[str] | None]:
    prepared_train = prepare_flow_training_frame(train_df)
    prepared_valid = prepare_flow_training_frame(valid_df)
    if len(prepared_train) < min_samples or prepared_train["flow_target"].nunique() < 2:
        return None, None
    if prepared_valid.empty:
        return None, None

    train_lgb = build_lightgbm_frame(prepared_train, feature_columns, categorical_columns)
    valid_lgb = build_lightgbm_frame(prepared_valid, feature_columns, categorical_columns)
    train_labels = pd.Categorical(prepared_train["flow_target"], categories=FLOW_CLASSES).codes
    valid_labels = pd.Categorical(prepared_valid["flow_target"], categories=FLOW_CLASSES).codes

    train_mask = train_labels >= 0
    valid_mask = valid_labels >= 0
    if train_mask.sum() < min_samples or valid_mask.sum() == 0:
        return None, None

    train_dataset = lgb.Dataset(
        train_lgb.loc[train_mask],
        label=train_labels[train_mask],
        categorical_feature=[c for c in categorical_columns if c in train_lgb.columns],
        free_raw_data=False,
    )
    valid_dataset = lgb.Dataset(
        valid_lgb.loc[valid_mask],
        label=valid_labels[valid_mask],
        categorical_feature=[c for c in categorical_columns if c in valid_lgb.columns],
        reference=train_dataset,
        free_raw_data=False,
    )

    params = {
        "objective": "multiclass",
        "metric": "multi_logloss",
        "num_class": len(FLOW_CLASSES),
        "learning_rate": config["model"]["learning_rate"],
        "num_leaves": 63,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbosity": -1,
        "seed": config["model"]["random_seed"],
    }
    model = train_lightgbm_with_optional_gpu(
        params,
        train_dataset,
        config,
        num_boost_round=config["model"]["iterations"],
        valid_sets=[valid_dataset],
        valid_names=["valid_flow"],
        callbacks=[lgb.log_evaluation(100)],
    )
    return model, FLOW_CLASSES.copy()


def predict_flow_probabilities(
    model: lgb.Booster | None,
    classes: list[str] | None,
    df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
) -> pd.DataFrame:
    enriched = df.copy()
    if model is None or not classes:
        return enriched

    frame = build_lightgbm_frame(df, feature_columns, categorical_columns)
    probabilities = np.asarray(model.predict(frame), dtype=float)
    for idx, class_name in enumerate(classes):
        enriched[f"flow_prob_{class_name}"] = probabilities[:, idx]
    return enriched


def evaluate_flow_model(
    model: lgb.Booster | None,
    classes: list[str] | None,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
) -> dict[str, Any]:
    if model is None or not classes:
        return {"status": "skipped"}
    return {
        "train": _evaluate_flow_split(model, classes, train_df, feature_columns, categorical_columns),
        "valid": _evaluate_flow_split(model, classes, valid_df, feature_columns, categorical_columns),
        "test": _evaluate_flow_split(model, classes, test_df, feature_columns, categorical_columns),
    }


def _evaluate_flow_split(
    model: lgb.Booster,
    classes: list[str],
    df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
) -> dict[str, float]:
    prepared = prepare_flow_training_frame(df)
    if prepared.empty:
        return {}
    frame = build_lightgbm_frame(prepared, feature_columns, categorical_columns)
    probabilities = np.asarray(model.predict(frame), dtype=float)
    return compute_multiclass_classification_metrics(prepared["flow_target"], probabilities, classes)


def save_flow_model(model: lgb.Booster | None, classes: list[str] | None, model_path: Path, meta_path: Path) -> None:
    if model is None or not classes:
        return
    model_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))
    meta_path.write_text(json.dumps({"classes": classes}, ensure_ascii=False, indent=2), encoding="utf-8")


def load_flow_model(model_path: Path, meta_path: Path) -> tuple[lgb.Booster | None, list[str] | None]:
    if not model_path.exists() or not meta_path.exists():
        return None, None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return lgb.Booster(model_file=str(model_path)), meta.get("classes")


def prepare_flow_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame["flow_target"] = frame["winning_style"].apply(normalize_winning_style) if "winning_style" in frame.columns else None
    return frame.dropna(subset=["flow_target"]).copy()


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
