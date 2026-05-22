from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import pandas as pd

from src.evaluation.metrics import compute_binary_classification_metrics


STAGED_TARGETS = {
    "exact_first": ("finish_position", 1, "exact1_prob"),
    "exact_second": ("finish_position", 2, "exact2_prob"),
    "exact_third": ("finish_position", 3, "exact3_prob"),
}


def train_staged_models(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    config: dict,
) -> dict[str, lgb.Booster]:
    models: dict[str, lgb.Booster] = {}
    for name, (source_col, rank_value, _) in STAGED_TARGETS.items():
        if source_col not in train_df.columns or source_col not in valid_df.columns:
            continue
        models[name] = train_binary_finish_model(
            train_df,
            valid_df,
            source_col,
            rank_value,
            feature_columns,
            categorical_columns,
            config,
        )
    return models


def train_binary_finish_model(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    source_col: str,
    rank_value: int,
    feature_columns: list[str],
    categorical_columns: list[str],
    config: dict,
) -> lgb.Booster:
    train_lgb = build_lightgbm_frame(train_df, feature_columns, categorical_columns)
    valid_lgb = build_lightgbm_frame(valid_df, feature_columns, categorical_columns)
    train_label = (pd.to_numeric(train_df[source_col], errors="coerce") == rank_value).astype(int)
    valid_label = (pd.to_numeric(valid_df[source_col], errors="coerce") == rank_value).astype(int)

    train_dataset = lgb.Dataset(
        train_lgb,
        label=train_label,
        categorical_feature=[c for c in categorical_columns if c in train_lgb.columns],
        free_raw_data=False,
    )
    valid_dataset = lgb.Dataset(
        valid_lgb,
        label=valid_label,
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
        valid_names=[f"valid_{source_col}_{rank_value}"],
        callbacks=[lgb.log_evaluation(100)],
    )


def predict_staged_probabilities(
    models: dict[str, lgb.Booster],
    df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
) -> pd.DataFrame:
    if not models:
        return df.copy()
    frame = build_lightgbm_frame(df, feature_columns, categorical_columns)
    enriched = df.copy()
    for name, (_, _, output_col) in STAGED_TARGETS.items():
        model = models.get(name)
        if model is None:
            continue
        enriched[output_col] = model.predict(frame)
    return enriched


def evaluate_staged_models(
    models: dict[str, lgb.Booster],
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
) -> dict[str, dict[str, dict[str, float]]]:
    if not models:
        return {"status": "skipped"}
    metrics: dict[str, dict[str, dict[str, float]]] = {}
    for name, (source_col, rank_value, output_col) in STAGED_TARGETS.items():
        model = models.get(name)
        if model is None:
            continue
        metrics[name] = {
            "train": evaluate_single_staged_model(model, train_df, source_col, rank_value, feature_columns, categorical_columns),
            "valid": evaluate_single_staged_model(model, valid_df, source_col, rank_value, feature_columns, categorical_columns),
            "test": evaluate_single_staged_model(model, test_df, source_col, rank_value, feature_columns, categorical_columns),
            "output_column": {"name": output_col},
        }
    return metrics


def evaluate_single_staged_model(
    model: lgb.Booster,
    df: pd.DataFrame,
    source_col: str,
    rank_value: int,
    feature_columns: list[str],
    categorical_columns: list[str],
) -> dict[str, float]:
    if df.empty or source_col not in df.columns:
        return {}
    frame = build_lightgbm_frame(df, feature_columns, categorical_columns)
    y_true = (pd.to_numeric(df[source_col], errors="coerce") == rank_value).astype(int)
    y_prob = model.predict(frame)
    return compute_binary_classification_metrics(y_true, y_prob)


def save_staged_models(models: dict[str, lgb.Booster], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        model.save_model(str(output_dir / f"{name}_lightgbm.txt"))


def load_staged_models(output_dir: Path) -> dict[str, lgb.Booster]:
    models: dict[str, lgb.Booster] = {}
    for name in STAGED_TARGETS:
        path = output_dir / f"{name}_lightgbm.txt"
        if path.exists():
            models[name] = lgb.Booster(model_file=str(path))
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
