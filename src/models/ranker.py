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
    "is_top3",
    "winning_style",
}


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def train_ranker(
    training_table: pd.DataFrame,
    config: dict,
) -> tuple[dict[str, Any], list[str], dict[str, Any], IsotonicRegression]:
    data_config = config.get("data", {})
    min_date = data_config.get("min_date")
    max_date = data_config.get("max_date")
    if min_date:
        training_table = training_table[training_table["race_date"] >= pd.Timestamp(min_date)].copy()
    if max_date:
        training_table = training_table[training_table["race_date"] <= pd.Timestamp(max_date)].copy()

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
    trifecta_calibrator = fit_trifecta_calibrator(
        models,
        ensemble_weights,
        valid_df,
        feature_columns,
        categorical_columns,
    )
    metrics = {
        "catboost": evaluate_model_bundle(catboost_model, "catboost", train_df, valid_df, test_df, feature_columns, categorical_columns),
        "lightgbm": evaluate_model_bundle(lightgbm_model, "lightgbm", train_df, valid_df, test_df, feature_columns, categorical_columns),
        "ensemble": evaluate_ensemble(models, ensemble_weights, train_df, valid_df, test_df, feature_columns, categorical_columns),
        "ensemble_weights": ensemble_weights,
        "trifecta": {
            "valid_raw": evaluate_trifecta(
                models,
                ensemble_weights,
                None,
                valid_df,
                feature_columns,
                categorical_columns,
            ),
            "valid_calibrated": evaluate_trifecta(
                models,
                ensemble_weights,
                trifecta_calibrator,
                valid_df,
                feature_columns,
                categorical_columns,
            ),
            "test_raw": evaluate_trifecta(
                models,
                ensemble_weights,
                None,
                test_df,
                feature_columns,
                categorical_columns,
            ),
            "test_calibrated": evaluate_trifecta(
                models,
                ensemble_weights,
                trifecta_calibrator,
                test_df,
                feature_columns,
                categorical_columns,
            ),
        },
    }
    return models, feature_columns, metrics, trifecta_calibrator


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

    scored = df[["race_id", "lane", "finish_position"]].copy()
    scored["score_raw"] = raw_scores
    scored["score_probability_like"] = scored.groupby("race_id")["score_raw"].transform(_softmax)
    scored["pred_rank"] = scored.groupby("race_id")["score_probability_like"].rank(
        ascending=False,
        method="first",
    )
    scored["score"] = scored["score_probability_like"]
    return scored


def summarize_rank_metrics(scored: pd.DataFrame) -> dict[str, float]:
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
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


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
    return base.sort_values(["race_id", "predicted_rank", "lane"])


def predict_trifecta_probabilities(
    models: dict[str, Any],
    feature_columns: list[str],
    future_df: pd.DataFrame,
    ensemble_weights: dict[str, float] | None = None,
    trifecta_calibrator: IsotonicRegression | None = None,
) -> pd.DataFrame:
    ranked = predict_race_order(models, feature_columns, future_df, ensemble_weights)
    rows: list[pd.DataFrame] = []
    for _, race_df in ranked.groupby("race_id", sort=False):
        trifecta = enumerate_trifecta_probabilities_from_scores(race_df)
        probs = trifecta["raw_probability"].to_numpy(dtype=float)
        if trifecta_calibrator is not None:
            probs = trifecta_calibrator.predict(probs)
        prob_sum = probs.sum()
        if prob_sum <= 0:
            probs = np.full_like(probs, 1.0 / len(probs))
        else:
            probs = probs / prob_sum
        trifecta["probability"] = probs
        rows.append(trifecta.sort_values("probability", ascending=False).reset_index(drop=True))

    if not rows:
        return pd.DataFrame(columns=["race_id", "trifecta", "raw_probability", "probability"])
    return pd.concat(rows, ignore_index=True)


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
    artifacts = config["artifacts"]
    catboost_model = CatBoostRanker()
    catboost_model.load_model(artifacts["catboost_model_path"])
    lightgbm_model = lgb.Booster(model_file=artifacts["lightgbm_model_path"])
    return {
        "catboost": catboost_model,
        "lightgbm": lightgbm_model,
    }


def load_ensemble_weights(path: Path) -> dict[str, float]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_trifecta_calibrator(path: Path) -> IsotonicRegression:
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
    raw_probs: list[float] = []
    labels: list[int] = []
    for _, race_df in race_probs.groupby("race_id", sort=False):
        trifecta = enumerate_trifecta_probabilities(race_df)
        raw_probs.extend(trifecta["raw_probability"].tolist())
        labels.extend(trifecta["is_actual"].astype(int).tolist())

    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(np.asarray(raw_probs, dtype=float), np.asarray(labels, dtype=float))
    return calibrator


def evaluate_trifecta(
    models: dict[str, Any],
    weights: dict[str, float],
    calibrator: IsotonicRegression | None,
    df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
) -> dict[str, float]:
    if df.empty:
        return {}

    race_probs = build_weighted_lane_probabilities(
        models,
        weights,
        df,
        feature_columns,
        categorical_columns,
    )
    race_count = 0
    top1_hits = 0
    top5_hits = 0
    log_losses: list[float] = []

    for _, race_df in race_probs.groupby("race_id", sort=False):
        trifecta = enumerate_trifecta_probabilities(race_df)
        probs = trifecta["raw_probability"].to_numpy(dtype=float)
        if calibrator is not None:
            probs = calibrator.predict(probs)
        prob_sum = probs.sum()
        if prob_sum <= 0:
            probs = np.full_like(probs, 1.0 / len(probs))
        else:
            probs = probs / prob_sum

        trifecta = trifecta.copy()
        trifecta["probability"] = probs
        trifecta = trifecta.sort_values("probability", ascending=False).reset_index(drop=True)
        actual_idx = int(trifecta["is_actual"].to_numpy().argmax())
        actual_prob = max(float(trifecta.loc[actual_idx, "probability"]), 1e-15)
        log_losses.append(-np.log(actual_prob))
        top1_hits += int(actual_idx == 0)
        top5_hits += int(actual_idx < 5)
        race_count += 1

    return {
        "race_count": float(race_count),
        "top1_hit_rate": top1_hits / race_count if race_count else 0.0,
        "top5_hit_rate": top5_hits / race_count if race_count else 0.0,
        "log_loss": float(np.mean(log_losses)) if log_losses else 0.0,
    }


def build_weighted_lane_probabilities(
    models: dict[str, Any],
    weights: dict[str, float],
    df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
) -> pd.DataFrame:
    base = sort_for_grouping(df)[["race_id", "lane", "finish_position"]].copy()
    combined = np.zeros(len(base), dtype=float)
    for model_type, model in models.items():
        scored = score_frame(model, model_type, df, feature_columns, categorical_columns)
        combined += weights.get(model_type, 0.0) * scored["score_probability_like"].to_numpy()

    base["lane_probability"] = combined
    return base


def enumerate_trifecta_probabilities(race_df: pd.DataFrame) -> pd.DataFrame:
    race_df = race_df.sort_values("lane").reset_index(drop=True)
    lane_to_prob = {int(row.lane): float(row.lane_probability) for row in race_df.itertuples()}
    actual_order = tuple(
        race_df.sort_values("finish_position").head(3)["lane"].astype(int).tolist()
    )
    rows: list[dict[str, Any]] = []

    for trifecta in itertools.permutations(sorted(lane_to_prob.keys()), 3):
        raw_probability = plackett_luce_probability(lane_to_prob, trifecta)
        rows.append(
            {
                "race_id": race_df["race_id"].iloc[0],
                "trifecta": "-".join(str(x) for x in trifecta),
                "raw_probability": raw_probability,
                "is_actual": trifecta == actual_order,
            }
        )
    return pd.DataFrame(rows)


def enumerate_trifecta_probabilities_from_scores(race_df: pd.DataFrame) -> pd.DataFrame:
    race_df = race_df.sort_values("lane").reset_index(drop=True)
    lane_to_prob = {int(row.lane): float(row.win_probability_like) for row in race_df.itertuples()}
    rows: list[dict[str, Any]] = []

    for trifecta in itertools.permutations(sorted(lane_to_prob.keys()), 3):
        rows.append(
            {
                "race_id": race_df["race_id"].iloc[0],
                "trifecta": "-".join(str(x) for x in trifecta),
                "raw_probability": plackett_luce_probability(lane_to_prob, trifecta),
            }
        )
    return pd.DataFrame(rows)


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
