"""Chronological, out-of-fold ST experiment, isolated from production models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd


ST_INPUT_COLUMNS = (
    "venue", "lane", "course", "age",
    "racer_prev_avg_st", "racer_prev_avg_st_5", "racer_prev_avg_st_10",
    "racer_prev_avg_st_30", "racer_prev_std_st_10",
    "racer_prev_best_st_5", "racer_prev_best_st_10", "racer_prev_best_st_30",
    "racer_prev_worst_st_5", "racer_prev_worst_st_10", "racer_prev_worst_st_30",
    "racer_venue_prev_avg_st", "racer_lane_prev_avg_st",
    "racer_venue_lane_prev_avg_st", "st_momentum_diff",
)
ST_FEATURE_COLUMNS = (
    "predicted_st", "predicted_st_proxy_diff", "predicted_st_race_rank",
    "predicted_st_diff_mean", "predicted_st_race_std",
    "predicted_st_inside_diff", "predicted_st_outside_diff",
)
DEFAULT_SETTINGS = {
    "enabled": False,
    "oof_folds": 3,
    "evaluation_fraction": 0.20,
    "max_train_races": 30000,
    "max_eval_races": 3000,
    "iterations": 80,
    "ranker_iterations": 80,
    "num_threads": 2,
    "min_training_rows": 120,
    "random_seed": 42,
}


def st_input_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    # Explicit allowlist: no current ST, result, payout, or result-derived features.
    if not set(columns).issubset(ST_INPUT_COLUMNS):
        raise ValueError("ST model input contains a non-pre-race feature")
    return frame.reindex(columns=columns).apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    ).astype("float32")


def normal_st_targets(frame: pd.DataFrame) -> np.ndarray:
    values = pd.to_numeric(frame["_actual_st"], errors="coerce").to_numpy(
        dtype=float, na_value=np.nan
    )
    valid = np.isfinite(values) & (values >= 0.0) & (values <= 1.0)
    if "finish_status" in frame:
        status = frame["finish_status"].fillna("").astype(str).str.normalize("NFKC").str.strip().str.upper()
        valid &= ~status.str.contains(r"F|L|フライング|出遅", regex=True).to_numpy()
    return np.where(valid, values, np.nan)


def fit_st_model(
    frame: pd.DataFrame, columns: list[str], settings: dict[str, Any]
) -> lgb.LGBMRegressor | None:
    targets = normal_st_targets(frame)
    usable = np.isfinite(targets)
    if int(usable.sum()) < int(settings["min_training_rows"]):
        return None
    model = lgb.LGBMRegressor(
        objective="regression_l1", n_estimators=int(settings["iterations"]),
        learning_rate=0.05, num_leaves=15, max_depth=4, min_child_samples=80,
        reg_lambda=5.0, n_jobs=int(settings["num_threads"]),
        random_state=int(settings["random_seed"]), verbosity=-1,
    )
    model.fit(st_input_frame(frame.loc[usable], columns), targets[usable])
    return model


def chronological_oof_st(
    frame: pd.DataFrame,
    columns: list[str],
    settings: dict[str, Any],
    progress: Callable[[str], None],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    dates = pd.to_datetime(frame["race_date"]).dt.normalize()
    blocks = np.array_split(np.sort(dates.unique()), int(settings["oof_folds"]) + 1)
    predictions = np.full(len(frame), np.nan)
    folds = []
    for number, block in enumerate(blocks[1:], 1):
        if not len(block):
            continue
        fit_mask = (dates < block[0]).to_numpy()
        predict_mask = dates.isin(block).to_numpy()
        model = fit_st_model(frame.loc[fit_mask], columns, settings)
        if model is None:
            continue
        predictions[predict_mask] = model.predict(st_input_frame(frame.loc[predict_mask], columns))
        fold = {
            "fold": number,
            "train_max_date": dates.loc[fit_mask].max().strftime("%Y-%m-%d"),
            "predict_min_date": pd.Timestamp(block[0]).strftime("%Y-%m-%d"),
            "predict_max_date": pd.Timestamp(block[-1]).strftime("%Y-%m-%d"),
            "train_rows": int(fit_mask.sum()), "prediction_rows": int(predict_mask.sum()),
        }
        folds.append(fold)
        progress(f"ST OOF fold {number}: train_rows={fit_mask.sum()}, prediction_rows={predict_mask.sum()}")
    return predictions, folds


def add_predicted_st_features(frame: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    result = frame.copy()
    result["predicted_st"] = predictions
    result["predicted_st_proxy_diff"] = result["predicted_st"] - pd.to_numeric(
        result["start_timing"], errors="coerce"
    )
    group = result.groupby("race_id", sort=False)["predicted_st"]
    result["predicted_st_race_rank"] = group.rank(method="average")
    result["predicted_st_diff_mean"] = result["predicted_st"] - group.transform("mean")
    result["predicted_st_race_std"] = group.transform("std")
    course = pd.to_numeric(result["course"], errors="coerce")
    lookup = pd.Series(
        predictions,
        index=pd.MultiIndex.from_arrays([result["race_id"], course]),
    )
    if not lookup.index.is_unique:
        raise ValueError("ST features require unique actual courses within each race")
    for name, offset in (("inside", -1), ("outside", 1)):
        adjacent = lookup.reindex(pd.MultiIndex.from_arrays([result["race_id"], course + offset]))
        result[f"predicted_st_{name}_diff"] = predictions - adjacent.to_numpy()
    return result


def _sample_races(frame: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    races = frame["race_id"].drop_duplicates()
    if limit > 0 and len(races) > limit:
        races = races.sample(n=limit, random_state=seed)
        frame = frame[frame["race_id"].isin(races)]
    return frame.copy().reset_index(drop=True)


def _complete_races(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    courses = pd.to_numeric(frame["course"], errors="coerce")
    lanes = pd.to_numeric(frame["lane"], errors="coerce")
    finishes = pd.to_numeric(frame["finish_position"], errors="coerce")
    checked = frame.assign(_course=courses, _lane=lanes, _finish=finishes)
    groups = checked.groupby("race_id", sort=False)
    valid = groups.size().eq(6)
    for column in ("_course", "_lane", "_finish"):
        valid &= groups[column].nunique().eq(6)
        valid &= groups[column].min().eq(1) & groups[column].max().eq(6)
    # Strictly keep unambiguous 1..6 courses/finishes; no ties or accident ranks.
    integers = ((courses % 1 == 0) & (lanes % 1 == 0) & (finishes % 1 == 0))
    valid &= integers.groupby(frame["race_id"]).all()
    if "finish_status" in frame:
        accident = frame["finish_status"].fillna("").astype(str).str.strip().ne("")
        valid &= ~accident.groupby(frame["race_id"]).any()
    return frame[frame["race_id"].isin(valid[valid].index)].copy()


def _st_errors(frame: pd.DataFrame, predictions: np.ndarray) -> dict[str, Any]:
    actual = normal_st_targets(frame)
    proxy = pd.to_numeric(frame["start_timing"], errors="coerce").to_numpy(
        dtype=float, na_value=np.nan
    )
    mask = np.isfinite(actual) & np.isfinite(proxy) & np.isfinite(predictions)
    if not mask.any():
        return {"sample_count": 0}
    proxy_error, model_error = proxy[mask] - actual[mask], predictions[mask] - actual[mask]
    result = {
        "sample_count": int(mask.sum()), "excluded_rows": int((~mask).sum()),
        "proxy_mae_seconds": float(np.abs(proxy_error).mean()),
        "model_mae_seconds": float(np.abs(model_error).mean()),
        "mae_improvement_seconds": float(np.abs(proxy_error).mean() - np.abs(model_error).mean()),
        "proxy_rmse_seconds": float(np.sqrt(np.mean(proxy_error ** 2))),
        "model_rmse_seconds": float(np.sqrt(np.mean(model_error ** 2))),
    }
    adjacent = pd.DataFrame({
        "race_id": frame["race_id"].to_numpy(), "course": frame["course"].to_numpy(),
        "actual": actual, "proxy": proxy, "model": predictions,
    }).sort_values(["race_id", "course"])
    differences = adjacent.groupby("race_id")[["actual", "proxy", "model"]].diff()
    comparable = differences.notna().all(axis=1) & differences["actual"].abs().ge(0.01)
    result["adjacent_comparison_count"] = int(comparable.sum())
    if comparable.any():
        for name in ("proxy", "model"):
            result[f"{name}_adjacent_advantage_accuracy"] = float((
                np.sign(differences.loc[comparable, name]) == np.sign(differences.loc[comparable, "actual"])
            ).mean())
    return result


def _rank_metrics(model: Any, frame: pd.DataFrame, features: list[str], categories: list[str]) -> dict:
    from src.models.ranker import (
        TRIFECTA_FAST_PERMUTATIONS, _fast_trifecta_probs_from_lane_scores,
        build_fast_trifecta_eval_context, build_lightgbm_frame, sort_for_grouping,
    )

    ordered = sort_for_grouping(frame).reset_index(drop=True)
    context = build_fast_trifecta_eval_context(ordered)
    scores = model.predict(build_lightgbm_frame(ordered, features, categories))
    scores = np.asarray(scores)[context["row_indices"]]
    probs = _fast_trifecta_probs_from_lane_scores(scores)
    order = np.argsort(-probs, axis=1, kind="stable")
    actual = context["actual_indices"]
    ranks = np.argmax(order == actual[:, None], axis=1) + 1
    winner = TRIFECTA_FAST_PERMUTATIONS[actual, 0]
    result = {
        "race_count": int(len(actual)),
        "boat_top1_accuracy": float(np.mean(np.argmax(scores, axis=1) == winner)),
        "log_loss": float(-np.log(np.clip(probs[np.arange(len(actual)), actual], 1e-12, 1.0)).mean()),
    }
    for n in (1, 3, 5, 10, 12, 20):
        result[f"top{n}_hit_rate"] = float(np.mean(ranks <= n))
    # Private arrays support paired comparisons but are not serialized in the report.
    return {"metrics": result, "actual_ranks": ranks, "winner_hits": np.argmax(scores, axis=1) == winner}


def predict_experiment_scores(bundle: dict, frame: pd.DataFrame) -> np.ndarray:
    """Score pre-race rows with the saved experimental ST + ranker pipeline."""
    from src.models.ranker import (
        apply_prediction_time_measurement_proxies, build_lightgbm_frame, sort_for_grouping,
    )

    prepared = apply_prediction_time_measurement_proxies(frame)
    st = bundle["st_model"].predict(st_input_frame(prepared, bundle["st_input_columns"]))
    prepared = add_predicted_st_features(prepared, st)
    prepared["_st_row_position"] = np.arange(len(prepared))
    ordered = sort_for_grouping(prepared)
    scores = bundle["ranker"].predict(build_lightgbm_frame(
        ordered, bundle["rank_features"], bundle["categorical_columns"]
    ))
    result = np.empty(len(prepared), dtype=float)
    result[ordered["_st_row_position"].to_numpy(dtype=int)] = scores
    return result


def run_st_prediction_experiment(
    train: pd.DataFrame, valid: pd.DataFrame, actual_train_st: np.ndarray,
    actual_valid_st: np.ndarray, features: list[str], categories: list[str],
    config: dict, artifact_path: Path, progress: Callable[[str], None],
    skip_evaluation: bool = False,
) -> dict[str, Any]:
    from src.models.ranker import train_lightgbm

    settings = {**DEFAULT_SETTINGS, **config.get("models", {}).get("st_prediction", {})}
    if int(settings["oof_folds"]) < 1 or not 0 < float(settings["evaluation_fraction"]) < 1:
        raise ValueError("ST prediction requires oof_folds >= 1 and 0 < evaluation_fraction < 1")
    if int(settings["num_threads"]) < 1 or int(settings["min_training_rows"]) < 1:
        raise ValueError("ST prediction thread and minimum row counts must be positive")
    if train.empty or valid.empty:
        return {"status": "insufficient_data"}
    dates = np.sort(pd.to_datetime(train["race_date"]).dt.normalize().dropna().unique())
    if len(dates) < int(settings["oof_folds"]) + 2:
        return {"status": "insufficient_dates"}
    cut_index = min(len(dates) - 1, max(1, int(len(dates) * (1 - float(settings["evaluation_fraction"])))))
    cutoff = pd.Timestamp(dates[cut_index])
    def prepare(frame: pd.DataFrame, actual: np.ndarray, before: bool, limit: int) -> pd.DataFrame:
        # Select on narrow metadata before copying hundreds of feature columns.
        metadata_columns = ["race_id", "race_date", "lane", "course", "finish_position"]
        if "finish_status" in frame:
            metadata_columns.append("finish_status")
        metadata = frame[metadata_columns].copy()
        metadata["_row_position"] = np.arange(len(frame))
        dates = pd.to_datetime(metadata["race_date"]).dt.normalize()
        metadata = metadata[dates < cutoff if before else dates >= cutoff]
        metadata = _sample_races(_complete_races(metadata), limit, int(settings["random_seed"]))
        positions = metadata["_row_position"].to_numpy(dtype=int)
        selected = frame.iloc[positions].copy().reset_index(drop=True)
        selected["_actual_st"] = np.asarray(actual)[positions]
        return selected

    train = prepare(train, actual_train_st, True, int(settings["max_train_races"]))
    valid = prepare(valid, actual_valid_st, False, int(settings["max_eval_races"]))
    if train.empty or valid.empty:
        return {"status": "insufficient_complete_races"}
    columns = [c for c in ST_INPUT_COLUMNS if c in train and pd.to_numeric(train[c], errors="coerce").notna().any()]
    if not any("prev" in c for c in columns):
        return {"status": "missing_st_history"}
    progress(f"ST experiment chronological split: cutoff={cutoff.date()}, train_races={train.race_id.nunique()}, eval_races={valid.race_id.nunique()}")
    st_columns = [*columns, "race_date", "_actual_st"]
    if "finish_status" in train:
        st_columns.append("finish_status")
    predictions, folds = chronological_oof_st(train[st_columns], columns, settings, progress)
    eligible = np.isfinite(predictions)
    if not eligible.any():
        return {"status": "insufficient_oof_history", "folds": folds}
    st_model = fit_st_model(train[st_columns], columns, settings)
    if st_model is None:
        return {"status": "insufficient_st_targets"}
    valid_predictions = st_model.predict(st_input_frame(valid, columns))
    rank_train = train.loc[eligible].reset_index(drop=True)
    augmented_train = add_predicted_st_features(rank_train, predictions[eligible])
    augmented_valid = add_predicted_st_features(valid, valid_predictions)
    # No evaluation labels are passed to either fit; iterations are fixed for this A/B.
    fit_config = {**config, "model": {
        **config["model"], "iterations": int(settings["ranker_iterations"]),
        "training_device": "cpu", "gpu": {"lightgbm": False},
    }}
    progress(f"ST experiment paired ranker training: races={rank_train.race_id.nunique()}, features_added={len(ST_FEATURE_COLUMNS)}")
    baseline = train_lightgbm(rank_train, rank_train, features, categories, fit_config, num_threads=int(settings["num_threads"]))
    augmented_features = [*features, *ST_FEATURE_COLUMNS]
    augmented = train_lightgbm(augmented_train, augmented_train, augmented_features, categories, fit_config, num_threads=int(settings["num_threads"]))
    bundle = {
        "version": 1, "st_model": st_model, "ranker": augmented, "baseline_ranker": baseline,
        "st_input_columns": columns, "rank_features": augmented_features,
        "baseline_features": features, "categorical_columns": categories,
        "training_max_date": pd.to_datetime(train.race_date).max().isoformat(),
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, artifact_path)
    report = {
        "status": "evaluation_skipped" if skip_evaluation else "completed",
        "scope": "chronological_paired_lightgbm_rank_only_uncalibrated_not_production_ensemble",
        "settings": settings, "st_input_columns": columns, "added_features": list(ST_FEATURE_COLUMNS),
        "evaluation_cutoff": cutoff.strftime("%Y-%m-%d"),
        "evaluation_max_date": pd.to_datetime(valid.race_date).max().strftime("%Y-%m-%d"),
        "st_training_races": int(train.race_id.nunique()),
        "paired_rank_training_races": int(rank_train.race_id.nunique()),
        "warmup_excluded_races": int(train.race_id.nunique() - rank_train.race_id.nunique()),
        "evaluation_races": int(valid.race_id.nunique()), "folds": folds,
        "artifact_path": str(artifact_path),
    }
    if skip_evaluation:
        return report
    report["st_evaluation"] = _st_errors(valid, valid_predictions)
    progress("ST experiment evaluating paired rankers")
    before = _rank_metrics(baseline, valid, features, categories)
    after = _rank_metrics(augmented, augmented_valid, augmented_features, categories)
    report["baseline"] = before["metrics"]
    report["with_predicted_st"] = after["metrics"]
    report["delta"] = {
        k: after["metrics"][k] - v for k, v in before["metrics"].items() if k != "race_count"
    }
    paired = {}
    for n in (3, 12):
        old, new = before["actual_ranks"] <= n, after["actual_ranks"] <= n
        differences = new.astype(float) - old.astype(float)
        paired[f"top{n}"] = {
            "gained_races": int((new & ~old).sum()), "lost_races": int((old & ~new).sum()),
            "delta_standard_error": float(differences.std(ddof=1) / np.sqrt(len(old))) if len(old) > 1 else None,
        }
    report["paired_comparison"] = paired
    errors = report["st_evaluation"]
    if errors.get("sample_count", 0):
        progress(f"ST experiment MAE: proxy={errors['proxy_mae_seconds']:.5f}s, model={errors['model_mae_seconds']:.5f}s")
    for key in ("boat_top1_accuracy", "top3_hit_rate", "top12_hit_rate"):
        progress(f"ST experiment {key}: baseline={report['baseline'][key]:.4%}, predicted_st={report['with_predicted_st'][key]:.4%}, delta={report['delta'][key] * 100:+.3f}pp")
    return report
