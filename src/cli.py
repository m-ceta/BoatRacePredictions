from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.api import (
    backfill_rowdata_files,
    build_dataset_from_rowdata_streaming,
    load_bundle,
    load_prediction_input,
)
from src.drive_restore import (
    DEFAULT_ARTIFACTS_DRIVE_FILE_URL,
    DEFAULT_DATA_DRIVE_FILE_URL,
    DEFAULT_ROWDATA_DRIVE_FILE_URL,
    download_and_restore_packages,
)
from src.evaluation.metrics import compute_trifecta_rerank_metrics
from src.features.builder import build_training_table, save_processed_tables
from src.features.streaming_builder import compare_processed_tables
from src.live import predict_today_race
from src.models.ranker import (
    cleanup_processed_intermediate_dirs,
    collect_garbage,
    evaluate_trifecta,
    fit_model_trifecta_calibrator,
    get_artifact_paths,
    load_config,
    load_classifier_artifacts,
    load_ensemble_weights,
    load_feature_columns,
    load_flow_artifacts,
    load_models,
    load_optional_trifecta_calibrator,
    load_staged_model_artifacts,
    load_trifecta_v2_model_artifact,
    load_trifecta_calibrator,
    prepare_training_table,
    predict_trifecta_probabilities,
    predict_race_order,
    save_artifacts,
    train_ranker,
    train_phase3_conditional_trifecta_model,
    train_trifecta_v2_model,
    optimize_rerank_inference_settings,
    optimize_phase3_calibration_window,
    with_conservative_rerank_weight,
    with_rank_penalty_settings,
    infer_feature_columns,
    infer_categorical_columns,
    infer_latest_available_race_date,
    load_training_splits_from_parquet,
    train_ranker_from_splits,
    with_latest_available_dates,
)
from src.models.training_device import with_training_device_override
from src.models.flow import evaluate_flow_model, train_flow_model
from src.models.staged import evaluate_staged_models, train_staged_models
from src.parsers.bk_parser import parse_entry_file, parse_result_file


def build_dataset_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rowdata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-date", type=str, default=None)
    args = parser.parse_args()
    summary = build_dataset_from_rowdata_streaming(
        rowdata_dir=args.rowdata,
        output_dir=args.output,
        max_date=args.max_date,
    )
    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))


def compare_processed_tables_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--actual", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1e-12)
    args = parser.parse_args()

    report = compare_processed_tables(args.expected, args.actual, atol=args.atol)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def backfill_rowdata_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rowdata", type=Path, required=True)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--kinds", type=str, default="BK")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    report = backfill_rowdata_files(
        rowdata_dir=args.rowdata,
        start_date=args.start,
        end_date=args.end,
        kinds=args.kinds,
        overwrite=args.overwrite,
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))


def package_download_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--rowdata-url", type=str, default=DEFAULT_ROWDATA_DRIVE_FILE_URL)
    parser.add_argument("--data-url", type=str, default=DEFAULT_DATA_DRIVE_FILE_URL)
    parser.add_argument("--artifacts-url", type=str, default=DEFAULT_ARTIFACTS_DRIVE_FILE_URL)
    parser.add_argument("--rowdata-zip-name", type=str, default="rowdata.zip")
    parser.add_argument("--data-zip-name", type=str, default="data.zip")
    parser.add_argument("--artifacts-zip-name", type=str, default="artifacts.zip")
    parser.add_argument("--skip-rowdata", action="store_true")
    parser.add_argument("--skip-data", action="store_true")
    parser.add_argument("--skip-artifacts", action="store_true")
    args = parser.parse_args()

    report = download_and_restore_packages(
        project_root=args.project_root,
        rowdata_drive_file_url=args.rowdata_url,
        data_drive_file_url=args.data_url,
        artifacts_drive_file_url=args.artifacts_url,
        rowdata_zip_name=args.rowdata_zip_name,
        data_zip_name=args.data_zip_name,
        artifacts_zip_name=args.artifacts_zip_name,
        restore_rowdata=not args.skip_rowdata,
        restore_data=not args.skip_data,
        restore_artifacts=not args.skip_artifacts,
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))


def train_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--training-device", choices=["cpu", "gpu"], default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    config = with_training_device_override(config, args.training_device)
    train_df, valid_df, test_df, config = load_training_splits_from_parquet(
        Path(config["data"]["training_table"]),
        config,
    )

    (
        models,
        feature_columns,
        metrics,
        trifecta_calibrator,
        classifier_models,
        flow_model,
        flow_classes,
        staged_models,
        trifecta_v2_model,
    ) = train_ranker_from_splits(train_df, valid_df, test_df, config)
    del train_df, valid_df, test_df
    collect_garbage()
    artifacts = get_artifact_paths(config)
    save_artifacts(
        models=models,
        feature_columns=feature_columns,
        metrics=metrics,
        trifecta_calibrator=trifecta_calibrator,
        catboost_model_path=artifacts["catboost_model_path"],
        lightgbm_model_path=artifacts["lightgbm_model_path"],
        features_path=artifacts["features_path"],
        ensemble_weights_path=artifacts["ensemble_weights_path"],
        trifecta_calibrator_path=artifacts["trifecta_calibrator_path"],
        metrics_path=artifacts["metrics_path"],
        classifier_models=classifier_models,
        classifier_output_dir=artifacts["classifier_dir"],
        flow_model=flow_model,
        flow_classes=flow_classes,
        flow_model_path=artifacts["flow_model_path"],
        flow_meta_path=artifacts["flow_meta_path"],
        staged_models=staged_models,
        staged_output_dir=artifacts["staged_dir"],
        trifecta_v2_model=trifecta_v2_model,
        trifecta_v2_model_path=artifacts["trifecta_v2_model_path"],
    )
    cleanup_processed_intermediate_dirs(config)


def predict_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"))
    parser.add_argument("--trifecta-output", type=Path, default=None)
    parser.add_argument("--odds", type=Path, default=None)
    parser.add_argument("--rerank-top-n", type=int, default=None)
    args = parser.parse_args()

    bundle = load_bundle(args.config)
    feature_columns = load_feature_columns(args.features)
    future_df = load_prediction_input(args.input)
    predictions = predict_race_order(bundle.models, feature_columns, future_df, bundle.ensemble_weights)
    predictions.to_csv(args.output, index=False, encoding="utf-8-sig")

    if args.trifecta_output is not None:
        odds_df = load_prediction_input(args.odds) if args.odds is not None else None
        trifecta = predict_trifecta_probabilities(
            models=bundle.models,
            feature_columns=feature_columns,
            future_df=future_df,
            ensemble_weights=bundle.ensemble_weights,
            trifecta_calibrator=bundle.trifecta_calibrator,
            classifier_models=bundle.classifier_models,
            flow_model=bundle.flow_model,
            flow_classes=bundle.flow_classes,
            staged_models=bundle.staged_models,
            trifecta_v2_model=bundle.trifecta_v2_model,
            odds_df=odds_df,
            use_v2=True,
            rerank_top_n=bundle.rerank_top_n if args.rerank_top_n is None else args.rerank_top_n,
        )
        trifecta.to_csv(args.trifecta_output, index=False, encoding="utf-8-sig")


def train_trifecta_v2_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--training-device", choices=["cpu", "gpu"], default=None)
    parser.add_argument("--max-races", type=int, default=1000)
    parser.add_argument("--eval-max-races", type=int, default=3000)
    parser.add_argument("--eval-rerank-top-n", type=int, default=24)
    parser.add_argument("--optimize-rerank", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    config = with_training_device_override(config, args.training_device)
    artifacts = get_artifact_paths(config)
    train_end = pd.Timestamp(config["split"]["train_end_date"])
    valid_end = pd.Timestamp(config["split"]["valid_end_date"])
    train_df, valid_df, test_df = load_training_splits(Path(config["data"]["training_table"]), config)
    eval_valid_df = sample_races_for_evaluation(valid_df, args.eval_max_races)
    eval_test_df = sample_races_for_evaluation(test_df, args.eval_max_races)

    schema_df = pd.concat(
        [train_df.head(200), valid_df.head(200), test_df.head(200)],
        ignore_index=True,
    )
    feature_columns = infer_feature_columns(schema_df)
    categorical_columns = infer_categorical_columns(schema_df, feature_columns)
    del schema_df
    collect_garbage()
    models = load_models(config)
    classifier_models = load_classifier_artifacts(config)
    ensemble_weights = load_ensemble_weights(artifacts["ensemble_weights_path"])
    trifecta_calibrator = load_trifecta_calibrator(artifacts["trifecta_calibrator_path"])

    flow_model, flow_classes = train_flow_model(train_df, valid_df, feature_columns, categorical_columns, config)
    collect_garbage()
    staged_models = train_staged_models(train_df, valid_df, feature_columns, categorical_columns, config)
    collect_garbage()
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
        max_races=args.max_races,
        config=config,
    )
    collect_garbage()
    trifecta_v3_model = train_phase3_conditional_trifecta_model(
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
        max_races=args.max_races,
        config=config,
    )
    collect_garbage()
    rerank_optimization = {}
    if args.optimize_rerank and not eval_valid_df.empty:
        rerank_optimization = optimize_rerank_inference_settings(
            models,
            ensemble_weights,
            eval_valid_df,
            feature_columns,
            categorical_columns,
            classifier_models=classifier_models,
            flow_model=flow_model,
            flow_classes=flow_classes,
            staged_models=staged_models,
            trifecta_v2_model=trifecta_v3_model,
        )
        optimized_top_n = int(rerank_optimization.get("best_top_n", args.eval_rerank_top_n))
        optimized_weight = float(rerank_optimization.get("best_conservative_weight", 0.92))
        optimized_penalty = float(rerank_optimization.get("best_rank_penalty_strength", 0.0))
        args.eval_rerank_top_n = optimized_top_n
        trifecta_v3_model = with_conservative_rerank_weight(trifecta_v3_model, optimized_weight)
        trifecta_v3_model = with_rank_penalty_settings(trifecta_v3_model, optimized_penalty, 5)
    calibration_optimization = optimize_phase3_calibration_window(
        models,
        ensemble_weights,
        valid_df,
        feature_columns,
        categorical_columns,
        classifier_models=classifier_models,
        flow_model=flow_model,
        flow_classes=flow_classes,
        staged_models=staged_models,
        trifecta_v2_model=trifecta_v3_model,
        rerank_top_n=args.eval_rerank_top_n,
    )
    calibration_window_days = int(calibration_optimization.get("best_window_days", 60))
    trifecta_v2_calibrator = fit_model_trifecta_calibrator(
        models,
        ensemble_weights,
        valid_df,
        feature_columns,
        categorical_columns,
        classifier_models=classifier_models,
        flow_model=flow_model,
        flow_classes=flow_classes,
        staged_models=staged_models,
        trifecta_v2_model=trifecta_v2_model,
        use_v2=True,
        rerank_top_n=args.eval_rerank_top_n,
    )
    trifecta_v3_calibrator = fit_model_trifecta_calibrator(
        models,
        ensemble_weights,
        valid_df,
        feature_columns,
        categorical_columns,
        classifier_models=classifier_models,
        flow_model=flow_model,
        flow_classes=flow_classes,
        staged_models=staged_models,
        trifecta_v2_model=trifecta_v3_model,
        use_v2=True,
        rerank_top_n=args.eval_rerank_top_n,
        calibration_window_days=calibration_window_days,
    )

    save_artifacts(
        models=models,
        feature_columns=feature_columns,
        metrics=json_load_or_empty(artifacts["metrics_path"]),
        trifecta_calibrator=trifecta_calibrator,
        catboost_model_path=artifacts["catboost_model_path"],
        lightgbm_model_path=artifacts["lightgbm_model_path"],
        features_path=artifacts["features_path"],
        ensemble_weights_path=artifacts["ensemble_weights_path"],
        trifecta_calibrator_path=artifacts["trifecta_calibrator_path"],
        metrics_path=artifacts["metrics_path"],
        classifier_models=classifier_models,
        classifier_output_dir=artifacts["classifier_dir"],
        flow_model=flow_model,
        flow_classes=flow_classes,
        flow_model_path=artifacts["flow_model_path"],
        flow_meta_path=artifacts["flow_meta_path"],
        staged_models=staged_models,
        staged_output_dir=artifacts["staged_dir"],
        trifecta_v2_model=trifecta_v3_model,
        trifecta_v2_model_path=artifacts["trifecta_v2_model_path"],
        trifecta_v2_calibrator=trifecta_v2_calibrator,
        trifecta_v2_calibrator_path=artifacts["trifecta_v2_calibrator_path"],
        trifecta_v3_calibrator=trifecta_v3_calibrator,
        trifecta_v3_calibrator_path=artifacts["trifecta_v3_calibrator_path"],
    )

    metrics = json_load_or_empty(artifacts["metrics_path"])
    metrics["flow_model_metrics"] = evaluate_flow_model(
        flow_model,
        flow_classes,
        train_df,
        eval_valid_df,
        eval_test_df,
        feature_columns,
        categorical_columns,
    )
    metrics["staged_model_metrics"] = evaluate_staged_models(
        staged_models,
        train_df,
        eval_valid_df,
        eval_test_df,
        feature_columns,
        categorical_columns,
    )
    metrics["trifecta_evaluation_scope"] = {
        "valid_races": int(eval_valid_df["race_id"].nunique()) if not eval_valid_df.empty else 0,
        "test_races": int(eval_test_df["race_id"].nunique()) if not eval_test_df.empty else 0,
        "eval_max_races": int(args.eval_max_races),
        "eval_rerank_top_n": int(args.eval_rerank_top_n),
    }
    if rerank_optimization:
        metrics["rerank_optimization"] = rerank_optimization
    metrics["calibration_optimization"] = calibration_optimization
    metrics["trifecta_v1_rerank_metrics"] = {
        "valid_calibrated": evaluate_trifecta(
            models,
            ensemble_weights,
            trifecta_calibrator,
            eval_valid_df,
            feature_columns,
            categorical_columns,
            classifier_models=classifier_models,
            flow_model=flow_model,
            flow_classes=flow_classes,
            staged_models=staged_models,
            trifecta_v2_model=trifecta_v2_model,
            use_v2=False,
            rerank_top_n=args.eval_rerank_top_n,
        ),
        "test_calibrated": evaluate_trifecta(
            models,
            ensemble_weights,
            trifecta_calibrator,
            eval_test_df,
            feature_columns,
            categorical_columns,
            classifier_models=classifier_models,
            flow_model=flow_model,
            flow_classes=flow_classes,
            staged_models=staged_models,
            trifecta_v2_model=trifecta_v2_model,
            use_v2=False,
            rerank_top_n=args.eval_rerank_top_n,
        ),
    }
    metrics["trifecta_v2_metrics"] = {
        "valid_calibrated": evaluate_trifecta(
            models,
            ensemble_weights,
            trifecta_v2_calibrator,
            eval_valid_df,
            feature_columns,
            categorical_columns,
            classifier_models=classifier_models,
            flow_model=flow_model,
            flow_classes=flow_classes,
            staged_models=staged_models,
            trifecta_v2_model=trifecta_v2_model,
            use_v2=True,
            rerank_top_n=args.eval_rerank_top_n,
        ),
        "test_calibrated": evaluate_trifecta(
            models,
            ensemble_weights,
            trifecta_v2_calibrator,
            eval_test_df,
            feature_columns,
            categorical_columns,
            classifier_models=classifier_models,
            flow_model=flow_model,
            flow_classes=flow_classes,
            staged_models=staged_models,
            trifecta_v2_model=trifecta_v2_model,
            use_v2=True,
            rerank_top_n=args.eval_rerank_top_n,
        ),
    }
    metrics["trifecta_v3_metrics"] = {
        "valid_calibrated": evaluate_trifecta(
            models,
            ensemble_weights,
            trifecta_v3_calibrator,
            eval_valid_df,
            feature_columns,
            categorical_columns,
            classifier_models=classifier_models,
            flow_model=flow_model,
            flow_classes=flow_classes,
            staged_models=staged_models,
            trifecta_v2_model=trifecta_v3_model,
            use_v2=True,
            rerank_top_n=args.eval_rerank_top_n,
        ),
        "test_calibrated": evaluate_trifecta(
            models,
            ensemble_weights,
            trifecta_v3_calibrator,
            eval_test_df,
            feature_columns,
            categorical_columns,
            classifier_models=classifier_models,
            flow_model=flow_model,
            flow_classes=flow_classes,
            staged_models=staged_models,
            trifecta_v2_model=trifecta_v3_model,
            use_v2=True,
            rerank_top_n=args.eval_rerank_top_n,
        ),
    }
    artifacts["metrics_path"].write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    cleanup_processed_intermediate_dirs(config)


def json_load_or_empty(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def sample_races_for_evaluation(df: pd.DataFrame, max_races: int) -> pd.DataFrame:
    if df.empty or max_races <= 0:
        return df.copy()
    races = df[["race_id", "race_date"]].drop_duplicates().sort_values(["race_date", "race_id"]).reset_index(drop=True)
    if len(races) <= max_races:
        return df.copy()
    indices = np.linspace(0, len(races) - 1, num=max_races, dtype=int)
    selected_ids = races.iloc[np.unique(indices)]["race_id"].tolist()
    return df[df["race_id"].isin(selected_ids)].copy()


def evaluate_trifecta_full_valid_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--chunk", type=str, default="month", choices=["month"])
    parser.add_argument("--date-from", type=str, default=None)
    parser.add_argument("--date-to", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    artifacts = get_artifact_paths(config)
    train_df, valid_df, test_df = load_training_splits(Path(config["data"]["training_table"]), config)
    if args.date_from is not None:
        valid_df = valid_df[valid_df["race_date"] >= pd.Timestamp(args.date_from)].copy()
        test_df = test_df[test_df["race_date"] >= pd.Timestamp(args.date_from)].copy()
    if args.date_to is not None:
        valid_df = valid_df[valid_df["race_date"] <= pd.Timestamp(args.date_to)].copy()
        test_df = test_df[test_df["race_date"] <= pd.Timestamp(args.date_to)].copy()
    schema_df = pd.concat([train_df.head(200), valid_df.head(200), test_df.head(200)], ignore_index=True)
    feature_columns = infer_feature_columns(schema_df)
    categorical_columns = infer_categorical_columns(schema_df, feature_columns)

    models = load_models(config)
    classifier_models = load_classifier_artifacts(config)
    flow_model, flow_classes = load_flow_artifacts(config)
    staged_models = load_staged_model_artifacts(config)
    trifecta_v3_model = load_trifecta_v2_model_artifact(config)
    ensemble_weights = load_ensemble_weights(artifacts["ensemble_weights_path"])
    v1_calibrator = load_trifecta_calibrator(artifacts["trifecta_calibrator_path"])
    v3_calibrator = load_optional_trifecta_calibrator(artifacts["trifecta_v3_calibrator_path"])
    rerank_top_n = int(config.get("inference", {}).get("trifecta_rerank_top_n", 10))

    valid_metrics = evaluate_trifecta_in_chunks(
        valid_df,
        models=models,
        weights=ensemble_weights,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        classifier_models=classifier_models,
        flow_model=flow_model,
        flow_classes=flow_classes,
        staged_models=staged_models,
        trifecta_v3_model=trifecta_v3_model,
        v1_calibrator=v1_calibrator,
        v3_calibrator=v3_calibrator,
        rerank_top_n=rerank_top_n,
    )
    test_metrics = evaluate_trifecta_in_chunks(
        test_df,
        models=models,
        weights=ensemble_weights,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        classifier_models=classifier_models,
        flow_model=flow_model,
        flow_classes=flow_classes,
        staged_models=staged_models,
        trifecta_v3_model=trifecta_v3_model,
        v1_calibrator=v1_calibrator,
        v3_calibrator=v3_calibrator,
        rerank_top_n=rerank_top_n,
    )

    metrics = json_load_or_empty(artifacts["metrics_path"])
    metrics["trifecta_evaluation_scope"] = {
        "valid_races": int(valid_df["race_id"].nunique()) if not valid_df.empty else 0,
        "test_races": int(test_df["race_id"].nunique()) if not test_df.empty else 0,
        "eval_rerank_top_n": rerank_top_n,
        "chunk": args.chunk,
        "evaluation_mode": "full_valid_chunked",
        "date_from": args.date_from,
        "date_to": args.date_to,
    }
    metrics["trifecta_v1_rerank_metrics"] = valid_metrics["v1"]
    metrics["trifecta_v3_metrics"] = valid_metrics["phase3"]
    metrics["trifecta_v1_rerank_metrics"]["test_calibrated"] = test_metrics["v1"].get("valid_calibrated", {})
    metrics["trifecta_v3_metrics"]["test_calibrated"] = test_metrics["phase3"].get("valid_calibrated", {})
    artifacts["metrics_path"].write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def iter_month_chunks(df: pd.DataFrame) -> list[pd.DataFrame]:
    if df.empty:
        return []
    frame = df.copy()
    frame["race_month"] = frame["race_date"].dt.to_period("M")
    chunks = [chunk.drop(columns=["race_month"]).copy() for _, chunk in frame.groupby("race_month", sort=True)]
    return chunks


def aggregate_metric_dicts(metric_dicts: list[dict]) -> dict:
    metric_dicts = [item for item in metric_dicts if item]
    if not metric_dicts:
        return {}
    aggregated: dict[str, object] = {}
    keys = set().union(*(d.keys() for d in metric_dicts))
    for key in keys:
        values = [d[key] for d in metric_dicts if key in d]
        if not values:
            continue
        if isinstance(values[0], dict):
            aggregated[key] = aggregate_metric_dicts(values)
            continue
        weight_key = "coverage_races" if key in {"rerank_top1_hit_rate", "rerank_mrr", "baseline_mrr", "mean_rank_improvement"} else "race_count"
        if key == "coverage_races":
            aggregated[key] = float(sum(float(v) for v in values))
            continue
        if key == "race_count":
            aggregated[key] = float(sum(float(v) for v in values))
            continue
        total_weight = 0.0
        weighted_sum = 0.0
        for item in metric_dicts:
            if key not in item:
                continue
            weight = float(item.get(weight_key, 0.0))
            weighted_sum += float(item[key]) * weight
            total_weight += weight
        aggregated[key] = (weighted_sum / total_weight) if total_weight > 0 else float(np.mean([float(v) for v in values]))
    return aggregated


def evaluate_trifecta_in_chunks(
    df: pd.DataFrame,
    *,
    models: dict,
    weights: dict,
    feature_columns: list[str],
    categorical_columns: list[str],
    classifier_models: dict,
    flow_model,
    flow_classes,
    staged_models: dict,
    trifecta_v3_model,
    v1_calibrator,
    v3_calibrator,
    rerank_top_n: int,
) -> dict[str, dict]:
    chunks = iter_month_chunks(df)
    v1_chunk_metrics: list[dict] = []
    v3_chunk_metrics: list[dict] = []
    for chunk in chunks:
        v1_chunk_metrics.append(
            evaluate_trifecta(
                models,
                weights,
                v1_calibrator,
                chunk,
                feature_columns,
                categorical_columns,
                classifier_models=classifier_models,
                flow_model=flow_model,
                flow_classes=flow_classes,
                staged_models=staged_models,
                trifecta_v2_model=trifecta_v3_model,
                use_v2=False,
                rerank_top_n=rerank_top_n,
            )
        )
        v3_chunk_metrics.append(
            evaluate_trifecta(
                models,
                weights,
                v3_calibrator,
                chunk,
                feature_columns,
                categorical_columns,
                classifier_models=classifier_models,
                flow_model=flow_model,
                flow_classes=flow_classes,
                staged_models=staged_models,
                trifecta_v2_model=trifecta_v3_model,
                use_v2=True,
                rerank_top_n=rerank_top_n,
            )
        )
    return {
        "v1": {"valid_calibrated": aggregate_metric_dicts(v1_chunk_metrics)},
        "phase3": {"valid_calibrated": aggregate_metric_dicts(v3_chunk_metrics)},
    }


def load_training_splits(training_table_path: Path, config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    latest_race_dates = pd.read_parquet(training_table_path, columns=["race_date"])
    config = with_latest_available_dates(config, infer_latest_available_race_date(latest_race_dates))
    data_config = config.get("data", {})
    min_date = pd.Timestamp(data_config.get("min_date")) if data_config.get("min_date") else None
    max_date = pd.Timestamp(data_config.get("max_date")) if data_config.get("max_date") else None
    train_end = pd.Timestamp(config["split"]["train_end_date"])
    valid_end = pd.Timestamp(config["split"]["valid_end_date"])

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
    valid_df = pd.read_parquet(
        training_table_path,
        filters=[("race_date", ">=", valid_start), ("race_date", "<=", valid_stop)],
    ) if valid_start <= valid_stop else pd.DataFrame()
    if test_stop is not None and test_start <= test_stop:
        test_df = pd.read_parquet(
            training_table_path,
            filters=[("race_date", ">=", test_start), ("race_date", "<=", test_stop)],
        )
    else:
        test_df = pd.DataFrame()

    return (
        prepare_training_table(train_df, config),
        prepare_training_table(valid_df, config) if not valid_df.empty else valid_df,
        prepare_training_table(test_df, config) if not test_df.empty else test_df,
    )


def predict_today_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--venue", type=str, required=True)
    parser.add_argument("--race-no", type=int, required=True)
    args = parser.parse_args()

    bundle = load_bundle(args.config)
    prediction = predict_today_race(bundle=bundle, venue=args.venue, race_no=args.race_no)
    print(prediction.text)


if __name__ == "__main__":
    raise SystemExit("Use boatrace-build / boatrace-train / boatrace-predict.")
