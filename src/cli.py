from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

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
    export_package_archives,
    restore_packages_from_zip_files,
)
from src.features.streaming_builder import compare_processed_tables
from src.live import predict_today_race
from src.recent_backtest import evaluate_recent_week_predictions
from src.recent_backtest import export_backtest_report_artifacts
from src.models.ranker import (
    cleanup_processed_intermediate_dirs,
    collect_garbage,
    evaluate_trifecta,
    fit_model_trifecta_calibrator,
    get_artifact_paths,
    get_default_rerank_top_n,
    get_phase3_settings,
    get_rerank_top_n,
    load_config,
    load_classifier_artifacts,
    load_ensemble_weights,
    load_feature_columns,
    load_flow_artifacts,
    load_models,
    load_optional_trifecta_calibrator,
    load_staged_model_artifacts,
    load_train_checkpoint,
    load_trifecta_v2_model_artifact,
    load_trifecta_v2_model_artifact_payload,
    load_trifecta_calibrator,
    mark_train_stage_completed,
    prepare_training_table,
    predict_trifecta_probabilities,
    predict_race_order,
    save_artifacts,
    save_train_checkpoint,
    save_trifecta_calibrator_artifact,
    save_trifecta_v2_model_artifact,
    optimize_trifecta_v2_blend_weight,
    train_phase3_conditional_trifecta_model,
    train_trifecta_v2_model,
    optimize_rerank_inference_settings_two_stage,
    optimize_dynamic_rerank_weights,
    optimize_phase3_calibration_window,
    with_conservative_rerank_weight,
    with_dynamic_rerank_weight_metadata,
    with_calibration_window_days,
    with_phase3_optimization_metadata,
    with_rank_penalty_settings,
    with_rerank_top_n,
    apply_prediction_time_measurement_proxies,
    infer_feature_columns,
    infer_categorical_columns,
    infer_latest_available_race_date,
    load_training_splits_from_parquet,
    select_feature_columns_for_set,
    train_ranker_from_splits,
    train_checkpoint_signature,
    train_stage_completed,
    with_latest_available_dates,
)
from src.models.training_device import with_training_device_override
from src.models.flow import evaluate_flow_model, save_flow_model, train_flow_model
from src.models.staged import evaluate_staged_models, save_staged_models, train_staged_models


LOG_TZ = ZoneInfo("Asia/Tokyo")


def current_log_time() -> str:
    return datetime.now(LOG_TZ).strftime("%Y-%m-%d %H:%M:%S JST")


def print_progress(command_name: str, message: str) -> None:
    print(f"[{current_log_time()}] [{command_name}] {message}", flush=True)


def build_dataset_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rowdata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-date", type=str, default=None)
    args = parser.parse_args()
    print_progress("boatrace-build", f"start: rowdata={args.rowdata}, output={args.output}, max_date={args.max_date}")
    summary = build_dataset_from_rowdata_streaming(
        rowdata_dir=args.rowdata,
        output_dir=args.output,
        max_date=args.max_date,
    )
    print_progress("boatrace-build", "completed")
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

    print_progress(
        "boatrace-backfill-rowdata",
        f"start: rowdata={args.rowdata}, start={args.start}, end={args.end}, kinds={args.kinds}, overwrite={args.overwrite}",
    )
    report = backfill_rowdata_files(
        rowdata_dir=args.rowdata,
        start_date=args.start,
        end_date=args.end,
        kinds=args.kinds,
        overwrite=args.overwrite,
    )
    print_progress("boatrace-backfill-rowdata", "completed")
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


def package_restore_local_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--source-dir", type=Path, default=Path("."))
    parser.add_argument("--rowdata-zip", type=Path, default=None)
    parser.add_argument("--data-zip", type=Path, default=None)
    parser.add_argument("--artifacts-zip", type=Path, default=None)
    parser.add_argument("--rowdata-zip-name", type=str, default="rowdata.zip")
    parser.add_argument("--data-zip-name", type=str, default="data.zip")
    parser.add_argument("--artifacts-zip-name", type=str, default="artifacts.zip")
    parser.add_argument("--skip-rowdata", action="store_true")
    parser.add_argument("--skip-data", action="store_true")
    parser.add_argument("--skip-artifacts", action="store_true")
    args = parser.parse_args()

    report = restore_packages_from_zip_files(
        project_root=args.project_root,
        source_dir=args.source_dir,
        rowdata_zip_path=args.rowdata_zip,
        data_zip_path=args.data_zip,
        artifacts_zip_path=args.artifacts_zip,
        rowdata_zip_name=args.rowdata_zip_name,
        data_zip_name=args.data_zip_name,
        artifacts_zip_name=args.artifacts_zip_name,
        restore_rowdata=not args.skip_rowdata,
        restore_data=not args.skip_data,
        restore_artifacts=not args.skip_artifacts,
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))


def package_export_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rowdata-zip-name", type=str, default="rowdata.zip")
    parser.add_argument("--data-zip-name", type=str, default="data.zip")
    parser.add_argument("--artifacts-zip-name", type=str, default="artifacts.zip")
    parser.add_argument("--skip-rowdata", action="store_true")
    parser.add_argument("--skip-data", action="store_true")
    parser.add_argument("--skip-artifacts", action="store_true")
    args = parser.parse_args()

    report = export_package_archives(
        project_root=args.project_root,
        output_dir=args.output_dir,
        rowdata_zip_name=args.rowdata_zip_name,
        data_zip_name=args.data_zip_name,
        artifacts_zip_name=args.artifacts_zip_name,
        export_rowdata=not args.skip_rowdata,
        export_data=not args.skip_data,
        export_artifacts=not args.skip_artifacts,
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))


def backtest_recent_week_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--rowdata", type=Path, default=Path("rowdata"))
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--stake", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--report-dir", type=Path, default=None)
    args = parser.parse_args()

    report = evaluate_recent_week_predictions(
        config_path=args.config,
        rowdata_dir=args.rowdata,
        days=args.days,
        stake_per_ticket=args.stake,
        top_k=args.top_k,
        start_date=pd.Timestamp(args.start).date() if args.start else None,
        end_date=pd.Timestamp(args.end).date() if args.end else None,
    )
    if args.report_dir is not None:
        base_name = f"backtest_{report['summary']['start_date']}_{report['summary']['end_date']}_top{args.top_k}"
        report["artifacts"] = export_backtest_report_artifacts(report, args.report_dir, base_name)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def train_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--training-device", choices=["cpu", "gpu"], default=None)
    parser.add_argument("--max-races", type=int, default=None, help="maximum train races for quick base-model experiments")
    parser.add_argument("--enable-lightgbm-variants", action="store_true")
    parser.add_argument("--disable-lightgbm-variants", action="store_true")
    parser.add_argument("--lightgbm-variant-workers", type=int, default=None)
    parser.add_argument("--ensemble-workers", type=int, default=None)
    parser.add_argument("--ensemble-max-eval-races", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="resume completed boatrace-train stages from artifacts/train_checkpoint.json")
    parser.add_argument("--reset-train-checkpoint", action="store_true", help="clear boatrace-train checkpoint before training")
    args = parser.parse_args()
    if args.enable_lightgbm_variants and args.disable_lightgbm_variants:
        parser.error("--enable-lightgbm-variants and --disable-lightgbm-variants cannot be used together")

    progress = print_train_progress
    progress("start")
    progress("loading config and training splits")
    config = load_config(args.config)
    config = with_training_device_override(config, args.training_device)
    config = with_lightgbm_variant_cli_overrides(
        config,
        enable=args.enable_lightgbm_variants,
        disable=args.disable_lightgbm_variants,
        variant_workers=args.lightgbm_variant_workers,
        ensemble_workers=args.ensemble_workers,
        ensemble_max_eval_races=args.ensemble_max_eval_races,
    )
    train_df, valid_df, test_df, config = load_training_splits_from_parquet(
        Path(config["data"]["training_table"]),
        config,
    )
    if args.max_races is not None and args.max_races > 0:
        train_df = sample_races_for_evaluation(train_df, args.max_races)
        progress(f"sampled train races for quick experiment: max_races={args.max_races}, train_races={race_count(train_df)}")
    progress(
        "loaded splits: "
        f"train_races={race_count(train_df)}, valid_races={race_count(valid_df)}, "
        f"test_races={race_count(test_df)}"
    )

    progress("training base models")
    (
        models,
        feature_columns,
        metrics,
        trifecta_calibrator,
        classifier_models,
        _flow_model,
        _flow_classes,
        _staged_models,
        _trifecta_v2_model,
    ) = train_ranker_from_splits(
        train_df,
        valid_df,
        test_df,
        config,
        progress_callback=progress,
        resume=args.resume,
        reset_train_checkpoint=args.reset_train_checkpoint,
    )
    del train_df, valid_df, test_df
    collect_garbage()
    artifacts = get_artifact_paths(config)
    progress("saving base model artifacts")
    save_artifacts(
        models=models,
        feature_columns=feature_columns,
        metrics=metrics,
        trifecta_calibrator=trifecta_calibrator,
        catboost_model_path=artifacts["catboost_model_path"],
        lightgbm_model_path=artifacts["lightgbm_model_path"],
        xgboost_model_path=artifacts["xgboost_model_path"],
        random_forest_model_path=artifacts["random_forest_model_path"],
        ridge_model_path=artifacts["ridge_model_path"],
        neural_model_path=artifacts["neural_model_path"],
        features_path=artifacts["features_path"],
        ensemble_weights_path=artifacts["ensemble_weights_path"],
        trifecta_calibrator_path=artifacts["trifecta_calibrator_path"],
        metrics_path=artifacts["metrics_path"],
        classifier_models=classifier_models,
        classifier_output_dir=artifacts["classifier_dir"],
    )
    progress("cleaning intermediate files")
    cleanup_processed_intermediate_dirs(config)
    progress("completed")


def with_lightgbm_variant_cli_overrides(
    config: dict,
    *,
    enable: bool = False,
    disable: bool = False,
    variant_workers: int | None = None,
    ensemble_workers: int | None = None,
    ensemble_max_eval_races: int | None = None,
) -> dict:
    updated = dict(config)
    models_config = dict(updated.get("models", {}) or {})
    variant_config = dict(models_config.get("lightgbm_variants", {}) or {})
    if enable:
        variant_config["enabled"] = True
    if disable:
        variant_config["enabled"] = False
    if variant_workers is not None:
        variant_config["parallel_workers"] = max(int(variant_workers), 1)
    models_config["lightgbm_variants"] = variant_config
    ensemble_config = dict(models_config.get("ensemble", {}) or {})
    if ensemble_workers is not None:
        ensemble_config["parallel_workers"] = max(int(ensemble_workers), 1)
    if ensemble_max_eval_races is not None:
        ensemble_config["max_eval_races"] = max(int(ensemble_max_eval_races), 0)
    models_config["ensemble"] = ensemble_config
    updated["models"] = models_config
    return updated


def predict_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"))
    parser.add_argument("--trifecta-output", type=Path, default=None)
    parser.add_argument("--odds", type=Path, default=None)
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
            odds_df=odds_df,
            use_v2=False,
            rerank_top_n=None,
        )
        trifecta.to_csv(args.trifecta_output, index=False, encoding="utf-8-sig")


def print_train_trifecta_v2_progress(message: str) -> None:
    print_progress("boatrace-train-trifecta-v2", message)


def print_train_progress(message: str) -> None:
    print_progress("boatrace-train", message)


def print_eval_trifecta_full_progress(message: str) -> None:
    print_progress("boatrace-eval-trifecta-full", message)


def race_count(df: pd.DataFrame) -> int:
    return int(df["race_id"].nunique()) if "race_id" in df.columns and not df.empty else 0


def trifecta_train_checkpoint_signature(
    config: dict,
    artifacts: dict[str, Path],
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    valid_tune_df: pd.DataFrame,
    final_eval_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    max_races: int,
    eval_max_races: int,
    final_eval_max_races: int,
    final_eval_months: int,
) -> str:
    base_train_checkpoint_signature = None
    base_train_checkpoint_path = artifacts.get("train_checkpoint_path")
    if base_train_checkpoint_path is not None and base_train_checkpoint_path.exists():
        try:
            base_train_checkpoint_signature = json.loads(
                base_train_checkpoint_path.read_text(encoding="utf-8")
            ).get("signature")
        except Exception:
            base_train_checkpoint_signature = None
    payload = {
        "base_signature": train_checkpoint_signature(config, train_df, valid_df, test_df),
        "base_train_checkpoint_signature": base_train_checkpoint_signature,
        "valid_tune_races": race_count(valid_tune_df),
        "final_eval_races": race_count(final_eval_df),
        "max_races": int(max_races),
        "eval_max_races": int(eval_max_races),
        "final_eval_max_races": int(final_eval_max_races),
        "final_eval_months": int(final_eval_months),
    }
    raw = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def train_trifecta_v2_main() -> None:
    raise SystemExit("boatrace-train-trifecta-v2 was removed. Use boatrace-train instead.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--training-device", choices=["cpu", "gpu"], default=None)
    parser.add_argument("--max-races", type=int, default=0, help="maximum train races for v2/v3 models. 0 uses all races.")
    parser.add_argument("--eval-max-races", type=int, default=10000, help="maximum races for tuning/optimization. 0 uses all races.")
    parser.add_argument("--final-eval-max-races", type=int, default=0, help="maximum races for final_eval metrics. 0 uses all races.")
    parser.add_argument("--optimize-rerank", action="store_true")
    parser.add_argument(
        "--optimize-rerank-workers",
        type=int,
        default=1,
        help="rerank optimization parallel workers. 1 keeps sequential execution, 0 uses cpu_count - 1.",
    )
    parser.add_argument("--reset-rerank-optimization", action="store_true")
    parser.add_argument("--resume", action="store_true", help="resume v2/v3 training stages when matching artifacts exist.")
    parser.add_argument(
        "--reset-trifecta-train-checkpoint",
        action="store_true",
        help="reset v2/v3 training checkpoint before running.",
    )
    args = parser.parse_args()

    progress = print_train_trifecta_v2_progress
    progress("start")
    progress("loading config and training splits")
    config = load_config(args.config)
    config = with_training_device_override(config, args.training_device)
    artifacts = get_artifact_paths(config)
    train_df, valid_df, test_df = load_training_splits(Path(config["data"]["training_table"]), config)
    final_eval_months = get_final_eval_months(config)
    valid_tune_df, final_eval_df = split_valid_for_final_eval(valid_df, final_eval_months)
    eval_valid_tune_df = sample_races_for_evaluation(valid_tune_df, args.eval_max_races)
    eval_final_df = sample_races_for_evaluation(final_eval_df, args.final_eval_max_races)
    eval_report_df = eval_final_df if not eval_final_df.empty else eval_valid_tune_df
    eval_test_df = sample_races_for_evaluation(test_df, args.eval_max_races)
    progress(
        "loaded splits: "
        f"train_races={race_count(train_df)}, valid_races={race_count(valid_df)}, "
        f"valid_tune_races={race_count(valid_tune_df)}, final_eval_races={race_count(final_eval_df)}, "
        f"test_races={race_count(test_df)}, eval_valid_tune_races={race_count(eval_valid_tune_df)}, "
        f"eval_final_races={race_count(eval_final_df)}, "
        f"eval_test_races={race_count(eval_test_df)}"
    )
    trifecta_checkpoint_path = artifacts["trifecta_train_checkpoint_path"]
    if args.reset_trifecta_train_checkpoint and trifecta_checkpoint_path.exists():
        progress(f"resetting trifecta train checkpoint: {trifecta_checkpoint_path}")
        trifecta_checkpoint_path.unlink()
    trifecta_signature = trifecta_train_checkpoint_signature(
        config,
        artifacts,
        train_df,
        valid_df,
        valid_tune_df,
        final_eval_df,
        test_df,
        max_races=args.max_races,
        eval_max_races=args.eval_max_races,
        final_eval_max_races=args.final_eval_max_races,
        final_eval_months=final_eval_months,
    )
    trifecta_checkpoint = load_train_checkpoint(trifecta_checkpoint_path, trifecta_signature)
    save_train_checkpoint(trifecta_checkpoint_path, trifecta_checkpoint)

    progress("inferring feature columns")
    schema_df = pd.concat(
        [train_df.head(200), valid_tune_df.head(200), final_eval_df.head(200), test_df.head(200)],
        ignore_index=True,
    )
    feature_columns = infer_feature_columns(schema_df)
    categorical_columns = infer_categorical_columns(schema_df, feature_columns)
    del schema_df
    collect_garbage()
    progress(f"inferred features: numeric_and_categorical={len(feature_columns)}, categorical={len(categorical_columns)}")

    progress("loading existing ranker/classifier artifacts")
    models = load_models(config)
    classifier_models = load_classifier_artifacts(config)
    ensemble_weights = load_ensemble_weights(artifacts["ensemble_weights_path"])
    ensemble_weights["scenario_metric_min_races"] = int(
        get_phase3_settings(config)["evaluation"].get("scenario_min_races", 100)
    )
    trifecta_calibrator = load_trifecta_calibrator(artifacts["trifecta_calibrator_path"])
    eval_rerank_top_n = get_default_rerank_top_n(config)

    if args.resume and train_stage_completed(
        trifecta_checkpoint,
        "flow_model",
        [artifacts["flow_model_path"], artifacts["flow_meta_path"]],
    ):
        progress("skipping flow model: checkpoint and artifacts match")
        flow_model, flow_classes = load_flow_artifacts(config)
    else:
        progress("training flow model")
        flow_model, flow_classes = train_flow_model(train_df, valid_tune_df, feature_columns, categorical_columns, config)
        save_flow_model(flow_model, flow_classes, artifacts["flow_model_path"], artifacts["flow_meta_path"])
        mark_train_stage_completed(
            trifecta_checkpoint_path,
            trifecta_checkpoint,
            "flow_model",
            {"train_races": race_count(train_df), "valid_tune_races": race_count(valid_tune_df)},
        )
    collect_garbage()
    if args.resume and train_stage_completed(trifecta_checkpoint, "staged_models", [artifacts["staged_dir"]]):
        progress("skipping staged models: checkpoint and artifacts match")
        staged_models = load_staged_model_artifacts(config)
        if not staged_models:
            progress("staged model artifacts are empty; retraining")
            staged_models = train_staged_models(train_df, valid_tune_df, feature_columns, categorical_columns, config)
            save_staged_models(staged_models, artifacts["staged_dir"])
            mark_train_stage_completed(
                trifecta_checkpoint_path,
                trifecta_checkpoint,
                "staged_models",
                {"model_count": len(staged_models)},
            )
    else:
        progress("training staged models")
        staged_models = train_staged_models(train_df, valid_tune_df, feature_columns, categorical_columns, config)
        save_staged_models(staged_models, artifacts["staged_dir"])
        mark_train_stage_completed(
            trifecta_checkpoint_path,
            trifecta_checkpoint,
            "staged_models",
            {"model_count": len(staged_models)},
        )
    collect_garbage()
    blend_metrics = trifecta_checkpoint.get("metrics", {}).get("trifecta_v2_blend_weight", {})
    if args.resume and train_stage_completed(trifecta_checkpoint, "trifecta_v2_blend_weight") and "v1_weight" in blend_metrics:
        trifecta_v2_v1_weight = float(blend_metrics["v1_weight"])
        progress(f"skipping trifecta v2 blend weight: checkpoint match, v1_weight={trifecta_v2_v1_weight:.4g}")
    else:
        progress("optimizing trifecta v2 blend weight")
        trifecta_v2_v1_weight = optimize_trifecta_v2_blend_weight(
            models,
            ensemble_weights,
            eval_valid_tune_df,
            feature_columns,
            categorical_columns,
            classifier_models=classifier_models,
            flow_model=flow_model,
            flow_classes=flow_classes,
            staged_models=staged_models,
        )
        mark_train_stage_completed(
            trifecta_checkpoint_path,
            trifecta_checkpoint,
            "trifecta_v2_blend_weight",
            {"v1_weight": float(trifecta_v2_v1_weight), "eval_valid_tune_races": race_count(eval_valid_tune_df)},
        )
    ensemble_weights["trifecta_v2_v1_weight"] = trifecta_v2_v1_weight
    progress(f"optimized trifecta v2 blend weight: v1_weight={trifecta_v2_v1_weight:.4g}")
    collect_garbage()
    if args.resume and train_stage_completed(
        trifecta_checkpoint,
        "trifecta_v2_model",
        [artifacts["trifecta_v2_phase2_model_path"]],
    ):
        progress("skipping trifecta v2 model: checkpoint and artifacts match")
        trifecta_v2_model = load_trifecta_v2_model_artifact_payload(artifacts["trifecta_v2_phase2_model_path"])
    else:
        progress(f"training trifecta v2 model: max_races={args.max_races}")
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
        save_trifecta_v2_model_artifact(trifecta_v2_model, artifacts["trifecta_v2_phase2_model_path"])
        mark_train_stage_completed(
            trifecta_checkpoint_path,
            trifecta_checkpoint,
            "trifecta_v2_model",
            {"max_races": int(args.max_races)},
        )
    collect_garbage()
    if trifecta_v2_model is None:
        raise RuntimeError("Trifecta v2 model artifact could not be loaded for resume.")
    if args.resume and train_stage_completed(
        trifecta_checkpoint,
        "trifecta_v3_model",
        [artifacts["trifecta_v3_base_model_path"]],
    ):
        progress("skipping phase3 conditional trifecta model: checkpoint and artifacts match")
        trifecta_v3_model = load_trifecta_v2_model_artifact_payload(artifacts["trifecta_v3_base_model_path"])
    else:
        progress(f"training phase3 conditional trifecta model: max_races={args.max_races}")
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
        save_trifecta_v2_model_artifact(trifecta_v3_model, artifacts["trifecta_v3_base_model_path"])
        mark_train_stage_completed(
            trifecta_checkpoint_path,
            trifecta_checkpoint,
            "trifecta_v3_model",
            {"max_races": int(args.max_races)},
        )
    if trifecta_v3_model is None:
        raise RuntimeError("Trifecta v3 model artifact could not be loaded for resume.")
    collect_garbage()
    rerank_optimization = {}
    if args.optimize_rerank and not eval_valid_tune_df.empty:
        rerank_checkpoint_path = artifacts["rerank_optimization_checkpoint_path"]
        if args.reset_rerank_optimization:
            checkpoint_pattern = f"{rerank_checkpoint_path.stem}*{rerank_checkpoint_path.suffix}"
            for existing_checkpoint in rerank_checkpoint_path.parent.glob(checkpoint_pattern):
                progress(f"resetting rerank optimization checkpoint: {existing_checkpoint}")
                existing_checkpoint.unlink()
        progress(f"optimizing rerank settings: checkpoint={rerank_checkpoint_path}")
        rerank_optimization = optimize_rerank_inference_settings_two_stage(
            models,
            ensemble_weights,
            eval_valid_tune_df,
            feature_columns,
            categorical_columns,
            classifier_models=classifier_models,
            flow_model=flow_model,
            flow_classes=flow_classes,
            staged_models=staged_models,
            trifecta_v2_model=trifecta_v3_model,
            config=config,
            checkpoint_path=rerank_checkpoint_path,
            progress_callback=progress,
            workers=args.optimize_rerank_workers,
        )
        optimized_top_n = int(rerank_optimization.get("best_top_n", eval_rerank_top_n))
        optimized_weight = float(rerank_optimization.get("best_conservative_weight", 0.92))
        optimized_penalty = float(rerank_optimization.get("best_rank_penalty_strength", 0.0))
        progress(
            "optimized rerank settings: "
            f"top_n={optimized_top_n}, weight={optimized_weight:.4g}, penalty={optimized_penalty:.4g}"
        )
        eval_rerank_top_n = optimized_top_n
        trifecta_v3_model = with_conservative_rerank_weight(trifecta_v3_model, optimized_weight)
        trifecta_v3_model = with_rank_penalty_settings(trifecta_v3_model, optimized_penalty, 5)
    elif args.optimize_rerank:
        progress("skipping rerank optimization: no evaluation races")
    else:
        progress("skipping rerank optimization: --optimize-rerank not set")
    trifecta_v3_model = with_rerank_top_n(trifecta_v3_model, eval_rerank_top_n)
    progress("optimizing dynamic rerank weight rules")
    dynamic_weight_optimization = optimize_dynamic_rerank_weights(
        models,
        ensemble_weights,
        eval_valid_tune_df,
        feature_columns,
        categorical_columns,
        classifier_models=classifier_models,
        flow_model=flow_model,
        flow_classes=flow_classes,
        staged_models=staged_models,
        trifecta_v2_model=trifecta_v3_model,
        rerank_top_n=eval_rerank_top_n,
        config=config,
        progress_callback=progress,
    )
    trifecta_v3_model = with_dynamic_rerank_weight_metadata(trifecta_v3_model, dynamic_weight_optimization)
    calibration_search_df = eval_valid_tune_df if not eval_valid_tune_df.empty else valid_tune_df
    progress(
        "optimizing calibration window: "
        f"rerank_top_n={eval_rerank_top_n}, races={race_count(calibration_search_df)}"
    )
    calibration_optimization = optimize_phase3_calibration_window(
        models,
        ensemble_weights,
        calibration_search_df,
        feature_columns,
        categorical_columns,
        classifier_models=classifier_models,
        flow_model=flow_model,
        flow_classes=flow_classes,
        staged_models=staged_models,
        trifecta_v2_model=trifecta_v3_model,
        rerank_top_n=eval_rerank_top_n,
        config=config,
    )
    calibration_window_days = int(calibration_optimization.get("best_window_days", 60))
    progress(f"optimized calibration window: days={calibration_window_days}")
    trifecta_v3_model = with_calibration_window_days(trifecta_v3_model, calibration_window_days)
    trifecta_v3_model = with_phase3_optimization_metadata(
        trifecta_v3_model,
        rerank_optimization=rerank_optimization,
        calibration_optimization=calibration_optimization,
    )
    v2_calibrator_metrics = trifecta_checkpoint.get("metrics", {}).get("trifecta_v2_calibrator", {})
    if (
        args.resume
        and train_stage_completed(
            trifecta_checkpoint,
            "trifecta_v2_calibrator",
            [artifacts["trifecta_v2_calibrator_path"]],
        )
        and int(v2_calibrator_metrics.get("rerank_top_n", -1)) == int(eval_rerank_top_n)
    ):
        progress("skipping trifecta v2 calibrator: checkpoint and artifacts match")
        trifecta_v2_calibrator = load_optional_trifecta_calibrator(artifacts["trifecta_v2_calibrator_path"])
    else:
        progress("fitting trifecta v2 calibrator")
        trifecta_v2_calibrator = fit_model_trifecta_calibrator(
            models,
            ensemble_weights,
            valid_tune_df,
            feature_columns,
            categorical_columns,
            classifier_models=classifier_models,
            flow_model=flow_model,
            flow_classes=flow_classes,
            staged_models=staged_models,
            trifecta_v2_model=trifecta_v2_model,
            use_v2=True,
            rerank_top_n=eval_rerank_top_n,
        )
        save_trifecta_calibrator_artifact(trifecta_v2_calibrator, artifacts["trifecta_v2_calibrator_path"])
        mark_train_stage_completed(
            trifecta_checkpoint_path,
            trifecta_checkpoint,
            "trifecta_v2_calibrator",
            {"rerank_top_n": int(eval_rerank_top_n), "valid_tune_races": race_count(valid_tune_df)},
        )
    v3_calibrator_metrics = trifecta_checkpoint.get("metrics", {}).get("trifecta_v3_calibrator", {})
    if (
        args.resume
        and train_stage_completed(
            trifecta_checkpoint,
            "trifecta_v3_calibrator",
            [artifacts["trifecta_v3_calibrator_path"]],
        )
        and int(v3_calibrator_metrics.get("rerank_top_n", -1)) == int(eval_rerank_top_n)
        and int(v3_calibrator_metrics.get("calibration_window_days", -1)) == int(calibration_window_days)
    ):
        progress("skipping trifecta v3 calibrator: checkpoint and artifacts match")
        trifecta_v3_calibrator = load_optional_trifecta_calibrator(artifacts["trifecta_v3_calibrator_path"])
    else:
        progress("fitting trifecta v3 calibrator")
        trifecta_v3_calibrator = fit_model_trifecta_calibrator(
            models,
            ensemble_weights,
            valid_tune_df,
            feature_columns,
            categorical_columns,
            classifier_models=classifier_models,
            flow_model=flow_model,
            flow_classes=flow_classes,
            staged_models=staged_models,
            trifecta_v2_model=trifecta_v3_model,
            use_v2=True,
            rerank_top_n=eval_rerank_top_n,
            calibration_window_days=calibration_window_days,
        )
        save_trifecta_calibrator_artifact(trifecta_v3_calibrator, artifacts["trifecta_v3_calibrator_path"])
        mark_train_stage_completed(
            trifecta_checkpoint_path,
            trifecta_checkpoint,
            "trifecta_v3_calibrator",
            {
                "rerank_top_n": int(eval_rerank_top_n),
                "calibration_window_days": int(calibration_window_days),
                "valid_tune_races": race_count(valid_tune_df),
            },
        )

    progress("saving model artifacts")
    save_artifacts(
        models=models,
        feature_columns=feature_columns,
        metrics=json_load_or_empty(artifacts["metrics_path"]),
        trifecta_calibrator=trifecta_calibrator,
        catboost_model_path=artifacts["catboost_model_path"],
        lightgbm_model_path=artifacts["lightgbm_model_path"],
        xgboost_model_path=artifacts["xgboost_model_path"],
        random_forest_model_path=artifacts["random_forest_model_path"],
        ridge_model_path=artifacts["ridge_model_path"],
        neural_model_path=artifacts["neural_model_path"],
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

    progress("evaluating flow and staged metrics")
    metrics = json_load_or_empty(artifacts["metrics_path"])
    metrics["flow_model_metrics"] = evaluate_flow_model(
        flow_model,
        flow_classes,
        train_df,
        eval_report_df,
        eval_test_df,
        feature_columns,
        categorical_columns,
    )
    metrics["staged_model_metrics"] = evaluate_staged_models(
        staged_models,
        train_df,
        eval_report_df,
        eval_test_df,
        feature_columns,
        categorical_columns,
    )
    metrics["trifecta_evaluation_scope"] = {
        "valid_races": int(valid_df["race_id"].nunique()) if not valid_df.empty else 0,
        "valid_tune_races": int(valid_tune_df["race_id"].nunique()) if not valid_tune_df.empty else 0,
        "final_eval_races": int(final_eval_df["race_id"].nunique()) if not final_eval_df.empty else 0,
        "eval_valid_tune_races": int(eval_valid_tune_df["race_id"].nunique()) if not eval_valid_tune_df.empty else 0,
        "eval_final_eval_races": int(eval_final_df["race_id"].nunique()) if not eval_final_df.empty else 0,
        "test_races": int(eval_test_df["race_id"].nunique()) if not eval_test_df.empty else 0,
        "eval_max_races": int(args.eval_max_races),
        "final_eval_max_races": int(args.final_eval_max_races),
        "eval_rerank_top_n": int(eval_rerank_top_n),
        "final_eval_months": int(final_eval_months),
        "scenario_min_races": int(ensemble_weights.get("scenario_metric_min_races", 100)),
    }
    if rerank_optimization:
        metrics["rerank_optimization"] = rerank_optimization
    metrics["dynamic_rerank_weight_optimization"] = dynamic_weight_optimization
    metrics["calibration_optimization"] = calibration_optimization
    progress("evaluating trifecta metrics")
    v1_final_metrics = evaluate_trifecta(
        models,
        ensemble_weights,
        trifecta_calibrator,
        eval_report_df,
        feature_columns,
        categorical_columns,
        classifier_models=classifier_models,
        flow_model=flow_model,
        flow_classes=flow_classes,
        staged_models=staged_models,
        trifecta_v2_model=trifecta_v2_model,
        use_v2=False,
        rerank_top_n=eval_rerank_top_n,
    )
    v1_test_metrics = evaluate_trifecta(
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
        rerank_top_n=eval_rerank_top_n,
    )
    metrics["trifecta_v1_rerank_metrics"] = {
        "valid_calibrated": v1_final_metrics,
        "final_eval_calibrated": v1_final_metrics,
        "test_calibrated": v1_test_metrics,
    }
    v2_final_metrics = evaluate_trifecta(
        models,
        ensemble_weights,
        trifecta_v2_calibrator,
        eval_report_df,
        feature_columns,
        categorical_columns,
        classifier_models=classifier_models,
        flow_model=flow_model,
        flow_classes=flow_classes,
        staged_models=staged_models,
        trifecta_v2_model=trifecta_v2_model,
        use_v2=True,
        rerank_top_n=eval_rerank_top_n,
    )
    v2_test_metrics = evaluate_trifecta(
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
        rerank_top_n=eval_rerank_top_n,
    )
    metrics["trifecta_v2_metrics"] = {
        "valid_calibrated": v2_final_metrics,
        "final_eval_calibrated": v2_final_metrics,
        "test_calibrated": v2_test_metrics,
    }
    v3_final_metrics = evaluate_trifecta(
        models,
        ensemble_weights,
        trifecta_v3_calibrator,
        eval_report_df,
        feature_columns,
        categorical_columns,
        classifier_models=classifier_models,
        flow_model=flow_model,
        flow_classes=flow_classes,
        staged_models=staged_models,
        trifecta_v2_model=trifecta_v3_model,
        use_v2=True,
        rerank_top_n=eval_rerank_top_n,
    )
    v3_test_metrics = evaluate_trifecta(
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
        rerank_top_n=eval_rerank_top_n,
    )
    metrics["trifecta_v3_metrics"] = {
        "valid_calibrated": v3_final_metrics,
        "final_eval_calibrated": v3_final_metrics,
        "test_calibrated": v3_test_metrics,
    }
    progress(f"writing metrics: {artifacts['metrics_path']}")
    artifacts["metrics_path"].write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    progress("cleaning processed intermediate directories")
    cleanup_processed_intermediate_dirs(config)
    progress("completed")


def json_load_or_empty(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def get_final_eval_months(config: dict) -> int:
    return max(0, int(config.get("split", {}).get("final_eval_months", 1)))


def split_valid_for_final_eval(
    valid_df: pd.DataFrame,
    final_eval_months: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if valid_df.empty or final_eval_months <= 0 or "race_date" not in valid_df.columns:
        return valid_df.copy(), valid_df.iloc[0:0].copy()

    frame = valid_df.copy()
    race_dates = pd.to_datetime(frame["race_date"])
    latest_date = race_dates.max().normalize()
    final_start = (latest_date - pd.DateOffset(months=final_eval_months) + pd.Timedelta(days=1)).normalize()

    final_mask = race_dates >= final_start
    valid_tune_df = frame.loc[~final_mask].copy()
    final_eval_df = frame.loc[final_mask].copy()

    if valid_tune_df.empty:
        return frame, frame.iloc[0:0].copy()
    return valid_tune_df, final_eval_df


def sample_races_for_evaluation(df: pd.DataFrame, max_races: int) -> pd.DataFrame:
    if df.empty or max_races <= 0:
        return df.copy()
    races = df[["race_id", "race_date"]].drop_duplicates().sort_values(["race_date", "race_id"]).reset_index(drop=True)
    if len(races) <= max_races:
        return df.copy()
    indices = np.linspace(0, len(races) - 1, num=max_races, dtype=int)
    selected_ids = races.iloc[np.unique(indices)]["race_id"].tolist()
    return df[df["race_id"].isin(selected_ids)].copy()


def _numeric_feature_frame(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, dict[str, dict[str, float]], list[str], list[str]]:
    numeric_parts: dict[str, pd.Series] = {}
    feature_stats: dict[str, dict[str, float]] = {}
    dropped_non_numeric: list[str] = []
    dropped_constant: list[str] = []

    row_count = len(df)
    for column in feature_columns:
        if column not in df.columns:
            dropped_non_numeric.append(column)
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        non_null = int(values.notna().sum())
        if non_null == 0:
            dropped_non_numeric.append(column)
            continue
        std = float(values.std(skipna=True))
        if not np.isfinite(std) or std <= 0.0:
            dropped_constant.append(column)
            continue
        numeric_parts[column] = values.astype("float32")
        feature_stats[column] = {
            "mean": float(values.mean(skipna=True)),
            "std": std,
            "missing_rate": float(1.0 - (non_null / row_count)) if row_count else 0.0,
            "non_null_count": float(non_null),
        }

    return pd.DataFrame(numeric_parts, index=df.index), feature_stats, dropped_non_numeric, dropped_constant


def write_feature_correlation_outputs(
    df: pd.DataFrame,
    feature_columns: list[str],
    output_dir: Path,
    *,
    split: str,
    feature_set: str,
    sample_races: int,
    pair_threshold: float,
    max_pairs: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    target_columns = [
        column
        for column in ("finish_position", "target_rank", "is_win", "is_top2", "is_top3")
        if column in df.columns
    ]
    numeric_features, feature_stats, dropped_non_numeric, dropped_constant = _numeric_feature_frame(df, feature_columns)
    target_rows: list[dict[str, float | str]] = []
    if not numeric_features.empty:
        for target in target_columns:
            target_values = pd.to_numeric(df[target], errors="coerce")
            if int(target_values.notna().sum()) == 0:
                continue
            pearson = numeric_features.corrwith(target_values, method="pearson")
            spearman = numeric_features.corrwith(target_values, method="spearman")
            for feature in numeric_features.columns:
                stats = feature_stats[feature]
                pearson_value = pearson.get(feature, np.nan)
                spearman_value = spearman.get(feature, np.nan)
                target_rows.append(
                    {
                        "feature": feature,
                        "target": target,
                        "pearson": float(pearson_value) if pd.notna(pearson_value) else np.nan,
                        "spearman": float(spearman_value) if pd.notna(spearman_value) else np.nan,
                        "abs_pearson": float(abs(pearson_value)) if pd.notna(pearson_value) else np.nan,
                        "abs_spearman": float(abs(spearman_value)) if pd.notna(spearman_value) else np.nan,
                        "feature_mean": stats["mean"],
                        "feature_std": stats["std"],
                        "feature_missing_rate": stats["missing_rate"],
                        "feature_non_null_count": stats["non_null_count"],
                    }
                )

    target_df = pd.DataFrame(target_rows)
    if not target_df.empty:
        target_df = target_df.sort_values(
            ["target", "abs_spearman", "abs_pearson", "feature"],
            ascending=[True, False, False, True],
            na_position="last",
        )
    target_path = output_dir / "feature_correlation_targets.csv"
    target_df.to_csv(target_path, index=False)

    pair_rows: list[dict[str, float | str]] = []
    high_pair_count = 0
    if numeric_features.shape[1] >= 2:
        corr = numeric_features.corr(method="pearson")
        columns = corr.columns.to_numpy()
        values = corr.to_numpy()
        upper_i, upper_j = np.triu_indices(len(columns), k=1)
        selected = np.where(np.abs(values[upper_i, upper_j]) >= pair_threshold)[0]
        high_pair_count = int(len(selected))
        if len(selected) > 0:
            order = np.argsort(np.abs(values[upper_i[selected], upper_j[selected]]))[::-1]
            for index in selected[order[:max_pairs]]:
                left = columns[upper_i[index]]
                right = columns[upper_j[index]]
                pearson_value = float(values[upper_i[index], upper_j[index]])
                pair_rows.append(
                    {
                        "feature_left": str(left),
                        "feature_right": str(right),
                        "pearson": pearson_value,
                        "abs_pearson": abs(pearson_value),
                    }
                )
    pair_path = output_dir / "feature_correlation_pairs.csv"
    pd.DataFrame(pair_rows).to_csv(pair_path, index=False)

    summary = {
        "split": split,
        "sample_races": int(sample_races),
        "race_count": race_count(df),
        "row_count": int(len(df)),
        "feature_set": feature_set,
        "feature_count": int(len(feature_columns)),
        "numeric_feature_count": int(numeric_features.shape[1]),
        "dropped_non_numeric_count": int(len(dropped_non_numeric)),
        "dropped_constant_count": int(len(dropped_constant)),
        "target_columns": target_columns,
        "pair_threshold": float(pair_threshold),
        "high_correlation_pair_count": high_pair_count,
        "reported_high_correlation_pair_count": int(len(pair_rows)),
        "target_correlation_rows": int(len(target_df)),
        "outputs": {
            "targets": str(target_path),
            "pairs": str(pair_path),
            "summary": str(output_dir / "feature_correlation_summary.json"),
        },
    }
    summary_path = output_dir / "feature_correlation_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def analyze_feature_correlation_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--split", choices=["train", "valid", "all"], default="valid")
    parser.add_argument("--sample-races", type=int, default=10000, help="0 means all selected races")
    parser.add_argument("--feature-set", type=str, default="full", help="full or legacy_20260712")
    parser.add_argument("--pair-threshold", type=float, default=0.95)
    parser.add_argument("--max-pairs", type=int, default=500)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    command_name = "boatrace-analyze-feature-correlation"
    print_progress(command_name, "loading config and training splits")
    config = load_config(args.config)
    train_df, valid_df, test_df, config = load_training_splits_from_parquet(
        Path(config["data"]["training_table"]),
        config,
    )
    if args.split == "train":
        df = train_df
    elif args.split == "valid":
        df = valid_df
    else:
        frames = [frame for frame in (train_df, valid_df, test_df) if not frame.empty]
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    del train_df, valid_df, test_df
    collect_garbage()

    print_progress(command_name, f"selected split={args.split}, races={race_count(df)}")
    if args.sample_races > 0:
        df = sample_races_for_evaluation(df, args.sample_races)
        print_progress(command_name, f"sampled races={race_count(df)}")
    print_progress(command_name, "applying prediction-time measurement proxies")
    df = apply_prediction_time_measurement_proxies(df)

    feature_columns = infer_feature_columns(df)
    categorical_columns = set(infer_categorical_columns(df, feature_columns))
    feature_columns = [column for column in feature_columns if column not in categorical_columns]
    feature_columns = select_feature_columns_for_set(feature_columns, args.feature_set)
    output_dir = args.output_dir or get_artifact_paths(config)["metrics_path"].parent
    print_progress(command_name, f"analyzing numeric features={len(feature_columns)}")
    summary = write_feature_correlation_outputs(
        df,
        feature_columns,
        output_dir,
        split=args.split,
        feature_set=args.feature_set,
        sample_races=args.sample_races,
        pair_threshold=args.pair_threshold,
        max_pairs=args.max_pairs,
    )
    print_progress(command_name, "completed")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def evaluate_trifecta_full_valid_main() -> None:
    raise SystemExit("boatrace-eval-trifecta-full was removed. Use boatrace-train metrics instead.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--chunk", type=str, default="month", choices=["month"])
    parser.add_argument("--date-from", type=str, default=None)
    parser.add_argument("--date-to", type=str, default=None)
    args = parser.parse_args()

    progress = print_eval_trifecta_full_progress
    progress("start")
    progress("loading config and training splits")
    config = load_config(args.config)
    artifacts = get_artifact_paths(config)
    train_df, valid_df, test_df = load_training_splits(Path(config["data"]["training_table"]), config)
    progress(
        "loaded splits: "
        f"train_races={race_count(train_df)}, valid_races={race_count(valid_df)}, "
        f"test_races={race_count(test_df)}"
    )
    if args.date_from is not None:
        progress(f"filtering date_from={args.date_from}")
        valid_df = valid_df[valid_df["race_date"] >= pd.Timestamp(args.date_from)].copy()
        if not test_df.empty and "race_date" in test_df.columns:
            test_df = test_df[test_df["race_date"] >= pd.Timestamp(args.date_from)].copy()
    if args.date_to is not None:
        progress(f"filtering date_to={args.date_to}")
        valid_df = valid_df[valid_df["race_date"] <= pd.Timestamp(args.date_to)].copy()
        if not test_df.empty and "race_date" in test_df.columns:
            test_df = test_df[test_df["race_date"] <= pd.Timestamp(args.date_to)].copy()
    final_eval_months = get_final_eval_months(config)
    valid_tune_df, final_eval_df = split_valid_for_final_eval(valid_df, final_eval_months)
    eval_valid_df = final_eval_df if not final_eval_df.empty else valid_tune_df
    progress(
        "evaluation scope: "
        f"valid_races={race_count(valid_df)}, valid_tune_races={race_count(valid_tune_df)}, "
        f"final_eval_races={race_count(final_eval_df)}, test_races={race_count(test_df)}, chunk={args.chunk}"
    )
    progress("inferring feature columns")
    schema_df = pd.concat(
        [train_df.head(200), valid_tune_df.head(200), final_eval_df.head(200), test_df.head(200)],
        ignore_index=True,
    )
    feature_columns = infer_feature_columns(schema_df)
    categorical_columns = infer_categorical_columns(schema_df, feature_columns)
    progress(f"inferred features: numeric_and_categorical={len(feature_columns)}, categorical={len(categorical_columns)}")

    progress("loading model artifacts")
    models = load_models(config)
    classifier_models = load_classifier_artifacts(config)
    flow_model, flow_classes = load_flow_artifacts(config)
    staged_models = load_staged_model_artifacts(config)
    trifecta_v3_model = load_trifecta_v2_model_artifact(config)
    ensemble_weights = load_ensemble_weights(artifacts["ensemble_weights_path"])
    ensemble_weights.setdefault(
        "scenario_metric_min_races",
        int(get_phase3_settings(config)["evaluation"].get("scenario_min_races", 100)),
    )
    v1_calibrator = load_trifecta_calibrator(artifacts["trifecta_calibrator_path"])
    v3_calibrator = load_optional_trifecta_calibrator(artifacts["trifecta_v3_calibrator_path"])
    rerank_top_n = get_rerank_top_n(trifecta_v3_model, get_default_rerank_top_n(config))
    progress(f"loaded model artifacts: rerank_top_n={rerank_top_n}")

    progress("evaluating final_eval chunks")
    valid_metrics = evaluate_trifecta_in_chunks(
        eval_valid_df,
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
        split_name="final_eval",
        progress_callback=progress,
    )
    progress("evaluating test chunks")
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
        split_name="test",
        progress_callback=progress,
    )

    progress(f"writing metrics: {artifacts['metrics_path']}")
    metrics = json_load_or_empty(artifacts["metrics_path"])
    metrics["trifecta_evaluation_scope"] = {
        "valid_races": int(valid_df["race_id"].nunique()) if not valid_df.empty else 0,
        "valid_tune_races": int(valid_tune_df["race_id"].nunique()) if not valid_tune_df.empty else 0,
        "final_eval_races": int(final_eval_df["race_id"].nunique()) if not final_eval_df.empty else 0,
        "test_races": int(test_df["race_id"].nunique()) if not test_df.empty else 0,
        "eval_rerank_top_n": rerank_top_n,
        "scenario_min_races": int(ensemble_weights.get("scenario_metric_min_races", 100)),
        "chunk": args.chunk,
        "evaluation_mode": "final_eval_chunked",
        "final_eval_months": int(final_eval_months),
        "date_from": args.date_from,
        "date_to": args.date_to,
    }
    metrics["trifecta_v1_rerank_metrics"] = valid_metrics["v1"]
    metrics["trifecta_v3_metrics"] = valid_metrics["phase3"]
    metrics["trifecta_v1_rerank_metrics"]["final_eval_calibrated"] = metrics["trifecta_v1_rerank_metrics"].get(
        "valid_calibrated", {}
    )
    metrics["trifecta_v3_metrics"]["final_eval_calibrated"] = metrics["trifecta_v3_metrics"].get(
        "valid_calibrated", {}
    )
    metrics["trifecta_v1_rerank_metrics"]["test_calibrated"] = test_metrics["v1"].get("valid_calibrated", {})
    metrics["trifecta_v3_metrics"]["test_calibrated"] = test_metrics["phase3"].get("valid_calibrated", {})
    artifacts["metrics_path"].write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    progress("completed")
    result = {
        "trifecta_evaluation_scope": metrics["trifecta_evaluation_scope"],
        "trifecta_v1_rerank_metrics": metrics["trifecta_v1_rerank_metrics"],
        "trifecta_v3_metrics": metrics["trifecta_v3_metrics"],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


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
        if key in {"coverage_races", "covered_races"}:
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
    split_name: str = "valid",
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, dict]:
    chunks = iter_month_chunks(df)
    v1_chunk_metrics: list[dict] = []
    v3_chunk_metrics: list[dict] = []
    if progress_callback is not None:
        progress_callback(f"{split_name}: chunks={len(chunks)}")
    for idx, chunk in enumerate(chunks, start=1):
        chunk_start = pd.to_datetime(chunk["race_date"]).min().date() if not chunk.empty else None
        chunk_end = pd.to_datetime(chunk["race_date"]).max().date() if not chunk.empty else None
        if progress_callback is not None:
            progress_callback(
                f"{split_name}: chunk {idx}/{len(chunks)} start "
                f"races={race_count(chunk)}, date_range={chunk_start}..{chunk_end}"
            )
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
        if progress_callback is not None:
            progress_callback(f"{split_name}: chunk {idx}/{len(chunks)} v1 complete")
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
        if progress_callback is not None:
            progress_callback(f"{split_name}: chunk {idx}/{len(chunks)} v3 complete")
    return {
        "v1": {"valid_calibrated": aggregate_metric_dicts(v1_chunk_metrics)},
        "phase3": {"valid_calibrated": aggregate_metric_dicts(v3_chunk_metrics)},
    }


def load_training_splits(training_table_path: Path, config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, valid_df, test_df, _synced_config = load_training_splits_from_parquet(training_table_path, config)
    return train_df, valid_df, test_df


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
