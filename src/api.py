from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.features.builder import build_training_table, save_processed_tables
from src.live import TodayRacePrediction, predict_today_race
from src.models.ranker import (
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
    predict_race_order,
    predict_trifecta_probabilities,
    save_artifacts,
    train_ranker,
    infer_latest_available_race_date,
    with_latest_available_dates,
)
from src.parsers.bk_parser import parse_entry_file, parse_result_file
from src.rowdata_sync import RowdataBackfillReport, backfill_rowdata


@dataclass(slots=True)
class BoatRaceModelBundle:
    config: dict[str, Any]
    models: dict[str, Any]
    feature_columns: list[str]
    ensemble_weights: dict[str, float]
    trifecta_calibrator: Any
    trifecta_v2_calibrator: Any | None
    trifecta_v3_calibrator: Any | None
    classifier_models: dict[str, Any]
    flow_model: Any | None
    flow_classes: list[str] | None
    staged_models: dict[str, Any]
    trifecta_v2_model: Any | None
    rerank_top_n: int


def build_dataset_from_rowdata(
    rowdata_dir: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    rowdata_path = Path(rowdata_dir)
    entries = []
    for path in sorted(rowdata_path.glob("B*.TXT")):
        entries.extend(item.to_dict() for item in parse_entry_file(path))

    results = []
    for path in sorted(rowdata_path.glob("K*.TXT")):
        results.extend(item.to_dict() for item in parse_result_file(path))

    entries_df = pd.DataFrame(entries)
    results_df = pd.DataFrame(results)
    training_table = build_training_table(entries_df, results_df)

    if output_dir is not None:
        save_processed_tables(entries_df, results_df, training_table, Path(output_dir))

    return {
        "entries": entries_df,
        "results": results_df,
        "training_table": training_table,
    }


def backfill_rowdata_files(
    rowdata_dir: str | Path,
    start_date: Any | None = None,
    end_date: Any | None = None,
    kinds: str | tuple[str, ...] | list[str] = "BK",
    overwrite: bool = False,
) -> RowdataBackfillReport:
    normalized_start = pd.Timestamp(start_date).date() if start_date is not None else None
    normalized_end = pd.Timestamp(end_date).date() if end_date is not None else None
    return backfill_rowdata(
        rowdata_dir=rowdata_dir,
        start_date=normalized_start,
        end_date=normalized_end,
        kinds=kinds,
        overwrite=overwrite,
    )


def train_from_config(config_path: str | Path) -> dict[str, Any]:
    config = load_config(Path(config_path))
    training_table = pd.read_parquet(config["data"]["training_table"])
    training_table["race_date"] = pd.to_datetime(training_table["race_date"])
    config = with_latest_available_dates(config, infer_latest_available_race_date(training_table))

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
    ) = train_ranker(training_table, config)
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
    return metrics


def load_bundle(config_path: str | Path = Path("configs/train.yaml")) -> BoatRaceModelBundle:
    config = load_config(Path(config_path))
    artifacts = get_artifact_paths(config)
    flow_model, flow_classes = load_flow_artifacts(config)
    return BoatRaceModelBundle(
        config=config,
        models=load_models(config),
        feature_columns=load_feature_columns(artifacts["features_path"]),
        ensemble_weights=load_ensemble_weights(artifacts["ensemble_weights_path"]),
        trifecta_calibrator=load_trifecta_calibrator(artifacts["trifecta_calibrator_path"]),
        trifecta_v2_calibrator=load_optional_trifecta_calibrator(artifacts["trifecta_v2_calibrator_path"]),
        trifecta_v3_calibrator=load_optional_trifecta_calibrator(artifacts["trifecta_v3_calibrator_path"]),
        classifier_models=load_classifier_artifacts(config),
        flow_model=flow_model,
        flow_classes=flow_classes,
        staged_models=load_staged_model_artifacts(config),
        trifecta_v2_model=load_trifecta_v2_model_artifact(config),
        rerank_top_n=int(config.get("inference", {}).get("trifecta_rerank_top_n", 24)),
    )


def predict_ranking(bundle: BoatRaceModelBundle, future_df: pd.DataFrame) -> pd.DataFrame:
    return predict_race_order(
        models=bundle.models,
        feature_columns=bundle.feature_columns,
        future_df=future_df,
        ensemble_weights=bundle.ensemble_weights,
    )


def predict_trifecta(
    bundle: BoatRaceModelBundle,
    future_df: pd.DataFrame,
    top_n: int | None = 20,
    odds_df: pd.DataFrame | None = None,
    use_v2: bool = True,
    rerank_top_n: int | None = None,
) -> pd.DataFrame:
    calibrator = bundle.trifecta_calibrator
    if use_v2:
        phase_name = bundle.trifecta_v2_model.get("phase") if isinstance(bundle.trifecta_v2_model, dict) else None
        if phase_name == "phase3_conditional" and bundle.trifecta_v3_calibrator is not None:
            calibrator = bundle.trifecta_v3_calibrator
        elif bundle.trifecta_v2_calibrator is not None:
            calibrator = bundle.trifecta_v2_calibrator
    effective_rerank_top_n = bundle.rerank_top_n if rerank_top_n is None else rerank_top_n
    trifecta = predict_trifecta_probabilities(
        models=bundle.models,
        feature_columns=bundle.feature_columns,
        future_df=future_df,
        ensemble_weights=bundle.ensemble_weights,
        trifecta_calibrator=calibrator,
        classifier_models=bundle.classifier_models,
        flow_model=bundle.flow_model,
        flow_classes=bundle.flow_classes,
        staged_models=bundle.staged_models,
        trifecta_v2_model=bundle.trifecta_v2_model,
        odds_df=odds_df,
        use_v2=use_v2,
        rerank_top_n=effective_rerank_top_n if use_v2 else None,
    )
    if top_n is None:
        return trifecta
    return (
        trifecta.sort_values(["race_id", "probability"], ascending=[True, False])
        .groupby("race_id", sort=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def load_prediction_input(path: str | Path) -> pd.DataFrame:
    input_path = Path(path)
    if input_path.suffix.lower() == ".parquet":
        return pd.read_parquet(input_path)
    return pd.read_csv(input_path)


def predict_today(
    venue: str,
    race_no: int,
    config_path: str | Path = Path("configs/train.yaml"),
    race_date: Any | None = None,
) -> TodayRacePrediction:
    bundle = load_bundle(config_path)
    return predict_today_race(bundle=bundle, venue=venue, race_no=race_no, race_date=race_date)
