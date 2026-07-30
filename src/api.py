from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from src.features.builder import build_training_table, save_processed_tables
from src.features.streaming_builder import BuildSummary, build_training_table_streaming
from src.models.ranker import (
    cleanup_processed_intermediate_dirs,
    collect_garbage,
    get_artifact_paths,
    load_config,
    load_classifier_artifacts,
    load_ensemble_weights,
    load_feature_columns,
    load_models,
    load_trifecta_calibrator,
    predict_race_order,
    predict_trifecta_probabilities,
    save_artifacts,
    load_training_splits_from_parquet,
    train_ranker,
    train_ranker_from_splits,
)
from src.parsers.bk_parser import parse_entry_file, parse_result_file
from src.rowdata_sync import RowdataBackfillReport, backfill_rowdata

if TYPE_CHECKING:
    from src.live import TodayRacePrediction


@dataclass(slots=True)
class BoatRaceModelBundle:
    config: dict[str, Any]
    models: dict[str, Any]
    feature_columns: list[str]
    ensemble_weights: dict[str, float]
    trifecta_calibrator: Any
    classifier_models: dict[str, Any]


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


def build_dataset_from_rowdata_streaming(
    rowdata_dir: str | Path,
    output_dir: str | Path,
    min_date: Any | None = None,
    max_date: Any | None = None,
) -> BuildSummary:
    normalized_min_date = pd.Timestamp(min_date).date() if min_date is not None else None
    normalized_max_date = pd.Timestamp(max_date).date() if max_date is not None else None
    return build_training_table_streaming(
        rowdata_dir=Path(rowdata_dir),
        output_dir=Path(output_dir),
        min_date=normalized_min_date,
        max_date=normalized_max_date,
    )


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
        _flow_model,
        _flow_classes,
        _staged_models,
        _trifecta_v2_model,
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
        xgboost_model_path=artifacts["xgboost_model_path"],
        features_path=artifacts["features_path"],
        ensemble_weights_path=artifacts["ensemble_weights_path"],
        trifecta_calibrator_path=artifacts["trifecta_calibrator_path"],
        metrics_path=artifacts["metrics_path"],
        classifier_models=classifier_models,
        classifier_output_dir=artifacts["classifier_dir"],
    )
    cleanup_processed_intermediate_dirs(config)
    return metrics


def load_bundle(config_path: str | Path = Path("configs/train.yaml")) -> BoatRaceModelBundle:
    config = load_config(Path(config_path))
    artifacts = get_artifact_paths(config)
    return BoatRaceModelBundle(
        config=config,
        models=load_models(config),
        feature_columns=load_feature_columns(artifacts["features_path"]),
        ensemble_weights=load_ensemble_weights(artifacts["ensemble_weights_path"]),
        trifecta_calibrator=load_trifecta_calibrator(artifacts["trifecta_calibrator_path"]),
        classifier_models=load_classifier_artifacts(config),
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
    use_v2: bool = False,
    rerank_top_n: int | None = None,
) -> pd.DataFrame:
    calibrator = bundle.trifecta_calibrator
    trifecta = predict_trifecta_probabilities(
        models=bundle.models,
        feature_columns=bundle.feature_columns,
        future_df=future_df,
        ensemble_weights=bundle.ensemble_weights,
        trifecta_calibrator=calibrator,
        classifier_models=bundle.classifier_models,
        odds_df=odds_df,
        use_v2=False,
        rerank_top_n=None,
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
    course_overrides: Any | None = None,
) -> "TodayRacePrediction":
    from src.live import predict_today_race

    bundle = load_bundle(config_path)
    return predict_today_race(
        bundle=bundle,
        venue=venue,
        race_no=race_no,
        race_date=race_date,
        course_overrides=course_overrides,
    )
