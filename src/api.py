from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.features.builder import build_training_table, save_processed_tables
from src.live import TodayRacePrediction, predict_today_race
from src.models.ranker import (
    load_config,
    load_ensemble_weights,
    load_feature_columns,
    load_models,
    load_trifecta_calibrator,
    predict_race_order,
    predict_trifecta_probabilities,
    save_artifacts,
    train_ranker,
)
from src.parsers.bk_parser import parse_entry_file, parse_result_file


@dataclass(slots=True)
class BoatRaceModelBundle:
    config: dict[str, Any]
    models: dict[str, Any]
    feature_columns: list[str]
    ensemble_weights: dict[str, float]
    trifecta_calibrator: Any


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


def train_from_config(config_path: str | Path) -> dict[str, Any]:
    config = load_config(Path(config_path))
    training_table = pd.read_parquet(config["data"]["training_table"])
    training_table["race_date"] = pd.to_datetime(training_table["race_date"])

    models, feature_columns, metrics, trifecta_calibrator = train_ranker(training_table, config)
    artifacts = config["artifacts"]
    save_artifacts(
        models=models,
        feature_columns=feature_columns,
        metrics=metrics,
        trifecta_calibrator=trifecta_calibrator,
        catboost_model_path=Path(artifacts["catboost_model_path"]),
        lightgbm_model_path=Path(artifacts["lightgbm_model_path"]),
        features_path=Path(artifacts["features_path"]),
        ensemble_weights_path=Path(artifacts["ensemble_weights_path"]),
        trifecta_calibrator_path=Path(artifacts["trifecta_calibrator_path"]),
        metrics_path=Path(artifacts["metrics_path"]),
    )
    return metrics


def load_bundle(config_path: str | Path = Path("configs/train.yaml")) -> BoatRaceModelBundle:
    config = load_config(Path(config_path))
    artifacts = config["artifacts"]
    return BoatRaceModelBundle(
        config=config,
        models=load_models(config),
        feature_columns=load_feature_columns(Path(artifacts["features_path"])),
        ensemble_weights=load_ensemble_weights(Path(artifacts["ensemble_weights_path"])),
        trifecta_calibrator=load_trifecta_calibrator(Path(artifacts["trifecta_calibrator_path"])),
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
) -> pd.DataFrame:
    trifecta = predict_trifecta_probabilities(
        models=bundle.models,
        feature_columns=bundle.feature_columns,
        future_df=future_df,
        ensemble_weights=bundle.ensemble_weights,
        trifecta_calibrator=bundle.trifecta_calibrator,
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
