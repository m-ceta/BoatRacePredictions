from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.api import load_bundle
from src.features.builder import build_training_table, save_processed_tables
from src.live import predict_today_race
from src.models.ranker import (
    load_config,
    load_ensemble_weights,
    load_feature_columns,
    load_models,
    predict_race_order,
    save_artifacts,
    train_ranker,
)
from src.parsers.bk_parser import parse_entry_file, parse_result_file


def build_dataset_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rowdata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    entries = []
    for path in sorted(args.rowdata.glob("B*.TXT")):
        entries.extend(item.to_dict() for item in parse_entry_file(path))

    results = []
    for path in sorted(args.rowdata.glob("K*.TXT")):
        results.extend(item.to_dict() for item in parse_result_file(path))

    entries_df = pd.DataFrame(entries)
    results_df = pd.DataFrame(results)
    training_table = build_training_table(entries_df, results_df)
    save_processed_tables(entries_df, results_df, training_table, args.output)


def train_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    training_table = pd.read_parquet(config["data"]["training_table"])
    training_table["race_date"] = pd.to_datetime(training_table["race_date"])

    models, feature_columns, metrics, trifecta_calibrator = train_ranker(training_table, config)
    artifacts = config["artifacts"]
    save_artifacts(
        models,
        feature_columns,
        metrics,
        trifecta_calibrator,
        Path(artifacts["catboost_model_path"]),
        Path(artifacts["lightgbm_model_path"]),
        Path(artifacts["features_path"]),
        Path(artifacts["ensemble_weights_path"]),
        Path(artifacts["trifecta_calibrator_path"]),
        Path(artifacts["metrics_path"]),
    )


def predict_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"))
    args = parser.parse_args()

    config = load_config(args.config)
    models = load_models(config)
    ensemble_weights = load_ensemble_weights(Path(config["artifacts"]["ensemble_weights_path"]))
    feature_columns = load_feature_columns(args.features)
    future_df = pd.read_csv(args.input)
    predictions = predict_race_order(models, feature_columns, future_df, ensemble_weights)
    predictions.to_csv(args.output, index=False, encoding="utf-8-sig")


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
