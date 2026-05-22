from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.models.ranker import (
    evaluate_trifecta,
    fit_model_trifecta_calibrator,
    get_artifact_paths,
    infer_categorical_columns,
    load_classifier_artifacts,
    load_config,
    load_ensemble_weights,
    load_feature_columns,
    load_flow_artifacts,
    load_models,
    load_optional_trifecta_calibrator,
    load_staged_model_artifacts,
    load_trifecta_calibrator,
    load_trifecta_v2_model_artifact,
    prepare_training_table,
)


CALIBRATION_PERIOD = ("2026-01-01", "2026-03-31")
EVAL_PERIODS = [
    ("2026-04-01", "2026-04-30"),
    ("2026-05-01", "2026-05-13"),
]


def aggregate_metric_dicts(metric_dicts: list[dict]) -> dict:
    metric_dicts = [m for m in metric_dicts if m]
    if not metric_dicts:
        return {}
    out: dict[str, object] = {}
    keys = set().union(*(m.keys() for m in metric_dicts))
    for key in keys:
        vals = [m[key] for m in metric_dicts if key in m]
        if not vals:
            continue
        if isinstance(vals[0], dict):
            out[key] = aggregate_metric_dicts(vals)
            continue
        if key in {"race_count", "coverage_races"}:
            out[key] = float(sum(float(v) for v in vals))
            continue
        weight_key = "coverage_races" if key in {"rerank_top1_hit_rate", "rerank_mrr", "baseline_mrr", "mean_rank_improvement"} else "race_count"
        total_weight = 0.0
        weighted_sum = 0.0
        for item in metric_dicts:
            if key not in item:
                continue
            weight = float(item.get(weight_key, 0.0))
            weighted_sum += float(item[key]) * weight
            total_weight += weight
        out[key] = (weighted_sum / total_weight) if total_weight > 0 else float(sum(float(v) for v in vals) / len(vals))
    return out


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs" / "train.yaml"
    config = load_config(config_path)
    artifacts = get_artifact_paths(config)
    training_table_path = root / config["data"]["training_table"]

    models = load_models(config)
    weights = load_ensemble_weights(artifacts["ensemble_weights_path"])
    feature_columns = load_feature_columns(artifacts["features_path"])
    classifier_models = load_classifier_artifacts(config)
    flow_model, flow_classes = load_flow_artifacts(config)
    staged_models = load_staged_model_artifacts(config)
    phase3_model = load_trifecta_v2_model_artifact(config)
    rerank_top_n = int(config.get("inference", {}).get("trifecta_rerank_top_n", 10))

    monthly_results: list[dict] = []
    v1_metrics: list[dict] = []
    v3_metrics: list[dict] = []
    progress_path = root / "artifacts" / "full_valid_progress.json"

    calibration_chunk = pd.read_parquet(
        training_table_path,
        filters=[
            ("race_date", ">=", pd.Timestamp(CALIBRATION_PERIOD[0])),
            ("race_date", "<=", pd.Timestamp(CALIBRATION_PERIOD[1])),
        ],
    )
    calibration_chunk = prepare_training_table(calibration_chunk, config)
    calibration_categorical_columns = infer_categorical_columns(calibration_chunk.head(200), feature_columns)
    v1_cal = fit_model_trifecta_calibrator(
        models=models,
        weights=weights,
        valid_df=calibration_chunk,
        feature_columns=feature_columns,
        categorical_columns=calibration_categorical_columns,
        classifier_models=classifier_models,
        flow_model=flow_model,
        flow_classes=flow_classes,
        staged_models=staged_models,
        trifecta_v2_model=phase3_model,
        use_v2=False,
        rerank_top_n=rerank_top_n,
    )
    v3_cal = fit_model_trifecta_calibrator(
        models=models,
        weights=weights,
        valid_df=calibration_chunk,
        feature_columns=feature_columns,
        categorical_columns=calibration_categorical_columns,
        classifier_models=classifier_models,
        flow_model=flow_model,
        flow_classes=flow_classes,
        staged_models=staged_models,
        trifecta_v2_model=phase3_model,
        use_v2=True,
        rerank_top_n=rerank_top_n,
        calibration_window_days=60,
    )

    for start, end in EVAL_PERIODS:
        chunk = pd.read_parquet(
            training_table_path,
            filters=[("race_date", ">=", pd.Timestamp(start)), ("race_date", "<=", pd.Timestamp(end))],
        )
        chunk = prepare_training_table(chunk, config)
        categorical_columns = infer_categorical_columns(chunk.head(200), feature_columns)
        common = dict(
            models=models,
            weights=weights,
            df=chunk,
            feature_columns=feature_columns,
            categorical_columns=categorical_columns,
            classifier_models=classifier_models,
            flow_model=flow_model,
            flow_classes=flow_classes,
            staged_models=staged_models,
            rerank_top_n=rerank_top_n,
        )
        v1 = evaluate_trifecta(calibrator=v1_cal, trifecta_v2_model=phase3_model, use_v2=False, **common)
        v3 = evaluate_trifecta(calibrator=v3_cal, trifecta_v2_model=phase3_model, use_v2=True, **common)
        record = {"start": start, "end": end, "v1": v1, "phase3": v3}
        monthly_results.append(record)
        v1_metrics.append(v1)
        v3_metrics.append(v3)
        progress_path.write_text(
            json.dumps(
                {
                    "calibration_period": CALIBRATION_PERIOD,
                    "completed": monthly_results,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    metrics = json.loads((root / artifacts["metrics_path"]).read_text(encoding="utf-8"))
    metrics["trifecta_evaluation_scope"] = {
        "valid_races": int(sum(item.get("phase3", {}).get("race_count", 0.0) for item in monthly_results)),
        "test_races": 0,
        "eval_rerank_top_n": rerank_top_n,
        "evaluation_mode": "leakage_free_calibrated_monthly_eval",
        "calibration_period": CALIBRATION_PERIOD,
        "periods": EVAL_PERIODS,
    }
    metrics["trifecta_v1_rerank_metrics"] = {"valid_calibrated": aggregate_metric_dicts(v1_metrics), "test_calibrated": {}}
    metrics["trifecta_v3_metrics"] = {"valid_calibrated": aggregate_metric_dicts(v3_metrics), "test_calibrated": {}}
    metrics["trifecta_v3_monthly_breakdown"] = monthly_results
    (root / artifacts["metrics_path"]).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    progress_path.write_text(
        json.dumps(
            {
                "calibration_period": CALIBRATION_PERIOD,
                "completed": monthly_results,
                "finalized": True,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
