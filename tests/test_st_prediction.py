from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

from src.models.st_prediction import (
    DEFAULT_SETTINGS, ST_FEATURE_COLUMNS, _complete_races, add_predicted_st_features,
    chronological_oof_st, normal_st_targets, predict_experiment_scores,
    run_st_prediction_experiment, st_input_frame,
)


def race_frame(days=24, suffix="train"):
    rows = []
    for day, date in enumerate(pd.date_range("2025-01-01", periods=days)):
        for lane in range(1, 7):
            history = 0.10 + lane * 0.01 + (day % 3) * 0.002
            rows.append({
                "race_id": f"{day:03d}_{suffix}", "race_date": date,
                "lane": lane, "course": lane, "venue": "15",
                "racer_prev_avg_st": history, "start_timing": history,
                "racer_prev_avg_exhibition": 6.7, "exhibition_time": 6.7,
                "finish_position": (lane + day) % 6 + 1,
                "target_rank": 6 - ((lane + day) % 6 + 1),
                "_actual_st": history + (day % 2) * 0.01,
            })
    return pd.DataFrame(rows)


def test_st_allowlist_excludes_result_time_inputs():
    frame = race_frame()
    with pytest.raises(ValueError, match="non-pre-race"):
        st_input_frame(frame, ["start_timing"])
    with pytest.raises(ValueError, match="non-pre-race"):
        st_input_frame(frame, ["finish_position"])
    features = st_input_frame(frame, ["course", "racer_prev_avg_st"])
    assert features.shape == (144, 2)


def test_normal_st_targets_excludes_missing_flying_and_late_starts():
    frame = pd.DataFrame({
        "_actual_st": [0.15, -0.01, np.nan, 1.5, 0.02, 0.4],
        "finish_status": [None, None, None, None, "F", "L"],
    })
    targets = normal_st_targets(frame)
    assert targets[0] == 0.15
    assert np.isnan(targets[1:]).all()


def test_oof_is_strictly_chronological_and_never_splits_a_day():
    frame = race_frame()
    settings = {**DEFAULT_SETTINGS, "iterations": 2, "min_training_rows": 6, "num_threads": 1}
    predictions, folds = chronological_oof_st(frame, ["course", "racer_prev_avg_st"], settings, lambda _: None)
    assert len(folds) == 3
    assert np.isnan(predictions[:36]).all()
    assert np.isfinite(predictions[36:]).all()
    assert all(f["train_max_date"] < f["predict_min_date"] for f in folds)
    changed = frame.copy()
    changed.loc[changed.race_date >= "2025-01-07", "_actual_st"] += 0.2
    changed_predictions, _ = chronological_oof_st(changed, ["course", "racer_prev_avg_st"], settings, lambda _: None)
    np.testing.assert_array_equal(predictions[36:72], changed_predictions[36:72])


def test_relative_features_use_course_not_boat_and_preserve_source():
    frame = race_frame(days=1)
    frame["course"] = [2, 1, 3, 4, 5, 6]
    original = frame.copy(deep=True)
    augmented = add_predicted_st_features(frame, np.array([0.2, 0.1, 0.3, 0.4, 0.5, 0.6]))
    assert augmented.loc[0, "predicted_st_inside_diff"] == pytest.approx(0.1)
    assert augmented.loc[0, "predicted_st_outside_diff"] == pytest.approx(-0.1)
    assert np.isnan(augmented.loc[1, "predicted_st_inside_diff"])
    assert set(ST_FEATURE_COLUMNS).issubset(augmented)
    pd.testing.assert_frame_equal(frame, original)


def test_incomplete_tied_or_accident_races_are_excluded():
    frame = race_frame(days=4)
    frame.loc[0, "course"] = 2
    frame.loc[6, "finish_position"] = frame.loc[7, "finish_position"]
    frame.loc[12, "finish_status"] = "F"
    assert _complete_races(frame).race_id.unique().tolist() == ["003_train"]


def run_experiment(tmp_path, valid_shift=0.0, skip_evaluation=False):
    train = race_frame(days=32)
    valid = race_frame(days=32, suffix="valid")
    raw_train = train.pop("_actual_st").to_numpy()
    raw_valid = valid.pop("_actual_st").to_numpy() + valid_shift
    config = {
        "model": {"iterations": 2, "learning_rate": 0.05, "random_seed": 42},
        "models": {"st_prediction": {
            "enabled": True, "iterations": 2, "ranker_iterations": 2,
            "min_training_rows": 6, "num_threads": 1,
            "max_train_races": 20, "max_eval_races": 4,
        }},
    }
    features = ["lane", "course", "start_timing", "racer_prev_avg_st"]
    artifact = tmp_path / f"experiment_{valid_shift}_{skip_evaluation}.joblib"
    messages = []
    report = run_st_prediction_experiment(
        train, valid, raw_train, raw_valid, features, [], config, artifact,
        messages.append, skip_evaluation=skip_evaluation,
    )
    return report, joblib.load(artifact), valid, messages


def test_experiment_saves_paired_metrics_and_reusable_pipeline(tmp_path):
    report, bundle, valid, messages = run_experiment(tmp_path)
    assert report["status"] == "completed"
    assert report["baseline"]["race_count"] == report["with_predicted_st"]["race_count"] == 4
    assert report["st_training_races"] == 20
    assert report["warmup_excluded_races"] > 0
    assert report["st_evaluation"]["sample_count"] == 24
    assert pd.Timestamp(bundle["training_max_date"]) < pd.Timestamp(report["evaluation_cutoff"])
    assert any("top3_hit_rate" in message for message in messages)
    predictions = predict_experiment_scores(bundle, valid)
    assert predictions.shape == (len(valid),)
    assert np.isfinite(predictions).all()
    shuffled = valid.sample(frac=1, random_state=42)
    np.testing.assert_array_equal(predict_experiment_scores(bundle, shuffled), predictions[shuffled.index])
    assert report["delta"]["top12_hit_rate"] == pytest.approx(
        report["with_predicted_st"]["top12_hit_rate"] - report["baseline"]["top12_hit_rate"]
    )


def test_evaluation_targets_do_not_affect_fitted_models(tmp_path):
    _, first, valid, _ = run_experiment(tmp_path)
    _, changed, _, _ = run_experiment(tmp_path, valid_shift=0.1)
    np.testing.assert_array_equal(
        predict_experiment_scores(first, valid), predict_experiment_scores(changed, valid)
    )


def test_skip_evaluation_still_saves_models(tmp_path):
    report, bundle, _, _ = run_experiment(tmp_path, skip_evaluation=True)
    assert report["status"] == "evaluation_skipped"
    assert "baseline" not in report
    assert "st_model" in bundle and "ranker" in bundle


def test_empty_validation_is_reported_without_training(tmp_path):
    report = run_st_prediction_experiment(
        race_frame(), pd.DataFrame(), np.zeros(144), np.empty(0), [], [], {},
        tmp_path / "unused.joblib", lambda _: None,
    )
    assert report["status"] == "insufficient_data"
