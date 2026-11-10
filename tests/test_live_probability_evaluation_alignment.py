from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.isotonic import IsotonicRegression

from src.models import ranker
from src.top12_confidence import apply_top12_probability_adjustment_table, sort_trifecta_candidates
from src.top3_hit_probability import (
    TOP3_HIT_PROBABILITY_FEATURES, attach_top3_hit_probability_columns,
    build_top3_hit_probability_training_frame,
)


class RecordingClassifier:
    def __init__(self):
        self.inputs = []

    def predict_proba(self, x):
        self.inputs.append(np.asarray(x).copy())
        probability = np.clip(np.asarray(x)[:, 0] / 120.0, 0.01, 0.99)
        return np.column_stack([1.0 - probability, probability])


def ranked_frame():
    return pd.DataFrame({
        "race_id": ["R1"] * 6 + ["R2"] * 6,
        "lane": list(range(1, 7)) * 2,
        "course": list(range(1, 7)) * 2,
        "finish_position": [1, 2, 3, 4, 5, 6, 3, 1, 2, 4, 5, 6],
        "win_probability_like": [.5, .2, .12, .08, .06, .04, .3, .2, .15, .14, .11, .10],
        "trifecta_payout": [1200.0] * 12,
    })


def correction_table():
    return {"default_factor": 1.0, "rules": [
        {"confidence_key": c, "rank_band": band, "factor": factor}
        for c in ("high", "middle", "low")
        for band, factor in (("top1", .25), ("top2_3", 2.0), ("top4_6", 1.5), ("outside_top12", .6))
    ]}


@pytest.mark.parametrize("ties", [False, True])
def test_fast_evaluation_uses_exact_live_features_and_corrected_rankings(ties):
    ranked = ranked_frame()
    calibrator = None
    if ties:
        calibrator = IsotonicRegression(out_of_bounds="clip").fit(
            [0.0, .001, .01, .02, .03, .1, 1.0], [0.0, .001, .04, .04, .04, .1, .2]
        )
    live = ranker.build_trifecta_prediction_frame(ranked, calibrator)
    live = apply_top12_probability_adjustment_table(live, correction_table())
    model = RecordingClassifier()
    payload = {"model": model, "feature_names": list(TOP3_HIT_PROBABILITY_FEATURES)}
    live = attach_top3_hit_probability_columns(live, payload, probability_col="adjusted_probability")
    live_inputs = np.vstack(model.inputs)
    model.inputs.clear()
    expected = ranker.compute_trifecta_metrics(live, probability_col="adjusted_probability")
    fast = ranker._evaluate_fast_v1_trifecta_metrics_from_ranked(
        ranked, calibrator, {}, payload, correction_table(),
    )
    np.testing.assert_allclose(np.vstack(model.inputs), live_inputs, rtol=0, atol=1e-12)
    for key in ("top1_hit_rate", "top3_hit_rate", "top12_hit_rate", "log_loss", "mean_top_probability"):
        assert fast[key] == pytest.approx(expected[key])
    summary = fast["top3_hit_probability_metrics"]
    assert summary["mean_probability"] == pytest.approx(expected["top3_hit_probability_metrics"]["mean_probability"])
    assert summary["actual_top3_hit_rate"] == pytest.approx(fast["top3_hit_rate"])
    assert fast["probability_source"] == "adjusted_probability"


def test_chunking_does_not_change_features_labels_or_probabilities():
    ranked = ranked_frame()
    live = ranker.build_trifecta_prediction_frame(ranked)
    probs = []
    actuals = []
    upsets = []
    tickets = ["-".join(str(x + 1) for x in p) for p in ranker.TRIFECTA_FAST_PERMUTATIONS]
    for _, group in live.groupby("race_id", sort=True):
        group = group.set_index("trifecta").loc[tickets]
        probs.append(group.probability.to_numpy())
        actuals.append(int(np.flatnonzero(group.is_actual)[0]))
        upsets.append(group.race_upset_score.iloc[0])
    model = RecordingClassifier()
    payload = {"model": model}
    args = (np.asarray(probs), np.asarray(actuals), ["R1", "R2"], np.asarray(upsets), correction_table(), payload)
    first, summary1 = ranker._prepare_live_evaluation_probabilities(*args, chunk_size=1)
    second, summary2 = ranker._prepare_live_evaluation_probabilities(*args, chunk_size=128)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first.sum(axis=1), 1.0)
    assert summary1 == summary2


def test_raw_evaluation_does_not_use_a_model_trained_on_adjusted_probabilities(monkeypatch):
    calls = []

    def record(*args, **kwargs):
        calls.append(kwargs)
        return {}

    monkeypatch.setattr(ranker, "evaluate_trifecta", record)
    ranker.evaluate_trifecta_v1_metrics(
        {}, {}, object(), pd.DataFrame(), pd.DataFrame(), [], [],
        top3_hit_probability_model={"model": object()}, probability_adjustment_table=correction_table(),
    )
    assert calls[0]["top3_hit_probability_model"] is None
    assert calls[2]["top3_hit_probability_model"] is None
    assert "probability_adjustment_table" not in calls[0]
    assert calls[1]["probability_adjustment_table"] == correction_table()
    assert calls[3]["probability_adjustment_table"] == correction_table()


def test_candidate_ties_are_independent_of_input_order():
    live = ranker.build_trifecta_prediction_frame(ranked_frame())
    live["probability"] = 1.0 / 120.0
    before, targets_before = build_top3_hit_probability_training_frame(live)
    after, targets_after = build_top3_hit_probability_training_frame(live.sample(frac=1, random_state=1))
    pd.testing.assert_frame_equal(before.sort_values("race_id").reset_index(drop=True), after.sort_values("race_id").reset_index(drop=True))
    assert dict(zip(before.race_id, targets_before)) == dict(zip(after.race_id, targets_after))
    assert sort_trifecta_candidates(live, "probability").iloc[0].trifecta == "1-2-3"


def test_dataframe_evaluation_also_uses_adjusted_probabilities(monkeypatch):
    ranked = ranked_frame()
    monkeypatch.setattr(ranker, "apply_prediction_time_measurement_proxies", lambda f: f)
    monkeypatch.setattr(ranker, "build_weighted_lane_probabilities", lambda *args, **kwargs: ranked)
    monkeypatch.setattr(ranker, "_evaluate_fast_v1_trifecta_metrics_from_ranked", lambda *args, **kwargs: None)
    payload = {"model": RecordingClassifier()}
    result = ranker.evaluate_trifecta(
        {}, {}, None, ranked, [], [], top3_hit_probability_model=payload,
        probability_adjustment_table=correction_table(),
    )
    live = ranker.build_trifecta_prediction_frame(ranked)
    live = apply_top12_probability_adjustment_table(live, correction_table())
    live = attach_top3_hit_probability_columns(live, payload, probability_col="adjusted_probability")
    expected = ranker.compute_trifecta_metrics(live, probability_col="adjusted_probability")
    assert result["top3_hit_probability_metrics"]["mean_probability"] == pytest.approx(
        expected["top3_hit_probability_metrics"]["mean_probability"]
    )
    assert result["top3_hit_rate"] == expected["top3_hit_rate"]


def test_resume_invalidates_only_old_probability_model_and_evaluation_stages():
    stages = ("lightgbm", "ensemble", "trifecta_v1_calibrator", "probability_adjustment",
              "st_prediction_experiment", "top3_hit_probability_model",
              "trifecta_v1_metrics", "trifecta_v1_model_metrics")
    checkpoint = {"completed": dict.fromkeys(stages, True), "metrics": {s: {} for s in stages}}
    removed = ranker.invalidate_legacy_probability_evaluation_stages(checkpoint)
    assert set(removed) == {"top3_hit_probability_model", "trifecta_v1_metrics", "trifecta_v1_model_metrics"}
    assert checkpoint["completed"]["lightgbm"]
    assert checkpoint["completed"]["ensemble"]
    assert checkpoint["completed"]["st_prediction_experiment"]
    checkpoint["top3_hit_probability_feature_version"] = ranker.TOP3_HIT_PROBABILITY_VERSION
    checkpoint["trifecta_evaluation_version"] = ranker.TRIFECTA_EVALUATION_VERSION
    checkpoint["completed"].update(dict.fromkeys(removed, True))
    assert ranker.invalidate_legacy_probability_evaluation_stages(checkpoint) == []
