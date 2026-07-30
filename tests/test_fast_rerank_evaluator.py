from __future__ import annotations

import numpy as np
import pandas as pd

from src.models import ranker


def test_fast_rerank_payload_metrics_use_rank_blend_and_baseline_mrr() -> None:
    raw_v1 = np.asarray([0.5, 0.3, 0.2], dtype=float)
    rerank_scores = np.asarray([0.1, 0.2, 0.9], dtype=float)
    base_rank_score, update_rank_score, overflow, flat_update = ranker._fast_rank_components(
        raw_v1,
        rerank_scores,
        rank_penalty_start=2,
    )
    payload = ranker.FastRerankRacePayload(
        race_id="R1",
        raw_v1=raw_v1,
        base_rank_score=base_rank_score,
        update_rank_score=update_rank_score,
        rank_penalty_overflow=overflow,
        actual_index=2,
        flat_update=flat_update,
    )

    metrics = ranker._evaluate_fast_rerank_payloads(
        [payload],
        conservative_weight=0.0,
        rank_penalty_strength=0.0,
        use_v2=True,
    )

    assert metrics["top1_hit_rate"] == 1.0
    assert metrics["top12_hit_rate"] == 1.0
    assert np.isclose(metrics["log_loss"], -np.log(2.0 / 3.0))
    assert metrics["rerank_metrics"]["rerank_mrr"] == 1.0
    assert np.isclose(metrics["rerank_metrics"]["baseline_mrr"], 1.0 / 3.0)
    assert metrics["rerank_metrics"]["mean_rank_improvement"] == 2.0


def test_fast_calibrated_metrics_normalize_probabilities_per_race() -> None:
    frame = pd.DataFrame(
        {
            "race_id": ["R1", "R1", "R1", "R2", "R2", "R2"],
            "trifecta": ["1-2-3", "1-3-2", "2-1-3", "1-2-3", "2-1-3", "3-1-2"],
            "raw_probability_v1": [0.1, 0.7, 0.2, 0.2, 0.5, 0.3],
            "raw_probability_v2": [0.2, 0.6, 0.2, 0.1, 0.8, 0.1],
            "is_actual": [False, True, False, False, True, False],
        }
    )
    calibrator = ranker._fit_isotonic_from_raw(
        frame["raw_probability_v2"].to_numpy(dtype=float),
        frame["is_actual"].astype(int).to_numpy(dtype=float),
    )

    metrics = ranker._fast_calibrated_metrics_from_frame(
        frame,
        raw_col="raw_probability_v2",
        calibrator=calibrator,
        baseline_raw_col="raw_probability_v1",
    )

    assert metrics["race_count"] == 2.0
    assert metrics["covered_races"] == 2.0
    assert metrics["top1_hit_rate"] == 1.0
    assert metrics["top12_hit_rate"] == 1.0
    assert metrics["log_loss"] >= 0.0
    assert metrics["rerank_metrics"]["coverage_races"] == 2.0


def test_fast_v1_trifecta_metrics_match_dataframe_metrics() -> None:
    ranked = pd.DataFrame(
        {
            "race_id": ["R1"] * 6 + ["R2"] * 6,
            "lane": [1, 2, 3, 4, 5, 6] * 2,
            "course": [1, 2, 3, 4, 5, 6, 1, 3, 2, 4, 5, 6],
            "finish_position": [1, 2, 3, 4, 5, 6, 3, 1, 2, 4, 5, 6],
            "win_probability_like": [0.42, 0.21, 0.16, 0.09, 0.07, 0.05, 0.18, 0.32, 0.24, 0.12, 0.08, 0.06],
        }
    )

    fast = ranker._evaluate_fast_v1_trifecta_metrics_from_ranked(ranked, calibrator=None, weights={})
    trifecta = ranker.build_trifecta_prediction_frame(ranked, trifecta_calibrator=None, use_v2=False)
    trifecta["probability"] = trifecta["probability_v1"]
    expected = ranker.compute_trifecta_metrics(trifecta, probability_col="probability")

    assert fast is not None
    for key in (
        "race_count",
        "covered_races",
        "top1_hit_rate",
        "top3_hit_rate",
        "top5_hit_rate",
        "top10_hit_rate",
        "top12_hit_rate",
        "log_loss",
        "brier_score",
        "mean_actual_probability",
        "mean_top_probability",
    ):
        assert np.isclose(float(fast[key]), float(expected[key]))
    subset_metrics = fast["entry_course_subset_metrics"]
    assert subset_metrics["lane_course_match"]["race_count"] == 1.0
    assert subset_metrics["lane_course_mismatch"]["race_count"] == 1.0


def test_fast_v1_calibrator_matches_dataframe_training_payload() -> None:
    ranked = pd.DataFrame(
        {
            "race_id": ["R1"] * 6 + ["R2"] * 6,
            "lane": [1, 2, 3, 4, 5, 6] * 2,
            "finish_position": [1, 2, 3, 4, 5, 6, 3, 1, 2, 4, 5, 6],
            "win_probability_like": [0.42, 0.21, 0.16, 0.09, 0.07, 0.05, 0.18, 0.32, 0.24, 0.12, 0.08, 0.06],
        }
    )
    fast_calibrator = ranker.fit_trifecta_calibrator_fast_from_ranked(ranked)
    trifecta = ranker.build_trifecta_prediction_frame(ranked, trifecta_calibrator=None, use_v2=False)
    dataframe_calibrator = ranker._fit_isotonic_from_raw(
        trifecta["raw_probability_v1"].to_numpy(dtype=float),
        trifecta["is_actual"].astype(int).to_numpy(dtype=float),
    )
    probe = np.linspace(0.0, 0.5, 20)

    assert fast_calibrator is not None
    assert np.allclose(fast_calibrator.predict(probe), dataframe_calibrator.predict(probe))


def test_dynamic_rerank_candidate_diagnostic_records_deltas() -> None:
    metrics = {
        "race_count": 100.0,
        "covered_races": 100.0,
        "top1_hit_rate": 0.12,
        "top3_hit_rate": 0.24,
        "top5_hit_rate": 0.35,
        "top10_hit_rate": 0.52,
        "top12_hit_rate": 0.56,
        "log_loss": 2.03,
        "brier_score": 0.08,
        "rerank_metrics": {"rerank_mrr": 0.41},
    }

    diagnostic = ranker._dynamic_rerank_candidate_diagnostic(
        "attack_or_collapse",
        0.8,
        metrics,
        baseline_top12=0.54,
        baseline_log_loss=2.0,
        log_loss_max_delta=0.03,
        default_weight=0.9,
    )

    assert diagnostic["subset"] == "attack_or_collapse"
    assert diagnostic["weight"] == 0.8
    assert not diagnostic["is_default_weight"]
    assert np.isclose(diagnostic["top12_delta"], 0.02)
    assert np.isclose(diagnostic["log_loss_delta"], 0.03)
    assert diagnostic["log_loss_within_guard"]
    assert diagnostic["rerank_mrr"] == 0.41
