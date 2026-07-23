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
