from __future__ import annotations

import numpy as np

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

