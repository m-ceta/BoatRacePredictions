from __future__ import annotations

import pandas as pd

from src.evaluation.metrics import compute_trifecta_metrics
from src.top12_confidence import attach_top12_confidence_columns


def _race_rows(race_id: str, actual_index: int, probabilities: list[float]) -> list[dict[str, object]]:
    return [
        {
            "race_id": race_id,
            "trifecta": f"1-2-{index + 1}",
            "probability": probability,
            "is_actual": index == actual_index,
        }
        for index, probability in enumerate(probabilities)
    ]


def test_attach_top12_confidence_columns_adds_race_level_score() -> None:
    probabilities = [0.30] + [0.04] * 11 + [0.002] * 108
    frame = pd.DataFrame(_race_rows("R1", 0, probabilities))

    scored = attach_top12_confidence_columns(frame)

    assert "top12_confidence_score" in scored.columns
    assert "top12_confidence_label" in scored.columns
    assert scored["top12_confidence_score"].nunique() == 1
    assert scored["top12_confidence_label"].iloc[0] == "高"


def test_compute_trifecta_metrics_includes_top12_confidence_metrics() -> None:
    high_confidence_probs = [0.30] + [0.04] * 11 + [0.002] * 108
    low_confidence_probs = [1.0 / 120.0] * 120
    rows = (
        _race_rows("R1", 3, high_confidence_probs)
        + _race_rows("R2", 30, low_confidence_probs)
    )

    metrics = compute_trifecta_metrics(pd.DataFrame(rows))

    confidence_metrics = metrics["top12_confidence_metrics"]
    assert confidence_metrics["high"]["race_count"] == 1.0
    assert confidence_metrics["high"]["top12_hit_rate"] == 1.0
    assert confidence_metrics["low"]["race_count"] == 1.0
    assert confidence_metrics["low"]["top12_hit_rate"] == 0.0
