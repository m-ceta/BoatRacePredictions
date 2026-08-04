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
    assert confidence_metrics["high"]["race_rate"] == 0.5
    assert confidence_metrics["high"]["top12_hit_rate"] == 1.0
    assert confidence_metrics["low"]["race_count"] == 1.0
    assert confidence_metrics["low"]["race_rate"] == 0.5
    assert confidence_metrics["low"]["top12_hit_rate"] == 0.0


def test_compute_trifecta_metrics_includes_payout_band_metrics() -> None:
    rows = []
    for race_id, actual_index, payout in [
        ("R1", 3, 9000.0),
        ("R2", 15, 12000.0),
        ("R3", 6, 60000.0),
        ("R4", 30, 150000.0),
    ]:
        probabilities = [1.0 / 120.0] * 120
        for row in _race_rows(race_id, actual_index, probabilities):
            row["trifecta_payout"] = payout
            rows.append(row)

    metrics = compute_trifecta_metrics(pd.DataFrame(rows))

    payout_metrics = metrics["payout_band_metrics"]
    assert payout_metrics["lt_10000"]["race_count"] == 1.0
    assert payout_metrics["lt_10000"]["race_rate"] == 0.25
    assert payout_metrics["lt_10000"]["top12_hit_rate"] == 1.0
    assert payout_metrics["gte_10000_lt_50000"]["top12_hit_rate"] == 0.0
    assert payout_metrics["gte_50000_lt_100000"]["top12_hit_rate"] == 1.0
    assert payout_metrics["gte_100000"]["top12_hit_rate"] == 0.0


def test_compute_trifecta_metrics_includes_uniform_ticket_recovery_metrics() -> None:
    rows = []
    for race_id, actual_index, payout in [
        ("R1", 2, 900.0),
        ("R2", 6, 5000.0),
    ]:
        probabilities = [1.0 / 120.0] * 120
        for row in _race_rows(race_id, actual_index, probabilities):
            row["trifecta_payout"] = payout
            rows.append(row)

    metrics = compute_trifecta_metrics(pd.DataFrame(rows))

    recovery = metrics["uniform_ticket_recovery_metrics"]
    assert recovery["top3"]["hit_rate"] == 0.5
    assert recovery["top3"]["total_stake"] == 600.0
    assert recovery["top3"]["total_return"] == 900.0
    assert recovery["top3"]["recovery_rate"] == 1.5
    assert recovery["top8"]["hit_rate"] == 1.0
    assert recovery["top8"]["total_stake"] == 1600.0
    assert recovery["top8"]["total_return"] == 5900.0


def test_compute_trifecta_metrics_includes_confidence_recovery_metrics() -> None:
    high_confidence_probs = [0.30] + [0.04] * 11 + [0.002] * 108
    low_confidence_probs = [1.0 / 120.0] * 120
    rows = []
    for row in _race_rows("R1", 3, high_confidence_probs):
        row["trifecta_payout"] = 1200.0
        rows.append(row)
    for row in _race_rows("R2", 30, low_confidence_probs):
        row["trifecta_payout"] = 5000.0
        rows.append(row)

    metrics = compute_trifecta_metrics(pd.DataFrame(rows))

    recovery = metrics["top12_confidence_recovery_metrics"]
    assert recovery["high"]["race_count"] == 1.0
    assert recovery["high"]["hit_rate"] == 1.0
    assert recovery["high"]["total_stake"] == 1200.0
    assert recovery["high"]["total_return"] == 1200.0
    assert recovery["high"]["recovery_rate"] == 1.0
    assert recovery["low"]["race_count"] == 1.0
    assert recovery["low"]["hit_rate"] == 0.0
    assert recovery["low"]["total_return"] == 0.0
