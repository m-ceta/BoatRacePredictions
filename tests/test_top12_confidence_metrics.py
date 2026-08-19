from __future__ import annotations

import pandas as pd

from src.evaluation.metrics import compute_trifecta_metrics
from src.top12_confidence import (
    apply_top12_probability_adjustment_table,
    attach_boat_top1_confidence_columns,
    attach_top12_confidence_columns,
    fit_top12_probability_adjustment_table,
)


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


def _permutation_race_rows(race_id: str, actual_trifecta: str, probabilities: dict[str, float]) -> list[dict[str, object]]:
    rows = []
    for first in range(1, 7):
        for second in range(1, 7):
            if second == first:
                continue
            for third in range(1, 7):
                if third in {first, second}:
                    continue
                trifecta = f"{first}-{second}-{third}"
                rows.append(
                    {
                        "race_id": race_id,
                        "trifecta": trifecta,
                        "probability": probabilities.get(trifecta, 0.001),
                        "is_actual": trifecta == actual_trifecta,
                        "trifecta_payout": 1200.0,
                    }
                )
    return rows


def test_attach_top12_confidence_columns_adds_race_level_score() -> None:
    probabilities = [0.30] + [0.04] * 11 + [0.002] * 108
    frame = pd.DataFrame(_race_rows("R1", 0, probabilities))

    scored = attach_top12_confidence_columns(frame)

    assert "top12_confidence_score" in scored.columns
    assert "top12_confidence_label" in scored.columns
    assert scored["top12_confidence_score"].nunique() == 1
    assert scored["top12_confidence_label"].iloc[0] == "高"


def test_attach_boat_top1_confidence_columns_adds_first_boat_score() -> None:
    rows = _permutation_race_rows(
        "R1",
        "1-2-3",
        {
            "1-2-3": 0.20,
            "1-3-2": 0.18,
            "1-2-4": 0.16,
            "2-1-3": 0.04,
            "3-1-2": 0.03,
        },
    )

    scored = attach_boat_top1_confidence_columns(pd.DataFrame(rows))

    assert "boat_top1_confidence_score" in scored.columns
    assert "boat_top1_confidence_label" in scored.columns
    assert scored["boat_top1_confidence_score"].nunique() == 1
    assert scored["predicted_first_boat"].iloc[0] == 1
    assert scored["predicted_first_boat_probability"].iloc[0] > 0.5


def test_probability_adjustment_table_adjusts_by_confidence_and_rank_band() -> None:
    rows = []
    for race_index in range(20):
        probabilities = [0.10] + [0.03] * 11 + [0.005] * 108
        rows.extend(_race_rows(f"R{race_index}", 0 if race_index < 4 else 30, probabilities))

    table = fit_top12_probability_adjustment_table(
        pd.DataFrame(rows),
        min_samples=1,
        factor_min=0.1,
        factor_max=3.0,
    )
    scored = attach_top12_confidence_columns(pd.DataFrame(_race_rows("PX", 0, [0.10] + [0.03] * 11 + [0.005] * 108)))
    adjusted = apply_top12_probability_adjustment_table(scored, table)

    assert "adjusted_probability" in adjusted.columns
    assert "probability_adjustment_factor" in adjusted.columns
    top_row = adjusted.sort_values("probability", ascending=False).iloc[0]
    assert top_row["probability_adjustment_factor"] > 1.0
    assert top_row["adjusted_probability"] > top_row["probability"]


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


def test_compute_trifecta_metrics_includes_boat_top1_confidence_metrics() -> None:
    rows = []
    rows.extend(
        _permutation_race_rows(
            "R1",
            "1-2-3",
            {
                "1-2-3": 0.20,
                "1-3-2": 0.18,
                "1-2-4": 0.16,
                "2-1-3": 0.04,
            },
        )
    )
    rows.extend(
        _permutation_race_rows(
            "R2",
            "3-1-2",
            {
                "1-2-3": 0.20,
                "1-3-2": 0.18,
                "1-2-4": 0.16,
                "3-1-2": 0.04,
            },
        )
    )

    metrics = compute_trifecta_metrics(pd.DataFrame(rows))

    confidence_metrics = metrics["boat_top1_confidence_metrics"]
    assert "high" in confidence_metrics
    assert confidence_metrics["high"]["race_count"] == 2.0
    assert confidence_metrics["high"]["boat_top1_hit_rate"] == 0.5
    assert confidence_metrics["high"]["top3_hit_rate"] == 0.5
    assert confidence_metrics["high"]["top12_hit_rate"] == 1.0
    assert confidence_metrics["high"]["top3_total_stake"] == 600.0
    assert confidence_metrics["high"]["top3_total_return"] == 1200.0
    assert confidence_metrics["high"]["top3_recovery_rate"] == 2.0

    matrix = metrics["top3_x_boat_top1_confidence_metrics"]
    assert matrix["high"]["high"]["race_count"] == 2.0
    assert matrix["high"]["high"]["boat_top1_hit_rate"] == 0.5
    assert matrix["high"]["high"]["top3_hit_rate"] == 0.5
    assert matrix["high"]["high"]["top12_hit_rate"] == 1.0
    assert matrix["high"]["high"]["top3_total_stake"] == 600.0
    assert matrix["high"]["high"]["top3_total_return"] == 1200.0
    assert matrix["high"]["high"]["top3_recovery_rate"] == 2.0
    assert matrix["high"]["high"]["mean_top3_confidence_score"] > 0.0
    assert matrix["high"]["high"]["mean_boat_top1_confidence_score"] > 0.0


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
    assert recovery["top1"]["hit_rate"] == 0.0
    assert recovery["top1"]["total_stake"] == 200.0
    assert recovery["top1"]["total_return"] == 0.0
    assert recovery["top3"]["hit_rate"] == 0.5
    assert recovery["top3"]["total_stake"] == 600.0
    assert recovery["top3"]["total_return"] == 900.0
    assert recovery["top3"]["recovery_rate"] == 1.5
    assert recovery["top8"]["hit_rate"] == 1.0
    assert recovery["top8"]["total_stake"] == 1600.0
    assert recovery["top8"]["total_return"] == 5900.0
    assert recovery["bottom8"]["hit_rate"] == 0.5
    assert recovery["bottom8"]["total_stake"] == 1600.0
    assert recovery["bottom8"]["total_return"] == 5000.0
    assert recovery["bottom6"]["hit_rate"] == 0.5
    assert recovery["bottom6"]["total_stake"] == 1200.0
    assert recovery["bottom6"]["total_return"] == 5000.0


def test_compute_trifecta_metrics_bottom_ticket_recovery_hits_lower_top12_predictions() -> None:
    rows = []
    for race_id, actual_index, payout in [
        ("R1", 4, 12000.0),
        ("R2", 6, 6000.0),
        ("R3", 11, 3000.0),
    ]:
        probabilities = [float(120 - index) for index in range(120)]
        total = sum(probabilities)
        probabilities = [probability / total for probability in probabilities]
        for row in _race_rows(race_id, actual_index, probabilities):
            row["trifecta_payout"] = payout
            rows.append(row)

    metrics = compute_trifecta_metrics(pd.DataFrame(rows))

    recovery = metrics["uniform_ticket_recovery_metrics"]
    assert recovery["bottom8"]["hit_rate"] == 1.0
    assert recovery["bottom8"]["total_stake"] == 2400.0
    assert recovery["bottom8"]["total_return"] == 21000.0
    assert recovery["bottom6"]["hit_rate"] == 2.0 / 3.0
    assert recovery["bottom6"]["total_stake"] == 1800.0
    assert recovery["bottom6"]["total_return"] == 9000.0


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


def test_compute_trifecta_metrics_includes_variable_ticket_recovery_metrics() -> None:
    high_confidence_probs = [0.30] + [0.04] * 11 + [0.002] * 108
    middle_confidence_probs = [0.08] + [0.025] * 11 + [0.005] * 108
    low_confidence_probs = [1.0 / 120.0] * 120
    rows = []
    for row in _race_rows("R1", 4, high_confidence_probs):
        row["trifecta_payout"] = 1000.0
        rows.append(row)
    for row in _race_rows("R2", 7, middle_confidence_probs):
        row["trifecta_payout"] = 3000.0
        rows.append(row)
    for row in _race_rows("R3", 0, low_confidence_probs):
        row["trifecta_payout"] = 10000.0
        rows.append(row)

    metrics = compute_trifecta_metrics(pd.DataFrame(rows))

    variable = metrics["variable_ticket_recovery_metrics"]
    assert variable["rule"] == {"high": "top3", "middle": "top3", "low": "skip"}
    assert variable["confidence_type"] == "top3"
    summary = variable["summary"]
    assert summary["race_count"] == 3.0
    assert summary["purchased_race_count"] == 2.0
    assert summary["purchase_rate"] == 2.0 / 3.0
    assert summary["average_ticket_count"] == (3.0 + 3.0) / 3.0
    assert summary["hit_rate"] == 0.0
    assert summary["overall_hit_rate"] == 0.0
    assert summary["total_stake"] == 600.0
    assert summary["total_return"] == 0.0
    assert summary["recovery_rate"] == 0.0
    assert variable["by_decision"]["skip"]["purchased_race_count"] == 0.0
    assert variable["by_decision"]["top3"]["hit_rate"] == 0.0

    top3_confidence = metrics["top3_confidence_metrics"]
    assert top3_confidence["high"]["top3_hit_rate"] == 0.0
    assert top3_confidence["middle"]["top3_hit_rate"] == 0.0
    assert top3_confidence["low"]["top3_hit_rate"] == 1.0

    confidence_strategy = metrics["top12_confidence_strategy_recovery_metrics"]
    assert set(confidence_strategy["high"]) == {"top1", "top3", "top5", "top8", "top12", "bottom8", "bottom6"}
    assert confidence_strategy["high"]["top1"]["hit_rate"] == 0.0
    assert confidence_strategy["high"]["top3"]["hit_rate"] == 0.0
    assert confidence_strategy["high"]["top5"]["hit_rate"] == 1.0
    assert confidence_strategy["high"]["top5"]["total_stake"] == 500.0
    assert confidence_strategy["high"]["top5"]["total_return"] == 1000.0
    assert confidence_strategy["middle"]["top5"]["hit_rate"] == 0.0
    assert confidence_strategy["middle"]["top8"]["hit_rate"] == 1.0
    assert confidence_strategy["middle"]["top8"]["total_stake"] == 800.0
    assert confidence_strategy["middle"]["top8"]["total_return"] == 3000.0
    assert confidence_strategy["low"]["top3"]["hit_rate"] == 1.0
    assert confidence_strategy["high"]["bottom8"]["hit_rate"] == 1.0
    assert confidence_strategy["high"]["bottom6"]["hit_rate"] == 0.0
    assert confidence_strategy["middle"]["bottom6"]["hit_rate"] == 1.0
    assert confidence_strategy["low"]["bottom8"]["hit_rate"] == 0.0
