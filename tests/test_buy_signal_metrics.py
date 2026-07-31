from __future__ import annotations

import pandas as pd

from src.evaluation.metrics import compute_trifecta_metrics


def test_compute_trifecta_metrics_includes_buy_signal_top12_metrics() -> None:
    rows = []
    for race_id, actual_index, buy_score, buy_decision, label in [
        ("R1", 3, 72.0, "買い", "買い候補"),
        ("R2", 15, 48.0, "見送り", "見送り"),
    ]:
        for index in range(20):
            rows.append(
                {
                    "race_id": race_id,
                    "trifecta": f"1-2-{index}",
                    "probability": 1.0 - index * 0.01,
                    "is_actual": index == actual_index,
                    "expected_value": 1.1 if buy_decision == "買い" else 0.9,
                    "buy_score": buy_score if index == 0 else 20.0,
                    "buy_score_label": label if index == 0 else "見送り",
                    "buy_decision": buy_decision if index == 0 else "見送り",
                }
            )

    metrics = compute_trifecta_metrics(pd.DataFrame(rows))

    buy_metrics = metrics["buy_signal_top12_metrics"]
    assert buy_metrics["by_buy_decision"]["buy"]["race_count"] == 1.0
    assert buy_metrics["by_buy_decision"]["buy"]["top12_hit_rate"] == 1.0
    assert buy_metrics["by_buy_decision"]["skip"]["race_count"] == 1.0
    assert buy_metrics["by_buy_decision"]["skip"]["top12_hit_rate"] == 0.0
    assert buy_metrics["by_score_label"]["buy"]["top12_hit_rate"] == 1.0
    assert buy_metrics["by_score_label"]["skip"]["top12_hit_rate"] == 0.0


def test_compute_trifecta_metrics_includes_payout_proxy_buy_score_top12_metrics() -> None:
    rows = []
    for race_id, actual_index, payout in [
        ("R1", 3, 5000.0),
        ("R2", 15, 500.0),
    ]:
        for index in range(20):
            rows.append(
                {
                    "race_id": race_id,
                    "trifecta": f"1-2-{index}",
                    "probability": 0.08 if race_id == "R1" and index == actual_index else 0.02,
                    "is_actual": index == actual_index,
                    "ticket_priority_score": 0.5,
                    "race_upset_score": 0.2,
                    "trifecta_payout": payout,
                }
            )

    metrics = compute_trifecta_metrics(pd.DataFrame(rows))

    proxy_metrics = metrics["payout_proxy_buy_score_top12_metrics"]
    assert proxy_metrics["strong_buy"]["race_count"] == 1.0
    assert proxy_metrics["strong_buy"]["top12_hit_rate"] == 1.0
    assert proxy_metrics["skip"]["race_count"] == 1.0
    assert proxy_metrics["skip"]["top12_hit_rate"] == 0.0
