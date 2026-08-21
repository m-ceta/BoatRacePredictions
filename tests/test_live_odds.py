from __future__ import annotations

import pandas as pd
import pytest

from src.live import attach_odds_and_value, select_buy_candidates


def test_attach_odds_and_value_marks_buy_from_top3_hit_probability_label() -> None:
    trifecta = pd.DataFrame(
        {
            "race_id": ["R1"] * 4,
            "trifecta": ["1-2-3", "1-3-2", "2-1-3", "2-3-1"],
            "probability": [0.20, 0.12, 0.08, 0.15],
            "trifecta_darkhorse_score": [0.20, 0.10, 0.65, 0.70],
            "scenario_line_fit_score": [0.10, 0.10, 0.55, 0.60],
            "top12_confidence_score": [72.0] * 4,
            "top12_confidence_label": ["中"] * 4,
            "top3_hit_probability": [0.62, 0.22, 0.58, 0.42],
            "top3_hit_probability_label": ["high", "low", "high", "middle"],
        }
    )
    odds = {
        "1-2-3": 12.0,
        "1-3-2": 30.0,
        "2-1-3": 30.0,
        "2-3-1": 12.5,
    }

    enriched = attach_odds_and_value(trifecta, odds)

    assert "break_even_odds" not in enriched.columns
    assert "recommended_min_odds" not in enriched.columns
    assert "ticket_hint" in enriched.columns
    assert "is_darkhorse_candidate" in enriched.columns
    assert "top12_confidence_score" in enriched.columns
    assert "top12_confidence_label" in enriched.columns
    assert enriched.loc[enriched["trifecta"] == "1-2-3", "expected_value"].iloc[0] == pytest.approx(2.4)
    assert enriched.loc[enriched["trifecta"] == "1-2-3", "top12_confidence_score"].iloc[0] == pytest.approx(72.0)
    assert enriched.loc[enriched["trifecta"] == "1-2-3", "recommended_bet_amount"].iloc[0] == 300
    assert enriched.loc[enriched["trifecta"] == "1-2-3", "buy_decision"].iloc[0] == "買い"
    assert enriched.loc[enriched["trifecta"] == "1-3-2", "buy_decision"].iloc[0] == "見送り"
    assert enriched.loc[enriched["trifecta"] == "1-3-2", "recommended_bet_amount"].iloc[0] == 0
    assert enriched.loc[enriched["trifecta"] == "2-3-1", "buy_decision"].iloc[0] == "検討"
    assert enriched.loc[enriched["trifecta"] == "2-3-1", "recommended_bet_amount"].iloc[0] == 100
    assert enriched.loc[enriched["trifecta"] == "2-1-3", "buy_decision"].iloc[0] == "見送り"
    assert enriched.loc[enriched["trifecta"] == "2-1-3", "recommended_bet_amount"].iloc[0] == 0


def test_select_buy_candidates_returns_only_buy_rows_when_available() -> None:
    trifecta = pd.DataFrame(
        {
            "race_id": ["R1"] * 4,
            "trifecta": ["1-2-3", "1-3-2", "2-1-3", "2-3-1"],
            "probability": [0.20, 0.12, 0.08, 0.15],
            "trifecta_darkhorse_score": [0.20, 0.10, 0.65, 0.70],
            "scenario_line_fit_score": [0.10, 0.10, 0.55, 0.60],
            "top12_confidence_score": [72.0, 72.0, 72.0, 72.0],
            "top12_confidence_label": ["中", "中", "中", "中"],
            "top3_hit_probability": [0.62, 0.22, 0.58, 0.42],
            "top3_hit_probability_label": ["high", "low", "high", "middle"],
        }
    )
    odds = {
        "1-2-3": 12.0,
        "1-3-2": 6.0,
        "2-1-3": 12.0,
        "2-3-1": 12.5,
    }

    candidates = select_buy_candidates(attach_odds_and_value(trifecta, odds))

    assert candidates["trifecta"].tolist() == ["1-2-3", "2-3-1"]


def test_attach_odds_and_value_uses_adjusted_probability_for_expected_value() -> None:
    trifecta = pd.DataFrame(
        {
            "race_id": ["R1"],
            "trifecta": ["1-2-3"],
            "probability": [0.12],
            "adjusted_probability": [0.06],
            "trifecta_darkhorse_score": [0.20],
            "scenario_line_fit_score": [0.10],
            "top12_confidence_score": [72.0],
            "top12_confidence_label": ["中"],
        }
    )

    enriched = attach_odds_and_value(trifecta, {"1-2-3": 20.0})

    assert enriched["expected_value_probability"].iloc[0] == pytest.approx(0.06)
    assert enriched["expected_value"].iloc[0] == pytest.approx(1.2)


def test_attach_odds_and_value_keeps_top3_outside_ticket_at_zero_yen() -> None:
    rows = []
    odds = {}
    trifectas = [
        "1-2-3",
        "1-2-4",
        "1-2-5",
        "1-2-6",
        "1-3-2",
        "1-3-4",
        "1-3-5",
        "1-3-6",
        "1-4-2",
        "1-4-3",
        "1-4-5",
        "1-4-6",
        "2-1-3",
    ]
    for index, trifecta in enumerate(trifectas):
        probability = 0.20 - (index * 0.01)
        rows.append(
            {
                "race_id": "R1",
                "trifecta": trifecta,
                "probability": probability,
                "trifecta_darkhorse_score": 0.1,
                "scenario_line_fit_score": 0.1,
                "top12_confidence_score": 70.0,
                "top12_confidence_label": "中",
                "top3_hit_probability": 0.60,
                "top3_hit_probability_label": "high",
            }
        )
        odds[trifecta] = 30.0

    enriched = attach_odds_and_value(pd.DataFrame(rows), odds)
    last_trifecta = rows[-1]["trifecta"]

    assert enriched.loc[enriched["trifecta"] == last_trifecta, "prediction_rank"].iloc[0] == 13
    assert enriched.loc[enriched["trifecta"] == last_trifecta, "expected_value"].iloc[0] > 1.0
    assert enriched.loc[enriched["trifecta"] == last_trifecta, "recommended_bet_amount"].iloc[0] == 0
    assert enriched.loc[enriched["trifecta"] == last_trifecta, "buy_decision"].iloc[0] == "見送り"
