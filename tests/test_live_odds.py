from __future__ import annotations

import pandas as pd
import pytest

from src.live import attach_odds_and_value, select_buy_candidates


def test_attach_odds_and_value_marks_buy_when_ev_and_odds_thresholds_pass() -> None:
    trifecta = pd.DataFrame(
        {
            "race_id": ["R1"] * 4,
            "trifecta": ["1-2-3", "1-3-2", "2-1-3", "2-3-1"],
            "probability": [0.12, 0.20, 0.08, 0.09],
            "trifecta_darkhorse_score": [0.20, 0.10, 0.65, 0.70],
            "scenario_line_fit_score": [0.10, 0.10, 0.55, 0.60],
        }
    )
    odds = {
        "1-2-3": 12.0,  # EV 1.44, odds threshold passed
        "1-3-2": 6.0,   # EV 1.2, odds threshold failed
        "2-1-3": 12.0,  # EV 0.96, EV threshold failed
        "2-3-1": 12.5,  # EV 1.125, odds threshold passed
    }

    enriched = attach_odds_and_value(trifecta, odds)

    assert "break_even_odds" not in enriched.columns
    assert "recommended_min_odds" not in enriched.columns
    assert "ticket_hint" in enriched.columns
    assert "is_darkhorse_candidate" in enriched.columns
    assert "fair_odds" in enriched.columns
    assert "odds_value_ratio" in enriched.columns
    assert "buy_score" in enriched.columns
    assert "buy_score_label" in enriched.columns
    assert enriched.loc[enriched["trifecta"] == "1-2-3", "expected_value"].iloc[0] == pytest.approx(1.44)
    assert enriched.loc[enriched["trifecta"] == "1-2-3", "fair_odds"].iloc[0] == pytest.approx(1 / 0.12)
    assert enriched.loc[enriched["trifecta"] == "1-2-3", "buy_score"].iloc[0] >= 50.0
    assert enriched.loc[enriched["trifecta"] == "1-2-3", "buy_decision"].iloc[0] == "買い"
    assert enriched.loc[enriched["trifecta"] == "1-3-2", "buy_decision"].iloc[0] == "見送り"
    assert enriched.loc[enriched["trifecta"] == "1-3-2", "buy_score"].iloc[0] < 50.0
    assert enriched.loc[enriched["trifecta"] == "2-1-3", "buy_decision"].iloc[0] == "見送り"
    assert enriched.loc[enriched["trifecta"] == "2-3-1", "buy_decision"].iloc[0] == "買い"


def test_select_buy_candidates_returns_only_buy_rows_when_available() -> None:
    trifecta = pd.DataFrame(
        {
            "race_id": ["R1"] * 4,
            "trifecta": ["1-2-3", "1-3-2", "2-1-3", "2-3-1"],
            "probability": [0.12, 0.20, 0.08, 0.09],
            "trifecta_darkhorse_score": [0.20, 0.10, 0.65, 0.70],
            "scenario_line_fit_score": [0.10, 0.10, 0.55, 0.60],
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
