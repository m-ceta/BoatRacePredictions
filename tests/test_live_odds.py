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
        }
    )
    odds = {
        "1-2-3": 10.0,  # EV 1.2, odds threshold passed
        "1-3-2": 6.0,   # EV 1.2, odds threshold failed
        "2-1-3": 12.0,  # EV 0.96, EV threshold failed
        "2-3-1": 11.5,  # EV 1.035, odds threshold passed
    }

    enriched = attach_odds_and_value(trifecta, odds)

    assert "break_even_odds" not in enriched.columns
    assert "recommended_min_odds" not in enriched.columns
    assert enriched.loc[enriched["trifecta"] == "1-2-3", "expected_value"].iloc[0] == pytest.approx(1.2)
    assert enriched.loc[enriched["trifecta"] == "1-2-3", "buy_decision"].iloc[0] == "買い"
    assert enriched.loc[enriched["trifecta"] == "1-3-2", "buy_decision"].iloc[0] == "見送り"
    assert enriched.loc[enriched["trifecta"] == "2-1-3", "buy_decision"].iloc[0] == "見送り"
    assert enriched.loc[enriched["trifecta"] == "2-3-1", "buy_decision"].iloc[0] == "買い"


def test_select_buy_candidates_returns_only_buy_rows_when_available() -> None:
    trifecta = pd.DataFrame(
        {
            "race_id": ["R1"] * 4,
            "trifecta": ["1-2-3", "1-3-2", "2-1-3", "2-3-1"],
            "probability": [0.12, 0.20, 0.08, 0.09],
        }
    )
    odds = {
        "1-2-3": 10.0,
        "1-3-2": 6.0,
        "2-1-3": 12.0,
        "2-3-1": 11.5,
    }

    candidates = select_buy_candidates(attach_odds_and_value(trifecta, odds))

    assert candidates["trifecta"].tolist() == ["1-2-3", "2-3-1"]
