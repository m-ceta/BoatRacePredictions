from __future__ import annotations

import pandas as pd


BUY_EXPECTED_VALUE_THRESHOLD = 1.0
BUY_MIN_ODDS = 12.0


def attach_expected_value_columns(
    trifecta_df: pd.DataFrame,
    probability_col: str = "probability",
    odds_col: str = "odds",
    expected_value_threshold: float = 1.10,
    kelly_cap: float = 0.02,
) -> pd.DataFrame:
    frame = trifecta_df.copy()
    if frame.empty or odds_col not in frame.columns:
        if "expected_value" not in frame.columns:
            frame["expected_value"] = pd.NA
        if "market_rank" not in frame.columns:
            frame["market_rank"] = pd.NA
        if "is_value_bet" not in frame.columns:
            frame["is_value_bet"] = False
        if "stake_fraction" not in frame.columns:
            frame["stake_fraction"] = 0.0
        return frame

    frame["expected_value"] = pd.to_numeric(frame[probability_col], errors="coerce") * pd.to_numeric(
        frame[odds_col],
        errors="coerce",
    )
    frame["market_rank"] = frame.groupby("race_id")[odds_col].rank(ascending=True, method="min")
    frame["is_value_bet"] = frame["expected_value"] >= float(expected_value_threshold)
    probability = pd.to_numeric(frame[probability_col], errors="coerce")
    odds = pd.to_numeric(frame[odds_col], errors="coerce")
    b = odds - 1.0
    raw = ((b * probability) - (1.0 - probability)) / b
    frame["stake_fraction"] = raw.where((odds > 1.0) & probability.notna() & odds.notna(), 0.0).clip(
        lower=0.0,
        upper=float(kelly_cap),
    )
    return frame


def _kelly_fraction(probability: float, odds: float, cap: float) -> float:
    if odds <= 1.0:
        return 0.0
    b = odds - 1.0
    q = 1.0 - probability
    raw = ((b * probability) - q) / b
    return float(min(max(raw, 0.0), cap))
