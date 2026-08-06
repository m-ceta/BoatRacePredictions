from __future__ import annotations

import pandas as pd


BUY_EXPECTED_VALUE_THRESHOLD = 1.0
BUY_MIN_ODDS = 12.0
RECOMMENDED_BET_TOP_N = 12
RECOMMENDED_BET_BANKROLL_YEN = 10000
RECOMMENDED_BET_UNIT_YEN = 100
RECOMMENDED_BET_MAX_PER_TICKET_YEN = 500
RECOMMENDED_BET_KELLY_CAP = 0.02


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


def attach_recommended_bet_amount_columns(
    trifecta_df: pd.DataFrame,
    probability_col: str = "probability",
    odds_col: str = "odds",
    rank_col: str = "prediction_rank",
    expected_value_col: str = "expected_value",
    expected_value_threshold: float = BUY_EXPECTED_VALUE_THRESHOLD,
    min_odds: float = BUY_MIN_ODDS,
    top_n: int = RECOMMENDED_BET_TOP_N,
    bankroll_yen: int = RECOMMENDED_BET_BANKROLL_YEN,
    unit_yen: int = RECOMMENDED_BET_UNIT_YEN,
    max_per_ticket_yen: int = RECOMMENDED_BET_MAX_PER_TICKET_YEN,
    kelly_cap: float = RECOMMENDED_BET_KELLY_CAP,
) -> pd.DataFrame:
    frame = trifecta_df.copy()
    if frame.empty or odds_col not in frame.columns:
        if "recommended_bet_amount" not in frame.columns:
            frame["recommended_bet_amount"] = 0
        if "recommended_bet_units" not in frame.columns:
            frame["recommended_bet_units"] = 0
        return frame

    probability = pd.to_numeric(frame[probability_col], errors="coerce")
    odds = pd.to_numeric(frame[odds_col], errors="coerce")
    if expected_value_col in frame.columns:
        expected_value = pd.to_numeric(frame[expected_value_col], errors="coerce")
    else:
        expected_value = pd.Series(pd.NA, index=frame.index, dtype="Float64")
    if expected_value.isna().all():
        expected_value = probability * odds
        frame[expected_value_col] = expected_value

    if rank_col in frame.columns:
        rank = pd.to_numeric(frame[rank_col], errors="coerce")
    else:
        rank = pd.Series(1.0, index=frame.index)

    b = odds - 1.0
    raw_kelly = ((b * probability) - (1.0 - probability)) / b
    stake_fraction = raw_kelly.where((odds > 1.0) & probability.notna() & odds.notna(), 0.0).clip(
        lower=0.0,
        upper=float(kelly_cap),
    )
    eligible = (
        (rank <= int(top_n))
        & (expected_value >= float(expected_value_threshold))
        & (odds >= float(min_odds))
        & probability.notna()
        & odds.notna()
    )
    raw_amount = (stake_fraction * int(bankroll_yen)).where(eligible, 0.0).clip(
        lower=0.0,
        upper=float(max_per_ticket_yen),
    )
    rounded_amount = (raw_amount // int(unit_yen)) * int(unit_yen)
    rounded_amount = rounded_amount.where(~eligible | (rounded_amount >= int(unit_yen)), int(unit_yen))

    frame["recommended_bet_amount"] = rounded_amount.fillna(0).astype(int)
    frame["recommended_bet_units"] = (frame["recommended_bet_amount"] // int(unit_yen)).astype(int)
    frame["stake_fraction"] = stake_fraction.fillna(0.0)
    return frame


def _kelly_fraction(probability: float, odds: float, cap: float) -> float:
    if odds <= 1.0:
        return 0.0
    b = odds - 1.0
    q = 1.0 - probability
    raw = ((b * probability) - q) / b
    return float(min(max(raw, 0.0), cap))
