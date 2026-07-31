from __future__ import annotations

import numpy as np
import pandas as pd


BUY_EXPECTED_VALUE_THRESHOLD = 1.0
BUY_MIN_ODDS = 12.0
BUY_SCORE_STRONG_THRESHOLD = 80.0
BUY_SCORE_BUY_THRESHOLD = 65.0
BUY_SCORE_KEEP_THRESHOLD = 50.0


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


def attach_buy_score_columns(
    odds_frame: pd.DataFrame,
    min_odds: float = BUY_MIN_ODDS,
    expected_value_threshold: float = BUY_EXPECTED_VALUE_THRESHOLD,
    confidence_score: float | None = None,
) -> pd.DataFrame:
    frame = odds_frame.copy()
    if frame.empty:
        return frame

    def numeric_series(column: str, default: float = 0.0) -> pd.Series:
        if column not in frame.columns:
            return pd.Series(default, index=frame.index, dtype=float)
        return pd.to_numeric(frame[column], errors="coerce").fillna(default)

    probability = numeric_series("probability")
    odds = numeric_series("odds")
    expected_value = numeric_series("expected_value")
    priority = numeric_series("ticket_priority_score")
    race_upset = numeric_series("race_upset_score")

    fair_odds = pd.Series(pd.NA, index=frame.index, dtype="Float64")
    positive_probability = probability > 0.0
    fair_odds.loc[positive_probability] = 1.0 / probability.loc[positive_probability]
    frame["fair_odds"] = fair_odds
    frame["odds_value_ratio"] = expected_value

    expected_value_score = ((expected_value - 0.8) / 0.7).clip(0.0, 1.0)
    odds_score = ((odds - float(min_odds)) / max(40.0 - float(min_odds), 1.0)).clip(0.0, 1.0)
    probability_score = (probability / 0.08).clip(0.0, 1.0)
    confidence = 0.5 if confidence_score is None else min(max(float(confidence_score), 0.0), 1.0)
    confidence_series = pd.Series(confidence, index=frame.index)
    upset_penalty = ((race_upset - 0.70) / 0.30).clip(0.0, 1.0)

    raw_score = (
        0.45 * expected_value_score
        + 0.20 * odds_score
        + 0.15 * probability_score
        + 0.10 * priority.clip(0.0, 1.0)
        + 0.10 * confidence_series
        - 0.10 * upset_penalty
    )
    eligible = (expected_value >= float(expected_value_threshold)) & (odds >= float(min_odds))
    raw_score = raw_score.mask(eligible, raw_score.clip(lower=0.50))
    raw_score = raw_score.mask(~eligible, raw_score.clip(upper=0.49))
    frame["buy_score"] = (raw_score.clip(0.0, 1.0) * 100.0).round(1)
    frame["buy_score_label"] = frame["buy_score"].map(label_buy_score)
    return frame


def label_buy_score(score: float) -> str:
    value = float(score)
    if value >= BUY_SCORE_STRONG_THRESHOLD:
        return "強く買い候補"
    if value >= BUY_SCORE_BUY_THRESHOLD:
        return "買い候補"
    if value >= BUY_SCORE_KEEP_THRESHOLD:
        return "抑え候補"
    return "見送り"


def _kelly_fraction(probability: float, odds: float, cap: float) -> float:
    if odds <= 1.0:
        return 0.0
    b = odds - 1.0
    q = 1.0 - probability
    raw = ((b * probability) - q) / b
    return float(min(max(raw, 0.0), cap))
