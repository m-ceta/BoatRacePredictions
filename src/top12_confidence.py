from __future__ import annotations

import numpy as np
import pandas as pd


TOP12_CONFIDENCE_HIGH_THRESHOLD = 75.0
TOP12_CONFIDENCE_MIDDLE_THRESHOLD = 60.0

_LABEL_HIGH = "\u9ad8"
_LABEL_MIDDLE = "\u4e2d"
_LABEL_LOW = "\u4f4e"


def attach_top12_confidence_columns(
    trifecta_df: pd.DataFrame,
    probability_col: str = "probability",
) -> pd.DataFrame:
    frame = trifecta_df.copy()
    if frame.empty or "race_id" not in frame.columns or probability_col not in frame.columns:
        return frame

    score_rows: list[pd.DataFrame] = []
    for _, race_df in frame.groupby("race_id", sort=False):
        scored = race_df.copy()
        score_data = calculate_top12_confidence_for_race(scored, probability_col=probability_col)
        for key, value in score_data.items():
            scored[key] = value
        score_rows.append(scored)
    return pd.concat(score_rows, ignore_index=True) if score_rows else frame


def calculate_top12_confidence_for_race(
    race_df: pd.DataFrame,
    probability_col: str = "probability",
) -> dict[str, float | str]:
    if race_df.empty or probability_col not in race_df.columns:
        return _empty_top12_confidence()

    ordered = race_df.sort_values(probability_col, ascending=False).reset_index(drop=True)
    probs = pd.to_numeric(ordered[probability_col], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
    total = float(probs.sum())
    if total > 0.0:
        probs = probs / total
    elif len(probs):
        probs = np.full(len(probs), 1.0 / len(probs), dtype=float)
    else:
        return _empty_top12_confidence()

    top12_mass = float(probs[:12].sum())
    top5_mass = float(probs[:5].sum())
    top1_top2_gap = float(probs[0] - probs[1]) if len(probs) >= 2 else float(probs[0])
    top12_margin = float(probs[11] - probs[12]) if len(probs) >= 13 else float(probs[min(len(probs) - 1, 11)])
    concentration = 1.0 - _normalized_entropy(probs)
    race_upset_score = _race_value(ordered, "race_upset_score", 0.0)

    top12_mass_score = _clip01((top12_mass - 0.10) / 0.45)
    top5_mass_score = _clip01((top5_mass - 0.04) / 0.25)
    gap_score = _clip01(top1_top2_gap / 0.06)
    margin_score = _clip01(top12_margin / 0.01)
    concentration_score = _clip01(concentration / 0.25)
    upset_penalty = _clip01((race_upset_score - 0.70) / 0.30)

    raw_score = (
        0.45 * top12_mass_score
        + 0.20 * top5_mass_score
        + 0.15 * gap_score
        + 0.10 * margin_score
        + 0.10 * concentration_score
        - 0.10 * upset_penalty
    )
    score = round(_clip01(raw_score) * 100.0, 1)
    recommended_ticket_count = recommended_ticket_count_from_top12_confidence(score)
    return {
        "top12_confidence_score": score,
        "top12_confidence_label": label_top12_confidence(score),
        "recommended_ticket_count": recommended_ticket_count,
        "recommended_ticket_label": recommended_ticket_label_from_count(recommended_ticket_count),
        "top12_probability_mass": round(top12_mass, 6),
        "top5_probability_mass": round(top5_mass, 6),
        "top1_top2_probability_gap": round(top1_top2_gap, 6),
        "top12_probability_margin": round(top12_margin, 6),
        "probability_concentration": round(concentration, 6),
    }


def label_top12_confidence(score: float) -> str:
    value = float(score)
    if value >= TOP12_CONFIDENCE_HIGH_THRESHOLD:
        return _LABEL_HIGH
    if value >= TOP12_CONFIDENCE_MIDDLE_THRESHOLD:
        return _LABEL_MIDDLE
    return _LABEL_LOW


def top12_confidence_label_key(value: object) -> str:
    text = str(value)
    if text == _LABEL_HIGH:
        return "high"
    if text == _LABEL_MIDDLE:
        return "middle"
    return "low"


def recommended_ticket_count_from_top12_confidence(value: object) -> int:
    if isinstance(value, str):
        label = top12_confidence_label_key(value)
    else:
        score = float(value)
        if score >= TOP12_CONFIDENCE_HIGH_THRESHOLD:
            label = "high"
        elif score >= TOP12_CONFIDENCE_MIDDLE_THRESHOLD:
            label = "middle"
        else:
            label = "low"
    if label == "high":
        return 5
    if label == "middle":
        return 8
    return 0


def recommended_ticket_label_from_count(ticket_count: int) -> str:
    count = int(ticket_count)
    if count > 0:
        return f"Top{count}"
    return "\u898b\u9001\u308a"


def _normalized_entropy(probs: np.ndarray) -> float:
    if len(probs) <= 1:
        return 0.0
    clipped = np.clip(probs, 1e-12, None)
    entropy = -float(np.sum(clipped * np.log(clipped)))
    return _clip01(entropy / float(np.log(len(probs))))


def _race_value(frame: pd.DataFrame, column: str, default: float) -> float:
    if column not in frame.columns:
        return float(default)
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.iloc[0]) if not values.empty else float(default)


def _clip01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _empty_top12_confidence() -> dict[str, float | str]:
    return {
        "top12_confidence_score": 0.0,
        "top12_confidence_label": _LABEL_LOW,
        "recommended_ticket_count": 0,
        "recommended_ticket_label": "\u898b\u9001\u308a",
        "top12_probability_mass": 0.0,
        "top5_probability_mass": 0.0,
        "top1_top2_probability_gap": 0.0,
        "top12_probability_margin": 0.0,
        "probability_concentration": 0.0,
    }
