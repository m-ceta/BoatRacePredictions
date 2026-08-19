from __future__ import annotations

import numpy as np
import pandas as pd


TOP12_CONFIDENCE_HIGH_THRESHOLD = 75.0
TOP12_CONFIDENCE_MIDDLE_THRESHOLD = 60.0
TOP3_CONFIDENCE_HIGH_THRESHOLD = 75.0
TOP3_CONFIDENCE_MIDDLE_THRESHOLD = 60.0
BOAT_TOP1_CONFIDENCE_HIGH_THRESHOLD = 75.0
BOAT_TOP1_CONFIDENCE_MIDDLE_THRESHOLD = 60.0

_LABEL_HIGH = "\u9ad8"
_LABEL_MIDDLE = "\u4e2d"
_LABEL_LOW = "\u4f4e"

PROBABILITY_ADJUSTMENT_VERSION = 1
PROBABILITY_ADJUSTMENT_DEFAULT_MIN_SAMPLES = 300
PROBABILITY_ADJUSTMENT_DEFAULT_FACTOR_MIN = 0.25
PROBABILITY_ADJUSTMENT_DEFAULT_FACTOR_MAX = 2.0


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


def attach_top3_confidence_columns(
    trifecta_df: pd.DataFrame,
    probability_col: str = "probability",
    *,
    update_recommendation: bool = True,
) -> pd.DataFrame:
    frame = trifecta_df.copy()
    if frame.empty or "race_id" not in frame.columns or probability_col not in frame.columns:
        return frame

    score_rows: list[pd.DataFrame] = []
    for _, race_df in frame.groupby("race_id", sort=False):
        scored = race_df.copy()
        score_data = calculate_top3_confidence_for_race(scored, probability_col=probability_col)
        for key, value in score_data.items():
            scored[key] = value
        if update_recommendation:
            scored["recommended_ticket_count"] = scored["top3_recommended_ticket_count"]
            scored["recommended_ticket_label"] = scored["top3_recommended_ticket_label"]
        score_rows.append(scored)
    return pd.concat(score_rows, ignore_index=True) if score_rows else frame


def attach_boat_top1_confidence_columns(
    trifecta_df: pd.DataFrame,
    probability_col: str = "probability",
) -> pd.DataFrame:
    frame = trifecta_df.copy()
    if frame.empty or "race_id" not in frame.columns or "trifecta" not in frame.columns or probability_col not in frame.columns:
        return frame

    score_rows: list[pd.DataFrame] = []
    for _, race_df in frame.groupby("race_id", sort=False):
        scored = race_df.copy()
        score_data = calculate_boat_top1_confidence_for_race(scored, probability_col=probability_col)
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


def calculate_top3_confidence_for_race(
    race_df: pd.DataFrame,
    probability_col: str = "probability",
) -> dict[str, float | str]:
    if race_df.empty or probability_col not in race_df.columns:
        return _empty_top3_confidence()

    ordered = race_df.sort_values(probability_col, ascending=False).reset_index(drop=True)
    probs = pd.to_numeric(ordered[probability_col], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
    total = float(probs.sum())
    if total > 0.0:
        probs = probs / total
    elif len(probs):
        probs = np.full(len(probs), 1.0 / len(probs), dtype=float)
    else:
        return _empty_top3_confidence()

    top3_mass = float(probs[:3].sum())
    top1_probability = float(probs[0])
    top1_top2_gap = float(probs[0] - probs[1]) if len(probs) >= 2 else float(probs[0])
    top3_margin = float(probs[2] - probs[3]) if len(probs) >= 4 else float(probs[min(len(probs) - 1, 2)])
    concentration = 1.0 - _normalized_entropy(probs)
    race_upset_score = _race_value(ordered, "race_upset_score", 0.0)

    top3_mass_score = _clip01((top3_mass - 0.03) / 0.16)
    top1_score = _clip01((top1_probability - 0.01) / 0.08)
    top1_gap_score = _clip01(top1_top2_gap / 0.035)
    top3_margin_score = _clip01(top3_margin / 0.012)
    concentration_score = _clip01(concentration / 0.25)
    upset_penalty = _clip01((race_upset_score - 0.75) / 0.25)

    raw_score = (
        0.35 * top3_mass_score
        + 0.20 * top1_score
        + 0.15 * top1_gap_score
        + 0.15 * top3_margin_score
        + 0.15 * concentration_score
        - 0.08 * upset_penalty
    )
    score = round(_clip01(raw_score) * 100.0, 1)
    recommended_ticket_count = recommended_ticket_count_from_top3_confidence(score)
    return {
        "top3_confidence_score": score,
        "top3_confidence_label": label_top3_confidence(score),
        "top3_recommended_ticket_count": recommended_ticket_count,
        "top3_recommended_ticket_label": recommended_ticket_label_from_count(recommended_ticket_count),
        "top3_probability_mass": round(top3_mass, 6),
        "top1_probability": round(top1_probability, 6),
        "top3_top4_probability_margin": round(top3_margin, 6),
    }


def calculate_boat_top1_confidence_for_race(
    race_df: pd.DataFrame,
    probability_col: str = "probability",
) -> dict[str, float | int | str]:
    if race_df.empty or "trifecta" not in race_df.columns or probability_col not in race_df.columns:
        return _empty_boat_top1_confidence()

    frame = race_df.copy()
    probs = pd.to_numeric(frame[probability_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    total = float(probs.sum())
    if total > 0.0:
        probs = probs / total
    elif len(probs):
        probs = pd.Series(np.full(len(probs), 1.0 / len(probs), dtype=float), index=frame.index)
    else:
        return _empty_boat_top1_confidence()

    first_boats = frame["trifecta"].map(_trifecta_first_boat)
    boat_probs = probs.groupby(first_boats).sum()
    boat_probs = boat_probs[boat_probs.index.notna()]
    if boat_probs.empty:
        return _empty_boat_top1_confidence()

    boat_probs = boat_probs.sort_values(ascending=False)
    predicted_boat = int(boat_probs.index[0])
    predicted_probability = float(boat_probs.iloc[0])
    second_probability = float(boat_probs.iloc[1]) if len(boat_probs) >= 2 else 0.0
    gap = predicted_probability - second_probability

    ordered = frame.assign(_probability=probs, _first_boat=first_boats).sort_values("_probability", ascending=False)
    top3_first_boats = ordered["_first_boat"].head(3)
    top3_same_first_boat_rate = float((top3_first_boats == predicted_boat).mean()) if len(top3_first_boats) else 0.0
    race_upset_score = _race_value(frame, "race_upset_score", 0.0)

    full_boat_probs = np.zeros(6, dtype=float)
    for boat, probability in boat_probs.items():
        boat_index = int(boat) - 1
        if 0 <= boat_index < len(full_boat_probs):
            full_boat_probs[boat_index] = float(probability)
    concentration = 1.0 - _normalized_entropy(full_boat_probs)

    probability_score = _clip01((predicted_probability - (1.0 / 6.0)) / 0.45)
    gap_score = _clip01(gap / 0.25)
    top3_same_first_boat_score = _clip01(top3_same_first_boat_rate)
    concentration_score = _clip01(concentration / 0.45)
    upset_penalty = _clip01((race_upset_score - 0.75) / 0.25)

    raw_score = (
        0.45 * probability_score
        + 0.25 * gap_score
        + 0.15 * top3_same_first_boat_score
        + 0.15 * concentration_score
        - 0.08 * upset_penalty
    )
    score = round(_clip01(raw_score) * 100.0, 1)
    return {
        "boat_top1_confidence_score": score,
        "boat_top1_confidence_label": label_boat_top1_confidence(score),
        "predicted_first_boat": predicted_boat,
        "predicted_first_boat_probability": round(predicted_probability, 6),
        "predicted_first_boat_gap": round(gap, 6),
        "predicted_first_boat_top3_share": round(top3_same_first_boat_rate, 6),
        "boat_top1_probability_concentration": round(concentration, 6),
    }


def label_top12_confidence(score: float) -> str:
    value = float(score)
    if value >= TOP12_CONFIDENCE_HIGH_THRESHOLD:
        return _LABEL_HIGH
    if value >= TOP12_CONFIDENCE_MIDDLE_THRESHOLD:
        return _LABEL_MIDDLE
    return _LABEL_LOW


def label_top3_confidence(score: float) -> str:
    value = float(score)
    if value >= TOP3_CONFIDENCE_HIGH_THRESHOLD:
        return _LABEL_HIGH
    if value >= TOP3_CONFIDENCE_MIDDLE_THRESHOLD:
        return _LABEL_MIDDLE
    return _LABEL_LOW


def label_boat_top1_confidence(score: float) -> str:
    value = float(score)
    if value >= BOAT_TOP1_CONFIDENCE_HIGH_THRESHOLD:
        return _LABEL_HIGH
    if value >= BOAT_TOP1_CONFIDENCE_MIDDLE_THRESHOLD:
        return _LABEL_MIDDLE
    return _LABEL_LOW


def top12_confidence_label_key(value: object) -> str:
    text = str(value)
    if text == _LABEL_HIGH:
        return "high"
    if text == _LABEL_MIDDLE:
        return "middle"
    return "low"


def top3_confidence_label_key(value: object) -> str:
    text = str(value)
    if text == _LABEL_HIGH:
        return "high"
    if text == _LABEL_MIDDLE:
        return "middle"
    return "low"


def boat_top1_confidence_label_key(value: object) -> str:
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


def recommended_ticket_count_from_top3_confidence(value: object) -> int:
    if isinstance(value, str):
        label = top3_confidence_label_key(value)
    else:
        score = float(value)
        if score >= TOP3_CONFIDENCE_HIGH_THRESHOLD:
            label = "high"
        elif score >= TOP3_CONFIDENCE_MIDDLE_THRESHOLD:
            label = "middle"
        else:
            label = "low"
    if label == "high":
        return 3
    if label == "middle":
        return 3
    return 0


def recommended_ticket_label_from_count(ticket_count: int) -> str:
    count = int(ticket_count)
    if count > 0:
        return f"Top{count}"
    return "\u898b\u9001\u308a"


def _trifecta_first_boat(value: object) -> int | None:
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text.split("-", 1)[0])
    except (TypeError, ValueError):
        return None


def fit_top12_probability_adjustment_table(
    trifecta_df: pd.DataFrame,
    probability_col: str = "probability",
    actual_col: str = "is_actual",
    min_samples: int = PROBABILITY_ADJUSTMENT_DEFAULT_MIN_SAMPLES,
    factor_min: float = PROBABILITY_ADJUSTMENT_DEFAULT_FACTOR_MIN,
    factor_max: float = PROBABILITY_ADJUSTMENT_DEFAULT_FACTOR_MAX,
) -> dict[str, object]:
    """Fit a conservative table that maps displayed probabilities closer to observed hit rates."""
    if trifecta_df.empty or probability_col not in trifecta_df.columns or actual_col not in trifecta_df.columns:
        return _empty_probability_adjustment_table(min_samples, factor_min, factor_max)

    frame = attach_top12_confidence_columns(trifecta_df, probability_col=probability_col)
    frame = frame.copy()
    frame["_probability"] = pd.to_numeric(frame[probability_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    frame["_is_actual"] = pd.to_numeric(frame[actual_col], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    frame["_prediction_rank"] = frame.groupby("race_id")["_probability"].rank(ascending=False, method="first")
    frame["_rank_band"] = frame["_prediction_rank"].map(_rank_band_from_rank)
    frame["_confidence_key"] = frame["top12_confidence_label"].map(top12_confidence_label_key)

    rules: list[dict[str, object]] = []
    for (confidence_key, rank_band), bucket in frame.groupby(["_confidence_key", "_rank_band"], sort=True):
        sample_count = int(len(bucket))
        mean_probability = float(bucket["_probability"].mean()) if sample_count else 0.0
        observed_hit_rate = float(bucket["_is_actual"].mean()) if sample_count else 0.0
        raw_factor = observed_hit_rate / mean_probability if mean_probability > 0.0 else 1.0
        shrink = sample_count / (sample_count + max(int(min_samples), 1))
        factor = 1.0 + (raw_factor - 1.0) * shrink
        factor = float(np.clip(factor, float(factor_min), float(factor_max)))
        rules.append(
            {
                "confidence_key": str(confidence_key),
                "rank_band": str(rank_band),
                "sample_count": float(sample_count),
                "race_count": float(bucket["race_id"].nunique()) if "race_id" in bucket.columns else 0.0,
                "mean_predicted_probability": mean_probability,
                "observed_hit_rate": observed_hit_rate,
                "raw_factor": float(raw_factor),
                "factor": factor,
            }
        )

    return {
        "version": PROBABILITY_ADJUSTMENT_VERSION,
        "probability_col": probability_col,
        "actual_col": actual_col,
        "min_samples": int(min_samples),
        "factor_min": float(factor_min),
        "factor_max": float(factor_max),
        "default_factor": 1.0,
        "rules": rules,
    }


def apply_top12_probability_adjustment_table(
    trifecta_df: pd.DataFrame,
    adjustment_table: dict[str, object] | None,
    probability_col: str = "probability",
    output_col: str = "adjusted_probability",
) -> pd.DataFrame:
    frame = trifecta_df.copy()
    if frame.empty or probability_col not in frame.columns:
        return frame

    probability = pd.to_numeric(frame[probability_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    if not adjustment_table:
        frame[output_col] = probability
        frame["probability_adjustment_factor"] = 1.0
        frame["probability_adjustment_rank_band"] = pd.NA
        return frame

    if "top12_confidence_label" not in frame.columns:
        frame = attach_top12_confidence_columns(frame, probability_col=probability_col)

    rank = frame.groupby("race_id")[probability_col].rank(ascending=False, method="first") if "race_id" in frame.columns else probability.rank(ascending=False, method="first")
    rank_band = rank.map(_rank_band_from_rank)
    confidence_key = frame.get("top12_confidence_label", pd.Series(_LABEL_LOW, index=frame.index)).map(
        top12_confidence_label_key
    )

    factor_lookup = {
        (str(rule.get("confidence_key")), str(rule.get("rank_band"))): float(rule.get("factor", 1.0))
        for rule in adjustment_table.get("rules", [])  # type: ignore[union-attr]
        if isinstance(rule, dict)
    }
    default_factor = float(adjustment_table.get("default_factor", 1.0))
    factors = [
        factor_lookup.get((str(confidence), str(band)), default_factor)
        for confidence, band in zip(confidence_key, rank_band, strict=False)
    ]
    factor_series = pd.Series(factors, index=frame.index, dtype=float)
    frame[output_col] = (probability * factor_series).clip(lower=0.0, upper=1.0)
    frame["probability_adjustment_factor"] = factor_series
    frame["probability_adjustment_rank_band"] = rank_band
    return frame


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


def _rank_band_from_rank(rank: object) -> str:
    try:
        value = int(float(rank))
    except (TypeError, ValueError):
        return "unknown"
    if value <= 1:
        return "top1"
    if value <= 3:
        return "top2_3"
    if value <= 6:
        return "top4_6"
    if value <= 12:
        return "top7_12"
    return "outside_top12"


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


def _empty_top3_confidence() -> dict[str, float | str]:
    return {
        "top3_confidence_score": 0.0,
        "top3_confidence_label": _LABEL_LOW,
        "top3_recommended_ticket_count": 0,
        "top3_recommended_ticket_label": "\u898b\u9001\u308a",
        "top3_probability_mass": 0.0,
        "top1_probability": 0.0,
        "top3_top4_probability_margin": 0.0,
    }


def _empty_boat_top1_confidence() -> dict[str, float | int | str]:
    return {
        "boat_top1_confidence_score": 0.0,
        "boat_top1_confidence_label": _LABEL_LOW,
        "predicted_first_boat": 0,
        "predicted_first_boat_probability": 0.0,
        "predicted_first_boat_gap": 0.0,
        "predicted_first_boat_top3_share": 0.0,
        "boat_top1_probability_concentration": 0.0,
    }


def _empty_probability_adjustment_table(
    min_samples: int,
    factor_min: float,
    factor_max: float,
) -> dict[str, object]:
    return {
        "version": PROBABILITY_ADJUSTMENT_VERSION,
        "probability_col": "probability",
        "actual_col": "is_actual",
        "min_samples": int(min_samples),
        "factor_min": float(factor_min),
        "factor_max": float(factor_max),
        "default_factor": 1.0,
        "rules": [],
    }
