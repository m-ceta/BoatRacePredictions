from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.top12_confidence import attach_boat_top1_confidence_columns, attach_top3_confidence_columns


TOP3_HIT_PROBABILITY_FEATURES: tuple[str, ...] = (
    "top3_confidence_score",
    "top3_probability_mass",
    "top1_probability",
    "top1_top2_probability_gap",
    "top3_top4_probability_margin",
    "top3_structure_tightness_score",
    "top3_same_first_boat_rate",
    "top3_same_first_second_pair_rate",
    "probability_entropy",
)

TOP3_HIT_PROBABILITY_BANDS: tuple[tuple[str, float, float | None], ...] = (
    ("p_ge_0_55", 0.55, None),
    ("p_0_50_0_55", 0.50, 0.55),
    ("p_0_45_0_50", 0.45, 0.50),
    ("p_0_40_0_45", 0.40, 0.45),
    ("p_0_35_0_40", 0.35, 0.40),
    ("p_0_30_0_35", 0.30, 0.35),
    ("p_lt_0_30", float("-inf"), 0.30),
)


@dataclass(slots=True)
class Top3HitProbabilityPayload:
    model: Any
    feature_names: list[str]
    training_metrics: dict[str, Any]
    version: int = 1


def fit_top3_hit_probability_model(
    trifecta_df: pd.DataFrame,
    probability_col: str = "probability",
) -> Top3HitProbabilityPayload:
    feature_frame, target = build_top3_hit_probability_training_frame(
        trifecta_df,
        probability_col=probability_col,
    )
    if feature_frame.empty or len(target) == 0:
        model = DummyClassifier(strategy="constant", constant=0)
        model.fit(np.zeros((1, len(TOP3_HIT_PROBABILITY_FEATURES))), np.asarray([0], dtype=int))
        return Top3HitProbabilityPayload(
            model=model,
            feature_names=list(TOP3_HIT_PROBABILITY_FEATURES),
            training_metrics={"status": "empty_training_frame", "race_count": 0.0},
        )

    x = feature_frame.loc[:, TOP3_HIT_PROBABILITY_FEATURES].to_numpy(dtype=float)
    y = target.astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        model = DummyClassifier(strategy="constant", constant=int(y[0]))
    else:
        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "logreg",
                    LogisticRegression(
                        max_iter=1000,
                        random_state=42,
                    ),
                ),
            ]
        )
    model.fit(x, y)
    probability = _predict_positive_probability(model, x)
    scored = feature_frame.copy()
    scored["top3_hit_probability"] = probability
    scored["top3_hit"] = y
    return Top3HitProbabilityPayload(
        model=model,
        feature_names=list(TOP3_HIT_PROBABILITY_FEATURES),
        training_metrics=summarize_top3_hit_probability_bands(scored),
    )


def build_top3_hit_probability_training_frame(
    trifecta_df: pd.DataFrame,
    probability_col: str = "probability",
) -> tuple[pd.DataFrame, pd.Series]:
    frame = _prepare_top3_hit_probability_frame(trifecta_df, probability_col=probability_col)
    records: list[dict[str, Any]] = []
    targets: list[int] = []
    for race_id, race_df in frame.groupby("race_id", sort=False):
        ordered = race_df.sort_values(probability_col, ascending=False).reset_index(drop=True)
        actual_positions = np.flatnonzero(ordered.get("is_actual", pd.Series(False, index=ordered.index)).to_numpy(dtype=bool))
        if len(actual_positions) != 1:
            continue
        top_row = ordered.iloc[0]
        record = {"race_id": str(race_id)}
        for column in TOP3_HIT_PROBABILITY_FEATURES:
            record[column] = float(top_row.get(column, 0.0) or 0.0)
        records.append(record)
        targets.append(int(int(actual_positions[0]) < 3))
    return pd.DataFrame(records), pd.Series(targets, dtype=int)


def attach_top3_hit_probability_columns(
    trifecta_df: pd.DataFrame,
    payload: Top3HitProbabilityPayload | dict[str, Any] | None,
    probability_col: str = "probability",
) -> pd.DataFrame:
    frame = _prepare_top3_hit_probability_frame(trifecta_df, probability_col=probability_col)
    if frame.empty:
        return _ensure_top3_hit_probability_columns(frame)
    if payload is None:
        return _attach_fallback_top3_hit_probability(frame)

    model = payload.model if isinstance(payload, Top3HitProbabilityPayload) else payload.get("model")
    feature_names = (
        payload.feature_names if isinstance(payload, Top3HitProbabilityPayload) else payload.get("feature_names")
    ) or list(TOP3_HIT_PROBABILITY_FEATURES)
    if model is None:
        return _attach_fallback_top3_hit_probability(frame)

    updates: list[pd.Series] = []
    for race_id, race_df in frame.groupby("race_id", sort=False):
        top_index = race_df.sort_values(probability_col, ascending=False).index[0]
        feature_values = {}
        for name in feature_names:
            raw_value = frame.at[top_index, name] if name in frame.columns else 0.0
            numeric_value = pd.to_numeric(pd.Series([raw_value]), errors="coerce").iloc[0]
            feature_values[name] = 0.0 if pd.isna(numeric_value) else float(numeric_value)
        features = pd.DataFrame([feature_values])
        probability = float(_predict_positive_probability(model, features.to_numpy(dtype=float))[0])
        updates.append(
            pd.Series(
                {
                    "race_id": race_id,
                    "top3_hit_probability": probability,
                    "top3_hit_probability_score": probability * 100.0,
                    "top3_hit_probability_label": label_top3_hit_probability(probability),
                }
            )
        )
    if not updates:
        return _ensure_top3_hit_probability_columns(frame)
    race_scores = pd.DataFrame(updates)
    frame = frame.merge(race_scores, on="race_id", how="left")
    return _ensure_top3_hit_probability_columns(frame)


def summarize_top3_hit_probability_bands(
    race_frame: pd.DataFrame,
    target_col: str = "top3_hit",
) -> dict[str, Any]:
    if race_frame.empty or "top3_hit_probability" not in race_frame.columns or target_col not in race_frame.columns:
        return {}
    probabilities = pd.to_numeric(race_frame["top3_hit_probability"], errors="coerce")
    targets = pd.to_numeric(race_frame[target_col], errors="coerce")
    valid = probabilities.notna() & targets.notna()
    if not bool(valid.any()):
        return {}
    frame = race_frame.loc[valid].copy()
    frame["top3_hit_probability"] = probabilities.loc[valid].astype(float)
    frame[target_col] = targets.loc[valid].astype(float)
    total = len(frame)
    metrics: dict[str, Any] = {
        "race_count": float(total),
        "mean_probability": float(frame["top3_hit_probability"].mean()),
        "actual_top3_hit_rate": float(frame[target_col].mean()),
        "bands": {},
    }
    for key, lower, upper in TOP3_HIT_PROBABILITY_BANDS:
        mask = frame["top3_hit_probability"] >= lower
        if upper is not None:
            mask &= frame["top3_hit_probability"] < upper
        subset = frame.loc[mask]
        if subset.empty:
            continue
        metrics["bands"][key] = {
            "race_count": float(len(subset)),
            "race_rate": float(len(subset) / total) if total else 0.0,
            "mean_probability": float(subset["top3_hit_probability"].mean()),
            "top3_hit_rate": float(subset[target_col].mean()),
            "mean_top3_confidence_score": float(subset.get("top3_confidence_score", pd.Series(dtype=float)).mean()),
            "mean_top3_structure_tightness_score": float(
                subset.get("top3_structure_tightness_score", pd.Series(dtype=float)).mean()
            ),
        }
    return metrics


def label_top3_hit_probability(probability: float) -> str:
    value = float(probability)
    if value >= 0.50:
        return "high"
    if value >= 0.35:
        return "middle"
    return "low"


def _prepare_top3_hit_probability_frame(trifecta_df: pd.DataFrame, probability_col: str) -> pd.DataFrame:
    frame = attach_top3_confidence_columns(trifecta_df, probability_col=probability_col)
    frame = attach_boat_top1_confidence_columns(frame, probability_col=probability_col)
    return frame


def _attach_fallback_top3_hit_probability(frame: pd.DataFrame) -> pd.DataFrame:
    if "top3_confidence_score" not in frame.columns:
        return _ensure_top3_hit_probability_columns(frame)
    score = pd.to_numeric(frame["top3_confidence_score"], errors="coerce").fillna(0.0)
    probability = (score / 100.0).clip(0.0, 1.0)
    frame = frame.copy()
    frame["top3_hit_probability"] = probability
    frame["top3_hit_probability_score"] = probability * 100.0
    frame["top3_hit_probability_label"] = probability.map(label_top3_hit_probability)
    return frame


def _ensure_top3_hit_probability_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "top3_hit_probability" not in frame.columns:
        frame["top3_hit_probability"] = 0.0
    if "top3_hit_probability_score" not in frame.columns:
        frame["top3_hit_probability_score"] = pd.to_numeric(
            frame["top3_hit_probability"],
            errors="coerce",
        ).fillna(0.0) * 100.0
    if "top3_hit_probability_label" not in frame.columns:
        frame["top3_hit_probability_label"] = pd.to_numeric(
            frame["top3_hit_probability"],
            errors="coerce",
        ).fillna(0.0).map(label_top3_hit_probability)
    return frame


def _predict_positive_probability(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x)
        if probabilities.ndim == 2 and probabilities.shape[1] >= 2:
            return np.asarray(probabilities[:, 1], dtype=float)
        if probabilities.ndim == 2 and probabilities.shape[1] == 1:
            classes = getattr(model, "classes_", None)
            if classes is not None and len(classes) == 1:
                return np.full(len(probabilities), float(int(classes[0]) == 1), dtype=float)
            return np.asarray(probabilities[:, 0], dtype=float)
    prediction = model.predict(x)
    return np.clip(np.asarray(prediction, dtype=float), 0.0, 1.0)
