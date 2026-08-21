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


IN_COURSE_PROBABILITY_FEATURES: tuple[str, ...] = (
    "race_escape_reliability_score",
    "race_attack_pressure",
    "race_inner_collapse_risk",
    "race_outer_link_risk",
    "race_outer_inner_avg_st_gap",
    "pre_race_attack_candidate_course",
    "pre_race_attack_candidate_score",
    "in_national_win_rate",
    "in_national_place_rate",
    "in_local_win_rate",
    "in_local_place_rate",
    "in_racer_prev_win_rate",
    "in_racer_prev_top3_rate",
    "in_racer_prev_avg_st_5",
    "in_racer_prev_avg_st_10",
    "in_racer_prev_best_st_30",
    "in_exhibition_time",
    "in_motor_place_rate",
    "in_boat_place_rate",
    "in_venue_course_prev_win_rate",
    "in_venue_course_prev_top2_rate",
    "in_venue_course_prev_top3_rate",
    "in_venue_course_prev_nige_rate",
    "in_flow_prob_nige",
    "max_outer_attack_score",
    "max_outer_makuri_rate",
    "max_outer_makurizashi_rate",
    "max_outer_sashi_rate",
    "course2_sashi_rate",
    "course2_makuri_rate",
    "course3_attack_rate",
    "course4_attack_rate",
)

IN_COURSE_PROBABILITY_BANDS: tuple[tuple[str, float, float | None], ...] = (
    ("p_ge_0_70", 0.70, None),
    ("p_0_60_0_70", 0.60, 0.70),
    ("p_0_50_0_60", 0.50, 0.60),
    ("p_0_40_0_50", 0.40, 0.50),
    ("p_lt_0_40", float("-inf"), 0.40),
)


@dataclass(slots=True)
class InCourseProbabilityPayload:
    model: Any
    feature_names: list[str]
    target_name: str
    training_metrics: dict[str, Any]
    version: int = 1


def fit_in_course_probability_model(
    lane_df: pd.DataFrame,
    *,
    target_name: str,
) -> InCourseProbabilityPayload:
    feature_frame, target = build_in_course_probability_training_frame(lane_df, target_name=target_name)
    if feature_frame.empty or len(target) == 0:
        model = DummyClassifier(strategy="constant", constant=0)
        model.fit(np.zeros((1, len(IN_COURSE_PROBABILITY_FEATURES))), np.asarray([0], dtype=int))
        return InCourseProbabilityPayload(
            model=model,
            feature_names=list(IN_COURSE_PROBABILITY_FEATURES),
            target_name=target_name,
            training_metrics={"status": "empty_training_frame", "race_count": 0.0},
        )

    x = feature_frame.loc[:, IN_COURSE_PROBABILITY_FEATURES].to_numpy(dtype=float)
    y = target.astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        model = DummyClassifier(strategy="constant", constant=int(y[0]))
    else:
        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("logreg", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        )
    model.fit(x, y)
    probabilities = _predict_positive_probability(model, x)
    scored = feature_frame.copy()
    scored["probability"] = probabilities
    scored["target"] = y
    return InCourseProbabilityPayload(
        model=model,
        feature_names=list(IN_COURSE_PROBABILITY_FEATURES),
        target_name=target_name,
        training_metrics=summarize_in_course_probability_bands(scored),
    )


def build_in_course_probability_training_frame(
    lane_df: pd.DataFrame,
    *,
    target_name: str,
) -> tuple[pd.DataFrame, pd.Series]:
    race_frame = build_in_course_probability_feature_frame(lane_df)
    if race_frame.empty:
        return race_frame, pd.Series(dtype=int)
    target = _build_target(lane_df, target_name=target_name)
    merged = race_frame.merge(target, on="race_id", how="inner")
    if merged.empty:
        return pd.DataFrame(columns=["race_id", *IN_COURSE_PROBABILITY_FEATURES]), pd.Series(dtype=int)
    return merged.loc[:, ["race_id", *IN_COURSE_PROBABILITY_FEATURES]], merged["target"].astype(int)


def attach_in_course_probability_columns(
    trifecta_df: pd.DataFrame,
    lane_df: pd.DataFrame,
    *,
    in_win_payload: InCourseProbabilityPayload | dict[str, Any] | None,
    in_collapse_payload: InCourseProbabilityPayload | dict[str, Any] | None,
) -> pd.DataFrame:
    frame = trifecta_df.copy()
    if frame.empty:
        return _ensure_in_course_probability_columns(frame)

    feature_frame = build_in_course_probability_feature_frame(lane_df)
    race_scores = feature_frame.loc[:, ["race_id"]].copy() if "race_id" in feature_frame.columns else pd.DataFrame()
    if not race_scores.empty:
        race_scores["in_win_probability"] = _predict_payload_probability(in_win_payload, feature_frame)
        race_scores["in_collapse_probability"] = _predict_payload_probability(in_collapse_payload, feature_frame)
        race_scores["in_win_probability_score"] = race_scores["in_win_probability"] * 100.0
        race_scores["in_collapse_probability_score"] = race_scores["in_collapse_probability"] * 100.0
        race_scores["in_win_probability_label"] = race_scores["in_win_probability"].map(label_in_course_probability)
        race_scores["in_collapse_probability_label"] = race_scores["in_collapse_probability"].map(
            label_in_course_probability
        )
        frame = frame.merge(race_scores, on="race_id", how="left")
    return _ensure_in_course_probability_columns(frame)


def summarize_in_course_probability_bands(
    race_frame: pd.DataFrame,
    *,
    probability_col: str = "probability",
    target_col: str = "target",
) -> dict[str, Any]:
    if race_frame.empty or probability_col not in race_frame.columns or target_col not in race_frame.columns:
        return {}
    probabilities = pd.to_numeric(race_frame[probability_col], errors="coerce")
    targets = pd.to_numeric(race_frame[target_col], errors="coerce")
    valid = probabilities.notna() & targets.notna()
    if not bool(valid.any()):
        return {}
    frame = race_frame.loc[valid].copy()
    frame[probability_col] = probabilities.loc[valid].astype(float)
    frame[target_col] = targets.loc[valid].astype(float)
    total = len(frame)
    metrics: dict[str, Any] = {
        "race_count": float(total),
        "mean_probability": float(frame[probability_col].mean()),
        "actual_hit_rate": float(frame[target_col].mean()),
        "bands": {},
    }
    for key, lower, upper in IN_COURSE_PROBABILITY_BANDS:
        mask = frame[probability_col] >= lower
        if upper is not None:
            mask &= frame[probability_col] < upper
        subset = frame.loc[mask]
        if subset.empty:
            continue
        metrics["bands"][key] = {
            "race_count": float(len(subset)),
            "race_rate": float(len(subset) / total) if total else 0.0,
            "mean_probability": float(subset[probability_col].mean()),
            "actual_hit_rate": float(subset[target_col].mean()),
        }
    return metrics


def label_in_course_probability(probability: float) -> str:
    value = float(probability)
    if value >= 0.65:
        return "high"
    if value >= 0.45:
        return "middle"
    return "low"


def build_in_course_probability_feature_frame(lane_df: pd.DataFrame) -> pd.DataFrame:
    if lane_df.empty or "race_id" not in lane_df.columns:
        return pd.DataFrame(columns=["race_id", *IN_COURSE_PROBABILITY_FEATURES])
    rows: list[dict[str, float | str]] = []
    for race_id, race_df in lane_df.groupby("race_id", sort=False):
        if race_df.empty:
            continue
        rows.append(_race_feature_row(str(race_id), race_df))
    if not rows:
        return pd.DataFrame(columns=["race_id", *IN_COURSE_PROBABILITY_FEATURES])
    frame = pd.DataFrame(rows)
    for column in IN_COURSE_PROBABILITY_FEATURES:
        if column not in frame.columns:
            frame[column] = 0.0
    return frame.loc[:, ["race_id", *IN_COURSE_PROBABILITY_FEATURES]]


def _race_feature_row(race_id: str, race_df: pd.DataFrame) -> dict[str, float | str]:
    frame = race_df.copy()
    course = pd.to_numeric(frame.get("course"), errors="coerce")
    lane = pd.to_numeric(frame.get("lane"), errors="coerce")
    in_rows = frame[course == 1]
    if in_rows.empty:
        in_rows = frame[lane == 1]
    in_row = in_rows.iloc[0] if not in_rows.empty else frame.iloc[0]
    outer_rows = frame[course.fillna(lane) >= 2]
    center_rows = frame[course.fillna(lane).isin([3, 4])]

    row: dict[str, float | str] = {"race_id": race_id}
    for column in (
        "race_escape_reliability_score",
        "race_attack_pressure",
        "race_inner_collapse_risk",
        "race_outer_link_risk",
        "race_outer_inner_avg_st_gap",
        "pre_race_attack_candidate_course",
        "pre_race_attack_candidate_score",
    ):
        row[column] = _race_value(frame, column)

    in_columns = (
        "national_win_rate",
        "national_place_rate",
        "local_win_rate",
        "local_place_rate",
        "racer_prev_win_rate",
        "racer_prev_top3_rate",
        "racer_prev_avg_st_5",
        "racer_prev_avg_st_10",
        "racer_prev_best_st_30",
        "exhibition_time",
        "motor_place_rate",
        "boat_place_rate",
        "venue_course_prev_win_rate",
        "venue_course_prev_top2_rate",
        "venue_course_prev_top3_rate",
        "venue_course_prev_nige_rate",
        "flow_prob_nige",
    )
    for column in in_columns:
        row[f"in_{column}"] = _row_value(in_row, column)

    row["max_outer_attack_score"] = _max_value(outer_rows, "pre_race_attack_score")
    row["max_outer_makuri_rate"] = _max_value(outer_rows, "flow_prob_makuri")
    row["max_outer_makurizashi_rate"] = _max_value(outer_rows, "flow_prob_makurizashi")
    row["max_outer_sashi_rate"] = _max_value(outer_rows, "flow_prob_sashi")
    row["course2_sashi_rate"] = _course_value(frame, 2, "flow_prob_sashi")
    row["course2_makuri_rate"] = _course_value(frame, 2, "flow_prob_makuri")
    row["course3_attack_rate"] = max(
        _course_value(frame, 3, "flow_prob_makuri"),
        _course_value(frame, 3, "flow_prob_makurizashi"),
    )
    row["course4_attack_rate"] = max(
        _course_value(frame, 4, "flow_prob_makuri"),
        _course_value(frame, 4, "flow_prob_makurizashi"),
        _max_value(center_rows, "pre_race_attack_score"),
    )
    return row


def _build_target(lane_df: pd.DataFrame, *, target_name: str) -> pd.DataFrame:
    records: list[dict[str, int | str]] = []
    for race_id, race_df in lane_df.groupby("race_id", sort=False):
        course = pd.to_numeric(race_df.get("course"), errors="coerce")
        lane = pd.to_numeric(race_df.get("lane"), errors="coerce")
        in_rows = race_df[course == 1]
        if in_rows.empty:
            in_rows = race_df[lane == 1]
        if in_rows.empty:
            continue
        finish = pd.to_numeric(in_rows.iloc[0].get("finish_position"), errors="coerce")
        if pd.isna(finish):
            continue
        finish_int = int(finish)
        if target_name == "in_win":
            target = int(finish_int == 1)
        elif target_name == "in_collapse":
            # Collapse here means the in-course boat failed to stay in the top 2.
            target = int(finish_int >= 3)
        else:
            raise ValueError(f"Unsupported in-course probability target: {target_name}")
        records.append({"race_id": str(race_id), "target": target})
    return pd.DataFrame(records)


def _predict_payload_probability(payload: InCourseProbabilityPayload | dict[str, Any] | None, features: pd.DataFrame) -> np.ndarray:
    if features.empty:
        return np.asarray([], dtype=float)
    if payload is None:
        return np.zeros(len(features), dtype=float)
    model = payload.model if isinstance(payload, InCourseProbabilityPayload) else payload.get("model")
    feature_names = (
        payload.feature_names if isinstance(payload, InCourseProbabilityPayload) else payload.get("feature_names")
    ) or list(IN_COURSE_PROBABILITY_FEATURES)
    if model is None:
        return np.zeros(len(features), dtype=float)
    x = features.reindex(columns=feature_names, fill_value=0.0).to_numpy(dtype=float)
    return _predict_positive_probability(model, x)


def _ensure_in_course_probability_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in ("in_win_probability", "in_collapse_probability"):
        if column not in frame.columns:
            frame[column] = 0.0
    for column in ("in_win_probability", "in_collapse_probability"):
        score_column = f"{column}_score"
        if score_column not in frame.columns:
            frame[score_column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0) * 100.0
        label_column = f"{column}_label"
        if label_column not in frame.columns:
            frame[label_column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0).map(
                label_in_course_probability
            )
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


def _race_value(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return 0.0
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.iloc[0]) if not values.empty else 0.0


def _row_value(row: pd.Series, column: str) -> float:
    if column not in row:
        return 0.0
    value = pd.to_numeric(row.get(column), errors="coerce")
    return 0.0 if pd.isna(value) else float(value)


def _max_value(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return 0.0
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.max()) if not values.empty else 0.0


def _course_value(frame: pd.DataFrame, course_value: int, column: str) -> float:
    if "course" not in frame.columns:
        return 0.0
    course = pd.to_numeric(frame["course"], errors="coerce")
    rows = frame[course == course_value]
    if rows.empty:
        return 0.0
    return _row_value(rows.iloc[0], column)
