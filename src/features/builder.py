from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


RACE_RELATIVE_GENERATED_EXACT_COLUMNS = {
    "lane_is_inner",
    "lane_is_outer",
    "entry_course_position",
    "entry_course_is_inner",
    "entry_course_is_outer",
    "entry_course_lane_gap",
    "pre_race_attack_score",
    "pre_race_attack_candidate_lane",
    "pre_race_attack_candidate_course",
    "pre_race_attack_candidate_score",
    "pre_race_attack_score_gap_candidate",
    "distance_from_attack_candidate",
    "is_attack_candidate",
    "is_inside_of_attack_candidate",
    "is_outside_of_attack_candidate",
    "attack_candidate_inner_count",
    "attack_candidate_outer_count",
    "race_escape_reliability_score",
    "race_attack_pressure",
    "race_inner_collapse_risk",
    "race_outer_link_risk",
    "race_inner3_avg_st",
    "race_outer3_avg_st",
    "race_outer_inner_avg_st_gap",
    "lane_st_gap_inner3_avg",
    "lane_st_gap_outer3_avg",
    "racer_prev_avg_st_window_best",
    "racer_prev_avg_st_window_worst",
    "racer_prev_avg_st_window_range",
    "start_timing_gap_st_window_best",
    "lane_escape_support_score",
    "lane_attack_pressure_gap",
    "lane_outer_link_fit",
}

DECISION_STYLE_KEYS = ("nige", "sashi", "makuri", "makurizashi", "nuki")
DECISION_STYLE_FLAG_COLUMNS = tuple(f"decision_style_{key}_win" for key in DECISION_STYLE_KEYS)

RACE_RELATIVE_GENERATED_SUFFIXES = (
    "_race_rank",
    "_race_rank_low",
    "_race_diff_mean",
    "_race_diff_mean_safe",
    "_race_diff_best",
    "_race_mean",
    "_race_std",
    "_race_zscore",
    "_gap_inner",
    "_gap_outer",
    "_gap_inner_mean",
    "_gap_outer_mean",
)


def build_training_table(entries: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    results = results.copy()
    if "trifecta_payout" not in results.columns:
        results["trifecta_payout"] = np.nan
    merged = entries.merge(
        results[
            [
                "race_id",
                "lane",
                "finish_position",
                "exhibition_time",
                "course",
                "start_timing",
                "weather",
                "wind_direction",
                "wind_speed_m",
                "wave_cm",
                "winning_style",
                "trifecta_payout",
            ]
        ],
        on=["race_id", "lane"],
        how="inner",
        validate="one_to_one",
    ).copy()

    merged["race_date"] = pd.to_datetime(merged["race_date"])
    merged = merged.sort_values(["race_date", "race_no", "lane"]).reset_index(drop=True)
    merged["target_rank"] = 7 - merged["finish_position"]
    merged["is_win"] = (merged["finish_position"] == 1).astype(int)
    merged["is_top2"] = (merged["finish_position"] <= 2).astype(int)
    merged["is_top3"] = (merged["finish_position"] <= 3).astype(int)

    merged = add_racer_history_features(merged)
    merged = add_race_relative_features(merged)
    merged = add_current_meet_features(merged)
    return merged


def add_racer_history_features(df: pd.DataFrame) -> pd.DataFrame:
    racer_sorted = df.sort_values(["racer_id", "race_date", "race_no"]).copy()
    racer_sorted = add_decision_style_flag_columns(racer_sorted)
    grouped = racer_sorted.groupby("racer_id", group_keys=False)

    racer_sorted["racer_prev_count"] = grouped.cumcount()
    racer_sorted["racer_prev_win_rate"] = grouped["is_win"].transform(
        lambda s: s.shift(1).rolling(30, min_periods=3).mean()
    )
    racer_sorted["racer_prev_top3_rate"] = grouped["is_top3"].transform(
        lambda s: s.shift(1).rolling(30, min_periods=3).mean()
    )
    racer_sorted["racer_prev_avg_finish"] = grouped["finish_position"].transform(
        lambda s: s.shift(1).rolling(30, min_periods=3).mean()
    )
    racer_sorted["racer_prev_avg_st"] = grouped["start_timing"].transform(
        lambda s: s.shift(1).rolling(30, min_periods=3).mean()
    )
    racer_sorted["racer_prev_avg_exhibition"] = grouped["exhibition_time"].transform(
        lambda s: s.shift(1).rolling(30, min_periods=3).mean()
    )
    racer_sorted["racer_prev_win_rate_5"] = grouped["is_win"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=2).mean()
    )
    racer_sorted["racer_prev_top3_rate_5"] = grouped["is_top3"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=2).mean()
    )
    racer_sorted["racer_prev_avg_finish_5"] = grouped["finish_position"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=2).mean()
    )
    racer_sorted["racer_prev_avg_finish_10"] = grouped["finish_position"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=3).mean()
    )
    racer_sorted["racer_prev_avg_st_5"] = grouped["start_timing"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=2).mean()
    )
    racer_sorted["racer_prev_avg_st_10"] = grouped["start_timing"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=3).mean()
    )
    racer_sorted["racer_prev_std_st_10"] = grouped["start_timing"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=3).std()
    )
    racer_sorted["racer_prev_best_st_5"] = grouped["start_timing"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=2).min()
    )
    racer_sorted["racer_prev_best_st_10"] = grouped["start_timing"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=3).min()
    )
    racer_sorted["racer_prev_best_st_30"] = grouped["start_timing"].transform(
        lambda s: s.shift(1).rolling(30, min_periods=3).min()
    )
    racer_sorted["racer_prev_worst_st_5"] = grouped["start_timing"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=2).max()
    )
    racer_sorted["racer_prev_worst_st_10"] = grouped["start_timing"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=3).max()
    )
    racer_sorted["racer_prev_worst_st_30"] = grouped["start_timing"].transform(
        lambda s: s.shift(1).rolling(30, min_periods=3).max()
    )
    racer_sorted["racer_prev_best_finish_5"] = grouped["finish_position"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=2).min()
    )
    racer_sorted["racer_prev_worst_finish_5"] = grouped["finish_position"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=2).max()
    )
    for style_key in DECISION_STYLE_KEYS:
        source = f"decision_style_{style_key}_win"
        racer_sorted[f"racer_prev_{style_key}_rate"] = grouped[source].transform(
            lambda s: s.shift(1).rolling(30, min_periods=3).mean()
        )

    venue_grouped = racer_sorted.groupby(["racer_id", "venue"], group_keys=False)
    racer_sorted["racer_venue_prev_win_rate"] = venue_grouped["is_win"].transform(
        lambda s: s.shift(1).rolling(15, min_periods=2).mean()
    )
    racer_sorted["racer_venue_prev_top3_rate"] = venue_grouped["is_top3"].transform(
        lambda s: s.shift(1).rolling(15, min_periods=2).mean()
    )

    lane_grouped = racer_sorted.groupby(["racer_id", "lane"], group_keys=False)
    racer_sorted["racer_lane_prev_win_rate"] = lane_grouped["is_win"].transform(
        lambda s: s.shift(1).rolling(15, min_periods=2).mean()
    )
    racer_sorted["racer_lane_prev_top3_rate"] = lane_grouped["is_top3"].transform(
        lambda s: s.shift(1).rolling(15, min_periods=2).mean()
    )
    racer_sorted["racer_lane_prev_avg_st"] = lane_grouped["start_timing"].transform(
        lambda s: s.shift(1).rolling(15, min_periods=2).mean()
    )
    racer_sorted["racer_lane_prev_avg_finish"] = lane_grouped["finish_position"].transform(
        lambda s: s.shift(1).rolling(15, min_periods=2).mean()
    )

    venue_lane_grouped = racer_sorted.groupby(["racer_id", "venue", "lane"], group_keys=False)
    racer_sorted["racer_venue_lane_prev_top3_rate"] = venue_lane_grouped["is_top3"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=2).mean()
    )
    racer_sorted["racer_venue_lane_prev_avg_st"] = venue_lane_grouped["start_timing"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=2).mean()
    )

    venue_course_sorted = racer_sorted.sort_values(["venue", "course", "race_date", "race_no", "lane"]).copy()
    venue_course_grouped = venue_course_sorted.dropna(subset=["course"]).groupby(["venue", "course"], group_keys=False)
    venue_course_sorted["venue_course_prev_win_rate"] = venue_course_grouped["is_win"].transform(
        lambda s: s.shift(1).rolling(200, min_periods=30).mean()
    )
    venue_course_sorted["venue_course_prev_top2_rate"] = venue_course_grouped["is_top2"].transform(
        lambda s: s.shift(1).rolling(200, min_periods=30).mean()
    )
    venue_course_sorted["venue_course_prev_top3_rate"] = venue_course_grouped["is_top3"].transform(
        lambda s: s.shift(1).rolling(200, min_periods=30).mean()
    )
    venue_course_sorted["venue_course_prev_avg_finish"] = venue_course_grouped["finish_position"].transform(
        lambda s: s.shift(1).rolling(200, min_periods=30).mean()
    )
    for style_key in DECISION_STYLE_KEYS:
        source = f"decision_style_{style_key}_win"
        venue_course_sorted[f"venue_course_prev_{style_key}_rate"] = venue_course_grouped[source].transform(
            lambda s: s.shift(1).rolling(200, min_periods=30).mean()
        )
    venue_course_columns = [
        "venue_course_prev_win_rate",
        "venue_course_prev_top2_rate",
        "venue_course_prev_top3_rate",
        "venue_course_prev_avg_finish",
        *[f"venue_course_prev_{style_key}_rate" for style_key in DECISION_STYLE_KEYS],
    ]
    racer_sorted[venue_course_columns] = venue_course_sorted[venue_course_columns].reindex(racer_sorted.index)

    venue_lane_sorted = racer_sorted.sort_values(["venue", "lane", "race_date", "race_no"]).copy()
    venue_lane_overall_grouped = venue_lane_sorted.groupby(["venue", "lane"], group_keys=False)
    venue_lane_sorted["venue_lane_prev_win_rate"] = venue_lane_overall_grouped["is_win"].transform(
        lambda s: s.shift(1).rolling(200, min_periods=30).mean()
    )
    venue_lane_sorted["venue_lane_prev_top2_rate"] = venue_lane_overall_grouped["is_top2"].transform(
        lambda s: s.shift(1).rolling(200, min_periods=30).mean()
    )
    venue_lane_sorted["venue_lane_prev_top3_rate"] = venue_lane_overall_grouped["is_top3"].transform(
        lambda s: s.shift(1).rolling(200, min_periods=30).mean()
    )
    venue_lane_sorted["venue_lane_prev_avg_finish"] = venue_lane_overall_grouped["finish_position"].transform(
        lambda s: s.shift(1).rolling(200, min_periods=30).mean()
    )
    venue_lane_columns = [
        "venue_lane_prev_win_rate",
        "venue_lane_prev_top2_rate",
        "venue_lane_prev_top3_rate",
        "venue_lane_prev_avg_finish",
    ]
    racer_sorted[venue_lane_columns] = venue_lane_sorted[venue_lane_columns].reindex(racer_sorted.index)

    course_grouped = racer_sorted.groupby(["racer_id", "course"], group_keys=False)
    racer_sorted["racer_course_prev_top3_rate"] = course_grouped["is_top3"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=2).mean()
    )
    racer_sorted["racer_course_prev_avg_finish"] = course_grouped["finish_position"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=2).mean()
    )
    for style_key in DECISION_STYLE_KEYS:
        source = f"decision_style_{style_key}_win"
        racer_sorted[f"racer_course_prev_{style_key}_rate"] = course_grouped[source].transform(
            lambda s: s.shift(1).rolling(15, min_periods=2).mean()
        )

    motor_grouped = racer_sorted.groupby("motor_no", group_keys=False)
    racer_sorted["motor_prev_top3_rate"] = motor_grouped["is_top3"].transform(
        lambda s: s.shift(1).rolling(40, min_periods=5).mean()
    )
    racer_sorted["motor_prev_win_rate"] = motor_grouped["is_win"].transform(
        lambda s: s.shift(1).rolling(40, min_periods=5).mean()
    )
    racer_sorted["motor_prev_avg_st"] = motor_grouped["start_timing"].transform(
        lambda s: s.shift(1).rolling(40, min_periods=5).mean()
    )
    boat_grouped = racer_sorted.groupby("boat_no", group_keys=False)
    racer_sorted["boat_prev_top3_rate"] = boat_grouped["is_top3"].transform(
        lambda s: s.shift(1).rolling(40, min_periods=5).mean()
    )
    racer_sorted["boat_prev_win_rate"] = boat_grouped["is_win"].transform(
        lambda s: s.shift(1).rolling(40, min_periods=5).mean()
    )
    racer_sorted["boat_prev_avg_st"] = boat_grouped["start_timing"].transform(
        lambda s: s.shift(1).rolling(40, min_periods=5).mean()
    )

    racer_sorted["st_momentum_diff"] = (
        racer_sorted["racer_prev_avg_st_5"] - racer_sorted["racer_prev_avg_st_10"]
    )
    racer_sorted["finish_momentum_diff"] = (
        racer_sorted["racer_prev_avg_finish_10"] - racer_sorted["racer_prev_avg_finish_5"]
    )
    racer_sorted = add_flow_probability_features(racer_sorted)
    return racer_sorted


def add_decision_style_flag_columns(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    style = frame.get("winning_style", pd.Series(pd.NA, index=frame.index)).map(_decision_style_key)
    is_winner = pd.to_numeric(frame.get("finish_position"), errors="coerce") == 1
    for style_key in DECISION_STYLE_KEYS:
        frame[f"decision_style_{style_key}_win"] = ((style == style_key) & is_winner).astype("int8")
    return frame


def add_flow_probability_features(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    for style_key in ("nige", "sashi", "makuri", "makurizashi"):
        frame[f"flow_prob_{style_key}"] = _weighted_feature_blend(
            frame,
            [
                (f"racer_course_prev_{style_key}_rate", 0.45),
                (f"racer_prev_{style_key}_rate", 0.25),
                (f"venue_course_prev_{style_key}_rate", 0.30),
            ],
        ).astype("float32")
    frame["racer_attack_style_score"] = (
        pd.to_numeric(frame.get("flow_prob_makuri", 0.0), errors="coerce").fillna(0.0)
        + pd.to_numeric(frame.get("flow_prob_makurizashi", 0.0), errors="coerce").fillna(0.0)
    ).clip(0.0, 1.0).astype("float32")
    frame["racer_sashi_style_score"] = pd.to_numeric(
        frame.get("flow_prob_sashi", 0.0),
        errors="coerce",
    ).fillna(0.0).clip(0.0, 1.0).astype("float32")
    return frame


def add_race_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    race_groups = df.groupby("race_id")
    entry_position = _entry_position_series(df)
    df["entry_course_position"] = entry_position.astype("float32")
    df["entry_course_is_inner"] = (entry_position <= 3).astype("int8")
    df["entry_course_is_outer"] = (entry_position >= 5).astype("int8")
    df["entry_course_lane_gap"] = (entry_position - pd.to_numeric(df["lane"], errors="coerce")).astype("float32")
    relative_columns = [
        "national_win_rate",
        "national_place_rate",
        "local_win_rate",
        "local_place_rate",
        "motor_place_rate",
        "boat_place_rate",
        "start_timing",
        "exhibition_time",
        "racer_prev_win_rate",
        "racer_prev_win_rate_5",
        "racer_prev_top3_rate",
        "racer_prev_top3_rate_5",
        "racer_prev_avg_finish",
        "racer_prev_avg_finish_5",
        "racer_prev_avg_finish_10",
        "racer_prev_avg_st",
        "racer_prev_avg_st_5",
        "racer_prev_avg_st_10",
        "racer_prev_best_st_5",
        "racer_prev_best_st_10",
        "racer_prev_best_st_30",
        "racer_prev_worst_st_5",
        "racer_prev_worst_st_10",
        "racer_prev_worst_st_30",
        "racer_prev_avg_exhibition",
        "racer_lane_prev_win_rate",
        "racer_lane_prev_top3_rate",
        "racer_lane_prev_avg_st",
        "racer_venue_lane_prev_top3_rate",
        "racer_venue_lane_prev_avg_st",
        "venue_course_prev_win_rate",
        "venue_course_prev_top2_rate",
        "venue_course_prev_top3_rate",
        "venue_course_prev_avg_finish",
        "venue_lane_prev_win_rate",
        "venue_lane_prev_top2_rate",
        "venue_lane_prev_top3_rate",
        "venue_lane_prev_avg_finish",
        "motor_prev_win_rate",
        "motor_prev_top3_rate",
        "motor_prev_avg_st",
        "boat_prev_win_rate",
        "boat_prev_top3_rate",
        "boat_prev_avg_st",
        "st_momentum_diff",
        "finish_momentum_diff",
        "racer_prev_nige_rate",
        "racer_prev_sashi_rate",
        "racer_prev_makuri_rate",
        "racer_prev_makurizashi_rate",
        "racer_course_prev_nige_rate",
        "racer_course_prev_sashi_rate",
        "racer_course_prev_makuri_rate",
        "racer_course_prev_makurizashi_rate",
        "venue_course_prev_nige_rate",
        "venue_course_prev_sashi_rate",
        "venue_course_prev_makuri_rate",
        "venue_course_prev_makurizashi_rate",
        "flow_prob_nige",
        "flow_prob_sashi",
        "flow_prob_makuri",
        "flow_prob_makurizashi",
        "racer_attack_style_score",
        "racer_sashi_style_score",
    ]

    for column in relative_columns:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        group_values = values.groupby(df["race_id"])
        df[f"{column}_race_rank"] = group_values.rank(ascending=False, method="min")
        df[f"{column}_race_diff_mean"] = values - group_values.transform("mean")

    df = add_measurement_relative_features(df, race_groups)
    df = add_neighbor_gap_features(df)
    df = add_inside_outside_mean_gap_features(df)
    df = add_proxy_st_structure_features(df)
    df = add_pre_race_attack_candidate_features(df)
    df = add_pre_race_flow_features(df)

    df["lane_is_inner"] = (df["lane"] <= 3).astype(int)
    df["lane_is_outer"] = (df["lane"] >= 5).astype(int)
    return df


def drop_race_relative_features(df: pd.DataFrame, preserve_missing_sources: bool = False) -> pd.DataFrame:
    """Remove derived race-relative columns before rebuilding them from proxy values."""
    drop_columns = [
        column
        for column in df.columns
        if _should_drop_race_relative_feature(column, df.columns, preserve_missing_sources)
    ]
    if not drop_columns:
        return df
    return df.drop(columns=drop_columns)


def _should_drop_race_relative_feature(
    column: str,
    available_columns: pd.Index | list[str],
    preserve_missing_sources: bool,
) -> bool:
    if column in RACE_RELATIVE_GENERATED_EXACT_COLUMNS:
        return True
    matched_suffix = next(
        (suffix for suffix in sorted(RACE_RELATIVE_GENERATED_SUFFIXES, key=len, reverse=True) if column.endswith(suffix)),
        None,
    )
    if matched_suffix is None:
        return False
    if not preserve_missing_sources:
        return True
    source_column = column[: -len(matched_suffix)]
    return source_column in available_columns


def add_measurement_relative_features(df: pd.DataFrame, race_groups: pd.core.groupby.DataFrameGroupBy) -> pd.DataFrame:
    lower_is_better_columns = [
        "start_timing",
        "exhibition_time",
        "racer_prev_avg_st",
        "racer_prev_avg_st_5",
        "racer_prev_avg_st_10",
        "racer_lane_prev_avg_st",
        "racer_venue_lane_prev_avg_st",
    ]
    higher_is_better_columns = [
        "national_win_rate",
        "national_place_rate",
        "local_win_rate",
        "local_place_rate",
        "racer_prev_win_rate_5",
        "racer_prev_win_rate",
        "racer_prev_top3_rate",
        "racer_prev_top3_rate_5",
        "motor_place_rate",
        "boat_place_rate",
        "motor_prev_win_rate",
        "venue_course_prev_win_rate",
        "venue_course_prev_top2_rate",
        "venue_course_prev_top3_rate",
        "venue_lane_prev_win_rate",
        "venue_lane_prev_top2_rate",
        "venue_lane_prev_top3_rate",
        "racer_prev_nige_rate",
        "racer_prev_sashi_rate",
        "racer_prev_makuri_rate",
        "racer_prev_makurizashi_rate",
        "racer_course_prev_nige_rate",
        "racer_course_prev_sashi_rate",
        "racer_course_prev_makuri_rate",
        "racer_course_prev_makurizashi_rate",
        "venue_course_prev_nige_rate",
        "venue_course_prev_sashi_rate",
        "venue_course_prev_makuri_rate",
        "venue_course_prev_makurizashi_rate",
        "flow_prob_nige",
        "flow_prob_sashi",
        "flow_prob_makuri",
        "flow_prob_makurizashi",
        "racer_attack_style_score",
        "racer_sashi_style_score",
    ]

    feature_frames: dict[str, pd.Series] = {}
    for column in lower_is_better_columns:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        group_values = values.groupby(df["race_id"])
        race_min = group_values.transform("min")
        race_mean = group_values.transform("mean")
        race_std = group_values.transform("std")
        feature_frames[f"{column}_race_rank_low"] = group_values.rank(ascending=True, method="min")
        feature_frames[f"{column}_race_diff_best"] = values - race_min
        feature_frames[f"{column}_race_diff_mean_safe"] = values - race_mean
        feature_frames[f"{column}_race_mean"] = race_mean
        feature_frames[f"{column}_race_std"] = race_std
        feature_frames[f"{column}_race_zscore"] = (values - race_mean) / race_std.replace(0, np.nan)

    for column in higher_is_better_columns:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        group_values = values.groupby(df["race_id"])
        race_max = group_values.transform("max")
        race_mean = group_values.transform("mean")
        race_std = group_values.transform("std")
        feature_frames[f"{column}_race_diff_best"] = values - race_max
        feature_frames[f"{column}_race_mean"] = race_mean
        feature_frames[f"{column}_race_std"] = race_std
        feature_frames[f"{column}_race_zscore"] = (values - race_mean) / race_std.replace(0, np.nan)
    if not feature_frames:
        return df
    return pd.concat([df, pd.DataFrame(feature_frames, index=df.index)], axis=1)


def add_neighbor_gap_features(df: pd.DataFrame) -> pd.DataFrame:
    gap_columns = [
        "racer_prev_avg_st",
        "racer_prev_avg_st_5",
        "racer_prev_avg_st_10",
        "racer_lane_prev_avg_st",
        "racer_venue_lane_prev_avg_st",
        "exhibition_time",
        "start_timing",
        "national_win_rate",
        "national_place_rate",
        "local_win_rate",
        "local_place_rate",
        "racer_prev_win_rate",
        "racer_prev_win_rate_5",
        "racer_prev_top3_rate",
        "racer_lane_prev_win_rate",
        "motor_place_rate",
        "motor_prev_win_rate",
        "boat_place_rate",
        "boat_prev_win_rate",
        "venue_course_prev_win_rate",
        "venue_course_prev_top2_rate",
        "venue_course_prev_top3_rate",
        "flow_prob_nige",
        "flow_prob_sashi",
        "flow_prob_makuri",
        "flow_prob_makurizashi",
        "racer_attack_style_score",
        "racer_sashi_style_score",
    ]
    ordered = df.copy()
    ordered["_entry_position"] = _entry_position_series(ordered)
    ordered = ordered.sort_values(["race_id", "_entry_position", "lane"]).copy()
    feature_frames: dict[str, pd.Series] = {}
    for column in gap_columns:
        if column not in ordered.columns:
            continue
        values = pd.to_numeric(ordered[column], errors="coerce")
        inner_values = values.groupby(ordered["race_id"]).shift(1)
        outer_values = values.groupby(ordered["race_id"]).shift(-1)
        feature_frames[f"{column}_gap_inner"] = values - inner_values
        feature_frames[f"{column}_gap_outer"] = values - outer_values
    if feature_frames:
        ordered = pd.concat([ordered, pd.DataFrame(feature_frames, index=ordered.index)], axis=1)
    return ordered.drop(columns="_entry_position").sort_index()


def add_inside_outside_mean_gap_features(df: pd.DataFrame) -> pd.DataFrame:
    gap_columns = [
        "national_win_rate",
        "national_place_rate",
        "local_win_rate",
        "local_place_rate",
        "racer_prev_win_rate",
        "racer_prev_win_rate_5",
        "racer_prev_top3_rate",
        "racer_lane_prev_win_rate",
        "motor_place_rate",
        "motor_prev_win_rate",
        "boat_place_rate",
        "boat_prev_win_rate",
        "exhibition_time",
        "start_timing",
        "venue_course_prev_win_rate",
        "venue_course_prev_top2_rate",
        "venue_course_prev_top3_rate",
        "flow_prob_nige",
        "flow_prob_sashi",
        "flow_prob_makuri",
        "flow_prob_makurizashi",
        "racer_attack_style_score",
        "racer_sashi_style_score",
    ]
    ordered = df.copy()
    ordered["_entry_position"] = _entry_position_series(ordered)
    ordered = ordered.sort_values(["race_id", "_entry_position", "lane"]).copy()
    feature_frames: dict[str, pd.Series] = {}
    for column in gap_columns:
        if column not in ordered.columns:
            continue
        values = pd.to_numeric(ordered[column], errors="coerce")
        race_keys = ordered["race_id"]
        total = values.groupby(race_keys).transform("sum")
        count = values.notna().astype("int64").groupby(race_keys).transform("sum")
        inner_sum = values.groupby(race_keys).cumsum() - values
        inner_count = values.notna().astype("int64").groupby(race_keys).cumsum() - values.notna().astype("int64")
        outer_sum = total - values.groupby(race_keys).cumsum()
        outer_count = count - values.notna().astype("int64").groupby(race_keys).cumsum()
        inner_mean = inner_sum / inner_count.replace(0, np.nan)
        outer_mean = outer_sum / outer_count.replace(0, np.nan)
        feature_frames[f"{column}_gap_inner_mean"] = values - inner_mean
        feature_frames[f"{column}_gap_outer_mean"] = values - outer_mean
    if feature_frames:
        ordered = pd.concat([ordered, pd.DataFrame(feature_frames, index=ordered.index)], axis=1)
    return ordered.drop(columns="_entry_position").sort_index()


def add_proxy_st_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "race_id" not in df.columns or "lane" not in df.columns:
        return df
    frame = df.copy()
    entry_position = _entry_position_series(frame)
    st = pd.to_numeric(frame.get("start_timing"), errors="coerce")
    race_keys = frame["race_id"]

    inner_st = st.where(entry_position <= 3)
    outer_st = st.where(entry_position >= 4)
    inner3_avg = inner_st.groupby(race_keys).transform("mean")
    outer3_avg = outer_st.groupby(race_keys).transform("mean")
    frame["race_inner3_avg_st"] = inner3_avg.astype("float32")
    frame["race_outer3_avg_st"] = outer3_avg.astype("float32")
    frame["race_outer_inner_avg_st_gap"] = (outer3_avg - inner3_avg).astype("float32")
    frame["lane_st_gap_inner3_avg"] = (st - inner3_avg).astype("float32")
    frame["lane_st_gap_outer3_avg"] = (st - outer3_avg).astype("float32")

    st_windows = [
        pd.to_numeric(frame[column], errors="coerce")
        for column in ("racer_prev_best_st_5", "racer_prev_best_st_10", "racer_prev_best_st_30")
        if column in frame.columns
    ]
    worst_st_windows = [
        pd.to_numeric(frame[column], errors="coerce")
        for column in ("racer_prev_worst_st_5", "racer_prev_worst_st_10", "racer_prev_worst_st_30")
        if column in frame.columns
    ]
    window_best = (
        pd.concat(st_windows, axis=1).min(axis=1, skipna=True)
        if st_windows
        else pd.Series(np.nan, index=frame.index)
    )
    window_worst = (
        pd.concat(worst_st_windows, axis=1).max(axis=1, skipna=True)
        if worst_st_windows
        else pd.Series(np.nan, index=frame.index)
    )
    frame["racer_prev_avg_st_window_best"] = window_best.astype("float32")
    frame["racer_prev_avg_st_window_worst"] = window_worst.astype("float32")
    frame["racer_prev_avg_st_window_range"] = (window_worst - window_best).astype("float32")
    frame["start_timing_gap_st_window_best"] = (st - window_best).astype("float32")
    return frame


def add_pre_race_attack_candidate_features(df: pd.DataFrame) -> pd.DataFrame:
    if "lane" not in df.columns:
        return df
    frame = df.copy()
    entry_position = _entry_position_series(frame)
    st_score = _race_scale_feature(frame, "racer_prev_avg_st_5", lower_is_better=True).fillna(
        _race_scale_feature(frame, "racer_prev_avg_st", lower_is_better=True)
    )
    exhibition_score = _race_scale_feature(frame, "exhibition_time", lower_is_better=True)
    top3_score = _race_scale_feature(frame, "racer_prev_top3_rate", lower_is_better=False)
    motor_score = _race_scale_feature(frame, "motor_place_rate", lower_is_better=False)
    venue_score = _race_scale_feature(frame, "venue_course_prev_top3_rate", lower_is_better=False).fillna(
        _race_scale_feature(frame, "venue_lane_prev_top3_rate", lower_is_better=False)
    )
    attack_style_score = _race_scale_feature(frame, "racer_attack_style_score", lower_is_better=False)
    lane_bias = entry_position.map({2: 0.08, 3: 0.13, 4: 0.15, 5: 0.08, 6: 0.05}).fillna(0.0)
    attack_score = (
        0.27 * st_score.fillna(0.0)
        + 0.16 * exhibition_score.fillna(0.0)
        + 0.16 * top3_score.fillna(0.0)
        + 0.13 * motor_score.fillna(0.0)
        + 0.11 * venue_score.fillna(0.0)
        + 0.09 * attack_style_score.fillna(0.0)
        + lane_bias
    )
    attack_score = attack_score.where(entry_position != 1, 0.0)
    frame["pre_race_attack_score"] = attack_score.astype("float32")

    candidate_score = frame.groupby("race_id")["pre_race_attack_score"].transform("max")
    candidate_lane_raw = (
        frame.assign(
            _candidate_lane=entry_position.where(
                (entry_position != 1)
                & (frame["pre_race_attack_score"] == candidate_score)
                & (candidate_score > 0)
            )
        )
        .groupby("race_id")["_candidate_lane"]
        .transform("min")
    )
    candidate_lane = candidate_lane_raw.fillna(0)
    has_candidate = candidate_lane > 0
    frame["pre_race_attack_candidate_course"] = candidate_lane.astype("float32")
    frame["pre_race_attack_candidate_lane"] = candidate_lane.astype("float32")
    frame["pre_race_attack_candidate_score"] = candidate_score.astype("float32")
    frame["pre_race_attack_score_gap_candidate"] = (
        frame["pre_race_attack_score"] - frame["pre_race_attack_candidate_score"]
    ).astype("float32")
    frame["distance_from_attack_candidate"] = (entry_position - candidate_lane).abs().where(has_candidate, 0).astype("float32")
    frame["is_attack_candidate"] = ((entry_position == candidate_lane) & has_candidate).astype("int8")
    frame["is_inside_of_attack_candidate"] = ((entry_position < candidate_lane) & has_candidate).astype("int8")
    frame["is_outside_of_attack_candidate"] = ((entry_position > candidate_lane) & has_candidate).astype("int8")
    frame["attack_candidate_inner_count"] = (candidate_lane - 1).clip(lower=0).where(has_candidate, 0).astype("float32")
    frame["attack_candidate_outer_count"] = (6 - candidate_lane).clip(lower=0).where(has_candidate, 0).astype("float32")
    return frame


def add_pre_race_flow_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "race_id" not in df.columns or "lane" not in df.columns:
        return df
    frame = df.copy()
    lane = pd.to_numeric(frame["lane"], errors="coerce")
    entry_position = _entry_position_series(frame)

    win_score = _race_scale_feature(frame, "national_win_rate", lower_is_better=False).fillna(
        _race_scale_feature(frame, "racer_prev_win_rate", lower_is_better=False)
    )
    local_score = _race_scale_feature(frame, "local_win_rate", lower_is_better=False).fillna(win_score)
    top3_score = _race_scale_feature(frame, "racer_prev_top3_rate", lower_is_better=False)
    st_score = _race_scale_feature(frame, "start_timing", lower_is_better=True).fillna(
        _race_scale_feature(frame, "racer_prev_avg_st_5", lower_is_better=True)
    )
    exhibition_score = _race_scale_feature(frame, "exhibition_time", lower_is_better=True)
    motor_score = _race_scale_feature(frame, "motor_place_rate", lower_is_better=False).fillna(
        _race_scale_feature(frame, "motor_prev_win_rate", lower_is_better=False)
    )
    venue_win_score = _race_scale_feature(frame, "venue_course_prev_win_rate", lower_is_better=False).fillna(
        _race_scale_feature(frame, "venue_lane_prev_win_rate", lower_is_better=False)
    )
    venue_top3_score = _race_scale_feature(frame, "venue_course_prev_top3_rate", lower_is_better=False).fillna(
        _race_scale_feature(frame, "venue_lane_prev_top3_rate", lower_is_better=False)
    )

    lane1_mask = entry_position == 1
    lane1_base = (
        0.25 * win_score.fillna(0.0)
        + 0.12 * local_score.fillna(0.0)
        + 0.18 * top3_score.fillna(0.0)
        + 0.16 * st_score.fillna(0.0)
        + 0.10 * exhibition_score.fillna(0.0)
        + 0.10 * motor_score.fillna(0.0)
        + 0.09 * venue_win_score.fillna(0.0)
    ).where(lane1_mask, np.nan)

    lane1_strength = lane1_base.groupby(frame["race_id"]).transform("max").fillna(0.0)
    attack_pressure = pd.to_numeric(
        frame.get("pre_race_attack_candidate_score", pd.Series(0.0, index=frame.index)),
        errors="coerce",
    ).fillna(0.0)
    attack_lane = pd.to_numeric(
        frame.get("pre_race_attack_candidate_course", frame.get("pre_race_attack_candidate_lane", pd.Series(0.0, index=frame.index))),
        errors="coerce",
    ).fillna(0.0)
    race_attack_pressure = attack_pressure.groupby(frame["race_id"]).transform("max").fillna(0.0)

    outer_lane_bias = entry_position.map({4: 0.08, 5: 0.12, 6: 0.14}).fillna(0.0)
    pre_race_attack_score = pd.to_numeric(
        frame.get("pre_race_attack_score", pd.Series(0.0, index=frame.index)),
        errors="coerce",
    ).fillna(0.0)
    outer_score = (
        0.32 * pre_race_attack_score
        + 0.22 * venue_top3_score.fillna(0.0)
        + 0.18 * motor_score.fillna(0.0)
        + 0.16 * st_score.fillna(0.0)
        + 0.12 * exhibition_score.fillna(0.0)
        + outer_lane_bias
    ).where(entry_position >= 4, 0.0)
    race_outer_link_risk = outer_score.groupby(frame["race_id"]).transform("max").fillna(0.0).clip(0.0, 1.0)
    race_escape_reliability = (lane1_strength * (1.0 - 0.35 * race_attack_pressure)).clip(0.0, 1.0)
    race_inner_collapse_risk = (
        (1.0 - race_escape_reliability) * (0.45 + 0.35 * race_attack_pressure)
        + 0.20 * race_outer_link_risk
    ).clip(0.0, 1.0)

    frame["race_escape_reliability_score"] = race_escape_reliability.astype("float32")
    frame["race_attack_pressure"] = race_attack_pressure.astype("float32")
    frame["race_inner_collapse_risk"] = race_inner_collapse_risk.astype("float32")
    frame["race_outer_link_risk"] = race_outer_link_risk.astype("float32")
    frame["lane_escape_support_score"] = (
        race_escape_reliability * entry_position.map({1: 1.0, 2: 0.65, 3: 0.45, 4: 0.20, 5: 0.10, 6: 0.05}).fillna(0.0)
    ).astype("float32")
    frame["lane_attack_pressure_gap"] = (
        pre_race_attack_score - race_attack_pressure
    ).astype("float32")
    frame["lane_outer_link_fit"] = (
        race_outer_link_risk
        * entry_position.map({1: 0.05, 2: 0.12, 3: 0.24, 4: 0.55, 5: 0.75, 6: 0.65}).fillna(0.0)
        * (1.0 + 0.15 * (entry_position == attack_lane).astype("float32"))
    ).clip(0.0, 1.0).astype("float32")
    return frame


def _entry_position_series(df: pd.DataFrame) -> pd.Series:
    fallback = pd.to_numeric(df.get("lane", pd.Series(np.nan, index=df.index)), errors="coerce")
    if "course" not in df.columns:
        return fallback
    course = pd.to_numeric(df["course"], errors="coerce")
    return course.fillna(fallback)


def _race_scale_feature(df: pd.DataFrame, column: str, *, lower_is_better: bool) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float32")
    values = pd.to_numeric(df[column], errors="coerce")
    group_values = values.groupby(df["race_id"])
    race_min = group_values.transform("min")
    race_max = group_values.transform("max")
    denom = (race_max - race_min).replace(0, np.nan)
    if lower_is_better:
        scaled = (race_max - values) / denom
    else:
        scaled = (values - race_min) / denom
    return scaled.clip(lower=0.0, upper=1.0).astype("float32")


def _decision_style_key(value: object) -> str:
    text = "" if value is None or pd.isna(value) else str(value)
    if "逃" in text:
        return "nige"
    if "まくり差" in text:
        return "makurizashi"
    if "まくり" in text:
        return "makuri"
    if "差" in text:
        return "sashi"
    if "抜" in text:
        return "nuki"
    return "unknown"


def _weighted_feature_blend(df: pd.DataFrame, weighted_columns: list[tuple[str, float]]) -> pd.Series:
    total = pd.Series(0.0, index=df.index, dtype="float64")
    weight_total = pd.Series(0.0, index=df.index, dtype="float64")
    for column, weight in weighted_columns:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        mask = values.notna()
        total = total + values.fillna(0.0) * float(weight)
        weight_total = weight_total + mask.astype("float64") * float(weight)
    blended = total / weight_total.replace(0.0, np.nan)
    return blended.fillna(0.0).clip(lower=0.0, upper=1.0)


def add_current_meet_features(df: pd.DataFrame) -> pd.DataFrame:
    parsed = [parse_current_meet_results(value) for value in df["current_meet_results"].fillna("")]
    parsed_df = pd.DataFrame.from_records(parsed, index=df.index)
    for column in parsed_df.columns:
        parsed_df[column] = pd.to_numeric(parsed_df[column], errors="coerce").astype("float32")

    meet_digits = df["current_meet_results"].fillna("").str.replace(r"[^0-9]", "", regex=True)
    extras = pd.DataFrame(index=df.index)
    extras["current_meet_race_count"] = meet_digits.str.len().astype("float32")
    extras["current_meet_avg_finish_raw"] = pd.to_numeric(
        meet_digits.apply(_average_finish_from_digits),
        errors="coerce",
    ).astype("float32")
    extras["current_meet_recent_form_gap"] = (
        extras["current_meet_avg_finish_raw"] - parsed_df["current_meet_last3_avg_finish"]
    ).astype("float32")
    extras["current_meet_top3_share"] = (
        parsed_df["current_meet_top3_count"] / extras["current_meet_race_count"].where(extras["current_meet_race_count"] != 0)
    ).astype("float32")
    return pd.concat([df, parsed_df, extras], axis=1)


def _average_finish_from_digits(value: str) -> float | None:
    if not value:
        return None
    digits = [int(ch) for ch in value if ch.isdigit()]
    if not digits:
        return None
    return sum(digits) / len(digits)


def parse_current_meet_results(value: str) -> dict[str, float | int | None]:
    digits = [int(ch) for ch in value if ch.isdigit()]
    starts = re.findall(r"[FSKL]", value)

    features: dict[str, float | int | None] = {
        "current_meet_last_finish": digits[-1] if digits else None,
        "current_meet_last2_avg_finish": _average_subset(digits[-2:]),
        "current_meet_last3_avg_finish": _average_subset(digits[-3:]),
        "current_meet_best_finish": min(digits) if digits else None,
        "current_meet_worst_finish": max(digits) if digits else None,
        "current_meet_top3_count": sum(1 for x in digits if x <= 3),
        "current_meet_win_count": sum(1 for x in digits if x == 1),
        "current_meet_start_issue_count": len(starts),
        "current_meet_f_count": starts.count("F"),
        "current_meet_s_count": starts.count("S"),
        "current_meet_k_count": starts.count("K"),
        "current_meet_l_count": starts.count("L"),
    }

    for finish in range(1, 7):
        features[f"current_meet_finish_{finish}_count"] = digits.count(finish)
    return features


def _average_subset(values: list[int]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def save_processed_tables(
    entries: pd.DataFrame,
    results: pd.DataFrame,
    training_table: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    entries.to_parquet(output_dir / "race_entries.parquet", index=False)
    results.to_parquet(output_dir / "race_results.parquet", index=False)
    training_table.to_parquet(output_dir / "training_table.parquet", index=False)
