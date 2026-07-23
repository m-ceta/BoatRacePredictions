from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


def build_training_table(entries: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
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
    racer_sorted["racer_prev_best_finish_5"] = grouped["finish_position"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=2).min()
    )
    racer_sorted["racer_prev_worst_finish_5"] = grouped["finish_position"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=2).max()
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
    venue_course_columns = [
        "venue_course_prev_win_rate",
        "venue_course_prev_top2_rate",
        "venue_course_prev_top3_rate",
        "venue_course_prev_avg_finish",
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
    return racer_sorted


def add_race_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    race_groups = df.groupby("race_id")
    relative_columns = [
        "national_win_rate",
        "national_place_rate",
        "local_win_rate",
        "local_place_rate",
        "motor_place_rate",
        "boat_place_rate",
        "racer_prev_win_rate",
        "racer_prev_top3_rate",
        "racer_prev_top3_rate_5",
        "racer_prev_avg_finish_5",
        "racer_prev_avg_finish_10",
        "racer_prev_avg_st",
        "racer_prev_avg_st_5",
        "racer_prev_avg_st_10",
        "racer_prev_avg_exhibition",
        "racer_lane_prev_top3_rate",
        "racer_venue_lane_prev_top3_rate",
        "venue_course_prev_win_rate",
        "venue_course_prev_top2_rate",
        "venue_course_prev_top3_rate",
        "venue_course_prev_avg_finish",
        "venue_lane_prev_win_rate",
        "venue_lane_prev_top2_rate",
        "venue_lane_prev_top3_rate",
        "venue_lane_prev_avg_finish",
        "motor_prev_top3_rate",
        "boat_prev_top3_rate",
        "st_momentum_diff",
        "finish_momentum_diff",
    ]

    for column in relative_columns:
        df[f"{column}_race_rank"] = race_groups[column].rank(ascending=False, method="min")
        df[f"{column}_race_diff_mean"] = df[column] - race_groups[column].transform("mean")

    df = add_measurement_relative_features(df, race_groups)
    df = add_neighbor_gap_features(df)
    df = add_pre_race_attack_candidate_features(df)

    df["lane_is_inner"] = (df["lane"] <= 3).astype(int)
    df["lane_is_outer"] = (df["lane"] >= 5).astype(int)
    return df


def add_measurement_relative_features(df: pd.DataFrame, race_groups: pd.core.groupby.DataFrameGroupBy) -> pd.DataFrame:
    lower_is_better_columns = [
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
        "racer_prev_win_rate",
        "racer_prev_top3_rate",
        "racer_prev_top3_rate_5",
        "motor_place_rate",
        "boat_place_rate",
        "venue_course_prev_win_rate",
        "venue_course_prev_top2_rate",
        "venue_course_prev_top3_rate",
        "venue_lane_prev_win_rate",
        "venue_lane_prev_top2_rate",
        "venue_lane_prev_top3_rate",
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
        "national_win_rate",
        "national_place_rate",
        "racer_prev_win_rate",
        "racer_prev_top3_rate",
        "motor_place_rate",
        "boat_place_rate",
        "venue_course_prev_win_rate",
        "venue_course_prev_top3_rate",
    ]
    ordered = df.sort_values(["race_id", "lane"]).copy()
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
    return ordered.sort_index()


def add_pre_race_attack_candidate_features(df: pd.DataFrame) -> pd.DataFrame:
    if "lane" not in df.columns:
        return df
    frame = df.copy()
    lane = pd.to_numeric(frame["lane"], errors="coerce")
    st_score = _race_scale_feature(frame, "racer_prev_avg_st_5", lower_is_better=True).fillna(
        _race_scale_feature(frame, "racer_prev_avg_st", lower_is_better=True)
    )
    exhibition_score = _race_scale_feature(frame, "exhibition_time", lower_is_better=True)
    top3_score = _race_scale_feature(frame, "racer_prev_top3_rate", lower_is_better=False)
    motor_score = _race_scale_feature(frame, "motor_place_rate", lower_is_better=False)
    venue_score = _race_scale_feature(frame, "venue_course_prev_top3_rate", lower_is_better=False).fillna(
        _race_scale_feature(frame, "venue_lane_prev_top3_rate", lower_is_better=False)
    )
    lane_bias = lane.map({2: 0.08, 3: 0.13, 4: 0.15, 5: 0.08, 6: 0.05}).fillna(0.0)
    attack_score = (
        0.30 * st_score.fillna(0.0)
        + 0.18 * exhibition_score.fillna(0.0)
        + 0.18 * top3_score.fillna(0.0)
        + 0.14 * motor_score.fillna(0.0)
        + 0.12 * venue_score.fillna(0.0)
        + lane_bias
    )
    attack_score = attack_score.where(lane != 1, 0.0)
    frame["pre_race_attack_score"] = attack_score.astype("float32")

    candidate_score = frame.groupby("race_id")["pre_race_attack_score"].transform("max")
    candidate_lane_raw = (
        frame.assign(
            _candidate_lane=lane.where(
                (lane != 1)
                & (frame["pre_race_attack_score"] == candidate_score)
                & (candidate_score > 0)
            )
        )
        .groupby("race_id")["_candidate_lane"]
        .transform("min")
    )
    candidate_lane = candidate_lane_raw.fillna(0)
    has_candidate = candidate_lane > 0
    frame["pre_race_attack_candidate_lane"] = candidate_lane.astype("float32")
    frame["pre_race_attack_candidate_score"] = candidate_score.astype("float32")
    frame["pre_race_attack_score_gap_candidate"] = (
        frame["pre_race_attack_score"] - frame["pre_race_attack_candidate_score"]
    ).astype("float32")
    frame["distance_from_attack_candidate"] = (lane - candidate_lane).abs().where(has_candidate, 0).astype("float32")
    frame["is_attack_candidate"] = ((lane == candidate_lane) & has_candidate).astype("int8")
    frame["is_inside_of_attack_candidate"] = ((lane < candidate_lane) & has_candidate).astype("int8")
    frame["is_outside_of_attack_candidate"] = ((lane > candidate_lane) & has_candidate).astype("int8")
    frame["attack_candidate_inner_count"] = (candidate_lane - 1).clip(lower=0).where(has_candidate, 0).astype("float32")
    frame["attack_candidate_outer_count"] = (6 - candidate_lane).clip(lower=0).where(has_candidate, 0).astype("float32")
    return frame


def _race_scale_feature(df: pd.DataFrame, column: str, *, lower_is_better: bool) -> pd.Series:
    if column not in df.columns:
        return pd.Series(pd.NA, index=df.index, dtype="float32")
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
