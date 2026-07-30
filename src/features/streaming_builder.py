from __future__ import annotations

import re
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.features.builder import add_current_meet_features, add_race_relative_features
from src.parsers.bk_parser import parse_entry_file, parse_result_file


ROWDATA_FILE_RE = re.compile(r"^[BK](?P<yy>\d{2})(?P<mm>\d{2})(?P<dd>\d{2})\.TXT$", re.IGNORECASE)

ENTRY_PARQUET_SCHEMA = pa.schema(
    [
        ("race_id", pa.string()),
        ("race_date", pa.date32()),
        ("venue", pa.string()),
        ("race_no", pa.int64()),
        ("race_title", pa.string()),
        ("leg_type", pa.string()),
        ("distance_m", pa.int64()),
        ("bet_type", pa.string()),
        ("lane", pa.int64()),
        ("racer_id", pa.int64()),
        ("racer_name", pa.string()),
        ("age", pa.int64()),
        ("branch", pa.string()),
        ("weight", pa.int64()),
        ("class_name", pa.string()),
        ("national_win_rate", pa.float64()),
        ("national_place_rate", pa.float64()),
        ("local_win_rate", pa.float64()),
        ("local_place_rate", pa.float64()),
        ("motor_no", pa.int64()),
        ("motor_place_rate", pa.float64()),
        ("boat_no", pa.int64()),
        ("boat_place_rate", pa.float64()),
        ("current_meet_results", pa.string()),
        ("early_lane_hint", pa.int64()),
    ]
)

RESULT_PARQUET_SCHEMA = pa.schema(
    [
        ("race_id", pa.string()),
        ("race_date", pa.date32()),
        ("venue", pa.string()),
        ("race_no", pa.int64()),
        ("lane", pa.int64()),
        ("finish_position", pa.int64()),
        ("finish_status", pa.string()),
        ("racer_id", pa.int64()),
        ("racer_name", pa.string()),
        ("motor_no", pa.int64()),
        ("boat_no", pa.int64()),
        ("exhibition_time", pa.float64()),
        ("course", pa.int64()),
        ("start_timing", pa.float64()),
        ("race_time", pa.float64()),
        ("weather", pa.string()),
        ("wind_direction", pa.string()),
        ("wind_speed_m", pa.int64()),
        ("wave_cm", pa.int64()),
        ("winning_style", pa.string()),
    ]
)


@dataclass(slots=True)
class BuildSummary:
    entries_rows: int
    results_rows: int
    training_rows: int
    min_date: str | None
    max_date: str | None
    output_dir: str

    def to_dict(self) -> dict[str, object]:
        return {
            "entries_rows": self.entries_rows,
            "results_rows": self.results_rows,
            "training_rows": self.training_rows,
            "min_date": self.min_date,
            "max_date": self.max_date,
            "output_dir": self.output_dir,
        }


class IncrementalParquetWriter:
    def __init__(self, output_path: Path, schema: pa.Schema | None = None) -> None:
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.schema = schema
        self._writer: pq.ParquetWriter | None = None

    def write(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        frame = _normalize_frame_for_arrow(frame)
        if self.schema is not None:
            frame = frame.reindex(columns=self.schema.names)
            table = pa.Table.from_pandas(frame, schema=self.schema, preserve_index=False, safe=False)
        else:
            table = pa.Table.from_pandas(frame, preserve_index=False)
        if self._writer is None:
            self._writer = pq.ParquetWriter(self.output_path, self.schema or table.schema)
        self._writer.write_table(table)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None


def _normalize_frame_for_arrow(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    for column in normalized.columns:
        series = normalized[column]
        if not pd.api.types.is_object_dtype(series):
            continue
        non_null = series.dropna()
        if non_null.empty or non_null.map(lambda value: isinstance(value, str)).all():
            normalized[column] = series.astype("string")
    return normalized


def parse_rowdata_file_date(path: Path) -> date | None:
    match = ROWDATA_FILE_RE.match(path.name)
    if not match:
        return None
    yy = int(match.group("yy"))
    year = 1900 + yy if yy >= 90 else 2000 + yy
    return date(year, int(match.group("mm")), int(match.group("dd")))


def iter_month_groups(rowdata_dir: Path, max_date: date | None = None) -> list[tuple[str, list[Path], list[Path]]]:
    grouped: dict[str, dict[str, list[Path]]] = {}
    for kind in ("B", "K"):
        for path in sorted(rowdata_dir.glob(f"{kind}*.TXT")):
            file_date = parse_rowdata_file_date(path)
            if file_date is None:
                continue
            if max_date is not None and file_date > max_date:
                continue
            month_key = f"{file_date.year:04d}{file_date.month:02d}"
            grouped.setdefault(month_key, {"B": [], "K": []})[kind].append(path)

    months: list[tuple[str, list[Path], list[Path]]] = []
    for month_key in sorted(grouped):
        months.append((month_key, grouped[month_key]["B"], grouped[month_key]["K"]))
    return months


def _build_base_chunk(entries_df: pd.DataFrame, results_df: pd.DataFrame) -> pd.DataFrame:
    merged = entries_df.merge(
        results_df[
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

    if merged.empty:
        return merged

    merged["race_date"] = pd.to_datetime(merged["race_date"])
    merged = merged.sort_values(["race_date", "race_no", "lane"]).reset_index(drop=True)
    merged["target_rank"] = 7 - merged["finish_position"]
    merged["is_win"] = (merged["finish_position"] == 1).astype(int)
    merged["is_top2"] = (merged["finish_position"] <= 2).astype(int)
    merged["is_top3"] = (merged["finish_position"] <= 3).astype(int)
    return merged


def _empty_like(columns: Iterable[str]) -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(dtype="object") for column in columns})


class HistoryCarryovers:
    def __init__(self) -> None:
        base_columns = [
            "racer_id",
            "venue",
            "lane",
            "course",
            "motor_no",
            "boat_no",
            "race_date",
            "race_no",
            "is_win",
            "is_top2",
            "is_top3",
            "finish_position",
            "start_timing",
            "exhibition_time",
        ]
        self.racer = _empty_like(base_columns)
        self.venue = _empty_like(base_columns)
        self.lane = _empty_like(base_columns)
        self.venue_lane = _empty_like(base_columns)
        self.venue_course = _empty_like(base_columns)
        self.venue_lane_overall = _empty_like(base_columns)
        self.course = _empty_like(base_columns)
        self.motor = _empty_like(base_columns)
        self.boat = _empty_like(base_columns)
        self.racer_counts: dict[int, int] = {}


def _rolling_mean(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    return series.shift(1).rolling(window, min_periods=min_periods).mean()


def _rolling_min(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    return series.shift(1).rolling(window, min_periods=min_periods).min()


def _rolling_max(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    return series.shift(1).rolling(window, min_periods=min_periods).max()


def _rolling_std(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    return series.shift(1).rolling(window, min_periods=min_periods).std()


def _apply_group_history_features(
    current_df: pd.DataFrame,
    carryover_df: pd.DataFrame,
    group_cols: list[str],
    feature_builders: dict[str, tuple[str, int, int, str]],
    max_tail: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_columns = list(
        {
            *group_cols,
            "race_date",
            "race_no",
            "is_win",
            "is_top2",
            "is_top3",
            "finish_position",
            "start_timing",
            "exhibition_time",
            "__row_id",
        }
    )
    combined = pd.concat(
        [
            carryover_df.reindex(columns=source_columns, fill_value=pd.NA),
            current_df[source_columns],
        ],
        ignore_index=True,
    )
    combined = combined.sort_values([*group_cols, "race_date", "race_no"]).reset_index(drop=True)
    grouped = combined.groupby(group_cols, group_keys=False, dropna=True)

    for feature_name, (source_col, window, min_periods, agg) in feature_builders.items():
        if agg == "mean":
            combined[feature_name] = grouped[source_col].transform(
                lambda s: _rolling_mean(s, window, min_periods)
            )
        elif agg == "min":
            combined[feature_name] = grouped[source_col].transform(
                lambda s: _rolling_min(s, window, min_periods)
            )
        elif agg == "max":
            combined[feature_name] = grouped[source_col].transform(
                lambda s: _rolling_max(s, window, min_periods)
            )
        elif agg == "std":
            combined[feature_name] = grouped[source_col].transform(
                lambda s: _rolling_std(s, window, min_periods)
            )
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unsupported aggregation: {agg}")

    current_features = (
        combined.loc[combined["__row_id"].notna(), ["__row_id", *feature_builders.keys()]]
        .copy()
        .set_index("__row_id")
        .sort_index()
    )

    new_carryover = (
        combined[
            group_cols
            + [
                "race_date",
                "race_no",
                "is_win",
                "is_top2",
                "is_top3",
                "finish_position",
                "start_timing",
                "exhibition_time",
            ]
        ]
        .groupby(group_cols, group_keys=False, dropna=True)
        .tail(max_tail)
        .reset_index(drop=True)
    )
    return current_features, new_carryover


def add_racer_history_features_streaming(base_chunk: pd.DataFrame, carryovers: HistoryCarryovers) -> pd.DataFrame:
    if base_chunk.empty:
        return base_chunk

    current = base_chunk.sort_values(["racer_id", "race_date", "race_no"]).copy()
    current["__row_id"] = range(len(current))

    current["racer_prev_count"] = (
        current["racer_id"].map(carryovers.racer_counts).fillna(0).astype("int64")
        + current.groupby("racer_id").cumcount()
    )

    racer_features, carryovers.racer = _apply_group_history_features(
        current,
        carryovers.racer,
        ["racer_id"],
        {
            "racer_prev_win_rate": ("is_win", 30, 3, "mean"),
            "racer_prev_top3_rate": ("is_top3", 30, 3, "mean"),
            "racer_prev_avg_finish": ("finish_position", 30, 3, "mean"),
            "racer_prev_avg_st": ("start_timing", 30, 3, "mean"),
            "racer_prev_avg_exhibition": ("exhibition_time", 30, 3, "mean"),
            "racer_prev_win_rate_5": ("is_win", 5, 2, "mean"),
            "racer_prev_top3_rate_5": ("is_top3", 5, 2, "mean"),
            "racer_prev_avg_finish_5": ("finish_position", 5, 2, "mean"),
            "racer_prev_avg_finish_10": ("finish_position", 10, 3, "mean"),
            "racer_prev_avg_st_5": ("start_timing", 5, 2, "mean"),
            "racer_prev_avg_st_10": ("start_timing", 10, 3, "mean"),
            "racer_prev_std_st_10": ("start_timing", 10, 3, "std"),
            "racer_prev_best_st_5": ("start_timing", 5, 2, "min"),
            "racer_prev_best_st_10": ("start_timing", 10, 3, "min"),
            "racer_prev_best_st_30": ("start_timing", 30, 3, "min"),
            "racer_prev_worst_st_5": ("start_timing", 5, 2, "max"),
            "racer_prev_worst_st_10": ("start_timing", 10, 3, "max"),
            "racer_prev_worst_st_30": ("start_timing", 30, 3, "max"),
            "racer_prev_best_finish_5": ("finish_position", 5, 2, "min"),
            "racer_prev_worst_finish_5": ("finish_position", 5, 2, "max"),
        },
        max_tail=30,
    )

    venue_features, carryovers.venue = _apply_group_history_features(
        current,
        carryovers.venue,
        ["racer_id", "venue"],
        {
            "racer_venue_prev_win_rate": ("is_win", 15, 2, "mean"),
            "racer_venue_prev_top3_rate": ("is_top3", 15, 2, "mean"),
        },
        max_tail=15,
    )

    lane_features, carryovers.lane = _apply_group_history_features(
        current,
        carryovers.lane,
        ["racer_id", "lane"],
        {
            "racer_lane_prev_win_rate": ("is_win", 15, 2, "mean"),
            "racer_lane_prev_top3_rate": ("is_top3", 15, 2, "mean"),
            "racer_lane_prev_avg_st": ("start_timing", 15, 2, "mean"),
            "racer_lane_prev_avg_finish": ("finish_position", 15, 2, "mean"),
        },
        max_tail=15,
    )

    venue_lane_features, carryovers.venue_lane = _apply_group_history_features(
        current,
        carryovers.venue_lane,
        ["racer_id", "venue", "lane"],
        {
            "racer_venue_lane_prev_top3_rate": ("is_top3", 10, 2, "mean"),
            "racer_venue_lane_prev_avg_st": ("start_timing", 10, 2, "mean"),
        },
        max_tail=10,
    )

    venue_course_current = current[current["course"].notna()].copy()
    venue_course_features = pd.DataFrame(index=current["__row_id"])
    if not venue_course_current.empty:
        computed, carryovers.venue_course = _apply_group_history_features(
            venue_course_current.sort_values(["venue", "course", "race_date", "race_no", "lane"]),
            carryovers.venue_course,
            ["venue", "course"],
            {
                "venue_course_prev_win_rate": ("is_win", 200, 30, "mean"),
                "venue_course_prev_top2_rate": ("is_top2", 200, 30, "mean"),
                "venue_course_prev_top3_rate": ("is_top3", 200, 30, "mean"),
                "venue_course_prev_avg_finish": ("finish_position", 200, 30, "mean"),
            },
            max_tail=200,
        )
        venue_course_features = computed

    venue_lane_overall_features, carryovers.venue_lane_overall = _apply_group_history_features(
        current.sort_values(["venue", "lane", "race_date", "race_no"]),
        carryovers.venue_lane_overall,
        ["venue", "lane"],
        {
            "venue_lane_prev_win_rate": ("is_win", 200, 30, "mean"),
            "venue_lane_prev_top2_rate": ("is_top2", 200, 30, "mean"),
            "venue_lane_prev_top3_rate": ("is_top3", 200, 30, "mean"),
            "venue_lane_prev_avg_finish": ("finish_position", 200, 30, "mean"),
        },
        max_tail=200,
    )

    course_current = current[current["course"].notna()].copy()
    course_features = pd.DataFrame(index=current["__row_id"])
    if not course_current.empty:
        computed, carryovers.course = _apply_group_history_features(
            course_current,
            carryovers.course,
            ["racer_id", "course"],
            {
                "racer_course_prev_top3_rate": ("is_top3", 10, 2, "mean"),
                "racer_course_prev_avg_finish": ("finish_position", 10, 2, "mean"),
            },
            max_tail=10,
        )
        course_features = computed

    motor_current = current[current["motor_no"].notna()].copy()
    motor_features = pd.DataFrame(index=current["__row_id"])
    if not motor_current.empty:
        computed, carryovers.motor = _apply_group_history_features(
            motor_current,
            carryovers.motor,
            ["motor_no"],
            {
                "motor_prev_top3_rate": ("is_top3", 40, 5, "mean"),
                "motor_prev_win_rate": ("is_win", 40, 5, "mean"),
                "motor_prev_avg_st": ("start_timing", 40, 5, "mean"),
            },
            max_tail=40,
        )
        motor_features = computed

    boat_current = current[current["boat_no"].notna()].copy()
    boat_features = pd.DataFrame(index=current["__row_id"])
    if not boat_current.empty:
        computed, carryovers.boat = _apply_group_history_features(
            boat_current,
            carryovers.boat,
            ["boat_no"],
            {
                "boat_prev_top3_rate": ("is_top3", 40, 5, "mean"),
                "boat_prev_win_rate": ("is_win", 40, 5, "mean"),
                "boat_prev_avg_st": ("start_timing", 40, 5, "mean"),
            },
            max_tail=40,
        )
        boat_features = computed

    feature_frames = [
        racer_features,
        venue_features,
        lane_features,
        venue_lane_features,
        venue_course_features,
        venue_lane_overall_features,
        course_features,
        motor_features,
        boat_features,
    ]
    merged_features = pd.concat(feature_frames, axis=1).sort_index()

    for column in merged_features.columns:
        current[column] = merged_features[column].reindex(current["__row_id"]).to_numpy()

    current["st_momentum_diff"] = current["racer_prev_avg_st_5"] - current["racer_prev_avg_st_10"]
    current["finish_momentum_diff"] = current["racer_prev_avg_finish_10"] - current["racer_prev_avg_finish_5"]
    chunk_counts = current.groupby("racer_id").size().to_dict()
    for racer_id, count in chunk_counts.items():
        carryovers.racer_counts[int(racer_id)] = carryovers.racer_counts.get(int(racer_id), 0) + int(count)
    return current.drop(columns="__row_id")


def build_training_table_streaming(
    rowdata_dir: Path,
    output_dir: Path,
    max_date: date | None = None,
) -> BuildSummary:
    output_dir = output_dir.resolve()
    temp_dir = output_dir.parent / f"{output_dir.name}_streaming_tmp_{uuid4().hex[:8]}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    entries_writer = IncrementalParquetWriter(temp_dir / "race_entries.parquet", schema=ENTRY_PARQUET_SCHEMA)
    results_writer = IncrementalParquetWriter(temp_dir / "race_results.parquet", schema=RESULT_PARQUET_SCHEMA)
    carryovers = HistoryCarryovers()
    bucket_dir = temp_dir / "base_buckets"
    bucket_dir.mkdir(parents=True, exist_ok=True)
    bucket_writers: dict[int, IncrementalParquetWriter] = {}

    entries_rows = 0
    results_rows = 0
    training_rows = 0
    min_seen_date: pd.Timestamp | None = None
    max_seen_date: pd.Timestamp | None = None

    def get_bucket_writer(bucket_id: int) -> IncrementalParquetWriter:
        writer = bucket_writers.get(bucket_id)
        if writer is None:
            writer = IncrementalParquetWriter(bucket_dir / f"base_bucket_{bucket_id:04d}.parquet")
            bucket_writers[bucket_id] = writer
        return writer

    try:
        for _, entry_paths, result_paths in iter_month_groups(rowdata_dir, max_date=max_date):
            entry_records = []
            for path in entry_paths:
                entry_records.extend(item.to_dict() for item in parse_entry_file(path))
            result_records = []
            for path in result_paths:
                result_records.extend(item.to_dict() for item in parse_result_file(path))

            entries_df = pd.DataFrame(entry_records)
            results_df = pd.DataFrame(result_records)

            if not entries_df.empty:
                entries_rows += len(entries_df)
                entries_writer.write(entries_df)
            if not results_df.empty:
                results_rows += len(results_df)
                results_writer.write(results_df)
            if entries_df.empty or results_df.empty:
                continue

            base_chunk = _build_base_chunk(entries_df, results_df)
            if base_chunk.empty:
                continue

            base_chunk["__bucket"] = (base_chunk["racer_id"].astype(int) // 100).astype(int)
            for bucket_id, bucket_frame in base_chunk.groupby("__bucket", sort=True):
                get_bucket_writer(int(bucket_id)).write(bucket_frame.drop(columns="__bucket"))
    finally:
        entries_writer.close()
        results_writer.close()
        for writer in bucket_writers.values():
            writer.close()

    history_month_dir = temp_dir / "history_months"
    history_month_dir.mkdir(parents=True, exist_ok=True)
    history_month_writers: dict[str, IncrementalParquetWriter] = {}

    def get_history_month_writer(month_key: str) -> IncrementalParquetWriter:
        writer = history_month_writers.get(month_key)
        if writer is None:
            writer = IncrementalParquetWriter(history_month_dir / f"history_{month_key}.parquet")
            history_month_writers[month_key] = writer
        return writer

    try:
        for bucket_path in sorted(bucket_dir.glob("base_bucket_*.parquet")):
            base_chunk = pd.read_parquet(bucket_path)
            enriched_chunk = add_racer_history_features_streaming(base_chunk, carryovers)
            month_keys = enriched_chunk["race_date"].dt.strftime("%Y%m")
            for month_key, month_frame in enriched_chunk.groupby(month_keys, sort=True):
                get_history_month_writer(str(month_key)).write(month_frame)
    finally:
        for writer in history_month_writers.values():
            writer.close()

    training_writer = IncrementalParquetWriter(temp_dir / "training_table.parquet")
    try:
        for history_path in sorted(history_month_dir.glob("history_*.parquet")):
            history_chunk = pd.read_parquet(history_path)
            history_chunk = add_race_relative_features(history_chunk)
            history_chunk = add_current_meet_features(history_chunk)

            training_rows += len(history_chunk)
            chunk_min = history_chunk["race_date"].min()
            chunk_max = history_chunk["race_date"].max()
            min_seen_date = chunk_min if min_seen_date is None else min(min_seen_date, chunk_min)
            max_seen_date = chunk_max if max_seen_date is None else max(max_seen_date, chunk_max)
            training_writer.write(history_chunk)
    finally:
        training_writer.close()

    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.move(str(temp_dir), str(output_dir))

    return BuildSummary(
        entries_rows=entries_rows,
        results_rows=results_rows,
        training_rows=training_rows,
        min_date=min_seen_date.strftime("%Y-%m-%d") if min_seen_date is not None else None,
        max_date=max_seen_date.strftime("%Y-%m-%d") if max_seen_date is not None else None,
        output_dir=str(output_dir),
    )


def _series_equal_with_tolerance(expected: pd.Series, actual: pd.Series, atol: float) -> bool:
    if pd.api.types.is_numeric_dtype(expected) and pd.api.types.is_numeric_dtype(actual):
        left = expected.astype("float64")
        right = actual.astype("float64")
        mask = left.isna() & right.isna()
        return bool(np.allclose(left[~mask], right[~mask], atol=atol, rtol=0.0, equal_nan=True))
    return expected.equals(actual)


def compare_processed_tables(expected_dir: Path, actual_dir: Path, atol: float = 1e-12) -> dict[str, object]:
    report: dict[str, object] = {}
    for file_name in ("race_entries.parquet", "race_results.parquet", "training_table.parquet"):
        expected = pd.read_parquet(expected_dir / file_name)
        actual = pd.read_parquet(actual_dir / file_name)

        sort_columns = ["race_id", "lane"]
        if "race_date" in expected.columns and "race_date" in actual.columns:
            sort_columns = ["race_date", "race_id", "lane"]
        expected = expected.sort_values(sort_columns).reset_index(drop=True)
        actual = actual.sort_values(sort_columns).reset_index(drop=True)

        equal = True
        mismatch_columns: list[str] = []
        if list(expected.columns) != list(actual.columns) or len(expected) != len(actual):
            equal = False
        else:
            for column in expected.columns:
                if not _series_equal_with_tolerance(expected[column], actual[column], atol=atol):
                    equal = False
                    mismatch_columns.append(column)
        report[file_name] = {
            "equal": equal,
            "expected_rows": int(len(expected)),
            "actual_rows": int(len(actual)),
            "expected_columns": list(expected.columns),
            "actual_columns": list(actual.columns),
        }
        if not equal:
            report[file_name]["mismatch_columns"] = mismatch_columns
    return report
