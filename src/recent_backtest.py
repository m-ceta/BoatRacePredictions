from __future__ import annotations

import re
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.api import load_bundle, predict_trifecta
from src.evaluation.metrics import compute_trifecta_metrics
from src.live import build_live_feature_frame, load_live_history_frame
from src.models.ranker import load_config
from src.parsers.bk_parser import (
    FULLWIDTH_SPACE,
    SECTION_CODE_RE,
    build_race_id,
    load_shift_jis_lines,
    parse_entry_file,
    parse_date_from_line,
    parse_result_file,
    parse_venue_from_line,
)
ROWDATA_FILE_RE = re.compile(r"^(?P<kind>[BK])(?P<yy>\d{2})(?P<mm>\d{2})(?P<dd>\d{2})\.TXT$", re.IGNORECASE)


RACE_HEADER_RE = re.compile(r"^\s*(?P<race_no>\d{1,2})R\b")
TRIFECTA_PAYOUT_RE = re.compile(
    r"^\s*(?:3連単|３連単)\s+(?P<trifecta>[1-6]-[1-6]-[1-6])\s+(?P<payout>\d+)\b"
)


def _parse_rowdata_file_date(path: Path) -> date | None:
    match = ROWDATA_FILE_RE.match(path.name)
    if not match:
        return None
    yy = int(match.group("yy"))
    year = 1900 + yy if yy >= 90 else 2000 + yy
    return date(year, int(match.group("mm")), int(match.group("dd")))


def _list_rowdata_dates(rowdata_dir: Path, kind: str) -> set[date]:
    dates: set[date] = set()
    for path in rowdata_dir.glob(f"{kind}*.TXT"):
        parsed = _parse_rowdata_file_date(path)
        if parsed is not None:
            dates.add(parsed)
    return dates


def _latest_available_rowdata_date(rowdata_dir: Path) -> date:
    common_dates = _list_rowdata_dates(rowdata_dir, "B") & _list_rowdata_dates(rowdata_dir, "K")
    if not common_dates:
        raise ValueError(f"Could not determine latest common B/K rowdata date in {rowdata_dir}")
    return max(common_dates)


def _load_entry_rows_for_period(
    rowdata_dir: Path,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    current = start_date
    while current <= end_date:
        path = rowdata_dir / f"B{current.strftime('%y%m%d')}.TXT"
        if path.exists():
            rows.extend(item.to_dict() for item in parse_entry_file(path))
        current += timedelta(days=1)

    frame = pd.DataFrame(rows)
    if not frame.empty and "race_date" in frame.columns:
        frame["race_date"] = pd.to_datetime(frame["race_date"], errors="coerce")
    return frame


def _load_result_rows_for_period(
    rowdata_dir: Path,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    current = start_date
    while current <= end_date:
        path = rowdata_dir / f"K{current.strftime('%y%m%d')}.TXT"
        if path.exists():
            rows.extend(item.to_dict() for item in parse_result_file(path))
        current += timedelta(days=1)

    frame = pd.DataFrame(rows)
    if not frame.empty and "race_date" in frame.columns:
        frame["race_date"] = pd.to_datetime(frame["race_date"], errors="coerce")
    return frame


def _build_recent_backtest_base_frame(entries_df: pd.DataFrame, results_df: pd.DataFrame) -> pd.DataFrame:
    if entries_df.empty or results_df.empty:
        return pd.DataFrame()

    merged = entries_df.merge(
        results_df[
            [
                "race_id",
                "lane",
                "finish_position",
                "finish_status",
                "exhibition_time",
                "course",
                "start_timing",
                "race_time",
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
    if not merged.empty:
        merged["race_date"] = pd.to_datetime(merged["race_date"], errors="coerce")
    return merged


def parse_trifecta_payouts_from_lines(lines: list[str]) -> pd.DataFrame:
    venue: str | None = None
    race_date: date | None = None
    current_race_no: int | None = None
    current_section_code: str | None = None
    rows: list[dict[str, Any]] = []

    for line in lines:
        section_code_match = SECTION_CODE_RE.match(line.strip())
        if section_code_match:
            current_section_code = section_code_match.group("section_code")

        section_venue = parse_venue_from_line(line)
        if section_venue:
            venue = section_venue

        section_date = parse_date_from_line(line)
        if section_date:
            race_date = section_date

        normalized = line.replace(FULLWIDTH_SPACE, " ")
        header_match = RACE_HEADER_RE.match(normalized)
        if header_match:
            current_race_no = int(header_match.group("race_no"))
            continue

        payout_match = TRIFECTA_PAYOUT_RE.match(normalized)
        venue_key = current_section_code or venue
        if payout_match and current_race_no is not None and venue_key is not None and race_date is not None:
            race_id = build_race_id(race_date, venue_key, current_race_no)
            rows.append(
                {
                    "race_id": race_id,
                    "race_date": pd.Timestamp(race_date),
                    "venue": venue or str(venue_key),
                    "race_no": int(current_race_no),
                    "actual_trifecta": payout_match.group("trifecta"),
                    "trifecta_payout": float(payout_match.group("payout")),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=["race_id", "race_date", "venue", "race_no", "actual_trifecta", "trifecta_payout"]
        )
    return pd.DataFrame(rows).drop_duplicates(subset=["race_id"], keep="first").reset_index(drop=True)


def parse_trifecta_payouts_from_file(path: Path) -> pd.DataFrame:
    return parse_trifecta_payouts_from_lines(load_shift_jis_lines(path))


def _load_recent_payouts(
    rowdata_dir: Path,
    start_date: date,
    end_date: date,
) -> tuple[pd.DataFrame, list[str]]:
    rows: list[pd.DataFrame] = []
    missing_files: list[str] = []
    current = start_date
    while current <= end_date:
        path = rowdata_dir / f"K{current.strftime('%y%m%d')}.TXT"
        if path.exists():
            payout_df = parse_trifecta_payouts_from_file(path)
            if not payout_df.empty:
                rows.append(payout_df)
        else:
            missing_files.append(str(path))
        current += timedelta(days=1)

    if not rows:
        return (
            pd.DataFrame(
                columns=["race_id", "race_date", "venue", "race_no", "actual_trifecta", "trifecta_payout"]
            ),
            missing_files,
        )
    return pd.concat(rows, ignore_index=True), missing_files


def _serialize_frame(frame: pd.DataFrame) -> list[dict[str, Any]]:
    serializable = frame.copy()
    for column in serializable.columns:
        if pd.api.types.is_datetime64_any_dtype(serializable[column]):
            serializable[column] = serializable[column].dt.strftime("%Y-%m-%d")
        elif pd.api.types.is_bool_dtype(serializable[column]):
            serializable[column] = serializable[column].astype(bool)
        elif pd.api.types.is_integer_dtype(serializable[column]):
            serializable[column] = serializable[column].astype(int)
        elif pd.api.types.is_float_dtype(serializable[column]):
            serializable[column] = serializable[column].astype(float)
    return serializable.to_dict(orient="records")


BACKTEST_RESULT_ONLY_COLUMNS = {
    "finish_position",
    "finish_status",
    "start_timing",
    "course",
    "exhibition_time",
    "race_time",
    "weather",
    "wind_direction",
    "wind_speed_m",
    "wave_cm",
    "winning_style",
    "target_rank",
    "is_win",
    "is_top2",
    "is_top3",
}


def prepare_recent_backtest_entry_frame(evaluation_rows: pd.DataFrame) -> pd.DataFrame:
    entry_frame = evaluation_rows.copy()
    for column in BACKTEST_RESULT_ONLY_COLUMNS:
        if column in entry_frame.columns:
            entry_frame = entry_frame.drop(columns=column)
    current_meet_feature_columns = [
        column
        for column in entry_frame.columns
        if column.startswith("current_meet_") and column != "current_meet_results"
    ]
    if current_meet_feature_columns:
        entry_frame = entry_frame.drop(columns=current_meet_feature_columns)
    return entry_frame


def _build_history_append_frame(evaluation_rows: pd.DataFrame) -> pd.DataFrame:
    history_columns = [
        "race_date",
        "race_no",
        "racer_id",
        "venue",
        "lane",
        "course",
        "motor_no",
        "boat_no",
        "finish_position",
        "start_timing",
        "exhibition_time",
    ]
    available = [column for column in history_columns if column in evaluation_rows.columns]
    history = evaluation_rows[available].copy()
    if history.empty:
        return history
    history["race_date"] = pd.to_datetime(history["race_date"], errors="coerce")
    finish = pd.to_numeric(history["finish_position"], errors="coerce")
    history["is_win"] = (finish == 1).astype(int)
    history["is_top3"] = (finish.fillna(999).astype(int) <= 3).astype(int)
    return history


def build_recent_backtest_prediction_frame(
    bundle,
    evaluation_rows: pd.DataFrame,
    start_date: date,
) -> pd.DataFrame:
    if evaluation_rows.empty:
        return evaluation_rows.copy()

    entry_frame = prepare_recent_backtest_entry_frame(evaluation_rows)
    history_df = load_live_history_frame(bundle.config, start_date)
    built_frames: list[pd.DataFrame] = []

    eval_columns = [
        column
        for column in ("race_id", "lane", "finish_position")
        if column in evaluation_rows.columns
    ]
    evaluation_frame = evaluation_rows[eval_columns].drop_duplicates(subset=["race_id", "lane"]).copy()

    for race_day, race_day_entries in entry_frame.groupby("race_date", sort=True):
        built = build_live_feature_frame(race_day_entries.copy(), history_df, bundle.feature_columns)
        built_frames.append(built)
        day_results = evaluation_rows[evaluation_rows["race_date"] == race_day].copy()
        history_append = _build_history_append_frame(day_results)
        if not history_append.empty:
            history_df = pd.concat([history_df, history_append], ignore_index=True)

    if not built_frames:
        return pd.DataFrame(columns=["race_id", *bundle.feature_columns, *eval_columns])

    prediction_frame = pd.concat(built_frames, ignore_index=True)
    prediction_frame = prediction_frame.merge(
        evaluation_frame,
        on=["race_id", "lane"],
        how="left",
        validate="one_to_one",
    )
    return prediction_frame


def evaluate_recent_week_predictions(
    config_path: str | Path = Path("configs/train.yaml"),
    rowdata_dir: str | Path = Path("rowdata"),
    days: int = 7,
    stake_per_ticket: int = 100,
    top_k: int = 1,
) -> dict[str, Any]:
    if days <= 0:
        raise ValueError("days must be positive")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if stake_per_ticket <= 0:
        raise ValueError("stake_per_ticket must be positive")

    config_path = Path(config_path)
    rowdata_dir = Path(rowdata_dir)
    config = load_config(config_path)
    latest_date = _latest_available_rowdata_date(rowdata_dir)
    start_date = latest_date - timedelta(days=days - 1)
    entries_week = _load_entry_rows_for_period(rowdata_dir, start_date, latest_date)
    results_week = _load_result_rows_for_period(rowdata_dir, start_date, latest_date)
    evaluation_week = _build_recent_backtest_base_frame(entries_week, results_week)
    if evaluation_week.empty:
        raise ValueError(f"No recent rowdata races found between {start_date} and {latest_date}")

    payout_df, missing_payout_files = _load_recent_payouts(rowdata_dir, start_date, latest_date)
    if payout_df.empty:
        raise ValueError(f"No trifecta payout data found in {rowdata_dir} between {start_date} and {latest_date}")

    common_race_ids = sorted(set(evaluation_week["race_id"].astype(str)) & set(payout_df["race_id"].astype(str)))
    if not common_race_ids:
        raise ValueError("No overlapping races found between rowdata entries/results and payout files")

    evaluation_week = evaluation_week[evaluation_week["race_id"].astype(str).isin(common_race_ids)].copy()
    payout_df = payout_df[payout_df["race_id"].astype(str).isin(common_race_ids)].copy()

    bundle = load_bundle(config_path)
    prediction_frame = build_recent_backtest_prediction_frame(bundle, evaluation_week, start_date)
    trifecta_all = predict_trifecta(bundle, prediction_frame, top_n=None, use_v2=True)
    trifecta_all = trifecta_all[trifecta_all["race_id"].astype(str).isin(common_race_ids)].copy()

    top_predictions = (
        trifecta_all.sort_values(["race_id", "probability"], ascending=[True, False])
        .groupby("race_id", sort=False)
        .head(top_k)
        .reset_index(drop=True)
    )
    top_predictions["prediction_rank"] = top_predictions.groupby("race_id", sort=False).cumcount() + 1

    race_meta = (
        evaluation_week[["race_id", "race_date", "venue", "race_no"]]
        .drop_duplicates(subset=["race_id"])
        .reset_index(drop=True)
    )

    ticket_details = top_predictions.merge(
        payout_df[["race_id", "actual_trifecta", "trifecta_payout"]],
        on="race_id",
        how="left",
        validate="many_to_one",
    ).merge(
        race_meta,
        on="race_id",
        how="left",
        validate="many_to_one",
    )
    ticket_details["hit"] = ticket_details["trifecta"].astype(str) == ticket_details["actual_trifecta"].astype(str)
    ticket_details["stake_amount"] = float(stake_per_ticket)
    ticket_details["return_amount"] = np.where(
        ticket_details["hit"],
        ticket_details["trifecta_payout"].astype(float) * float(stake_per_ticket) / 100.0,
        0.0,
    )
    ticket_details = ticket_details[
        [
            "race_date",
            "venue",
            "race_no",
            "race_id",
            "prediction_rank",
            "trifecta",
            "probability",
            "actual_trifecta",
            "trifecta_payout",
            "hit",
            "stake_amount",
            "return_amount",
        ]
    ].sort_values(["race_date", "race_no", "prediction_rank"])

    candidate_text = (
        ticket_details.sort_values(["race_id", "prediction_rank"])
        .groupby("race_id", sort=False)["trifecta"]
        .apply(lambda values: " / ".join(values.astype(str).tolist()))
        .rename("predicted_tickets")
        .reset_index()
    )
    race_summary = (
        ticket_details.groupby("race_id", sort=False)
        .agg(
            race_date=("race_date", "first"),
            venue=("venue", "first"),
            race_no=("race_no", "first"),
            tickets_bought=("prediction_rank", "count"),
            race_hit=("hit", "max"),
            total_stake=("stake_amount", "sum"),
            total_return=("return_amount", "sum"),
            actual_trifecta=("actual_trifecta", "first"),
            actual_payout=("trifecta_payout", "first"),
        )
        .reset_index()
        .merge(candidate_text, on="race_id", how="left", validate="one_to_one")
    )
    race_summary["recovery_rate"] = np.where(
        race_summary["total_stake"] > 0,
        race_summary["total_return"] / race_summary["total_stake"],
        0.0,
    )
    race_summary = race_summary.sort_values(["race_date", "race_no"]).reset_index(drop=True)

    daily_summary = (
        race_summary.groupby("race_date", sort=False)
        .agg(
            race_count=("race_id", "count"),
            hit_races=("race_hit", "sum"),
            total_stake=("total_stake", "sum"),
            total_return=("total_return", "sum"),
        )
        .reset_index()
    )
    daily_summary["hit_rate"] = np.where(
        daily_summary["race_count"] > 0,
        daily_summary["hit_races"] / daily_summary["race_count"],
        0.0,
    )
    daily_summary["recovery_rate"] = np.where(
        daily_summary["total_stake"] > 0,
        daily_summary["total_return"] / daily_summary["total_stake"],
        0.0,
    )

    accuracy_metrics = compute_trifecta_metrics(trifecta_all, probability_col="probability")

    race_count = int(race_summary["race_id"].nunique())
    ticket_count = int(len(ticket_details))
    hit_races = int(race_summary["race_hit"].sum())
    hit_tickets = int(ticket_details["hit"].sum())
    total_stake = float(ticket_details["stake_amount"].sum())
    total_return = float(ticket_details["return_amount"].sum())

    summary = {
        "latest_available_date": latest_date.isoformat(),
        "start_date": start_date.isoformat(),
        "end_date": latest_date.isoformat(),
        "days_requested": int(days),
        "available_race_dates": [value.strftime("%Y-%m-%d") for value in sorted(race_summary["race_date"].dt.date.unique())],
        "race_count": race_count,
        "ticket_count": ticket_count,
        "top_k": int(top_k),
        "stake_per_ticket": int(stake_per_ticket),
        "total_stake": total_stake,
        "hit_races": hit_races,
        "hit_tickets": hit_tickets,
        "race_hit_rate": hit_races / race_count if race_count else 0.0,
        "ticket_hit_rate": hit_tickets / ticket_count if ticket_count else 0.0,
        "total_return": total_return,
        "recovery_rate": total_return / total_stake if total_stake else 0.0,
        "top1_hit_rate": float(accuracy_metrics.get("top1_hit_rate", 0.0)),
        "top3_hit_rate": float(accuracy_metrics.get("top3_hit_rate", 0.0)),
        "top5_hit_rate": float(accuracy_metrics.get("top5_hit_rate", 0.0)),
        "missing_payout_files": missing_payout_files,
    }

    return {
        "summary": summary,
        "daily_summary": _serialize_frame(daily_summary),
        "race_summary": _serialize_frame(race_summary),
        "ticket_details": _serialize_frame(ticket_details),
    }
