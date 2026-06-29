from __future__ import annotations

import re
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.api import load_bundle, predict_trifecta
from src.evaluation.metrics import compute_trifecta_metrics
from src.models.ranker import load_config
from src.parsers.bk_parser import (
    FULLWIDTH_SPACE,
    SECTION_CODE_RE,
    build_race_id,
    load_shift_jis_lines,
    parse_date_from_line,
    parse_venue_from_line,
)


RACE_HEADER_RE = re.compile(r"^\s*(?P<race_no>\d{1,2})R\b")
TRIFECTA_PAYOUT_RE = re.compile(
    r"^\s*(?:3連単|３連単)\s+(?P<trifecta>[1-6]-[1-6]-[1-6])\s+(?P<payout>\d+)\b"
)


def _latest_available_race_date(training_table_path: Path) -> date:
    race_dates = pd.read_parquet(training_table_path, columns=["race_date"])
    latest = pd.to_datetime(race_dates["race_date"], errors="coerce").max()
    if pd.isna(latest):
        raise ValueError(f"Could not determine latest race_date from {training_table_path}")
    return latest.date()


def _load_training_rows_for_period(
    training_table_path: Path,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    filters = [
        ("race_date", ">=", pd.Timestamp(start_date)),
        ("race_date", "<=", pd.Timestamp(end_date)),
    ]
    try:
        frame = pd.read_parquet(training_table_path, filters=filters)
    except Exception:
        frame = pd.read_parquet(training_table_path)
        race_dates = pd.to_datetime(frame["race_date"], errors="coerce").dt.date
        frame = frame[(race_dates >= start_date) & (race_dates <= end_date)].copy()
    frame["race_date"] = pd.to_datetime(frame["race_date"], errors="coerce")
    return frame


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
    training_table_path = Path(config["data"]["training_table"])

    latest_date = _latest_available_race_date(training_table_path)
    start_date = latest_date - timedelta(days=days - 1)
    training_week = _load_training_rows_for_period(training_table_path, start_date, latest_date)
    if training_week.empty:
        raise ValueError(f"No training rows found between {start_date} and {latest_date}")

    payout_df, missing_payout_files = _load_recent_payouts(rowdata_dir, start_date, latest_date)
    if payout_df.empty:
        raise ValueError(f"No trifecta payout data found in {rowdata_dir} between {start_date} and {latest_date}")

    common_race_ids = sorted(set(training_week["race_id"].astype(str)) & set(payout_df["race_id"].astype(str)))
    if not common_race_ids:
        raise ValueError("No overlapping races found between training_table and payout files")

    training_week = training_week[training_week["race_id"].astype(str).isin(common_race_ids)].copy()
    payout_df = payout_df[payout_df["race_id"].astype(str).isin(common_race_ids)].copy()

    bundle = load_bundle(config_path)
    trifecta_all = predict_trifecta(bundle, training_week, top_n=None, use_v2=True)
    trifecta_all = trifecta_all[trifecta_all["race_id"].astype(str).isin(common_race_ids)].copy()

    top_predictions = (
        trifecta_all.sort_values(["race_id", "probability"], ascending=[True, False])
        .groupby("race_id", sort=False)
        .head(top_k)
        .reset_index(drop=True)
    )
    top_predictions["prediction_rank"] = top_predictions.groupby("race_id", sort=False).cumcount() + 1

    race_meta = (
        training_week[["race_id", "race_date", "venue", "race_no"]]
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
