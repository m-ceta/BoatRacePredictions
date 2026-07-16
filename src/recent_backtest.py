from __future__ import annotations

import re
from datetime import date, timedelta
from math import ceil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.api import load_bundle, predict_trifecta
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

VENUE_NAMES = {
    "01": "桐生",
    "02": "戸田",
    "03": "江戸川",
    "04": "平和島",
    "05": "多摩川",
    "06": "浜名湖",
    "07": "蒲郡",
    "08": "常滑",
    "09": "津",
    "10": "三国",
    "11": "びわこ",
    "12": "住之江",
    "13": "尼崎",
    "14": "鳴門",
    "15": "丸亀",
    "16": "児島",
    "17": "宮島",
    "18": "徳山",
    "19": "下関",
    "20": "若松",
    "21": "芦屋",
    "22": "福岡",
    "23": "唐津",
    "24": "大村",
}


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


def _resolve_backtest_period(
    rowdata_dir: Path,
    days: int,
    start_date: date | None = None,
    end_date: date | None = None,
) -> tuple[date, date]:
    if start_date is not None and end_date is not None:
        if start_date > end_date:
            raise ValueError("start_date must be on or before end_date")
        return start_date, end_date
    if start_date is not None:
        return start_date, start_date + timedelta(days=days - 1)
    latest_date = _latest_available_rowdata_date(rowdata_dir) if end_date is None else end_date
    return latest_date - timedelta(days=days - 1), latest_date


def _normalize_race_type(leg_type: str | None, race_title: str | None = None) -> str:
    text = " ".join(part for part in [str(leg_type or ""), str(race_title or "")] if part).strip()
    if not text:
        return "不明"
    if "準優勝" in text:
        return "準優勝戦"
    if "優勝戦" in text:
        return "優勝戦"
    if (
        "ドリーム" in text
        or "選抜" in text
        or "特選" in text
        or "特賞" in text
        or clean_text_label(text).endswith("特")
        or clean_text_label(text).endswith("選")
    ):
        return "特賞・選抜"
    if "予選" in text:
        return "予選"
    return "一般戦"


def clean_text_label(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


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
    history["is_top2"] = (finish.fillna(999).astype(int) <= 2).astype(int)
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


def compute_ticket_rank_hit_rates(
    ticket_details: pd.DataFrame,
    race_count: int,
    top_ks: tuple[int, ...] = (1, 3, 5),
) -> dict[str, float]:
    if race_count <= 0 or ticket_details.empty:
        return {f"top{top_k}_hit_rate": 0.0 for top_k in top_ks}

    metrics: dict[str, float] = {}
    for top_k in top_ks:
        covered_races = int(
            ticket_details.loc[ticket_details["prediction_rank"] <= int(top_k)]
            .groupby("race_id", sort=False)["hit"]
            .max()
            .sum()
        )
        metrics[f"top{top_k}_hit_rate"] = covered_races / race_count
    return metrics


def evaluate_recent_week_predictions(
    config_path: str | Path = Path("configs/train.yaml"),
    rowdata_dir: str | Path = Path("rowdata"),
    days: int = 7,
    stake_per_ticket: int = 100,
    top_k: int = 1,
    start_date: date | None = None,
    end_date: date | None = None,
) -> dict[str, Any]:
    if days <= 0:
        raise ValueError("days must be positive")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if stake_per_ticket <= 0:
        raise ValueError("stake_per_ticket must be positive")

    config_path = Path(config_path)
    rowdata_dir = Path(rowdata_dir)
    load_config(config_path)
    start_date, end_date = _resolve_backtest_period(rowdata_dir, days=days, start_date=start_date, end_date=end_date)
    latest_date = end_date
    entries_week = _load_entry_rows_for_period(rowdata_dir, start_date, end_date)
    results_week = _load_result_rows_for_period(rowdata_dir, start_date, end_date)
    evaluation_week = _build_recent_backtest_base_frame(entries_week, results_week)
    if evaluation_week.empty:
        raise ValueError(f"No recent rowdata races found between {start_date} and {end_date}")

    payout_df, missing_payout_files = _load_recent_payouts(rowdata_dir, start_date, end_date)
    if payout_df.empty:
        raise ValueError(f"No trifecta payout data found in {rowdata_dir} between {start_date} and {end_date}")

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
        evaluation_week[["race_id", "race_date", "venue", "race_no", "race_title", "leg_type"]]
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
            "race_title",
            "leg_type",
            "race_id",
            "prediction_rank",
            "trifecta",
            "probability",
            "race_upset_score",
            "race_upset_label",
            "trifecta_darkhorse_score",
            "is_darkhorse_candidate",
            "ticket_priority_score",
            "ticket_hint",
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
            race_title=("race_title", "first"),
            leg_type=("leg_type", "first"),
            tickets_bought=("prediction_rank", "count"),
            race_hit=("hit", "max"),
            race_upset_score=("race_upset_score", "first"),
            race_upset_label=("race_upset_label", "first"),
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

    race_count = int(race_summary["race_id"].nunique())
    ticket_count = int(len(ticket_details))
    hit_races = int(race_summary["race_hit"].sum())
    hit_tickets = int(ticket_details["hit"].sum())
    total_stake = float(ticket_details["stake_amount"].sum())
    total_return = float(ticket_details["return_amount"].sum())
    rank_hit_rates = compute_ticket_rank_hit_rates(ticket_details, race_count)

    summary = {
        "latest_available_date": latest_date.isoformat(),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
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
        "top1_hit_rate": float(rank_hit_rates.get("top1_hit_rate", 0.0)),
        "top3_hit_rate": float(rank_hit_rates.get("top3_hit_rate", 0.0)),
        "top5_hit_rate": float(rank_hit_rates.get("top5_hit_rate", 0.0)),
        "missing_payout_files": missing_payout_files,
    }

    return {
        "summary": summary,
        "daily_summary": _serialize_frame(daily_summary),
        "race_summary": _serialize_frame(race_summary),
        "ticket_details": _serialize_frame(ticket_details),
    }


def _summarize_hit_frame(frame: pd.DataFrame, group_column: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[group_column, "レース数", "的中レース数", "的中率", "投資額", "払戻額", "回収率"])
    summary = (
        frame.groupby(group_column, sort=False)
        .agg(
            レース数=("race_id", "count"),
            的中レース数=("race_hit", "sum"),
            投資額=("total_stake", "sum"),
            払戻額=("total_return", "sum"),
        )
        .reset_index()
    )
    summary["的中率"] = np.where(summary["レース数"] > 0, summary["的中レース数"] / summary["レース数"], 0.0)
    summary["回収率"] = np.where(summary["投資額"] > 0, summary["払戻額"] / summary["投資額"], 0.0)
    return summary.sort_values([group_column]).reset_index(drop=True)


def _format_rate_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    formatted = frame.copy()
    for column in columns:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(lambda value: f"{float(value) * 100:.2f}%")
    return formatted


def _build_japanese_report_frames(report: dict[str, Any]) -> dict[str, pd.DataFrame]:
    race_summary = pd.DataFrame(report.get("race_summary", [])).copy()
    ticket_details = pd.DataFrame(report.get("ticket_details", [])).copy()
    daily_summary = pd.DataFrame(report.get("daily_summary", [])).copy()

    for frame in (race_summary, ticket_details, daily_summary):
        if "race_date" in frame.columns:
            frame["race_date"] = pd.to_datetime(frame["race_date"], errors="coerce")

    if not race_summary.empty:
        race_summary["venue_code"] = race_summary["race_id"].astype(str).str.split("_").str[1]
        race_summary["レース場"] = race_summary["venue_code"].map(VENUE_NAMES).fillna(race_summary["venue_code"])
        race_summary["レースNo"] = pd.to_numeric(race_summary["race_no"], errors="coerce").astype("Int64")
        race_summary["レース種別"] = race_summary.apply(
            lambda row: _normalize_race_type(row.get("leg_type"), row.get("race_title")),
            axis=1,
        )
        race_summary["日付"] = race_summary["race_date"].dt.strftime("%Y-%m-%d")
        race_summary["的中"] = race_summary["race_hit"].map(lambda value: "的中" if bool(value) else "不的中")

    if not ticket_details.empty:
        ticket_details["venue_code"] = ticket_details["race_id"].astype(str).str.split("_").str[1]
        ticket_details["レース場"] = ticket_details["venue_code"].map(VENUE_NAMES).fillna(ticket_details["venue_code"])
        ticket_details["レースNo"] = pd.to_numeric(ticket_details["race_no"], errors="coerce").astype("Int64")
        ticket_details["日付"] = ticket_details["race_date"].dt.strftime("%Y-%m-%d")
        ticket_details["的中"] = ticket_details["hit"].map(lambda value: "的中" if bool(value) else "不的中")

    if not daily_summary.empty:
        daily_summary["日付"] = daily_summary["race_date"].dt.strftime("%Y-%m-%d")

    venue_summary = _summarize_hit_frame(race_summary, "レース場")
    race_no_summary = _summarize_hit_frame(race_summary, "レースNo")
    race_type_summary = _summarize_hit_frame(race_summary, "レース種別")

    race_summary_ja = race_summary[
        [
            "日付",
            "レース場",
            "レースNo",
            "レース種別",
            "predicted_tickets",
            "race_upset_score",
            "race_upset_label",
            "actual_trifecta",
            "actual_payout",
            "的中",
            "total_stake",
            "total_return",
            "recovery_rate",
        ]
    ].rename(
        columns={
            "predicted_tickets": "予想買い目",
            "actual_trifecta": "結果3連単",
            "actual_payout": "払戻",
            "total_stake": "投資額",
            "total_return": "払戻額",
            "recovery_rate": "回収率",
        }
    ) if not race_summary.empty else pd.DataFrame()

    ticket_details_ja = ticket_details[
        [
            "日付",
            "レース場",
            "レースNo",
            "prediction_rank",
            "trifecta",
            "probability",
            "race_upset_score",
            "race_upset_label",
            "trifecta_darkhorse_score",
            "is_darkhorse_candidate",
            "ticket_priority_score",
            "ticket_hint",
            "actual_trifecta",
            "trifecta_payout",
            "的中",
            "stake_amount",
            "return_amount",
        ]
    ].rename(
        columns={
            "prediction_rank": "予想順位",
            "trifecta": "予想3連単",
            "probability": "予想確率",
            "actual_trifecta": "結果3連単",
            "trifecta_payout": "払戻",
            "stake_amount": "投資額",
            "return_amount": "払戻額",
        }
    ) if not ticket_details.empty else pd.DataFrame()

    daily_summary_ja = daily_summary[
        ["日付", "race_count", "hit_races", "hit_rate", "total_stake", "total_return", "recovery_rate"]
    ].rename(
        columns={
            "race_count": "レース数",
            "hit_races": "的中レース数",
            "hit_rate": "的中率",
            "total_stake": "投資額",
            "total_return": "払戻額",
            "recovery_rate": "回収率",
        }
    ) if not daily_summary.empty else pd.DataFrame()

    return {
        "daily_summary": daily_summary_ja,
        "race_summary": race_summary_ja,
        "ticket_details": ticket_details_ja,
        "venue_summary": venue_summary,
        "race_no_summary": race_no_summary,
        "race_type_summary": race_type_summary,
    }


def _pick_japanese_font_family() -> str:
    from matplotlib import font_manager

    for family in ("Yu Gothic", "Meiryo", "MS Gothic", "Noto Sans CJK JP", "IPAexGothic"):
        try:
            path = font_manager.findfont(family, fallback_to_default=False)
        except Exception:
            path = ""
        if path:
            return family
    return "sans-serif"


def _render_table_pages(pdf, title: str, frame: pd.DataFrame, rows_per_page: int = 24) -> None:
    import matplotlib.pyplot as plt

    if frame.empty:
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis("off")
        ax.set_title(title, loc="left", fontsize=14, fontweight="bold")
        ax.text(0.01, 0.9, "データはありません。", fontsize=11, va="top")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        return

    page_count = ceil(len(frame) / rows_per_page)
    for page in range(page_count):
        chunk = frame.iloc[page * rows_per_page : (page + 1) * rows_per_page].copy()
        fig, ax = plt.subplots(figsize=(16.54, 11.69))
        ax.axis("off")
        ax.set_title(f"{title} ({page + 1}/{page_count})", loc="left", fontsize=14, fontweight="bold")
        table = ax.table(
            cellText=chunk.astype(str).values.tolist(),
            colLabels=chunk.columns.tolist(),
            loc="upper left",
            cellLoc="left",
            colLoc="left",
            bbox=[0.0, 0.0, 1.0, 0.92],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.2)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def _build_trend_lines(summary_by_group: pd.DataFrame, label: str) -> list[str]:
    if summary_by_group.empty:
        return [f"{label}: データなし"]
    best_hit = summary_by_group.sort_values("的中率", ascending=False).iloc[0]
    worst_hit = summary_by_group.sort_values("的中率", ascending=True).iloc[0]
    best_recovery = summary_by_group.sort_values("回収率", ascending=False).iloc[0]
    return [
        f"{label} 的中率上位: {best_hit.iloc[0]} ({best_hit['的中率'] * 100:.2f}%)",
        f"{label} 的中率下位: {worst_hit.iloc[0]} ({worst_hit['的中率'] * 100:.2f}%)",
        f"{label} 回収率上位: {best_recovery.iloc[0]} ({best_recovery['回収率'] * 100:.2f}%)",
    ]


def export_backtest_report_artifacts(
    report: dict[str, Any],
    output_dir: str | Path,
    base_name: str,
) -> dict[str, str]:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    plt.rcParams["font.family"] = _pick_japanese_font_family()
    japanese_frames = _build_japanese_report_frames(report)
    summary = report["summary"]

    daily_csv = output_path / f"{base_name}_日別集計.csv"
    race_csv = output_path / f"{base_name}_レース別結果.csv"
    ticket_csv = output_path / f"{base_name}_買い目詳細.csv"
    pdf_path = output_path / f"{base_name}_集計レポート.pdf"

    japanese_frames["daily_summary"].to_csv(daily_csv, index=False, encoding="utf-8-sig")
    japanese_frames["race_summary"].to_csv(race_csv, index=False, encoding="utf-8-sig")
    japanese_frames["ticket_details"].to_csv(ticket_csv, index=False, encoding="utf-8-sig")

    venue_summary = _format_rate_columns(japanese_frames["venue_summary"], ["的中率", "回収率"])
    race_no_summary = _format_rate_columns(japanese_frames["race_no_summary"], ["的中率", "回収率"])
    race_type_summary = _format_rate_columns(japanese_frames["race_type_summary"], ["的中率", "回収率"])
    daily_summary = _format_rate_columns(japanese_frames["daily_summary"], ["的中率", "回収率"])

    trend_lines = (
        _build_trend_lines(japanese_frames["venue_summary"], "レース場")
        + _build_trend_lines(japanese_frames["race_no_summary"], "レースNo")
        + _build_trend_lines(japanese_frames["race_type_summary"], "レース種別")
    )

    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis("off")
        ax.set_title("3連単バックテスト集計レポート", loc="left", fontsize=18, fontweight="bold")
        lines = [
            f"対象期間: {summary['start_date']} ～ {summary['end_date']}",
            f"買い方: 1レース {summary['top_k']} 点買い / 1点 {summary['stake_per_ticket']}円",
            f"レース数: {summary['race_count']}",
            f"購入点数: {summary['ticket_count']}",
            f"的中レース数: {summary['hit_races']}",
            f"的中率: {summary['race_hit_rate'] * 100:.2f}%",
            f"投資額: {summary['total_stake']:,.0f}円",
            f"払戻額: {summary['total_return']:,.0f}円",
            f"回収率: {summary['recovery_rate'] * 100:.2f}%",
            "",
            "傾向メモ:",
            *trend_lines,
        ]
        ax.text(0.01, 0.95, "\n".join(lines), va="top", fontsize=12)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        _render_table_pages(pdf, "日別集計", daily_summary, rows_per_page=28)
        _render_table_pages(pdf, "レース場別集計", venue_summary, rows_per_page=28)
        _render_table_pages(pdf, "レースNo別集計", race_no_summary, rows_per_page=28)
        _render_table_pages(pdf, "レース種別別集計", race_type_summary, rows_per_page=28)
        _render_table_pages(pdf, "レース別結果", japanese_frames["race_summary"], rows_per_page=28)

    return {
        "daily_csv": str(daily_csv),
        "race_csv": str(race_csv),
        "ticket_csv": str(ticket_csv),
        "pdf": str(pdf_path),
    }
