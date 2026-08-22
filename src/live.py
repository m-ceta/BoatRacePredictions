from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import pandas as pd
from bs4 import BeautifulSoup

try:
    import lhafile
except ImportError:  # pragma: no cover - optional runtime dependency
    lhafile = None

from src.features.builder import (
    DECISION_STYLE_KEYS,
    add_current_meet_features,
    add_decision_style_flag_columns,
    add_flow_probability_features,
    add_race_relative_features,
)
from src.features.scenario import scenario_description, scenario_display_name
from src.models.ranker import attach_darkhorse_odds_context, predict_race_order, predict_trifecta_probabilities
from src.odds.expected_value import (
    BUY_EXPECTED_VALUE_THRESHOLD,
    BUY_MIN_ODDS,
    RECOMMENDED_BET_TOP_N,
)
from src.in_course_probability import attach_in_course_probability_columns
from src.top3_hit_probability import attach_top3_hit_probability_columns
from src.parsers.bk_parser import parse_entry_text
from src.top12_confidence import (
    apply_top12_probability_adjustment_table,
    attach_boat_top1_confidence_columns,
    attach_top12_confidence_columns,
    attach_top3_confidence_columns,
)
from src.today_schedule import (
    choose_default_today_race_no as _choose_default_today_race_no,
    choose_default_today_venue as _choose_default_today_venue,
    current_jst_date,
    fetch_daily_race_schedule as _fetch_daily_race_schedule,
    parse_program_race_schedule as _parse_program_race_schedule,
)


VENUE_CODE_MAP = {
    "01": "01",
    "02": "02",
    "03": "03",
    "04": "04",
    "05": "05",
    "06": "06",
    "07": "07",
    "08": "08",
    "09": "09",
    "10": "10",
    "11": "11",
    "12": "12",
    "13": "13",
    "14": "14",
    "15": "15",
    "16": "16",
    "17": "17",
    "18": "18",
    "19": "19",
    "20": "20",
    "21": "21",
    "22": "22",
    "23": "23",
    "24": "24",
    "桐生": "01",
    "戸田": "02",
    "江戸川": "03",
    "平和島": "04",
    "多摩川": "05",
    "浜名湖": "06",
    "蒲郡": "07",
    "常滑": "08",
    "津": "09",
    "三国": "10",
    "びわこ": "11",
    "住之江": "12",
    "尼崎": "13",
    "鳴門": "14",
    "丸亀": "15",
    "児島": "16",
    "宮島": "17",
    "徳山": "18",
    "下関": "19",
    "若松": "20",
    "芦屋": "21",
    "福岡": "22",
    "唐津": "23",
    "大村": "24",
}

_HTML_TITLE_RE = re.compile(r"<title>(.*?)</title>", re.IGNORECASE | re.DOTALL)
LIVE_JST = ZoneInfo("Asia/Tokyo")


@dataclass(slots=True)
class TodayRacePrediction:
    venue: str
    venue_code: str
    race_no: int
    race_date: date
    ranking: pd.DataFrame
    trifecta: pd.DataFrame
    boatrace_title: str | None
    odds: pd.DataFrame | None = None
    race_scenario_id: str = "S6_MIXED_OTHER"
    race_scenario_name: str = "混合・その他型"
    race_scenario_description: str = "勝ち筋や上位構成が一つの典型に寄り切らない決着型です。"
    confidence_score: float = 0.0
    race_upset_score: float = 0.0
    race_upset_label: str = "堅め"
    confidence_label: str = "低"
    buy_candidates: pd.DataFrame | None = None

    @property
    def text(self) -> str:
        top = self.trifecta.iloc[0]
        top_trifecta = str(top["trifecta"])
        first, second, third = [int(x) for x in top_trifecta.split("-")]
        first_prob = self._position_probability(first, 0)
        second_prob = self._position_probability(second, 1)
        third_prob = self._position_probability(third, 2)
        lines = [
            f"予想着順: 1着: {first}, 2着: {second}, 3着: {third}\n"
            f"予想確立：1着: {format_percent(first_prob)}, 2着: {format_percent(second_prob)}, 3着: {format_percent(third_prob)}",
            f"予想信頼度: {self.confidence_label} ({format_percent(self.confidence_score)})",
            f"3連単予想本命: {top_trifecta}",
            f"3連単予想確率: {format_percent(float(top['probability']))}",
        ]

        lines.insert(2, f"レース荒れ度: {self.race_upset_label} ({format_percent(self.race_upset_score)})")

        lines.insert(2, f"決着パターン: {self.race_scenario_name} ({self.race_scenario_id})")
        lines.insert(3, f"決着イメージ: {self.race_scenario_description}")

        if self.odds is not None and not self.odds.empty:
            top_odds = self.odds.iloc[0]
            lines.extend(
                [
                    f"オッズ評価上位: {top_odds['trifecta']}",
                    f"現在オッズ: {float(top_odds['odds']):.1f}倍",
                    f"補正後確率: {format_percent(float(top_odds.get('expected_value_probability', top_odds.get('probability', 0.0)) or 0.0))}",
                    f"期待値: {float(top_odds['expected_value']):.2f}",
                    f"推奨購入金額: {int(float(top_odds.get('recommended_bet_amount', 0) or 0))}円",
                    f"Top3的中確率: {format_percent(float(top_odds.get('top3_hit_probability', 0.0) or 0.0))} ({top_odds.get('top3_hit_probability_label', '-')})",
                    f"判定: {top_odds['buy_decision']}",
                ]
            )
        if self.buy_candidates is not None and not self.buy_candidates.empty:
            lines.append("買い候補一覧:")
            for row in self.buy_candidates.head(5).itertuples(index=False):
                lines.append(
                    f"{row.trifecta} | 確率 {format_percent(float(row.probability))} | オッズ {float(row.odds):.1f}倍 | "
                    f"補正後確率 {format_percent(float(getattr(row, 'expected_value_probability', getattr(row, 'probability', 0.0)) or 0.0))} | "
                    f"期待値 {float(row.expected_value):.2f} | "
                    f"推奨 {int(float(getattr(row, 'recommended_bet_amount', 0) or 0))}円 | "
                    f"Top3的中 {format_percent(float(getattr(row, 'top3_hit_probability', 0.0) or 0.0))} ({getattr(row, 'top3_hit_probability_label', '-')}) | "
                    f"判定 {row.buy_decision}"
                )
        lines.append("Top10 3連単候補:")
        for row in self.trifecta.head(10).itertuples(index=False):
            lines.append(
                f"{row.trifecta} | 確率 {format_percent(float(row.probability))} | "
                f"Top3的中 {format_percent(float(getattr(row, 'top3_hit_probability', 0.0) or 0.0))} ({getattr(row, 'top3_hit_probability_label', '-')})"
            )
        return "\n".join(lines)

    def _position_probability(self, lane: int, index: int) -> float:
        return float(
            self.trifecta.loc[
                self.trifecta["trifecta"].str.split("-").str[index].astype(int) == lane,
                "probability",
            ].sum()
        )


def predict_today_race(
    bundle: Any,
    venue: str,
    race_no: int,
    race_date: date | None = None,
    history_df: pd.DataFrame | None = None,
    course_overrides: Mapping[int, int] | Sequence[int] | str | None = None,
) -> TodayRacePrediction:
    target_date = normalize_target_date(race_date)
    venue_code = normalize_venue_code(venue)
    program_text = fetch_mbrace_program_text(target_date)
    boatrace_title = fetch_boatrace_race_title(target_date, venue_code, int(race_no))

    entries = pd.DataFrame([x.to_dict() for x in parse_entry_text(program_text)])
    if entries.empty:
        raise ValueError(f"No entry data found in schedule file for {target_date.isoformat()}.")

    race_entries = entries[
        entries["race_id"].str.contains(f"_{venue_code}_")
        & (entries["race_no"] == int(race_no))
    ].copy()
    if race_entries.empty:
        raise ValueError(
            f"No entry data found for venue={venue} race_no={race_no} on {target_date.isoformat()}."
        )

    race_entries = apply_course_overrides(race_entries, course_overrides)

    history = history_df
    if history is None:
        history = load_live_history_frame(bundle.config, target_date)

    feature_frame = build_live_feature_frame(
        race_entries,
        history,
        bundle.feature_columns,
    )
    ranking = predict_race_order(
        models=bundle.models,
        feature_columns=bundle.feature_columns,
        future_df=feature_frame,
        ensemble_weights=bundle.ensemble_weights,
    )
    trifecta = predict_trifecta_probabilities(
        models=bundle.models,
        feature_columns=bundle.feature_columns,
        future_df=feature_frame,
        ensemble_weights=bundle.ensemble_weights,
        trifecta_calibrator=bundle.trifecta_calibrator,
        classifier_models=bundle.classifier_models,
        use_v2=False,
        rerank_top_n=None,
    )
    trifecta = attach_top12_confidence_columns(trifecta)
    trifecta = apply_top12_probability_adjustment_table(
        trifecta,
        bundle.probability_adjustment_table,
        probability_col="probability",
        output_col="adjusted_probability",
    )
    trifecta = attach_top3_confidence_columns(
        trifecta,
        probability_col="adjusted_probability" if "adjusted_probability" in trifecta.columns else "probability",
    )
    trifecta = attach_boat_top1_confidence_columns(
        trifecta,
        probability_col="adjusted_probability" if "adjusted_probability" in trifecta.columns else "probability",
    )
    trifecta = attach_top3_hit_probability_columns(
        trifecta,
        bundle.top3_hit_probability_model,
        probability_col="adjusted_probability" if "adjusted_probability" in trifecta.columns else "probability",
    )
    trifecta = attach_in_course_probability_columns(
        trifecta,
        feature_frame,
        in_win_payload=getattr(bundle, "in_win_probability_model", None),
        in_collapse_payload=getattr(bundle, "in_collapse_probability_model", None),
    )
    confidence_score = calculate_prediction_confidence(ranking, trifecta)
    odds = fetch_boatrace_trifecta_odds(target_date, venue_code, int(race_no))
    odds_frame = None
    buy_candidates = None
    if odds:
        odds_frame = attach_odds_and_value(trifecta, odds)
        buy_candidates = select_buy_candidates(odds_frame)
    race_upset_score = 0.0
    race_scenario_id = "S6_MIXED_OTHER"
    race_upset_label = "堅め"
    if "race_upset_score" in trifecta.columns and trifecta["race_upset_score"].notna().any():
        race_upset_score = float(pd.to_numeric(trifecta["race_upset_score"], errors="coerce").dropna().iloc[0])
    if "race_upset_label" in trifecta.columns and trifecta["race_upset_label"].notna().any():
        race_upset_label = str(trifecta["race_upset_label"].dropna().iloc[0])
    if "race_scenario_id" in trifecta.columns and trifecta["race_scenario_id"].notna().any():
        race_scenario_id = str(trifecta["race_scenario_id"].dropna().iloc[0])
    return TodayRacePrediction(
        venue=venue,
        venue_code=venue_code,
        race_no=int(race_no),
        race_date=target_date,
        ranking=ranking,
        trifecta=trifecta,
        boatrace_title=boatrace_title,
        odds=odds_frame,
        race_scenario_id=race_scenario_id,
        race_scenario_name=scenario_display_name(race_scenario_id),
        race_scenario_description=scenario_description(race_scenario_id),
        confidence_score=confidence_score,
        confidence_label=label_prediction_confidence(confidence_score),
        race_upset_score=race_upset_score,
        race_upset_label=race_upset_label,
        buy_candidates=buy_candidates,
    )


def normalize_course_overrides(
    course_overrides: Mapping[int, int] | Sequence[int] | str | None,
) -> dict[int, int]:
    if course_overrides is None:
        return {}
    if isinstance(course_overrides, str):
        value = course_overrides.strip()
        if not value:
            return {}
        if "=" in value:
            parsed: dict[int, int] = {}
            for item in value.split(","):
                token = item.strip()
                if not token:
                    continue
                if "=" not in token:
                    raise ValueError("Course override items must be formatted as lane=course.")
                lane_text, course_text = token.split("=", 1)
                parsed[int(lane_text.strip())] = int(course_text.strip())
            return _validate_course_overrides(parsed)
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if len(parts) == 1 and len(parts[0]) == 6 and parts[0].isdigit():
            parts = list(parts[0])
        if len(parts) != 6:
            raise ValueError("Course list must contain 6 values, for example: 1,2,3,4,5,6.")
        return _validate_course_overrides({lane: int(course) for lane, course in enumerate(parts, start=1)})
    if isinstance(course_overrides, Mapping):
        return _validate_course_overrides({int(lane): int(course) for lane, course in course_overrides.items()})
    values = list(course_overrides)
    if len(values) != 6:
        raise ValueError("Course override sequence must contain 6 values.")
    return _validate_course_overrides({lane: int(course) for lane, course in enumerate(values, start=1)})


def apply_course_overrides(
    race_entries: pd.DataFrame,
    course_overrides: Mapping[int, int] | Sequence[int] | str | None,
) -> pd.DataFrame:
    frame = race_entries.copy()
    if "course" not in frame.columns:
        frame["course"] = frame["lane"]
    frame["course"] = pd.to_numeric(frame["course"], errors="coerce").fillna(
        pd.to_numeric(frame["lane"], errors="coerce")
    )
    overrides = normalize_course_overrides(course_overrides)
    if overrides:
        for lane, course in overrides.items():
            frame.loc[pd.to_numeric(frame["lane"], errors="coerce") == lane, "course"] = course
    _validate_race_entry_courses(frame)
    frame["course"] = pd.to_numeric(frame["course"], errors="coerce").astype(int)
    return frame


def _validate_course_overrides(overrides: dict[int, int]) -> dict[int, int]:
    for lane, course in overrides.items():
        if lane < 1 or lane > 6:
            raise ValueError(f"Lane must be between 1 and 6: {lane}")
        if course < 1 or course > 6:
            raise ValueError(f"Course must be between 1 and 6: {course}")
    return overrides


def _validate_race_entry_courses(frame: pd.DataFrame) -> None:
    lane_values = pd.to_numeric(frame.get("lane"), errors="coerce")
    course_values = pd.to_numeric(frame.get("course"), errors="coerce")
    if lane_values.isna().any() or course_values.isna().any():
        raise ValueError("Lane and course must be available for all race entries.")
    courses = [int(value) for value in course_values.tolist()]
    invalid = [course for course in courses if course < 1 or course > 6]
    if invalid:
        raise ValueError(f"Course must be between 1 and 6: {invalid[0]}")
    if len(courses) == 6 and len(set(courses)) != 6:
        raise ValueError("Course assignments must be unique across the 6 lanes.")


def fetch_daily_race_schedule(target_date: date | None = None) -> dict[str, dict[int, time]]:
    return _fetch_daily_race_schedule(normalize_target_date(target_date))


def parse_program_race_schedule(program_text: str) -> dict[str, dict[int, time]]:
    return _parse_program_race_schedule(program_text)


def choose_default_today_venue(
    schedule: dict[str, dict[int, time]],
    preferred_venue: str = "15",
) -> str:
    return _choose_default_today_venue(schedule, preferred_venue=preferred_venue)


def choose_default_today_race_no(
    schedule: dict[str, dict[int, time]],
    venue_code: str,
    now: datetime | None = None,
) -> int:
    return _choose_default_today_race_no(schedule, venue_code=venue_code, now=now)


def load_live_history_frame(config: dict[str, Any], target_date: date) -> pd.DataFrame:
    processed_dir = Path(config["data"].get("processed_dir", "data/processed"))
    race_results_path = processed_dir / "race_results.parquet"
    rolling_years = float(config.get("data", {}).get("rolling_years", 3) or 3)
    rolling_months = max(1, int(round(rolling_years * 12)))
    min_history_date = (pd.Timestamp(target_date) - pd.DateOffset(months=rolling_months)).date()

    if race_results_path.exists():
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
            "winning_style",
        ]
        history = pd.read_parquet(
            race_results_path,
            columns=history_columns,
            filters=[
                ("race_date", ">=", min_history_date),
                ("race_date", "<", target_date),
            ],
        )
        history["is_win"] = (pd.to_numeric(history["finish_position"], errors="coerce") == 1).astype(int)
        history["is_top2"] = (
            pd.to_numeric(history["finish_position"], errors="coerce").fillna(999).astype(int) <= 2
        ).astype(int)
        history["is_top3"] = (
            pd.to_numeric(history["finish_position"], errors="coerce").fillna(999).astype(int) <= 3
        ).astype(int)
        history["race_date"] = pd.to_datetime(history["race_date"])
        return history

    history_columns = [
        "race_date",
        "race_no",
        "racer_id",
        "venue",
        "lane",
        "course",
        "motor_no",
        "boat_no",
        "is_win",
        "is_top2",
        "is_top3",
        "finish_position",
        "start_timing",
        "exhibition_time",
        "winning_style",
    ]
    try:
        history = pd.read_parquet(
            config["data"]["training_table"],
            columns=history_columns,
            filters=[
                ("race_date", ">=", min_history_date),
                ("race_date", "<", target_date),
            ],
        )
    except Exception:
        history = pd.read_parquet(
            config["data"]["training_table"],
            columns=[column for column in history_columns if column != "is_top2"],
            filters=[
                ("race_date", ">=", min_history_date),
                ("race_date", "<", target_date),
            ],
        )
    if "is_top2" not in history.columns:
        finish = pd.to_numeric(history.get("finish_position"), errors="coerce")
        history["is_top2"] = (finish.fillna(999).astype(int) <= 2).astype(int)
    history["race_date"] = pd.to_datetime(history["race_date"])
    return history


def build_live_feature_frame(
    race_entries: pd.DataFrame,
    history_df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    categorical_feature_names = {
        "venue",
        "leg_type",
        "bet_type",
        "class_name",
        "branch",
        "weather",
        "wind_direction",
    }
    hist = history_df[pd.to_datetime(history_df["race_date"]) < pd.Timestamp(race_entries["race_date"].iloc[0])].copy()
    frame = race_entries.copy()
    frame["race_date"] = pd.to_datetime(frame["race_date"])

    frame = merge_recent_group_features(frame, hist)
    frame = fill_live_measurement_proxies(frame)
    frame = add_current_meet_features(frame)
    frame = add_race_relative_features(frame)

    for column in feature_columns:
        if column not in frame.columns:
            frame[column] = pd.NA
        elif column not in categorical_feature_names:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    display_columns = [
        "race_id",
        "race_date",
        "venue",
        "race_no",
        "lane",
        "racer_name",
        "class_name",
        "branch",
        "age",
        "motor_no",
        "boat_no",
        "course",
        "exhibition_time",
        "start_timing",
    ]
    output_columns: list[str] = []
    for column in [*display_columns, *feature_columns]:
        if column in frame.columns and column not in output_columns:
            output_columns.append(column)
    return frame[output_columns].copy()


def fill_live_measurement_proxies(frame: pd.DataFrame) -> pd.DataFrame:
    filled = frame.copy()

    if "start_timing" not in filled.columns:
        filled["start_timing"] = pd.NA
    filled["start_timing"] = pd.to_numeric(filled["start_timing"], errors="coerce")
    if "racer_venue_prev_avg_st" in filled.columns:
        filled["start_timing"] = filled["start_timing"].fillna(
            pd.to_numeric(filled["racer_venue_prev_avg_st"], errors="coerce")
        )
    if "racer_prev_avg_st" in filled.columns:
        filled["start_timing"] = filled["start_timing"].fillna(
            pd.to_numeric(filled["racer_prev_avg_st"], errors="coerce")
        )

    if "course" not in filled.columns:
        filled["course"] = pd.NA
    filled["course"] = pd.to_numeric(filled["course"], errors="coerce")
    if "racer_venue_prev_avg_course" in filled.columns:
        filled["course"] = filled["course"].fillna(
            pd.to_numeric(filled["racer_venue_prev_avg_course"], errors="coerce")
        )
    if "racer_prev_avg_course" in filled.columns:
        filled["course"] = filled["course"].fillna(
            pd.to_numeric(filled["racer_prev_avg_course"], errors="coerce")
        )

    if "exhibition_time" not in filled.columns:
        filled["exhibition_time"] = pd.NA
    filled["exhibition_time"] = pd.to_numeric(filled["exhibition_time"], errors="coerce")
    if "racer_venue_prev_avg_exhibition" in filled.columns:
        filled["exhibition_time"] = filled["exhibition_time"].fillna(
            pd.to_numeric(filled["racer_venue_prev_avg_exhibition"], errors="coerce")
        )
    if "racer_prev_avg_exhibition" in filled.columns:
        filled["exhibition_time"] = filled["exhibition_time"].fillna(
            pd.to_numeric(filled["racer_prev_avg_exhibition"], errors="coerce")
        )

    return filled


def merge_recent_group_features(frame: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    if "course" not in frame.columns:
        frame = frame.copy()
        frame["course"] = frame["lane"]
    frame["course"] = pd.to_numeric(frame["course"], errors="coerce").fillna(
        pd.to_numeric(frame["lane"], errors="coerce")
    )
    if "is_top2" not in hist.columns:
        hist = hist.copy()
        finish = pd.to_numeric(hist.get("finish_position"), errors="coerce")
        hist["is_top2"] = (finish.fillna(999).astype(int) <= 2).astype(int)
    hist = add_decision_style_flag_columns(hist)

    racer_ids = set(frame["racer_id"].dropna().tolist())
    motor_nos = set(frame["motor_no"].dropna().tolist())
    boat_nos = set(frame["boat_no"].dropna().tolist())
    venues = set(frame["venue"].dropna().tolist())
    lanes = set(frame["lane"].dropna().tolist())

    hist_racer = hist[hist["racer_id"].isin(racer_ids)].copy()
    hist_motor = hist[hist["motor_no"].isin(motor_nos)].copy()
    hist_boat = hist[hist["boat_no"].isin(boat_nos)].copy()

    racer_recent = recent_group_agg(
        hist_racer,
        ["racer_id"],
        {
            "is_win": [("racer_prev_win_rate", 30, "mean"), ("racer_prev_win_rate_5", 5, "mean")],
            "is_top3": [("racer_prev_top3_rate", 30, "mean"), ("racer_prev_top3_rate_5", 5, "mean")],
            "finish_position": [
                ("racer_prev_avg_finish", 30, "mean"),
                ("racer_prev_avg_finish_5", 5, "mean"),
                ("racer_prev_avg_finish_10", 10, "mean"),
                ("racer_prev_best_finish_5", 5, "min"),
                ("racer_prev_worst_finish_5", 5, "max"),
            ],
            "start_timing": [
                ("racer_prev_avg_st", 30, "mean"),
                ("racer_prev_avg_st_5", 5, "mean"),
                ("racer_prev_avg_st_10", 10, "mean"),
                ("racer_prev_std_st_10", 10, "std"),
                ("racer_prev_best_st_5", 5, "min"),
                ("racer_prev_best_st_10", 10, "min"),
                ("racer_prev_best_st_30", 30, "min"),
                ("racer_prev_worst_st_5", 5, "max"),
                ("racer_prev_worst_st_10", 10, "max"),
                ("racer_prev_worst_st_30", 30, "max"),
            ],
            "exhibition_time": [("racer_prev_avg_exhibition", 30, "mean")],
            "course": [("racer_prev_avg_course", 30, "mean")],
            **{
                f"decision_style_{style_key}_win": [(f"racer_prev_{style_key}_rate", 30, "mean")]
                for style_key in DECISION_STYLE_KEYS
            },
        },
    )
    frame = frame.merge(racer_recent, on=["racer_id"], how="left")
    frame["racer_prev_count"] = (
        hist_racer.groupby("racer_id").size().rename("racer_prev_count").reindex(frame["racer_id"]).to_numpy()
    )

    hist_racer_venue = hist_racer[hist_racer["venue"].isin(venues)].copy()
    frame = frame.merge(
        recent_group_agg(
            hist_racer_venue,
            ["racer_id", "venue"],
            {
                "is_win": [("racer_venue_prev_win_rate", 15, "mean")],
                "is_top3": [("racer_venue_prev_top3_rate", 15, "mean")],
                "start_timing": [("racer_venue_prev_avg_st", 15, "mean")],
                "exhibition_time": [("racer_venue_prev_avg_exhibition", 15, "mean")],
                "course": [("racer_venue_prev_avg_course", 15, "mean")],
            },
        ),
        on=["racer_id", "venue"],
        how="left",
    )
    frame = frame.merge(
        recent_group_agg(
            hist_racer[hist_racer["lane"].isin(lanes)].copy(),
            ["racer_id", "lane"],
            {
                "is_win": [("racer_lane_prev_win_rate", 15, "mean")],
                "is_top3": [("racer_lane_prev_top3_rate", 15, "mean")],
                "start_timing": [("racer_lane_prev_avg_st", 15, "mean")],
                "finish_position": [("racer_lane_prev_avg_finish", 15, "mean")],
            },
        ),
        on=["racer_id", "lane"],
        how="left",
    )
    frame = frame.merge(
        recent_group_agg(
            hist_racer_venue[hist_racer_venue["lane"].isin(lanes)].copy(),
            ["racer_id", "venue", "lane"],
            {
                "is_top3": [("racer_venue_lane_prev_top3_rate", 10, "mean")],
                "start_timing": [("racer_venue_lane_prev_avg_st", 10, "mean")],
            },
        ),
        on=["racer_id", "venue", "lane"],
        how="left",
    )
    if "course" in hist_racer.columns:
        frame = frame.merge(
            recent_group_agg(
                hist_racer.dropna(subset=["course"]).assign(course=lambda d: d["course"].astype(int)),
                ["racer_id", "course"],
                {
                    "is_top3": [("racer_course_prev_top3_rate", 10, "mean")],
                    "finish_position": [("racer_course_prev_avg_finish", 10, "mean")],
                    **{
                        f"decision_style_{style_key}_win": [(f"racer_course_prev_{style_key}_rate", 15, "mean")]
                        for style_key in DECISION_STYLE_KEYS
                    },
                },
            ),
            on=["racer_id", "course"],
            how="left",
        )

    hist_venue = hist[hist["venue"].isin(venues)].copy()
    if "course" in hist_venue.columns and not hist_venue.empty:
        frame = frame.merge(
            recent_group_agg(
                hist_venue.dropna(subset=["course"]).assign(course=lambda d: d["course"].astype(int)),
                ["venue", "course"],
                {
                    "is_win": [("venue_course_prev_win_rate", 200, "mean")],
                    "is_top2": [("venue_course_prev_top2_rate", 200, "mean")],
                    "is_top3": [("venue_course_prev_top3_rate", 200, "mean")],
                    "finish_position": [("venue_course_prev_avg_finish", 200, "mean")],
                    **{
                        f"decision_style_{style_key}_win": [(f"venue_course_prev_{style_key}_rate", 200, "mean")]
                        for style_key in DECISION_STYLE_KEYS
                    },
                },
            ),
            on=["venue", "course"],
            how="left",
        )

    frame = frame.merge(
        recent_group_agg(
            hist_venue[hist_venue["lane"].isin(lanes)].copy(),
            ["venue", "lane"],
            {
                "is_win": [("venue_lane_prev_win_rate", 200, "mean")],
                "is_top2": [("venue_lane_prev_top2_rate", 200, "mean")],
                "is_top3": [("venue_lane_prev_top3_rate", 200, "mean")],
                "finish_position": [("venue_lane_prev_avg_finish", 200, "mean")],
            },
        ),
        on=["venue", "lane"],
        how="left",
    )

    frame = frame.merge(
        recent_group_agg(
            hist_motor,
            ["motor_no"],
            {
                "is_top3": [("motor_prev_top3_rate", 40, "mean")],
                "is_win": [("motor_prev_win_rate", 40, "mean")],
                "start_timing": [("motor_prev_avg_st", 40, "mean")],
            },
        ),
        on=["motor_no"],
        how="left",
    )
    frame = frame.merge(
        recent_group_agg(
            hist_boat,
            ["boat_no"],
            {
                "is_top3": [("boat_prev_top3_rate", 40, "mean")],
                "is_win": [("boat_prev_win_rate", 40, "mean")],
                "start_timing": [("boat_prev_avg_st", 40, "mean")],
            },
        ),
        on=["boat_no"],
        how="left",
    )

    required_recent_columns = [
        "racer_prev_win_rate",
        "racer_prev_win_rate_5",
        "racer_prev_top3_rate",
        "racer_prev_top3_rate_5",
        "racer_prev_avg_finish",
        "racer_prev_avg_finish_5",
        "racer_prev_avg_finish_10",
        "racer_prev_best_finish_5",
        "racer_prev_worst_finish_5",
        "racer_prev_avg_st",
        "racer_prev_avg_st_5",
        "racer_prev_avg_st_10",
        "racer_prev_std_st_10",
        "racer_prev_best_st_5",
        "racer_prev_best_st_10",
        "racer_prev_best_st_30",
        "racer_prev_worst_st_5",
        "racer_prev_worst_st_10",
        "racer_prev_worst_st_30",
        "racer_prev_avg_exhibition",
        "racer_prev_avg_course",
        "racer_venue_prev_win_rate",
        "racer_venue_prev_top3_rate",
        "racer_venue_prev_avg_st",
        "racer_venue_prev_avg_exhibition",
        "racer_venue_prev_avg_course",
        "racer_lane_prev_win_rate",
        "racer_lane_prev_top3_rate",
        "racer_lane_prev_avg_st",
        "racer_lane_prev_avg_finish",
        "racer_venue_lane_prev_top3_rate",
        "racer_venue_lane_prev_avg_st",
        "racer_course_prev_top3_rate",
        "racer_course_prev_avg_finish",
        "venue_course_prev_win_rate",
        "venue_course_prev_top2_rate",
        "venue_course_prev_top3_rate",
        "venue_course_prev_avg_finish",
        "venue_lane_prev_win_rate",
        "venue_lane_prev_top2_rate",
        "venue_lane_prev_top3_rate",
        "venue_lane_prev_avg_finish",
        "motor_prev_top3_rate",
        "motor_prev_win_rate",
        "motor_prev_avg_st",
        "boat_prev_top3_rate",
        "boat_prev_win_rate",
        "boat_prev_avg_st",
        *[f"racer_prev_{style_key}_rate" for style_key in DECISION_STYLE_KEYS],
        *[f"racer_course_prev_{style_key}_rate" for style_key in DECISION_STYLE_KEYS],
        *[f"venue_course_prev_{style_key}_rate" for style_key in DECISION_STYLE_KEYS],
    ]
    for column in required_recent_columns:
        if column not in frame.columns:
            frame[column] = pd.NA

    st_5 = pd.to_numeric(frame.get("racer_prev_avg_st_5"), errors="coerce")
    st_10 = pd.to_numeric(frame.get("racer_prev_avg_st_10"), errors="coerce")
    finish_5 = pd.to_numeric(frame.get("racer_prev_avg_finish_5"), errors="coerce")
    finish_10 = pd.to_numeric(frame.get("racer_prev_avg_finish_10"), errors="coerce")
    frame["st_momentum_diff"] = st_5 - st_10
    frame["finish_momentum_diff"] = finish_10 - finish_5
    frame = add_flow_probability_features(frame)
    frame["lane_is_inner"] = (frame["lane"] <= 3).astype(int)
    frame["lane_is_outer"] = (frame["lane"] >= 5).astype(int)
    return frame


def recent_group_agg(
    hist: pd.DataFrame,
    group_cols: list[str],
    spec: dict[str, list[tuple[str, int, str]]],
) -> pd.DataFrame:
    output_columns = [out_col for targets in spec.values() for out_col, _, _ in targets]
    empty_columns = [*group_cols, *output_columns]
    if hist.empty:
        return pd.DataFrame(columns=empty_columns)

    rows = []
    for key, group in hist.sort_values(["race_date", "race_no"]).groupby(group_cols, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = {col: val for col, val in zip(group_cols, key)}
        for source_col, targets in spec.items():
            series = pd.to_numeric(group[source_col], errors="coerce").dropna()
            for out_col, window, agg in targets:
                tail = series.tail(window)
                if tail.empty:
                    row[out_col] = pd.NA
                elif agg == "mean":
                    row[out_col] = float(tail.mean())
                elif agg == "min":
                    row[out_col] = float(tail.min())
                elif agg == "max":
                    row[out_col] = float(tail.max())
                elif agg == "std":
                    row[out_col] = float(tail.std()) if len(tail) >= 2 else pd.NA
                else:
                    raise ValueError(f"Unsupported aggregation: {agg}")
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=empty_columns)
    return pd.DataFrame(rows, columns=empty_columns)


def fetch_mbrace_program_text(target_date: date) -> str:
    archive_bytes = fetch_mbrace_program_archive(target_date)
    return extract_lzh_text(archive_bytes)


def fetch_mbrace_program_archive(target_date: date) -> bytes:
    return fetch_mbrace_daily_archive(target_date, kind="B")


def fetch_mbrace_result_text(target_date: date) -> str:
    archive_bytes = fetch_mbrace_result_archive(target_date)
    return extract_lzh_text(archive_bytes)


def fetch_mbrace_result_archive(target_date: date) -> bytes:
    return fetch_mbrace_daily_archive(target_date, kind="K")


def fetch_mbrace_daily_archive(target_date: date, kind: str) -> bytes:
    kind = kind.upper()
    if kind not in {"B", "K"}:
        raise ValueError(f"Unsupported mbrace archive kind: {kind}")
    yyyy_mm = target_date.strftime("%Y%m")
    yy_mm_dd = target_date.strftime("%y%m%d")
    url = f"https://www1.mbrace.or.jp/od2/{kind}/{yyyy_mm}/{kind.lower()}{yy_mm_dd}.lzh"
    try:
        return fetch_bytes(url, referer=f"https://www1.mbrace.or.jp/od2/{kind}/{yyyy_mm}/mday.html")
    except HTTPError as exc:
        if exc.code == 404:
            label = "schedule" if kind == "B" else "result"
            raise ValueError(f"Mbrace {label} archive not found for {target_date.isoformat()}: {url}") from exc
        raise


def fetch_boatrace_race_title(target_date: date, venue_code: str, race_no: int) -> str | None:
    hd = target_date.strftime("%Y%m%d")
    url = f"https://www.boatrace.jp/owpc/pc/race/racelist?jcd={venue_code}&hd={hd}&rno={int(race_no):02d}"
    html = fetch_text(url, referer="https://www.boatrace.jp/")
    match = _HTML_TITLE_RE.search(html)
    if not match:
        return None
    title = re.sub(r"\s+", " ", match.group(1)).strip()
    if "システムエラー" in title or title == "BOAT RACE オフィシャルウェブサイト":
        return None
    return title


def fetch_boatrace_trifecta_odds(target_date: date, venue_code: str, race_no: int) -> dict[str, float] | None:
    hd = target_date.strftime("%Y%m%d")
    url = f"https://www.boatrace.jp/owpc/pc/race/odds3t?jcd={venue_code}&hd={hd}&rno={int(race_no):02d}"
    html = fetch_text(url, referer="https://www.boatrace.jp/")
    match = _HTML_TITLE_RE.search(html)
    title = re.sub(r"\s+", " ", match.group(1)).strip() if match else ""
    if title == "BOAT RACE オフィシャルウェブサイト" or "システムエラー" in title:
        return None
    return parse_trifecta_odds_html(html)


def fetch_boatrace_exhibition_courses(
    target_date: date,
    venue_code: str,
    race_no: int,
) -> tuple[int, int, int, int, int, int] | None:
    hd = target_date.strftime("%Y%m%d")
    url = f"https://www.boatrace.jp/owpc/pc/race/beforeinfo?jcd={venue_code}&hd={hd}&rno={int(race_no):02d}"
    try:
        html = fetch_text(url, referer="https://www.boatrace.jp/")
    except Exception:
        return None
    match = _HTML_TITLE_RE.search(html)
    title = re.sub(r"\s+", " ", match.group(1)).strip() if match else ""
    if title == "BOAT RACE オフィシャルウェブサイト" or "システムエラー" in title:
        return None
    return parse_start_exhibition_courses_html(html)


def parse_start_exhibition_courses_html(html: str) -> tuple[int, int, int, int, int, int] | None:
    soup = BeautifulSoup(html, "html.parser")
    numbers: list[int] = []
    for block in soup.select(".table1_boatImage1"):
        number = _extract_start_exhibition_boat_number(block)
        if number is not None:
            numbers.append(number)
        if len(numbers) >= 6:
            break
    if len(numbers) < 6:
        numbers = []
        for span in soup.select(".table1_boatImage1Number"):
            value = parse_int_text(span.get_text(" ", strip=True))
            if value is not None:
                numbers.append(value)
            if len(numbers) >= 6:
                break
    if len(numbers) != 6 or sorted(numbers) != [1, 2, 3, 4, 5, 6]:
        return None
    lane_to_course = {lane: course for course, lane in enumerate(numbers, start=1)}
    return tuple(lane_to_course[lane] for lane in range(1, 7))  # type: ignore[return-value]


def _extract_start_exhibition_boat_number(block: Any) -> int | None:
    number_span = block.select_one(".table1_boatImage1Number")
    if number_span is not None:
        number = parse_int_text(number_span.get_text(" ", strip=True))
        if number is not None:
            return number
        class_text = " ".join(str(value) for value in number_span.get("class", []))
        match = re.search(r"is-type([1-6])", class_text)
        if match:
            return int(match.group(1))
    image = block.find("img")
    if image is not None:
        for attr in ("alt", "title", "src"):
            value = str(image.get(attr, ""))
            match = re.search(r"(?:^|[_-])([1-6])(?:\D|$)", value)
            if match:
                return int(match.group(1))
    return None


def parse_int_text(text: str) -> int | None:
    match = re.search(r"[1-6]", str(text))
    if not match:
        return None
    return int(match.group(0))


def parse_trifecta_odds_html(html: str) -> dict[str, float] | None:
    soup = BeautifulSoup(html, "html.parser")
    candidate_tables = []
    for table in soup.find_all("table"):
        odds_cells = table.select("td.oddsPoint")
        if len(odds_cells) >= 100:
            candidate_tables.append((len(odds_cells), table))
    if not candidate_tables:
        return None

    odds_table = sorted(candidate_tables, key=lambda item: item[0], reverse=True)[0][1]
    tbody = odds_table.find("tbody")
    if tbody is None:
        return None

    rows = tbody.find_all("tr", recursive=False)
    rowspan_state: dict[int, tuple[str, int]] = {}
    grid: list[list[str]] = []
    total_columns = 18

    for tr in rows:
        row = [""] * total_columns
        for col_idx, (value, remaining) in list(rowspan_state.items()):
            row[col_idx] = value
            if remaining <= 1:
                del rowspan_state[col_idx]
            else:
                rowspan_state[col_idx] = (value, remaining - 1)

        col = 0
        for td in tr.find_all("td", recursive=False):
            while col < total_columns and row[col]:
                col += 1
            if col >= total_columns:
                break
            value = td.get_text(" ", strip=True)
            row[col] = value
            rowspan = int(td.get("rowspan", 1))
            if rowspan > 1:
                rowspan_state[col] = (value, rowspan - 1)
            col += 1
        grid.append(row)

    odds_map: dict[str, float] = {}
    for row in grid:
        for first in range(1, 7):
            offset = (first - 1) * 3
            second = row[offset].strip()
            third = row[offset + 1].strip()
            odds_text = row[offset + 2].strip()
            if not second or not third or not odds_text:
                continue
            if not second.isdigit() or not third.isdigit():
                continue
            odds_value = parse_odds_value(odds_text)
            if odds_value is None:
                continue
            odds_map[f"{first}-{int(second)}-{int(third)}"] = odds_value
    return odds_map or None


def parse_odds_value(text: str) -> float | None:
    cleaned = text.replace(",", "").strip()
    if cleaned in {"", "-", "欠場", "発売前"}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def attach_odds_and_value(
    trifecta: pd.DataFrame,
    odds_map: dict[str, float],
    expected_value_threshold: float = BUY_EXPECTED_VALUE_THRESHOLD,
    min_odds: float = BUY_MIN_ODDS,
) -> pd.DataFrame:
    frame = trifecta.copy()
    frame["odds"] = frame["trifecta"].map(odds_map)
    frame = frame.dropna(subset=["odds"]).copy()
    if frame.empty:
        return frame

    frame["probability"] = pd.to_numeric(frame["probability"], errors="coerce")
    value_probability_col = "adjusted_probability" if "adjusted_probability" in frame.columns else "probability"
    frame[value_probability_col] = pd.to_numeric(frame[value_probability_col], errors="coerce")
    frame["odds"] = pd.to_numeric(frame["odds"], errors="coerce")
    if "race_id" in frame.columns:
        frame["prediction_rank"] = frame.groupby("race_id")[value_probability_col].rank(
            ascending=False,
            method="first",
        )
    else:
        frame["prediction_rank"] = frame[value_probability_col].rank(ascending=False, method="first")
    frame["prediction_rank"] = frame["prediction_rank"].astype("Int64")
    frame["expected_value_probability"] = frame[value_probability_col]
    frame["expected_value"] = frame["odds"] * frame["expected_value_probability"]
    frame = attach_darkhorse_odds_context(frame)
    frame = attach_top3_hit_probability_buy_decision(frame)
    frame = frame.sort_values(
        ["top3_hit_probability", "prediction_rank", "expected_value", "expected_value_probability"],
        ascending=[False, True, False, False],
    ).reset_index(drop=True)
    return frame


def attach_top3_hit_probability_buy_decision(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "top3_hit_probability" in result.columns:
        probability = pd.to_numeric(result["top3_hit_probability"], errors="coerce").fillna(0.0)
    else:
        probability = pd.Series(0.0, index=result.index)
        result["top3_hit_probability"] = probability
    rank = pd.to_numeric(result.get("prediction_rank"), errors="coerce").fillna(999.0)
    label = result.get("top3_hit_probability_label")
    if label is None:
        label = probability.map(lambda value: "high" if value >= 0.50 else ("middle" if value >= 0.35 else "low"))
        result["top3_hit_probability_label"] = label
    else:
        label = label.astype(str)

    top3_candidate = rank <= 3
    high = top3_candidate & (label == "high")
    middle = top3_candidate & (label == "middle")

    result["buy_decision_source"] = "top3_hit_probability"
    result["buy_decision"] = "見送り"
    result.loc[middle, "buy_decision"] = "検討"
    result.loc[high, "buy_decision"] = "買い"
    result["recommended_bet_amount"] = 0
    result.loc[middle, "recommended_bet_amount"] = 100
    result.loc[high, "recommended_bet_amount"] = 300
    result["recommended_bet_units"] = (result["recommended_bet_amount"] // 100).astype(int)
    return result


def select_buy_candidates(odds_frame: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    if odds_frame.empty:
        return odds_frame
    buy_only = odds_frame[odds_frame["buy_decision"] != "見送り"].copy()
    if "recommended_bet_amount" in odds_frame.columns:
        amount = pd.to_numeric(odds_frame["recommended_bet_amount"], errors="coerce").fillna(0.0)
        buy_only = odds_frame[amount > 0].copy()
    sort_columns = [
        "top3_hit_probability",
        "recommended_bet_amount",
        "expected_value",
        "probability",
    ]
    sort_columns = [column for column in sort_columns if column in odds_frame.columns]
    if not buy_only.empty:
        return buy_only.sort_values(
            sort_columns,
            ascending=[False] * len(sort_columns),
        ).head(top_n).reset_index(drop=True)
    return odds_frame.sort_values(
        sort_columns,
        ascending=[False] * len(sort_columns),
    ).head(top_n).reset_index(drop=True)


def calculate_prediction_confidence(ranking: pd.DataFrame, trifecta: pd.DataFrame) -> float:
    race_ranking = ranking.sort_values("predicted_rank").reset_index(drop=True)
    top_win = float(race_ranking.iloc[0]["win_probability_like"])
    second_win = float(race_ranking.iloc[1]["win_probability_like"]) if len(race_ranking) > 1 else 0.0
    top_trifecta = float(trifecta.iloc[0]["probability"])
    next_trifecta = float(trifecta.iloc[1]["probability"]) if len(trifecta) > 1 else 0.0
    score = 0.5 * top_win + 0.3 * max(top_win - second_win, 0.0) + 0.2 * max(top_trifecta - next_trifecta, 0.0)
    return float(min(max(score, 0.0), 1.0))


def label_prediction_confidence(score: float) -> str:
    if score >= 0.22:
        return "高"
    if score >= 0.16:
        return "中"
    return "低"


def format_percent(value: float, cap: float | None = None) -> str:
    display_value = value
    if cap is not None:
        display_value = min(display_value, cap)
    return f"{display_value * 100:.1f}%"


def extract_lzh_text(archive_bytes: bytes) -> str:
    seven_zip = find_seven_zip()
    with tempfile.TemporaryDirectory(prefix="boatrace_lzh_") as tmpdir:
        temp_dir = Path(tmpdir)
        archive_path = temp_dir / "program.lzh"
        archive_path.write_bytes(archive_bytes)
        result = subprocess.run(
            [seven_zip, "e", str(archive_path), f"-o{temp_dir}", "-y"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to extract LZH archive with 7-Zip: {result.stderr.strip()}")

        extracted = sorted(path for path in temp_dir.iterdir() if path.is_file() and path.name != archive_path.name)
        if not extracted:
            raise RuntimeError("No file was extracted from the schedule archive.")

        for path in extracted:
            data = path.read_bytes()
            for encoding in ("cp932", "shift_jis", "utf-8", "latin1"):
                try:
                    text = data.decode(encoding)
                except UnicodeDecodeError:
                    continue
                if "競走成績" not in text and "番組表" not in text and "艇" not in text:
                    continue
                return text
        return extracted[0].read_text(encoding="cp932", errors="ignore")


def extract_lzh_first_file_bytes(archive_bytes: bytes) -> bytes:
    seven_zip = find_seven_zip()
    with tempfile.TemporaryDirectory(prefix="boatrace_lzh_") as tmpdir:
        temp_dir = Path(tmpdir)
        archive_path = temp_dir / "program.lzh"
        archive_path.write_bytes(archive_bytes)
        result = subprocess.run(
            [seven_zip, "e", str(archive_path), f"-o{temp_dir}", "-y"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to extract LZH archive with 7-Zip: {result.stderr.strip()}")
        extracted = sorted(path for path in temp_dir.iterdir() if path.is_file() and path.name != archive_path.name)
        if not extracted:
            raise RuntimeError("No file was extracted from the schedule archive.")
        return extracted[0].read_bytes()


def decode_rowdata_bytes(data: bytes) -> str:
    for encoding in ("cp932", "shift_jis", "utf-8", "latin1"):
        try:
            text = data.decode(encoding)
        except UnicodeDecodeError:
            continue
        if "競走成績" not in text and "番組表" not in text and "艇" not in text:
            continue
        return text
    return data.decode("cp932", errors="ignore")


def find_seven_zip() -> str:
    candidates = [
        Path(r"C:\Program Files\7-Zip\7z.exe"),
        Path(r"C:\Program Files\7-Zip\7z.EXE"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise RuntimeError("7-Zip is required to extract .lzh schedule files but was not found.")


def extract_lzh_text(archive_bytes: bytes) -> str:  # noqa: F811
    extracted = extract_lzh_entries(archive_bytes)
    for _, data in extracted:
        for encoding in ("cp932", "shift_jis", "utf-8", "latin1"):
            try:
                text = data.decode(encoding)
            except UnicodeDecodeError:
                continue
            if "競走成績" not in text and "番組表" not in text and "艇" not in text:
                continue
            return text
    return decode_rowdata_bytes(extracted[0][1])


def extract_lzh_first_file_bytes(archive_bytes: bytes) -> bytes:  # noqa: F811
    return extract_lzh_entries(archive_bytes)[0][1]


def extract_lzh_entries(archive_bytes: bytes) -> list[tuple[str, bytes]]:
    extraction_errors: list[str] = []
    try:
        return extract_lzh_entries_with_lhafile(archive_bytes)
    except Exception as exc:
        extraction_errors.append(f"lhafile: {exc}")

    seven_zip = find_seven_zip()
    if seven_zip is not None:
        try:
            return extract_lzh_entries_with_seven_zip(archive_bytes, seven_zip)
        except Exception as exc:
            extraction_errors.append(f"7-Zip: {exc}")

    detail = " / ".join(extraction_errors) if extraction_errors else "no extractor available"
    raise RuntimeError(
        "Failed to extract .lzh archive. Install the Python package 'lhafile' or 7-Zip. "
        f"Details: {detail}"
    )


def extract_lzh_entries_with_lhafile(archive_bytes: bytes) -> list[tuple[str, bytes]]:
    if lhafile is None:
        raise RuntimeError("lhafile is not installed")
    archive_class = getattr(lhafile, "Lhafile", None) or getattr(lhafile, "LhaFile", None)
    if archive_class is None:
        raise RuntimeError("lhafile does not expose Lhafile/LhaFile")

    with tempfile.TemporaryDirectory(prefix="boatrace_lzh_") as tmpdir:
        temp_dir = Path(tmpdir)
        archive_path = temp_dir / "program.lzh"
        archive_path.write_bytes(archive_bytes)
        archive = archive_class(str(archive_path))
        try:
            if hasattr(archive, "namelist"):
                names = list(archive.namelist())
            elif hasattr(archive, "infolist"):
                names = [
                    getattr(info, "filename", getattr(info, "name", str(info)))
                    for info in archive.infolist()
                ]
            else:
                raise RuntimeError("lhafile archive object does not expose namelist/infolist")
            if not names:
                raise RuntimeError("No file was extracted from the schedule archive.")
            return [(name, archive.read(name)) for name in names]
        finally:
            close_fn = getattr(archive, "close", None)
            if callable(close_fn):
                close_fn()


def extract_lzh_entries_with_seven_zip(archive_bytes: bytes, seven_zip: str) -> list[tuple[str, bytes]]:
    with tempfile.TemporaryDirectory(prefix="boatrace_lzh_") as tmpdir:
        temp_dir = Path(tmpdir)
        archive_path = temp_dir / "program.lzh"
        archive_path.write_bytes(archive_bytes)
        result = subprocess.run(
            [seven_zip, "e", str(archive_path), f"-o{temp_dir}", "-y"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to extract LZH archive with 7-Zip: {result.stderr.strip()}")
        extracted = sorted(path for path in temp_dir.iterdir() if path.is_file() and path.name != archive_path.name)
        if not extracted:
            raise RuntimeError("No file was extracted from the schedule archive.")
        return [(path.name, path.read_bytes()) for path in extracted]


def find_seven_zip() -> str | None:  # noqa: F811
    candidates = [
        Path(r"C:\Program Files\7-Zip\7z.exe"),
        Path(r"C:\Program Files\7-Zip\7z.EXE"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def fetch_text(url: str, referer: str | None = None) -> str:
    data, content_type = fetch_response(url, referer=referer)
    encoding = content_type or "cp932"
    return data.decode(encoding, errors="ignore")


def fetch_bytes(url: str, referer: str | None = None) -> bytes:
    data, _ = fetch_response(url, referer=referer)
    return data


def fetch_response(url: str, referer: str | None = None) -> tuple[bytes, str | None]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        ),
        "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
    }
    if referer:
        headers["Referer"] = referer
    req = Request(url, headers=headers)
    with urlopen(req, timeout=30) as resp:
        data = resp.read()
        content_type = resp.headers.get_content_charset()
    return data, content_type


def normalize_venue_code(venue: str) -> str:
    cleaned = venue.replace("ボートレース", "").replace("競艇場", "").replace("競走場", "").strip()
    if cleaned in VENUE_CODE_MAP:
        return VENUE_CODE_MAP[cleaned]
    raise ValueError(f"Unknown venue: {venue}")


def normalize_target_date(value: date | str | None) -> date:
    if value is None:
        return current_jst_date()
    if isinstance(value, date):
        return value
    return pd.Timestamp(value).date()
