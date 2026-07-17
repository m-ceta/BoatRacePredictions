from __future__ import annotations

import pandas as pd

import src.recent_backtest as recent_backtest
from src.recent_backtest import parse_trifecta_payouts_from_lines
from src.recent_backtest import build_recent_backtest_prediction_frame
from src.recent_backtest import compute_ticket_rank_hit_rates
from src.recent_backtest import _latest_available_rowdata_date
from src.recent_backtest import _normalize_race_type
from src.recent_backtest import _resolve_backtest_period
from src.recent_backtest import prepare_recent_backtest_entry_frame


def test_parse_trifecta_payouts_from_lines_reads_race_detail_payouts() -> None:
    lines = """
STARTK
24KBGN
大　村［成績］      5/24

   第 1日          2026/ 5/24                             ボートレース大　村

   1R       予選　　　　                 H1800m  晴　  風  北西　 2m  波　  1cm
-------------------------------------------------------------------------------
        ３連単   1-4-2     7650  人気    36

   2R       予選　　　　                 H1800m  晴　  風  北西　 1m  波　  1cm
-------------------------------------------------------------------------------
        ３連単   3-6-2    23210  人気    81
""".splitlines()

    payouts = parse_trifecta_payouts_from_lines(lines)

    assert payouts["race_id"].tolist() == [
        "2026-05-24_24_01",
        "2026-05-24_24_02",
    ]
    assert payouts["actual_trifecta"].tolist() == ["1-4-2", "3-6-2"]
    assert payouts["trifecta_payout"].tolist() == [7650.0, 23210.0]


def test_prepare_recent_backtest_entry_frame_removes_result_only_columns() -> None:
    frame = pd.DataFrame(
        [
            {
                "race_id": "2026-05-24_24_01",
                "lane": 1,
                "racer_id": 1001,
                "start_timing": 0.12,
                "course": 1,
                "exhibition_time": 6.82,
                "finish_position": 1,
                "weather": "晴",
                "race_time": 110.3,
                "target_rank": 1,
            }
        ]
    )

    prepared = prepare_recent_backtest_entry_frame(frame)

    assert "lane" in prepared.columns
    assert "racer_id" in prepared.columns
    assert "start_timing" not in prepared.columns
    assert "course" not in prepared.columns
    assert "exhibition_time" not in prepared.columns
    assert "finish_position" not in prepared.columns
    assert "weather" not in prepared.columns
    assert "race_time" not in prepared.columns
    assert "target_rank" not in prepared.columns


def test_build_recent_backtest_prediction_frame_keeps_feature_race_columns(monkeypatch) -> None:
    evaluation_rows = pd.DataFrame(
        [
            {
                "race_id": "2026-05-24_24_01",
                "race_date": pd.Timestamp("2026-05-24"),
                "lane": 1,
                "race_no": 1,
                "venue": "大村",
                "racer_id": 1001,
                "finish_position": 1,
            }
        ]
    )

    class DummyBundle:
        config = {"data": {}}
        feature_columns = ["lane", "race_no", "venue", "racer_id"]

    monkeypatch.setattr(recent_backtest, "load_live_history_frame", lambda config, target_date: pd.DataFrame())
    monkeypatch.setattr(
        recent_backtest,
        "build_live_feature_frame",
        lambda race_entries, history_df, feature_columns: race_entries[["race_id", *feature_columns]].copy(),
    )

    prediction_frame = build_recent_backtest_prediction_frame(
        DummyBundle(),
        evaluation_rows,
        start_date=pd.Timestamp("2026-05-24").date(),
    )

    assert "race_no" in prediction_frame.columns
    assert "venue" in prediction_frame.columns
    assert "race_no_x" not in prediction_frame.columns
    assert "race_no_y" not in prediction_frame.columns
    assert prediction_frame.loc[0, "finish_position"] == 1


def test_build_recent_backtest_prediction_frame_appends_prior_day_history(monkeypatch) -> None:
    evaluation_rows = pd.DataFrame(
        [
            {
                "race_id": "2026-05-24_24_01",
                "race_date": pd.Timestamp("2026-05-24"),
                "lane": 1,
                "race_no": 1,
                "venue": "大村",
                "racer_id": 1001,
                "motor_no": 11,
                "boat_no": 21,
                "finish_position": 1,
                "course": 1,
                "start_timing": 0.12,
                "exhibition_time": 6.8,
            },
            {
                "race_id": "2026-05-25_24_01",
                "race_date": pd.Timestamp("2026-05-25"),
                "lane": 1,
                "race_no": 1,
                "venue": "大村",
                "racer_id": 1001,
                "motor_no": 11,
                "boat_no": 21,
                "finish_position": 2,
                "course": 2,
                "start_timing": 0.14,
                "exhibition_time": 6.9,
            },
        ]
    )

    class DummyBundle:
        config = {"data": {}}
        feature_columns = ["lane", "race_no", "venue", "racer_id"]

    history_sizes: list[int] = []

    monkeypatch.setattr(recent_backtest, "load_live_history_frame", lambda config, target_date: pd.DataFrame())

    def fake_build_live_feature_frame(race_entries, history_df, feature_columns):
        history_sizes.append(len(history_df))
        return race_entries[["race_id", *feature_columns]].copy()

    monkeypatch.setattr(recent_backtest, "build_live_feature_frame", fake_build_live_feature_frame)

    build_recent_backtest_prediction_frame(
        DummyBundle(),
        evaluation_rows,
        start_date=pd.Timestamp("2026-05-24").date(),
    )

    assert history_sizes == [0, 1]


def test_latest_available_rowdata_date_uses_common_bk_dates(tmp_path) -> None:
    (tmp_path / "B260601.TXT").write_text("", encoding="utf-8")
    (tmp_path / "K260601.TXT").write_text("", encoding="utf-8")
    (tmp_path / "B260602.TXT").write_text("", encoding="utf-8")
    (tmp_path / "K260603.TXT").write_text("", encoding="utf-8")

    latest = _latest_available_rowdata_date(tmp_path)

    assert latest.isoformat() == "2026-06-01"


def test_normalize_race_type_groups_special_labels() -> None:
    assert _normalize_race_type("一般") == "一般戦"
    assert _normalize_race_type("予選特賞") == "特賞・選抜"
    assert _normalize_race_type("優勝戦") == "優勝戦"
    assert _normalize_race_type("準優勝戦") == "準優勝戦"


def test_resolve_backtest_period_prefers_explicit_range(tmp_path) -> None:
    start, end = _resolve_backtest_period(
        tmp_path,
        days=7,
        start_date=pd.Timestamp("2026-06-01").date(),
        end_date=pd.Timestamp("2026-06-29").date(),
    )

    assert start.isoformat() == "2026-06-01"
    assert end.isoformat() == "2026-06-29"


def test_compute_ticket_rank_hit_rates_uses_bought_tickets_only() -> None:
    ticket_details = pd.DataFrame(
        [
            {"race_id": "R1", "prediction_rank": 1, "hit": False},
            {"race_id": "R1", "prediction_rank": 2, "hit": True},
            {"race_id": "R1", "prediction_rank": 3, "hit": False},
            {"race_id": "R2", "prediction_rank": 1, "hit": False},
            {"race_id": "R2", "prediction_rank": 2, "hit": False},
            {"race_id": "R2", "prediction_rank": 3, "hit": False},
            {"race_id": "R3", "prediction_rank": 1, "hit": True},
            {"race_id": "R3", "prediction_rank": 2, "hit": False},
        ]
    )

    metrics = compute_ticket_rank_hit_rates(ticket_details, race_count=3)

    assert metrics["top1_hit_rate"] == 1 / 3
    assert metrics["top3_hit_rate"] == 2 / 3
    assert metrics["top5_hit_rate"] == 2 / 3
    assert metrics["top12_hit_rate"] == 2 / 3
