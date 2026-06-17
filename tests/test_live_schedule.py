from __future__ import annotations

from datetime import datetime, time

from src.today_schedule import (
    choose_default_today_race_no,
    choose_default_today_venue,
    parse_program_race_schedule,
)


def test_parse_program_race_schedule_extracts_deadlines() -> None:
    text = "\n".join(
        [
            "15BBGN",
            "  １Ｒ  予　選　　　          Ｈ１８００ｍ  電話投票締切予定時刻１０：５７ 連複",
            "  ２Ｒ  予　選　　　          Ｈ１８００ｍ  電話投票締切予定時刻１１：１９ 連複",
            "24BBGN",
            "  １Ｒ  予　選　　　          Ｈ１８００ｍ  電話投票締切予定時刻１０：５０ 連複",
        ]
    )

    schedule = parse_program_race_schedule(text)

    assert schedule == {
        "15": {1: time(10, 57), 2: time(11, 19)},
        "24": {1: time(10, 50)},
    }


def test_choose_default_today_venue_prefers_marugame_then_smallest() -> None:
    assert choose_default_today_venue({"15": {1: time(10, 57)}, "24": {1: time(10, 50)}}) == "15"
    assert choose_default_today_venue({"24": {1: time(10, 50)}, "18": {1: time(10, 51)}}) == "18"


def test_choose_default_today_race_no_prefers_nearest_future() -> None:
    schedule = {"15": {1: time(10, 57), 2: time(11, 19), 12: time(16, 7)}}

    assert choose_default_today_race_no(schedule, "15", now=datetime(2026, 6, 17, 11, 0)) == 2
    assert choose_default_today_race_no(schedule, "15", now=datetime(2026, 6, 17, 17, 0)) == 12
