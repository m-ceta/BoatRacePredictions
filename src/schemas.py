from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date
from typing import Any


@dataclass(slots=True)
class RaceEntry:
    race_id: str
    race_date: date
    venue: str
    race_no: int
    race_title: str
    leg_type: str
    distance_m: int | None
    bet_type: str | None
    lane: int
    racer_id: int
    racer_name: str
    age: int | None
    branch: str | None
    weight: int | None
    class_name: str | None
    national_win_rate: float | None
    national_place_rate: float | None
    local_win_rate: float | None
    local_place_rate: float | None
    motor_no: int | None
    motor_place_rate: float | None
    boat_no: int | None
    boat_place_rate: float | None
    current_meet_results: str | None
    early_lane_hint: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RaceResult:
    race_id: str
    race_date: date
    venue: str
    race_no: int
    lane: int
    finish_position: int
    finish_status: str | None
    racer_id: int
    racer_name: str
    motor_no: int | None
    boat_no: int | None
    exhibition_time: float | None
    course: int | None
    start_timing: float | None
    race_time: float | None
    weather: str | None
    wind_direction: str | None
    wind_speed_m: int | None
    wave_cm: int | None
    winning_style: str | None
    trifecta_payout: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
