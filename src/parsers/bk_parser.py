from __future__ import annotations

import re
from datetime import date
from pathlib import Path

from src.schemas import RaceEntry, RaceResult

FULLWIDTH_SPACE = "\u3000"

ENTRY_HEADER_RE = re.compile(
    r"^\s*(?P<race_no>\d{1,2})Ｒ\s+(?P<leg_type>.+?)\s+Ｈ(?P<distance>\d+)ｍ.*?(?P<bet_type>連[複単])?\s*$"
)
ENTRY_LINE_RE = re.compile(
    r"^\s*(?P<lane>[1-6])\s+"
    r"(?P<racer_id>\d{4})"
    r"(?P<racer_name>.+?)"
    r"(?P<age>\d{1,2})"
    r"(?P<branch>[^\d]{2,3})"
    r"(?P<weight>\d{2})"
    r"(?P<class_name>[AB]\d)\s+"
    r"(?P<national_win>\d+\.\d{2})\s*"
    r"(?P<national_place>\d+\.\d{2})\s+"
    r"(?P<local_win>\d+\.\d{2})\s*"
    r"(?P<local_place>\d+\.\d{2})\s+"
    r"(?P<motor_no>\d+)\s+"
    r"(?P<motor_place>\d+\.\d{2})\s*"
    r"(?P<boat_no>\d{1,3})\s*"
    r"(?P<boat_place>\d+\.\d{2})\s*"
    r"(?P<tail>.*)$"
)
RESULT_HEADER_RE = re.compile(
    r"^\s*(?P<race_no>\d{1,2})R\s+(?P<leg_type>.+?)\s+H(?P<distance>\d+)m\s+"
    r"(?P<weather>.+?)\s+風\s+(?P<wind_direction>.+?)\s+(?P<wind_speed>\d+)m\s+波\s+(?P<wave>\d+)cm\s*$"
)
RESULT_LINE_RE = re.compile(
    r"^\s*(?P<finish>[0-9A-Z]{1,2})\s+"
    r"(?P<lane>[1-6])\s+"
    r"(?P<racer_id>\d{4})\s+"
    r"(?P<racer_name>.+?)\s+"
    r"(?P<motor_no>\d+)\s+"
    r"(?P<boat_no>\d+)\s+"
    r"(?P<tail>.*)$"
)
WINNING_STYLE_RE = re.compile(r"決まり手\s+(?P<style>\S+)")
DATE_JP_RE = re.compile(r"(?P<year>\d{2,4})年\s*(?P<month>\d{1,2})月\s*(?P<day>\d{1,2})日")
DATE_SLASH_RE = re.compile(r"(?P<year>\d{2,4})/\s*(?P<month>\d{1,2})/\s*(?P<day>\d{1,2})")
VENUE_SUFFIX_RE = re.compile(r"(?P<venue>[^\d\s]{1,6})\s*(?:競艇場|競走場|ボートレース)")
VENUE_RESULT_RE = re.compile(r"^(?P<venue>[^\d\s]{1,6})\s*［成績］")
SECTION_CODE_RE = re.compile(r"^(?P<section_code>\d{2})[BK]BGN$")
RACE_TIME_RE = re.compile(r"^\d+\.\d+\.\d+$")
KNOWN_WINNING_STYLES = {
    "逃げ",
    "差し",
    "まくり",
    "まくり差し",
    "抜き",
    "恵まれ",
}


def load_shift_jis_lines(path: Path) -> list[str]:
    return path.read_text(encoding="cp932", errors="ignore").splitlines()


def parse_entry_file(path: Path) -> list[RaceEntry]:
    lines = load_shift_jis_lines(path)
    return parse_entry_lines(lines)


def parse_entry_text(text: str) -> list[RaceEntry]:
    return parse_entry_lines(text.splitlines())


def parse_entry_lines(lines: list[str]) -> list[RaceEntry]:
    venue = extract_venue(lines)
    race_date = extract_date(lines)
    race_title = extract_race_title(lines)
    entries: list[RaceEntry] = []
    current_race_no: int | None = None
    current_leg_type: str | None = None
    current_distance: int | None = None
    current_bet_type: str | None = None
    current_section_code: str | None = None
    separator_count = 0
    collecting_entries = False
    collected_for_race = 0

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
        section_title = parse_race_title_from_line(line)
        if section_title:
            race_title = section_title

        normalized = line.replace(FULLWIDTH_SPACE, " ")
        header_match = ENTRY_HEADER_RE.match(normalized)
        if header_match:
            current_race_no = int(header_match.group("race_no"))
            current_leg_type = header_match.group("leg_type")
            current_distance = int(header_match.group("distance"))
            current_bet_type = header_match.group("bet_type")
            separator_count = 0
            collecting_entries = False
            collected_for_race = 0
            continue

        if current_race_no is None:
            continue

        if line.strip().startswith("-"):
            separator_count += 1
            collecting_entries = separator_count >= 2
            continue

        if not collecting_entries or collected_for_race >= 6:
            continue

        entry_match = ENTRY_LINE_RE.match(normalized)
        if not entry_match:
            continue

        meet_results, early_hint = split_meet_results(entry_match.group("tail"))
        raw_name = clean_name(entry_match.group("racer_name"))
        race_id = build_race_id(race_date, current_section_code or venue, current_race_no)
        entries.append(
            RaceEntry(
                race_id=race_id,
                race_date=race_date,
                venue=venue,
                race_no=current_race_no,
                race_title=race_title,
                leg_type=clean_text(current_leg_type) or "",
                distance_m=current_distance,
                bet_type=current_bet_type,
                lane=int(entry_match.group("lane")),
                racer_id=int(entry_match.group("racer_id")),
                racer_name=raw_name,
                age=to_int(entry_match.group("age")),
                branch=clean_text(entry_match.group("branch")),
                weight=to_int(entry_match.group("weight")),
                class_name=entry_match.group("class_name"),
                national_win_rate=to_float(entry_match.group("national_win")),
                national_place_rate=to_float(entry_match.group("national_place")),
                local_win_rate=to_float(entry_match.group("local_win")),
                local_place_rate=to_float(entry_match.group("local_place")),
                motor_no=to_int(entry_match.group("motor_no")),
                motor_place_rate=to_float(entry_match.group("motor_place")),
                boat_no=to_int(entry_match.group("boat_no")),
                boat_place_rate=to_float(entry_match.group("boat_place")),
                current_meet_results=meet_results,
                early_lane_hint=early_hint,
            )
        )
        collected_for_race += 1
    return entries


def parse_result_file(path: Path) -> list[RaceResult]:
    lines = load_shift_jis_lines(path)
    return parse_result_lines(lines)


def parse_result_text(text: str) -> list[RaceResult]:
    return parse_result_lines(text.splitlines())


def parse_result_lines(lines: list[str]) -> list[RaceResult]:
    venue = extract_venue(lines)
    race_date = extract_date(lines)
    results: list[RaceResult] = []
    current_race_no: int | None = None
    current_weather: str | None = None
    current_wind_direction: str | None = None
    current_wind_speed: int | None = None
    current_wave: int | None = None
    current_winning_style: str | None = None
    current_section_code: str | None = None
    separator_count = 0
    collecting_results = False
    collected_for_race = 0

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
        header_match = RESULT_HEADER_RE.match(normalized)
        if header_match:
            current_race_no = int(header_match.group("race_no"))
            current_weather = clean_text(header_match.group("weather"))
            current_wind_direction = clean_text(header_match.group("wind_direction"))
            current_wind_speed = to_int(header_match.group("wind_speed"))
            current_wave = to_int(header_match.group("wave"))
            current_winning_style = None
            separator_count = 0
            collecting_results = False
            collected_for_race = 0
            continue

        style_match = WINNING_STYLE_RE.search(normalized)
        if style_match:
            current_winning_style = clean_text(style_match.group("style"))
            continue

        table_style = parse_winning_style_from_result_table_header(normalized)
        if table_style:
            current_winning_style = table_style
            continue

        if current_race_no is None:
            continue

        race_id = build_race_id(race_date, current_section_code or venue, current_race_no)

        if line.strip().startswith("-"):
            separator_count += 1
            collecting_results = separator_count >= 1
            continue

        if not collecting_results or collected_for_race >= 6:
            continue

        result_match = RESULT_LINE_RE.match(normalized)
        if not result_match:
            continue

        finish_position, finish_status = parse_finish_token(result_match.group("finish"))
        exhibition_time, course, start_timing, race_time = parse_result_tail(result_match.group("tail"))
        results.append(
            RaceResult(
                race_id=race_id,
                race_date=race_date,
                venue=venue,
                race_no=current_race_no,
                lane=int(result_match.group("lane")),
                finish_position=finish_position,
                finish_status=finish_status,
                racer_id=int(result_match.group("racer_id")),
                racer_name=clean_name(result_match.group("racer_name")),
                motor_no=to_int(result_match.group("motor_no")),
                boat_no=to_int(result_match.group("boat_no")),
                exhibition_time=exhibition_time,
                course=course,
                start_timing=start_timing,
                race_time=race_time,
                weather=current_weather,
                wind_direction=current_wind_direction,
                wind_speed_m=current_wind_speed,
                wave_cm=current_wave,
                winning_style=current_winning_style,
            )
        )
        collected_for_race += 1
    return results


def extract_venue(lines: list[str]) -> str:
    for line in lines[:10]:
        text = clean_text(line)
        if not text:
            continue
        if "競艇場" in text:
            return text.split("競艇場")[0].replace(" ", "").strip()
        if "競走場" in text:
            return text.split("競走場")[0].replace(" ", "").strip()
        if "ボートレース" in text:
            return text.split("ボートレース")[0].replace(" ", "").strip()
        if "［成績］" in text or "[成績]" in text:
            text = text.replace("[成績]", "［成績］")
            return text.split("［成績］")[0].replace(" ", "").strip()
    raise ValueError("Venue not found in file header.")


def extract_date(lines: list[str]) -> date:
    for line in lines[:20]:
        parsed = parse_date_from_line(line)
        if parsed:
            return parsed
    raise ValueError("Race date not found in file header.")


def extract_race_title(lines: list[str]) -> str:
    for line in lines[3:8]:
        title = parse_race_title_from_line(line)
        if title:
            return title
    return ""


def build_race_id(race_date: date, venue_key: str, race_no: int) -> str:
    return f"{race_date.isoformat()}_{venue_key}_{race_no:02d}"


def parse_race_time(text: str) -> float | None:
    cleaned = clean_text(text)
    if not cleaned or cleaned.startswith("."):
        return None
    if RACE_TIME_RE.match(cleaned):
        parts = cleaned.split(".")
        minutes = int(parts[0])
        seconds = int(parts[1])
        decimals = int(parts[2])
        return minutes * 60 + seconds + decimals / 10
    try:
        return float(cleaned)
    except ValueError:
        return None


def clean_text(value: str | None) -> str | None:
    if value is None:
        return None
    collapsed = " ".join(value.replace(FULLWIDTH_SPACE, " ").split())
    return collapsed or None


def clean_name(value: str) -> str:
    cleaned = clean_text(value)
    return (cleaned or "").replace(" ", "")


def to_int(value: str | None) -> int | None:
    if value is None:
        return None
    value = clean_text(value)
    if not value:
        return None
    return int(value)


def to_float(value: str | None) -> float | None:
    if value is None:
        return None
    value = clean_text(value)
    if not value:
        return None
    return float(value)


def parse_winning_style_from_result_table_header(line: str) -> str | None:
    normalized = clean_text(line)
    marker = None
    for candidate in ("レースタイム", "ﾚｰｽﾀｲﾑ"):
        if normalized and candidate in normalized:
            marker = candidate
            break
    if not normalized or marker is None:
        return None
    tail = normalized.split(marker, 1)[-1].strip()
    if not tail:
        return None
    for style in sorted(KNOWN_WINNING_STYLES, key=len, reverse=True):
        if tail.startswith(style):
            return style
    return None


def split_meet_results(tail: str | None) -> tuple[str | None, int | None]:
    cleaned = clean_text(tail)
    if not cleaned:
        return None, None

    parts = cleaned.split()
    if len(parts) == 1:
        return parts[0], None

    if parts[-1].isdigit():
        return " ".join(parts[:-1]) or None, int(parts[-1])
    return cleaned, None


def normalize_year(raw_year: str) -> int:
    year = int(raw_year)
    if year < 100:
        return 1900 + year if year >= 80 else 2000 + year
    return year


def parse_finish_token(token: str) -> tuple[int, str | None]:
    cleaned = clean_text(token) or ""
    if cleaned.isdigit():
        return int(cleaned), None
    return 6, cleaned


def parse_result_tail(tail: str) -> tuple[float | None, int | None, float | None, float | None]:
    tokens = (clean_text(tail) or "").split()
    if not tokens:
        return None, None, None, None

    course_index = next((i for i, token in enumerate(tokens) if token.isdigit()), None)
    exhibition = None
    if course_index is not None:
        for token in tokens[:course_index]:
            exhibition = parse_optional_float_token(token)
            if exhibition is not None:
                break
        course = int(tokens[course_index])
        start = None
        start_index = None
        for i, token in enumerate(tokens[course_index + 1 :], start=course_index + 1):
            start = parse_optional_float_token(token)
            if start is not None:
                start_index = i
                break
        race_tokens = tokens[start_index + 1 :] if start_index is not None else []
    else:
        course = None
        start = None
        race_tokens = []

    race_time = parse_race_time(" ".join(race_tokens)) if race_tokens else None
    return exhibition, course, start, race_time


def parse_optional_float_token(token: str) -> float | None:
    cleaned = clean_text(token)
    if not cleaned or cleaned == ".":
        return None
    if cleaned and cleaned[0].isalpha():
        cleaned = cleaned[1:]
    if not cleaned or cleaned == ".":
        return None
    return float(cleaned)


def parse_venue_from_line(line: str) -> str | None:
    text = clean_text(line)
    if not text:
        return None
    text = text.replace("[成績]", "［成績］")
    result_match = VENUE_RESULT_RE.search(text)
    if result_match:
        return result_match.group("venue")
    suffix_match = VENUE_SUFFIX_RE.search(text)
    if suffix_match:
        return suffix_match.group("venue")
    return None


def parse_date_from_line(line: str) -> date | None:
    if not line:
        return None
    jp = DATE_JP_RE.search(line)
    if jp:
        return date(normalize_year(jp.group("year")), int(jp.group("month")), int(jp.group("day")))
    slash = DATE_SLASH_RE.search(line)
    if slash:
        return date(
            normalize_year(slash.group("year")),
            int(slash.group("month")),
            int(slash.group("day")),
        )
    return None


def parse_race_title_from_line(line: str) -> str | None:
    text = clean_text(line)
    if not text:
        return None
    if "＊＊＊" in text or "第" in text or "競艇場" in text or "競走場" in text or "成績" in text:
        return None
    if "レース" in text or "競走" in text:
        return text
    return None
