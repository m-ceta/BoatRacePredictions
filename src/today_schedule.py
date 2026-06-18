from __future__ import annotations

import re
import subprocess
import tempfile
from datetime import date, datetime, time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

try:
    import lhafile
except ImportError:  # pragma: no cover
    lhafile = None


SECTION_BBGN_RE = re.compile(r"^(?P<section_code>\d{2})BBGN$")
ENTRY_DEADLINE_RE = re.compile(
    r"^\s*(?P<race_no>[0-9０-９]{1,2})\s*R?.*?電話投票締切予定時刻\s*(?P<hour>[0-9０-９]{1,2})[:：](?P<minute>[0-9０-９]{2})"
)
FULLWIDTH_DIGIT_TRANS = str.maketrans("０１２３４５６７８９：", "0123456789:")
JST = ZoneInfo("Asia/Tokyo")


def current_jst_datetime() -> datetime:
    return datetime.now(JST)


def current_jst_date() -> date:
    return current_jst_datetime().date()


def fetch_daily_race_schedule(target_date: date | None = None) -> dict[str, dict[int, time]]:
    schedule_date = target_date or current_jst_date()
    program_text = fetch_mbrace_program_text(schedule_date)
    return parse_program_race_schedule(program_text)


def parse_program_race_schedule(program_text: str) -> dict[str, dict[int, time]]:
    schedule: dict[str, dict[int, time]] = {}
    current_section_code: str | None = None
    for raw_line in program_text.splitlines():
        line = raw_line.translate(FULLWIDTH_DIGIT_TRANS)
        section_match = SECTION_BBGN_RE.match(line.strip())
        if section_match:
            current_section_code = section_match.group("section_code")
            schedule.setdefault(current_section_code, {})
            continue

        if current_section_code is None:
            continue

        deadline_match = ENTRY_DEADLINE_RE.match(line)
        if deadline_match is None:
            continue

        race_no = int(deadline_match.group("race_no"))
        hour = int(deadline_match.group("hour"))
        minute = int(deadline_match.group("minute"))
        schedule.setdefault(current_section_code, {})[race_no] = time(hour=hour, minute=minute)
    return {venue: races for venue, races in schedule.items() if races}


def choose_default_today_venue(
    schedule: dict[str, dict[int, time]],
    preferred_venue: str = "15",
) -> str:
    available = sorted(schedule.keys())
    if not available:
        return preferred_venue
    if preferred_venue in schedule:
        return preferred_venue
    return available[0]


def choose_default_today_race_no(
    schedule: dict[str, dict[int, time]],
    venue_code: str,
    now: datetime | None = None,
) -> int:
    race_schedule = schedule.get(venue_code, {})
    if not race_schedule:
        return 12

    now_time = _normalize_now(now).time()
    future_races = sorted(race_no for race_no, deadline in race_schedule.items() if deadline >= now_time)
    if future_races:
        return future_races[0]
    return sorted(race_schedule.keys())[-1]


def filter_future_schedule(
    schedule: dict[str, dict[int, time]],
    now: datetime | None = None,
) -> dict[str, dict[int, time]]:
    now_time = _normalize_now(now).time()
    filtered: dict[str, dict[int, time]] = {}
    for venue_code, race_map in schedule.items():
        future_races = {
            race_no: deadline
            for race_no, deadline in race_map.items()
            if deadline >= now_time
        }
        if future_races:
            filtered[venue_code] = future_races
    return filtered


def _normalize_now(now: datetime | None) -> datetime:
    if now is None:
        return current_jst_datetime()
    if now.tzinfo is None:
        return now.replace(tzinfo=JST)
    return now.astimezone(JST)


def fetch_mbrace_program_text(target_date: date) -> str:
    archive_bytes = fetch_mbrace_daily_archive(target_date, kind="B")
    return extract_lzh_text(archive_bytes)


def fetch_mbrace_daily_archive(target_date: date, kind: str) -> bytes:
    kind = kind.upper()
    yyyy_mm = target_date.strftime("%Y%m")
    yy_mm_dd = target_date.strftime("%y%m%d")
    url = f"https://www1.mbrace.or.jp/od2/{kind}/{yyyy_mm}/{kind.lower()}{yy_mm_dd}.lzh"
    try:
        return fetch_bytes(url, referer=f"https://www1.mbrace.or.jp/od2/{kind}/{yyyy_mm}/mday.html")
    except HTTPError as exc:
        if exc.code == 404:
            raise ValueError(f"Mbrace schedule archive not found for {target_date.isoformat()}: {url}") from exc
        raise


def fetch_bytes(url: str, referer: str | None = None) -> bytes:
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
        return resp.read()


def extract_lzh_text(archive_bytes: bytes) -> str:
    extracted = extract_lzh_entries(archive_bytes)
    for _, data in extracted:
        for encoding in ("cp932", "shift_jis", "utf-8", "latin1"):
            try:
                text = data.decode(encoding)
            except UnicodeDecodeError:
                continue
            if "競艇" not in text and "電話投票" not in text and "番組表" not in text:
                continue
            return text
    return extracted[0][1].decode("cp932", errors="ignore")


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


def find_seven_zip() -> str | None:
    candidates = [
        Path(r"C:\Program Files\7-Zip\7z.exe"),
        Path(r"C:\Program Files\7-Zip\7z.EXE"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None
