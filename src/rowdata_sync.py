from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from src.live import extract_lzh_first_file_bytes, fetch_mbrace_daily_archive


SUPPORTED_ROW_KINDS = ("B", "K")


@dataclass(slots=True)
class RowdataBackfillReport:
    rowdata_dir: Path
    start_date: date
    end_date: date
    kinds: tuple[str, ...]
    downloaded_files: list[Path]
    skipped_existing_files: list[Path]
    unavailable_files: list[str]

    @property
    def downloaded_count(self) -> int:
        return len(self.downloaded_files)

    @property
    def skipped_count(self) -> int:
        return len(self.skipped_existing_files)

    @property
    def unavailable_count(self) -> int:
        return len(self.unavailable_files)

    def to_dict(self) -> dict[str, object]:
        return {
            "rowdata_dir": str(self.rowdata_dir),
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "kinds": list(self.kinds),
            "downloaded_files": [str(path) for path in self.downloaded_files],
            "skipped_existing_files": [str(path) for path in self.skipped_existing_files],
            "unavailable_files": list(self.unavailable_files),
            "downloaded_count": self.downloaded_count,
            "skipped_count": self.skipped_count,
            "unavailable_count": self.unavailable_count,
        }


def normalize_row_kinds(kinds: str | tuple[str, ...] | list[str]) -> tuple[str, ...]:
    if isinstance(kinds, str):
        values = [value.upper() for value in kinds if value.strip()]
    else:
        values = [str(value).upper() for value in kinds]
    normalized = tuple(value for value in values if value in SUPPORTED_ROW_KINDS)
    if not normalized:
        raise ValueError("At least one rowdata kind from {'B', 'K'} is required.")
    return normalized


def rowdata_file_path(rowdata_dir: str | Path, kind: str, target_date: date) -> Path:
    directory = Path(rowdata_dir)
    return directory / f"{kind.upper()}{target_date.strftime('%y%m%d')}.TXT"


def existing_rowdata_dates(rowdata_dir: str | Path, kind: str) -> set[date]:
    directory = Path(rowdata_dir)
    dates: set[date] = set()
    for path in directory.glob(f"{kind.upper()}*.TXT"):
        stem = path.stem.upper()
        if len(stem) != 7:
            continue
        token = stem[1:]
        try:
            parsed = date(2000 + int(token[:2]), int(token[2:4]), int(token[4:6]))
        except ValueError:
            continue
        dates.add(parsed)
    return dates


def infer_backfill_range(
    rowdata_dir: str | Path,
    kinds: tuple[str, ...],
    start_date: date | None = None,
    end_date: date | None = None,
) -> tuple[date, date]:
    today = date.today()
    resolved_end = end_date or today
    if start_date is not None:
        return start_date, resolved_end

    existing_dates: set[date] = set()
    for kind in kinds:
        existing_dates.update(existing_rowdata_dates(rowdata_dir, kind))
    if not existing_dates:
        return resolved_end, resolved_end
    inferred_start = max(existing_dates) + timedelta(days=1)
    return inferred_start, resolved_end


def iter_dates(start_date: date, end_date: date) -> list[date]:
    if start_date > end_date:
        return []
    current = start_date
    dates: list[date] = []
    while current <= end_date:
        dates.append(current)
        current += timedelta(days=1)
    return dates


def download_rowdata_file(
    rowdata_dir: str | Path,
    target_date: date,
    kind: str,
    overwrite: bool = False,
) -> tuple[str, Path]:
    destination = rowdata_file_path(rowdata_dir, kind, target_date)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        return "skipped", destination

    archive_bytes = fetch_mbrace_daily_archive(target_date, kind)
    raw_bytes = extract_lzh_first_file_bytes(archive_bytes)
    destination.write_bytes(raw_bytes)
    return "downloaded", destination


def backfill_rowdata(
    rowdata_dir: str | Path,
    start_date: date | None = None,
    end_date: date | None = None,
    kinds: str | tuple[str, ...] | list[str] = "BK",
    overwrite: bool = False,
) -> RowdataBackfillReport:
    normalized_kinds = normalize_row_kinds(kinds)
    resolved_start, resolved_end = infer_backfill_range(
        rowdata_dir=rowdata_dir,
        kinds=normalized_kinds,
        start_date=start_date,
        end_date=end_date,
    )
    downloaded_files: list[Path] = []
    skipped_files: list[Path] = []
    unavailable_files: list[str] = []

    for target_date in iter_dates(resolved_start, resolved_end):
        for kind in normalized_kinds:
            try:
                status, path = download_rowdata_file(
                    rowdata_dir=rowdata_dir,
                    target_date=target_date,
                    kind=kind,
                    overwrite=overwrite,
                )
            except ValueError:
                unavailable_files.append(f"{kind}{target_date.strftime('%y%m%d')}")
                continue
            if status == "downloaded":
                downloaded_files.append(path)
            else:
                skipped_files.append(path)

    return RowdataBackfillReport(
        rowdata_dir=Path(rowdata_dir),
        start_date=resolved_start,
        end_date=resolved_end,
        kinds=normalized_kinds,
        downloaded_files=downloaded_files,
        skipped_existing_files=skipped_files,
        unavailable_files=unavailable_files,
    )
