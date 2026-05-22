from __future__ import annotations

from datetime import date

from src.rowdata_sync import (
    existing_rowdata_dates,
    infer_backfill_range,
    normalize_row_kinds,
    rowdata_file_path,
)


def test_normalize_row_kinds_accepts_string() -> None:
    assert normalize_row_kinds("bk") == ("B", "K")


def test_rowdata_file_path_formats_name(tmp_path) -> None:
    path = rowdata_file_path(tmp_path, "b", date(2026, 5, 22))
    assert path.name == "B260522.TXT"


def test_existing_rowdata_dates_reads_yyMMdd(tmp_path) -> None:
    (tmp_path / "B260520.TXT").write_text("dummy", encoding="utf-8")
    (tmp_path / "B260521.TXT").write_text("dummy", encoding="utf-8")
    assert existing_rowdata_dates(tmp_path, "B") == {date(2026, 5, 20), date(2026, 5, 21)}


def test_infer_backfill_range_defaults_to_next_day(tmp_path) -> None:
    (tmp_path / "B260520.TXT").write_text("dummy", encoding="utf-8")
    (tmp_path / "K260520.TXT").write_text("dummy", encoding="utf-8")
    start_date, end_date = infer_backfill_range(
        rowdata_dir=tmp_path,
        kinds=("B", "K"),
        end_date=date(2026, 5, 22),
    )
    assert start_date == date(2026, 5, 21)
    assert end_date == date(2026, 5, 22)
