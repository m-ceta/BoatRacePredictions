from __future__ import annotations

from datetime import date, datetime

from src.rowdata_sync import (
    existing_rowdata_dates,
    infer_backfill_range,
    infer_default_backfill_end_date,
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


def test_infer_default_backfill_end_date_uses_yesterday_during_daytime() -> None:
    assert infer_default_backfill_end_date(datetime(2026, 6, 4, 7, 0, 0)) == date(2026, 6, 3)
    assert infer_default_backfill_end_date(datetime(2026, 6, 4, 20, 59, 59)) == date(2026, 6, 3)


def test_infer_default_backfill_end_date_uses_today_outside_daytime() -> None:
    assert infer_default_backfill_end_date(datetime(2026, 6, 4, 6, 59, 59)) == date(2026, 6, 4)
    assert infer_default_backfill_end_date(datetime(2026, 6, 4, 21, 0, 0)) == date(2026, 6, 4)
