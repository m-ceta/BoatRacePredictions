from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.cli import resolve_build_min_date
from src.features.streaming_builder import iter_month_groups


def touch_rowdata_pair(rowdata_dir: Path, yymmdd: str) -> None:
    (rowdata_dir / f"B{yymmdd}.TXT").write_text("", encoding="utf-8")
    (rowdata_dir / f"K{yymmdd}.TXT").write_text("", encoding="utf-8")


def test_resolve_build_min_date_defaults_to_rolling_years_plus_warmup(tmp_path: Path) -> None:
    rowdata_dir = tmp_path / "rowdata"
    rowdata_dir.mkdir()
    touch_rowdata_pair(rowdata_dir, "260723")
    config_path = tmp_path / "train.yaml"
    config_path.write_text(yaml.safe_dump({"data": {"rolling_years": 3.5}}), encoding="utf-8")

    resolved = resolve_build_min_date(rowdata_dir, "auto", None, config_path)

    expected = (pd.Timestamp("2026-07-23") - pd.Timedelta(days=int((3.5 + 2.0) * 365.25 + 0.9999))).date()
    assert resolved == expected.isoformat()


def test_resolve_build_min_date_can_be_disabled(tmp_path: Path) -> None:
    rowdata_dir = tmp_path / "rowdata"
    rowdata_dir.mkdir()
    touch_rowdata_pair(rowdata_dir, "260723")

    assert resolve_build_min_date(rowdata_dir, "none", None, tmp_path / "missing.yaml") is None


def test_iter_month_groups_filters_min_date(tmp_path: Path) -> None:
    rowdata_dir = tmp_path / "rowdata"
    rowdata_dir.mkdir()
    touch_rowdata_pair(rowdata_dir, "210101")
    touch_rowdata_pair(rowdata_dir, "260723")

    groups = iter_month_groups(rowdata_dir, min_date=pd.Timestamp("2022-01-01").date())

    assert [month for month, _, _ in groups] == ["202607"]
