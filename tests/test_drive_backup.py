from __future__ import annotations

import zipfile
from pathlib import Path

from src.drive_backup import create_zip_from_directories, create_zip_from_directory, extract_drive_folder_id


def test_extract_drive_folder_id_from_url() -> None:
    url = "https://drive.google.com/drive/folders/19HHxA5r4T_IqMrDyNqU3qRUoRkhT87OL?usp=drive_link"
    assert extract_drive_folder_id(url) == "19HHxA5r4T_IqMrDyNqU3qRUoRkhT87OL"


def test_create_zip_from_directory(tmp_path: Path) -> None:
    source = tmp_path / "rowdata"
    source.mkdir()
    (source / "B260524.TXT").write_text("sample", encoding="utf-8")
    output = tmp_path / "rowdata.zip"

    create_zip_from_directory(source, output)

    with zipfile.ZipFile(output) as zf:
        assert "rowdata/B260524.TXT" in zf.namelist()


def test_create_zip_from_directories(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    configs = tmp_path / "configs"
    data = tmp_path / "data"
    artifacts.mkdir()
    configs.mkdir()
    data.mkdir()
    (artifacts / "metrics.json").write_text("{}", encoding="utf-8")
    (configs / "train.yaml").write_text("model: {}", encoding="utf-8")
    (data / "sample.txt").write_text("x", encoding="utf-8")
    output = tmp_path / "drp.zip"

    create_zip_from_directories([artifacts, configs, data], output)

    with zipfile.ZipFile(output) as zf:
        names = set(zf.namelist())
        assert "artifacts/metrics.json" in names
        assert "configs/train.yaml" in names
        assert "data/sample.txt" in names
