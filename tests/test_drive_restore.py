from __future__ import annotations

import zipfile
from pathlib import Path

from src import drive_restore


def test_extract_drive_file_id_from_share_link() -> None:
    file_id = drive_restore.extract_drive_file_id(
        "https://drive.google.com/file/d/14w8W6xqi-NmnePs7waYhrUrxYD378YHq/view?usp=drive_link"
    )
    assert file_id == "14w8W6xqi-NmnePs7waYhrUrxYD378YHq"


def test_download_and_restore_packages_replaces_target_directories(tmp_path, monkeypatch) -> None:
    project_root = tmp_path
    (project_root / "rowdata").mkdir()
    (project_root / "data").mkdir()
    (project_root / "artifacts").mkdir()
    (project_root / "rowdata" / "old.txt").write_text("old", encoding="utf-8")
    (project_root / "data" / "old.txt").write_text("old", encoding="utf-8")
    (project_root / "artifacts" / "old.txt").write_text("old", encoding="utf-8")

    source_root = tmp_path / "source_archives"
    source_root.mkdir()
    rowdata_zip = source_root / "rowdata.zip"
    brp_zip = source_root / "brp.zip"

    with zipfile.ZipFile(rowdata_zip, "w") as zf:
        zf.writestr("rowdata/new_row.txt", "row")
    with zipfile.ZipFile(brp_zip, "w") as zf:
        zf.writestr("data/new_data.txt", "data")
        zf.writestr("artifacts/new_artifact.txt", "artifact")

    mapping = {
        "row-file-id": rowdata_zip,
        "brp-file-id": brp_zip,
    }

    def fake_download(file_id: str, output_path: Path) -> Path:
        output_path.write_bytes(mapping[file_id].read_bytes())
        return output_path

    monkeypatch.setattr(drive_restore, "_download_drive_file", fake_download)

    report = drive_restore.download_and_restore_packages(
        project_root=project_root,
        brp_drive_file_url="brp-file-id",
        rowdata_drive_file_url="row-file-id",
    )

    assert report.restored_targets == ["rowdata", "data", "artifacts"]
    assert (project_root / "rowdata" / "new_row.txt").read_text(encoding="utf-8") == "row"
    assert (project_root / "data" / "new_data.txt").read_text(encoding="utf-8") == "data"
    assert (project_root / "artifacts" / "new_artifact.txt").read_text(encoding="utf-8") == "artifact"
    assert not (project_root / "rowdata" / "old.txt").exists()
    assert not (project_root / "data" / "old.txt").exists()
    assert not (project_root / "artifacts" / "old.txt").exists()
