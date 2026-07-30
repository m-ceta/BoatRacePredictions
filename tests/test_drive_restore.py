from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

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
    data_zip = source_root / "data.zip"
    artifacts_zip = source_root / "artifacts.zip"

    with zipfile.ZipFile(rowdata_zip, "w") as zf:
        zf.writestr("rowdata/new_row.txt", "row")
    with zipfile.ZipFile(data_zip, "w") as zf:
        zf.writestr("data/new_data.txt", "data")
    with zipfile.ZipFile(artifacts_zip, "w") as zf:
        zf.writestr("artifacts/new_artifact.txt", "artifact")

    mapping = {
        "row-file-id": rowdata_zip,
        "data-file-id": data_zip,
        "artifacts-file-id": artifacts_zip,
    }

    def fake_download(file_id: str, output_path: Path) -> Path:
        output_path.write_bytes(mapping[file_id].read_bytes())
        return output_path

    monkeypatch.setattr(drive_restore, "_download_drive_file", fake_download)

    report = drive_restore.download_and_restore_packages(
        project_root=project_root,
        rowdata_drive_file_url="row-file-id",
        data_drive_file_url="data-file-id",
        artifacts_drive_file_url="artifacts-file-id",
    )

    assert report.restored_targets == ["rowdata", "data", "artifacts"]
    assert (project_root / "rowdata" / "new_row.txt").read_text(encoding="utf-8") == "row"
    assert (project_root / "data" / "new_data.txt").read_text(encoding="utf-8") == "data"
    assert (project_root / "artifacts" / "new_artifact.txt").read_text(encoding="utf-8") == "artifact"
    assert not (project_root / "rowdata" / "old.txt").exists()
    assert not (project_root / "data" / "old.txt").exists()
    assert not (project_root / "artifacts" / "old.txt").exists()


def test_download_and_restore_packages_can_skip_targets(tmp_path, monkeypatch) -> None:
    project_root = tmp_path
    (project_root / "data").mkdir()
    (project_root / "artifacts").mkdir()
    (project_root / "data" / "keep.txt").write_text("keep", encoding="utf-8")

    source_root = tmp_path / "source_archives"
    source_root.mkdir()
    artifacts_zip = source_root / "artifacts.zip"
    with zipfile.ZipFile(artifacts_zip, "w") as zf:
        zf.writestr("artifacts/new_artifact.txt", "artifact")

    def fake_download(file_id: str, output_path: Path) -> Path:
        output_path.write_bytes(artifacts_zip.read_bytes())
        return output_path

    monkeypatch.setattr(drive_restore, "_download_drive_file", fake_download)

    report = drive_restore.download_and_restore_packages(
        project_root=project_root,
        data_drive_file_url="",
        artifacts_drive_file_url="artifacts-file-id",
        restore_rowdata=False,
        restore_data=False,
        restore_artifacts=True,
    )

    assert report.restored_targets == ["artifacts"]
    assert (project_root / "data" / "keep.txt").read_text(encoding="utf-8") == "keep"
    assert (project_root / "artifacts" / "new_artifact.txt").read_text(encoding="utf-8") == "artifact"


def test_restore_packages_from_zip_files_uses_mounted_package_directory(tmp_path) -> None:
    project_root = tmp_path / "project"
    source_dir = tmp_path / "drive"
    project_root.mkdir()
    source_dir.mkdir()
    (project_root / "data").mkdir()
    (project_root / "data" / "old.txt").write_text("old", encoding="utf-8")

    data_zip = source_dir / "data.zip"
    with zipfile.ZipFile(data_zip, "w") as zf:
        zf.writestr("data/new_data.txt", "data")

    report = drive_restore.restore_packages_from_zip_files(
        project_root=project_root,
        source_dir=source_dir,
        restore_rowdata=False,
        restore_data=True,
        restore_artifacts=False,
    )

    assert report.restored_targets == ["data"]
    assert report.data_zip == data_zip
    assert (project_root / "data" / "new_data.txt").read_text(encoding="utf-8") == "data"
    assert not (project_root / "data" / "old.txt").exists()


def test_export_package_archives_writes_top_level_directories(tmp_path) -> None:
    project_root = tmp_path / "project"
    output_dir = tmp_path / "drive"
    (project_root / "data").mkdir(parents=True)
    (project_root / "artifacts").mkdir()
    (project_root / "data" / "processed.txt").write_text("data", encoding="utf-8")
    (project_root / "artifacts" / "model.txt").write_text("model", encoding="utf-8")

    report = drive_restore.export_package_archives(
        project_root=project_root,
        output_dir=output_dir,
        export_rowdata=False,
        export_data=True,
        export_artifacts=True,
    )

    assert report.exported_targets == ["data", "artifacts"]
    assert report.data_zip == output_dir / "data.zip"
    assert report.artifacts_zip == output_dir / "artifacts.zip"
    with zipfile.ZipFile(output_dir / "data.zip") as zf:
        assert zf.read("data/processed.txt").decode("utf-8") == "data"
    with zipfile.ZipFile(output_dir / "artifacts.zip") as zf:
        assert zf.read("artifacts/model.txt").decode("utf-8") == "model"


def test_export_package_archives_removes_unnecessary_data_directories(tmp_path) -> None:
    project_root = tmp_path / "project"
    output_dir = tmp_path / "drive"
    (project_root / "data" / "processed").mkdir(parents=True)
    (project_root / "data" / "data" / "processed").mkdir(parents=True)
    (project_root / "data" / "processed_streaming_tmp_abcd").mkdir()
    (project_root / "data" / "processed" / "training_table.parquet").write_text("current", encoding="utf-8")
    (project_root / "data" / "data" / "processed" / "training_table.parquet").write_text("old", encoding="utf-8")
    (project_root / "data" / "processed_streaming_tmp_abcd" / "partial.parquet").write_text("tmp", encoding="utf-8")

    drive_restore.export_package_archives(
        project_root=project_root,
        output_dir=output_dir,
        export_rowdata=False,
        export_data=True,
        export_artifacts=False,
    )

    assert not (project_root / "data" / "data").exists()
    assert not (project_root / "data" / "processed_streaming_tmp_abcd").exists()
    with zipfile.ZipFile(output_dir / "data.zip") as zf:
        names = set(zf.namelist())
    assert "data/processed/training_table.parquet" in names
    assert "data/data/processed/training_table.parquet" not in names
    assert "data/processed_streaming_tmp_abcd/partial.parquet" not in names


class _FakeResponse:
    def __init__(
        self,
        body: bytes,
        *,
        content_type: str = "application/octet-stream",
        cookies: dict[str, str] | None = None,
        status_code: int = 200,
    ) -> None:
        self._body = body
        self.headers = {"Content-Type": content_type}
        self.cookies = cookies or {}
        self.status_code = status_code
        self.encoding = "utf-8"

    @property
    def text(self) -> str:
        return self._body.decode(self.encoding, errors="replace")

    def iter_content(self, chunk_size: int = 1024 * 1024):
        for index in range(0, len(self._body), chunk_size):
            yield self._body[index : index + chunk_size]

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def close(self) -> None:
        return None


class _FakeSession:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self._responses = responses

    def get(self, *args, **kwargs):
        if not self._responses:
            raise AssertionError("No more fake responses configured")
        return self._responses.pop(0)


def test_download_drive_file_follows_html_confirm_form(tmp_path, monkeypatch) -> None:
    zip_path = tmp_path / "payload.zip"
    zip_bytes_path = tmp_path / "source.zip"
    with zipfile.ZipFile(zip_bytes_path, "w") as zf:
        zf.writestr("data/file.txt", "ok")
    zip_bytes = zip_bytes_path.read_bytes()
    html = b"""
    <html><body>
      <form id="download-form" action="https://drive.usercontent.google.com/download">
        <input type="hidden" name="id" value="file-id">
        <input type="hidden" name="confirm" value="token123">
      </form>
    </body></html>
    """
    session = _FakeSession(
        [
            _FakeResponse(html, content_type="text/html; charset=utf-8"),
            _FakeResponse(zip_bytes),
        ]
    )
    monkeypatch.setattr(drive_restore.requests, "Session", lambda: session)

    result = drive_restore._download_drive_file("file-id", zip_path)

    assert result == zip_path
    assert zipfile.is_zipfile(zip_path)


def test_download_drive_file_raises_helpful_error_for_html_page(tmp_path, monkeypatch) -> None:
    zip_path = tmp_path / "payload.zip"
    html = b"<html><body>Sign in required</body></html>"
    session = _FakeSession([_FakeResponse(html, content_type="text/html; charset=utf-8")])
    monkeypatch.setattr(drive_restore.requests, "Session", lambda: session)

    with pytest.raises(RuntimeError, match="Google Drive returned an HTML page"):
        drive_restore._download_drive_file("file-id", zip_path)
