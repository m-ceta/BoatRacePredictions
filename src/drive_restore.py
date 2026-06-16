from __future__ import annotations

import os
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import parse_qs, urlparse

import requests


DEFAULT_ROWDATA_DRIVE_FILE_URL = os.environ.get(
    "BOATRACE_ROWDATA_DRIVE_FILE_URL",
    "https://drive.google.com/file/d/1mtjumyk9k43UlGa7c2URfAZmUFgt_9En/view?usp=drive_link",
)
DEFAULT_DATA_DRIVE_FILE_URL = os.environ.get(
    "BOATRACE_DATA_DRIVE_FILE_URL",
    "",
)
DEFAULT_ARTIFACTS_DRIVE_FILE_URL = os.environ.get(
    "BOATRACE_ARTIFACTS_DRIVE_FILE_URL",
    "",
)
LEGACY_BRP_DRIVE_FILE_URL = os.environ.get(
    "BOATRACE_BRP_DRIVE_FILE_URL",
    "https://drive.google.com/file/d/14w8W6xqi-NmnePs7waYhrUrxYD378YHq/view?usp=drive_link",
)


@dataclass(slots=True)
class DriveRestoreReport:
    rowdata_zip: Path | None
    data_zip: Path | None
    artifacts_zip: Path | None
    restored_targets: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "rowdata_zip": str(self.rowdata_zip) if self.rowdata_zip is not None else None,
            "data_zip": str(self.data_zip) if self.data_zip is not None else None,
            "artifacts_zip": str(self.artifacts_zip) if self.artifacts_zip is not None else None,
            "restored_targets": self.restored_targets,
        }


def extract_drive_file_id(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme and parsed.netloc:
        if "/file/d/" in parsed.path:
            return parsed.path.rstrip("/").split("/file/d/")[-1].split("/")[0]
        query = parse_qs(parsed.query)
        if "id" in query and query["id"]:
            return query["id"][0]
    return value.strip()


def _download_drive_file(file_id: str, output_path: Path) -> Path:
    session = requests.Session()
    base_url = "https://drive.google.com/uc"
    params = {"export": "download", "id": file_id}
    response = session.get(base_url, params=params, stream=True, timeout=120)
    response.raise_for_status()

    confirm_token = None
    for cookie_key, cookie_value in response.cookies.items():
        if cookie_key.startswith("download_warning"):
            confirm_token = cookie_value
            break
    if confirm_token:
        response.close()
        response = session.get(
            base_url,
            params={**params, "confirm": confirm_token},
            stream=True,
            timeout=120,
        )
        response.raise_for_status()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as fh:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                fh.write(chunk)
    response.close()
    return output_path


def _replace_directory_from_extracted(extracted_root: Path, source_name: str, destination_dir: Path) -> None:
    source_dir = extracted_root / source_name
    if not source_dir.exists():
        raise FileNotFoundError(f"Archive does not contain '{source_name}/'.")
    if destination_dir.exists():
        shutil.rmtree(destination_dir)
    shutil.copytree(source_dir, destination_dir)


def _extract_zip_to_temp(zip_path: Path, temp_dir: Path) -> Path:
    extract_root = temp_dir / zip_path.stem
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_root)
    return extract_root


def _download_unique_archives(
    root: Path,
    requested: list[tuple[str, str, str]],
) -> dict[str, Path]:
    downloaded: dict[str, Path] = {}
    for _, url, zip_name in requested:
        if not url:
            continue
        file_id = extract_drive_file_id(url)
        if file_id in downloaded:
            continue
        downloaded[file_id] = _download_drive_file(file_id, root / zip_name)
    return downloaded


def download_and_restore_packages(
    project_root: Path,
    rowdata_drive_file_url: str = DEFAULT_ROWDATA_DRIVE_FILE_URL,
    data_drive_file_url: str = DEFAULT_DATA_DRIVE_FILE_URL,
    artifacts_drive_file_url: str = DEFAULT_ARTIFACTS_DRIVE_FILE_URL,
    rowdata_zip_name: str = "rowdata.zip",
    data_zip_name: str = "data.zip",
    artifacts_zip_name: str = "artifacts.zip",
    restore_rowdata: bool = True,
    restore_data: bool = True,
    restore_artifacts: bool = True,
) -> DriveRestoreReport:
    root = project_root.resolve()

    data_url = data_drive_file_url or LEGACY_BRP_DRIVE_FILE_URL
    artifacts_url = artifacts_drive_file_url or LEGACY_BRP_DRIVE_FILE_URL

    requests_to_download: list[tuple[str, str, str]] = []
    if restore_rowdata:
        requests_to_download.append(("rowdata", rowdata_drive_file_url, rowdata_zip_name))
    if restore_data:
        requests_to_download.append(("data", data_url, data_zip_name))
    if restore_artifacts:
        requests_to_download.append(("artifacts", artifacts_url, artifacts_zip_name))

    for target_name, url, _ in requests_to_download:
        if not url:
            raise ValueError(f"{target_name} の Google Drive 共有リンクが指定されていません。")

    downloads = _download_unique_archives(root, requests_to_download)

    restored_targets: list[str] = []
    rowdata_zip = root / rowdata_zip_name if restore_rowdata else None
    data_zip = root / data_zip_name if restore_data else None
    artifacts_zip = root / artifacts_zip_name if restore_artifacts else None

    with TemporaryDirectory(dir=root) as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        if restore_rowdata:
            rowdata_extract_root = _extract_zip_to_temp(
                downloads[extract_drive_file_id(rowdata_drive_file_url)],
                temp_dir,
            )
            _replace_directory_from_extracted(rowdata_extract_root, "rowdata", root / "rowdata")
            restored_targets.append("rowdata")

        if restore_data:
            data_extract_root = _extract_zip_to_temp(
                downloads[extract_drive_file_id(data_url)],
                temp_dir,
            )
            _replace_directory_from_extracted(data_extract_root, "data", root / "data")
            restored_targets.append("data")

        if restore_artifacts:
            artifacts_extract_root = _extract_zip_to_temp(
                downloads[extract_drive_file_id(artifacts_url)],
                temp_dir,
            )
            _replace_directory_from_extracted(artifacts_extract_root, "artifacts", root / "artifacts")
            restored_targets.append("artifacts")

    return DriveRestoreReport(
        rowdata_zip=rowdata_zip,
        data_zip=data_zip,
        artifacts_zip=artifacts_zip,
        restored_targets=restored_targets,
    )
