from __future__ import annotations

import os
import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import parse_qs, urljoin, urlparse

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

_HTML_FORM_ACTION_RE = re.compile(
    r"""<form[^>]+id=["']download-form["'][^>]+action=["']([^"']+)["']""",
    re.IGNORECASE,
)
_HTML_INPUT_RE = re.compile(
    r"""<input[^>]+type=["']hidden["'][^>]+name=["']([^"']+)["'][^>]+value=["']([^"']*)["']""",
    re.IGNORECASE,
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


@dataclass(slots=True)
class DrivePackageExportReport:
    rowdata_zip: Path | None
    data_zip: Path | None
    artifacts_zip: Path | None
    exported_targets: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "rowdata_zip": str(self.rowdata_zip) if self.rowdata_zip is not None else None,
            "data_zip": str(self.data_zip) if self.data_zip is not None else None,
            "artifacts_zip": str(self.artifacts_zip) if self.artifacts_zip is not None else None,
            "exported_targets": self.exported_targets,
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


def _extract_confirm_form(html: str, base_url: str) -> tuple[str, dict[str, str]] | None:
    action_match = _HTML_FORM_ACTION_RE.search(html)
    if not action_match:
        return None
    action_url = urljoin(base_url, action_match.group(1))
    params = {name: value for name, value in _HTML_INPUT_RE.findall(html)}
    return action_url, params


def _read_response_text(response: requests.Response) -> str:
    response.encoding = response.encoding or "utf-8"
    return response.text


def _validate_zip_file(zip_path: Path) -> None:
    if zipfile.is_zipfile(zip_path):
        return
    snippet = zip_path.read_bytes()[:512].decode("utf-8", errors="replace")
    raise RuntimeError(
        "Downloaded file is not a zip archive. "
        "Google Drive may have returned an HTML page instead of the file. "
        "Check that the file is shared publicly and downloadable without login. "
        f"Downloaded content starts with: {snippet[:200]!r}"
    )


def _stream_response_to_file(response: requests.Response, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as fh:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                fh.write(chunk)
    return output_path


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

    content_type = response.headers.get("Content-Type", "")
    if "text/html" in content_type.lower():
        html = _read_response_text(response)
        response.close()
        confirm_form = _extract_confirm_form(html, base_url)
        if confirm_form is not None:
            action_url, form_params = confirm_form
            response = session.get(action_url, params=form_params, stream=True, timeout=120)
            response.raise_for_status()
        else:
            raise RuntimeError(
                "Google Drive returned an HTML page instead of the requested zip file. "
                "The file may require login, may not be shared publicly, or may be rate-limited."
            )

    _stream_response_to_file(response, output_path)
    response.close()
    _validate_zip_file(output_path)
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


def _resolve_zip_path(source_dir: Path, zip_path: Path | None, zip_name: str) -> Path:
    return zip_path if zip_path is not None else source_dir / zip_name


def _restore_target_from_zip(
    *,
    zip_path: Path,
    target_name: str,
    project_root: Path,
    temp_dir: Path,
) -> None:
    if not zip_path.exists():
        raise FileNotFoundError(f"{target_name} zip file was not found: {zip_path}")
    _validate_zip_file(zip_path)
    extract_root = _extract_zip_to_temp(zip_path, temp_dir)
    _replace_directory_from_extracted(extract_root, target_name, project_root / target_name)


def _zip_directory(source_dir: Path, zip_path: Path) -> Path:
    if not source_dir.exists() or not source_dir.is_dir():
        raise FileNotFoundError(f"Package source directory was not found: {source_dir}")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = zip_path.with_name(f"{zip_path.name}.tmp")
    if temp_path.exists():
        temp_path.unlink()
    with zipfile.ZipFile(temp_path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(source_dir.parent))
    temp_path.replace(zip_path)
    return zip_path


def cleanup_data_directory_before_export(data_dir: Path) -> list[Path]:
    if not data_dir.exists() or not data_dir.is_dir():
        return []
    data_root = data_dir.resolve()
    candidates = [
        data_root / "data",
        *data_root.glob("*_streaming_tmp_*"),
    ]
    removed: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if not resolved.exists() or not resolved.is_dir():
            continue
        if resolved == data_root or data_root not in resolved.parents:
            raise ValueError(f"Refusing to remove path outside data directory: {resolved}")
        shutil.rmtree(resolved)
        removed.append(resolved)
    return removed


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


def restore_packages_from_zip_files(
    project_root: Path,
    source_dir: Path | None = None,
    rowdata_zip_path: Path | None = None,
    data_zip_path: Path | None = None,
    artifacts_zip_path: Path | None = None,
    rowdata_zip_name: str = "rowdata.zip",
    data_zip_name: str = "data.zip",
    artifacts_zip_name: str = "artifacts.zip",
    restore_rowdata: bool = True,
    restore_data: bool = True,
    restore_artifacts: bool = True,
) -> DriveRestoreReport:
    root = project_root.resolve()
    source = (source_dir or root).resolve()
    rowdata_zip = _resolve_zip_path(source, rowdata_zip_path, rowdata_zip_name) if restore_rowdata else None
    data_zip = _resolve_zip_path(source, data_zip_path, data_zip_name) if restore_data else None
    artifacts_zip = _resolve_zip_path(source, artifacts_zip_path, artifacts_zip_name) if restore_artifacts else None

    restored_targets: list[str] = []
    with TemporaryDirectory(dir=root) as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        if restore_rowdata and rowdata_zip is not None:
            _restore_target_from_zip(
                zip_path=rowdata_zip,
                target_name="rowdata",
                project_root=root,
                temp_dir=temp_dir,
            )
            restored_targets.append("rowdata")
        if restore_data and data_zip is not None:
            _restore_target_from_zip(
                zip_path=data_zip,
                target_name="data",
                project_root=root,
                temp_dir=temp_dir,
            )
            restored_targets.append("data")
        if restore_artifacts and artifacts_zip is not None:
            _restore_target_from_zip(
                zip_path=artifacts_zip,
                target_name="artifacts",
                project_root=root,
                temp_dir=temp_dir,
            )
            restored_targets.append("artifacts")

    return DriveRestoreReport(
        rowdata_zip=rowdata_zip,
        data_zip=data_zip,
        artifacts_zip=artifacts_zip,
        restored_targets=restored_targets,
    )


def export_package_archives(
    project_root: Path,
    output_dir: Path,
    rowdata_zip_name: str = "rowdata.zip",
    data_zip_name: str = "data.zip",
    artifacts_zip_name: str = "artifacts.zip",
    export_rowdata: bool = True,
    export_data: bool = True,
    export_artifacts: bool = True,
) -> DrivePackageExportReport:
    root = project_root.resolve()
    output = output_dir.resolve()
    exported_targets: list[str] = []
    rowdata_zip = output / rowdata_zip_name if export_rowdata else None
    data_zip = output / data_zip_name if export_data else None
    artifacts_zip = output / artifacts_zip_name if export_artifacts else None

    if export_rowdata and rowdata_zip is not None:
        _zip_directory(root / "rowdata", rowdata_zip)
        exported_targets.append("rowdata")
    if export_data and data_zip is not None:
        cleanup_data_directory_before_export(root / "data")
        _zip_directory(root / "data", data_zip)
        exported_targets.append("data")
    if export_artifacts and artifacts_zip is not None:
        _zip_directory(root / "artifacts", artifacts_zip)
        exported_targets.append("artifacts")

    return DrivePackageExportReport(
        rowdata_zip=rowdata_zip,
        data_zip=data_zip,
        artifacts_zip=artifacts_zip,
        exported_targets=exported_targets,
    )
