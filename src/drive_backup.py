from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qs, urlparse


DEFAULT_DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/19HHxA5r4T_IqMrDyNqU3qRUoRkhT87OL?usp=drive_link"
DEFAULT_SCOPES = ["https://www.googleapis.com/auth/drive.file"]


@dataclass(slots=True)
class DriveBackupReport:
    rowdata_zip: Path
    drp_zip: Path
    uploaded_files: list[dict[str, str]]
    folder_id: str

    def to_dict(self) -> dict[str, object]:
        return {
            "rowdata_zip": str(self.rowdata_zip),
            "drp_zip": str(self.drp_zip),
            "uploaded_files": self.uploaded_files,
            "folder_id": self.folder_id,
        }


def extract_drive_folder_id(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme and parsed.netloc:
        if "/folders/" in parsed.path:
            return parsed.path.rstrip("/").split("/folders/")[-1].split("/")[0]
        query = parse_qs(parsed.query)
        if "id" in query and query["id"]:
            return query["id"][0]
    return value.strip()


def create_zip_from_directory(source_dir: Path, output_zip: Path) -> Path:
    source_dir = source_dir.resolve()
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(source_dir.rglob("*")):
            if not path.is_file():
                continue
            zf.write(path, arcname=str(Path(source_dir.name) / path.relative_to(source_dir)))
    return output_zip


def create_zip_from_directories(source_dirs: Iterable[Path], output_zip: Path) -> Path:
    normalized = [path.resolve() for path in source_dirs]
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for source_dir in normalized:
            for path in sorted(source_dir.rglob("*")):
                if not path.is_file():
                    continue
                zf.write(path, arcname=str(Path(source_dir.name) / path.relative_to(source_dir)))
    return output_zip


def load_drive_credentials(
    credentials_path: Path,
    token_path: Path,
    scopes: list[str] | None = None,
):
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    scopes = scopes or DEFAULT_SCOPES
    creds = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), scopes)
    if creds and creds.valid:
        return creds
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        flow = InstalledAppFlow.from_client_secrets_file(str(credentials_path), scopes)
        creds = flow.run_local_server(port=0)
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(creds.to_json(), encoding="utf-8")
    return creds


def build_drive_service(credentials_path: Path, token_path: Path):
    from googleapiclient.discovery import build

    creds = load_drive_credentials(credentials_path, token_path)
    return build("drive", "v3", credentials=creds)


def find_existing_drive_file(service, folder_id: str, file_name: str) -> dict | None:
    escaped_name = file_name.replace("'", "\\'")
    query = f"name = '{escaped_name}' and '{folder_id}' in parents and trashed = false"
    response = (
        service.files()
        .list(
            q=query,
            spaces="drive",
            fields="files(id, name)",
            pageSize=10,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        )
        .execute()
    )
    files = response.get("files", [])
    return files[0] if files else None


def upload_or_replace_drive_file(service, folder_id: str, file_path: Path) -> dict[str, str]:
    from googleapiclient.http import MediaFileUpload

    existing = find_existing_drive_file(service, folder_id, file_path.name)
    media = MediaFileUpload(str(file_path), mimetype="application/zip", resumable=True)
    if existing:
        result = (
            service.files()
            .update(
                fileId=existing["id"],
                media_body=media,
                fields="id, name",
                supportsAllDrives=True,
            )
            .execute()
        )
        return {"id": result["id"], "name": result["name"], "action": "updated"}

    metadata = {"name": file_path.name, "parents": [folder_id]}
    result = (
        service.files()
        .create(
            body=metadata,
            media_body=media,
            fields="id, name",
            supportsAllDrives=True,
        )
        .execute()
    )
    return {"id": result["id"], "name": result["name"], "action": "created"}


def package_and_upload_to_drive(
    project_root: Path,
    folder_url_or_id: str = DEFAULT_DRIVE_FOLDER_URL,
    credentials_path: Path | None = None,
    token_path: Path | None = None,
    rowdata_zip_name: str = "rowdata.zip",
    drp_zip_name: str = "drp.zip",
) -> DriveBackupReport:
    root = project_root.resolve()
    folder_id = extract_drive_folder_id(folder_url_or_id)
    credentials = credentials_path or (root / "google_drive_credentials.json")
    token = token_path or (root / "artifacts" / "google-drive-token.json")

    rowdata_zip = create_zip_from_directory(root / "rowdata", root / rowdata_zip_name)
    drp_zip = create_zip_from_directories(
        [root / "artifacts", root / "configs", root / "data"],
        root / drp_zip_name,
    )

    service = build_drive_service(credentials, token)
    uploaded_files = [
        upload_or_replace_drive_file(service, folder_id, rowdata_zip),
        upload_or_replace_drive_file(service, folder_id, drp_zip),
    ]
    return DriveBackupReport(
        rowdata_zip=rowdata_zip,
        drp_zip=drp_zip,
        uploaded_files=uploaded_files,
        folder_id=folder_id,
    )
