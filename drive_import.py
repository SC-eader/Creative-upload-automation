# drive_import.py
from __future__ import annotations
import io
import pathlib
import tempfile
from typing import List, Dict

import streamlit as st
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2.service_account import Credentials

DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

def get_drive_service_from_secrets():
    """Create an authenticated Drive API client using service-account JSON from st.secrets."""
    info = dict(st.secrets["gcp_service_account"])
    creds = Credentials.from_service_account_info(info, scopes=DRIVE_SCOPES)
    # cache_discovery=False avoids file writes in some environments
    return build("drive", "v3", credentials=creds, cache_discovery=False)

def extract_drive_folder_id(url_or_id: str) -> str:
    """Accepts a Drive folder URL or raw ID and returns the folder ID."""
    s = (url_or_id or "").strip()
    if not s:
        raise ValueError("Provide a Google Drive folder URL or ID.")
    if "drive.google.com" in s and "/folders/" in s:
        return s.split("/folders/")[1].split("?")[0].split("/")[0]
    return s  # assume already an ID

def list_drive_videos_in_folder(service, folder_id: str) -> List[Dict]:
    """List video files (id, name, mimeType, size) under the folder (My Drive or Shared Drives)."""
    items, page_token = [], None
    query = (
        f"'{folder_id}' in parents and trashed=false "
        "and (mimeType contains 'video' or mimeType='video/mp4' or mimeType='video/mpeg')"
    )
    while True:
        resp = service.files().list(
            q=query,
            fields="nextPageToken, files(id, name, mimeType, size)",
            pageToken=page_token,
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
            corpora="allDrives",
            pageSize=1000,
        ).execute()
        items.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return items

def download_drive_file_to_tmp(service, file_id: str, filename_hint: str | None = None) -> Dict:
    """Download a Drive file to a temp file, return {'name': str, 'path': str}."""
    request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    name = (filename_hint or f"{file_id}.mp4").strip()
    suffix = pathlib.Path(name).suffix.lower() or ".mp4"
    if suffix not in (".mp4", ".mpeg4"):
        suffix = ".mp4"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        fh = io.FileIO(tmp.name, "wb")
        downloader = MediaIoBaseDownload(fh, request, chunksize=1024 * 1024)  # 1MB chunks
        done = False
        while not done:
            _, done = downloader.next_chunk()
        fh.close()
        return {"name": name if name.lower().endswith((".mp4", ".mpeg4")) else f"{name}{suffix}", "path": tmp.name}

def import_drive_folder_videos(folder_url_or_id: str) -> List[Dict]:
    """High-level helper: list & download all videos inside a Drive folder."""
    service = get_drive_service_from_secrets()
    folder_id = extract_drive_folder_id(folder_url_or_id)
    files = list_drive_videos_in_folder(service, folder_id)
    if not files:
        raise ValueError("No video files found (or access denied). Check folder permissions for the service account.")
    return [download_drive_file_to_tmp(service, f["id"], filename_hint=f.get("name")) for f in files]