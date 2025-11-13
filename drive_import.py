# drive_import.py
from __future__ import annotations
import io
import os
import pathlib
import tempfile
from typing import List, Dict, Optional, Callable

import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.errors import HttpError
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

VIDEO_EXTS = {".mp4", ".mpeg4", ".mov", ".mkv"}  # add more if you want

def list_drive_videos_in_folder(service, folder_id: str) -> List[Dict]:
    """
    List *all* files in a folder (with pagination, supports shared drives),
    keep items that are (mimeType starts with 'video/') or have a known video extension.
    Returns [{'id': ..., 'name': ...}, ...]
    """
    items: List[Dict] = []
    page_token = None
    q = f"'{folder_id}' in parents and trashed=false"
    fields = "nextPageToken, files(id, name, mimeType)"

    while True:
        resp = service.files().list(
            q=q,
            spaces="drive",
            fields=fields,
            pageSize=1000,
            pageToken=page_token,
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
            corpora="allDrives",
            orderBy="name_natural"
        ).execute()
        for f in resp.get("files", []):
            name = f.get("name") or ""
            ext = pathlib.Path(name).suffix.lower()
            if (f.get("mimeType","").startswith("video/")) or (ext in VIDEO_EXTS):
                items.append({"id": f["id"], "name": name})
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return items


def download_drive_file_to_tmp(service, file_id: str, filename_hint: Optional[str] = None, *, max_retries: int = 5) -> Dict:
    """
    Stream a Drive file directly to a temporary file on disk and return {'name','path'}.
    Retries transient errors with exponential backoff.
    """
    # We first query the metadata to get the name (cheaper than guessing)
    try:
        meta = service.files().get(fileId=file_id, fields="name", supportsAllDrives=True).execute()
        name = meta.get("name") or filename_hint or f"{file_id}.mp4"
    except Exception:
        name = filename_hint or f"{file_id}.mp4"

    # Ensure extension
    ext = pathlib.Path(name).suffix.lower()
    if ext == "" or ext not in VIDEO_EXTS:
        name = f"{name}.mp4"

    suffix = pathlib.Path(name).suffix or ".mp4"

    # Request media
    request = service.files().get_media(fileId=file_id, supportsAllDrives=True)

    # Stream to disk, not RAM
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        path = tmp.name
    fh = open(path, "wb")
    try:
        downloader = MediaIoBaseDownload(fh, request, chunksize=1024 * 1024)
        done = False
        attempt = 0
        while not done:
            try:
                status, done = downloader.next_chunk()
            except HttpError as e:
                attempt += 1
                if attempt > max_retries:
                    raise
                # basic exponential backoff
                import time
                time.sleep(min(2 ** attempt, 30))
            except Exception as e:
                attempt += 1
                if attempt > max_retries:
                    raise
                import time
                time.sleep(min(2 ** attempt, 30))
    finally:
        fh.close()

    return {"name": pathlib.Path(name).name, "path": path}

def import_drive_folder_videos_parallel(
    folder_url_or_id: str,
    max_workers: int = 4,
    on_progress: Optional[Callable[[int, int, str, Optional[str]], None]] = None,
) -> List[Dict]:
    """
    List & download all videos in parallel.
    Calls on_progress(done, total, file_name, error_message_or_None) after each file finishes.
    Returns successfully downloaded [{'name','path'}, ...].
    """
    # Enumerate first with a single service (cheap calls)
    svc_list = get_drive_service_from_secrets()
    folder_id = extract_drive_folder_id(folder_url_or_id)
    files = list_drive_videos_in_folder(svc_list, folder_id)
    total = len(files)
    done = 0
    results: List[Dict] = []
    errors: List[str] = []

    if total == 0:
        if on_progress: on_progress(0, 0, "", None)
        return results

    def _one(meta: Dict) -> Dict:
        # Create a fresh service per worker to avoid concurrency issues
        svc = get_drive_service_from_secrets()
        return download_drive_file_to_tmp(svc, meta["id"], filename_hint=meta.get("name"))

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut_map = {ex.submit(_one, f): f for f in files}
        for fut in as_completed(fut_map):
            src = fut_map[fut]
            name = src.get("name", "(no name)")
            err_msg = None
            try:
                out = fut.result()
                results.append(out)
            except Exception as e:
                err_msg = str(e)
                errors.append(f"{name}: {err_msg}")
            finally:
                done += 1
                if on_progress:
                    on_progress(done, total, name, err_msg)

    if errors:
        import logging
        logging.warning("Some Drive downloads failed: %s", errors)

    return results