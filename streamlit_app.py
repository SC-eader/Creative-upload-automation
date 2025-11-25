"""Streamlit app: bulk upload per-game videos from Drive and create Meta creative tests."""
from __future__ import annotations

import os
from typing import Dict, List
from datetime import datetime, timedelta, timezone
import tempfile
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

logger = logging.getLogger(__name__)
import requests
import pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
from streamlit.components.v1 import html as components_html 

try:
    from drive_import import import_drive_folder_videos_parallel as import_drive_folder_videos
    _DRIVE_IMPORT_SUPPORTS_PROGRESS = True
except ImportError:
    from drive_import import import_drive_folder_videos  # old signature: (folder_url_or_id) -> list[{"name","path"}]
    _DRIVE_IMPORT_SUPPORTS_PROGRESS = False
from unity_ads import (
    render_unity_settings_panel,
    get_unity_settings,
    upload_unity_creatives_to_campaign,
)
VERBOSE_UPLOAD_LOG = False

# ----- UI/Validation helpers --------------------------------------------------
try:
    MAX_UPLOAD_MB = int(st.get_option("server.maxUploadSize"))
except Exception:
    MAX_UPLOAD_MB = 200  # Streamlit default if option missing

def init_state():
    """Ensure we have places to store uploads and per-game settings in session state."""
    if "uploads" not in st.session_state:
        st.session_state.uploads = {}
    if "settings" not in st.session_state:
        st.session_state.settings = {}

def init_remote_state():
    """Ensure we have a place to store server-downloaded (URL) videos per game."""
    if "remote_videos" not in st.session_state:
        st.session_state.remote_videos = {}  # {game: [ {"name":..., "path":...}, ... ]}

def ensure_settings_state():
    """Ensure we have a per-game dict in session_state for settings."""
    if "settings" not in st.session_state:
        st.session_state.settings = {}

def game_tabs(n: int) -> List[str]:
    """Return the fixed list of 10 game names (tabs)."""
    return [
        "XP HERO",
        "Dino Universe",
        "Snake Clash",
        "Pizza Ready",
        "Cafe Life",
        "Suzy's Restaurant",
        "Office Life",
        "Lumber Chopper",
        "Burger Please",
        "Prison Life",
    ]

def validate_count(files: List) -> tuple[bool, str]:
    """Ensure at least one video is uploaded and that all files are videos (.mp4/.mpeg4)."""
    if not files:
        return False, "Please upload at least one video (.mp4 or .mpeg4)."

    allowed = {".mp4", ".mpeg4"}
    bad = []
    for u in files:
        # Handle both UploadedFile objects and dicts from Drive imports
        name = getattr(u, "name", None) or (u.get("name") if isinstance(u, dict) else None)
        if not name:
            continue
        if pathlib.Path(name).suffix.lower() not in allowed:
            bad.append(name)

    if bad:
        return (
            False,
            f"Only video files are allowed (.mp4/.mpeg4). "
            f"Remove non-video files: {', '.join(bad[:5])}{'…' if len(bad) > 5 else ''}",
        )
    return True, f"{len(files)} video(s) ready."

def validate_page_binding(account, page_id: str) -> dict:
    """
    Ensure page_id is numeric, readable by token, and fetch IG actor (if any).
    Raises a RuntimeError with a precise hint if not usable.
    """
    _require_fb()
    from facebook_business.adobjects.page import Page
    pid = str(page_id).strip()
    if not pid.isdigit():
        raise RuntimeError(f"Page ID must be numeric. Got: {page_id!r}")
    try:
        p = Page(pid).api_get(fields=["id", "name", "instagram_business_account"])
    except Exception as e:
        raise RuntimeError(
            f"Page validation failed for PAGE_ID={pid}. "
            "Use a real Facebook Page ID and ensure the token can read it."
        ) from e
    iba = (p.get("instagram_business_account") or {}).get("id")
    return {"id": p["id"], "name": p["name"], "instagram_business_account_id": iba}


def _fname_any(u):
    """Return a filename for either a Streamlit UploadedFile or a {'name','path'} dict."""
    return getattr(u, "name", None) or (u.get("name") if isinstance(u, dict) else "")

def _dedupe_by_name(files):
    """Keep first occurrence of each filename (case-insensitive)."""
    seen = set()
    out = []
    for u in files or []:
        n = (_fname_any(u) or "").strip().lower()
        if n and n not in seen:
            seen.add(n)
            out.append(u)
    return out

def _run_drive_import(folder_url_or_id: str, max_workers: int, on_progress=None):
    """
    Calls the Drive import function in a version-agnostic way.
    If the parallel importer is available, we pass workers + on_progress.
    Otherwise, we call the legacy function and emulate a simple progress callback.
    """
    if _DRIVE_IMPORT_SUPPORTS_PROGRESS:
        return import_drive_folder_videos(folder_url_or_id, max_workers=max_workers, on_progress=on_progress)

    # Legacy path: no workers/progress in the older function.
    files = import_drive_folder_videos(folder_url_or_id)
    total = len(files)
    if on_progress:
        done = 0
        for f in files:
            done += 1
            on_progress(done, total, f.get("name", ""), None)
    return files
# ----- Settings helpers -------------------------------------------------------
def sanitize_store_url(raw: str) -> str:
    """
    Keep only the parts required for Meta:
      - Google Play: keep ?id=<package> (drop other params)
      - App Store: keep path; drop query/fragment
      - Other hosts: return as-is
    """
    from urllib.parse import urlsplit, urlunsplit, parse_qs, urlencode

    if not raw:
        return raw

    parts = urlsplit(raw)
    host = parts.netloc.lower()

    # Google Play: we MUST preserve the 'id' query parameter
    if "play.google.com" in host:
        qs = parse_qs(parts.query)
        pkg = (qs.get("id") or [None])[0]
        if not pkg:
            # Allow "details?id=..." style only; if missing, raise a helpful error
            raise ValueError("Google Play URL must include ?id=<package>. Example: https://play.google.com/store/apps/details?id=io.supercent.weaponrpg")
        # Rebuild with ONLY the id param
        new_query = urlencode({"id": pkg})
        return urlunsplit((parts.scheme, parts.netloc, parts.path or "/store/apps/details", new_query, ""))

    # Apple App Store: ID is in the path; query can be dropped safely
    if "apps.apple.com" in host:
        return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))

    # Other hosts: return unchanged
    return raw

ASIA_SEOUL = timezone(timedelta(hours=9))  # KST (+09:00)

def next_sat_0900_kst(today: datetime | None = None) -> str:
    """Return default start_iso = next Saturday 00:00 in KST."""
    now = (today or datetime.now(ASIA_SEOUL)).astimezone(ASIA_SEOUL)
    base = now.replace(hour=0, minute=0, second=0, microsecond=0)
    # Monday=0 ... Saturday=5, Sunday=6
    days_until_sat = (5 - base.weekday()) % 7 or 7
    start_dt = (base + timedelta(days=days_until_sat)).replace(hour=9, minute=0)
    return start_dt.isoformat()

def compute_budget_from_settings(files: list, settings: dict, fallback_per_video: int = 10) -> int:
    """
    Budget per day = (#eligible videos) × (per-video budget chosen by user).
    Supports UploadedFile and {'name':..., 'path':...} dicts.
    """
    allowed = {".mp4", ".mpeg4"}
    def _name(u):
        return getattr(u, "name", None) or (u.get("name") if isinstance(u, dict) else "")
    n_videos = sum(1 for u in (files or []) if pathlib.Path(_name(u)).suffix.lower() in allowed)
    per_video = int(settings.get("budget_per_video_usd", fallback_per_video))
    return max(1, n_videos * per_video) if n_videos else per_video

def dollars_to_minor(usd: float) -> int:
    """Convert USD to Meta minor units ($1 → 100)."""
    return int(round(usd * 100))
ANDROID_OS_CHOICES = {
    "None (any)": None,
    "6.0+": "Android_ver_6.0_and_above",
    "7.0+": "Android_ver_7.0_and_above",
    "8.0+": "Android_ver_8.0_and_above",
    "9.0+": "Android_ver_9.0_and_above",
    "10.0+": "Android_ver_10.0_and_above",
    "11.0+": "Android_ver_11.0_and_above",
    "12.0+": "Android_ver_12.0_and_above",
    "13.0+": "Android_ver_13.0_and_above",
    "14.0+": "Android_ver_14.0_and_above",
}

IOS_OS_CHOICES = {
    "None (any)": None,
    "11.0+": "iOS_ver_11.0_and_above",
    "12.0+": "iOS_ver_12.0_and_above",
    "13.0+": "iOS_ver_13.0_and_above",
    "14.0+": "iOS_ver_14.0_and_above",
    "15.0+": "iOS_ver_15.0_and_above",
    "16.0+": "iOS_ver_16.0_and_above",
    "17.0+": "iOS_ver_17.0_and_above",
    "18.0+": "iOS_ver_18.0_and_above",
}

OPT_GOAL_LABEL_TO_API = {
    "앱 설치수 극대화": "APP_INSTALLS",
    "앱 이벤트 수 극대화": "APP_EVENTS",
    "전환값 극대화": "VALUE",
    "링크 클릭수 극대화": "LINK_CLICKS",
}

def build_targeting_from_settings(country: str, age_min: int, settings: dict) -> dict:
    """
    Build Meta targeting dict from UI settings.
    Uses user_os (version tokens) to filter platform instead of deprecated operating_systems.
    """
    os_choice = settings.get("os_choice", "Both")
    min_android = settings.get("min_android_os_token")  # e.g., "Android_ver_6.0_and_above" or None
    min_ios = settings.get("min_ios_os_token")          # e.g., "iOS_ver_15.0_and_above" or None

    # Base targeting
    targeting = {
        "geo_locations": {"countries": [country]},
        "age_min": max(13, int(age_min)),
        "publisher_platforms": ["facebook", "instagram"],
        "device_platforms": ["mobile"],
    }

    # IMPORTANT: Do NOT send 'operating_systems' (deprecated/invalid).
    # Enforce OS family via user_os tokens only.
    user_os: list[str] = []

    if os_choice == "Android only":
        # If no explicit minimum was chosen, fall back to a sane default so it truly filters to Android.
        token = min_android or "Android_ver_6.0_and_above"
        user_os.append(token)

    elif os_choice == "iOS only":
        # If no explicit minimum was chosen, fall back to a sane default for iOS-only filtering.
        token = min_ios or "iOS_ver_11.0_and_above"
        user_os.append(token)

    else:  # Both
        # Include whichever minimums were selected; if neither was given, we simply won't filter by OS.
        if min_android:
            user_os.append(min_android)
        if min_ios:
            user_os.append(min_ios)

    if user_os:
        targeting["user_os"] = user_os

    return targeting

def make_ad_name(filename: str, prefix: str | None) -> str:
    """Build ad name from filename and optional prefix."""
    return f"{prefix.strip()}_{filename}" if prefix else filename

# ----- Meta (Facebook) Marketing API wiring ----------------------------------

try:
    from facebook_business.api import FacebookAdsApi
    from facebook_business.adobjects.adaccount import AdAccount
    from facebook_business.adobjects.adset import AdSet
    from facebook_business.adobjects.adcreative import AdCreative
    from facebook_business.adobjects.ad import Ad
    FB_AVAILABLE = True
    FB_IMPORT_ERROR = ""
except Exception as _e:
    FB_AVAILABLE = False
    FB_IMPORT_ERROR = f"{type(_e).__name__}: {_e}"

def _require_fb():
    """Raise a clear error if Facebook SDK isn't available."""
    if not FB_AVAILABLE:
        raise RuntimeError(
            "facebook-business SDK not available. Install it with:\n"
            "  pip install facebook-business\n"
            f"Import error: {FB_IMPORT_ERROR}"
        )

def init_fb_from_secrets(ad_account_id: str | None = None) -> AdAccount:
    """
    Initialize the Meta SDK using ONLY the access token, and return an AdAccount.
    If ad_account_id is None, fall back to the XP HERO ad account.
    """
    _require_fb()
    token = st.secrets.get("access_token", "").strip()
    if not token:
        raise RuntimeError("Missing access_token in st.secrets. Put it in .streamlit/secrets.toml")

    # Initialize with access token only (no app_id/app_secret)
    FacebookAdsApi.init(access_token=token)

    # Default to XP HERO account if none is provided
    default_act_id = "act_692755193188182"
    act_id = ad_account_id or default_act_id
    return AdAccount(act_id) # XP HERO ad account

def create_creativetest_adset(
    account: AdAccount,
    *,
    campaign_id: str,
    adset_name: str,
    targeting: dict,
    daily_budget_usd: int,
    start_iso: str,
    optimization_goal: str,  # API token string like "APP_INSTALLS"
    promoted_object: dict | None = None,
    end_iso: str | None = None,  # optional: only used if provided
) -> str:
    """Create an active ad set with the given name/settings; return adset_id."""
    from facebook_business.adobjects.adset import AdSet

    params = {
        "name": adset_name,
        "campaign_id": campaign_id,
        "daily_budget": dollars_to_minor(daily_budget_usd),
        "billing_event": AdSet.BillingEvent.impressions,
        "optimization_goal": getattr(
            AdSet.OptimizationGoal,
            optimization_goal.lower(),
            AdSet.OptimizationGoal.app_installs,
        ),
        "bid_strategy": "LOWEST_COST_WITHOUT_CAP",
        "targeting": targeting,
        "status": AdSet.Status.active,
        "start_time": start_iso,
    }

    # Only include end_time if the user explicitly set one
    if end_iso:
        params["end_time"] = end_iso

    if promoted_object:
        params["promoted_object"] = promoted_object

    adset = account.create_ad_set(fields=[], params=params)
    return adset["id"]

def _save_uploadedfile_tmp(u) -> str:
    """
    Return a local path for a video source.
    - If u is a Streamlit UploadedFile -> persist to a temp file and return its path
    - If u is a dict like {'name':..., 'path':...} (from URL) -> return its existing path
    """
    # URL-imported object
    if isinstance(u, dict) and "path" in u and "name" in u:
        return u["path"]
    # Streamlit UploadedFile
    if hasattr(u, "getbuffer"):
        suffix = pathlib.Path(u.name).suffix.lower() or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(u.getbuffer())
            return tmp.name
    raise ValueError("Unsupported video object type for saving.")

def upload_videos_create_ads(
    account: AdAccount,
    *,
    page_id: str,
    adset_id: str,
    uploaded_files: list,
    ad_name_prefix: str | None = None,
    max_workers: int = 6,  # <-- Renamed from max_workers_save for clarity
    store_url: str | None = None,
    try_instagram: bool = True,
):
    """
    Upload videos (resumable), wait for processing, then create a video creative and an ad.
    - Uses chunked upload to avoid 390/1363030 timeouts.
    - Adds INSTALL_MOBILE_APP CTA when APP_INSTALLS flow is used (store_url supplied).
    - If Instagram actor is unavailable, automatically drops IG from placements for this ad creation.
    Returns [{"name","ad_id"}] and shows per-file errors in the UI.
    """
    from facebook_business.adobjects.advideo import AdVideo
    from facebook_business.adobjects.page import Page
    from facebook_business.adobjects.adcreative import AdCreative
    from facebook_business.adobjects.ad import Ad
    from facebook_business.exceptions import FacebookRequestError
    import time
    import pathlib

    allowed = {".mp4", ".mpeg4"}

    def _is_video(u):
        n = _fname_any(u) or "video.mp4"
        return pathlib.Path(n).suffix.lower() in allowed

    videos = _dedupe_by_name([u for u in (uploaded_files or []) if _is_video(u)])

    # ---------- helpers ----------
    def _persist_to_tmp(u):
        return {"name": _fname_any(u) or "video.mp4", "path": _save_uploadedfile_tmp(u)}

    def simple_video_upload(path: str) -> str:
        """
        Fallback: single-shot upload via SDK.
        Use only when resumable keeps failing (small/fast files often work fine).
        Returns the new video_id.
        """
        v = account.create_ad_video(params={"file": path, "content_category": "VIDEO_GAMING"})
        return v["id"]

    def upload_video_resumable(path: str) -> str:
        """
        Chunked upload to /{act_id}/advideos using the official 3-phase protocol.
        1) start  -> get (upload_session_id, video_id, start_offset, end_offset)
        2) transfer -> send EXACTLY [start_offset, end_offset) bytes each time, loop until offsets reach EOF
        3) finish
        Retries transient errors (HTTP 5xx, code=390). Verifies that total sent bytes == file_size.
        Returns the uploaded video_id.
        """
        import os, time, requests

        token = (st.secrets.get("access_token") or "").strip()
        if not token:
            raise RuntimeError("Missing access_token in st.secrets")

        act = account.get_id()  # e.g., "act_692755193188182"
        base = f"https://graph.facebook.com/v24.0/{act}/advideos"
        file_size = os.path.getsize(path)

        def _post(data, files=None, max_retries=5):
            delays = [0, 2, 4, 8, 12]
            last = None
            for i, d in enumerate(delays[:max_retries], 1):
                if d:
                    time.sleep(d)
                try:
                    r = requests.post(
                        base,
                        data={**data, "access_token": token},
                        files=files,
                        timeout=180,
                    )
                    if r.status_code >= 500:
                        last = RuntimeError(f"HTTP {r.status_code}: {r.text[:400]}")
                        continue
                    j = r.json()
                    if "error" in j:
                        code = j["error"].get("code")
                        # Transient upload timeout
                        if code in (390,) and i < max_retries:
                            last = RuntimeError(j["error"].get("message"))
                            continue
                        raise RuntimeError(j["error"].get("message", str(j["error"])))
                    return j
                except Exception as e:
                    last = e
            raise last or RuntimeError("advideos POST failed")

        # ---- 1) start (file_size REQUIRED) ----
        start_resp = _post({
            "upload_phase": "start",
            "file_size": str(file_size),
            "content_category": "VIDEO_GAMING",
        })
        upload_session_id = start_resp["upload_session_id"]
        video_id = start_resp["video_id"]
        start_offset = int(start_resp.get("start_offset", 0))
        end_offset = int(start_resp.get("end_offset", 0))

        sent_bytes = 0

        # ---- 2) transfer (loop until offsets converge to file_size) ----
        with open(path, "rb") as f:
            while True:
                # Done only when both offsets == file_size (server has everything)
                if start_offset == end_offset == file_size:
                    if VERBOSE_UPLOAD_LOG:
                        st.write(f"[Upload] ✅ All bytes acknowledged ({sent_bytes}/{file_size}).")
                    break

                # If Graph returns a stall window (no progress yet), ask again (no file chunk)
                if end_offset <= start_offset:
                    if VERBOSE_UPLOAD_LOG:
                        st.write(f"[Upload] ↻ Asking for next window at {start_offset}")
                    tr = _post({
                        "upload_phase": "transfer",
                        "upload_session_id": upload_session_id,
                        "start_offset": str(start_offset),
                    })
                    start_offset = int(tr.get("start_offset", start_offset))
                    end_offset   = int(tr.get("end_offset", end_offset or file_size))
                    continue

                # Send exactly the requested window [start_offset, end_offset)
                to_read = end_offset - start_offset
                f.seek(start_offset)
                chunk = f.read(to_read)
                if not chunk or len(chunk) != to_read:
                    raise RuntimeError(f"Read {len(chunk) if chunk else 0} bytes; expected {to_read}.")

                files = {"video_file_chunk": ("chunk.bin", chunk, "application/octet-stream")}
                tr = _post({
                    "upload_phase": "transfer",
                    "upload_session_id": upload_session_id,
                    "start_offset": str(start_offset),
                }, files=files)

                sent_bytes += to_read
                new_start = int(tr.get("start_offset", start_offset + to_read))
                new_end   = int(tr.get("end_offset", end_offset))

                if VERBOSE_UPLOAD_LOG:
                    st.write(
                        f"[Upload] Sent [{start_offset},{end_offset}) → "
                        f"ack: start={new_start}, end={new_end}, sent={sent_bytes}/{file_size}"
                    )

                start_offset, end_offset = new_start, new_end

                # Safety valve: if Graph says we've passed EOF, normalize to file_size
                if start_offset > file_size:
                    start_offset = file_size
                if end_offset > file_size:
                    end_offset = file_size

        # Sanity check before finish
        if sent_bytes != file_size:
            raise RuntimeError(f"Uploaded bytes ({sent_bytes}) != file size ({file_size}).")

        # ---- 3) finish ----
        try:
            _post({"upload_phase": "finish", "upload_session_id": upload_session_id})
            return video_id
        except Exception:
            # Last resort: small files often succeed with single-shot upload
            st.info(f"Resumable finish failed for {os.path.basename(path)} — trying fallback upload once.")
            v = account.create_ad_video(params={"file": path, "content_category": "VIDEO_GAMING"})
            return v["id"]

    def wait_all_videos_ready(video_ids: list[str], *, timeout_s: int = 300, sleep_s: int = 5) -> dict[str, bool]:
        """
        FAST-PATH: skip polling AdVideo.status.

        We assume Meta will finish encoding in the background and immediately
        proceed to creative/ad creation. This removes the extra "Encoding..." wait
        and significantly speeds up the end-to-end flow.

        Returns {video_id: True} for all provided IDs so the rest of the pipeline
        can run unchanged.
        """
        return {vid: True for vid in (video_ids or [])}

    def resolve_instagram_actor_id(page_id: str) -> str | None:
        """
        Try to get the IG actor connected to the Page. If none, return None.
        """
        try:
            p = Page(page_id).api_get(fields=["instagram_business_account"])
            iba = p.get("instagram_business_account") or {}
            return iba.get("id")
        except Exception:
            return None

    # ---------- Stage 1: persist to temp (parallel I/O only) ----------
    persisted, persist_errors = [], []
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as ex: # <-- Use max_workers
        futs = {ex.submit(_persist_to_tmp, u): _fname_any(u) for u in videos}
        for fut, nm in futs.items():
            try:
                persisted.append(fut.result())
            except Exception as e:
                persist_errors.append(f"{nm}: {e}")

    if persist_errors:
        st.warning("Some files failed to prepare:\n- " + "\n- ".join(persist_errors))

    # Prepare IG actor if required
    # Prefer a pre-validated IG actor id passed via cfg; else try to resolve from Page
    ig_actor_id = None
    try:
        # When available upstream, stash into st.session_state for this run
        ig_actor_id = st.session_state.get("ig_actor_id_from_page") or None
    except Exception:
        pass
    if try_instagram and not ig_actor_id:
        ig_actor_id = resolve_instagram_actor_id(page_id)

    # ---------- Stage 2: API calls ----------
    # Phase A) Upload all videos first (no waiting yet)
    uploads, api_errors = [], []
    total = len(persisted)
    progress = st.progress(0, text=f"Uploading 0/{total} videos…") if total else None

    def _upload_one(item):
        """Upload a single video file to Meta and return its info."""
        name, path = item["name"], item["path"]
        vid = upload_video_resumable(path)
        return {"name": name, "path": path, "video_id": vid}

    done = 0
    if total:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=max_workers) as ex: # <-- Use max_workers
            future_to_item = {ex.submit(_upload_one, item): item for item in persisted}
            for fut in as_completed(future_to_item):
                item = future_to_item[fut]
                name = item["name"]
                try:
                    res = fut.result()
                    res["ready"] = False
                    uploads.append(res)
                    done += 1
                    if progress is not None:
                        pct = int(done / total * 100)
                        progress.progress(pct, text=f"Uploading {done}/{total} videos…")
                except Exception as e:
                    api_errors.append(f"{name}: upload failed: {e}")
    
    if progress:
        progress.empty() # Clear upload progress bar


    # Phase B) Poll all videos concurrently (round-robin) with a single progress bar
    # (Skipped by FAST-PATH)

    # Phase C) Create creatives & ads
    results = []

    # ✅ NEW: wait for ALL uploaded videos together (batch) instead of one-by-one
    video_ids = [u["video_id"] for u in uploads]
    ready_map = wait_all_videos_ready(video_ids, timeout_s=300, sleep_s=5)

    # =======================================================================
    # START: 🚀 NEW PARALLEL CREATION LOGIC
    # =======================================================================
    
    def _process_one_video(up):
        """
        Helper to fetch thumbnail, create creative, and create ad for one video.
        This will be run in a thread.
        """
        # We must import these here for thread safety with the SDK
        from facebook_business.adobjects.advideo import AdVideo
        from facebook_business.adobjects.adcreative import AdCreative
        from facebook_business.adobjects.ad import Ad
        from facebook_business.exceptions import FacebookRequestError
        import time

        name, video_id = up["name"], up["video_id"]

        try:
            # --- Fetch thumbnail ---
            video_info = AdVideo(video_id).api_get(fields=["picture"])
            thumbnail_url = video_info.get("picture")
            if not thumbnail_url:
                raise RuntimeError("Video processed but no 'picture' (thumbnail) URL was returned.")

            # --- Internal create helper (copied from original loop) ---
            def _create_once(allow_ig: bool) -> str:
                vd = {
                    "video_id": video_id,
                    "title": name,
                    "message": "",
                    "image_url": thumbnail_url,
                }

                if store_url:
                    vd["call_to_action"] = {
                        "type": "INSTALL_MOBILE_APP",
                        "value": {"link": store_url},
                    }

                spec = {"page_id": page_id, "video_data": vd}

                if allow_ig and ig_actor_id:
                    spec["instagram_actor_id"] = ig_actor_id

                creative = account.create_ad_creative(
                    fields=[],
                    params={"name": name, "object_story_spec": spec},
                )
                ad = account.create_ad(
                    fields=[],
                    params={
                        "name": make_ad_name(name, ad_name_prefix),
                        "adset_id": adset_id,
                        "creative": {"creative_id": creative["id"]},
                        "status": Ad.Status.active,
                    },
                )
                return ad["id"]

            # --- Try/Except logic (copied from original loop) ---
            try:
                ad_id = _create_once(True)
            except FacebookRequestError as e:
                msg = (e.api_error_message() or "").lower()
                if "instagram" in msg or "not ready" in msg or "processing" in msg:
                    time.sleep(5) # Sleep is fine, it's in a thread
                    ad_id = _create_once(False)
                else:
                    raise
            
            # Return success
            return {"success": True, "result": {"name": name, "ad_id": ad_id}}

        except Exception as e:
            # Return failure
            return {"success": False, "error": f"{name}: creative/ad failed: {e}"}

    # --- NEW: Run _process_one_video in a ThreadPool ---
    total_c = len(uploads)
    progress_c = st.progress(0, text=f"Creating 0/{total_c} ads…") if total_c else None
    done_c = 0

    if total_c:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        with ThreadPoolExecutor(max_workers=max_workers) as ex: # <-- Use max_workers
            future_to_video = {ex.submit(_process_one_video, up): up for up in uploads}
            
            for fut in as_completed(future_to_video):
                res = fut.result()
                
                done_c += 1
                if progress_c:
                    pct = int(done_c / total_c * 100)
                    progress_c.progress(pct, text=f"Creating {done_c}/{total_c} ads…")

                if res["success"]:
                    results.append(res["result"])
                else:
                    api_errors.append(res["error"])
    
    if progress_c:
        progress_c.empty()

    # =======================================================================
    # END: 🚀 NEW PARALLEL CREATION LOGIC
    # =======================================================================

    # The old serial loop (from line 790 to 862) is now replaced
    # by the ThreadPoolExecutor block above.

    # --- Cleanup & Return ---
    if progress:
        progress.empty()

    # Log errors to the UI
    if api_errors:
        st.error(
            f"{len(api_errors)} video(s) failed during creation:\n"
            + "\n".join(f"- {e}" for e in api_errors[:20])
            + ("\n..." if len(api_errors) > 20 else "")
        )
    
    return results


def _plan_upload(account: AdAccount, *, campaign_id: str, adset_prefix: str, page_id: str, uploaded_files: list, settings: dict) -> dict:
    """Compute what would be created (no writes): ad set name, budget/schedule, and ad names."""
    # Schedule: start is required (default = next Saturday 00:00 KST),
    # end is optional (we won't automatically turn off if it's empty).
    start_iso = settings.get("start_iso")
    if not start_iso:
        # use your helper that returns the default start
        start_iso = next_sat_0900_kst()

    # end date is optional; keep it in the plan dict for display only (may be None / "")
    end_iso = settings.get("end_iso")

    # Ad set suffix: user-selected n
    n = int(settings.get("suffix_number") or 1)
    suffix_str = f"{n}th"

    # Optional: add launch date (start_iso) as YYYYmmdd, e.g. _35th_20251122
    launch_date_suffix = ""
    if settings.get("add_launch_date"):
        try:
            dt = datetime.fromisoformat(start_iso)
            launch_date_suffix = "_" + dt.strftime("%y%m%d")
        except Exception:
            launch_date_suffix = ""

    adset_name = f"{adset_prefix}_{suffix_str}{launch_date_suffix}"

    # Videos (local + any server-downloaded)
    allowed = {".mp4", ".mpeg4"}
    remote = st.session_state.remote_videos.get(settings.get("game_key", ""), []) or []

    def _name(u):
        return getattr(u, "name", None) or (u.get("name") if isinstance(u, dict) else "")

    def _is_video(u):
        return pathlib.Path(_name(u)).suffix.lower() in allowed

    vids_local = [u for u in (uploaded_files or []) if _is_video(u)]
    vids_all = _dedupe_by_name(vids_local + [rv for rv in remote if _is_video(rv)])

    # Budget per day = per-video × count
    budget_usd_per_day = compute_budget_from_settings(vids_all, settings)

    ad_name_prefix = settings.get("ad_name_prefix") if settings.get("ad_name_mode") == "Prefix + filename" else None
    ad_names = [make_ad_name(_name(u), ad_name_prefix) for u in vids_all]

    return {
        "campaign_id": campaign_id,
        "adset_name": adset_name,
        "country": settings.get("country", "US"),
        "age_min": int(settings.get("age_min", 18)),
        "budget_usd_per_day": int(budget_usd_per_day),
        "start_iso": start_iso,
        "end_iso": end_iso,  # may be None/empty; only for summary/UI
        "page_id": page_id,
        "n_videos": len(vids_all),
        "ad_names": ad_names,
        # ▼ extra metadata for summary
        "campaign_name": settings.get("campaign_name"),  # will be set by caller
        "app_store": settings.get("app_store"),
        "opt_goal_label": settings.get("opt_goal_label"),
    }

def upload_to_facebook(game_name: str, uploaded_files: list, settings: dict, *, simulate: bool = False):
    """Create the chosen ad set and one paused ad per video (simulate=True returns plan only)."""

    mapping = {
        "XP HERO": {
            "account_id": "act_692755193188182",
            "campaign_id": "120218934861590118",
            "campaign_name": "weaponrpg_aos_facebook_us_creativetest",
            "adset_prefix": "weaponrpg_aos_facebook_us_creativetest",
            "page_id": st.secrets["page_id_xp"],  # MUST be a Facebook Page ID, NOT the ad account ID
        },
        "Dino Universe": {
            "account_id": "act_1400645283898971",  # <- put the Dino ad account here
            "campaign_id": "120203672340130431",
            "campaign_name": "ageofdinosaurs_aos_facebook_us_test_6th+",
            "adset_prefix": "ageofdinosaurs_aos_facebook_us_test",
            "page_id": st.secrets["page_id_dino"],
        },
        "Snake Clash": {
            "account_id": "act_837301614677763",  # <- Snake account
            "campaign_id": "120201313657080615",
            "campaign_name": "linkedcubic_aos_facebook_us_test_14th above",
            "adset_prefix": "linkedcubic_aos_facebook_us_test",
            "page_id": st.secrets["page_id_snake"],
        },
        "Pizza Ready": {
            "account_id": "act_939943337267153",  # <- Pizza account
            "campaign_id": "120200161907250465",
            "campaign_name": "pizzaidle_aos_facebook_us_test_12th+",
            "adset_prefix": "pizzaidle_aos_facebook_us_test",
            "page_id": st.secrets["page_id_pizza"],
        },
        "Cafe Life": {
            "account_id": "act_1425841598550220",  # cafe ad account
            "campaign_id": "120231530818850361",
            "campaign_name": "cafelife_aos_facebook_us_creativetest",
            "adset_prefix": "cafelife_aos_facebook_us_creativetest",
            "page_id": st.secrets["page_id_cafe"],
        },
        "Suzy's Restaurant": {
            "account_id": "act_953632226485498",  # suzy ad account
            "campaign_id": "120217220153800643",
            "campaign_name": "suzyrest_aos_facebook_us_creativetest",
            "adset_prefix": "suzyrest_aos_facebook_us_creativetest",
            "page_id": st.secrets["page_id_suzy"],
        },
        "Office Life": {
            "account_id": "act_733192439468531",  # office ad account
            "campaign_id": "120228464454680636",
            "campaign_name": "corporatetycoon_aos_facebook_us_creativetest",
            "adset_prefix": "corporatetycoon_aos_facebook_us_creativetest",
            "page_id": st.secrets["page_id_office"],
        },
        "Lumber Chopper": {
            "account_id": "act_1372896617079122",  # lumber ad account
            "campaign_id": "120224569359980144",
            "campaign_name": "lumberchopper_aos_facebook_us_creativetest",
            "adset_prefix": "lumberchopper_aos_facebook_us_creativetest",
            "page_id": st.secrets["page_id_lumber"],
        },
        "Burger Please": {
            "account_id": "act_3546175519039834",  # burger ad account
            "campaign_id": "120200361364790724",
            "campaign_name": "burgeridle_aos_facebook_us_test_30th+",
            "adset_prefix": "burgeridle_aos_facebook_us_test",
            "page_id": st.secrets["page_id_burger"],
        },
        "Prison Life": {
            "account_id": "act_510600977962388",  # prison ad account
            "campaign_id": "120212520882120614",
            "campaign_name": "prison_aos_facebook_us_install_test",
            "adset_prefix": "prison_aos_facebook_us_install_test",
            "page_id": st.secrets["page_id_prison"],
        },
    }

    if game_name not in mapping:
        raise ValueError(f"No FB mapping configured for game: {game_name}")
    cfg = mapping[game_name]

    # ⬇️ NEW: initialize the correct ad account for this game
    account = init_fb_from_secrets(cfg["account_id"])
      # --- Validate Page ID early and capture IG actor if present ---
    page_check = validate_page_binding(account, cfg["page_id"])
    # Optional: if user wants IG but Page has no IG actor, we can still run (we'll drop IG later).
    ig_actor_id_from_page = page_check.get("instagram_business_account_id")
    # --- Validate that cfg["page_id"] is a real Page (not the ad account number) ---
    try:
        acct_num = account.get_id().replace("act_", "")
        pid = str(cfg["page_id"])
        if pid in (acct_num, f"act_{acct_num}"):
            raise RuntimeError(
                "Configured PAGE_ID equals the Ad Account ID. "
                "Set st.secrets['page_id'] to your Facebook Page ID (NOT 'act_...')."
            )

        from facebook_business.adobjects.page import Page
        _page_probe = Page(pid).api_get(fields=["id", "name"])
        if not _page_probe or not _page_probe.get("id"):
            raise RuntimeError("Provided PAGE_ID is not readable with this token.")
    except Exception as _pg_err:
        raise RuntimeError(
            f"Page validation failed for PAGE_ID={cfg['page_id']}. "
            "Use a real Facebook Page ID and ensure asset access from this ad account/token."
        ) from _pg_err

    # Build plan (no writes)
    settings = dict(settings or {})
    settings["campaign_name"] = cfg.get("campaign_name")
    plan = _plan_upload(
    account,
    campaign_id=cfg["campaign_id"],
    adset_prefix=cfg["adset_prefix"],
    page_id=cfg["page_id"],
    uploaded_files=uploaded_files,
    settings=settings,
)
    if simulate:
        return plan  # caller renders the plan; nothing is created

    # Targeting (country/age + OS filtering)
    targeting = build_targeting_from_settings(
        country=plan["country"],
        age_min=plan["age_min"],
        settings=settings,
    )

    # Optimization goal
        # ----- Optimization goal + promoted_object -----
    # ----- Optimization goal + promoted_object -----
    # ----- Optimization goal + promoted_object -----
    opt_goal_label = settings.get("opt_goal_label") or "앱 설치수 극대화"
    opt_goal_api = OPT_GOAL_LABEL_TO_API.get(opt_goal_label, "APP_INSTALLS")

    store_label = settings.get("app_store")  # "Google Play 스토어" or "Apple App Store"
    store_url = (settings.get("store_url") or "").strip()
    fb_app_id = (settings.get("fb_app_id") or "").strip()

    # sanitize store_url (Meta often rejects tracking params like fbclid)
    if store_url:
        store_url = sanitize_store_url(store_url)  # drop query/fragment

    # Build promoted_object only for app objectives
    promoted_object = None
    if opt_goal_api in ("APP_INSTALLS", "APP_EVENTS", "VALUE"):
        if not store_url:
            raise RuntimeError(
                "App objective selected. Please enter a valid store URL in Settings (Google Play or App Store)."
            )
        # Only send fields Meta accepts here
        promoted_object = {
            "object_store_url": store_url,
            **({"application_id": fb_app_id} if fb_app_id else {}),
        }

    # ----- Create ad set (ALWAYS run this, passing promoted_object if any) -----
    adset_id = None
    try:
        adset_id = create_creativetest_adset(
            account=account,
            campaign_id=cfg["campaign_id"],
            adset_name=plan["adset_name"],
            targeting=targeting,
            daily_budget_usd=plan["budget_usd_per_day"],
            start_iso=plan["start_iso"],
            optimization_goal=opt_goal_api,
            promoted_object=promoted_object,
            end_iso=plan.get("end_iso"),  # may be None; only used if present
)
    except Exception:
        # Let outer try/except print the full traceback on screen
        raise

    if not adset_id:
        raise RuntimeError(
            "Ad set was not created (no ID returned). Check the error above and fix settings/permissions."
        )

    # ----- Create ads (one per MP4) -----
    ad_name_prefix = settings.get("ad_name_prefix") if settings.get("ad_name_mode") == "Prefix + filename" else None
    try:
        st.session_state["ig_actor_id_from_page"] = ig_actor_id_from_page
    except Exception:
        pass

    upload_videos_create_ads(
        account=account,
        page_id=str(cfg["page_id"]),
        adset_id=adset_id,
        uploaded_files=uploaded_files,
        ad_name_prefix=ad_name_prefix,
        
        # store_url=(settings.get("store_url") or "").strip() or None, # <-- OLD (used raw settings)
        store_url=store_url, # <-- FIX: Use the SANITIZED store_url variable from line 1073
        
        try_instagram=True,
    ) 

    plan["adset_id"] = adset_id
    return plan
# ----- Streamlit UI -----------------------------------------------------------

st.set_page_config(page_title="Creative 자동 업로드", page_icon="🎮", layout="wide")

st.title("🎮 Creative 자동 업로드")
st.caption("게임별 크리에이티브를 다운받고, 설정에 따라 자동으로 업로드합니다.")
init_state()
init_remote_state()

# --- XP HERO default app config (App ID + Store URL) ---
GAME_DEFAULTS = {
    "XP HERO": {
        "fb_app_id": "519275767201283",
        "store_url": "https://play.google.com/store/apps/details?id=io.supercent.weaponrpg",
    },
    "Dino Universe": {
        "fb_app_id": "1665399243918955",
        "store_url": "https://play.google.com/store/apps/details?id=io.supercent.ageofdinosaurs",
    },
    "Snake Clash": {
        "fb_app_id": "1205179980183812",
        "store_url": "https://play.google.com/store/apps/details?id=io.supercent.linkedcubic",
    },
    "Pizza Ready": {
        "fb_app_id": "1475920199615616",
        "store_url": "https://play.google.com/store/apps/details?id=io.supercent.pizzaidle",
    },
    "Cafe Life": {
        "fb_app_id": "1343040866909064",
        "store_url": "https://play.google.com/store/apps/details?id=com.fireshrike.h2",
    },
    "Suzy's Restaurant": {
        "fb_app_id": "836273807918279",
        "store_url": "https://play.google.com/store/apps/details?id=com.corestudiso.suzyrest",
    },
    "Office Life": {
        "fb_app_id": "1570824996873176",
        "store_url": "https://play.google.com/store/apps/details?id=com.funreal.corporatetycoon",
    },
    "Lumber Chopper": {
        "fb_app_id": "2824067207774178",
        "store_url": "https://play.google.com/store/apps/details?id=dasi.prs2.lumberchopper",
    },
    "Burger Please": {
        "fb_app_id": "2967105673598896",
        "store_url": "https://play.google.com/store/apps/details?id=io.supercent.burgeridle",
    },
    "Prison Life": {
        "fb_app_id": "6564765833603067",
        "store_url": "https://play.google.com/store/apps/details?id=io.supercent.prison",
    },
}

# Apply defaults WITHOUT overwriting existing user settings
for game, defaults in GAME_DEFAULTS.items():
    cur = st.session_state.settings.get(game, {})  # may be {}
    if not cur.get("fb_app_id") and defaults.get("fb_app_id"):
        cur["fb_app_id"] = defaults["fb_app_id"]
    if not cur.get("store_url") and defaults.get("store_url"):
        cur["store_url"] = defaults["store_url"]
    st.session_state.settings[game] = cur
# --- end per-game defaults ---
# --- end XP HERO defaults ---

NUM_GAMES = 10
GAMES = game_tabs(NUM_GAMES)

accepted_types = ["mp4", "mpeg4"]


_tabs = st.tabs(GAMES)

for i, game in enumerate(GAMES):
    with _tabs[i]:
        # 전체 영역을 고정된 2열 레이아웃으로: 왼쪽(게임/Drive), 오른쪽(Settings)
        left_col, right_col = st.columns([2, 1], gap="large")

        # =========================
        # LEFT COLUMN: 게임 이름 + 플랫폼 선택 + Drive/버튼들
        # =========================
                # =========================
        # LEFT COLUMN: 게임 이름 + 플랫폼 선택 + Drive/버튼들
        # =========================
        with left_col:
            left_card = st.container(border=True)
            with left_card:
                st.subheader(game)

                # --- Platform selector (Unity는 그대로 유지) ---
                platform = st.radio(
                    "플랫폼 선택",
                    ["Facebook", "Unity Ads"],
                    index=0,
                    horizontal=True,
                    key=f"platform_{i}",
                )

                if platform == "Facebook":
                    # --- Import videos from Google Drive folder (server-side) ---
                    st.markdown("**구글 드라이브에서 Creative Videos를 가져옵니다**")
                    drv_input = st.text_input(
                        "Drive folder URL or ID",
                        key=f"drive_folder_{i}",
                        placeholder="https://drive.google.com/drive/folders/<FOLDER_ID>",
                    )

                    # Advanced option: hidden by default
                    with st.expander("Advanced import options", expanded=False):
                        workers = st.number_input(
                            "Parallel workers",
                            min_value=1,
                            max_value=16,
                            value=8,
                            key=f"drive_workers_{i}",
                            help="Higher = more simultaneous downloads (faster) but more load / chance of throttling.",
                        )

                    if st.button("드라이브에서 Creative 가져오기", key=f"drive_import_{i}"):
                        try:
                            overall = st.progress(0, text="0/0 • waiting…")
                            log_box = st.empty()
                            lines: List[str] = []
                            imported_accum: List[Dict] = []

                            import time
                            last_flush = [0.0]  # <-- mutable holder instead of nonlocal

                            def _on_progress(done: int, total: int, name: str, err: str | None):
                                pct = int((done / max(total, 1)) * 100)
                                label = f"{done}/{total}"
                                if name:
                                    label += f" • {name}"
                                if err:
                                    lines.append(f"❌ {name}  —  {err}")
                                else:
                                    lines.append(f"✅ {name}")

                                now = time.time()
                                # Only update UI every ~0.3s or on final item
                                if (now - last_flush[0]) > 0.3 or done == total:
                                    overall.progress(pct, text=label)
                                    log_box.write("\n".join(lines[-200:]))
                                    last_flush[0] = now

                            with st.status("Importing videos from Drive folder...", expanded=True) as status:
                                imported = _run_drive_import(
                                    drv_input,
                                    max_workers=int(workers),
                                    on_progress=_on_progress,
                                )
                                imported_accum.extend(imported)
                                lst = st.session_state.remote_videos.get(game, [])
                                lst.extend(imported_accum)
                                st.session_state.remote_videos[game] = lst

                                status.update(
                                    label=f"Drive import complete: {len(imported_accum)} file(s)",
                                    state="complete",
                                )
                                if isinstance(imported, dict) and imported.get("errors"):
                                    st.warning("Some files failed:\n- " + "\n- ".join(imported["errors"]))

                            st.success(f"Imported {len(imported_accum)} video(s) from the folder.")
                            if len(imported_accum) < 1:
                                st.info("No eligible videos found. Check access, file types, or folder contents.")
                        except Exception as e:
                            st.exception(e)
                            st.error(
                                "Could not import from this folder. "
                                "Make sure your service account has access and the folder contains videos."
                            )

                    # --- Shared list & clear button for all remote videos (URL + Drive) ---
                    remote_list = st.session_state.remote_videos.get(game, [])

                    st.caption("다운로드된 Creatives:")
                    if remote_list:
                        for it in remote_list[:50]:
                            st.write("•", it["name"])
                    else:
                        st.write("- (현재 저장된 URL/Drive 영상 없음)")

                    # 🔹 URL/Drive 영상만 지우는 버튼 (항상 표시)
                    if st.button("URL/Drive 영상만 초기화", key=f"clearurl_{i}"):
                        if remote_list:
                            st.session_state.remote_videos[game] = []
                            st.info("Cleared URL/Drive videos for this game.")
                            st.rerun()
                        else:
                            st.info("삭제할 URL/Drive 영상이 없습니다.")

                    ok_msg_placeholder = st.empty()
                    cont = st.button("Creative Test 업로드하기", key=f"continue_{i}")
                    clr = st.button("전체 초기화", key=f"clear_{i}")

                else:
                    # =========================
                    # UNITY ADS FLOW
                    # =========================
                    st.markdown("### Unity Ads")

                    remote_list = st.session_state.remote_videos.get(game, []) or []
                    st.caption("다운로드된 Creatives (Unity에 업로드 예정):")
                    if remote_list:
                        for it in remote_list[:50]:
                            st.write("•", it["name"])
                        if len(remote_list) > 50:
                            st.write(f"... 외 {len(remote_list) - 50}개")
                    else:
                        st.write("- (현재 저장된 URL/Drive 영상 없음)")

                    unity_ok_placeholder = st.empty()
                    cont_unity = st.button("Unity Ads에 업로드하기", key=f"unity_continue_{i}")
                    clr_unity = st.button("전체 초기화 (Unity용)", key=f"unity_clear_{i}")

        # =========================
        # RIGHT COLUMN: Settings (Facebook 선택일 때만)
        # =========================
                # =========================
        # RIGHT COLUMN: Settings (플랫폼별)
        # =========================
        if platform == "Facebook":
            with right_col:
                right_card = st.container(border=True)
                with right_card:
                    ensure_settings_state()

                    # 🔹 게임 이름과 묶인 Settings 헤더
                    st.markdown(f"### {game} Settings")

                    cur = st.session_state.settings.get(game, {})

                    # 1) 광고 세트 이름: campaign_name + "_nth"
                                        # 1) 광고 세트 이름: campaign_name + "_nth"
                    suffix_number = st.number_input(
                        "광고 세트 접미사 n(…_nth)",
                        min_value=1,
                        step=1,
                        value=int(cur.get("suffix_number", 1)),
                        help="Ad set will be named as <campaign_name>_<n>th or <campaign_name>_<n>th_YYYYmmdd",
                        key=f"suffix_{i}",
                    )

                    # 선택: 시작 날짜(launch Saturday)의 YYYYmmdd를 광고 세트 이름에 추가
                    add_launch_date = st.checkbox(
                        "Launch 날짜 추가",
                        value=bool(cur.get("add_launch_date", False)),
                        key=f"add_launch_date_{i}",
                        help="예: weaponrpg_aos_facebook_us_creativetest_35th_20251122",
                    )

                    # 2) 앱 홍보 - 스토어 선택 (기본: Google Play)
                    app_store = st.selectbox(
                        "모바일 앱 스토어",
                        ["Google Play 스토어", "Apple App Store"],
                        index=0 if cur.get("app_store", "Google Play 스토어") == "Google Play 스토어" else 1,
                        key=f"appstore_{i}",
                    )

                    # 3) 앱 연결 정보
                    fb_app_id = st.text_input(
                        "Facebook App ID",
                        value=cur.get("fb_app_id", ""),
                        key=f"fbappid_{i}",
                        help="설치 추적을 연결하려면 FB App ID를 입력하세요(선택).",
                    )
                    store_url = st.text_input(
                        "구글 스토어 URL",
                        value=cur.get("store_url", ""),
                        key=f"storeurl_{i}",
                        help="예) https://play.google.com/store/apps/details?id=... (쿼리스트링/트래킹 파라미터 제거 권장)",
                    )

                    # 4) 성과 목표 (기본: 앱 설치수 극대화)
                    opt_goal_label = st.selectbox(
                        "성과 목표",
                        list(OPT_GOAL_LABEL_TO_API.keys()),
                        index=list(OPT_GOAL_LABEL_TO_API.keys()).index(cur.get("opt_goal_label", "앱 설치수 극대화")),
                        key=f"optgoal_{i}",
                    )

                    # 5) 기여 설정 (표시용 안내)
                    st.caption("기여 설정: 클릭 1일(기본), 참여한 조회/조회 없음 — Facebook에서 고정/제한될 수 있습니다.")

                    # 6) 예산 (per-video × 개수)
                    budget_per_video_usd = st.number_input(
                        "영상 1개당 일일 예산 (USD)",
                        min_value=1,
                        value=int(cur.get("budget_per_video_usd", 10)),
                        key=f"budget_per_video_{i}",
                        help="총 일일 예산 = (업로드/선택된 영상 수) × 이 값",
                    )

                    # 7) 예약 (기본: 토 9:00)
                    default_start_iso = next_sat_0900_kst()
                    start_iso = st.text_input(
                        "시작 날짜/시간 (ISO, KST)",
                        value=cur.get("start_iso", default_start_iso),
                        help="예: 2025-11-15T00:00:00+09:00 (종료일은 자동으로 꺼지지 않도록 설정하지 않습니다)",
                        key=f"start_{i}",
)

                    # 8) 타겟 위치 (기본: United States)
                    country = st.text_input("국가", value=cur.get("country", "US"), key=f"country_{i}")

                    # 9) 최소 연령 (기본 18)
                    age_min = st.number_input(
                        "최소 연령",
                        min_value=13,
                        value=int(cur.get("age_min", 18)),
                        key=f"age_{i}",
                    )

                    # 10) OS/버전 (기본: Android only, 6.0+)
                    os_choice = st.selectbox(
                        "Target OS",
                        ["Both", "Android only", "iOS only"],
                        index={"Both": 0, "Android only": 1, "iOS only": 2}[cur.get("os_choice", "Android only")],
                        key=f"os_choice_{i}",
                    )

                    if os_choice in ("Both", "Android only"):
                        min_android_label = st.selectbox(
                            "Min Android version",
                            list(ANDROID_OS_CHOICES.keys()),
                            index=list(ANDROID_OS_CHOICES.keys()).index(cur.get("min_android_label", "6.0+")),
                            key=f"min_android_{i}",
                        )
                    else:
                        min_android_label = "None (any)"

                    if os_choice in ("Both", "iOS only"):
                        min_ios_label = st.selectbox(
                            "Min iOS version",
                            list(IOS_OS_CHOICES.keys()),
                            index=list(IOS_OS_CHOICES.keys()).index(cur.get("min_ios_label", "None (any)")),
                            key=f"min_ios_{i}",
                        )
                    else:
                        min_ios_label = "None (any)"

                    min_android_os_token = ANDROID_OS_CHOICES[min_android_label] if os_choice in ("Both", "Android only") else None
                    min_ios_os_token = IOS_OS_CHOICES[min_ios_label] if os_choice in ("Both", "iOS only") else None

                    # (선택) 광고 이름 규칙
                    ad_name_mode = st.selectbox(
                        "Ad name",
                        ["Use video filename", "Prefix + filename"],
                        index=1 if cur.get("ad_name_mode") == "Prefix + filename" else 0,
                        key=f"adname_mode_{i}",
                    )
                    ad_name_prefix = ""
                    if ad_name_mode == "Prefix + filename":
                        ad_name_prefix = st.text_input(
                            "Ad name prefix",
                            value=cur.get("ad_name_prefix", ""),
                            key=f"adname_prefix_{i}",
                        )

                    # --- Save settings back into session_state ---
                    st.session_state.settings[game] = {
                        "suffix_number": int(suffix_number),
                        "add_launch_date": bool(add_launch_date),
                        "app_store": app_store,
                        "fb_app_id": fb_app_id.strip(),
                        "store_url": store_url.strip(),
                        "opt_goal_label": opt_goal_label,
                        "budget_per_video_usd": int(budget_per_video_usd),
                        "start_iso": start_iso.strip(),
                        "country": (country or "US").strip(),
                        "age_min": int(age_min),
                        "os_choice": os_choice,
                        "min_android_label": min_android_label,
                        "min_ios_label": min_ios_label,
                        "min_android_os_token": min_android_os_token,
                        "min_ios_os_token": min_ios_os_token,
                        "ad_name_mode": ad_name_mode,
                        "ad_name_prefix": ad_name_prefix.strip(),
                        "game_key": game,
                    }

        elif platform == "Unity Ads":
            # 👉 Unity용 설정 패널도 테두리 카드 안에 렌더링
            with right_col:
                unity_card = st.container(border=True)
                render_unity_settings_panel(unity_card, game, i)

        # --- Handle button actions after BOTH columns are drawn (Facebook only) ---
        if platform == "Facebook":
            if cont:
                # Only use server-downloaded (Drive) videos now
                remote_list = st.session_state.remote_videos.get(game, [])
                combined = remote_list

                ok, msg = validate_count(combined)
                if not ok:
                    ok_msg_placeholder.error(msg)
                else:
                    try:
                        st.session_state.uploads[game] = combined
                        settings = st.session_state.settings.get(game, {})
                        plan = upload_to_facebook(game, combined, settings)

                        # (기존 _render_summary 정의 및 사용 그대로 유지)
                        def _render_summary(plan: dict, settings: dict, created: bool):
                            ...
                        if isinstance(plan, dict) and plan.get("adset_id"):
                            ok_msg_placeholder.success(ok_msg_placeholder.success(msg + " Uploaded to Meta (ads created as ACTIVE, scheduled by start time)."))
                            _render_summary(plan, settings, created=True)
                        else:
                            ok_msg_placeholder.error(
                                "Meta upload did not return an ad set ID. "
                                "Check the error above and your settings/permissions."
                            )
                            if isinstance(plan, dict):
                                _render_summary(plan, settings, created=False)
                    except Exception as e:
                        import traceback
                        st.exception(e)
                        tb = traceback.format_exc()
                        st.error("Meta upload failed. See full error below ⬇️")
                        st.code(tb, language="python")

            if clr:
                st.session_state.uploads.pop(game, None)
                st.session_state.remote_videos.pop(game, None)
                st.session_state.settings.pop(game, None)
                st.session_state[f"clear_uploader_flag_{i}"] = True
                ok_msg_placeholder.info("Cleared saved uploads, URL videos, and settings for this game.")
                st.rerun()
                # --- Unity Ads: handle upload & clear actions ---
        if platform == "Unity Ads":
            unity_settings = get_unity_settings(game)

            if 'cont_unity' in locals() and cont_unity:
                remote_list = st.session_state.remote_videos.get(game, []) or []
                ok, msg = validate_count(remote_list)
                if not ok:
                    unity_ok_placeholder.error(msg)
                else:
                    try:
                        summary = upload_unity_creatives_to_campaign(
                            game=game,
                            videos=remote_list,
                            settings=unity_settings,
                        )

                        n_creatives = len(summary.get("creative_ids") or [])
                        removed = summary.get("removed_ids") or []
                        errors = summary.get("errors") or []

                        if n_creatives > 0:
                            unity_ok_placeholder.success(
                                f"{msg} Unity Ads에 {n_creatives}개 크리에이티브(creative packs)를 생성하고 "
                                f"캠페인에 연결했습니다."
                            )
                        else:
                            unity_ok_placeholder.warning(
                                "Unity Ads 호출은 성공했지만 생성된 크리에이티브 ID가 없습니다. "
                                "Unity 대시보드에서 실제 상태를 확인해 주세요."
                            )

                        if removed:
                            st.caption(
                                f"캠페인에서 이전 크리에이티브 {len(removed)}개를 제거했습니다 "
                                f"(예: {removed[:10]})"
                            )

                        if errors:
                            st.error(
                                "일부 단계에서 오류가 발생했습니다:\n"
                                + "\n".join(f"- {e}" for e in errors[:20])
                                + ("\n..." if len(errors) > 20 else "")
                            )

                    except Exception as e:
                        import traceback
                        st.exception(e)
                        tb = traceback.format_exc()
                        unity_ok_placeholder.error("Unity Ads 업로드 실패. 아래 오류 로그를 확인하세요.")
                        st.code(tb, language="python")

            if 'clr_unity' in locals() and clr_unity:
                st.session_state.uploads.pop(game, None)
                st.session_state.remote_videos.pop(game, None)
                st.session_state.settings.pop(game, None)
                st.session_state.unity_settings.pop(game, None)
                st.session_state[f"clear_uploader_flag_{i}"] = True
                unity_ok_placeholder.info("해당 게임의 업로드/설정(페북+유니티)을 모두 초기화했습니다.")
                st.rerun()



# Summary table
st.subheader("업로드 완료된 게임")
if st.session_state.uploads:
    data = {"게임": [], "업로드 파일": []}
    for g, files in st.session_state.uploads.items():
        data["게임"].append(g)
        data["업로드 파일"].append(len(files))
    st.dataframe(data, hide_index=True)
else:
    st.info("No uploads saved yet. Go to a tab and click **Creative Test 업로드하기** after importing videos.")
