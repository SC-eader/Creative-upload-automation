"""
Streamlit UI: 10 games, each with an upload area that expects 10–12 creative files.

Run:
  streamlit run streamlit_app.py
# Tip: for better Google Drive reliability, install gdown:
#   pip install gdown
# If absent, we still try direct downloads with robust retries.

Notes:
- Supported files: JPG/PNG/MP4 (tweak below).
f"- Each tab accepts **video files only**: {', '.join(t.upper() for t in accepted_types)} (limit {MAX_UPLOAD_MB}MB per file)."- "Creative Test 업로드하기" creates a (paused) ad set + ads via Meta Marketing API using per-title settings.
"""

from __future__ import annotations

import os
from typing import Dict, List
import io
from datetime import datetime, timedelta, timezone
import tempfile
import logging
import requests
import pathlib
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
from streamlit.components.v1 import html as components_html 

try:
    from drive_import import import_drive_folder_videos_parallel as import_drive_folder_videos
    _DRIVE_IMPORT_SUPPORTS_PROGRESS = True
except ImportError:
    from drive_import import import_drive_folder_videos  # old signature: (folder_url_or_id) -> list[{"name","path"}]
    _DRIVE_IMPORT_SUPPORTS_PROGRESS = False

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

def ext(fname: str) -> str:
    """Return lowercase file extension including the dot (e.g., '.jpg')."""
    return os.path.splitext(fname)[1].lower()

def test_fb_setup() -> dict:
    """Quick read-only check: account info and presence of the XP HERO campaign."""
    acct = init_fb_from_secrets()
    info = acct.api_get(fields=["name", "account_status", "currency"])
    has_ct = False
    for c in acct.get_campaigns(fields=["id","name"], params={"limit": 200}):
        if c.get("id") == "120218934861590118":
            has_ct = True
            break
    return {"account": info, "creative_test_campaign_found": has_ct}

def debug_fb_identity() -> dict:
    """
    Minimal identity check with the current token.
    Returns {'me': {id,name}, 'account': {name,currency}} without using app creds.
    """
    _require_fb()
    from facebook_business.adobjects.user import User

    me = User(fbid="me").api_get(fields=["id", "name"])
    acct = AdAccount("act_692755193188182").api_get(fields=["name", "currency"])

    return {"me": {"id": me["id"], "name": me["name"]}, "account": {"name": acct["name"], "currency": acct["currency"]}}

# --- Friendly FB error helper ---
def _friendly_fb_error(e: Exception) -> str:
    """
    Turn common Marketing API errors into actionable guidance.
    Returns a short HTML string for st.markdown(..., unsafe_allow_html=True).
    """
    try:
        from facebook_business.exceptions import FacebookRequestError
        if isinstance(e, FacebookRequestError):
            code = getattr(e, "api_error_code", lambda: None)()
            sub = getattr(e, "api_error_subcode", lambda: None)()
            msg = getattr(e, "api_error_message", lambda: "")()

            # Permission errors
            if code == 200 and sub == 1815066:
                return (
                    "<b>Permission error (code 200 / subcode 1815066)</b><br/>"
                    "이 광고계정에서 광고를 <b>생성</b>할 권한이 없습니다.<br/><br/>"
                    "<b>해결 방법</b><ol>"
                    "<li>Business Settings → Ad Accounts → <code>act_692755193188182</code> → People → "
                    "당신(또는 시스템 사용자)에게 <b>Advertiser</b> 이상 권한 부여</li>"
                    "<li>액세스 토큰에 <code>ads_management</code> 포함 및 자산 연결 확인</li>"
                    "<li>해당 Page가 이 광고계정과 Connected assets로 연결</li>"
                    "<li>앱 Live / 비즈니스 인증 & 2FA 확인</li>"
                    "</ol>"
                    f"<div style='color:#6b7280'>Raw: {msg}</div>"
                )

            # Invalid Page ID inside object_story_spec (your current case)
            if code == 100 and sub == 1443120:
                return (
                    "<b>Page ID error (code 100 / subcode 1443120)</b><br/>"
                    "object_story_spec.page_id 로 <b>광고계정 ID</b>가 전달되었습니다.<br/>"
                    "<b>해결 방법</b><ol>"
                    "<li><code>page_id</code> 에는 <b>Facebook Page ID</b> 를 넣으세요 (예: 1xxxxxxxxxxxxxx).</li>"
                    "<li><code>act_...</code> 또는 그 숫자만 전달하면 안 됩니다.</li>"
                    "<li>토큰/비즈니스 자산 연결로 그 Page 에 접근 가능한지도 확인하세요.</li>"
                    "</ol>"
                    f"<div style='color:#6b7280'>Raw: {msg}</div>"
                )

            if code == 200:
                return (
                    "<b>Permission error</b> — 토큰 권한/자산 연결을 확인하세요.<br/>"
                    f"<div style='color:#6b7280'>Raw: {msg} (code={code}, subcode={sub})</div>"
                )
    except Exception:
        pass
    return "Meta API call failed. Check token scopes, ad account role, Page/ad account connections, and payload."
def list_manageable_pages() -> list[dict]:
    """Return [{'id': '<PAGE_ID>', 'name': '<Page Name>'}] for Pages this token can access."""
    _require_fb()
    from facebook_business.adobjects.user import User
    token = (st.secrets.get("access_token") or "").strip()
    if not token:
        raise RuntimeError("Missing access_token in st.secrets")
    # FacebookAdsApi.init already called in init_fb_from_secrets when needed
    pages = User(fbid="me").get_accounts(fields=["id", "name"], params={"limit": 500})
    return [{"id": p["id"], "name": p["name"]} for p in pages]

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
        raise RuntimeError(f"Page validation failed for PAGE_ID={pid}. Use a real Facebook Page ID and ensure the token can read it.") from e
    iba = (p.get("instagram_business_account") or {}).get("id")
    return {"id": p["id"], "name": p["name"], "instagram_business_account_id": iba}


def preflight_permissions(target_act_id: str, page_id: str | None = None) -> dict:
    """
    Read-only checks to verify that the current token can CREATE in the target ad account.
    Returns a dict with diagnostic info and pass/fail booleans.
    """
    _require_fb()
    from facebook_business.adobjects.user import User
    from facebook_business.adobjects.adaccount import AdAccount

    me = User(fbid="me").api_get(fields=["id", "name"])
    # Fetch ad accounts the token can see + listed permissions strings
    adaccts = User(fbid="me").get_ad_accounts(
        fields=["id", "name", "account_status", "permissions"], params={"limit": 500}
    )
    acct_rows = [a for a in adaccts]
    acct_map = {a["id"]: a for a in acct_rows}

    have_target = target_act_id in acct_map
    perm_list = (acct_map.get(target_act_id, {}).get("permissions") or []) if have_target else []
    # Creation requires at least ADVERTISE (or ADMIN)
    can_create = any(p in ("ADVERTISE", "ADMIN") for p in perm_list)

    page_ok = None
    if page_id:
        try:
            # Minimal probe: can we read page? (If not, system user/page asset may be missing)
            from facebook_business.adobjects.page import Page
            _p = Page(page_id).api_get(fields=["id", "name"])
            page_ok = True if _p and _p.get("id") else False
        except Exception:
            page_ok = False

    return {
        "me": {"id": me["id"], "name": me["name"]},
        "target_account_seen": have_target,
        "target_permissions": perm_list,
        "can_create_in_target": bool(can_create),
        "page_access_ok": page_ok,
        "adaccounts_overview": [
            {
                "id": a.get("id"),
                "name": a.get("name"),
                "status": a.get("account_status"),
                "permissions": a.get("permissions"),
            }
            for a in acct_rows
        ],
    }
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

def next_sat_0000_and_mon_1200_kst(today: datetime | None = None) -> tuple[str, str]:
    """Return default (start_iso, end_iso) = next Saturday 00:00 and next Monday 12:00 in KST."""
    now = (today or datetime.now(ASIA_SEOUL)).astimezone(ASIA_SEOUL)
    base = now.replace(hour=0, minute=0, second=0, microsecond=0)
    # Monday=0 ... Saturday=5, Sunday=6
    days_until_sat = (5 - base.weekday()) % 7 or 7
    start_dt = (base + timedelta(days=days_until_sat)).replace(hour=0, minute=0)
    end_dt = (start_dt + timedelta(days=2)).replace(hour=12, minute=0)  # Sat -> Mon 12:00
    return start_dt.isoformat(), end_dt.isoformat()

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

def _normalize_drive_url(url: str) -> str:
    """Turn common Google Drive share links into direct-download links."""
    url = url.strip()
    if "drive.google.com/file/d/" in url:
        try:
            file_id = url.split("/file/d/")[1].split("/")[0]
            return f"https://drive.google.com/uc?export=download&id={file_id}"
        except Exception:
            return url
    return url

def fetch_url_to_tmp(url: str, timeout: int = 900) -> dict:
    """
    Download a remote video URL to a temp file and return {'name': str, 'path': str}.
    - Retries with exponential backoff on transient network/SSL/server errors.
    - Smaller chunk size to reduce TLS buffer errors on some CDNs.
    - Google Drive fallback via gdown for large files / confirm tokens.
    """
    import tempfile, pathlib, time
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    def _direct_download(u: str) -> tuple[str, str]:
        sess = requests.Session()
        # robust retry policy
        retry = Retry(
            total=5,
            connect=5,
            read=5,
            backoff_factor=1.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET", "HEAD"]),
            raise_on_status=False,
        )
        sess.mount("http://", HTTPAdapter(max_retries=retry))
        sess.mount("https://", HTTPAdapter(max_retries=retry))

        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; CreativeUploader/1.0)",
            "Accept": "*/*",
            "Accept-Encoding": "identity",  # avoid gzip chunking oddities on some CDNs
            "Connection": "keep-alive",
        }

        # follow redirects; separate connect/read timeouts
        with sess.get(u, headers=headers, stream=True, timeout=(10, timeout), allow_redirects=True) as r:
            r.raise_for_status()
            ctype = r.headers.get("Content-Type", "").lower()
            disp = r.headers.get("Content-Disposition", "")
            # filename
            name = None
            if "filename=" in disp:
                name = disp.split("filename=")[-1].strip(' \'"')
            if not name:
                name = u.split("?")[0].rstrip("/").split("/")[-1] or "video.mp4"
            # ensure extension when content-type says video
            if not name.lower().endswith((".mp4", ".mpeg4")) and ("video" in ctype or "mp4" in ctype):
                name = f"{name}.mp4"
            # quick sanity
            if not ("video" in ctype or name.lower().endswith((".mp4", ".mpeg4"))):
                raise ValueError(f"URL is not a video. Content-Type={ctype!r}")

            # write to tmp in smaller chunks (256KB)
            suffix = pathlib.Path(name).suffix.lower() or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                for chunk in r.iter_content(chunk_size=256 * 1024):
                    if chunk:
                        tmp.write(chunk)
                local_path = tmp.name

        return pathlib.Path(name).name, local_path

    def _maybe_gdrive(u: str) -> tuple[str, str] | None:
        # Use gdown for Drive links (handles confirm prompts/large files/throttling)
        try:
            from urllib.parse import urlparse
            netloc = urlparse(u).netloc.lower()
            if "drive.google.com" not in netloc:
                return None
            try:
                import gdown  # type: ignore
            except Exception:
                return None  # gdown not installed; skip
            # gdown will figure out id/confirm; fuzzy handles various link shapes
            import tempfile
            out_fd, out_path = tempfile.mkstemp(suffix=".mp4")
            os.close(out_fd)
            ok = gdown.download(url=u, output=out_path, quiet=True, fuzzy=True)
            if not ok:
                return None
            # pick a decent display name
            name = u.split("?")[0].rstrip("/").split("/")[-1] or "video.mp4"
            if not name.lower().endswith((".mp4", ".mpeg4")):
                name += ".mp4"
            return name, out_path
        except Exception:
            return None

    # Normalize common Drive share link
    url = _normalize_drive_url(url)

    # Try Drive-specialized path first (if applicable)
    gd = _maybe_gdrive(url)
    if gd:
        name, path = gd
        return {"name": name, "path": path}

    # Otherwise use robust direct downloader with retries
    last_err = None
    for attempt in range(1, 6):
        try:
            name, path = _direct_download(url)
            return {"name": name, "path": path}
        except Exception as e:
            last_err = e
            # small sleep with backoff
            time.sleep(min(2 * attempt, 10))

    raise RuntimeError(f"Download failed after retries: {last_err}")

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

def next_nth_suffix(account: AdAccount, prefix: str) -> str:
    """Scan existing ad sets and return next ordinal like '35th' for a given prefix."""
    ordinals = []
    for a in account.get_ad_sets(fields=["name"], params={"limit": 200}):
        n = a.get("name", "")
        if n.startswith(prefix) and n.split("_")[-1].endswith("th"):
            try:
                ordinals.append(int(n.split("_")[-1].replace("th", "")))
            except ValueError:
                pass
    return f"{(max(ordinals) + 1) if ordinals else 1}th"

def create_creativetest_adset(
    account: AdAccount,
    *,
    campaign_id: str,
    adset_name: str,
    targeting: dict,
    daily_budget_usd: int,
    start_iso: str,
    end_iso: str,
    optimization_goal: str,  # API token string like "APP_INSTALLS"
    promoted_object: dict | None = None,
) -> str:
    """Create a paused ad set with the given name/settings; return adset_id."""
    from facebook_business.adobjects.adset import AdSet

    adset = account.create_ad_set(
        fields=[],
        params={
            "name": adset_name,
            "campaign_id": campaign_id,
            "daily_budget": dollars_to_minor(daily_budget_usd),
            "billing_event": AdSet.BillingEvent.impressions,
            "optimization_goal": getattr(AdSet.OptimizationGoal, optimization_goal.lower(), AdSet.OptimizationGoal.app_installs),
            "bid_strategy": "LOWEST_COST_WITHOUT_CAP",
            "targeting": targeting,
            "status": AdSet.Status.paused,
            "start_time": start_iso,
            "end_time": end_iso,
            **({"promoted_object": promoted_object} if promoted_object else {}),
        },
    )
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
    max_workers_save: int = 4,
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
                    st.write(f"[Upload] ✅ All bytes acknowledged ({sent_bytes}/{file_size}).")
                    break

                # If Graph returns a stall window (no progress yet), ask again (no file chunk)
                if end_offset <= start_offset:
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

                st.write(f"[Upload] Sent [{start_offset},{end_offset}) → ack: start={new_start}, end={new_end}, sent={sent_bytes}/{file_size}")

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

    def wait_until_video_ready(video_id: str, *, timeout_s: int = 300, sleep_s: int = 5) -> bool:
        """
        Poll the uploaded AdVideo until it's usable. v24 exposes 'status' only.
        Returns True if we observe a ready-ish status before timeout.
        Defensive: if timeout_s <= 0 (misconfig), default to 300s to avoid spam loops.
        """
        from time import sleep, time
        from facebook_business.adobjects.advideo import AdVideo

        if timeout_s is None:
            timeout_s = 300  # guard against 0-second timeouts

        READY = {"ready", "processed", "finished", "available", "live", "published"}
        start = time()
        last = None

        # Compact, bounded polling — no repeated Streamlit warnings on every rerun
        while time() - start < timeout_s:
            try:
                v = AdVideo(video_id).api_get(fields=["status"])
                status = v.get("status")
                s = status if isinstance(status, str) else (
                    (status or {}).get("video_status")
                    or (status or {}).get("processing_phase")
                    or (status or {}).get("status")
                )
                last = s
                if isinstance(s, str) and s.lower() in READY:
                    st.write(f"[Encode] Video {video_id} ready: {s}")
                    return True
            except Exception as e:
                # Silent backoff; asset propagation can 404/4xx transiently
                last = getattr(e, "args", [str(e)])[0]

            sleep(sleep_s)

        st.info(f"[Encode] Video {video_id} not confirmed ready within {timeout_s}s (last={last}). Proceeding.")
        return False

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
    with ThreadPoolExecutor(max_workers=max_workers_save) as ex:
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

    # ---------- Stage 2: API calls (serial) ----------
        # ---------- Stage 2: API calls ----------
    # Phase A) Upload all videos first (no waiting yet)
    uploads, api_errors = [], []
    for item in persisted:  
        name, path = item["name"], item["path"]
        try:
            vid = upload_video_resumable(path)
            uploads.append({"name": name, "path": path, "video_id": vid, "ready": False})
            st.write(f"⬆️ Uploaded {name} → video_id={vid}")
        except Exception as e:
            api_errors.append(f"{name}: upload failed: {e}")

    # Phase B) Poll all videos concurrently (round-robin) with a single progress bar

    # Phase C) Create creatives & ads
    results = []
    for up in uploads:
        name, video_id = up["name"], up["video_id"]

        # ✅ Wait up to 300s for this specific video to be ready
        wait_until_video_ready(video_id, timeout_s=300, sleep_s=5)

        # --- START FIX: Fetch the auto-generated thumbnail ---
        try:
            # 'picture' is the field for the default thumbnail URL
            video_info = AdVideo(video_id).api_get(fields=["picture"])
            thumbnail_url = video_info.get("picture")
            if not thumbnail_url:
                raise RuntimeError("Video processed but no 'picture' (thumbnail) URL was returned.")
        except Exception as e:
            api_errors.append(f"{name}: Failed to fetch thumbnail: {e}")
            continue # Skip this video, move to the next
        # --- END FIX ---


        def _create_once(allow_ig: bool) -> str:
            
            # --- START FIX: Add image_url to video_data ---
            vd = {
                "video_id": video_id,
                "title": name,
                "message": "",
                "image_url": thumbnail_url  # Add the thumbnail URL here
            }
            # --- END FIX ---

            if store_url:
                # This 'store_url' variable is passed into the function
                vd["call_to_action"] = {"type": "INSTALL_MOBILE_APP", "value": {"link": store_url}}
            
            spec = {"page_id": page_id, "video_data": vd}
            
            if allow_ig and ig_actor_id:
                spec["instagram_actor_id"] = ig_actor_id
            creative = account.create_ad_creative(fields=[], params={"name": name, "object_story_spec": spec})
            ad = account.create_ad(
                fields=[],
                params={
                    "name": make_ad_name(name, ad_name_prefix),
                    "adset_id": adset_id,
                    "creative": {"creative_id": creative["id"]},
                    "status": Ad.Status.paused,
                },
            )
            return ad["id"]

        try:
            try:
                ad_id = _create_once(True)
            except FacebookRequestError as e:
                # If IG or not-ready issues, retry once without IG after a short wait
                msg = (e.api_error_message() or "").lower()
                if "instagram" in msg or "not ready" in msg or "processing" in msg:
                    time.sleep(5)
                    ad_id = _create_once(False)
                else:
                    raise
            results.append({"name": name, "ad_id": ad_id})
        except Exception as e:
            api_errors.append(f"{name}: creative/ad failed: {e}")

    if api_errors:
        st.error("Some ads failed to create:\n- " + "\n- ".join(api_errors))

    return results


def _plan_upload(account: AdAccount, *, campaign_id: str, adset_prefix: str, page_id: str, uploaded_files: list, settings: dict) -> dict:
    """Compute what would be created (no writes): ad set name, budget/schedule, and ad names."""
    # Schedule
    start_iso = settings.get("start_iso")
    end_iso = settings.get("end_iso")
    if not (start_iso and end_iso):
        start_iso, end_iso = next_sat_0000_and_mon_1200_kst()

    # Ad set suffix: user-selected n
    n = int(settings.get("suffix_number") or 1)
    suffix_str = f"{n}th"
    adset_name = f"{adset_prefix}_{suffix_str}"

    # Videos (local + any server-downloaded)
    allowed = {".mp4", ".mpeg4"}
    remote = st.session_state.remote_videos.get(settings.get("game_key", ""), []) or []
    def _name(u): return getattr(u, "name", None) or (u.get("name") if isinstance(u, dict) else "")
    def _is_video(u): return pathlib.Path(_name(u)).suffix.lower() in allowed
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
        "end_iso": end_iso,
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
            end_iso=plan["end_iso"],
            optimization_goal=opt_goal_api,
            promoted_object=promoted_object,
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
st.caption("Collect, validate, and upload creatives per game with configurable settings.")

with st.expander("🔧 Debug: server upload settings", expanded=False):
    try:
        st.write("server.maxUploadSize =", st.get_option("server.maxUploadSize"))
        st.write("server.maxMessageSize =", st.get_option("server.maxMessageSize"))
    except Exception as e:
        st.write("Could not read options:", e)
init_state()
init_remote_state()

# --- XP HERO default app config (App ID + Store URL) ---
GAME_DEFAULTS = {
    "XP HERO": {
        "fb_app_id": "519275767201283",
        "store_url": "https://play.google.com/store/apps/details?id=io.supercent.weaponrpg",
    },
    "Dino Universe": {
        "fb_app_id": "722279627640461",
        "store_url": "https://play.google.com/store/apps/details?id=io.supercent.ageofdinosaurs",
    },
    "Snake Clash": {
        "fb_app_id": "102629376270269",
        "store_url": "https://play.google.com/store/apps/details?id=io.supercent.linkedcubic",
    },
    "Pizza Ready": {
        "fb_app_id": "115469331609377",
        "store_url": "https://play.google.com/store/apps/details?id=io.supercent.pizzaidle",
    },
    "Cafe Life": {
        "fb_app_id": "607125849162050",
        "store_url": "https://play.google.com/store/apps/details?id=com.fireshrike.h2",
    },
    "Suzy's Restaurant": {
        "fb_app_id": "608844445639579",
        "store_url": "https://play.google.com/store/apps/details?id=com.corestudiso.suzyrest",
    },
    "Office Life": {
        "fb_app_id": "743103455548945",
        "store_url": "https://play.google.com/store/apps/details?id=com.funreal.corporatetycoon",
    },
    "Lumber Chopper": {
        "fb_app_id": "729295830272629",
        "store_url": "https://play.google.com/store/apps/details?id=dasi.prs2.lumberchopper",
    },
    "Burger Please": {
        "fb_app_id": "729295830272629",
        "store_url": "https://play.google.com/store/apps/details?id=io.supercent.burgeridle",
    },
    "Prison Life": {
        "fb_app_id": "368211056367446",
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

# Global notes panel
with st.expander("How this works", expanded=True):
    st.markdown(
        f"- There are **{NUM_GAMES}** tabs.\n"
        f"- Each tab accepts **video files only**: {', '.join(t.upper() for t in accepted_types)}.\n"
        f"- There is **no limit** on the number of videos you can upload per tab.\n"
        "- Use the Settings panel to customize targeting and schedule per game.\n"
        "- Click **Creative Test 업로드하기** to create a paused ad set + ads on Meta (or Dry run to preview).\n"
        "- Use **Clear** to remove previously saved uploads/settings for that game."
    )

_tabs = st.tabs(GAMES)

for i, game in enumerate(GAMES):
    with _tabs[i]:
        st.subheader(game)
        left, right = st.columns([2, 1], gap="large")

        # ----- LEFT: uploader + live preview + ACTIONS -----
        with left:
            # 1) If we set the clear flag in the previous run, clear the uploader BEFORE creating it
            if st.session_state.get(f"clear_uploader_flag_{i}"):
                st.session_state.pop(f"uploader_{i}", None)          # remove widget state
                st.session_state.pop(f"clear_uploader_flag_{i}", None)

            uploaded = st.file_uploader(
                "Upload video files (MP4/MPEG4) — any count",
                type=accepted_types,
                accept_multiple_files=True,
                key=f"uploader_{i}",
                help=f"Hold Shift/Cmd/Ctrl to select multiple videos. Limit {MAX_UPLOAD_MB}MB per file.",
            )

            # live preview (lightweight)
            if uploaded:
                allowed = {".mp4", ".mpeg4"}
                vids = [u for u in uploaded if pathlib.Path(u.name).suffix.lower() in allowed]
                non_video = [u.name for u in uploaded if pathlib.Path(u.name).suffix.lower() not in allowed]

                if non_video:
                    st.warning("Non-video files will be ignored: " + ", ".join(non_video[:5]) + ("…" if len(non_video) > 5 else ""))

                if vids:
                    st.markdown("**Videos**")
                    for u in vids:
                        st.write("•", u.name)

            # NEW: Clear only the current selection in the uploader (does not touch saved uploads)
            if st.button("선택 파일 모두 지우기", key=f"clear_selected_{i}", help="현재 탭에서 방금 선택한 파일들을 모두 해제합니다."):
                st.session_state[f"clear_uploader_flag_{i}"] = True   # set flag
                st.rerun()
            
            st.markdown("**Add video by URL (server-side download)**")
            url_val = st.text_input("Paste a direct video link or Drive *file* link", key=f"urlinput_{i}", placeholder="https://…")
            if st.button("Add URL video", key=f"addurl_{i}"):
                try:
                    meta = fetch_url_to_tmp(url_val)  # keeps your existing single-file URL fetcher
                    lst = st.session_state.remote_videos.get(game, [])
                    lst.append(meta)
                    st.session_state.remote_videos[game] = lst
                    st.success(f"Added: {meta['name']}")
                except Exception as e:
                    st.exception(e)
                    try:
                        hint = _friendly_fb_error(e)
                        st.markdown(hint, unsafe_allow_html=True)
                    except Exception:
                        pass
                    ok_msg_placeholder.error("Meta upload failed due to permissions. See guidance above.")

            # --- Import videos from Google Drive folder (server-side) ---
            # --- Import videos from Google Drive folder (server-side) ---
            st.markdown("**Import all videos from a Google Drive folder (server-side)**")
            drv_input = st.text_input(
                "Drive folder URL or ID",
                key=f"drive_folder_{i}",
                placeholder="https://drive.google.com/drive/folders/<FOLDER_ID>",
            )
            workers = st.number_input("Parallel workers", 1, 8, value=3, key=f"drive_workers_{i}")

            if st.button("Import videos from folder", key=f"drive_import_{i}"):
                try:
                    overall = st.progress(0, text="0/0 • waiting…")
                    log_box = st.empty()
                    lines: List[str] = []
                    imported_accum: List[Dict] = []

                    def _on_progress(done: int, total: int, name: str, err: str | None):
                        pct = int((done / max(total, 1)) * 100)
                        label = f"{done}/{total}"
                        if name:
                            label += f" • {name}"
                        if err:
                            lines.append(f"❌ {name}  —  {err}")
                        else:
                            lines.append(f"✅ {name}")
                        overall.progress(pct, text=label)
                        # keep last ~200 lines visible
                        log_box.write("\n".join(lines[-200:]))

                    with st.status("Importing videos from Drive folder...", expanded=True) as status:
                        from drive_import import import_drive_folder_videos_parallel
                        imported = import_drive_folder_videos_parallel(
                            drv_input, max_workers=int(workers), on_progress=_on_progress
                        )
                        imported_accum.extend(imported)
                        lst = st.session_state.remote_videos.get(game, [])
                        lst.extend(imported_accum)
                        st.session_state.remote_videos[game] = lst
                        # finalize status
                        status.update(label=f"Drive import complete: {len(imported_accum)} file(s)", state="complete")
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
            if remote_list:
                st.caption("Server-downloaded videos:")
                for it in remote_list[:50]:
                    st.write("•", it["name"])
                if st.button("Clear URL/Drive videos", key=f"clearurl_{i}"):
                    st.session_state.remote_videos[game] = []
                    st.info("Cleared remote videos for this game.")
                    st.rerun()                                          # next run will clear before creating widget

            # --- ACTION BUTTONS ---
            st.markdown("### Actions")
            dry_run = st.checkbox("Dry run (no API writes)", value=True, key=f"dryrun_{i}")

            if st.button("🔍 Preflight: permissions & token", key=f"preflight_{i}"):
                try:
                    acct = "act_692755193188182"
                    page = st.secrets.get("page_id", "")
                    diag = preflight_permissions(acct, page)
                    st.success(f"Token belongs to: {diag['me']['name']} ({diag['me']['id']})")

                    c1, c2, c3 = st.columns([1,1,1])
                    with c1: st.metric("Sees target ad account", "Yes" if diag["target_account_seen"] else "No")
                    with c2: st.metric("Can create in target", "Yes" if diag["can_create_in_target"] else "No")
                    with c3: st.metric("Page access OK", "-" if diag["page_access_ok"] is None else ("Yes" if diag["page_access_ok"] else "No"))

                    st.caption("Permissions on target ad account")
                    st.code(", ".join(diag["target_permissions"]) or "(none)", language=None)

                    with st.expander("All ad accounts visible to this token", expanded=False):
                        for row in diag["adaccounts_overview"]:
                            st.write(f"- {row['id']} — {row['name']} (status={row['status']})")
                            st.write("  permissions:", ", ".join(row.get("permissions") or []))
                except Exception as e:
                    st.exception(e)
                    st.error("Preflight failed. Check token and network.")
            # Read-only connection check to confirm token, account, and campaign visibility
            if st.button("🔎 Check Facebook connection (read-only)", key=f"fbcheck_{i}"):
                try:
                    result = test_fb_setup()
                    st.success(f"Account: {result['account']['name']} | Currency: {result['account']['currency']}")
                    st.write("Creative test campaign present:", result["creative_test_campaign_found"])
                except Exception as e:
                    st.exception(e)
                    st.error("Facebook connection failed. See error above.")

            ok_msg_placeholder = st.empty()
            cont = st.button("Creative Test 업로드하기", key=f"continue_{i}")
            clr = st.button("업로드 파일 초기화", key=f"clear_{i}")

        # ----- RIGHT: SETTINGS PANEL -----
        with right:
            ensure_settings_state()
            st.markdown("### Settings")

            cur = st.session_state.settings.get(game, {})

        

            # 1) 광고 세트 이름: campaign_name + "_nth"
            suffix_number = st.number_input(
                "광고 세트 접미사 n (…_nth)",
                min_value=1,
                step=1,
                value=int(cur.get("suffix_number", 1)),
                help="Ad set will be named as <campaign_name>_<n>th",
                key=f"suffix_{i}",
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

            # 7) 예약 (기본: 토 00:00 → 월 12:00 KST)
            default_start_iso, default_end_iso = next_sat_0000_and_mon_1200_kst()
            start_iso = st.text_input(
                "시작 날짜/시간 (ISO, KST)",
                value=cur.get("start_iso", default_start_iso),
                help="예: 2025-11-15T00:00:00+09:00",
                key=f"start_{i}",
            )
            end_iso = st.text_input(
                "종료 날짜/시간 (ISO, KST)",
                value=cur.get("end_iso", default_end_iso),
                help="예: 2025-11-17T12:00:00+09:00",
                key=f"end_{i}",
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
                "app_store": app_store,
                "fb_app_id": fb_app_id.strip(),
                "store_url": store_url.strip(),
                "opt_goal_label": opt_goal_label,
                "budget_per_video_usd": int(budget_per_video_usd),
                "start_iso": start_iso.strip(),
                "end_iso": end_iso.strip(),
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
        # --- Handle button actions after UI is drawn ---
        if cont:
            # Combine browser uploads + server-downloaded URL videos
            remote_list = st.session_state.remote_videos.get(game, [])
            combined = (uploaded or []) + remote_list

            ok, msg = validate_count(combined)
            if not ok:
                ok_msg_placeholder.error(msg)
            else:
                try:
                    st.session_state.uploads[game] = uploaded  # keep native uploads separately if you like
                    settings = st.session_state.settings.get(game, {})
                    plan = upload_to_facebook(game, combined, settings, simulate=dry_run)
                    def _render_summary(plan: dict, settings: dict, created: bool):
                        """
                        Render a styled summary card of the planned/created upload.
                        Uses components_html (universal), falls back to st.markdown with unsafe HTML,
                        and finally to a native Streamlit layout if HTML fails.
                        """
                        # ---- values ----
                        if not isinstance(plan, dict):
                            st.error("No plan data to display (upload did not return a plan).")
                            return
                        if settings is None:
                            settings = {}
                        store_url = (settings.get("store_url") or "").strip()
                        fb_app_id = (settings.get("fb_app_id") or "").strip()
                        per_video = int(settings.get("budget_per_video_usd", 10))
                        os_choice = settings.get("os_choice", "Both")
                        min_android = settings.get("min_android_label", "None (any)")
                        min_ios = settings.get("min_ios_label", "None (any)")
                        campaign_name = plan.get("campaign_name") or settings.get("campaign_name") or "—"
                        app_store = plan.get("app_store") or settings.get("app_store") or "—"
                        opt_goal_label = plan.get("opt_goal_label") or settings.get("opt_goal_label") or "앱 설치수 극대화"

                        # Display-only fix: turn "..._1th" → "..._1st"
                        def _ordinalize(adset_name: str) -> str:
                            try:
                                suffix = adset_name.rsplit("_", 1)[-1]
                                n = int(suffix[:-2]) if suffix.endswith("th") else None
                                if not n:
                                    return adset_name
                                def ord_suffix(k):
                                    return "th" if 11 <= (k % 100) <= 13 else {1:"st", 2:"nd", 3:"rd"}.get(k % 10, "th")
                                nice = f"{n}{ord_suffix(n)}"
                                return adset_name[:-len(suffix)] + nice
                            except Exception:
                                return adset_name

                        adset_name_disp = _ordinalize(plan.get("adset_name", "—"))

                        # ---- build HTML once (BEFORE rendering) ----
                        html = f"""
                        <style>
                        .summary-card {{ border:1px solid #e5e7eb; border-radius:12px; padding:16px 18px; background:#fff; }}
                        .kv {{ display:grid; grid-template-columns: 200px 1fr; row-gap:8px; column-gap:14px; }}
                        .label {{ color:#6b7280; font-weight:600; }}
                        .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }}
                        .chip {{ display:inline-block; background:#f3f4f6; border:1px solid #e5e7eb; border-radius:999px; padding:4px 10px; margin-right:6px; margin-bottom:6px; font-size:0.875rem; }}
                        </style>
                        <div class="summary-card">
                        <div class="kv">
                            <div class="label">Campaign</div><div class="mono">{campaign_name}</div>
                            <div class="label">Campaign ID</div><div class="mono">{plan.get('campaign_id','—')}</div>
                            <div class="label">Ad Set Name</div><div class="mono">{adset_name_disp}</div>

                            <div class="label">App Store</div><div>{app_store}</div>
                            <div class="label">Optimization Goal</div><div>{opt_goal_label}</div>

                            <div class="label">Country/Age</div><div>{plan['country']} / {plan['age_min']}+</div>
                            <div class="label">Budget (USD/day)</div>
                            <div class="mono">${plan['budget_usd_per_day']} <span style="color:#6b7280;">(= ${per_video} × {plan['n_videos']} videos)</span></div>
                            <div class="label">Schedule (KST)</div><div class="mono">{plan['start_iso']} → {plan['end_iso']}</div>

                            <div class="label">OS Targeting</div>
                            <div>
                            <span class="chip">{os_choice}</span>
                            <span class="chip">Android ≥ {min_android}</span>
                            <span class="chip">iOS ≥ {min_ios}</span>
                            </div>

                            <div class="label"># of videos</div><div class="mono">{plan['n_videos']}</div>
                            {f"<div class='label'>Store URL</div><div class='mono'>{store_url}</div>" if store_url else ""}
                            {f"<div class='label'>Facebook App ID</div><div class='mono'>{fb_app_id}</div>" if fb_app_id else ""}
                        </div>
                        </div>
                        """

                        # ---- heading + creation note ----
                        st.markdown("#### 📋 Creative Test Summary")
                        if created and plan.get("adset_id"):
                            st.success(f"Created Ad Set ID: `{plan['adset_id']}`")

                        # ---- render: components_html → markdown(unsafe) → native fallback ----
                        rendered = False
                        try:
                            components_html(html, height=360, scrolling=False)
                            rendered = True
                        except Exception:
                            pass

                        if not rendered:
                            try:
                                st.markdown(html, unsafe_allow_html=True)
                                rendered = True
                            except Exception:
                                rendered = False

                        if not rendered:
                            # Native fallback (no HTML/CSS)
                            def row(lbl, val):
                                c1, c2 = st.columns([1, 3])
                                with c1: st.caption(lbl)
                                with c2: st.code(str(val), language=None)

                            row("Campaign", campaign_name)
                            row("Campaign ID", plan.get("campaign_id","—"))
                            row("Ad Set Name", adset_name_disp)
                            row("App Store", app_store)
                            row("Optimization Goal", opt_goal_label)
                            row("Country/Age", f"{plan['country']} / {plan['age_min']}+")
                            row("Budget (USD/day)", f"${plan['budget_usd_per_day']} (= ${per_video} × {plan['n_videos']} videos)")
                            row("Schedule (KST)", f"{plan['start_iso']} → {plan['end_iso']}")
                            st.caption("OS Targeting")
                            st.write(f"• {os_choice}")
                            st.write(f"• Android ≥ {min_android}")
                            st.write(f"• iOS ≥ {min_ios}")
                            row("# of videos", plan['n_videos'])
                            if store_url: row("Store URL", store_url)
                            if fb_app_id: row("Facebook App ID", fb_app_id)

                        # Ad names list
                        if plan.get("ad_names"):
                            with st.expander("Ad names to be created", expanded=False):
                                for nm in plan["ad_names"]:
                                    st.write("•", nm)

                    if dry_run:
                        ok_msg_placeholder.info("Dry run only — nothing was created.")
                        _render_summary(plan, settings, created=False)
                    else:
                        # Success only if we got a valid dict and adset_id
                        if isinstance(plan, dict) and plan.get("adset_id"):
                            ok_msg_placeholder.success(msg + " Uploaded to Meta (ads created as PAUSED).")
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
            # Clear saved data for this game
            st.session_state.uploads.pop(game, None)
            st.session_state.remote_videos.pop(game, None)  # also clear URL/Drive videos
            st.session_state.settings.pop(game, None)

            # Mark the uploader to be cleared on the NEXT run,
            # BEFORE the widget is created again.
            st.session_state[f"clear_uploader_flag_{i}"] = True

            ok_msg_placeholder.info("Cleared saved uploads, URL videos, and settings for this game.")
            st.rerun()

st.divider()

# Summary table
st.subheader("Summary")
if st.session_state.uploads:
    data = {"Game": [], "Files Saved": []}
    for g, files in st.session_state.uploads.items():
        data["Game"].append(g)
        data["Files Saved"].append(len(files))
    st.dataframe(data, hide_index=True)
else:
    st.info("No uploads saved yet. Go to a tab and click **Continue** after uploading.")
