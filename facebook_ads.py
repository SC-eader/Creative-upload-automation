"""Facebook/Meta helpers for Creative 자동 업로드 Streamlit app."""

from __future__ import annotations

from typing import Dict, List, Any
from datetime import datetime, timedelta, timezone
import logging
import pathlib
import tempfile
import os

import requests
import streamlit as st

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------
# Date / timezone helpers
# --------------------------------------------------------------------
ASIA_SEOUL = timezone(timedelta(hours=9))

def next_sat_0900_kst(today: datetime | None = None) -> str:
    """
    Compute start_iso in KST:
      - start: next Saturday 09:00
    Returned string is ISO8601 with +09:00 offset.
    """
    now = (today or datetime.now(ASIA_SEOUL)).astimezone(ASIA_SEOUL)
    base = now.replace(hour=0, minute=0, second=0, microsecond=0)
    # Monday=0 ... Saturday=5, Sunday=6
    days_until_sat = (5 - base.weekday()) % 7 or 7
    start_dt = (base + timedelta(days=days_until_sat)).replace(hour=9, minute=0)
    return start_dt.isoformat()

# --------------------------------------------------------------------
# Settings helpers (store URL, budget, targeting)
# --------------------------------------------------------------------
def sanitize_store_url(raw: str) -> str:
    """
    Normalize store URLs for Meta:
      - Google Play: keep ?id=<package> only
      - App Store: drop query/fragment
      - Other hosts: return as-is
    """
    from urllib.parse import urlsplit, urlunsplit, parse_qs, urlencode

    if not raw:
        return raw

    parts = urlsplit(raw)
    host = parts.netloc.lower()

    # Google Play: MUST preserve 'id' param only
    if "play.google.com" in host:
        qs = parse_qs(parts.query)
        pkg = (qs.get("id") or [None])[0]
        if not pkg:
            raise ValueError(
                "Google Play URL must include ?id=<package>. "
                "Example: https://play.google.com/store/apps/details?id=io.supercent.weaponrpg"
            )
        new_query = urlencode({"id": pkg})
        return urlunsplit(
            (parts.scheme, parts.netloc, parts.path or "/store/apps/details", new_query, "")
        )

    # Apple App Store: keep path only
    if "apps.apple.com" in host:
        return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))

    # Other hosts: unchanged
    return raw

def compute_budget_from_settings(files: list, settings: dict, fallback_per_video: int = 10) -> int:
    """
    Budget per day = (#eligible videos) × per-video budget.
    Counts only .mp4/.mpeg4.
    """
    allowed = {".mp4", ".mpeg4"}

    def _name(u):
        return getattr(u, "name", None) or (u.get("name") if isinstance(u, dict) else "")

    n_videos = sum(
        1 for u in (files or []) if pathlib.Path(_name(u)).suffix.lower() in allowed
    )
    per_video = int(settings.get("budget_per_video_usd", fallback_per_video))
    return max(1, n_videos * per_video) if n_videos else per_video

def dollars_to_minor(usd: float) -> int:
    """Convert USD → Meta 'minor' units (1 USD → 100)."""
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
    Uses Advantage+ placements (no publisher_platforms/device_platforms),
    and optional OS-family filtering via user_os.
    """
    os_choice = settings.get("os_choice", "Both")
    min_android = settings.get("min_android_os_token")
    min_ios = settings.get("min_ios_os_token")

    targeting = {
        "geo_locations": {"countries": [country]},
        "age_min": max(13, int(age_min)),
    }

    user_os: list[str] = []

    if os_choice == "Android only":
        token = min_android or "Android_ver_6.0_and_above"
        user_os.append(token)
    elif os_choice == "iOS only":
        token = min_ios or "iOS_ver_11.0_and_above"
        user_os.append(token)
    else:  # Both
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

# --------------------------------------------------------------------
# Session-state helpers for FB settings
# --------------------------------------------------------------------
def _ensure_settings_state() -> None:
    if "settings" not in st.session_state:
        st.session_state.settings = {}

def get_fb_settings(game: str) -> dict:
    """Return per-game FB settings dict (creating container if needed)."""
    _ensure_settings_state()
    return st.session_state.settings.get(game, {})

# --------------------------------------------------------------------
# Default per-game App IDs + Store URLs
# --------------------------------------------------------------------
GAME_DEFAULTS: Dict[str, Dict[str, str]] = {
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

def init_fb_game_defaults() -> None:
    """
    Apply FB app_id/store_url defaults per game without overwriting
    what the user has already saved in st.session_state.settings.
    """
    _ensure_settings_state()
    for game, defaults in GAME_DEFAULTS.items():
        cur = st.session_state.settings.get(game, {}) or {}
        if not cur.get("fb_app_id") and defaults.get("fb_app_id"):
            cur["fb_app_id"] = defaults["fb_app_id"]
        if not cur.get("store_url") and defaults.get("store_url"):
            cur["store_url"] = defaults["store_url"]
        st.session_state.settings[game] = cur

# --------------------------------------------------------------------
# Meta SDK and account helpers
# --------------------------------------------------------------------
try:
    from facebook_business.api import FacebookAdsApi
    from facebook_business.adobjects.adaccount import AdAccount
    from facebook_business.adobjects.adset import AdSet
    from facebook_business.adobjects.adcreative import AdCreative
    from facebook_business.adobjects.ad import Ad
    from facebook_business.exceptions import FacebookRequestError
    FB_AVAILABLE = True
    FB_IMPORT_ERROR = ""
except Exception as _e:
    FB_AVAILABLE = False
    FB_IMPORT_ERROR = f"{type(_e).__name__}: {_e}"

def _require_fb() -> None:
    """Raise a clear error if the Facebook SDK is missing."""
    if not FB_AVAILABLE:
        raise RuntimeError(
            "facebook-business SDK not available. Install it with:\n"
            "  pip install facebook-business\n"
            f"Import error: {FB_IMPORT_ERROR}"
        )

def init_fb_from_secrets(ad_account_id: str | None = None) -> "AdAccount":
    """
    Initialize Meta SDK using only access_token from st.secrets,
    and return an AdAccount (default: XP HERO account if none given).
    """
    _require_fb()
    token = st.secrets.get("access_token", "").strip()
    if not token:
        raise RuntimeError("Missing access_token in st.secrets. Put it in .streamlit/secrets.toml")

    FacebookAdsApi.init(access_token=token)

    default_act_id = "act_692755193188182"  # XP HERO default
    act_id = ad_account_id or default_act_id
    return AdAccount(act_id)

def validate_page_binding(account: "AdAccount", page_id: str) -> dict:
    """
    Ensure page_id is numeric/readable and fetch IG actor (if present).
    Returns {'id','name','instagram_business_account_id'}.
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

# --------------------------------------------------------------------
# File helpers for uploads
# --------------------------------------------------------------------
VERBOSE_UPLOAD_LOG = False

def _fname_any(u) -> str:
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

def _save_uploadedfile_tmp(u) -> str:
    """
    Persist a video source to disk and return its path.
    Supports UploadedFile and {'name','path'} dicts.
    """
    if isinstance(u, dict) and "path" in u and "name" in u:
        return u["path"]
    if hasattr(u, "getbuffer"):
        suffix = pathlib.Path(u.name).suffix.lower() or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(u.getbuffer())
            return tmp.name
    raise ValueError("Unsupported video object type for saving.")

# --------------------------------------------------------------------
# Resumable upload + ad creation
# --------------------------------------------------------------------
def upload_videos_create_ads(
    account: "AdAccount",
    *,
    page_id: str,
    adset_id: str,
    uploaded_files: list,
    ad_name_prefix: str | None = None,
    max_workers: int = 6,
    store_url: str | None = None,
    try_instagram: bool = True,
):
    """
    Upload videos (resumable), wait for processing, then create creatives + ads in parallel.
    Returns a list of {'name','ad_id'} and shows errors in Streamlit UI.
    """
    from facebook_business.adobjects.advideo import AdVideo
    from facebook_business.adobjects.page import Page
    from facebook_business.adobjects.adcreative import AdCreative
    from facebook_business.adobjects.ad import Ad
    from facebook_business.exceptions import FacebookRequestError
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time

    allowed = {".mp4", ".mpeg4"}

    def _is_video(u):
        n = _fname_any(u) or "video.mp4"
        return pathlib.Path(n).suffix.lower() in allowed

    videos = _dedupe_by_name([u for u in (uploaded_files or []) if _is_video(u)])

    def _persist_to_tmp(u):
        return {"name": _fname_any(u) or "video.mp4", "path": _save_uploadedfile_tmp(u)}

    def simple_video_upload(path: str) -> str:
        v = account.create_ad_video(params={"file": path, "content_category": "VIDEO_GAMING"})
        return v["id"]

    def upload_video_resumable(path: str) -> str:
        """
        Chunked upload to /{act_id}/advideos using the official 3-phase protocol.
        Retries transient errors and verifies total bytes sent before finishing.
        """
        token = (st.secrets.get("access_token") or "").strip()
        if not token:
            raise RuntimeError("Missing access_token in st.secrets")

        act = account.get_id()
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
                        if code in (390,) and i < max_retries:
                            last = RuntimeError(j["error"].get("message"))
                            continue
                        raise RuntimeError(j["error"].get("message", str(j["error"])))
                    return j
                except Exception as e:
                    last = e
            raise last or RuntimeError("advideos POST failed")

        start_resp = _post(
            {"upload_phase": "start", "file_size": str(file_size), "content_category": "VIDEO_GAMING"}
        )
        upload_session_id = start_resp["upload_session_id"]
        video_id = start_resp["video_id"]
        start_offset = int(start_resp.get("start_offset", 0))
        end_offset = int(start_resp.get("end_offset", 0))

        sent_bytes = 0

        with open(path, "rb") as f:
            while True:
                if start_offset == end_offset == file_size:
                    if VERBOSE_UPLOAD_LOG:
                        st.write(f"[Upload] ✅ All bytes acknowledged ({sent_bytes}/{file_size}).")
                    break

                if end_offset <= start_offset:
                    if VERBOSE_UPLOAD_LOG:
                        st.write(f"[Upload] ↻ Asking for next window at {start_offset}")
                    tr = _post(
                        {
                            "upload_phase": "transfer",
                            "upload_session_id": upload_session_id,
                            "start_offset": str(start_offset),
                        }
                    )
                    start_offset = int(tr.get("start_offset", start_offset))
                    end_offset = int(tr.get("end_offset", end_offset or file_size))
                    continue

                to_read = end_offset - start_offset
                f.seek(start_offset)
                chunk = f.read(to_read)
                if not chunk or len(chunk) != to_read:
                    raise RuntimeError(f"Read {len(chunk) if chunk else 0} bytes; expected {to_read}.")

                files = {"video_file_chunk": ("chunk.bin", chunk, "application/octet-stream")}
                tr = _post(
                    {
                        "upload_phase": "transfer",
                        "upload_session_id": upload_session_id,
                        "start_offset": str(start_offset),
                    },
                    files=files,
                )

                sent_bytes += to_read
                new_start = int(tr.get("start_offset", start_offset + to_read))
                new_end = int(tr.get("end_offset", end_offset))

                if VERBOSE_UPLOAD_LOG:
                    st.write(
                        f"[Upload] Sent [{start_offset},{end_offset}) → "
                        f"ack: start={new_start}, end={new_end}, sent={sent_bytes}/{file_size}"
                    )

                start_offset, end_offset = new_start, new_end
                if start_offset > file_size:
                    start_offset = file_size
                if end_offset > file_size:
                    end_offset = file_size

        if sent_bytes != file_size:
            raise RuntimeError(f"Uploaded bytes ({sent_bytes}) != file size ({file_size}).")

        try:
            _post({"upload_phase": "finish", "upload_session_id": upload_session_id})
            return video_id
        except Exception:
            st.info(
                f"Resumable finish failed for {os.path.basename(path)} — trying fallback upload once."
            )
            v = account.create_ad_video(params={"file": path, "content_category": "VIDEO_GAMING"})
            return v["id"]

    def wait_all_videos_ready(video_ids: list[str], *, timeout_s: int = 120, sleep_s: int = 3) -> dict[str, bool]:
        """Poll advideo thumbnails for all video_ids to avoid blank previews."""
        from facebook_business.adobjects.advideo import AdVideo
        import time

        ready = {vid: False for vid in video_ids}
        deadline = time.time() + timeout_s

        while time.time() < deadline:
            all_done = True
            for vid in video_ids:
                if ready[vid]:
                    continue
                try:
                    info = AdVideo(vid).api_get(fields=["thumbnails", "picture"])
                    has_pic = bool(info.get("picture"))
                    has_thumbs = bool(info.get("thumbnails"))
                    if has_pic or has_thumbs:
                        ready[vid] = True
                    else:
                        all_done = False
                except Exception:
                    all_done = False
            if all_done:
                break
            time.sleep(sleep_s)
        return ready

    def resolve_instagram_actor_id(page_id: str) -> str | None:
        try:
            p = Page(page_id).api_get(fields=["instagram_business_account"])
            iba = p.get("instagram_business_account") or {}
            return iba.get("id")
        except Exception:
            return None

    # Stage 1: persist to temp (parallel)
    persisted, persist_errors = [], []
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_persist_to_tmp, u): _fname_any(u) for u in videos}
        for fut, nm in futs.items():
            try:
                persisted.append(fut.result())
            except Exception as e:
                persist_errors.append(f"{nm}: {e}")

    if persist_errors:
        st.warning("Some files failed to prepare:\n- " + "\n- ".join(persist_errors))

    ig_actor_id = None
    try:
        ig_actor_id = st.session_state.get("ig_actor_id_from_page") or None
    except Exception:
        pass
    if try_instagram and not ig_actor_id:
        ig_actor_id = resolve_instagram_actor_id(page_id)

    uploads, api_errors = [], []
    total = len(persisted)
    progress = st.progress(0, text=f"Uploading 0/{total} videos…") if total else None

    def _upload_one(item):
        name, path = item["name"], item["path"]
        vid = upload_video_resumable(path)
        return {"name": name, "path": path, "video_id": vid}

    done = 0
    if total:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
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
        progress.empty()

    # Wait for all videos to have thumbnails
    video_ids = [u["video_id"] for u in uploads]
    _ = wait_all_videos_ready(video_ids, timeout_s=300, sleep_s=5)

    # Create creatives + ads in parallel
    results = []
    total_c = len(uploads)
    progress_c = st.progress(0, text=f"Creating 0/{total_c} ads…") if total_c else None
    done_c = 0

    def _process_one_video(up):
        from facebook_business.adobjects.advideo import AdVideo
        from facebook_business.adobjects.adcreative import AdCreative
        from facebook_business.adobjects.ad import Ad
        from facebook_business.exceptions import FacebookRequestError
        import time

        name, video_id = up["name"], up["video_id"]
        try:
            video_info = AdVideo(video_id).api_get(fields=["picture"])
            thumbnail_url = video_info.get("picture")
            if not thumbnail_url:
                raise RuntimeError("Video processed but no 'picture' (thumbnail) URL was returned.")

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

            try:
                ad_id = _create_once(True)
            except FacebookRequestError as e:
                msg = (e.api_error_message() or "").lower()
                if "instagram" in msg or "not ready" in msg or "processing" in msg:
                    time.sleep(5)
                    ad_id = _create_once(False)
                else:
                    raise

            return {"success": True, "result": {"name": name, "ad_id": ad_id}}
        except Exception as e:
            return {"success": False, "error": f"{name}: creative/ad failed: {e}"}

    if total_c:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
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

    if api_errors:
        st.error(
            f"{len(api_errors)} video(s) failed during creation:\n"
            + "\n".join(f"- {e}" for e in api_errors[:20])
            + ("\n..." if len(api_errors) > 20 else "")
        )

    return results

# --------------------------------------------------------------------
# Ad set planning + creation
# --------------------------------------------------------------------
def _plan_upload(
    account: "AdAccount",
    *,
    campaign_id: str,
    adset_prefix: str,
    page_id: str,
    uploaded_files: list,
    settings: dict,
) -> dict:
    """
    Compute planned ad set name/budget/schedule/ad names from settings
    and available videos (local + remote_videos).
    """
    start_iso = settings.get("start_iso") or next_sat_0900_kst()
    end_iso = settings.get("end_iso")

    n = int(settings.get("suffix_number") or 1)
    suffix_str = f"{n}th"

    launch_date_suffix = ""
    if settings.get("add_launch_date"):
        try:
            dt = datetime.fromisoformat(start_iso)
            launch_date_suffix = "_" + dt.strftime("%y%m%d")
        except Exception:
            launch_date_suffix = ""

    adset_name = f"{adset_prefix}_{suffix_str}{launch_date_suffix}"

    allowed = {".mp4", ".mpeg4"}
    remote = st.session_state.remote_videos.get(settings.get("game_key", ""), []) or []

    def _name(u):
        return getattr(u, "name", None) or (u.get("name") if isinstance(u, dict) else "")

    def _is_video(u):
        return pathlib.Path(_name(u)).suffix.lower() in allowed

    vids_local = [u for u in (uploaded_files or []) if _is_video(u)]
    vids_all = _dedupe_by_name(vids_local + [rv for rv in remote if _is_video(rv)])

    budget_usd_per_day = compute_budget_from_settings(vids_all, settings)

    ad_name_prefix = (
        settings.get("ad_name_prefix") if settings.get("ad_name_mode") == "Prefix + filename" else None
    )
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
        "campaign_name": settings.get("campaign_name"),
        "app_store": settings.get("app_store"),
        "opt_goal_label": settings.get("opt_goal_label"),
    }

def create_creativetest_adset(
    account: "AdAccount",
    *,
    campaign_id: str,
    adset_name: str,
    targeting: dict,
    daily_budget_usd: int,
    start_iso: str,
    optimization_goal: str,
    promoted_object: dict | None = None,
    end_iso: str | None = None,
) -> str:
    """
    Create an ACTIVE ad set for a creative test and return its ID.
    """
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

    if end_iso:
        params["end_time"] = end_iso
    if promoted_object:
        params["promoted_object"] = promoted_object

    adset = account.create_ad_set(fields=[], params=params)
    return adset["id"]

# --------------------------------------------------------------------
# Per-game mapping + main entry
# --------------------------------------------------------------------
FB_GAME_MAPPING: Dict[str, Dict[str, Any]] = {
    "XP HERO": {
        "account_id": "act_692755193188182",
        "campaign_id": "120218934861590118",
        "campaign_name": "weaponrpg_aos_facebook_us_creativetest",
        "adset_prefix": "weaponrpg_aos_facebook_us_creativetest",
        "page_id_key": "page_id_xp",
    },
    "Dino Universe": {
        "account_id": "act_1400645283898971",
        "campaign_id": "120203672340130431",
        "campaign_name": "ageofdinosaurs_aos_facebook_us_test_6th+",
        "adset_prefix": "ageofdinosaurs_aos_facebook_us_test",
        "page_id_key": "page_id_dino",
    },
    "Snake Clash": {
        "account_id": "act_837301614677763",
        "campaign_id": "120201313657080615",
        "campaign_name": "linkedcubic_aos_facebook_us_test_14th above",
        "adset_prefix": "linkedcubic_aos_facebook_us_test",
        "page_id_key": "page_id_snake",
    },
    "Pizza Ready": {
        "account_id": "act_939943337267153",
        "campaign_id": "120200161907250465",
        "campaign_name": "pizzaidle_aos_facebook_us_test_12th+",
        "adset_prefix": "pizzaidle_aos_facebook_us_test",
        "page_id_key": "page_id_pizza",
    },
    "Cafe Life": {
        "account_id": "act_1425841598550220",
        "campaign_id": "120231530818850361",
        "campaign_name": "cafelife_aos_facebook_us_creativetest",
        "adset_prefix": "cafelife_aos_facebook_us_creativetest",
        "page_id_key": "page_id_cafe",
    },
    "Suzy's Restaurant": {
        "account_id": "act_953632226485498",
        "campaign_id": "120217220153800643",
        "campaign_name": "suzyrest_aos_facebook_us_creativetest",
        "adset_prefix": "suzyrest_aos_facebook_us_creativetest",
        "page_id_key": "page_id_suzy",
    },
    "Office Life": {
        "account_id": "act_733192439468531",
        "campaign_id": "120228464454680636",
        "campaign_name": "corporatetycoon_aos_facebook_us_creativetest",
        "adset_prefix": "corporatetycoon_aos_facebook_us_creativetest",
        "page_id_key": "page_id_office",
    },
    "Lumber Chopper": {
        "account_id": "act_1372896617079122",
        "campaign_id": "120224569359980144",
        "campaign_name": "lumberchopper_aos_facebook_us_creativetest",
        "adset_prefix": "lumberchopper_aos_facebook_us_creativetest",
        "page_id_key": "page_id_lumber",
    },
    "Burger Please": {
        "account_id": "act_3546175519039834",
        "campaign_id": "120200361364790724",
        "campaign_name": "burgeridle_aos_facebook_us_test_30th+",
        "adset_prefix": "burgeridle_aos_facebook_us_test",
        "page_id_key": "page_id_burger",
    },
    "Prison Life": {
        "account_id": "act_510600977962388",
        "campaign_id": "120212520882120614",
        "campaign_name": "prison_aos_facebook_us_install_test",
        "adset_prefix": "prison_aos_facebook_us_install_test",
        "page_id_key": "page_id_prison",
    },
}

def upload_to_facebook(
    game_name: str,
    uploaded_files: list,
    settings: dict,
    *,
    simulate: bool = False,
) -> dict:
    """
    Main entry: create ad set + ads for a game using current settings.
    If simulate=True, just return the plan (no writes).
    """
    if game_name not in FB_GAME_MAPPING:
        raise ValueError(f"No FB mapping configured for game: {game_name}")

    cfg = FB_GAME_MAPPING[game_name]
    account = init_fb_from_secrets(cfg["account_id"])

    page_id_key = cfg.get("page_id_key")
    if not page_id_key or page_id_key not in st.secrets:
        raise RuntimeError(f"Missing {page_id_key!r} in st.secrets for game {game_name}")
    page_id = st.secrets[page_id_key]

    # Validate page and capture IG actor
    page_check = validate_page_binding(account, page_id)
    ig_actor_id_from_page = page_check.get("instagram_business_account_id")

    # Extra safety: ensure page_id != ad account id
    try:
        acct_num = account.get_id().replace("act_", "")
        pid = str(page_id)
        if pid in (acct_num, f"act_{acct_num}"):
            raise RuntimeError(
                "Configured PAGE_ID equals the Ad Account ID. "
                "Set st.secrets[page_id_*] to your Facebook Page ID (NOT 'act_...')."
            )
        from facebook_business.adobjects.page import Page
        _probe = Page(pid).api_get(fields=["id", "name"])
        if not _probe or not _probe.get("id"):
            raise RuntimeError("Provided PAGE_ID is not readable with this token.")
    except Exception as _pg_err:
        raise RuntimeError(
            f"Page validation failed for PAGE_ID={page_id}. "
            "Use a real Facebook Page ID and ensure asset access from this ad account/token."
        ) from _pg_err

    # Build plan (no writes yet)
    settings = dict(settings or {})
    settings["campaign_name"] = cfg.get("campaign_name")
    plan = _plan_upload(
        account=account,
        campaign_id=cfg["campaign_id"],
        adset_prefix=cfg["adset_prefix"],
        page_id=str(page_id),
        uploaded_files=uploaded_files,
        settings=settings,
    )
    if simulate:
        return plan

    # Targeting
    targeting = build_targeting_from_settings(
        country=plan["country"],
        age_min=plan["age_min"],
        settings=settings,
    )

    # Optimization goal + promoted_object
    opt_goal_label = settings.get("opt_goal_label") or "앱 설치수 극대화"
    opt_goal_api = OPT_GOAL_LABEL_TO_API.get(opt_goal_label, "APP_INSTALLS")

    store_label = settings.get("app_store")
    store_url = (settings.get("store_url") or "").strip()
    fb_app_id = (settings.get("fb_app_id") or "").strip()

    if store_url:
        store_url = sanitize_store_url(store_url)

    promoted_object = None
    if opt_goal_api in ("APP_INSTALLS", "APP_EVENTS", "VALUE"):
        if not store_url:
            raise RuntimeError(
                "App objective selected. Please enter a valid store URL in Settings "
                "(Google Play or App Store)."
            )
        promoted_object = {
            "object_store_url": store_url,
            **({"application_id": fb_app_id} if fb_app_id else {}),
        }

    adset_id = create_creativetest_adset(
        account=account,
        campaign_id=cfg["campaign_id"],
        adset_name=plan["adset_name"],
        targeting=targeting,
        daily_budget_usd=plan["budget_usd_per_day"],
        start_iso=plan["start_iso"],
        optimization_goal=opt_goal_api,
        promoted_object=promoted_object,
        end_iso=plan.get("end_iso"),
    )

    if not adset_id:
        raise RuntimeError(
            "Ad set was not created (no ID returned). Check the error above and fix settings/permissions."
        )

    ad_name_prefix = (
        settings.get("ad_name_prefix") if settings.get("ad_name_mode") == "Prefix + filename" else None
    )

    try:
        st.session_state["ig_actor_id_from_page"] = ig_actor_id_from_page
    except Exception:
        pass

    upload_videos_create_ads(
        account=account,
        page_id=str(page_id),
        adset_id=adset_id,
        uploaded_files=uploaded_files,
        ad_name_prefix=ad_name_prefix,
        store_url=store_url,
        try_instagram=True,
    )

    plan["adset_id"] = adset_id
    return plan

# --------------------------------------------------------------------
# Settings panel UI (right column)
# --------------------------------------------------------------------
def render_facebook_settings_panel(container, game: str, idx: int) -> None:
    """
    Render the Facebook settings panel for a single game and save
    values into st.session_state.settings[game].
    """
    _ensure_settings_state()
    cur = st.session_state.settings.get(game, {})

    with container:
        st.markdown(f"#### {game} Facebook Settings")

        suffix_number = st.number_input(
            "광고 세트 접미사 n(…_nth)",
            min_value=1,
            step=1,
            value=int(cur.get("suffix_number", 1)),
            help="Ad set will be named as <campaign_name>_<n>th or <campaign_name>_<n>th_YYMMDD",
            key=f"suffix_{idx}",
        )

        app_store = st.selectbox(
            "모바일 앱 스토어",
            ["Google Play 스토어", "Apple App Store"],
            index=0
            if cur.get("app_store", "Google Play 스토어") == "Google Play 스토어"
            else 1,
            key=f"appstore_{idx}",
        )

        fb_app_id = st.text_input(
            "Facebook App ID",
            value=cur.get("fb_app_id", ""),
            key=f"fbappid_{idx}",
            help="설치 추적을 연결하려면 FB App ID를 입력하세요(선택).",
        )
        store_url = st.text_input(
            "구글 스토어 URL",
            value=cur.get("store_url", ""),
            key=f"storeurl_{idx}",
            help="예) https://play.google.com/store/apps/details?id=... "
                 "(쿼리스트링/트래킹 파라미터 제거 권장)",
        )

        opt_goal_label = st.selectbox(
            "성과 목표",
            list(OPT_GOAL_LABEL_TO_API.keys()),
            index=list(OPT_GOAL_LABEL_TO_API.keys()).index(
                cur.get("opt_goal_label", "앱 설치수 극대화")
            ),
            key=f"optgoal_{idx}",
        )

        st.caption(
            "기여 설정: 클릭 1일(기본), 참여한 조회/조회 없음 — "
            "Facebook에서 고정/제한될 수 있습니다."
        )

        budget_per_video_usd = st.number_input(
            "영상 1개당 일일 예산 (USD)",
            min_value=1,
            value=int(cur.get("budget_per_video_usd", 10)),
            key=f"budget_per_video_{idx}",
            help="총 일일 예산 = (업로드/선택된 영상 수) × 이 값",
        )

        default_start_iso = next_sat_0900_kst()
        start_iso = st.text_input(
            "시작 날짜/시간 (ISO, KST)",
            value=cur.get("start_iso", default_start_iso),
            help="예: 2025-11-15T00:00:00+09:00 "
                 "(종료일은 자동으로 꺼지지 않도록 설정하지 않습니다)",
            key=f"start_{idx}",
        )

        launch_date_example = ""
        try:
            dt_preview = datetime.fromisoformat(start_iso.strip())
            launch_date_example = dt_preview.strftime("%y%m%d")
        except Exception:
            launch_date_example = ""

        add_launch_date = st.checkbox(
            "Launch 날짜 추가",
            value=bool(cur.get("add_launch_date", False)),
            key=f"add_launch_date_{idx}",
            help=(
                f"시작 날짜/시간의 날짜(YYMMDD)를 광고 세트 이름 끝에 추가합니다. "
                f"예: …_{int(suffix_number)}th_{launch_date_example or 'YYMMDD'}"
            ),
        )

        country = st.text_input(
            "국가",
            value=cur.get("country", "US"),
            key=f"country_{idx}",
        )

        age_min = st.number_input(
            "최소 연령",
            min_value=13,
            value=int(cur.get("age_min", 18)),
            key=f"age_{idx}",
        )

        os_choice = st.selectbox(
            "Target OS",
            ["Both", "Android only", "iOS only"],
            index={"Both": 0, "Android only": 1, "iOS only": 2}[
                cur.get("os_choice", "Android only")
            ],
            key=f"os_choice_{idx}",
        )

        if os_choice in ("Both", "Android only"):
            min_android_label = st.selectbox(
                "Min Android version",
                list(ANDROID_OS_CHOICES.keys()),
                index=list(ANDROID_OS_CHOICES.keys()).index(
                    cur.get("min_android_label", "6.0+")
                ),
                key=f"min_android_{idx}",
            )
        else:
            min_android_label = "None (any)"

        if os_choice in ("Both", "iOS only"):
            min_ios_label = st.selectbox(
                "Min iOS version",
                list(IOS_OS_CHOICES.keys()),
                index=list(IOS_OS_CHOICES.keys()).index(
                    cur.get("min_ios_label", "None (any)")
                ),
                key=f"min_ios_{idx}",
            )
        else:
            min_ios_label = "None (any)"

        min_android_os_token = (
            ANDROID_OS_CHOICES[min_android_label]
            if os_choice in ("Both", "Android only")
            else None
        )
        min_ios_os_token = (
            IOS_OS_CHOICES[min_ios_label]
            if os_choice in ("Both", "iOS only")
            else None
        )

        ad_name_mode = st.selectbox(
            "Ad name",
            ["Use video filename", "Prefix + filename"],
            index=1 if cur.get("ad_name_mode") == "Prefix + filename" else 0,
            key=f"adname_mode_{idx}",
        )
        ad_name_prefix = ""
        if ad_name_mode == "Prefix + filename":
            ad_name_prefix = st.text_input(
                "Ad name prefix",
                value=cur.get("ad_name_prefix", ""),
                key=f"adname_prefix_{idx}",
            )

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