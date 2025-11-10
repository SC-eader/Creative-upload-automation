"""
Streamlit UI: 10 games, each with an upload area that expects 10–12 creative files.

Run:
  streamlit run streamlit_facebook_creative_uploader.py

Notes:
- Supported files: JPG/PNG/MP4 (tweak below).
f"- Each tab accepts **video files only**: {', '.join(t.upper() for t in accepted_types)} (limit {MAX_UPLOAD_MB}MB per file)."- "Creative Test 업로드하기" creates a (paused) ad set + ads via Meta Marketing API using per-title settings.
"""

from __future__ import annotations

import os
from typing import Dict, List
from datetime import datetime, timedelta, timezone
import tempfile
import logging
import requests
from types import SimpleNamespace

import streamlit as st

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
    bad = [u.name for u in files if pathlib.Path(u.name).suffix.lower() not in allowed]
    if bad:
        return False, f"Only video files are allowed (.mp4/.mpeg4). Remove non-video files: {', '.join(bad[:5])}{'…' if len(bad)>5 else ''}"
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

# ----- Settings helpers -------------------------------------------------------

ASIA_SEOUL = timezone(timedelta(hours=9))  # KST (+09:00)

def next_sat_0000_and_mon_1000_kst(today: datetime | None = None) -> tuple[str, str]:
    """Return default (start_iso, end_iso) = next Saturday 00:00 and next Monday 10:00 in KST."""
    now = (today or datetime.now(ASIA_SEOUL)).astimezone(ASIA_SEOUL)
    base = now.replace(hour=0, minute=0, second=0, microsecond=0)
    # Monday=0 ... Saturday=5, Sunday=6
    days_until_sat = (5 - base.weekday()) % 7 or 7
    start_dt = (base + timedelta(days=days_until_sat)).replace(hour=0, minute=0)
    end_dt = (start_dt + timedelta(days=2)).replace(hour=10, minute=0)  # Sat -> Mon 10:00
    return start_dt.isoformat(), end_dt.isoformat()

def infer_budget_usd(uploaded_files: list, fallback: int = 10) -> int:
    """Return recommended daily budget: #videos × $10 (or fallback if none)."""
    n_videos = sum(1 for u in (uploaded_files or []) if pathlib.Path(u.name).suffix.lower() == ".mp4")
    return max(1, n_videos * 10) if n_videos else fallback

def dollars_to_minor(usd: float) -> int:
    """Convert USD to Meta minor units ($1 → 100)."""
    return int(round(usd * 100))

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

def fetch_url_to_tmp(url: str, timeout: int = 600) -> dict:
    """Download a remote video URL to a temp file and return {'name': str, 'path': str}."""
    url = _normalize_drive_url(url)
    headers = {"User-Agent": "Mozilla/5.0"}
    with requests.get(url, headers=headers, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        ctype = r.headers.get("Content-Type", "").lower()
        disp = r.headers.get("Content-Disposition", "")
        # Guess filename
        name = None
        if "filename=" in disp:
            name = disp.split("filename=")[-1].strip(' \'"')
        if not name:
            name = url.split("?")[0].rstrip("/").split("/")[-1] or "video.mp4"
        # Ensure extension
        if not name.lower().endswith((".mp4", ".mpeg4")):
            if "mp4" in ctype or "video" in ctype:
                name = f"{name}.mp4"

        # Validate content-type
        if not ("video" in ctype or name.lower().endswith((".mp4", ".mpeg4"))):
            raise ValueError(f"URL is not a video. Content-Type={ctype!r}")

        # Stream to temp file (1MB chunks)
        suffix = pathlib.Path(name).suffix.lower() or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    tmp.write(chunk)
            local_path = tmp.name

    return {"name": pathlib.Path(name).name, "path": local_path}

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

def init_fb_from_secrets() -> AdAccount:
    """Initialize Meta SDK from Streamlit secrets; return the XP HERO AdAccount."""
    _require_fb()
    FacebookAdsApi.init(
        access_token=st.secrets.get("access_token", ""),
        app_id=st.secrets.get("app_id", ""),
        app_secret=st.secrets.get("app_secret", ""),
    )
    return AdAccount("act_692755193188182")   # XP HERO ad account

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
    adset_prefix: str,
    country: str,
    daily_budget_usd: int,
    age_min: int,
    start_iso: str,
    end_iso: str,
    suffix_number: int | None = None,
    promoted_object: dict | None = None,
    optimization_goal: str = "APP_INSTALLS",
) -> str:
    """Create a paused ad set with user-chosen settings; return adset_id."""
    suffix = f"{suffix_number}th" if suffix_number else next_nth_suffix(account, adset_prefix)
    name = f"{adset_prefix}_{suffix}"
    adset = account.create_ad_set(
        fields=[],
        params={
            "name": name,
            "campaign_id": campaign_id,
            "daily_budget": dollars_to_minor(daily_budget_usd),
            "billing_event": AdSet.BillingEvent.impressions,
            "optimization_goal": getattr(
                AdSet.OptimizationGoal, optimization_goal.lower(), AdSet.OptimizationGoal.app_installs
            ),
            "bid_strategy": "LOWEST_COST_WITHOUT_CAP",
            "targeting": {
                "geo_locations": {"countries": [country]},
                "age_min": max(13, int(age_min)),
                "publisher_platforms": ["facebook", "instagram"],
                "device_platforms": ["mobile"],
            },
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
):
    """For each MP4: upload video → create creative → create paused ad in the given ad set."""
    allowed = {".mp4", ".mpeg4"}
    videos = [u for u in uploaded_files if pathlib.Path(u.name).suffix.lower() in allowed]
    for u in videos:
        path = _save_uploadedfile_tmp(u)
        video = account.create_ad_video(params={"file": path})
        creative = account.create_ad_creative(
            fields=[],
            params={
                "name": u.name,
                "object_story_spec": {
                    "page_id": page_id,
                    "video_data": {
                        "video_id": video["id"],
                        "title": u.name,
                        "message": "",
                    },
                },
            },
        )
        account.create_ad(
            fields=[],
            params={
                "name": make_ad_name(u.name, ad_name_prefix),
                "adset_id": adset_id,
                "creative": {"creative_id": creative["id"]},
                "status": Ad.Status.paused,
            },
        )

def _plan_upload(account: AdAccount, *, campaign_id: str, adset_prefix: str, page_id: str, uploaded_files: list, settings: dict) -> dict:
    """Compute what would be created (no writes): ad set name, budget/schedule, and ad names."""
    # Schedule
    start_iso = settings.get("start_iso")
    end_iso = settings.get("end_iso")
    if not (start_iso and end_iso):
        start_iso, end_iso = next_sat_0000_and_mon_1000_kst()

    # Ad set suffix
    if settings.get("suffix_number"):
        suffix_str = f"{int(settings['suffix_number'])}th"
    else:
        suffix_str = next_nth_suffix(account, adset_prefix)  # read-only scan of ad sets

    # Videos & ad names
    allowed = {".mp4", ".mpeg4"}
    remote = st.session_state.remote_videos.get(settings.get("game_key", ""), [])
    vids = [u for u in uploaded_files if pathlib.Path(u.name).suffix.lower() in allowed] + remote

    ad_name_prefix = settings.get("ad_name_prefix") if settings.get("ad_name_mode") == "Prefix + filename" else None
    def _fname(x): return x.name if hasattr(x, "name") else x["name"]
    ad_names = [make_ad_name(_fname(u), ad_name_prefix) for u in vids]

    return {
        "campaign_id": campaign_id,
        "adset_name": f"{adset_prefix}_{suffix_str}",
        "country": settings.get("country", "US"),
        "age_min": int(settings.get("age_min", 18)),
        "budget_usd_per_day": int(settings.get("daily_budget_usd", infer_budget_usd(uploaded_files))),
        "start_iso": start_iso,
        "end_iso": end_iso,
        "page_id": page_id,
        "n_videos": len(vids),
        "ad_names": ad_names,
    }

def upload_to_facebook(game_name: str, uploaded_files: list, settings: dict, *, simulate: bool = False):
    """Create the chosen ad set and one paused ad per video (simulate=True returns plan only)."""
    account = init_fb_from_secrets()

    mapping = {
        "XP HERO": {
            "campaign_id": "120218934861590118",  # weaponrpg_aos_facebook_us_creativetest
            "adset_prefix": "weaponrpg_aos_facebook_us_creativetest",
            "page_id": st.secrets["page_id"],
        }
    }
    if game_name not in mapping:
        raise ValueError(f"No FB mapping configured for game: {game_name}")
    cfg = mapping[game_name]

    # Build plan (no writes)
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

    # Optional promoted_object (left None by default)
    promoted_object = None
    if settings.get("app_store") == "Google Play 스토어":
        # Fill these when you’re ready to track installs end-to-end:
        # promoted_object = {
        #     "application_id": "YOUR_FB_APP_ID",
        #     "object_store_url": "https://play.google.com/store/apps/details?id=YOUR_APP_ID",
        #     "application_store": "GOOGLE_PLAY",
        # }
        promoted_object = None

    # Create ad set as per plan
    suffix_num = None
    try:
        # Only pass numeric suffix if user explicitly provided it
        if settings.get("suffix_number"):
            suffix_num = int(settings["suffix_number"])
    except Exception:
        suffix_num = None

    adset_id = create_creativetest_adset(
        account=account,
        campaign_id=cfg["campaign_id"],
        adset_prefix=cfg["adset_prefix"],
        country=plan["country"],
        daily_budget_usd=plan["budget_usd_per_day"],
        age_min=plan["age_min"],
        start_iso=plan["start_iso"],
        end_iso=plan["end_iso"],
        suffix_number=suffix_num,
        promoted_object=promoted_object,
        optimization_goal="APP_INSTALLS",
    )

    # Create ads (one per MP4)
    ad_name_prefix = settings.get("ad_name_prefix") if settings.get("ad_name_mode") == "Prefix + filename" else None
    upload_videos_create_ads(
        account=account,
        page_id=cfg["page_id"],
        adset_id=adset_id,
        uploaded_files=uploaded_files,
        ad_name_prefix=ad_name_prefix,
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
            url_val = st.text_input("Paste a direct or Drive link", key=f"urlinput_{i}", placeholder="https://...")
            if st.button("Add URL video", key=f"addurl_{i}"):
                try:
                    meta = fetch_url_to_tmp(url_val)
                    lst = st.session_state.remote_videos.get(game, [])
                    lst.append(meta)
                    st.session_state.remote_videos[game] = lst
                    st.success(f"Added: {meta['name']}")
                except Exception as e:
                    st.exception(e)
                    st.error("Could not fetch this URL. Check the link or permissions.")

            remote_list = st.session_state.remote_videos.get(game, [])
            if remote_list:
                st.caption("Server-downloaded videos:")
                for it in remote_list[:20]:
                    st.write("•", it["name"])
                if st.button("Clear URL videos", key=f"clearurl_{i}"):
                    st.session_state.remote_videos[game] = []
                    st.info("Cleared URL videos for this game.")
                    st.rerun()                                            # next run will clear before creating widget

            # --- ACTION BUTTONS ---
            st.markdown("### Actions")
            dry_run = st.checkbox("Dry run (no API writes)", value=True, key=f"dryrun_{i}")

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

            saved = st.session_state.uploads.get(game, [])
            suggested_budget = infer_budget_usd(saved or uploaded or [])
            cur = st.session_state.settings.get(game, {})
            default_start_iso, default_end_iso = next_sat_0000_and_mon_1000_kst()

            country = st.text_input("Country", value=cur.get("country", "US"), key=f"country_{i}")
            daily_budget_usd = st.number_input(
                "Daily Budget (USD/day)", min_value=1, value=int(cur.get("daily_budget_usd", suggested_budget)), key=f"daily_budget_{i}"
            )

            try:
                suffix_default = int(cur.get("suffix_number")) if cur.get("suffix_number") not in (None, "", "None") else 0
            except (ValueError, TypeError):
                suffix_default = 0

            suffix_number = st.number_input(
                "Ad set last suffix (e.g., 35)",
                min_value=0,
                step=1,
                value=suffix_default,
                help="Leave 0 to auto-pick next (e.g., 35th).",
                key=f"suffix_{i}",
            )

            app_store = st.selectbox(
                "App Store", ["Google Play 스토어", "Apple App Store"],
                index=0 if cur.get("app_store", "Google Play 스토어") == "Google Play 스토어" else 1,
                key=f"appstore_{i}",
            )
            optimize_goal_label = st.selectbox(
                "Optimize towards", ["결과당 비용 - 모바일 앱 설치"], index=0, key=f"optgoal_{i}"
            )
            age_min = st.number_input("Age (min)", min_value=13, value=int(cur.get("age_min", 18)), key=f"age_{i}")

            ad_name_mode = st.selectbox(
                "Ad name", ["Use video filename", "Prefix + filename"],
                index=1 if cur.get("ad_name_mode") == "Prefix + filename" else 0,
                key=f"adname_mode_{i}",
            )

            ad_name_prefix = ""
            if ad_name_mode == "Prefix + filename":
                ad_name_prefix = st.text_input("Ad name prefix", value=cur.get("ad_name_prefix", ""), key=f"adname_prefix_{i}")

            st.selectbox("Creative type", ["Video ad"], index=0, disabled=True, key=f"ctype_{i}")

            start_iso = st.text_input(
                "Start date/time (ISO, KST)",
                value=cur.get("start_iso", default_start_iso),
                help="e.g., 2025-11-15T00:00:00+09:00",
                key=f"start_{i}",
            )
            end_iso = st.text_input(
                "End date/time (ISO, KST)",
                value=cur.get("end_iso", default_end_iso),
                help="e.g., 2025-11-17T10:00:00+09:00",
                key=f"end_{i}",
            )

            st.session_state.settings[game] = {
            "country": (country or "US").strip(),
            "daily_budget_usd": int(daily_budget_usd),
            "suffix_number": int(suffix_number) if int(suffix_number) > 0 else None,
            "app_store": app_store,
            "age_min": int(age_min),
            "ad_name_mode": ad_name_mode,
            "ad_name_prefix": ad_name_prefix.strip(),
            "start_iso": start_iso.strip(),
            "end_iso": end_iso.strip(),
            "game_key": game,  # <-- add this
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
                    if dry_run:
                        ok_msg_placeholder.info("Dry run only — no ad set or ads were created.")
                        st.write("**Planned Ad Set:**", plan["adset_name"])
                        st.write("**Campaign ID:**", plan["campaign_id"])
                        st.write("**Country/Age:**", plan["country"], "/", plan["age_min"], "+")
                        st.write("**Budget (USD/day):**", plan["budget_usd_per_day"])
                        st.write("**Schedule (KST):**", f"{plan['start_iso']} → {plan['end_iso']}")
                        st.write("**# of videos:**", plan["n_videos"])
                        if plan["ad_names"]:
                            st.markdown("**Ad names to be created:**")
                            for nm in plan["ad_names"]:
                                st.write("•", nm)
                    else:
                        ok_msg_placeholder.success(msg + " Uploaded to Meta (ads created as PAUSED).")
                        if plan.get("adset_id"):
                            st.write("**Created Ad Set ID:**", plan["adset_id"])
                except Exception as e:
                    st.exception(e)
                    ok_msg_placeholder.error("Meta upload failed. See error above.")


        if clr:
            st.session_state.uploads.pop(game, None)
            st.session_state.remote_videos.pop(game, None)  # also clear URL videos
            st.session_state[f"uploader_{i}"] = None
            st.session_state.settings.pop(game, None)
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
