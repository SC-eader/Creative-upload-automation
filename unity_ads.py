"""Unity Ads helpers for Creative 자동 업로드 Streamlit app."""

from __future__ import annotations

from typing import Dict, List, Any
from datetime import datetime, timedelta, timezone
import logging
import pathlib
import re

import streamlit as st

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------
# Session-state helpers (settings per game)
# --------------------------------------------------------------------
def _ensure_unity_settings_state() -> None:
    """Make sure we have st.session_state.unity_settings for per-game Unity settings."""
    if "unity_settings" not in st.session_state:
        st.session_state.unity_settings = {}


def get_unity_settings(game: str) -> Dict:
    """Return the Unity settings dict for the given game (empty if not set yet)."""
    _ensure_unity_settings_state()
    return st.session_state.unity_settings.get(game, {})


# --------------------------------------------------------------------
# Date helper: next Sat 00:00 → next Mon 12:00 (KST)
# --------------------------------------------------------------------
ASIA_SEOUL = timezone(timedelta(hours=9))


def next_sat_0000_and_mon_1200_kst(today: datetime | None = None) -> tuple[str, str]:
    """
    Compute (start_iso, end_iso) in KST:
      - start: next Saturday 00:00
      - end:   next Monday 12:00
    Returned strings are ISO8601 with +09:00 offset.
    """
    now = (today or datetime.now(ASIA_SEOUL)).astimezone(ASIA_SEOUL)
    base = now.replace(hour=0, minute=0, second=0, microsecond=0)
    # Monday=0 ... Saturday=5, Sunday=6
    days_until_sat = (5 - base.weekday()) % 7 or 7
    start_dt = (base + timedelta(days=days_until_sat)).replace(hour=0, minute=0)
    end_dt = (start_dt + timedelta(days=2)).replace(hour=12, minute=0)  # Sat → Mon 12:00
    return start_dt.isoformat(), end_dt.isoformat()


# --------------------------------------------------------------------
# Naming helper: video### from filename
# --------------------------------------------------------------------
def unity_creative_name_from_filename(filename: str) -> str:
    """
    Build a Unity creative/pack name like 'video123' from a filename.
    - If the name contains a 3-digit code (e.g. 'video123_9x16.mp4', '123.mp4'),
      it uses that → 'video123'.
    - If no 3-digit code is found, it falls back to 'video000'.
    """
    stem = pathlib.Path(filename).stem  # drop extension
    # last 3-digit group in the stem (e.g. 123 from video123_9x16)
    m = re.search(r"(\d{3})(?!.*\d)", stem)
    code = m.group(1) if m else "000"
    return f"video{code}"


# --------------------------------------------------------------------
# Unity settings UI (right column per game)
# --------------------------------------------------------------------
def render_unity_settings_panel(right_col, game: str, idx: int) -> None:
    """
    Render the Unity Ads settings panel on the right side for a single game tab.

    right_col: Streamlit container (the right column).
    game:     Game name (used as key in session_state).
    idx:      Tab index, only used to keep widget keys unique.
    """
    _ensure_unity_settings_state()

    with right_col:
        st.markdown(f"### {game} Unity Settings")

        cur = st.session_state.unity_settings.get(game, {})

        unity_title_id = st.text_input(
            "Unity Title ID",
            value=cur.get("title_id", ""),
            key=f"unity_title_{idx}",
            help="각 게임의 Unity Ads(ironSource) Title ID",
        )

        unity_campaign_id = st.text_input(
            "기본 Campaign ID (선택)",
            value=cur.get("campaign_id", ""),
            key=f"unity_campaign_{idx}",
            help="이 캠페인 설정을 복사해서 크리에이티브 테스트를 만들고 싶을 때 사용 (선택).",
        )

        unity_org_id = st.text_input(
            "Unity Org ID",
            value=cur.get("org_id", ""),
            key=f"unity_org_{idx}",
            help="서비스 계정이 속한 Organization ID",
        )

        unity_client_id = st.text_input(
            "Unity Service client_id",
            value=cur.get("client_id", ""),
            key=f"unity_client_id_{idx}",
        )

        unity_client_secret = st.text_input(
            "Unity Service client_secret",
            value=cur.get("client_secret", ""),
            key=f"unity_client_secret_{idx}",
            type="password",
        )

        unity_daily_budget = st.number_input(
            "Unity 일일 예산 (USD, 선택)",
            min_value=0,
            value=int(cur.get("daily_budget_usd", 0)),
            key=f"unity_budget_{idx}",
        )

        # Save updated values back into session_state
        st.session_state.unity_settings[game] = {
            "title_id": unity_title_id.strip(),
            "campaign_id": unity_campaign_id.strip(),
            "org_id": unity_org_id.strip(),
            "client_id": unity_client_id.strip(),
            "client_secret": unity_client_secret.strip(),
            "daily_budget_usd": int(unity_daily_budget),
        }


# --------------------------------------------------------------------
# Main Unity upload entry point used by streamlit_app.py
# --------------------------------------------------------------------
def upload_unity_creatives_to_campaign(
    *,
    game: str,
    videos: List[Dict[str, Any]],
    settings: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Take imported videos for a game and prepare a Unity campaign update.

    This function currently:
      - Validates required Unity settings (title_id, campaign_id, org/client).
      - Computes default start/end (next Sat 00:00 → next Mon 12:00 KST).
      - Derives creative pack names like 'video123' from each filename.
      - Returns a summary dict used by the Streamlit UI.

    TODO (when you plug in the real Unity Advertising Management API):
      1) Get an access token using org_id / client_id / client_secret.
      2) For each video:
         - Upload the asset for the given title_id.
         - Create a creative pack with name = unity_creative_name_from_filename(name).
      3) Update the target campaign:
         - Unassign previous creatives.
         - Assign the new ones.
         - Set start/end dates and enable auto-start.
    """
    # ---- 1) Validate settings ----
    title_id = (settings.get("title_id") or "").strip()
    campaign_id = (settings.get("campaign_id") or "").strip()
    org_id = (settings.get("org_id") or "").strip()
    client_id = (settings.get("client_id") or "").strip()
    client_secret = (settings.get("client_secret") or "").strip()
    daily_budget_usd = int(settings.get("daily_budget_usd") or 0)

    missing = []
    if not title_id:
        missing.append("Unity Title ID")
    if not campaign_id:
        missing.append("Campaign ID")
    if not org_id:
        missing.append("Org ID")
    if not client_id:
        missing.append("Service client_id")
    if not client_secret:
        missing.append("Service client_secret")

    if missing:
        raise RuntimeError(
            "Unity Ads 설정이 부족합니다. 다음 항목을 채워주세요:\n- "
            + "\n- ".join(missing)
        )

    # ---- 2) Compute schedule (next Sat → next Mon) ----
    start_iso, end_iso = next_sat_0000_and_mon_1200_kst()

    # ---- 3) Derive creative names: video### ----
    creative_names: List[str] = []
    for v in videos or []:
        # videos from Drive import are dicts: {"name": ..., "path": ...}
        name = v.get("name") or "video000.mp4"
        creative_name = unity_creative_name_from_filename(name)
        creative_names.append(creative_name)

    # Log for debugging inside Streamlit (optional)
    logger.info(
        "Unity upload plan for game=%s, campaign=%s: %d videos -> %s",
        game,
        campaign_id,
        len(creative_names),
        creative_names,
    )

    # ------------------------------------------------------------------
    # 4) TODO: real Unity API calls go here
    # ------------------------------------------------------------------
    #
    # Pseudo-code (fill in with real endpoints from Unity docs):
    #
    #   token = _unity_get_access_token(org_id, client_id, client_secret)
    #
    #   new_creative_ids = []
    #   for video, cname in zip(videos, creative_names):
    #       asset_id = _unity_upload_video_asset(token, title_id, video["path"])
    #       creative_id = _unity_create_creative_pack(
    #           token=token,
    #           title_id=title_id,
    #           name=cname,
    #           asset_id=asset_id,
    #       )
    #       new_creative_ids.append(creative_id)
    #
    #   removed_ids = _unity_update_campaign_creatives(
    #       token=token,
    #       campaign_id=campaign_id,
    #       new_creative_ids=new_creative_ids,
    #   )
    #
    #   _unity_update_campaign_schedule(
    #       token=token,
    #       campaign_id=campaign_id,
    #       start=start_iso,
    #       end=end_iso,
    #       daily_budget_usd=daily_budget_usd,
    #       auto_start=True,
    #   )
    #
    # For now we just simulate the response so the Streamlit UI works.

    simulated_creative_ids = creative_names  # stand-in until real IDs exist
    removed_ids: List[str] = []  # will be real IDs once you implement removal
    errors: List[str] = []       # collect API errors here later

    return {
        "game": game,
        "campaign_id": campaign_id,
        "title_id": title_id,
        "start_iso": start_iso,
        "end_iso": end_iso,
        "daily_budget_usd": daily_budget_usd,
        "creative_ids": simulated_creative_ids,
        "removed_ids": removed_ids,
        "errors": errors,
    }