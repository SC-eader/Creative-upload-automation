"""Unity Ads helpers for Creative 자동 업로드 Streamlit app."""

from __future__ import annotations

from typing import Dict, List, Any
from datetime import datetime, timedelta, timezone
import logging
import pathlib
import re
import os
import json
import hashlib

import time
import requests
import streamlit as st

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------
# Unity config from secrets.toml
# --------------------------------------------------------------------
unity_cfg = st.secrets.get("unity", {}) or {}

UNITY_ORG_ID_DEFAULT = unity_cfg.get("organization_id", "")
UNITY_CLIENT_ID_DEFAULT = unity_cfg.get("client_id", "")
UNITY_CLIENT_SECRET_DEFAULT = unity_cfg.get("client_secret", "")
UNITY_AUTH_HEADER_DEFAULT = unity_cfg.get("authorization_header", "")

_raw_game_ids = unity_cfg.get("game_ids", {}) or {}
_raw_campaign_ids = unity_cfg.get("campaign_ids", {}) or {}

UNITY_GAME_IDS: Dict[str, str] = {
    str(game): str(title_id)
    for game, title_id in _raw_game_ids.items()
}

UNITY_CAMPAIGN_IDS: Dict[str, List[str]] = {}
for game, val in _raw_campaign_ids.items():
    if isinstance(val, dict) and "ids" in val:
        UNITY_CAMPAIGN_IDS[str(game)] = [str(x) for x in (val.get("ids") or [])]
    elif isinstance(val, (list, tuple)):
        UNITY_CAMPAIGN_IDS[str(game)] = [str(x) for x in val]
    elif isinstance(val, str):
        UNITY_CAMPAIGN_IDS[str(game)] = [val]

UNITY_BASE_URL = "https://services.api.unity.com/advertise/v1"

# --------------------------------------------------------------------
# Session-state helpers
# --------------------------------------------------------------------
def _ensure_unity_settings_state() -> None:
    if "unity_settings" not in st.session_state:
        st.session_state.unity_settings = {}

def get_unity_settings(game: str) -> Dict:
    _ensure_unity_settings_state()
    return st.session_state.unity_settings.get(game, {})

# --------------------------------------------------------------------
# Unity settings UI
# --------------------------------------------------------------------
def render_unity_settings_panel(right_col, game: str, idx: int) -> None:
    _ensure_unity_settings_state()

    with right_col:
        st.markdown(f"#### {game} Unity Settings")
        cur = st.session_state.unity_settings.get(game, {})

        secret_title_id = str(UNITY_GAME_IDS.get(game, ""))
        secret_campaign_ids = UNITY_CAMPAIGN_IDS.get(game, []) or []
        default_campaign_id_val = secret_campaign_ids[0] if secret_campaign_ids else ""

        title_key = f"unity_title_{idx}"
        campaign_key = f"unity_campaign_{idx}"

        if st.session_state.get(title_key) == "" and secret_title_id:
            st.session_state[title_key] = secret_title_id
        if (not secret_campaign_ids and st.session_state.get(campaign_key) == "" and default_campaign_id_val):
            st.session_state[campaign_key] = default_campaign_id_val

        current_title_id = cur.get("title_id") or st.session_state.get(title_key) or secret_title_id
        current_campaign_id = cur.get("campaign_id") or st.session_state.get(campaign_key) or default_campaign_id_val

        default_org_id = cur.get("org_id") or UNITY_ORG_ID_DEFAULT
        default_client_id = cur.get("client_id") or UNITY_CLIENT_ID_DEFAULT
        default_client_secret = cur.get("client_secret") or UNITY_CLIENT_SECRET_DEFAULT

        unity_title_id = current_title_id
        st.session_state[title_key] = unity_title_id
        unity_campaign_id = current_campaign_id
        st.session_state[campaign_key] = unity_campaign_id
        unity_org_id = default_org_id
        st.session_state[f"unity_org_{idx}"] = unity_org_id
        unity_client_id = default_client_id
        st.session_state[f"unity_client_id_{idx}"] = unity_client_id
        unity_client_secret = default_client_secret
        st.session_state[f"unity_client_secret_{idx}"] = unity_client_secret

        unity_daily_budget = 0

        st.markdown("#### Playable 선택")
        drive_playables = [
            v for v in (st.session_state.remote_videos.get(game, []) if "remote_videos" in st.session_state else [])
            if "playable" in (v.get("name") or "").lower()
        ]
        drive_options = [p["name"] for p in drive_playables]
        prev_drive_playable = cur.get("selected_playable", "")

        selected_drive_playable = st.selectbox(
            "Drive에서 가져온 플레이어블",
            options=["(선택 안 함)"] + drive_options,
            index=(drive_options.index(prev_drive_playable) + 1) if prev_drive_playable in drive_options else 0,
            key=f"unity_playable_{idx}",
        )
        chosen_drive_playable = selected_drive_playable if selected_drive_playable != "(선택 안 함)" else ""

        existing_labels: List[str] = ["(선택 안 함)"]
        existing_id_by_label: Dict[str, str] = {}
        prev_existing_label = cur.get("existing_playable_label", "")

        try:
            org_for_list = (unity_org_id or UNITY_ORG_ID_DEFAULT).strip()
            title_for_list = (unity_title_id or secret_title_id).strip()

            if org_for_list and title_for_list:
                # STRICT FILTER: Playables only
                playable_creatives = _unity_list_playable_creatives(org_id=org_for_list, title_id=title_for_list)
                for cr in playable_creatives:
                    cr_id = str(cr.get("id") or "")
                    cr_name = cr.get("name") or "(no name)"
                    cr_type = cr.get("type", "")
                    if not cr_id: continue
                    label = f"{cr_name} ({cr_type}) [{cr_id}]"
                    existing_labels.append(label)
                    existing_id_by_label[label] = cr_id
        except Exception as e:
            st.info(f"Unity playable 목록을 불러오지 못했습니다: {e}")

        try:
            existing_default_idx = existing_labels.index(prev_existing_label)
        except ValueError:
            existing_default_idx = 0

        selected_existing_label = st.selectbox(
            "Unity에 이미 있는 playable",
            options=existing_labels,
            index=existing_default_idx,
            key=f"unity_existing_playable_{idx}",
        )

        existing_playable_id = ""
        if selected_existing_label != "(선택 안 함)":
            existing_playable_id = existing_id_by_label.get(selected_existing_label, "")

        st.warning(
            "Unity creative pack은 **9:16 영상 1개 + 16:9 영상 1개 + 1개의 playable** 조합을 기준으로 생성됩니다."
        )

        st.session_state.unity_settings[game] = {
            "title_id": (unity_title_id or "").strip(),
            "campaign_id": (unity_campaign_id or "").strip(),
            "org_id": (unity_org_id or "").strip(),
            "daily_budget_usd": int(unity_daily_budget),
            "selected_playable": chosen_drive_playable,
            "existing_playable_id": existing_playable_id,
            "existing_playable_label": selected_existing_label,
        }

# --------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------

def _get_upload_state_key(game: str, campaign_id: str) -> str:
    """Generate a unique key for tracking upload state per game/campaign."""
    return f"unity_upload_state_{game}_{campaign_id}"

def _init_upload_state(game: str, campaign_id: str, videos: List[Dict]) -> Dict:
    """
    Initialize or load upload state for resumability.
    
    State structure:
    {
        "video_creatives": {
            "video_filename.mp4": "creative_id_12345" or None
        },
        "playable_creative": "creative_id_67890" or None,
        "creative_packs": {
            "pack_name": "pack_id_abc" or None
        },
        "completed_packs": ["pack_id_1", "pack_id_2"],
        "total_expected": 10
    }
    """
    key = _get_upload_state_key(game, campaign_id)
    
    if key not in st.session_state:
        # Initialize new state
        state = {
            "video_creatives": {},
            "playable_creative": None,
            "creative_packs": {},
            "completed_packs": [],
            "total_expected": 0
        }
        
        # Pre-populate video names
        for v in videos or []:
            name = v.get("name", "")
            if "playable" not in name.lower():
                state["video_creatives"][name] = None
        
        st.session_state[key] = state
    
    return st.session_state[key]


ASIA_SEOUL = timezone(timedelta(hours=9))

def next_sat_0000_kst(today: datetime | None = None) -> str:
    now = (today or datetime.now(ASIA_SEOUL)).astimezone(ASIA_SEOUL)
    base = now.replace(hour=0, minute=0, second=0, microsecond=0)
    days_until_sat = (5 - base.weekday()) % 7 or 7
    start_dt = (base + timedelta(days=days_until_sat)).replace(hour=9, minute=0)
    return start_dt.isoformat()

def unity_creative_name_from_filename(filename: str) -> str:
    stem = pathlib.Path(filename).stem
    m = re.search(r"(\d{3})(?!.*\d)", stem)
    code = m.group(1) if m else "000"
    return f"video{code}"

# --------------------------------------------------------------------
# API Helpers
# --------------------------------------------------------------------
def _unity_headers() -> dict:
    if not UNITY_AUTH_HEADER_DEFAULT:
        raise RuntimeError("unity.authorization_header is missing in secrets.toml")
    return {"Authorization": UNITY_AUTH_HEADER_DEFAULT, "Content-Type": "application/json"}

def _unity_post(path: str, json_body: dict) -> dict:
    url = f"{UNITY_BASE_URL.rstrip('/')}/{path.lstrip('/')}"
    last_error: Exception | None = None

    for attempt in range(8):
        try:
            resp = requests.post(url, headers=_unity_headers(), json=json_body, timeout=60)
            
            if resp.status_code == 429:
                detail = resp.text[:400]
                if "quota" in detail.lower():
                    raise RuntimeError(f"Unity Quota Exceeded (STOPPING): {detail}")
                
                sleep_sec = 2 ** (attempt + 1)
                logger.warning(f"Unity 429 Rate Limit (attempt {attempt+1}/8). Sleeping {sleep_sec}s...")
                time.sleep(sleep_sec)
                continue

            if not resp.ok:
                raise RuntimeError(f"Unity POST {path} failed ({resp.status_code}): {resp.text[:400]}")

            return resp.json()
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request failed: {e}. Retrying...")
            time.sleep(2)
            last_error = e

    raise last_error or RuntimeError(f"Unity POST {path} failed after retries.")

def _unity_put(path: str, json_body: dict) -> dict:
    url = f"{UNITY_BASE_URL.rstrip('/')}/{path.lstrip('/')}"
    resp = requests.put(url, headers=_unity_headers(), json=json_body, timeout=60)
    if not resp.ok:
        raise RuntimeError(f"Unity PUT {path} failed ({resp.status_code}): {resp.text[:400]}")
    return resp.json()

def _unity_get(path: str, params: dict | None = None) -> dict:
    url = f"{UNITY_BASE_URL.rstrip('/')}/{path.lstrip('/')}"
    resp = requests.get(url, headers=_unity_headers(), params=params or {}, timeout=60)
    if not resp.ok:
        raise RuntimeError(f"Unity GET {path} failed ({resp.status_code}): {resp.text[:400]}")
    return resp.json()

def _unity_delete(path: str) -> None:
    url = f"{UNITY_BASE_URL.rstrip('/')}/{path.lstrip('/')}"
    resp = requests.delete(url, headers=_unity_headers(), timeout=60)
    if not resp.ok:
        raise RuntimeError(f"Unity DELETE {path} failed ({resp.status_code}): {resp.text[:400]}")

# --------------------------------------------------------------------
# Creative Helpers
# --------------------------------------------------------------------
def _unity_list_assigned_creative_packs(*, org_id: str, title_id: str, campaign_id: str) -> List[dict]:
    path = f"organizations/{org_id}/apps/{title_id}/campaigns/{campaign_id}/assigned-creative-packs"
    
    # 1. No limit param (as it caused 400 error)
    meta = _unity_get(path)

    # 2. Return the correct list structure
    if isinstance(meta, list): return meta
    if isinstance(meta, dict):
        # FIX: Check 'results' which is standard for this API
        if isinstance(meta.get("results"), list): return meta["results"]
        if isinstance(meta.get("items"), list): return meta["items"]
        if isinstance(meta.get("data"), list): return meta["data"]
        
        for v in meta.values():
            if isinstance(v, list): return v

    return []

def _unity_assign_creative_pack(*, org_id: str, title_id: str, campaign_id: str, creative_pack_id: str) -> None:
    path = f"organizations/{org_id}/apps/{title_id}/campaigns/{campaign_id}/assigned-creative-packs"
    _unity_post(path, {"id": creative_pack_id})

def _unity_unassign_creative_pack(*, org_id: str, title_id: str, campaign_id: str, assigned_creative_pack_id: str) -> None:
    path = f"organizations/{org_id}/apps/{title_id}/campaigns/{campaign_id}/assigned-creative-packs/{assigned_creative_pack_id}"
    _unity_delete(path)

def _unity_unassign_with_retry(*, org_id: str, title_id: str, campaign_id: str, assigned_creative_pack_id: str, max_retries: int = 3) -> None:
    """Unassign with exponential backoff on rate limit."""
    for attempt in range(max_retries):
        try:
            _unity_unassign_creative_pack(
                org_id=org_id,
                title_id=title_id,
                campaign_id=campaign_id,
                assigned_creative_pack_id=assigned_creative_pack_id
            )
            return  # Success
        except RuntimeError as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait_time = 2 ** (attempt + 1)  # 2, 4, 8 seconds
                logger.warning(f"Rate limit hit, waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                raise  # Give up after retries

def _unity_get_creative(*, org_id: str, title_id: str, creative_id: str) -> dict:
    path = f"organizations/{org_id}/apps/{title_id}/creatives/{creative_id}"
    return _unity_get(path)

def _unity_create_video_creative(*, org_id: str, title_id: str, video_path: str, name: str, language: str = "en") -> str:
    if not os.path.isfile(video_path):
        raise RuntimeError(f"Video path does not exist: {video_path!r}")

    # FIX: Use original name for fileName, not temp file path
    display_filename = name
    if not display_filename.lower().endswith(".mp4"):
        display_filename += ".mp4"

    creative_info = {
        "name": name, 
        "language": language, 
        "video": {"fileName": display_filename}
    }
    
    url = f"{UNITY_BASE_URL.rstrip('/')}/organizations/{org_id}/apps/{title_id}/creatives"
    headers = {"Authorization": UNITY_AUTH_HEADER_DEFAULT}

    for attempt in range(8):
        try:
            with open(video_path, "rb") as f:
                files = {
                    "creativeInfo": (None, json.dumps(creative_info), "application/json"),
                    "videoFile": (display_filename, f, "video/mp4"),
                }
                resp = requests.post(url, headers=headers, files=files, timeout=300)

            if resp.status_code == 429:
                detail = (resp.text or "")[:400].lower()
                if "quota" in detail:
                    raise RuntimeError(f"Unity Quota Exceeded (STOPPING): {detail}")
                sleep_sec = 5 * (attempt + 1)
                time.sleep(sleep_sec)
                continue

            if not resp.ok:
                raise RuntimeError(f"Unity create creative failed ({resp.status_code}): {resp.text[:400]}")

            body = resp.json()
            return str(body.get("id") or body.get("creativeId"))
        except Exception as e:
            if "Quota Exceeded" in str(e): raise e
            time.sleep(5)

    raise RuntimeError("Unity create creative failed after multiple retries.")

def _unity_create_playable_creative(*, org_id: str, title_id: str, playable_path: str, name: str, language: str = "en") -> str:
    if not os.path.isfile(playable_path):
        raise RuntimeError(f"Playable path does not exist: {playable_path!r}")

    file_name = os.path.basename(playable_path)
    creative_info = {"name": name, "language": language, "playable": {"fileName": file_name}}
    url = f"{UNITY_BASE_URL.rstrip('/')}/organizations/{org_id}/apps/{title_id}/creatives"
    headers = {"Authorization": UNITY_AUTH_HEADER_DEFAULT}

    for attempt in range(8):
        try:
            with open(playable_path, "rb") as f:
                files = {
                    "creativeInfo": (None, json.dumps(creative_info), "application/json"),
                    "playableFile": (file_name, f, "text/html"),
                }
                resp = requests.post(url, headers=headers, files=files, timeout=300)

            if resp.status_code == 429:
                time.sleep(3 * (attempt + 1))
                continue

            if not resp.ok:
                raise RuntimeError(f"Unity create playable failed ({resp.status_code}): {resp.text[:400]}")

            body = resp.json()
            return str(body.get("id") or body.get("creativeId"))
        except Exception as e:
            time.sleep(3)

    raise RuntimeError("Unity create playable creative failed after retries.")

def _unity_create_creative_pack(*, org_id: str, title_id: str, pack_name: str, creative_ids: List[str], pack_type: str = "video") -> str:
    clean_ids = [str(x) for x in creative_ids if x]
    
    if len(clean_ids) < 2:
        raise RuntimeError(f"Not enough creative IDs to create a pack: {clean_ids}")

    payload = {
        "name": pack_name,
        "creativeIds": clean_ids,
        "type": pack_type,
    }

    path = f"organizations/{org_id}/apps/{title_id}/creative-packs"
    meta = _unity_post(path, payload)
    
    creative_pack_id = meta.get("id") or meta.get("creativePackId")
    if not creative_pack_id:
        raise RuntimeError(f"Unity creative pack response missing id: {meta}")

    return str(creative_pack_id)

def _unity_list_playable_creatives(*, org_id: str, title_id: str) -> List[dict]:
    path = f"organizations/{org_id}/apps/{title_id}/creatives"
    meta = _unity_get(path)

    items: List[dict] = []
    if isinstance(meta, list): items = meta
    elif isinstance(meta, dict):
        if isinstance(meta.get("items"), list): items = meta["items"]
        elif isinstance(meta.get("data"), list): items = meta["data"]
        else:
            for v in meta.values():
                if isinstance(v, list): items.extend(v)

    playables: List[dict] = []
    for cr in items:
        if not isinstance(cr, dict): continue
        t = (cr.get("type") or "").lower()
        if "playable" in t or "cpe" in t:
             playables.append(cr)

    return playables


def _save_upload_state(game: str, campaign_id: str, state: Dict):
    """Save upload state to session."""
    key = _get_upload_state_key(game, campaign_id)
    st.session_state[key] = state


def _clear_upload_state(game: str, campaign_id: str):
    """Clear upload state (call when upload completes successfully)."""
    key = _get_upload_state_key(game, campaign_id)
    if key in st.session_state:
        del st.session_state[key]


def _check_existing_creative(org_id: str, title_id: str, name: str) -> str | None:
    """
    Check if a creative with this name already exists in Unity.
    Returns creative_id if found, None otherwise.
    """
    try:
        path = f"organizations/{org_id}/apps/{title_id}/creatives"
        meta = _unity_get(path, params={"limit": 100})
        
        items = []
        if isinstance(meta, list):
            items = meta
        elif isinstance(meta, dict):
            items = meta.get("items") or meta.get("data") or []
        
        for creative in items:
            if creative.get("name") == name:
                return str(creative.get("id", ""))
        
        return None
    except Exception as e:
        logger.warning(f"Could not check existing creative: {e}")
        return None


def _check_existing_pack(org_id: str, title_id: str, pack_name: str) -> str | None:
    """
    Check if a creative pack with this name already exists.
    Returns pack_id if found, None otherwise.
    """
    try:
        path = f"organizations/{org_id}/apps/{title_id}/creative-packs"
        meta = _unity_get(path, params={"limit": 100})
        
        items = []
        if isinstance(meta, list):
            items = meta
        elif isinstance(meta, dict):
            items = meta.get("items") or meta.get("data") or []
        
        for pack in items:
            if pack.get("name") == pack_name:
                return str(pack.get("id", ""))
        
        return None
    except Exception as e:
        logger.warning(f"Could not check existing pack: {e}")
        return None
# --------------------------------------------------------------------
# Main Helpers
# --------------------------------------------------------------------

def upload_unity_creatives_to_campaign(
    *, 
    game: str, 
    videos: List[Dict[str, Any]], 
    settings: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Upload Unity creatives and packs with resume support.
    
    If upload fails midway, calling again will:
    - Skip already uploaded creatives
    - Only upload missing items
    - Resume from where it left off
    """
    title_id = (settings.get("title_id") or "").strip() or str(UNITY_GAME_IDS.get(game, ""))
    campaign_id = (settings.get("campaign_id") or "").strip()
    if not campaign_id:
        ids_for_game = UNITY_CAMPAIGN_IDS.get(game) or []
        if ids_for_game:
            campaign_id = str(ids_for_game[0])
    
    org_id = (settings.get("org_id") or "").strip() or UNITY_ORG_ID_DEFAULT
    
    if not all([title_id, campaign_id, org_id]):
        raise RuntimeError("Unity Settings Missing for upload.")

    start_iso = next_sat_0000_kst()
    errors: List[str] = []
    
    # Initialize upload state (loads existing if resuming)
    upload_state = _init_upload_state(game, campaign_id, videos)
    
    # Show resume info if applicable
    existing_packs = len([p for p in upload_state["completed_packs"] if p])
    if existing_packs > 0:
        st.info(
            f"📦 **Resuming Upload**\n\n"
            f"Found {existing_packs} previously created pack(s).\n"
            f"Will skip already uploaded items and continue from where we left off."
        )

    # ========================================
    # 1. PLAYABLE HANDLING
    # ========================================
    playable_name = settings.get("selected_playable") or ""
    existing_playable_id = settings.get("existing_playable_id") or ""
    playable_creative_id: str | None = upload_state.get("playable_creative")

    if not playable_creative_id:
        if playable_name:
            playable_item = next((v for v in (videos or []) if v.get("name") == playable_name), None)
            if playable_item:
                try:
                    # Check if already uploaded
                    playable_creative_id = _check_existing_creative(org_id, title_id, playable_name)
                    
                    if playable_creative_id:
                        st.info(f"✅ Found existing playable: {playable_name}")
                    else:
                        st.info(f"⬆️ Uploading playable: {playable_name}")
                        playable_creative_id = _unity_create_playable_creative(
                            org_id=org_id, 
                            title_id=title_id, 
                            playable_path=playable_item["path"], 
                            name=playable_name
                        )
                    
                    upload_state["playable_creative"] = playable_creative_id
                    _save_upload_state(game, campaign_id, upload_state)
                    
                except Exception as e:
                    errors.append(f"Playable creation failed: {e}")
                    playable_creative_id = None

        if not playable_creative_id and existing_playable_id:
            playable_creative_id = str(existing_playable_id)
            upload_state["playable_creative"] = playable_creative_id
            _save_upload_state(game, campaign_id, upload_state)

    # Validate Playable
    if playable_creative_id:
        try:
            p_details = _unity_get_creative(org_id=org_id, title_id=title_id, creative_id=playable_creative_id)
            p_type = (p_details.get("type") or "").lower()
            
            if "playable" not in p_type and "cpe" not in p_type:
                error_msg = f"CRITICAL: Playable ID ({playable_creative_id}) is type '{p_type}'. Must be 'playable'."
                errors.append(error_msg)
                return {"game": game, "campaign_id": campaign_id, "errors": errors, "creative_ids": upload_state["completed_packs"]}
        except Exception as e:
            errors.append(f"Could not validate Playable ID: {e}")
            return {"game": game, "campaign_id": campaign_id, "errors": errors, "creative_ids": upload_state["completed_packs"]}
    else:
        errors.append("No Playable End Card selected.")
        return {"game": game, "campaign_id": campaign_id, "errors": errors, "creative_ids": upload_state["completed_packs"]}

    # ========================================
    # 2. VIDEO PAIRING
    # ========================================
    subjects: dict[str, list[dict]] = {}
    for v in videos or []:
        n = v.get("name") or ""
        if "playable" in n.lower():
            continue
        base = n.split("_")[0]
        subjects.setdefault(base, []).append(v)

    total_pairs = len(subjects)
    upload_state["total_expected"] = total_pairs
    _save_upload_state(game, campaign_id, upload_state)
    
    if total_pairs == 0:
        st.warning("No video pairs found to upload.")
        return {
            "game": game,
            "campaign_id": campaign_id,
            "errors": errors,
            "creative_ids": upload_state["completed_packs"]
        }

    processed_count = 0
    progress_bar = st.progress(0, text=f"Starting upload... (0/{total_pairs})")
    
    # Status container for real-time updates
    status_container = st.empty()

    # ========================================
    # 3. PROCESSING LOOP
    # ========================================
    for base, items in subjects.items():
        portrait = next((x for x in items if "1080x1920" in (x.get("name") or "")), None)
        landscape = next((x for x in items if "1920x1080" in (x.get("name") or "")), None)

        if not portrait or not landscape:
            errors.append(f"{base}: Missing Portrait or Landscape video.")
            processed_count += 1
            progress_bar.progress(
                int(processed_count / total_pairs * 100),
                text=f"❌ Skipped {base} (Missing videos) - {processed_count}/{total_pairs}"
            )
            continue
        
        # Generate pack name
        clean_base = base.replace("_", "")
        raw_p_name = playable_name if playable_name else settings.get("existing_playable_label", "").split(" ")[0]
        clean_p = pathlib.Path(raw_p_name).stem.replace("_unityads", "").replace("_", "")
        final_pack_name = f"{clean_base}_{clean_p}"
        
        # Check if pack already exists
        if final_pack_name in upload_state["creative_packs"] and upload_state["creative_packs"][final_pack_name]:
            pack_id = upload_state["creative_packs"][final_pack_name]
            if pack_id not in upload_state["completed_packs"]:
                upload_state["completed_packs"].append(pack_id)
                _save_upload_state(game, campaign_id, upload_state)
            
            processed_count += 1
            progress_bar.progress(
                int(processed_count / total_pairs * 100),
                text=f"✅ Already uploaded: {base} - {processed_count}/{total_pairs}"
            )
            status_container.success(f"✅ Skipped (already exists): {final_pack_name}")
            continue

        try:
            progress_bar.progress(
                int(processed_count / total_pairs * 100),
                text=f"⬆️ Uploading {base} ({processed_count + 1}/{total_pairs})..."
            )
            
            # Upload portrait video (check if exists first)
            p_id = upload_state["video_creatives"].get(portrait["name"])
            if not p_id:
                p_id = _check_existing_creative(org_id, title_id, portrait["name"])
                
            if not p_id:
                status_container.info(f"⬆️ Uploading portrait: {portrait['name']}")
                p_id = _unity_create_video_creative(
                    org_id=org_id, 
                    title_id=title_id, 
                    video_path=portrait["path"], 
                    name=portrait["name"]
                )
                upload_state["video_creatives"][portrait["name"]] = p_id
                _save_upload_state(game, campaign_id, upload_state)
                time.sleep(1)
            else:
                status_container.success(f"✅ Found existing: {portrait['name']}")
            
            # Upload landscape video (check if exists first)
            l_id = upload_state["video_creatives"].get(landscape["name"])
            if not l_id:
                l_id = _check_existing_creative(org_id, title_id, landscape["name"])
                
            if not l_id:
                status_container.info(f"⬆️ Uploading landscape: {landscape['name']}")
                l_id = _unity_create_video_creative(
                    org_id=org_id, 
                    title_id=title_id, 
                    video_path=landscape["path"], 
                    name=landscape["name"]
                )
                upload_state["video_creatives"][landscape["name"]] = l_id
                _save_upload_state(game, campaign_id, upload_state)
                time.sleep(1)
            else:
                status_container.success(f"✅ Found existing: {landscape['name']}")

            pack_creatives = [p_id, l_id, playable_creative_id]
            
            # Check if pack already exists
            pack_id = _check_existing_pack(org_id, title_id, final_pack_name)
            
            if not pack_id:
                status_container.info(f"📦 Creating pack: {final_pack_name}")
                pack_id = _unity_create_creative_pack(
                    org_id=org_id,
                    title_id=title_id,
                    pack_name=final_pack_name,
                    creative_ids=pack_creatives,
                    pack_type="video+playable"
                )
            else:
                status_container.success(f"✅ Found existing pack: {final_pack_name}")
            
            upload_state["creative_packs"][final_pack_name] = pack_id
            upload_state["completed_packs"].append(pack_id)
            _save_upload_state(game, campaign_id, upload_state)
            
            status_container.success(f"✅ Completed: {final_pack_name}")
            time.sleep(0.5)

        except Exception as e:
            msg = str(e)
            if "Quota Exceeded" in msg or "429" in msg:
                errors.append(f"⚠️ Rate limit reached at {base}. Progress saved - you can retry!")
                status_container.error(
                    f"⚠️ **Rate Limit Reached**\n\n"
                    f"Progress saved: {len(upload_state['completed_packs'])}/{total_pairs} packs created.\n"
                    f"Click '크리에이티브/팩 생성' again to resume from where we left off."
                )
                break
            
            logger.exception(f"Unity pack creation failed for {base}")
            errors.append(f"{base}: {msg}")
            status_container.error(f"❌ Failed: {base} - {msg}")

        finally:
            processed_count += 1
            pct = int(processed_count / total_pairs * 100)
            completed = len(upload_state["completed_packs"])
            progress_bar.progress(
                pct, 
                text=f"Progress: {completed}/{total_pairs} packs created"
            )

    progress_bar.empty()
    status_container.empty()
    
    # Final summary
    total_created = len(upload_state["completed_packs"])
    
    if total_created == total_pairs:
        st.success(
            f"🎉 **Upload Complete!**\n\n"
            f"Successfully created **{total_created}/{total_pairs}** creative packs."
        )
        # Clear state on successful completion
        _clear_upload_state(game, campaign_id)
    elif total_created > 0:
        st.warning(
            f"⚠️ **Partial Upload**\n\n"
            f"Created **{total_created}/{total_pairs}** creative packs.\n"
            f"Click '크리에이티브/팩 생성' again to continue uploading remaining packs."
        )
    
    return {
        "game": game,
        "campaign_id": campaign_id,
        "start_iso": start_iso,
        "creative_ids": upload_state["completed_packs"],
        "errors": errors,
        "removed_ids": [],
        "total_created": total_created,
        "total_expected": total_pairs
    }

def apply_unity_creative_packs_to_campaign(*, game: str, creative_pack_ids: List[str], settings: Dict[str, Any]) -> Dict[str, Any]:
    if not creative_pack_ids:
        raise RuntimeError("No creative pack IDs to apply.")

    title_id = (settings.get("title_id") or "").strip() or str(UNITY_GAME_IDS.get(game, ""))
    campaign_id = (settings.get("campaign_id") or "").strip()
    org_id = (settings.get("org_id") or "").strip() or UNITY_ORG_ID_DEFAULT

    if not all([title_id, campaign_id, org_id]):
         raise RuntimeError("Unity settings missing for apply step.")

    removed_ids: List[str] = []
    assigned_packs: List[str] = []
    errors: List[str] = []

    # --- PROGRESS BAR FOR APPLY STEP ---
    progress_bar = st.progress(0, text="Fetching existing assignments...")

    # 1. Unassign existing (Loop to handle pagination)
    max_loops = 20 # Safety limit
    loop_count = 0
    
    while loop_count < max_loops:
        try:
            assigned = _unity_list_assigned_creative_packs(org_id=org_id, title_id=title_id, campaign_id=campaign_id)
            if not assigned:
                break
                
            total_unassign = len(assigned)
            loop_count += 1
            
            for idx, item in enumerate(assigned):
                assigned_id = item.get("id") or item.get("assignedCreativePackId")
                if assigned_id:
                    # Update progress bar
                    progress_bar.progress(
                        int((idx + 1) / max(total_unassign, 1) * 50), 
                        text=f"Unassigning batch {loop_count}: {idx + 1}/{total_unassign}..."
                    )
                    
                    try:
                        _unity_unassign_with_retry(  # 새 함수 사용
                            org_id=org_id,
                            title_id=title_id,
                            campaign_id=campaign_id,
                            assigned_creative_pack_id=str(assigned_id)
                        )
                        removed_ids.append(str(assigned_id))
                        time.sleep(1.0)  # 0.2 → 1.0으로 증가
                    except Exception as e:
                        errors.append(f"Unassign error {assigned_id}: {e}")
            
            # Short sleep between pages
            time.sleep(1)
            
        except Exception as e:
            errors.append(f"List assigned error: {e}")
            break

    # 2. Assign new
    total_assign = len(creative_pack_ids)
    count_a = 0
    
    for pack_id in creative_pack_ids:
        count_a += 1
        pct = 50 + int(count_a / max(total_assign, 1) * 50)
        progress_bar.progress(pct, text=f"Assigning new packs {count_a}/{total_assign}...")
        
        try:
            _unity_assign_creative_pack(org_id=org_id, title_id=title_id, campaign_id=campaign_id, creative_pack_id=str(pack_id))
            assigned_packs.append(str(pack_id))
            time.sleep(0.5) 
        except Exception as e:
            errors.append(f"Assign error {pack_id}: {e}")

    progress_bar.empty()

    return {
        "game": game,
        "campaign_id": campaign_id,
        "assigned_packs": assigned_packs,
        "removed_assignments": removed_ids,
        "errors": errors,
    }