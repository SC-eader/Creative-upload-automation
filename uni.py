"""Marketer-side Unity helpers for Creative 자동 업로드.

Overrides specific UI/logic from unity_ads.py for the 'Marketer' mode:
1. Simplified Settings UI (Campaign Selector).
2. Dynamic Campaign Fetching from Unity API.
3. AOS/iOS Selection Toggle.
"""

from __future__ import annotations

import streamlit as st
import logging

# Import base module to access constants and helpers
import unity_ads as uni_ops
import game_manager # Import game_manager to read raw config directly

from unity_ads import (
    UNITY_ORG_ID_DEFAULT,
    _unity_get,
    upload_unity_creatives_to_campaign,
    apply_unity_creative_packs_to_campaign,
    get_unity_settings,
    _ensure_unity_settings_state,
    _unity_list_playable_creatives
)

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# 1. Cached Campaign Fetcher
# -------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def fetch_unity_campaigns_cached(org_id: str, title_id: str) -> list[dict]:
    """Fetch campaigns for a specific Unity Game ID."""
    if not org_id or not title_id:
        return []
        
    path = f"organizations/{org_id}/apps/{title_id}/campaigns"
    try:
        resp = _unity_get(path)
        items = []
        if isinstance(resp, list): items = resp
        elif isinstance(resp, dict):
            if "results" in resp: items = resp["results"]
            elif "items" in resp: items = resp["items"]
            elif "data" in resp: items = resp["data"]
            
        campaigns = []
        for c in items:
            c_id = c.get("id") or c.get("campaignId")
            c_name = c.get("name") or "(No Name)"
            if c_id:
                campaigns.append({"id": c_id, "name": c_name})
        return campaigns
    except Exception as e:
        logger.error(f"Failed to fetch Unity campaigns: {e}")
        return []

# -------------------------------------------------------------------------
# 2. Overridden UI: Render Settings
# -------------------------------------------------------------------------
def render_unity_settings_panel(container, game: str, idx: int) -> None:
    """
    Marketer version of Unity Settings.
    - OS Toggle (AOS/iOS)
    - Campaign Dropdown
    - Playable Selection
    """
    _ensure_unity_settings_state()
    
    # 1. Get Config directly from Game Manager to see separate OS IDs
    try:
        raw_config = game_manager.get_game_config(game, "unity")
    except Exception:
        raw_config = {}

    android_id = raw_config.get("android_game_id", "")
    ios_id = raw_config.get("ios_game_id", "")
    
    # Fallback for legacy "game_id" if separate ones aren't found
    if not android_id and not ios_id:
        android_id = raw_config.get("game_id", "")

    org_id = raw_config.get("org_id") or UNITY_ORG_ID_DEFAULT

    if not android_id and not ios_id:
        with container:
            st.error(f"Missing Unity Game ID for {game}")
        return

    cur = st.session_state.unity_settings.get(game, {})

    with container:
        st.markdown(f"#### {game} Unity Settings")
        
        # --- OS SELECTION ---
        # Determine available options
        os_options = []
        if android_id: os_options.append("Android")
        if ios_id: os_options.append("iOS")
        
        if not os_options:
             os_options = ["Android"] # Default fallback
        
        selected_os = st.radio("Target OS", os_options, horizontal=True, key=f"os_sel_{idx}")
        
        # Pick the correct Title ID based on selection
        active_title_id = android_id if selected_os == "Android" else ios_id
        
        if not active_title_id:
            st.warning(f"No Game ID found for {selected_os}")
        else:
            # --- CAMPAIGN SELECTOR ---
            campaigns = fetch_unity_campaigns_cached(org_id, active_title_id)
            
            selected_campaign_id = ""
            
            if not campaigns:
                st.warning("캠페인 목록을 불러올 수 없습니다.")
                st.caption(f"Game ID: {active_title_id}")
            else:
                options = sorted(campaigns, key=lambda x: x["name"])
                labels = [f"{c['name']} ({c['id']})" for c in options]
                ids = [c['id'] for c in options]
                
                # Try to keep previous selection if valid
                current_val = cur.get("campaign_id", "")
                try:
                    sel_idx = ids.index(current_val)
                except ValueError:
                    sel_idx = 0
                    
                sel_label = st.selectbox(
                    "캠페인 선택",
                    options=labels,
                    index=sel_idx,
                    key=f"mk_uni_camp_{idx}"
                )
                selected_campaign_id = ids[labels.index(sel_label)]
                st.caption(f"ID: `{selected_campaign_id}`")

            # --- PLAYABLE SELECTION (Standard) ---
            st.markdown("#### Playable 선택")
            drive_playables = [
                v for v in (st.session_state.remote_videos.get(game, []) if "remote_videos" in st.session_state else [])
                if "playable" in (v.get("name") or "").lower()
            ]
            drive_options = [p["name"] for p in drive_playables]
            prev_drive = cur.get("selected_playable", "")
            
            sel_drive = st.selectbox(
                "Drive에서 가져온 플레이어블",
                options=["(선택 안 함)"] + drive_options,
                index=(drive_options.index(prev_drive) + 1) if prev_drive in drive_options else 0,
                key=f"mk_uni_drv_play_{idx}"
            )
            chosen_drive = sel_drive if sel_drive != "(선택 안 함)" else ""

            # Fetch Existing Playables
            existing_labels = ["(선택 안 함)"]
            existing_id_map = {}
            
            if org_id and active_title_id:
                try:
                    playables = _unity_list_playable_creatives(org_id=org_id, title_id=active_title_id)
                    for p in playables:
                        pid = str(p.get("id") or "")
                        pname = p.get("name") or "(no name)"
                        ptype = p.get("type", "")
                        if pid:
                            lbl = f"{pname} ({ptype}) [{pid}]"
                            existing_labels.append(lbl)
                            existing_id_map[lbl] = pid
                except Exception:
                    pass
                
            prev_exist_label = cur.get("existing_playable_label", "")
            try:
                ex_idx = existing_labels.index(prev_exist_label)
            except ValueError:
                ex_idx = 0
                
            sel_exist = st.selectbox(
                "Unity에 이미 있는 playable",
                options=existing_labels,
                index=ex_idx,
                key=f"mk_uni_exist_play_{idx}"
            )
            chosen_exist_id = existing_id_map.get(sel_exist, "")

            # --- SAVE SETTINGS ---
            # IMPORTANT: We save the *Active* title_id (AOS or iOS) into the state
            # so the upload function knows which one to use without needing to know logic.
            st.session_state.unity_settings[game] = {
                "title_id": active_title_id,
                "campaign_id": selected_campaign_id,
                "org_id": org_id,
                "selected_playable": chosen_drive,
                "existing_playable_id": chosen_exist_id,
                "existing_playable_label": sel_exist,
                "target_os": selected_os 
            }