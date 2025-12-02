"""Marketer-side Facebook helpers for Creative 자동 업로드.

Overrides specific UI/logic from facebook_ads.py for the 'Marketer' mode:
1. Simplified Settings UI (Campaign -> Ad Set -> Creative Type).
2. Uses the selected Ad Set ID directly.
"""

from __future__ import annotations

import streamlit as st
from facebook_business.adobjects.adaccount import AdAccount
from facebook_business.adobjects.campaign import Campaign
from facebook_business.adobjects.adset import AdSet

# Import everything from the base module
from facebook_ads import (
    FB_GAME_MAPPING,
    GAME_DEFAULTS,
    OPT_GOAL_LABEL_TO_API,
    init_fb_from_secrets,
    validate_page_binding,
    _plan_upload,
    build_targeting_from_settings,
    create_creativetest_adset,
    upload_videos_create_ads,
    sanitize_store_url,
    next_sat_0900_kst,
    init_fb_game_defaults,
)

# -------------------------------------------------------------------------
# 1. Cached Data Fetchers
# -------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def fetch_active_campaigns_cached(account_id: str) -> list[dict]:
    """Fetch ACTIVE campaigns for the given ad account."""
    try:
        account = init_fb_from_secrets(account_id)
        campaigns = account.get_campaigns(
            fields=[Campaign.Field.name, Campaign.Field.id],
            params={"effective_status": ["ACTIVE"], "limit": 100}
        )
        return [{"id": c["id"], "name": c["name"]} for c in campaigns]
    except Exception as e:
        print(f"Error fetching campaigns for {account_id}: {e}")
        return []

@st.cache_data(ttl=300, show_spinner=False)
def fetch_active_adsets_cached(account_id: str, campaign_id: str) -> list[dict]:
    """Fetch ACTIVE ad sets for a specific campaign."""
    try:
        # Re-init not strictly necessary if global session exists, but safe
        # We assume init_fb_from_secrets was called in fetch_active_campaigns
        from facebook_business.adobjects.campaign import Campaign
        camp = Campaign(campaign_id)
        adsets = camp.get_ad_sets(
            fields=[AdSet.Field.name, AdSet.Field.id],
            params={"effective_status": ["ACTIVE"], "limit": 100}
        )
        return [{"id": a["id"], "name": a["name"]} for a in adsets]
    except Exception as e:
        print(f"Error fetching ad sets for campaign {campaign_id}: {e}")
        return []

# -------------------------------------------------------------------------
# 2. Overridden UI: Render Settings
# -------------------------------------------------------------------------
def render_facebook_settings_panel(container, game: str, idx: int) -> None:
    if "settings" not in st.session_state:
        st.session_state.settings = {}

    cur = st.session_state.settings.get(game, {})

    # -- Apply hidden defaults --
    hidden_defaults = {
        "suffix_number": 1,
        "app_store": "Google Play 스토어",
        "opt_goal_label": "앱 설치수 극대화",
        "budget_per_video_usd": 10,
        "start_iso": next_sat_0900_kst(),
        "country": "US",
        "age_min": 18,
        "os_choice": "Android only",
        "ad_name_mode": "Use video filename",
        "add_launch_date": False,
        "creative_type": "단일 이미지/영상", # Default
    }
    for k, v in hidden_defaults.items():
        if k not in cur:
            cur[k] = v

    st.session_state.settings[game] = cur

    with container:
        st.markdown(f"#### {game} Facebook Settings (Marketer)")

        if game not in FB_GAME_MAPPING:
            st.error(f"Configuration missing for {game}")
            return

        cfg = FB_GAME_MAPPING[game]
        account_id = cfg["account_id"]
        default_camp_id = cfg.get("campaign_id", "")

        # -------------------------------------------------------
        # STEP 1: CAMPAIGN SELECTION
        # -------------------------------------------------------
        campaigns = fetch_active_campaigns_cached(account_id)
        
        selected_camp_id = None
        if not campaigns:
            st.warning("활성화된 캠페인이 없습니다. 기본 설정을 로드합니다.")
            selected_camp_id = default_camp_id
        else:
            camp_map = {c["id"]: c["name"] for c in campaigns}
            if default_camp_id and default_camp_id not in camp_map:
                camp_map[default_camp_id] = "(Default Config Campaign)"

            options = sorted(camp_map.items(), key=lambda x: x[1])
            option_labels = [f"{name} ({cid})" for cid, name in options]
            option_ids = [cid for cid, name in options]

            current_val = cur.get("campaign_id") or default_camp_id
            try:
                sel_idx = option_ids.index(current_val)
            except ValueError:
                sel_idx = 0

            sel_label = st.selectbox(
                "캠페인 선택",
                options=option_labels,
                index=sel_idx,
                key=f"mk_camp_sel_{idx}"
            )
            selected_camp_id = option_ids[option_labels.index(sel_label)]

        cur["campaign_id"] = selected_camp_id

        # -------------------------------------------------------
        # STEP 2: AD SET SELECTION
        # -------------------------------------------------------
        selected_adset_id = None
        if selected_camp_id:
            adsets = fetch_active_adsets_cached(account_id, selected_camp_id)
            
            if not adsets:
                st.warning(f"선택한 캠페인({selected_camp_id})에 활성화된 광고 세트가 없습니다.")
            else:
                adset_map = {a["id"]: a["name"] for a in adsets}
                as_options = sorted(adset_map.items(), key=lambda x: x[1])
                as_labels = [f"{name} ({aid})" for aid, name in as_options]
                as_ids = [aid for aid, name in as_options]

                prev_adset = cur.get("adset_id")
                try:
                    as_idx = as_ids.index(prev_adset) if prev_adset in as_ids else 0
                except ValueError:
                    as_idx = 0

                sel_as_label = st.selectbox(
                    "광고 세트 선택",
                    options=as_labels,
                    index=as_idx,
                    key=f"mk_adset_sel_{idx}",
                )
                selected_adset_id = as_ids[as_labels.index(sel_as_label)]

        cur["adset_id"] = selected_adset_id

        # -------------------------------------------------------
        # STEP 3: CREATIVE TYPE SELECTION
        # -------------------------------------------------------
        # -------------------------------------------------------
        # STEP 3: CREATIVE TYPE SELECTION
        # -------------------------------------------------------
        if selected_adset_id:
            creative_type = st.selectbox(
                "Creative Type",
                ["단일 이미지/영상", "다이나믹"],
                index=0 if cur.get("creative_type") == "단일 이미지/영상" else 1,
                key=f"mk_ctype_{idx}",
                help="단일: 각 영상을 개별 광고로 생성 / 다이나믹: 여러 소재를 하나의 Dynamic Creative로 묶음"
            )
            cur["creative_type"] = creative_type

            # If Dynamic is selected, show Aspect Ratio selection
            if creative_type == "다이나믹":
                dco_aspect = st.selectbox(
                    "소재 비율 (Aspect Ratio)",
                    ["세로 (9:16)", "가로 (16:9)", "정방향 (1:1)"],
                    index=0, 
                    key=f"mk_dco_ratio_{idx}"
                )
                cur["dco_aspect_ratio"] = dco_aspect

        st.session_state.settings[game] = cur

        # Info Box
        if selected_camp_id and selected_adset_id:
            c_type_str = cur.get("creative_type", "단일 이미지/영상")
            st.success(f"Target: `{selected_adset_id}`\nType: **{c_type_str}**")
        elif selected_camp_id:
             st.info("광고 세트를 선택해주세요.")

# -------------------------------------------------------------------------
# 3. Overridden Logic: Upload
# -------------------------------------------------------------------------
def upload_to_facebook(
    game_name: str,
    uploaded_files: list,
    settings: dict,
    *,
    simulate: bool = False,
) -> dict:
    """
    Marketer version:
    - Uses settings['campaign_id'] and settings['adset_id'].
    - Respects settings['creative_type'].
    - SKIPS Ad Set creation.
    """
    if game_name not in FB_GAME_MAPPING:
        raise ValueError(f"No FB mapping configured for game: {game_name}")

    cfg = FB_GAME_MAPPING[game_name]
    account = init_fb_from_secrets(cfg["account_id"])

    # 1. Validation
    page_id_key = cfg.get("page_id_key")
    if not page_id_key or page_id_key not in st.secrets:
        raise RuntimeError(f"Missing {page_id_key!r} in st.secrets")
    page_id = st.secrets[page_id_key]
    validate_page_binding(account, page_id)

    # 2. Grab Target IDs
    target_campaign_id = settings.get("campaign_id")
    target_adset_id = settings.get("adset_id")
    creative_type = settings.get("creative_type", "단일 이미지/영상")

    if not target_campaign_id:
        raise RuntimeError("캠페인이 선택되지 않았습니다.")
    if not target_adset_id:
        raise RuntimeError("광고 세트가 선택되지 않았습니다.")

    # 3. Plan
    plan = {
        "campaign_id": target_campaign_id,
        "adset_id": target_adset_id,
        "adset_name": "(Existing Ad Set)",
        "page_id": str(page_id),
        "n_videos": len(uploaded_files),
        "ad_names": [getattr(u, "name", "video") for u in uploaded_files],
        "creative_type": creative_type
    }
    
    if simulate:
        return plan

    # 4. Upload Logic
    ad_name_prefix = (
        settings.get("ad_name_prefix") if settings.get("ad_name_mode") == "Prefix + filename" else None
    )
    store_url = (settings.get("store_url") or "").strip()

    if creative_type == "다이나믹":
        # Placeholder for Future Logic:
        # Dynamic Creative (DCO) usually requires uploading all videos, then creating 
        # ONE AdCreative with object_story_spec.template_data containing multiple assets.
        # Currently, upload_videos_create_ads creates 1 Ad per 1 Video.
        st.warning("⚠️ '다이나믹' 모드는 현재 '단일 이미지/영상' 로직으로 처리됩니다. (추후 구현 예정)")
        
        # Fallback to standard upload for now so it doesn't crash
        upload_videos_create_ads(
            account=account,
            page_id=str(page_id),
            adset_id=target_adset_id,
            uploaded_files=uploaded_files,
            ad_name_prefix=ad_name_prefix,
            store_url=store_url,
            try_instagram=True,
        )
    else:
        # Standard: 1 Video -> 1 Ad
        upload_videos_create_ads(
            account=account,
            page_id=str(page_id),
            adset_id=target_adset_id,
            uploaded_files=uploaded_files,
            ad_name_prefix=ad_name_prefix,
            store_url=store_url,
            try_instagram=True,
        )

    return plan