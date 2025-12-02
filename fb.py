"""Marketer-side Facebook helpers for Creative 자동 업로드.

Overrides specific UI/logic from facebook_ads.py for the 'Marketer' mode:
1. Simplified Settings UI (Campaign -> Ad Set -> Creative Type).
2. Uses the selected Ad Set ID directly.
3. Auto-optimizes ad set (cleans up low performers).
4. Clones settings (headline/text) from existing ads.
"""
from __future__ import annotations

import streamlit as st
import logging
import time
import pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import FB SDK objects
from facebook_business.adobjects.adaccount import AdAccount
from facebook_business.adobjects.campaign import Campaign
from facebook_business.adobjects.adset import AdSet
from facebook_business.adobjects.adcreative import AdCreative
from facebook_business.adobjects.ad import Ad
from facebook_business.exceptions import FacebookRequestError

# Import everything from the base module
import facebook_ads as fb_ops
from facebook_ads import (
    FB_GAME_MAPPING,
    GAME_DEFAULTS,
    OPT_GOAL_LABEL_TO_API,
    init_fb_from_secrets,
    validate_page_binding,
    _plan_upload,
    build_targeting_from_settings,
    create_creativetest_adset,
    sanitize_store_url,
    next_sat_0900_kst,
    init_fb_game_defaults,
    make_ad_name,
)

logger = logging.getLogger(__name__)

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
        from facebook_business.adobjects.campaign import Campaign
        from facebook_business.adobjects.adset import AdSet
        
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
# 2. Optimization Logic (Cleanup)
# -------------------------------------------------------------------------
def cleanup_low_performing_ads(account, adset_id: str, new_files_count: int) -> None:
    """
    Ensures the Ad Set has enough slots (Max 50) for new files.
    """
    from facebook_business.adobjects.ad import Ad
    from datetime import datetime
    
    MAX_ADS = 50
    
    # Fetch current active/paused ads
    existing_ads = account.get_ads(
        params={
            "filtering": [
                {"field": "adset.id", "operator": "EQUAL", "value": adset_id},
                {"field": "effective_status", "operator": "IN", "value": ["ACTIVE", "PAUSED", "PENDING_REVIEW", "DISAPPROVED", "WITH_ISSUES", "ADSET_PAUSED", "CAMPAIGN_PAUSED"]}
            ],
            "limit": 100,
        },
        fields=[Ad.Field.id, Ad.Field.name, Ad.Field.created_time]
    )
    
    current_count = len(existing_ads)
    slots_needed = (current_count + new_files_count) - MAX_ADS
    
    if slots_needed <= 0:
        return  # Plenty of space

    st.info(f"Ad Set Full ({current_count}/50). Attempting to free {slots_needed} slots...")

    now = datetime.now()
    protected_ads = set()
    candidates = []

    for ad in existing_ads:
        try:
            c_time_str = ad["created_time"].replace("+0000", "")
            c_time = datetime.fromisoformat(c_time_str)
            age_days = (now - c_time).days
            if age_days <= 7:
                protected_ads.add(ad["id"])
            else:
                candidates.append(ad)
        except Exception:
            protected_ads.add(ad["id"])

    if not candidates:
        raise RuntimeError("크리에이티브 최적화가 필요합니다 (모든 광고가 7일 이내 생성됨).")

    def get_spend_map(preset):
        insights = account.get_insights(
            params={
                "level": "ad",
                "filtering": [{"field": "adset.id", "operator": "EQUAL", "value": adset_id}],
                "date_preset": preset,
            },
            fields=["spend", "ad_id"]
        )
        return {x["ad_id"]: float(x.get("spend", 0)) for x in insights}

    # Phase 1: Spend < $1 in last 14 days
    spend_map_14d = get_spend_map("last_14d")
    candidates_14d_check = []
    for ad in candidates:
        spend = spend_map_14d.get(ad["id"], 0.0)
        candidates_14d_check.append((spend, ad))
    
    candidates_14d_check.sort(key=lambda x: x[0])

    deleted_count = 0
    remaining_candidates = [] 

    for spend, ad in candidates_14d_check:
        if slots_needed <= 0: break
        if spend < 1.0:
            try:
                Ad(ad["id"]).api_update(params={"status": "ARCHIVED"})
                st.write(f"🗑️ Cleaned up (14d < $1): {ad['name']} (${spend})")
                slots_needed -= 1
                deleted_count += 1
            except Exception as e:
                print(f"Failed to archive {ad['id']}: {e}")
        else:
            remaining_candidates.append(ad)

    if slots_needed <= 0:
        st.success(f"Space cleared! Removed {deleted_count} ads.")
        return

    # Phase 2: Spend < $1 in last 7 days
    spend_map_7d = get_spend_map("last_7d")
    candidates_7d_check = []
    for ad in remaining_candidates:
        spend = spend_map_7d.get(ad["id"], 0.0)
        candidates_7d_check.append((spend, ad))
    candidates_7d_check.sort(key=lambda x: x[0])

    for spend, ad in candidates_7d_check:
        if slots_needed <= 0: break
        if spend < 1.0:
            try:
                Ad(ad["id"]).api_update(params={"status": "ARCHIVED"})
                st.write(f"🗑️ Cleaned up (7d < $1): {ad['name']} (${spend})")
                slots_needed -= 1
                deleted_count += 1
            except Exception: pass

    if slots_needed > 0:
        raise RuntimeError("크리에이티브 최적화가 필요합니다 (삭제할 수 있는 저효율 소재가 부족합니다).")
    else:
        st.success(f"Space cleared! Removed {deleted_count} ads.")

# -------------------------------------------------------------------------
# 3. Template Fetcher
# -------------------------------------------------------------------------
def fetch_reference_creative_data(account, adset_id: str) -> dict:
    """Finds the most recent active ad and extracts its text/headline/CTA."""
    try:
        ads = account.get_ads(
            params={
                "filtering": [
                    {"field": "adset.id", "operator": "EQUAL", "value": adset_id},
                    {"field": "effective_status", "operator": "IN", "value": ["ACTIVE", "PAUSED"]}
                ],
                "limit": 5,
                "fields": [Ad.Field.creative, Ad.Field.created_time]
            }
        )
        if not ads: return {}
        ads.sort(key=lambda x: x.get("created_time", ""), reverse=True)
        ref_ad = ads[0]
        creative_id = ref_ad.get("creative", {}).get("id")
        if not creative_id: return {}

        creative = AdCreative(creative_id).api_get(fields=["object_story_spec", "asset_feed_spec"])
        data = {"message": None, "headline": None, "call_to_action": None}

        # Single Video Spec
        spec = creative.get("object_story_spec")
        if spec:
            video_data = spec.get("video_data")
            if video_data:
                data["message"] = video_data.get("message")
                data["headline"] = video_data.get("title")
                data["call_to_action"] = video_data.get("call_to_action")
                return data

        # Dynamic Spec
        feed = creative.get("asset_feed_spec")
        if feed:
            if feed.get("bodies"): data["message"] = feed["bodies"][0].get("text")
            if feed.get("titles"): data["headline"] = feed["titles"][0].get("text")
            if feed.get("call_to_action_types"): data["call_to_action"] = {"type": feed["call_to_action_types"][0]}
            
        return data
    except Exception as e:
        logger.warning(f"Failed to fetch reference creative: {e}")
        return {}

# -------------------------------------------------------------------------
# 4. Settings UI (This was missing!)
# -------------------------------------------------------------------------
def render_facebook_settings_panel(container, game: str, idx: int) -> None:
    if "settings" not in st.session_state:
        st.session_state.settings = {}

    cur = st.session_state.settings.get(game, {})
    
    # Defaults
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
        "creative_type": "단일 이미지/영상",
    }
    for k, v in hidden_defaults.items():
        if k not in cur: cur[k] = v

    st.session_state.settings[game] = cur

    with container:
        st.markdown(f"#### {game} Facebook Settings (Marketer)")

        if game not in FB_GAME_MAPPING:
            st.error(f"Configuration missing for {game}")
            return

        cfg = FB_GAME_MAPPING[game]
        account_id = cfg["account_id"]
        default_camp_id = cfg.get("campaign_id", "")

        # 1. Campaign
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
            try: sel_idx = option_ids.index(current_val)
            except ValueError: sel_idx = 0

            sel_label = st.selectbox("캠페인 선택", options=option_labels, index=sel_idx, key=f"mk_camp_sel_{idx}")
            selected_camp_id = option_ids[option_labels.index(sel_label)]

        cur["campaign_id"] = selected_camp_id

        # 2. Ad Set
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
                try: as_idx = as_ids.index(prev_adset) if prev_adset in as_ids else 0
                except ValueError: as_idx = 0

                sel_as_label = st.selectbox("광고 세트 선택", options=as_labels, index=as_idx, key=f"mk_adset_sel_{idx}")
                selected_adset_id = as_ids[as_labels.index(sel_as_label)]

        cur["adset_id"] = selected_adset_id

        # 3. Creative Type
        if selected_adset_id:
            creative_type = st.selectbox(
                "Creative Type",
                ["단일 이미지/영상", "다이나믹"],
                index=0 if cur.get("creative_type") == "단일 이미지/영상" else 1,
                key=f"mk_ctype_{idx}",
                help="단일: 각 영상을 개별 광고로 생성 / 다이나믹: 여러 소재를 하나의 Dynamic Creative로 묶음"
            )
            cur["creative_type"] = creative_type

            if creative_type == "다이나믹":
                dco_aspect = st.selectbox(
                    "소재 비율 (Aspect Ratio)",
                    ["세로 (9:16)", "가로 (16:9)", "정방향 (1:1)"],
                    index=0, 
                    key=f"mk_dco_ratio_{idx}"
                )
                cur["dco_aspect_ratio"] = dco_aspect

        st.session_state.settings[game] = cur

        if selected_camp_id and selected_adset_id:
            c_type_str = cur.get("creative_type", "단일 이미지/영상")
            st.success(f"Target: `{selected_adset_id}`\nType: **{c_type_str}**")
        elif selected_camp_id:
             st.info("광고 세트를 선택해주세요.")

# -------------------------------------------------------------------------
# 5. Specialized Upload Function (Clones Settings)
# -------------------------------------------------------------------------
def upload_videos_create_ads_cloned(
    account: AdAccount,
    *,
    page_id: str,
    adset_id: str,
    uploaded_files: list,
    ad_name_prefix: str | None = None,
    store_url: str | None = None,
    try_instagram: bool = True,
    template_data: dict | None = None,
):
    """
    Modified version of upload_videos_create_ads.
    Applies 'template_data' (message, headline, CTA) to the new ads.
    """
    allowed = {".mp4", ".mpeg4"}
    def _is_video(u):
        n = fb_ops._fname_any(u) or "video.mp4"
        return pathlib.Path(n).suffix.lower() in allowed

    videos = fb_ops._dedupe_by_name([u for u in (uploaded_files or []) if _is_video(u)])
    
    # 1. Persist to temp
    persisted = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {}
        for u in videos:
            f = ex.submit(fb_ops._save_uploadedfile_tmp, u)
            futs[f] = fb_ops._fname_any(u)
        
        for fut, nm in futs.items():
            try:
                p = fut.result()
                persisted.append({"name": nm, "path": p})
            except Exception as e:
                st.error(f"File prep failed {nm}: {e}")

    # 2. Upload Videos
    uploads = []
    total = len(persisted)
    progress = st.progress(0, text="Uploading videos (Marketer Mode)...")
    
    for i, item in enumerate(persisted):
        try:
            # Simple upload for simplicity. Ideally copy resumable logic if needed.
            v = account.create_ad_video(params={
                "file": item["path"], 
                "content_category": "VIDEO_GAMING"
            })
            uploads.append({"name": item["name"], "video_id": v["id"]})
        except Exception as e:
            st.error(f"Upload failed for {item['name']}: {e}")
        progress.progress((i + 1) / total)
    
    progress.empty()
    time.sleep(5) # Wait for thumbnails

    # 3. Create Ads with Cloned Settings
    results = []
    api_errors = []
    
    ig_actor_id = None
    try:
        from facebook_business.adobjects.page import Page
        p = Page(page_id).api_get(fields=["instagram_business_account"])
        ig_actor_id = p.get("instagram_business_account", {}).get("id")
    except: pass

    template = template_data or {}
    
    def _create_ad_process(up):
        name = up["name"]
        vid = up["video_id"]
        
        try:
            from facebook_business.adobjects.advideo import AdVideo
            vinfo = AdVideo(vid).api_get(fields=["picture"])
            thumb = vinfo.get("picture")
            
            # Use template or fallback
            headline = template.get("headline") or name
            message = template.get("message") or ""
            cta = template.get("call_to_action")
            
            if cta and store_url and "value" in cta:
                 cta["value"]["link"] = store_url
            elif not cta and store_url:
                 cta = {"type": "INSTALL_MOBILE_APP", "value": {"link": store_url}}

            video_data = {
                "video_id": vid,
                "image_url": thumb,
                "title": headline,    
                "message": message,   
                "call_to_action": cta 
            }
            
            spec = {"page_id": page_id, "video_data": video_data}
            if try_instagram and ig_actor_id:
                spec["instagram_actor_id"] = ig_actor_id
                
            creative = account.create_ad_creative(params={
                "name": name,
                "object_story_spec": spec
            })
            
            ad_name = make_ad_name(name, ad_name_prefix)
            account.create_ad(params={
                "name": ad_name,
                "adset_id": adset_id,
                "creative": {"creative_id": creative["id"]},
                "status": "ACTIVE"
            })
            return True, name
        except Exception as e:
            return False, f"{name}: {e}"

    with ThreadPoolExecutor(max_workers=5) as ex:
        futs = [ex.submit(_create_ad_process, u) for u in uploads]
        for f in as_completed(futs):
            ok, res = f.result()
            if ok: results.append(res)
            else: api_errors.append(res)

    if api_errors:
        st.error("Some ads failed to create:\n" + "\n".join(api_errors))
        
    return results

# -------------------------------------------------------------------------
# 6. Main Entry Point
# -------------------------------------------------------------------------
def upload_to_facebook(
    game_name: str,
    uploaded_files: list,
    settings: dict,
    *,
    simulate: bool = False,
) -> dict:
    if game_name not in FB_GAME_MAPPING:
        raise ValueError(f"No FB mapping configured for game: {game_name}")

    cfg = FB_GAME_MAPPING[game_name]
    account = init_fb_from_secrets(cfg["account_id"])

    page_id_key = cfg.get("page_id_key")
    if not page_id_key or page_id_key not in st.secrets:
        raise RuntimeError(f"Missing {page_id_key!r} in st.secrets")
    page_id = st.secrets[page_id_key]
    validate_page_binding(account, page_id)

    target_campaign_id = settings.get("campaign_id")
    target_adset_id = settings.get("adset_id")
    creative_type = settings.get("creative_type", "단일 이미지/영상")

    if not target_campaign_id: raise RuntimeError("캠페인이 선택되지 않았습니다.")
    if not target_adset_id: raise RuntimeError("광고 세트가 선택되지 않았습니다.")

    plan = {
        "campaign_id": target_campaign_id,
        "adset_id": target_adset_id,
        "adset_name": "(Existing Ad Set)",
        "page_id": str(page_id),
        "n_videos": len(uploaded_files),
        "creative_type": creative_type
    }
    if simulate: return plan

    # 4. Cleanup Logic
    if creative_type == "단일 이미지/영상":
        try:
            cleanup_low_performing_ads(
                account=account, 
                adset_id=target_adset_id, 
                new_files_count=len(uploaded_files)
            )
        except RuntimeError as re:
            raise re
        except Exception as e:
            st.warning(f"Optimization check failed: {e}")

    # 5. Fetch Template
    template_data = fetch_reference_creative_data(account, target_adset_id)
    if template_data.get("headline") or template_data.get("message"):
        st.info(f"📋 Copying settings from existing ad:\n- Headline: {template_data.get('headline')}\n- Message: {template_data.get('message')[:30]}...")
    else:
        st.warning("⚠️ No existing active ads found to copy settings from. Using defaults.")

    ad_name_prefix = (
        settings.get("ad_name_prefix") if settings.get("ad_name_mode") == "Prefix + filename" else None
    )
    store_url = (settings.get("store_url") or "").strip()

    # 6. Upload
    if creative_type == "단일 이미지/영상":
        upload_videos_create_ads_cloned(
            account=account,
            page_id=str(page_id),
            adset_id=target_adset_id,
            uploaded_files=uploaded_files,
            ad_name_prefix=ad_name_prefix,
            store_url=store_url,
            template_data=template_data
        )
    
    elif creative_type == "다이나믹":
        st.warning("⚠️ '다이나믹' 모드 구현 중... 현재는 단일 이미지 로직(설정 복사 포함)으로 실행됩니다.")
        upload_videos_create_ads_cloned(
            account=account,
            page_id=str(page_id),
            adset_id=target_adset_id,
            uploaded_files=uploaded_files,
            ad_name_prefix=ad_name_prefix,
            store_url=store_url,
            template_data=template_data
        )

    return plan