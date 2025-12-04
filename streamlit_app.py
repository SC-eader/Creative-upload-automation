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
    from drive_import import import_drive_folder_videos  # old signature
    _DRIVE_IMPORT_SUPPORTS_PROGRESS = False

# game_manager must be imported before ops modules to ensure config loading
import game_manager 

# --------------------------------------------------------------------------
# OPERATIONS MODULES (Full Admin Access)
# --------------------------------------------------------------------------
import facebook_ads as fb_ops
import unity_ads as uni_ops

# --------------------------------------------------------------------------
# MARKETER MODULES (Simplified/Restricted Access)
# --------------------------------------------------------------------------
# Assumption: You have an fb.py for Facebook Marketer logic (similar to unity_marketer)
import fb as fb_marketer 
# UPDATED: Import the specific file we created
import uni as uni_marketer 


# ----- UI/Validation helpers --------------------------------------------------
try:
    MAX_UPLOAD_MB = int(st.get_option("server.maxUploadSize"))
except Exception:
    MAX_UPLOAD_MB = 200


def init_state():
    """Set up st.session_state containers for uploads and settings if missing."""
    if "uploads" not in st.session_state:
        st.session_state.uploads = {}
    if "settings" not in st.session_state:
        st.session_state.settings = {}


def init_remote_state():
    """Set up st.session_state container for Drive-imported videos per game if missing."""
    if "remote_videos" not in st.session_state:
        st.session_state.remote_videos = {}  # {game: [ {"name":..., "path":...}, ... ]}


def validate_count(files: List) -> tuple[bool, str]:
    """Check there is at least one .mp4/.mpeg4 file and no invalid types."""
    if not files:
        return False, "Please upload at least one video (.mp4 or .mpeg4)."

    allowed = {".mp4", ".mpeg4"}
    bad = []
    for u in files:
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


def _run_drive_import(folder_url_or_id: str, max_workers: int, on_progress=None):
    """Wrapper for Drive import (new parallel API or legacy)."""
    if _DRIVE_IMPORT_SUPPORTS_PROGRESS:
        return import_drive_folder_videos(folder_url_or_id, max_workers=max_workers, on_progress=on_progress)

    files = import_drive_folder_videos(folder_url_or_id)
    total = len(files)
    if on_progress:
        done = 0
        for f in files:
            done += 1
            on_progress(done, total, f.get("name", ""), None)
    return files


# ----- Streamlit base config + shared init ------------------------------------
st.set_page_config(
    page_title="Creative 자동 업로드",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_state()
init_remote_state()
fb_ops.init_fb_game_defaults()


# Sidebar: 페이지 선택 (운영 / 마케터)
with st.sidebar:
    st.markdown("### 페이지 선택")

    if "page" not in st.session_state:
        st.session_state["page"] = "Creative 자동 업로드"

    main_clicked = st.button(
        "운영 (Ops)",
        key="page_main_btn",
        use_container_width=True,
    )
    marketer_clicked = st.button(
        "마케터 (Marketer)",
        key="page_marketer_btn",
        use_container_width=True,
    )

    if main_clicked:
        st.session_state["page"] = "Creative 자동 업로드"
    if marketer_clicked:
        st.session_state["page"] = "Creative 자동 업로드 - 마케터"

    page = st.session_state["page"]
    st.caption(f"현재 페이지: **{page}**")


# ======================================================================
# 공통 메인 앱 렌더러 (운영 / 마케터 공용)
# ======================================================================

def render_main_app(title: str, fb_module, unity_module, is_marketer: bool = False) -> None:
    """Render the full Creative 자동 업로드 UI with the given page title and helper modules."""
    st.title(title)
    st.caption("게임별 크리에이티브를 다운받고, 설정에 따라 자동으로 업로드합니다.")

    # --- MARKETER ONLY: Add New Game Sidebar ---
    if is_marketer:
        with st.sidebar:
            st.divider() 
            with st.expander("➕ Add New Game", expanded=False):
                with st.form("add_game_form"):
                    st.caption("Add a new game configuration locally.")
                    new_game_name = st.text_input("Game Name (e.g. My New RPG)")
                    
                    st.markdown("**Facebook Details**")
                    new_fb_act = st.text_input("Ad Account ID", placeholder="act_12345678")
                    new_fb_page = st.text_input("Page ID", placeholder="1234567890")
                    
                    st.markdown("**Unity Details**")
                    new_unity_id = st.text_input("Unity Game ID (Optional)")
                    
                    submitted = st.form_submit_button("Save Game")
                    
                    if submitted:
                        if not new_game_name or not new_fb_act or not new_fb_page:
                            st.error("Name, Ad Account, and Page ID are required.")
                        else:
                            # --- VALIDATION STEP ---
                            status_box = st.status("Validating IDs with Meta...", expanded=True)
                            try:
                                # 1. Auth
                                fb_ops.init_fb_from_secrets()
                                from facebook_business.adobjects.adaccount import AdAccount
                                from facebook_business.adobjects.page import Page
                                
                                # 2. Validate Ad Account
                                status_box.write("Checking Ad Account...")
                                act = AdAccount(new_fb_act.strip())
                                act_info = act.api_get(fields=["name", "account_status"])
                                status_box.write(f"✅ Found Account: {act_info.get('name')}")

                                # 3. Validate Page
                                status_box.write("Checking Page Access...")
                                pg = Page(new_fb_page.strip())
                                pg_info = pg.api_get(fields=["name", "is_published"])
                                status_box.write(f"✅ Found Page: {pg_info.get('name')}")
                                
                                # 4. Success -> Save
                                status_box.update(label="Validation Successful!", state="complete")
                                
                                game_manager.save_new_game(
                                    new_game_name, new_fb_act, new_fb_page, new_unity_id
                                )
                                st.success(f"Saved **{new_game_name}** successfully!")
                                import time
                                time.sleep(1.5)
                                st.rerun()

                            except Exception as e:
                                status_box.update(label="Validation Failed", state="error")
                                st.error(f"❌ Invalid ID or Permission Error:\n{e}")

    # Load Games
    GAMES = game_manager.get_all_game_names(include_custom=is_marketer)
    
    if not GAMES:
        st.error("No games found.")
        return

    _tabs = st.tabs(GAMES)
    
    for i, game in enumerate(GAMES):
        with _tabs[i]:
            left_col, right_col = st.columns([2, 1], gap="large")

            # =========================
            # LEFT COLUMN: Inputs
            # =========================
            with left_col:
                left_card = st.container(border=True)
                with left_card:
                    st.subheader(game)

                    # --- Platform Selection ---
                    platform = st.radio(
                        "플랫폼 선택",
                        ["Facebook", "Unity Ads"],
                        index=0,
                        horizontal=True,
                        key=f"platform_{i}",
                    )

                    if platform == "Facebook":
                        st.markdown("### Facebook")
                    else:
                        st.markdown("### Unity Ads")

                    # --- Drive Import ---
                    st.markdown("**구글 드라이브에서 Creative Videos를 가져옵니다**")
                    drv_input = st.text_input(
                        "Drive folder URL or ID",
                        key=f"drive_folder_{i}",
                        placeholder="https://drive.google.com/drive/folders/<FOLDER_ID>",
                    )

                    with st.expander("Advanced import options", expanded=False):
                        workers = st.number_input(
                            "Parallel workers",
                            min_value=1, max_value=16, value=8,
                            key=f"drive_workers_{i}",
                        )

                    if st.button("드라이브에서 Creative 가져오기", key=f"drive_import_{i}"):
                        try:
                            overall = st.progress(0, text="0/0 • waiting…")
                            log_box = st.empty()
                            lines: List[str] = []
                            import time
                            last_flush = [0.0]

                            def _on_progress(done: int, total: int, name: str, err: str | None):
                                pct = int((done / max(total, 1)) * 100)
                                label = f"{done}/{total}"
                                if name: label += f" • {name}"
                                if err: lines.append(f"❌ {name}  —  {err}")
                                else: lines.append(f"✅ {name}")

                                now = time.time()
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
                                lst = st.session_state.remote_videos.get(game, [])
                                lst.extend(imported)
                                st.session_state.remote_videos[game] = lst
                                status.update(label=f"Drive import complete: {len(imported)} file(s)", state="complete")
                                if isinstance(imported, dict) and imported.get("errors"):
                                    st.warning("Some files failed:\n- " + "\n".join(imported["errors"]))

                            st.success(f"Imported {len(imported)} video(s).")
                        except Exception as e:
                            st.exception(e)
                            st.error("Could not import from folder.")

                    # --- List & Clear ---
                    remote_list = st.session_state.remote_videos.get(game, [])
                    st.caption("다운로드된 Creatives:")
                    if remote_list:
                        for it in remote_list[:50]: st.write("•", it["name"])
                        if len(remote_list) > 50: st.write(f"... 외 {len(remote_list) - 50}개")
                    else:
                        st.write("- (없음)")

                    if st.button("URL/Drive 영상만 초기화", key=f"clearurl_{i}"):
                        if remote_list:
                            st.session_state.remote_videos[game] = []
                            st.rerun()

                    # --- Action Buttons ---
                    if platform == "Facebook":
                        ok_msg_placeholder = st.empty()
                        btn_label = "Creative 업로드하기" if is_marketer else "Creative Test 업로드하기"
                        cont = st.button(btn_label, key=f"continue_{i}")
                        clr = st.button("전체 초기화", key=f"clear_{i}")
                    else:
                        unity_ok_placeholder = st.empty()
                        cont_unity_create = st.button("크리에이티브/팩 생성", key=f"unity_create_{i}")
                        cont_unity_apply = st.button("캠페인에 적용", key=f"unity_apply_{i}")
                        clr_unity = st.button("전체 초기화 (Unity용)", key=f"unity_clear_{i}")

            # =========================
            # RIGHT COLUMN: Settings
            # =========================
            if platform == "Facebook":
                with right_col:
                    fb_card = st.container(border=True)
                    # Pass the module (either fb_ops or fb_marketer)
                    fb_module.render_facebook_settings_panel(fb_card, game, i)

            elif platform == "Unity Ads":
                with right_col:
                    unity_card = st.container(border=True)
                    # Pass the module (either uni_ops or uni_marketer)
                    unity_module.render_unity_settings_panel(unity_card, game, i)

            # =========================
            # EXECUTION LOGIC
            # =========================
            
            # --- FACEBOOK ---
            if platform == "Facebook" and cont:
                remote_list = st.session_state.remote_videos.get(game, [])
                ok, msg = validate_count(remote_list)
                if not ok:
                    ok_msg_placeholder.error(msg)
                else:
                    try:
                        st.session_state.uploads[game] = remote_list
                        settings = st.session_state.settings.get(game, {})
                        
                        # Call upload on the active module
                        plan = fb_module.upload_to_facebook(game, remote_list, settings)
                        
                        if isinstance(plan, dict) and plan.get("adset_id"):
                            ok_msg_placeholder.success(msg + " Uploaded to Meta successfully.")
                        else:
                            ok_msg_placeholder.error("Meta upload failed. Check logs.")
                    except Exception as e:
                        import traceback
                        st.error("Meta upload failed.")
                        st.code(traceback.format_exc(), language="python")
            
            if platform == "Facebook" and clr:
                st.session_state.uploads.pop(game, None)
                st.session_state.remote_videos.pop(game, None)
                st.session_state.settings.pop(game, None)
                st.rerun()

            # --- UNITY ---
            if platform == "Unity Ads":
                # Get settings using the passed module (polymorphic)
                unity_settings = unity_module.get_unity_settings(game)

                if "unity_created_packs" not in st.session_state:
                    st.session_state.unity_created_packs = {}

                # 1. Create Packs
                if "cont_unity_create" in locals() and cont_unity_create:
                    remote_list = st.session_state.remote_videos.get(game, []) or []
                    ok, msg = validate_count(remote_list)
                    if not ok:
                        unity_ok_placeholder.error(msg)
                    else:
                        try:
                            # Polymorphic call to upload logic
                            summary = unity_module.upload_unity_creatives_to_campaign(
                                game=game,
                                videos=remote_list,
                                settings=unity_settings,
                            )
                            pack_ids = summary.get("creative_ids") or []
                            errors = summary.get("errors") or []
                            
                            st.session_state.unity_created_packs[game] = list(pack_ids)

                            if pack_ids:
                                unity_ok_placeholder.success(f"Unity Ads: Created {len(pack_ids)} packs.")
                            else:
                                unity_ok_placeholder.warning("Unity Ads: No packs created.")
                                
                            if errors:
                                st.error("Errors:\n" + "\n".join(errors))
                        except Exception as e:
                            import traceback
                            unity_ok_placeholder.error("Unity creation failed.")
                            st.code(traceback.format_exc(), language="python")

                # 2. Apply Packs
                if "cont_unity_apply" in locals() and cont_unity_apply:
                    pack_ids = st.session_state.unity_created_packs.get(game) or []
                    if not pack_ids:
                        unity_ok_placeholder.error("No created packs to apply.")
                    else:
                        try:
                            # Polymorphic call to apply logic
                            result = unity_module.apply_unity_creative_packs_to_campaign(
                                game=game,
                                creative_pack_ids=pack_ids,
                                settings=unity_settings,
                            )
                            assigned = result.get("assigned_packs") or []
                            errors = result.get("errors") or []
                            
                            if assigned:
                                unity_ok_placeholder.success(f"Assigned {len(assigned)} packs to campaign.")
                            else:
                                unity_ok_placeholder.warning("No packs assigned.")
                            
                            if errors:
                                st.error("Errors:\n" + "\n".join(errors))
                        except Exception as e:
                            import traceback
                            unity_ok_placeholder.error("Unity apply failed.")
                            st.code(traceback.format_exc(), language="python")

                if "clr_unity" in locals() and clr_unity:
                    st.session_state.uploads.pop(game, None)
                    st.session_state.remote_videos.pop(game, None)
                    st.session_state.unity_settings.pop(game, None)
                    if "unity_created_packs" in st.session_state:
                        st.session_state.unity_created_packs.pop(game, None)
                    st.rerun()

    # Summary table
    st.subheader("업로드 완료된 게임")
    if st.session_state.uploads:
        data = {"게임": [], "업로드 파일": []}
        for g, files in st.session_state.uploads.items():
            data["게임"].append(g)
            data["업로드 파일"].append(len(files))
        st.dataframe(data, hide_index=True)


# ======================================================================
# PAGE ROUTING
# ======================================================================
if page == "Creative 자동 업로드":
    # OPS MODE: Use full modules
    render_main_app("🎮 Creative 자동 업로드", fb_ops, uni_ops, is_marketer=False)
else:
    # MARKETER MODE: Use restricted/simplified modules
    render_main_app("🎮 Creative 자동 업로드 - 마케터", fb_marketer, uni_marketer, is_marketer=True)