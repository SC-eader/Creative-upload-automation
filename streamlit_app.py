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

# 운영용 모듈 (기존 facebook_ads / unity_ads)
import facebook_ads as fb_ops
import unity_ads as uni_ops

# 마케터용 모듈 (fb.py / uni.py – 처음엔 facebook_ads / unity_ads 를 그대로 re-export)
import fb as fb_marketer
import uni as uni_marketer

import game_manager
# ----- UI/Validation helpers --------------------------------------------------
try:
    MAX_UPLOAD_MB = int(st.get_option("server.maxUploadSize"))
except Exception:
    MAX_UPLOAD_MB = 200  # Streamlit default if option missing


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
    Wrapper for Drive import (new parallel API or legacy).

    Calls on_progress(done, total, name, error_or_None) if provided.
    """
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
    initial_sidebar_state="expanded",  # 사이드바 기본 확장
)

init_state()
init_remote_state()
fb_ops.init_fb_game_defaults()  # 운영 기본값 초기화 (마케터도 같은 state 공유)


# Sidebar: 페이지 선택 (운영 / 마케터)
with st.sidebar:
    st.markdown("### 페이지 선택")

    if "page" not in st.session_state:
        st.session_state["page"] = "Creative 자동 업로드"

    main_clicked = st.button(
        "운영",
        key="page_main_btn",
        use_container_width=True,
    )
    marketer_clicked = st.button(
        "마케터",
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
# 공통 메인 앱 렌더러 (운영 / 마케터 공용) – 모듈만 다르게 주입
# ======================================================================

def render_main_app(title: str, fb_module, unity_module, is_marketer: bool = False) -> None:
    """Render the full Creative 자동 업로드 UI with the given page title and helper modules."""
    st.title(title)
    st.caption("게임별 크리에이티브를 다운받고, 설정에 따라 자동으로 업로드합니다.")

    # --- MARKETER ONLY: Add New Game Sidebar ---
    if is_marketer:
        with st.sidebar:
            st.divider() # Visual separation
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
                                # Do NOT save if error occurs

    # Load Games: Marketers see ALL, Operations see DEFAULT only
    GAMES = game_manager.get_all_game_names(include_custom=is_marketer)
    
    # Render Tabs
    if not GAMES:
        st.error("No games found.")
        return

    _tabs = st.tabs(GAMES)
    

    for i, game in enumerate(GAMES):
        with _tabs[i]:
            # 전체 영역을 고정된 2열 레이아웃으로: 왼쪽(게임/Drive), 오른쪽(Settings)
            left_col, right_col = st.columns([2, 1], gap="large")

            # =========================
            # LEFT COLUMN
            # =========================
            with left_col:
                left_card = st.container(border=True)
                with left_card:
                    st.subheader(game)

                    # --- 플랫폼 선택: 게임 제목 바로 아래 ---
                    platform = st.radio(
                        "플랫폼 선택",
                        ["Facebook", "Unity Ads"],
                        index=0,
                        horizontal=True,
                        key=f"platform_{i}",
                    )

                    # 플랫폼별 섹션 헤더
                    if platform == "Facebook":
                        st.markdown("### Facebook")
                    else:
                        st.markdown("### Unity Ads")

                    # --- 공통: 구글 드라이브에서 Creative Videos 가져오기 (Facebook/Unity 공용) ---
                    st.markdown("**구글 드라이브에서 Creative Videos를 가져옵니다**")
                    drv_input = st.text_input(
                        "Drive folder URL or ID",
                        key=f"drive_folder_{i}",
                        placeholder="https://drive.google.com/drive/folders/<FOLDER_ID>",
                    )

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

                            import time
                            last_flush = [0.0]

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

                                status.update(
                                    label=f"Drive import complete: {len(imported)} file(s)",
                                    state="complete",
                                )
                                if isinstance(imported, dict) and imported.get("errors"):
                                    st.warning(
                                        "Some files failed:\n- "
                                        + "\n".join(imported["errors"])
                                    )

                            st.success(f"Imported {len(imported)} video(s) from the folder.")
                            if len(imported) < 1:
                                st.info("No eligible videos found. Check access, file types, or folder contents.")
                        except Exception as e:
                            st.exception(e)
                            st.error(
                                "Could not import from this folder. "
                                "Make sure your service account has access and the folder contains videos."
                            )

                    # --- 공통: 현재 다운로드된/저장된 리스트 + 초기화 ---
                    remote_list = st.session_state.remote_videos.get(game, [])

                    st.caption("다운로드된 Creatives:")
                    if remote_list:
                        for it in remote_list[:50]:
                            st.write("•", it["name"])
                        if len(remote_list) > 50:
                            st.write(f"... 외 {len(remote_list) - 50}개")
                    else:
                        st.write("- (현재 저장된 URL/Drive 영상 없음)")

                    if st.button("URL/Drive 영상만 초기화", key=f"clearurl_{i}"):
                        if remote_list:
                            st.session_state.remote_videos[game] = []
                            st.info("Cleared URL/Drive videos for this game.")
                            st.rerun()
                        else:
                            st.info("삭제할 URL/Drive 영상이 없습니다.")

                    # --- 플랫폼별 버튼들 ---
                    if platform == "Facebook":
                        ok_msg_placeholder = st.empty()
                        
                        # Change label based on mode
                        btn_label = "Creative 업로드하기" if is_marketer else "Creative Test 업로드하기"
                        
                        cont = st.button(btn_label, key=f"continue_{i}")
                        clr = st.button("전체 초기화", key=f"clear_{i}")
                    else:
                        unity_ok_placeholder = st.empty()

                        cont_unity_create = st.button(
                            "크리에이티브/팩 생성",
                            key=f"unity_create_{i}",
                            help="Drive에서 가져온 영상으로 Unity creative + creative packs를 만듭니다 (캠페인에는 아직 적용 안 함).",
                        )

                        cont_unity_apply = st.button(
                            "캠페인에 적용",
                            key=f"unity_apply_{i}",
                            help="방금 생성한 creative packs만 캠페인에 assign하고, 이전 iteration pack들은 unassign 합니다.",
                        )

                        clr_unity = st.button("전체 초기화 (Unity용)", key=f"unity_clear_{i}")

            # =========================
            # RIGHT COLUMN: Settings (플랫폼별)
            # =========================
            if platform == "Facebook":
                with right_col:
                    fb_card = st.container(border=True)
                    fb_module.render_facebook_settings_panel(fb_card, game, i)

            elif platform == "Unity Ads":
                with right_col:
                    unity_card = st.container(border=True)
                    unity_module.render_unity_settings_panel(unity_card, game, i)

            # --- Handle button actions after BOTH columns are drawn ---
            # FACEBOOK FLOW --------------------------------------------------
            if platform == "Facebook":
                if cont:
                    remote_list = st.session_state.remote_videos.get(game, [])
                    combined = remote_list

                    ok, msg = validate_count(combined)
                    if not ok:
                        ok_msg_placeholder.error(msg)
                    else:
                        try:
                            st.session_state.uploads[game] = combined
                            settings = st.session_state.settings.get(game, {})
                            plan = fb_module.upload_to_facebook(game, combined, settings)

                            def _render_summary(plan: dict, settings: dict, created: bool):
                                ...

                            if isinstance(plan, dict) and plan.get("adset_id"):
                                ok_msg_placeholder.success(
                                    msg + " Uploaded to Meta (ads created as ACTIVE, scheduled by start time)."
                                )
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

            # UNITY ADS FLOW --------------------------------------------------
            if platform == "Unity Ads":
                unity_settings = unity_module.get_unity_settings(game)

                if "unity_created_packs" not in st.session_state:
                    st.session_state.unity_created_packs = {}  # {game: [pack_id, ...]}

                # 1) CREATE creatives + packs
                if "cont_unity_create" in locals() and cont_unity_create:
                    remote_list = st.session_state.remote_videos.get(game, []) or []

                    ok, msg = validate_count(remote_list)
                    if not ok:
                        unity_ok_placeholder.error(msg)
                    else:
                        if not (
                            unity_settings.get("selected_playable")
                            or unity_settings.get("existing_playable_id")
                        ):
                            unity_ok_placeholder.warning(
                                "현재 선택된 playable이 없습니다. Unity creative pack은 "
                                "9:16 영상 1개 + 16:9 영상 1개 + 1개의 playable 조합이 권장됩니다."
                            )

                        try:
                            summary = unity_module.upload_unity_creatives_to_campaign(
                                game=game,
                                videos=remote_list,
                                settings=unity_settings,
                            )

                            pack_ids = summary.get("creative_ids") or []
                            errors = summary.get("errors") or []

                            st.session_state.unity_created_packs[game] = list(pack_ids)

                            n_packs = len(pack_ids)
                            if n_packs > 0:
                                unity_ok_placeholder.success(
                                    f"{msg} Unity Ads에 {n_packs}개 creative pack을 생성했습니다.\n"
                                    "이제 '캠페인에 적용' 버튼으로 해당 pack들을 캠페인에 assign 할 수 있습니다."
                                )
                            else:
                                unity_ok_placeholder.warning(
                                    "Unity Ads 호출은 성공했지만 생성된 creative pack ID가 없습니다. "
                                    "Unity 대시보드에서 실제 상태를 확인해 주세요."
                                )

                            if errors:
                                st.error(
                                    "캠페인 적용 중 일부 오류가 발생했습니다:\n"
                                    + "\n".join(f"- {e}" for e in errors[:20])
                                    + ("\n..." if len(errors) > 20 else "")
                                )

                        except Exception as e:
                            import traceback
                            st.exception(e)
                            tb = traceback.format_exc()
                            unity_ok_placeholder.error("Unity Ads 크리에이티브/팩 생성 실패. 아래 오류 로그를 확인하세요.")
                            st.code(tb, language="python")

                # 2) APPLY packs to campaign
                if "cont_unity_apply" in locals() and cont_unity_apply:
                    pack_ids = st.session_state.unity_created_packs.get(game) or []
                    if not pack_ids:
                        unity_ok_placeholder.error(
                            "적용할 creative pack이 없습니다. 먼저 '크리에이티브/팩 생성' 버튼을 눌러주세요."
                        )
                    else:
                        try:
                            result = unity_module.apply_unity_creative_packs_to_campaign(
                                game=game,
                                creative_pack_ids=pack_ids,
                                settings=unity_settings,
                            )

                            assigned = result.get("assigned_packs") or []
                            removed = result.get("removed_assignments") or []
                            errors = result.get("errors") or []

                            if assigned:
                                unity_ok_placeholder.success(
                                    f"캠페인에 {len(assigned)}개 creative pack을 assign했습니다.\n"
                                    "이전 iteration의 pack들은 모두 unassign 되었습니다."
                                )
                            else:
                                unity_ok_placeholder.warning(
                                    "캠페인에 새로 assign된 creative pack이 없습니다. "
                                    "Unity 대시보드에서 캠페인 상태를 확인해 주세요."
                                )

                            if removed:
                                st.caption(
                                    f"기존 assigned creative pack {len(removed)}개를 unassign 했습니다."
                                )

                            if errors:
                                st.error(
                                    "캠페인 적용 중 일부 오류가 발생했습니다:\n"
                                    + "\n".join(f"- {e}" for e in errors[:20])
                                    + ("\n..." if len(errors) > 20 else "")
                                )

                        except Exception as e:
                            import traceback
                            st.exception(e)
                            tb = traceback.format_exc()
                            unity_ok_placeholder.error("Unity 캠페인 적용 실패. 아래 오류 로그를 확인하세요.")
                            st.code(tb, language="python")

                # 3) CLEAR (Unity + Facebook for this game)
                if "clr_unity" in locals() and clr_unity:
                    st.session_state.uploads.pop(game, None)
                    st.session_state.remote_videos.pop(game, None)
                    st.session_state.settings.pop(game, None)
                    st.session_state.unity_settings.pop(game, None)
                    if "unity_created_packs" in st.session_state:
                        st.session_state.unity_created_packs.pop(game, None)

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


# ======================================================================
# PAGE ROUTING: 운영 / 마케터 – 모듈 세트만 다르게
# ======================================================================
if page == "Creative 자동 업로드":
    # 운영: facebook_ads / unity_ads 사용
    render_main_app("🎮 Creative 자동 업로드", fb_ops, uni_ops, is_marketer=False)
else:  # "Creative 자동 업로드 - 마케터"
    # 마케터: fb.py / uni.py 사용
    render_main_app("🎮 Creative 자동 업로드 - 마케터", fb_marketer, uni_marketer, is_marketer=True)