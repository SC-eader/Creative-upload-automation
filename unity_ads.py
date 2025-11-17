"""Unity Ads helpers for Creative 자동 업로드 Streamlit app."""

from __future__ import annotations

from typing import Dict
import streamlit as st


def _ensure_unity_settings_state() -> None:
    """Make sure we have st.session_state.unity_settings for per-game Unity settings."""
    if "unity_settings" not in st.session_state:
        st.session_state.unity_settings = {}


def get_unity_settings(game: str) -> Dict:
    """Return the Unity settings dict for the given game (empty if not set yet)."""
    _ensure_unity_settings_state()
    return st.session_state.unity_settings.get(game, {})


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
