# game_manager.py
import json
import os
import streamlit as st

CONFIG_FILE = "games_config.json"

# Base list of games (Operation mode defaults)
DEFAULT_GAME_NAMES = [
    "XP HERO", "Dino Universe", "Snake Clash", "Pizza Ready", "Cafe Life",
    "Suzy's Restaurant", "Office Life", "Lumber Chopper", "Burger Please", "Prison Life"
]

def load_custom_config() -> dict:
    """Loads custom games from local JSON file."""
    if not os.path.exists(CONFIG_FILE):
        return {}
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def get_all_game_names(include_custom: bool = True) -> list:
    """
    Returns list of game names. 
    - If include_custom=True (Marketer): Returns Defaults + Custom games.
    - If include_custom=False (Operation): Returns Defaults only.
    """
    if not include_custom:
        return list(DEFAULT_GAME_NAMES)
        
    custom = load_custom_config()
    # Combine and deduplicate while preserving order of defaults
    all_games = list(DEFAULT_GAME_NAMES)
    for name in custom.keys():
        if name not in all_games:
            all_games.append(name)
    return all_games

def save_new_game(game_name: str, fb_account_id: str, fb_page_id: str, unity_game_id: str = ""):
    """Saves a new game configuration to disk."""
    config = load_custom_config()
    
    config[game_name] = {
        "facebook": {
            "account_id": fb_account_id.strip(),
            "page_id": fb_page_id.strip(),
            "campaign_id": "", 
            "adset_prefix": f"{game_name.lower().replace(' ', '')}_creative_test",
        },
        "unity": {
            "game_id": unity_game_id.strip()
        }
    }
    
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
        
    st.cache_data.clear()

def get_game_config(game_name: str, platform: str) -> dict:
    """Retrieves config for a specific game/platform."""
    custom = load_custom_config()
    return custom.get(game_name, {}).get(platform, {})