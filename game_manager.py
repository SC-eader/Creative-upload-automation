# game_manager.py
import streamlit as st
from google.cloud import bigquery
from google.oauth2 import service_account
import datetime
import json

# --- BIGQUERY CONFIGURATION ---
# We point specifically to your new table
BQ_PROJECT_ID = "roas-test-456808"
BQ_DATASET_ID = "marketing_datascience"
BQ_TABLE_ID = "game_configs"
FULL_TABLE_ID = f"{BQ_PROJECT_ID}.{BQ_DATASET_ID}.{BQ_TABLE_ID}"

# Hardcoded defaults (Legacy support - these will always be visible)
DEFAULT_GAME_NAMES = [
    "XP HERO", "Dino Universe", "Snake Clash", "Pizza Ready", "Cafe Life",
    "Suzy's Restaurant", "Office Life", "Lumber Chopper", "Burger Please", "Prison Life"
]

def get_bq_client():
    """Initialize BigQuery Client from Streamlit Secrets."""
    # We reuse the same GCP secrets you use for Drive Import
    # Ensure st.secrets["gcp_service_account"] exists!
    if "gcp_service_account" not in st.secrets:
        raise RuntimeError("Missing 'gcp_service_account' in secrets.toml")
        
    creds_info = st.secrets["gcp_service_account"]
    creds = service_account.Credentials.from_service_account_info(creds_info)
    return bigquery.Client(credentials=creds, project=creds_info["project_id"])

@st.cache_data(ttl=600) # Cache data for 10 minutes to save BQ costs/speed up UI
def load_custom_config() -> dict:
    """
    Fetch all games from BigQuery.
    Returns: { "GameName": { "facebook": {...}, "unity": {...} } }
    """
    client = get_bq_client()
    
    # Query the 'config' JSON column directly
    query = f"""
        SELECT game_name, config
        FROM `{FULL_TABLE_ID}`
    """
    
    try:
        query_job = client.query(query)
        results = query_job.result()
        
        config_map = {}
        for row in results:
            # 'row.config' comes back as a Python dictionary automatically
            # if the column type is JSON. If it's STRING, we json.loads() it.
            try:
                data = row.config
                if isinstance(data, str):
                    data = json.loads(data)
                config_map[row.game_name] = data
            except Exception as e:
                print(f"Skipping row {row.game_name}: {e}")
                
        return config_map
        
    except Exception as e:
        # Graceful fallback if table is empty or permission fails
        print(f"BigQuery Load Error: {e}") 
        return {}

def get_all_game_names(include_custom: bool = True) -> list:
    """Returns merged list of Default + BigQuery games."""
    if not include_custom:
        return list(DEFAULT_GAME_NAMES)
        
    custom = load_custom_config()
    all_games = list(DEFAULT_GAME_NAMES)
    
    # Add BQ games, avoiding duplicates (BigQuery overrides defaults if names match)
    for name in custom.keys():
        if name not in all_games:
            all_games.append(name)
            
    return all_games

def save_new_game(game_name: str, fb_account_id: str, fb_page_id: str, unity_game_id: str = "", fb_app_id: str = ""):
    """
    Bundles the IDs into a JSON object and inserts it into BigQuery.
    """
    client = get_bq_client()
    
    # 1. Create the Universal Config Object
    # This structure is what allows us to add 'tiktok', 'google', etc. later!
    config_payload = {
        "facebook": {
            "account_id": fb_account_id.strip(),
            "page_id": fb_page_id.strip(),
            "app_id": fb_app_id.strip(), # Saved here!
            "campaign_id": "",
            "adset_prefix": f"{game_name.lower().replace(' ', '')}_creative_test"
        },
        "unity": {
            "game_id": unity_game_id.strip()
        }
        # Future: "tiktok": { ... }
    }
    
    # 2. Prepare the Row for BigQuery
    # Note: For BQ JSON type, we pass the dict directly.
    rows_to_insert = [
        {
            "game_name": game_name,
            "config": json.dumps(config_payload), # Serialize to string for safe insertion
            "updated_at": datetime.datetime.now().isoformat()
        }
    ]

    # 3. Insert
    errors = client.insert_rows_json(FULL_TABLE_ID, rows_to_insert)
    
    if errors:
        raise RuntimeError(f"BigQuery Insert Errors: {errors}")
        
    # 4. Clear Cache so the new game appears immediately in the UI
    st.cache_data.clear()

def get_game_config(game_name: str, platform: str) -> dict:
    """
    Retrieves config for a specific game/platform.
    Example: get_game_config("MyGame", "facebook") -> returns the FB dict
    """
    custom = load_custom_config()
    
    # 'custom' is now { "GameName": { "facebook": {...}, "unity": {...} } }
    game_data = custom.get(game_name, {})
    return game_data.get(platform, {})