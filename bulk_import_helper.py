"""
bulk_import_helper.py
Helper script to fetch assets from Meta & Unity and generate BigQuery SQL.
"""
import streamlit as st
import requests
import json
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BulkImport")

# --- 1. FACEBOOK FETCH ---
def find_app_id_from_campaigns(ad_account):
    """Scans recent campaigns to find the 'promoted_object' (App ID)."""
    try:
        campaigns = ad_account.get_campaigns(
            fields=['name', 'promoted_object'],
            params={'limit': 20}
        )
        for camp in campaigns:
            promo_obj = camp.get('promoted_object')
            if promo_obj and 'application_id' in promo_obj:
                return promo_obj['application_id']
    except Exception:
        pass
    return ""

def fetch_facebook_assets():
    """Fetches all Ad Accounts and Pages from Meta."""
    logger.info("Fetching Facebook Assets...")
    try:
        from facebook_business.api import FacebookAdsApi
        from facebook_business.adobjects.user import User
        from facebook_business.adobjects.adaccount import AdAccount
        
        my_app_id = st.secrets["facebook"]["app_id"]
        my_app_secret = st.secrets["facebook"]["app_secret"]
        my_access_token = st.secrets["facebook"]["access_token"]
        
        FacebookAdsApi.init(my_app_id, my_app_secret, my_access_token)
        me = User(fbid='me')
        
        accounts = me.get_ad_accounts(fields=['name', 'account_id', 'account_status'])
        ad_accounts = []
        
        print(f"Scanning {len(accounts)} Ad Accounts for App IDs...")
        
        for acc in accounts:
            if acc['account_status'] == 1: # ACTIVE only
                acc_obj = AdAccount(f"act_{acc['account_id']}")
                found_app_id = find_app_id_from_campaigns(acc_obj)
                ad_accounts.append({
                    "name": acc['name'],
                    "id": f"act_{acc['account_id']}",
                    "app_id": found_app_id
                })

        pages = me.get_accounts(fields=['name', 'id', 'is_published'])
        fb_pages = []
        for pg in pages:
            fb_pages.append({"name": pg['name'], "id": pg['id']})
            
        logger.info(f"Found {len(ad_accounts)} Active Ad Accounts and {len(fb_pages)} Pages.")
        return ad_accounts, fb_pages

    except Exception as e:
        logger.error(f"Facebook Fetch Failed: {e}")
        return [], []

# --- 2. UNITY FETCH (WITH STORE MERGING) ---
def fetch_unity_apps():
    """Fetches Unity Apps and returns a DICTIONARY merged by name."""
    logger.info("Fetching Unity Apps...")
    
    org_id = st.secrets["unity"].get("organization_id")
    api_key = st.secrets["unity"].get("authorization_header")
    
    if not org_id or not api_key:
        logger.error("Missing Unity Org ID or Auth Header.")
        return {}

    url = f"https://services.api.unity.com/advertise/v1/organizations/{org_id}/apps"
    headers = {"Authorization": api_key, "Content-Type": "application/json"}
    
    merged_apps = {} # Key: Game Name, Value: {android_id: ..., ios_id: ...}

    try:
        resp = requests.get(url, headers=headers, timeout=10)
        data = resp.json()
        results = data.get("results", []) or data.get("items", [])
        
        for app in results:
            name = app.get("name")
            game_id = app.get("id") or app.get("gameId")
            store = (app.get("store") or "").lower() # 'apple_app_store' or 'google_play'

            if name not in merged_apps:
                merged_apps[name] = {"name": name, "android_id": "", "ios_id": ""}
            
            # Smart Assignment
            if "google" in store:
                merged_apps[name]["android_id"] = game_id
            elif "apple" in store:
                merged_apps[name]["ios_id"] = game_id
            else:
                # Fallback if store is unknown, assign to android if empty, else ios
                if not merged_apps[name]["android_id"]:
                    merged_apps[name]["android_id"] = game_id
                else:
                    merged_apps[name]["ios_id"] = game_id

        return merged_apps

    except Exception as e:
        logger.error(f"Unity Fetch Failed: {e}")
        return {}

# --- 3. MATCHING LOGIC ---
def normalize(s):
    if not s: return ""
    return s.lower().replace(" ", "").replace("-", "").replace("_", "")

def match_assets(fb_accounts, fb_pages, unity_apps_dict):
    matches = []
    
    # Iterate over the MERGED Unity apps
    for game_name, uni_data in unity_apps_dict.items():
        norm_name = normalize(game_name)
        
        best_fb = None
        for acc in fb_accounts:
            if norm_name in normalize(acc['name']):
                best_fb = acc
                break
        
        best_page = None
        for pg in fb_pages:
            if norm_name in normalize(pg['name']):
                best_page = pg
                break
        
        matches.append({
            "game_name": game_name,
            "unity_android_id": uni_data['android_id'],
            "unity_ios_id": uni_data['ios_id'],
            "fb_account_id": best_fb['id'] if best_fb else "MISSING",
            "fb_app_id": best_fb['app_id'] if best_fb else "",
            "fb_page_id": best_page['id'] if best_page else "MISSING"
        })
        
    return matches

# --- 4. SQL GENERATOR ---
def generate_sql(matches):
    print("\n" + "="*50)
    print("       GENERATED BIGQUERY SQL QUERY (MERGED OS)")
    print("="*50 + "\n")
    
    print("INSERT INTO `roas-test-456808.marketing_datascience.game_configs` (game_name, config, updated_at) VALUES")
    
    rows = []
    for m in matches:
        if m['fb_account_id'] == "MISSING" and m['fb_page_id'] == "MISSING":
            continue

        # --- NEW CONFIG STRUCTURE ---
        config_payload = {
            "facebook": {
                "account_id": m['fb_account_id'],
                "page_id": m['fb_page_id'],
                "app_id": m['fb_app_id'],
                "campaign_id": "",
                "adset_prefix": f"{m['game_name'].lower().replace(' ', '')}_creative_test"
            },
            "unity": {
                "android_game_id": m['unity_android_id'],
                "ios_game_id": m['unity_ios_id']
            }
        }
        
        safe_name = m['game_name'].replace("'", "\\'")
        json_str = json.dumps(config_payload).replace("'", "\\'")
        
        rows.append(f"('{safe_name}', JSON '{json_str}', CURRENT_TIMESTAMP())")
    
    print(",\n".join(rows) + ";")
    print("\n" + "="*50)

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("Starting Smart Bulk Import Helper (With OS Merge)...")
    fb_accs, fb_pgs = fetch_facebook_assets()
    uni_apps_dict = fetch_unity_apps()
    
    if fb_accs or uni_apps_dict:
        matched_data = match_assets(fb_accs, fb_pgs, uni_apps_dict)
        generate_sql(matched_data)
    else:
        print("Failed to fetch data.")