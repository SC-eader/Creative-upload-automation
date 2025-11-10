from facebook_business.api import FacebookAdsApi
from facebook_business.adobjects.adaccount import AdAccount


FacebookAdsApi.init(access_token=ACCESS_TOKEN, app_id=APP_ID, app_secret=APP_SECRET)

acct = AdAccount(AD_ACCOUNT_ID)
info = acct.api_get(fields=["name", "account_status", "currency"])
print("✅ Connected to:", info["name"])
print("Status:", info["account_status"])
print("Currency:", info["currency"])

print("\nListing a few campaigns:")
for c in acct.get_campaigns(fields=["id", "name"], params={"limit": 5}):
    print("-", c["id"], c["name"])