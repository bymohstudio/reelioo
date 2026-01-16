import requests
import logging

log = logging.getLogger(__name__)

# --- CONFIG ---
# Keys that bypass the API check completely.
# useful for dev, testing, or giving free access to friends.
BYPASS_KEYS = {
    "dev-mode": "Developer Access",
    "beta-tester": "Beta Access",
    "reelioo-god": "God Mode",
    "free-trial-bypass": "Manual Override"
}


def verify_gumroad_license(license_key):
    """
    Verifies a license key.
    1. Checks Internal Bypass Keys first.
    2. Checks Gumroad API second.
    """
    # 1. CHECK INTERNAL BYPASS
    if license_key in BYPASS_KEYS:
        return True, BYPASS_KEYS[license_key]

    # 2. CHECK GUMROAD API
    PRODUCT_PERMALINK = "reelioo-pro"

    url = "https://api.gumroad.com/v2/licenses/verify"
    data = {
        "product_permalink": PRODUCT_PERMALINK,
        "license_key": license_key,
        "increment_uses_count": "false"
    }

    try:
        r = requests.post(url, data=data, timeout=10)

        if r.status_code == 404:
            return False, "License does not exist."

        res = r.json()

        if not res.get('success'):
            return False, "Invalid License Key."

        purchase = res.get('purchase', {})

        if purchase.get('refunded') or purchase.get('chargebacked'):
            return False, "License was refunded or revoked."

        if purchase.get('subscription_cancelled_at'):
            return False, "Subscription has been cancelled."

        if purchase.get('subscription_failed_at'):
            return False, "Payment failed. Please update card on Gumroad."

        return True, "Active"

    except Exception as e:
        log.error(f"Gumroad API Error: {e}")
        return False, "Connection Error."