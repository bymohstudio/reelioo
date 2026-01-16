import os

import requests
import urllib.parse
import random
from dotenv import load_dotenv

# 1. Load Environment Variables
load_dotenv()

# Note: We do NOT include send_discord_alert here because
# you already have it defined in views.py

def send_marketing_prompt(symbol, score, whale_status):
    """
    Sends a private alert to your Marketing Channel with a
    'Click to Tweet' button for ENTRIES.
    """
    webhook_url = os.getenv('DISCORD_MARKETING_WEBHOOK_URL')
    if not webhook_url: return

    # A. Visuals: Minimalist Radar
    emojis = ["💠", "📡", "⚡", "👁️"]
    selected_emoji = random.choice(emojis)

    # B. Text: "We see what others don't."
    tweet_text = (
        f"{selected_emoji} {symbol}\n\n"
        f"Institutional vectors aligned.\n"
        f"Volatility expansion imminent.\n\n"
        f"Score: {score}/100\n"
        f"Status: {whale_status}\n\n"
        f"Data > Noise.\n"
        f"#Quant #Crypto #{symbol}"
    )

    # C. Links
    encoded_text = urllib.parse.quote(tweet_text)
    web_link = f"https://twitter.com/intent/tweet?text={encoded_text}"
    mobile_link = f"twitter://post?message={encoded_text}"

    # D. Discord Preview
    payload = {
        "username": "Reelioo Marketing",
        "embeds": [{
            "title": f"📡 FORECAST: {symbol}",
            "description": f"**Style:** Clean Quant\n**Tone:** 'We know something.'",
            "color": 3447003,  # Blue
            "fields": [
                {"name": "📱 App Link", "value": f"[Click to Tweet]({mobile_link})", "inline": True},
                {"name": "💻 Web Link", "value": f"[Click to Tweet]({web_link})", "inline": True}
            ]
        }]
    }

    try:
        requests.post(webhook_url, json=payload)
    except Exception as e:
        print(f"Marketing Entry Fail: {e}")


def send_win_prompt(symbol, roi, duration_hours):
    """
    Sends a private alert to your Marketing Channel with a
    'Click to Tweet' button for WINS.
    """
    webhook_url = os.getenv('DISCORD_MARKETING_WEBHOOK_URL')
    if not webhook_url: return

    # --- 1. SMART TIME FORMATTING ---
    # A. Time Format
    if duration_hours < 1:
        time_str = f"{int(duration_hours * 60)}m"
    else:
        h = int(duration_hours)
        m = int((duration_hours - h) * 60)
        time_str = f"{h}h {m}m"

    # B. Text: "Pure Math."
    tweet_text = (
        f"✅ {symbol}\n\n"
        f"+{roi}% captured in {time_str}.\n\n"
        f"Pure execution. No emotions.\n"
        f"The edge is real.\n\n"
        f"#{symbol} #Reelioo"
    )

    # C. Links
    encoded_text = urllib.parse.quote(tweet_text)
    web_link = f"https://twitter.com/intent/tweet?text={encoded_text}"
    mobile_link = f"twitter://post?message={encoded_text}"

    # D. Discord Preview
    payload = {
        "username": "Reelioo Marketing",
        "embeds": [{
            "title": f"💰 WIN: {symbol} (+{roi}%)",
            "description": f"**Style:** Minimalist Flex\n**Tone:** 'Easy money.'",
            "color": 5763719,  # Green
            "fields": [
                {"name": "📱 App Link", "value": f"[Click to Tweet]({mobile_link})", "inline": True},
                {"name": "💻 Web Link", "value": f"[Click to Tweet]({web_link})", "inline": True}
            ]
        }]
    }

    try:
        requests.post(webhook_url, json=payload)
    except Exception as e:
        print(f"Marketing Win Fail: {e}")
