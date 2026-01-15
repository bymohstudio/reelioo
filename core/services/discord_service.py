import requests
import urllib.parse
import random
from django.conf import settings

# Note: We do NOT include send_discord_alert here because
# you already have it defined in views.py

def send_marketing_prompt(symbol, score, whale_status):
    """
    Sends a private alert to your Marketing Channel with a
    'Click to Tweet' button for ENTRIES.
    """
    webhook_url = getattr(settings, 'DISCORD_MARKETING_WEBHOOK_URL', '')
    if not webhook_url: return

    # 1. Visual Hooks
    emojis = ["🌊", "📡", "⚡", "👁️", "💠"]
    selected_emoji = random.choice(emojis)

    # 2. Deterministic Context
    context_lines = [
        "Structural anomaly detected.",
        "Volatility compression active.",
        "Order flow divergence identified.",
        "Market liquidity engaging.",
        "Deterministic setup sequence."
    ]

    if whale_status == "ACTIVE":
        selected_context = "Institutional volume signature detected."
    elif score >= 85:
        selected_context = "High-velocity momentum expansion imminent."
    else:
        selected_context = random.choice(context_lines)

    # 3. Build the Tweet Text
    tweet_text = (
        f"{selected_emoji} SYSTEM ALERT: ${symbol}\n\n"
        f"Signal Strength: {score}/100\n"
        f"{selected_context}\n\n"
        f"Tracking execution vectors...\n\n"
        f"#Crypto #{symbol} #Trading"
    )

    # 4. Create the Magic Link
    encoded_text = urllib.parse.quote(tweet_text)
    twitter_link = f"https://twitter.com/intent/tweet?text={encoded_text}"

    # 5. Send to Discord
    payload = {
        "username": "Reelioo Marketing",
        "embeds": [{
            "title": f"📢 ENTRY OPPORTUNITY: {symbol}",
            "description": f"**Score:** {score}/100\n**Whale:** {whale_status}",
            "color": 15105570,  # Orange
            "fields": [
                {
                    "name": "⚡ ACTION REQUIRED",
                    "value": f"👉 **[CLICK TO TWEET ENTRY]({twitter_link})**",
                    "inline": False
                }
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
    webhook_url = getattr(settings, 'DISCORD_MARKETING_WEBHOOK_URL', '')
    if not webhook_url: return

    # 1. Format Time
    if duration_hours < 1:
        time_str = f"{int(duration_hours * 60)}m"
    else:
        time_str = f"{round(duration_hours, 1)}h"

    # 2. Build Tweet Text
    tweet_text = (
        f"✅ TARGET NEUTRALIZED: ${symbol}\n\n"
        f"• Result: +{roi}% Captured\n"
        f"• Time: {time_str}\n\n"
        f"Calculated precision. No guessing.\n\n"
        f"#{symbol} #CryptoWins"
    )

    # 3. Create Magic Link
    encoded_text = urllib.parse.quote(tweet_text)
    twitter_link = f"https://twitter.com/intent/tweet?text={encoded_text}"

    # 4. Send to Discord
    payload = {
        "username": "Reelioo Marketing",
        "embeds": [{
            "title": f"💰 WIN CONFIRMED: {symbol}",
            "description": f"**ROI:** +{roi}%\n**Time:** {time_str}",
            "color": 5763719,  # Green
            "fields": [
                {
                    "name": "⚡ ACTION REQUIRED",
                    "value": f"👉 **[CLICK TO TWEET RECEIPT]({twitter_link})**",
                    "inline": False
                }
            ]
        }]
    }

    try:
        requests.post(webhook_url, json=payload)
    except Exception as e:
        print(f"Marketing Win Fail: {e}")