import os
import tweepy
import logging
import random

log = logging.getLogger(__name__)


class TwitterBot:
    def __init__(self):
        # Configure these in your .env file
        self.api_key = os.getenv("TWITTER_API_KEY")
        self.api_secret = os.getenv("TWITTER_API_SECRET")
        self.access_token = os.getenv("TWITTER_ACCESS_TOKEN")
        self.access_secret = os.getenv("TWITTER_ACCESS_SECRET")
        self.bearer_token = os.getenv("TWITTER_BEARER_TOKEN")

        self.client = None
        if self.api_key and self.access_token:
            try:
                # Client for v2 API
                self.client = tweepy.Client(
                    bearer_token=self.bearer_token,
                    consumer_key=self.api_key,
                    consumer_secret=self.api_secret,
                    access_token=self.access_token,
                    access_token_secret=self.access_secret
                )
            except Exception as e:
                log.error(f"Twitter Auth Failed: {e}")

    def post_entry_signal(self, symbol, score, whale_status):
        """
        Subtle hype tweet. Doesn't promise a moonshot, just indicates 'Activity'.
        Uses terms like 'Structure', 'Flow', 'Dynamics'.
        """
        if not self.client: return

        # Visual Hooks
        emojis = ["🌊", "📡", "⚡", "👁️", "💠"]
        selected_emoji = random.choice(emojis)

        # Deterministic / Structural Context
        context_lines = [
            "Structural anomaly detected.",
            "Volatility compression active.",
            "Order flow divergence identified.",
            "Market liquidity engaging.",
            "Deterministic setup sequence."
        ]

        # High Confidence Context
        if whale_status == "ACTIVE":
            selected_context = "Institutional volume signature detected."
        elif score >= 85:
            selected_context = "High-velocity momentum expansion imminent."
        else:
            selected_context = random.choice(context_lines)

        tweet = (
            f"{selected_emoji} SYSTEM ALERT: ${symbol}\n\n"
            f"Signal Strength: {score}/100\n"
            f"{selected_context}\n\n"
            f"Tracking execution vectors...\n\n"
            f"#Crypto #{symbol} #Trading"
        )

        try:
            self.client.create_tweet(text=tweet)
        except Exception as e:
            log.error(f"Entry Tweet Failed: {e}")

    def post_win_receipt(self, symbol, roi, duration_hours):
        """
        The 'Receipt'. Shows the result.
        Focuses on 'Precision' and 'Calculated' outcome.
        """
        if not self.client: return

        # Format time nicely
        if duration_hours < 1:
            time_str = f"{int(duration_hours * 60)}m"
        else:
            time_str = f"{round(duration_hours, 1)}h"

        tweet = (
            f"✅ TARGET NEUTRALIZED: ${symbol}\n\n"
            f"• Result: +{roi}% Captured\n"
            f"• Time: {time_str}\n\n"
            f"Calculated precision. No guessing.\n\n"
            f"#{symbol} #CryptoWins"
        )

        try:
            self.client.create_tweet(text=tweet)
        except Exception as e:
            log.error(f"Win Tweet Failed: {e}")