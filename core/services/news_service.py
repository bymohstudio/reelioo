import os
import logging
from datetime import datetime
from django.core.cache import cache
from openai import OpenAI

log = logging.getLogger(__name__)


class NewsService:
    """
    Reelioo V2 Neural Engine.
    Generates high-fidelity institutional market notes.
    """

    @staticmethod
    def get_smart_insights(symbol="BTC"):
        coin = symbol.replace("USDT", "").replace("-PERP", "").upper()
        cache_key = f"desk_note_v7:{coin}"

        # 1. Cache Check
        cached = cache.get(cache_key)
        if cached: return cached

        # 2. OpenAI Generation
        try:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

            # PROMPT: Enforce "TAG|Message" format
            prompt = (
                f"Analyze {coin} market structure. Return a single string in this exact format: 'TAG|Message'. "
                f"The TAG must be 1-3 words, uppercase (e.g. LIQUIDITY GRAB, WHALE BUYING, STOP HUNT). "
                f"The Message must be concise (max 12 words) and actionable. "
                f"Example output: 'ORDER BLOCK|High volume rejection at 60k confirms support.'"
            )

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4, max_tokens=40
            )

            content = response.choices[0].message.content.strip().replace('"', '')

            # Fallback if AI forgets format
            if "|" not in content:
                content = f"MARKET NOTE|{content}"

            if content:
                cache.set(cache_key, content, timeout=7200)
                return content

        except Exception as e:
            log.error(f"AI Error: {e}")

        # 3. Fallback
        return f"VOLATILITY ALERT|Liquidity thin at current levels, expect rapid moves."