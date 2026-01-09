import os
import logging
from django.core.cache import cache
from openai import OpenAI

log = logging.getLogger(__name__)


class NewsService:
    """
    Reelioo V2 Neural Engine.
    Generates high-fidelity institutional market notes.
    """

    @staticmethod
    def get_smart_insights(symbol="BTC", mode="INTRADAY"):
        coin = symbol.replace("USDT", "").replace("-PERP", "").upper()

        # FIX 1: Make cache key unique to the MODE (Scalp vs Intraday)
        # FIX 2: Version bump to v8 to invalidate old stuck keys
        cache_key = f"desk_note_v8:{coin}:{mode}"

        # 1. Cache Check
        cached = cache.get(cache_key)
        if cached: return cached

        # 2. OpenAI Generation
        try:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

            # FIX 3: Inject the MODE into the prompt so the AI context changes
            context_str = "short-term scalping" if mode == "SCALP" else "intraday swing trading"

            prompt = (
                f"Analyze {coin} market structure for {context_str}. Return a single string in this exact format: 'TAG|Message'. "
                f"The TAG must be 1-3 words, uppercase (e.g. LIQUIDITY SWEEP, MOMENTUM SHIFT, RANGE BOUND). "
                f"The Message must be concise (max 12 words) and actionable. "
                f"Example output: 'ORDER BLOCK|High volume rejection at 60k confirms support.'"
            )

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,  # Slightly higher temp for more variety
                max_tokens=50
            )

            content = response.choices[0].message.content.strip().replace('"', '')

            if "|" not in content:
                content = f"MARKET NOTE|{content}"

            if content:
                # FIX 4: Reduce timeout from 7200s (2hr) to 900s (15 min)
                cache.set(cache_key, content, timeout=900)
                return content

        except Exception as e:
            log.error(f"AI Error: {e}")

        return f"VOLATILITY ALERT|Liquidity thin, expect rapid moves."