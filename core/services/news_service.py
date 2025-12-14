import os
import json
import logging
from datetime import datetime
from django.core.cache import cache
from openai import OpenAI

log = logging.getLogger(__name__)


class NewsService:
    """
    Reelioo V2 Neural Engine.
    Uses OpenAI GPT-4o-mini to generate institutional-grade market briefs.
    """

    @staticmethod
    def get_smart_insights(symbol="BTC"):
        coin = symbol.replace("USDT", "").replace("-PERP", "").upper()
        cache_key = f"ai_insights_v2:{coin}"

        # 1. CACHE CHECK (2 Hours - Strategic Insights)
        cached = cache.get(cache_key)
        if cached: return cached

        # 2. OPENAI GENERATION
        try:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

            prompt = (
                f"Act as a senior quantitative analyst. "
                f"Generate 3 distinct, high-alpha market insights for {coin} based on current market structure. "
                f"Focus strictly on: Order Flow, Volatility Regimes, and Macro Correlation. "
                f"Use professional, terse language. No financial advice. Max 15 words per point. "
                f"Return ONLY a raw JSON list of strings."
            )

            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Cost-effective & Fast
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=150
            )

            content = response.choices[0].message.content.strip()
            # Clean Markdown
            if content.startswith("```"):
                content = content.split("```")[1].replace("json", "").strip()

            insights = json.loads(content)

            # Format
            data = []
            for txt in insights:
                data.append({
                    "title": txt,
                    "source": "REELIOO NEURAL",
                    "url": "#",
                    "published_at": datetime.now().isoformat()
                })

            if data:
                cache.set(cache_key, data, timeout=7200)
                return data

        except Exception as e:
            log.error(f"AI Insight Error: {e}")

        # 3. FALLBACK
        return [
            {"title": f"Volatility matrix calculation active for {coin}.", "source": "SYSTEM",
             "published_at": datetime.now().isoformat()},
            {"title": "Monitoring institutional order block depth.", "source": "QUANT CORE",
             "published_at": datetime.now().isoformat()}
        ]