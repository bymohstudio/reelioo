# core/services/ai_commentary_service.py

import logging
import re
from typing import Dict, Any
from datetime import datetime
import pytz

try:
    from openai import OpenAI
    client = OpenAI()
except ImportError:
    client = None

logger = logging.getLogger(__name__)


class AICommentaryService:
    """
    Ultra-Premium Institutional Desk Note Generator
    """

    @staticmethod
    def _clean_markdown(text: str) -> str:
        if not text:
            return ""

        # ------------------------------------------------------------
        # 1) Convert Markdown Headings → Rounded Premium Pills
        # ------------------------------------------------------------
        text = re.sub(
            r'###?\s+(.*?)\n',
            r'<div class="inline-block px-2 py-1 rounded-lg bg-slate-800/60 '
            r'text-blue-300 border border-blue-500/30 text-[11px] font-semibold '
            r'tracking-wide mb-2">\1</div>',
            text + "\n"
        )

        # ------------------------------------------------------------
        # 2) Bold Text (**text**) → Premium Metric Badge
        # ------------------------------------------------------------
        text = re.sub(
            r'\*\*(.*?)\*\*',
            r'<span class="inline-block px-2 py-0.5 mx-1 rounded bg-slate-700 '
            r'border border-slate-500 text-white text-[11px] font-semibold '
            r'tracking-wide shadow-sm">\1</span>',
            text
        )

        # ------------------------------------------------------------
        # 3) Hashtags (#TEXT) → Blue Tag Badge
        # ------------------------------------------------------------
        text = re.sub(
            r'#([A-Za-z0-9_]+)',
            r'<span class="inline-block px-2 py-0.5 mx-1 rounded bg-blue-900/40 '
            r'border border-blue-500/40 text-blue-300 text-[10px] font-bold '
            r'uppercase tracking-wide">\1</span>',
            text
        )

        # ------------------------------------------------------------
        # 4) Convert Bullet Points (GPT usually generates "- item")
        # ------------------------------------------------------------
        lines = text.split("\n")
        html = ""
        inside_list = False

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            if stripped.startswith("- ") or stripped.startswith("* "):
                content = stripped[2:]
                if not inside_list:
                    html += "<ul class='space-y-1.5 list-disc pl-4 text-slate-300 marker:text-blue-500'>"
                    inside_list = True
                html += f"<li class='leading-relaxed text-[13px]'>{content}</li>"
            else:
                if inside_list:
                    html += "</ul>"
                    inside_list = False
                html += f"<p class='mb-1.5 text-[13px] text-slate-300'>{stripped}</p>"

        if inside_list:
            html += "</ul>"

        return html

    # =============================================================
    #   MAIN COMMENTARY GENERATOR
    # =============================================================
    @staticmethod
    def generate_quant_commentary(result) -> str:
        try:
            extras = getattr(result, "extras", {}) or {}
            score = float(getattr(result, "score", 0))
            direction = getattr(result, "direction", "NEUTRAL")
            vol = getattr(result, "volatility_regime", "NORMAL")
            market = getattr(result, "market_type", "EQUITY")
            symbol = getattr(result, "symbol", "UNKNOWN")

            # --------------------------------------------------------
            # HEADER PILLS (BIAS, RISKS, VOLATILITY, WHALE, MANIPULATION)
            # --------------------------------------------------------
            pills_html = "<div class='flex flex-wrap gap-2 mb-3'>"

            # BIAS PILL
            if score >= 55:
                pills_html += """<span class="px-2 py-1 rounded bg-emerald-500/20 text-emerald-300 
                                  border border-emerald-500/50 text-[10px] font-bold">BULLISH BIAS</span>"""
            elif score <= 45:
                pills_html += """<span class="px-2 py-1 rounded bg-red-500/20 text-red-300 
                                  border border-red-500/50 text-[10px] font-bold">BEARISH BIAS</span>"""
            else:
                pills_html += """<span class="px-2 py-1 rounded bg-slate-500/20 text-slate-300 
                                  border border-slate-500/50 text-[10px] font-bold">NEUTRAL</span>"""

            # VOLATILITY PILL
            pills_html += f"""
                <span class="px-2 py-1 rounded bg-indigo-500/20 text-indigo-300 
                border border-indigo-500/50 text-[10px] font-bold">{vol.upper()} VOL</span>
            """

            # MANIPULATION
            if extras.get("liquidity_grab_risk") == "HIGH":
                pills_html += """<span class="px-2 py-1 rounded bg-yellow-500/20 text-yellow-300 
                                  border border-yellow-500/50 text-[10px] font-bold">LIQUIDITY GRAB</span>"""

            if extras.get("stop_run_risk") == "HIGH":
                pills_html += """<span class="px-2 py-1 rounded bg-red-500/20 text-red-300 
                                  border border-red-500/50 text-[10px] font-bold">STOP HUNT</span>"""

            # WHALE VOLUME
            if float(extras.get("whale_activity_z", 0)) >= 1.5:
                pills_html += """<span class="px-2 py-1 rounded bg-blue-500/20 text-blue-300 
                                  border border-blue-500/50 text-[10px] font-bold">WHALE ACTIVITY</span>"""

            pills_html += "</div>"

            # --------------------------------------------------------
            # AI REQUEST: SHORT + INSTITUTIONAL + NO HEADINGS
            # --------------------------------------------------------
            commentary_text = ""

            if client:
                try:
                    system_prompt = (
                        "You are a Senior Quant Analyst. "
                        "Generate a **VERY SHORT** 3–4 line institutional desk note. "
                        "No headings like ## or ###. "
                        "Use compact bullet points only. "
                        "Highlight: Bias, Liquidity, Whale Activity, Risk. "
                        "Use **bold** for numbers & metrics. "
                        "Use #TAGS for market phenomena. "
                        "Never write paragraphs. Only bullets."
                    )

                    user_prompt = (
                        f"Asset: {symbol}\n"
                        f"Bias: {direction} at {score:.2f}%\n"
                        f"Volatility: {vol}\n"
                        f"Liquidity Risk: {extras.get('liquidity_grab_risk')}\n"
                        f"Whale Z-Score: {float(extras.get('whale_activity_z',0)):.2f}\n"
                        f"Market: {market}\n"
                    )

                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        max_tokens=90,
                        temperature=0.6,
                    )
                    raw = response.choices[0].message.content
                    commentary_text = AICommentaryService._clean_markdown(raw)

                except Exception as e:
                    logger.error(f"OpenAI Failed: {e}")
                    commentary_text = ""

            # FALLBACK IF AI FAILS
            if not commentary_text:
                fallback = f"""
                - **{score:.2f}%** {direction} structure with stable flow.
                - Volatility regime: **{vol}**.
                - Liquidity conditions: {extras.get("liquidity_grab_risk", "LOW")}.
                - Whale Activity: Z = **{float(extras.get('whale_activity_z',0)):.2f}**.
                """
                commentary_text = AICommentaryService._clean_markdown(fallback)

            # --------------------------------------------------------
            # MARKET STATUS DISCLAIMER
            # --------------------------------------------------------
            ist = pytz.timezone("Asia/Kolkata")
            now = datetime.now(ist)
            start = now.replace(hour=9, minute=15, second=0)
            end = now.replace(hour=15, minute=30, second=0)

            if market == "CRYPTO":
                status_msg = "LIVE • 24/7 MARKET"
            else:
                status_msg = "MARKET OPEN • LIVE" if (start <= now <= end) else "MARKET CLOSED • EOD DATA"

            disclaimer = f"""
                <div class='mt-3 pt-2 border-t border-white/5 text-[10px] 
                text-slate-500 uppercase tracking-wide'>{status_msg}</div>
            """

            return pills_html + commentary_text + disclaimer

        except Exception as e:
            logger.error(str(e))
            return "<p class='text-slate-400'>Analysis generated.</p>"


generate_quant_commentary = AICommentaryService.generate_quant_commentary
