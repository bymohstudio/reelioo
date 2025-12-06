import os
import json
import logging
from typing import Dict, Any
from openai import OpenAI

logger = logging.getLogger(__name__)

class AICommentary:
    """
    AI Brain tailored for Indian Markets (Nifty/BankNifty) and Crypto.
    """
    
    @staticmethod
    def _render_html(symbol, analysis, context, detailed_insight_html):
        score = analysis.get("score", 50)
        direction = analysis.get("direction", "FLAT")
        meta = analysis.get("meta", {})
        regime = meta.get("vol_regime", "Normal").replace("_", " ")
        
        if score >= 70:
            score_class = "text-emerald-400"
            score_bg = "bg-emerald-500/10 border-emerald-500/20"
        elif score <= 30:
            score_class = "text-red-400"
            score_bg = "bg-red-500/10 border-red-500/20"
        else:
            score_class = "text-blue-400"
            score_bg = "bg-blue-500/10 border-blue-500/20"

        if direction == "UP":
            trend_badge = "bg-emerald-500/10 text-emerald-400 border-emerald-500/20"
        elif direction == "DOWN":
            trend_badge = "bg-red-500/10 text-red-400 border-red-500/20"
        else:
            trend_badge = "bg-slate-800 text-slate-400 border-slate-700"

        return f"""
        <div class="space-y-4 font-light">
            <div class="flex flex-wrap items-center gap-2 mb-3">
                <span class="px-2 py-1 rounded text-[10px] font-bold uppercase tracking-wider bg-slate-800 text-slate-400 border border-slate-700">
                    {context}
                </span>
                <span class="px-2 py-1 rounded text-[10px] font-bold uppercase tracking-wider border {trend_badge}">
                    {direction} TREND
                </span>
                <span class="px-2 py-1 rounded text-[10px] font-bold uppercase tracking-wider border {score_bg} {score_class}">
                    SCORE: {score}
                </span>
            </div>
            
            <div class="text-slate-300 text-sm leading-relaxed">
                {detailed_insight_html}
            </div>
        </div>
        """

    @classmethod
    def _call_openai(cls, symbol: str, data: Dict[str, Any], market_context: str) -> str:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            return cls._fallback_logic(symbol, data, market_context)

        system_prompt = "You are a Senior Technical Analyst at a top Mumbai proprietary trading desk. Write concise, professional notes."
        
        user_prompt = f"""
        Analyze {symbol} ({market_context}).
        
        METRICS:
        - Trend: {data.get('direction')} (Score: {data.get('score')})
        - Levels: Entry {data.get('entry')}, Target {data.get('target')}, Stop {data.get('stop')}
        - Factors: {json.dumps(data.get('factors', {}))}
        
        Write a 3 sentence commentary:
        1. Structure: Bullish/Bearish/Sideways? Why?
        2. Execution: Buy Dips? Fade Rallies? Breakout watch?
        3. Risk: Key level to watch.
        
        HTML FORMAT:
        Use <span class="text-emerald-400 font-bold"> for bullish words, <span class="text-red-400 font-bold"> for bearish.
        """

        try:
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                temperature=0.3, max_tokens=200
            )
            return response.choices[0].message.content.replace("```html", "").replace("```", "")
        except Exception:
            return cls._fallback_logic(symbol, data, market_context)

    @staticmethod
    def _fallback_logic(symbol, data, context):
        score = data.get("score", 50)
        target = data.get("target", 0)
        if score > 60:
            return f"Bullish structure detected in {context}. Look for entries near support targeting <span class='text-emerald-400'>{target}</span>."
        elif score < 40:
            return f"Bearish pressure dominant in {context}. Fade rallies targeting <span class='text-emerald-400'>{target}</span>."
        return "Market is sideways. Wait for a clear breakout."

    @classmethod
    def generate_equity_commentary(cls, symbol, analysis):
        return cls._render_html(symbol, analysis, "Indian Equity", cls._call_openai(symbol, analysis, "NSE Equity"))

    @classmethod
    def generate_futures_commentary(cls, symbol, analysis):
        return cls._render_html(symbol, analysis, "NFO Futures", cls._call_openai(symbol, analysis, "Index/Stock Futures"))

    @classmethod
    def generate_options_commentary(cls, symbol, analysis):
        return cls._render_html(symbol, analysis, "NFO Options", cls._call_openai(symbol, analysis, "Options Premium Action"))

    @classmethod
    def generate_crypto_commentary(cls, symbol, analysis):
        return cls._render_html(symbol, analysis, "Crypto (INR)", cls._call_openai(symbol, analysis, "Crypto Assets"))