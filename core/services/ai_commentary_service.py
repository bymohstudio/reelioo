import os
import json
import logging
from typing import Dict, Any

from openai import OpenAI

logger = logging.getLogger(__name__)


class AICommentary:
    """
    AI desk note & explainer on top of Quant + ML engines.

    IMPORTANT:
    - This layer is descriptive only.
    - It must NEVER modify numbers, levels or trading logic.
    - All numeric fields come from QuantResult / SignalResult.
    """

    # ------------------------------------------------------------------
    # HTML RENDERER
    # ------------------------------------------------------------------
    @staticmethod
    def _render_html(symbol, analysis: Dict[str, Any], context: str, detailed_insight_html: str) -> str:
        """
        Build a compact institutional-style card with pills and AI note.

        `analysis` is expected to be a dict-like view of QuantResult.
        """
        # Defensive .get() usage – if anything is missing, we fall back
        score = float(analysis.get("score", 50) or 50)
        direction = str(analysis.get("direction", "NEUTRAL") or "NEUTRAL").upper()
        ml_edge = float(analysis.get("ml_edge", 50) or 50)
        trend_score = float(analysis.get("trend_score", 0) or 0)
        vol_regime = str(analysis.get("volatility_regime", "NORMAL") or "NORMAL").upper()
        grade = str(analysis.get("signal_quality", "C") or "C").upper()

        extras = analysis.get("extras") or {}
        opt_strategy = extras.get("opt_strategy") or extras.get("strategy") or ""
        opt_instrument = extras.get("opt_instrument") or extras.get("instrument") or ""
        theta_risk = extras.get("theta_risk") or "-"
        sentiment = str(extras.get("sentiment") or direction).upper()

        # Direction pill styling
        if direction == "BUY":
            trend_badge = "border-emerald-500/40 bg-emerald-500/10 text-emerald-300"
            dir_label = "BULLISH BIAS"
        elif direction == "SELL":
            trend_badge = "border-red-500/40 bg-red-500/10 text-red-300"
            dir_label = "BEARISH BIAS"
        else:
            trend_badge = "border-slate-500/40 bg-slate-500/10 text-slate-200"
            dir_label = "NEUTRAL / NO-TRADE"

        # Score pill
        if score >= 70:
            score_class = "text-emerald-300"
            score_bg = "bg-emerald-500/10 border-emerald-500/30"
            score_label = "HIGH CONVICTION"
        elif score <= 35:
            score_class = "text-red-300"
            score_bg = "bg-red-500/10 border-red-500/30"
            score_label = "LOW CONFIDENCE"
        else:
            score_class = "text-amber-300"
            score_bg = "bg-amber-500/10 border-amber-500/30"
            score_label = "MIXED / TACTICAL"

        # Volatility pill
        if vol_regime == "HIGH":
            vol_class = "bg-purple-500/10 border-purple-500/40 text-purple-300"
        elif vol_regime == "LOW":
            vol_class = "bg-sky-500/10 border-sky-500/40 text-sky-300"
        else:
            vol_class = "bg-slate-800/80 border-slate-600 text-slate-200"

        # ML Edge pill
        edge_side = "BULLISH" if ml_edge >= 50 else "BEARISH"
        edge_strength = abs(ml_edge - 50)
        if edge_strength >= 25:
            edge_class = "bg-blue-500/10 border-blue-500/40 text-blue-300"
            edge_label = "STRONG EDGE"
        elif edge_strength >= 10:
            edge_class = "bg-blue-500/5 border-blue-500/30 text-blue-200"
            edge_label = "Tactical Edge"
        else:
            edge_class = "bg-slate-900/80 border-slate-700 text-slate-300"
            edge_label = "Flat / No Edge"

        # Grade pill
        if grade == "A":
            grade_class = "bg-emerald-500/10 border-emerald-500/40 text-emerald-300"
            grade_label = "A GRADE SETUP"
        elif grade == "B":
            grade_class = "bg-amber-500/10 border-amber-500/40 text-amber-300"
            grade_label = "B GRADE / TACTICAL"
        else:
            grade_class = "bg-slate-900/80 border-slate-700 text-slate-300"
            grade_label = "C GRADE / AVOID RISK"

        is_options = str(analysis.get("market_type", "") or "").upper() == "OPTIONS"

        detailed_html = (detailed_insight_html or "").strip()
        if not detailed_html:
            detailed_html = "<p class='text-slate-400 text-sm'>No AI desk note available. Using rule-based summary only.</p>"

        return f"""\
<div class="space-y-4 font-light">
  <div class="flex flex-wrap items-center gap-2 mb-3">
    <span class="px-2 py-1 rounded-full text-[10px] font-semibold uppercase tracking-[0.2em] bg-slate-900/80 text-slate-300 border border-slate-700/80">
      {context}
    </span>
    <span class="px-2 py-1 rounded-full text-[10px] font-semibold uppercase tracking-[0.2em] {trend_badge}">
      {dir_label}
    </span>
    <span class="px-2 py-1 rounded-full text-[10px] font-semibold uppercase tracking-[0.2em] {score_bg} {score_class}">
      {score_label} · {score:.1f}%
    </span>
    <span class="px-2 py-1 rounded-full text-[10px] font-semibold uppercase tracking-[0.2em] {vol_class}">
      VOL: {vol_regime}
    </span>
    <span class="px-2 py-1 rounded-full text-[10px] font-semibold uppercase tracking-[0.2em] {edge_class}">
      ML EDGE: {ml_edge:.1f}% · {edge_side} · {edge_label}
    </span>
    <span class="px-2 py-1 rounded-full text-[10px] font-semibold uppercase tracking-[0.2em] {grade_class}">
      GRADE {grade}
    </span>
  </div>

  <div class="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs md:text-[13px] text-slate-300">
    <div class="space-y-1">
      <p class="font-semibold text-slate-200 tracking-wider text-[11px] uppercase">
        STRUCTURE &amp; BIAS
      </p>
      <p class="text-slate-300/90">
        <span class="font-mono text-[11px] px-1.5 py-0.5 rounded bg-slate-900/70 border border-slate-700/80 mr-1">
          {symbol}
        </span>
        <span class="font-semibold">{sentiment.title()}</span> regime with
        <span class="font-semibold">{abs(trend_score):.0f}/100</span> trend strength and
        <span class="font-semibold">{vol_regime.title()}</span> volatility.
      </p>
    </div>

    <div class="space-y-1">
      <p class="font-semibold text-slate-200 tracking-wider text-[11px] uppercase">
        EXECUTION LENS
      </p>
      {"<p class='text-slate-300/90'>Options focus: <span class='font-semibold'>" + opt_strategy + "</span> via <span class='font-semibold'>" + opt_instrument + "</span> (theta: " + str(theta_risk) + ").</p>" if is_options and opt_strategy and opt_instrument else "<p class='text-slate-300/90'>Use position sizing and scaling rules – treat this as guidance, not a blind signal.</p>"}
    </div>
  </div>

  <div class="mt-2 border-t border-slate-800/80 pt-3">
    <p class="text-[11px] font-semibold tracking-[0.25em] text-slate-500 uppercase mb-1">
      AI DESK NOTE
    </p>
    <div class="text-slate-300 text-sm leading-relaxed space-y-1">
      {detailed_html}
    </div>
  </div>
</div>
"""  # noqa: E501

    # ------------------------------------------------------------------
    # OPENAI CALL
    # ------------------------------------------------------------------
    @classmethod
    def _call_openai(cls, symbol: str, data: Dict[str, Any], market_context: str) -> str:
        """
        Call OpenAI with strict instructions to avoid hallucinated numbers.

        If anything fails (no key, API error), we fall back to a small
        deterministic template in `_fallback_logic`.
        """
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set – using fallback commentary.")
            return cls._fallback_logic(symbol, data, market_context)

        client = OpenAI(api_key=api_key)

        payload = {
            "symbol": symbol,
            "market_context": market_context,
            "direction": data.get("direction"),
            "score": data.get("score"),
            "ml_edge": data.get("ml_edge"),
            "trend_score": data.get("trend_score"),
            "volatility_regime": data.get("volatility_regime"),
            "signal_quality": data.get("signal_quality"),
            "entry": data.get("entry"),
            "target": data.get("target"),
            "stop": data.get("stop"),
        }

        system_prompt = (
            "You are a senior sell-side technical analyst on an institutional trading desk. "
            "Your job is to explain what the existing QUANT + ML engine is already signalling. "
            "You must NEVER invent numbers, levels, indicators, or probabilities that are not explicitly provided. "
            "Do not guess future prices. Do not mention any indicator by name unless it is implied by the metrics. "
            "Keep the tone calm, professional, and risk-aware. Focus on structure, execution, and risk."
        )

        user_prompt = (
            "Using ONLY the metrics in the JSON below, write a short HTML fragment (2–4 sentences) that:\n"
            "• Describes the current structure (bullish / bearish / sideways) and how strong it is.\n"
            "• Suggests how a disciplined trader might frame execution (e.g., buy dips / fade rallies / wait).\n"
            "• Highlights risk and invalidation without giving trade advice.\n\n"
            "Hard constraints:\n"
            "- Do NOT invent any numeric values, price levels, or probabilities.\n"
            "- If you are unsure, say that the signal is mixed and emphasize risk control.\n"
            "- Output MUST be valid inline HTML using only <p>, <span>, <strong>, <em>, and <br> tags.\n"
            "- Do NOT wrap the output in <html>, <body>, or code fences.\n\n"
            "METRICS JSON:\n"
            + json.dumps(payload, default=float)
        )

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=260,
            )
            content = response.choices[0].message.content or ""
            content = content.replace("```html", "").replace("```", "").strip()
            return content
        except Exception as e:
            logger.exception("OpenAI commentary failed, falling back: %s", e)
            return cls._fallback_logic(symbol, data, market_context)

    # ------------------------------------------------------------------
    # FALLBACK RULE-BASED COMMENTARY
    # ------------------------------------------------------------------
    @staticmethod
    def _fallback_logic(symbol: str, data: Dict[str, Any], context: str) -> str:
        """
        Very small rule-based commentary.

        This is deliberately conservative and uses only the existing metrics.
        """
        score = float(data.get("score", 50) or 50)
        direction = str(data.get("direction", "NEUTRAL") or "NEUTRAL").upper()
        target = data.get("target")
        vol_regime = str(data.get("volatility_regime", "NORMAL") or "NORMAL").upper()

        base = f"{symbol} in {context}: "

        if score >= 65 and direction == "BUY":
            core = "shows a bullish structure with a reasonably strong upside bias."
        elif score >= 65 and direction == "SELL":
            core = "shows a bearish structure with sellers in control."
        elif score <= 35 and direction == "NEUTRAL":
            core = "has very weak directional edge – price action is largely noise."
        else:
            core = "is currently mixed with no clear high-conviction edge."

        vol_line = f" Current volatility regime is {vol_regime.title().lower()}, adjust position sizing accordingly."
        tgt_line = ""
        if target:
            tgt_line = f" Use the provided target level ({target}) as a reference, not a guarantee."

        return (
            "<p class='text-slate-300 text-sm'>"
            + base
            + core
            + vol_line
            + tgt_line
            + "</p>"
        )

    # ------------------------------------------------------------------
    # PUBLIC HELPERS
    # ------------------------------------------------------------------
    @classmethod
    def generate_equity_commentary(cls, symbol, analysis: Dict[str, Any]) -> str:
        note = cls._call_openai(symbol, analysis, "NSE EQUITY")
        return cls._render_html(symbol, analysis, "Indian Equity", note)

    @classmethod
    def generate_futures_commentary(cls, symbol, analysis: Dict[str, Any]) -> str:
        note = cls._call_openai(symbol, analysis, "INDEX / STOCK FUTURES")
        return cls._render_html(symbol, analysis, "NFO Futures", note)

    @classmethod
    def generate_options_commentary(cls, symbol, analysis: Dict[str, Any]) -> str:
        note = cls._call_openai(symbol, analysis, "OPTIONS PREMIUM ACTION")
        return cls._render_html(symbol, analysis, "NFO Options", note)

    @classmethod
    def generate_crypto_commentary(cls, symbol, analysis: Dict[str, Any]) -> str:
        note = cls._call_openai(symbol, analysis, "CRYPTO (INR)")
        return cls._render_html(symbol, analysis, "Crypto (INR)", note)
