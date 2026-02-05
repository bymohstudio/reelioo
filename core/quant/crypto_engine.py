from __future__ import annotations
from types import SimpleNamespace
import pandas as pd
import numpy as np
import logging
from core.services.marketdata_service import MarketService

log = logging.getLogger(__name__)


class CryptoQuantEngine:
    """
    REELIOO ENGINE v34 – DUAL CORE (SAFE FOR CRON)

    MODES:
    ------
    1. SWING (Default/Cron): Uses EMA 200. Strict. Catches major trends only.
    2. SCALP (Manual/Hunter): Uses EMA 50. Fast. Catches pumps & reversals.

    This ensures your Discord alerts remain high-quality (Swing),
    while allowing you to hunt volatility in the dashboard (Scalp).
    """

    def __init__(self):
        self.ATR_LEN = 14
        self.MASS_LEN = 20
        self.BASE_RISK = 0.015
        self.MAX_STRETCH = 0.05

    def analyze(self, df: pd.DataFrame, trade_style="INTRADAY", market_context=None, symbol=None):
        price = 0.0

        try:
            # Need history for EMA calculation
            if len(df) < 210: return self._neutral(0, "Initializing Data...")

            df = df.copy()
            close = df["close"]
            high = df["high"]
            low = df["low"]
            volume = df["volume"]
            open_p = df["open"]
            price = float(close.iloc[-1])

            # ===================================================================
            # 1. DYNAMIC CONFIGURATION
            # ===================================================================
            # CRON/SWING Defaults (Strict EMA 200)
            struct_len = 200
            min_strength = 1.1
            watch_strength = 0.6

            # SCALP Override (Fast EMA 50)
            if trade_style == "SCALP":
                struct_len = 50
                min_strength = 0.85
                watch_strength = 0.5

            # ===================================================================
            # 2. MARKET MOMENTUM (RETAIL FRIENDLY)
            # ===================================================================
            tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
            sigma = tr.rolling(self.ATR_LEN).mean()
            current_sigma = float(sigma.iloc[-1])

            # Velocity & Mass -> Strength
            velocity = close.diff() / (sigma + 1e-9)
            mass = volume / (volume.rolling(self.MASS_LEN).mean() + 1e-9)

            trend_strength = (mass * velocity).ewm(span=3).mean()
            strength_now = float(trend_strength.iloc[-1])

            # ===================================================================
            # 3. STRUCTURAL BIAS (DYNAMIC)
            # ===================================================================
            eq = close.ewm(span=struct_len).mean().iloc[-1]
            stretch_pct = (price - eq) / eq

            bull_struct = price > eq
            bear_struct = price < eq

            # ===================================================================
            # 4. ADVANCED CONFLUENCE
            # ===================================================================

            # A. DIVERGENCE (Leading Indicator)
            price_change = close.diff(5).iloc[-1]
            strength_change = trend_strength.diff(5).iloc[-1]

            bullish_divergence = price_change < 0 and strength_change > 0
            bearish_divergence = price_change > 0 and strength_change < 0

            # B. ORDER BOOK FLOW
            obi_score = 0.0
            smart_money_bias = "NEUTRAL"

            if symbol:
                try:
                    data = MarketService.get_order_book_snapshot(symbol)
                    if data:
                        bids = np.array(data['bids'], dtype=float)
                        asks = np.array(data['asks'], dtype=float)
                        bid_vol = np.sum(bids[:20, 1])
                        ask_vol = np.sum(asks[:20, 1])
                        obi_score = (bid_vol - ask_vol) / (bid_vol + ask_vol)

                        if obi_score > 0.15:
                            smart_money_bias = "BULLISH"
                        elif obi_score < -0.15:
                            smart_money_bias = "BEARISH"
                except:
                    pass

            # ===================================================================
            # 5. RISK MANAGEMENT
            # ===================================================================
            candle_range = high.iloc[-1] - low.iloc[-1]
            wick_ratio_u = (high.iloc[-1] - max(close.iloc[-1], open_p.iloc[-1])) / (candle_range + 1e-9)
            wick_ratio_l = (min(close.iloc[-1], open_p.iloc[-1]) - low.iloc[-1]) / (candle_range + 1e-9)

            is_trap = False
            trap_limit = 0.40 if trade_style == "SCALP" else 0.35

            if wick_ratio_u > trap_limit: is_trap = True
            if wick_ratio_l > trap_limit: is_trap = True

            # ===================================================================
            # 6. SIGNAL GENERATION (DUAL CORE)
            # ===================================================================
            bias = "HOLD"
            lane = "⚫ FLAT"
            score = 50

            # LONG SETUP
            if ((bull_struct or bullish_divergence) and
                    strength_now > min_strength and
                    smart_money_bias == "BULLISH" and
                    not is_trap):

                bias = "LONG"
                score = 90 if bullish_divergence else 80
                lane = "⚡ SCALP ENTRY" if trade_style == "SCALP" else "🔥 TREND FOLLOWING"

            # SHORT SETUP
            elif ((bear_struct or bearish_divergence) and
                  strength_now < -min_strength and
                  smart_money_bias == "BEARISH" and
                  not is_trap):

                bias = "SHORT"
                score = 90 if bearish_divergence else 80
                lane = "⚡ SCALP ENTRY" if trade_style == "SCALP" else "🔥 TREND FOLLOWING"

            # WATCH LOGIC (Visual Activity for Users)
            elif abs(strength_now) > watch_strength:
                bias = "WATCH"
                score = 65
                # If momentum is high but structure failed (e.g. Pump below EMA 200 in Swing Mode)
                if abs(strength_now) > min_strength and not (bull_struct or bear_struct):
                    lane = "⚠️ BLOCKED BY TREND"
                else:
                    lane = "👀 MOMENTUM BUILDING"

            else:
                bias = "HOLD"
                lane = "⚫ CONSOLIDATION"

            # =======================================================================
            # 7. TARGETS & OUTPUTS
            # =======================================================================
            stop = t1 = t2 = t3 = 0.0
            rr = 0.0
            risk_pct = 0.0

            if bias in ["LONG", "SHORT"]:
                direction = 1 if bias == "LONG" else -1
                stop_mult = 1.0 if trade_style == "SCALP" else 1.5

                stop_dist = current_sigma * stop_mult
                stop = price - (direction * stop_dist)
                t1 = price + (direction * stop_dist * 2.0)
                t2 = price + (direction * stop_dist * 4.0)
                t3 = price + (direction * stop_dist * 8.0)
                risk_pct = self.BASE_RISK * 1.5
                rr = 2.0

            # =======================================================================
            # 8. VECTORS (Clean Retail Terms)
            # =======================================================================
            regime = "RANGE"
            if abs(strength_now) > 1.0: regime = "TREND"
            if abs(strength_now) < 0.5: regime = "CONSOLIDATION"

            regime_color = "gray"
            if bias == "LONG":
                regime_color = "green"
            elif bias == "SHORT":
                regime_color = "red"
            elif bias == "WATCH":
                regime_color = "violet"

            top_features = []

            def calc_pct(val, min_v, max_v, target_min=60, target_max=99):
                norm = (abs(val) - min_v) / (max_v - min_v)
                norm = max(0.0, min(1.0, norm))
                return int(target_min + (norm * (target_max - target_min)))

            if smart_money_bias != "NEUTRAL":
                desc = "Major Order Book Support" if smart_money_bias == "BULLISH" else "Major Order Book Resistance"
                val = calc_pct(obi_score, 0.15, 0.5, 80, 99)
                top_features.append({"desc": desc, "importance": val})

            if bullish_divergence:
                top_features.append({"desc": "Bullish Volume Divergence", "importance": 95})
            elif bearish_divergence:
                top_features.append({"desc": "Bearish Volume Divergence", "importance": 95})

            if abs(strength_now) > 0.8:
                desc = "Strong Upside Momentum" if strength_now > 0 else "Strong Downside Momentum"
                val = calc_pct(strength_now, 0.8, 3.0, 70, 90)
                top_features.append({"desc": desc, "importance": val})

            if not (bull_struct or bear_struct) and abs(strength_now) > min_strength:
                top_features.append({"desc": "Counter Trend Risk", "importance": 90})

            if not top_features:
                top_features.append({"desc": "Awaiting Order Flow", "importance": 50})

            top_features = top_features[:3]

            vol_scalar = float(mass.iloc[-1])
            whale_label = "NORMAL"
            if vol_scalar > 2.0: whale_label = "HIGH"

            return SimpleNamespace(
                bias=bias, lane=lane, score=score, price=price,
                entry=price if bias in ["LONG", "SHORT"] else 0.0,
                stop=round(stop, 4), target1=round(t1, 4), target2=round(t2, 4), target3=round(t3, 4),
                rr_ratio=round(rr, 2), risk_pct=round(risk_pct * 100, 2),
                regime=regime, regime_color=regime_color,
                whale_state="ACTIVE" if vol_scalar > 1.5 else "BASELINE",
                whale_label=whale_label,
                top_features=top_features,
                narrative="High Probability Setup Detected" if bias != "HOLD" else "Analyzing Market Depth...",
                lifecycle="ACTIVE" if bias != "HOLD" else "WAITING"
            )

        except Exception as e:
            log.error(f"Engine Crash: {e}")
            return self._neutral(price, "System Error")

    def _neutral(self, price, reason):
        return SimpleNamespace(
            bias="HOLD", lane="⚫ FLAT", score=50, price=price,
            entry=0, stop=0, target1=0, target2=0, target3=0, rr_ratio=0, risk_pct=0,
            regime="NEUTRAL", regime_color="gray",
            whale_state="BASELINE", whale_label="---",
            top_features=[{"desc": "Initializing...", "importance": 10}],
            narrative=reason, lifecycle="WAITING"
        )