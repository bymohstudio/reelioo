from __future__ import annotations
from types import SimpleNamespace
import pandas as pd
import numpy as np
import logging
from core.services.marketdata_service import MarketService

log = logging.getLogger(__name__)


class CryptoQuantEngine:
    """
    REELIOO ENGINE v35 – INSTITUTIONAL BALANCED CORE

    Philosophy:
    -----------
    - No perfect signals → only probabilistic edge
    - Confluence scoring replaces rigid filters
    - Order flow is enhancer, not blocker
    - Designed for REAL market behavior (not theory)
    """

    def __init__(self):
        self.ATR_LEN = 14
        self.MASS_LEN = 20
        self.BASE_RISK = 0.012
        self.MAX_STRETCH = 0.06

    def analyze(self, df: pd.DataFrame, trade_style="INTRADAY", market_context=None, symbol=None):
        price = 0.0

        try:
            if len(df) < 210:
                return self._neutral(0, "Initializing Data...")

            df = df.copy()

            close = df["close"]
            high = df["high"]
            low = df["low"]
            volume = df["volume"]
            open_p = df["open"]

            price = float(close.iloc[-1])

            # ==========================================================
            # 1. CONFIG (ADAPTIVE)
            # ==========================================================
            if trade_style == "SCALP":
                struct_len = 50
                min_strength = 0.65
                threshold = 2.2
                stop_mult = 1.0
            else:
                struct_len = 200
                min_strength = 0.85
                threshold = 2.6
                stop_mult = 1.4

            # ==========================================================
            # 2. VOLATILITY + MOMENTUM
            # ==========================================================
            tr = pd.concat([
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs()
            ], axis=1).max(axis=1)

            sigma = tr.rolling(self.ATR_LEN).mean()
            current_sigma = float(sigma.iloc[-1])

            velocity = close.diff() / (sigma + 1e-9)
            mass = volume / (volume.rolling(self.MASS_LEN).mean() + 1e-9)

            trend_strength = (mass * velocity).ewm(span=4).mean()
            strength_now = float(trend_strength.iloc[-1])

            # ==========================================================
            # 3. STRUCTURE + STRETCH
            # ==========================================================
            eq = close.ewm(span=struct_len).mean().iloc[-1]
            stretch_pct = (price - eq) / eq

            bull_struct = price > eq
            bear_struct = price < eq

            overstretched = abs(stretch_pct) > self.MAX_STRETCH

            # ==========================================================
            # 4. DIVERGENCE
            # ==========================================================
            price_change = close.diff(5).iloc[-1]
            strength_change = trend_strength.diff(5).iloc[-1]

            bullish_div = price_change < 0 and strength_change > 0
            bearish_div = price_change > 0 and strength_change < 0

            # ==========================================================
            # 5. ORDER BOOK (SOFT EDGE)
            # ==========================================================
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

                        if obi_score > 0.1:
                            smart_money_bias = "BULLISH"
                        elif obi_score < -0.1:
                            smart_money_bias = "BEARISH"
                except:
                    pass

            # ==========================================================
            # 6. TRAP FILTER (ONLY EXTREME)
            # ==========================================================
            candle_range = high.iloc[-1] - low.iloc[-1]

            wick_u = (high.iloc[-1] - max(close.iloc[-1], open_p.iloc[-1])) / (candle_range + 1e-9)
            wick_l = (min(close.iloc[-1], open_p.iloc[-1]) - low.iloc[-1]) / (candle_range + 1e-9)

            is_trap = wick_u > 0.6 or wick_l > 0.6

            # ==========================================================
            # 7. 🧠 CONFLUENCE ENGINE
            # ==========================================================
            long_score = 0
            short_score = 0

            # Structure (base)
            if bull_struct: long_score += 1
            if bear_struct: short_score += 1

            # Momentum (core driver)
            if strength_now > min_strength: long_score += 1.2
            if strength_now < -min_strength: short_score += 1.2

            # Divergence (high alpha)
            if bullish_div: long_score += 1.5
            if bearish_div: short_score += 1.5

            # Order flow (booster only)
            if smart_money_bias == "BULLISH": long_score += 0.8
            if smart_money_bias == "BEARISH": short_score += 0.8

            # Stretch penalty (avoid late entries)
            if overstretched:
                if stretch_pct > 0:
                    long_score -= 1
                else:
                    short_score -= 1

            # Trap penalty
            if is_trap:
                long_score -= 1
                short_score -= 1

            # ==========================================================
            # 8. DECISION LOGIC
            # ==========================================================
            bias = "HOLD"
            lane = "⚫ CONSOLIDATION"
            score = 50

            if long_score >= threshold and long_score > short_score:
                bias = "LONG"
                lane = "🔥 TREND FOLLOWING"
                score = int(75 + min(long_score * 5, 20))

            elif short_score >= threshold and short_score > long_score:
                bias = "SHORT"
                lane = "🔥 TREND FOLLOWING"
                score = int(75 + min(short_score * 5, 20))

            elif max(long_score, short_score) > 1.6:
                bias = "WATCH"
                lane = "👀 BUILDING PRESSURE"
                score = 65

            # ==========================================================
            # 9. TARGET SYSTEM (REALISTIC RR)
            # ==========================================================
            stop = t1 = t2 = t3 = 0.0
            rr = 0.0
            risk_pct = 0.0

            if bias in ["LONG", "SHORT"]:
                direction = 1 if bias == "LONG" else -1

                stop_dist = current_sigma * stop_mult

                stop = price - (direction * stop_dist)
                t1 = price + (direction * stop_dist * 1.8)
                t2 = price + (direction * stop_dist * 3.0)
                t3 = price + (direction * stop_dist * 5.5)

                rr = 2.0
                risk_pct = self.BASE_RISK

            # ==========================================================
            # 10. MARKET REGIME
            # ==========================================================
            regime = "RANGE"
            if abs(strength_now) > 1:
                regime = "TREND"
            elif abs(strength_now) < 0.5:
                regime = "CONSOLIDATION"

            regime_color = "gray"
            if bias == "LONG":
                regime_color = "green"
            elif bias == "SHORT":
                regime_color = "red"
            elif bias == "WATCH":
                regime_color = "violet"

            # ==========================================================
            # 11. FEATURES (UI SAFE)
            # ==========================================================
            top_features = []

            if bullish_div:
                top_features.append({"desc": "Bullish Divergence", "importance": 95})
            elif bearish_div:
                top_features.append({"desc": "Bearish Divergence", "importance": 95})

            if abs(strength_now) > 0.8:
                desc = "Strong Momentum Up" if strength_now > 0 else "Strong Momentum Down"
                top_features.append({"desc": desc, "importance": 85})

            if smart_money_bias != "NEUTRAL":
                desc = "Order Flow Support" if smart_money_bias == "BULLISH" else "Order Flow Resistance"
                top_features.append({"desc": desc, "importance": 80})

            if overstretched:
                top_features.append({"desc": "Overextended Move", "importance": 88})

            if not top_features:
                top_features.append({"desc": "Neutral Market", "importance": 50})

            top_features = top_features[:3]

            # ==========================================================
            # FINAL OUTPUT (UNCHANGED STRUCTURE)
            # ==========================================================
            return SimpleNamespace(
                bias=bias,
                lane=lane,
                score=score,
                price=price,
                entry=price if bias in ["LONG", "SHORT"] else 0.0,
                stop=round(stop, 4),
                target1=round(t1, 4),
                target2=round(t2, 4),
                target3=round(t3, 4),
                rr_ratio=round(rr, 2),
                risk_pct=round(risk_pct * 100, 2),
                regime=regime,
                regime_color=regime_color,
                whale_state="ACTIVE" if mass.iloc[-1] > 1.5 else "BASELINE",
                whale_label="HIGH" if mass.iloc[-1] > 2 else "NORMAL",
                top_features=top_features,
                narrative="Tradeable Edge Detected" if bias != "HOLD" else "Waiting for Alignment",
                lifecycle="ACTIVE" if bias != "HOLD" else "WAITING"
            )

        except Exception as e:
            log.error(f"Engine Crash: {e}")
            return self._neutral(price, "System Error")

    def _neutral(self, price, reason):
        return SimpleNamespace(
            bias="HOLD",
            lane="⚫ FLAT",
            score=50,
            price=price,
            entry=0,
            stop=0,
            target1=0,
            target2=0,
            target3=0,
            rr_ratio=0,
            risk_pct=0,
            regime="NEUTRAL",
            regime_color="gray",
            whale_state="BASELINE",
            whale_label="---",
            top_features=[{"desc": "Initializing...", "importance": 10}],
            narrative=reason,
            lifecycle="WAITING"
        )