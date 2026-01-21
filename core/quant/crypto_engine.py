from __future__ import annotations
from types import SimpleNamespace
import pandas as pd
import numpy as np
import logging
from core.services.marketdata_service import MarketService

log = logging.getLogger(__name__)


class CryptoQuantEngine:
    """
    REELIOO ENGINE v33 – THE ORACLE (RETAIL FRIENDLY OUTPUT)

    CHANGES:
    --------
    1. CLEAN OUTPUT: Removed all brackets '()' and technical jargon from UI text.
    2. SYNTAX FIX: Fixed the indentation of the try/except block.
    3. RETAIL TERMS: Simplified descriptions (e.g., 'Hidden Buy' -> 'Smart Accumulation').
    """

    def __init__(self):
        self.ATR_LEN = 14
        self.MASS_LEN = 20
        self.STRUCT_LEN = 200  # Ironclad Trend Filter
        self.BASE_RISK = 0.015
        self.MAX_STRETCH = 0.04

    def analyze(self, df: pd.DataFrame, trade_style="INTRADAY", market_context=None, symbol=None):
        price = 0.0

        try:
            if len(df) < 210: return self._neutral(0, "Initializing Data...")

            df = df.copy()
            close = df["close"]
            high = df["high"]
            low = df["low"]
            volume = df["volume"]
            open_p = df["open"]
            price = float(close.iloc[-1])

            # ===================================================================
            # 1. PHYSICS VECTORS
            # ===================================================================
            tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
            sigma = tr.rolling(self.ATR_LEN).mean()
            current_sigma = float(sigma.iloc[-1])

            velocity = close.diff() / (sigma + 1e-9)
            vol_mean = volume.rolling(self.MASS_LEN).mean()
            mass = volume / (vol_mean + 1e-9)

            # Weighted Force: Recent data matters more
            force = (mass * velocity).ewm(span=3).mean()
            force_now = float(force.iloc[-1])
            acceleration = force.diff().iloc[-1]

            # ===================================================================
            # 2. STRUCTURAL BIAS (The 200 EMA "King")
            # ===================================================================
            eq = close.ewm(span=self.STRUCT_LEN).mean().iloc[-1]
            stretch_pct = (price - eq) / eq

            bull_struct = price > eq
            bear_struct = price < eq

            # ===================================================================
            # 3. PREDICTIVE LAYERS (THE "FUTURE" CHECK)
            # ===================================================================

            # A. PREDICTIVE DIVERGENCE (CVD Proxy)
            price_change = close.diff(5).iloc[-1]
            force_change = force.diff(5).iloc[-1]

            bullish_divergence = price_change < 0 and force_change > 0  # Whales buying the dip
            bearish_divergence = price_change > 0 and force_change < 0  # Whales selling the rip

            # B. ORDER BOOK INTENT (The "Oracle")
            obi_score = 0.0
            smart_money_bias = "NEUTRAL"

            if symbol:
                try:
                    data = MarketService.get_order_book_snapshot(symbol)
                    if data:
                        bids = np.array(data['bids'], dtype=float)
                        asks = np.array(data['asks'], dtype=float)
                        # Look deeper (top 20 levels) for "True Intent"
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
            # 4. TRAP & WICK PROTECTION
            # ===================================================================
            candle_range = high.iloc[-1] - low.iloc[-1]
            wick_ratio_u = (high.iloc[-1] - max(close.iloc[-1], open_p.iloc[-1])) / (candle_range + 1e-9)
            wick_ratio_l = (min(close.iloc[-1], open_p.iloc[-1]) - low.iloc[-1]) / (candle_range + 1e-9)

            is_trap = False
            if wick_ratio_u > 0.35: is_trap = True  # Rejection from top
            if wick_ratio_l > 0.35: is_trap = True  # Rejection from bottom

            # ===================================================================
            # 5. SIGNAL GENERATION (PREDICTIVE MODE)
            # ===================================================================
            bias = "HOLD"
            lane = "⚫ HOLD"
            score = 50

            MIN_FORCE = 1.2

            # LONG SCENARIO
            if ((bull_struct or bullish_divergence) and
                    force_now > MIN_FORCE and
                    smart_money_bias == "BULLISH" and
                    not is_trap):

                bias = "LONG"
                score = 90 if bullish_divergence else 80
                lane = "🔮 PREDICTION" if bullish_divergence else "🚀 MOMENTUM"

            # SHORT SCENARIO
            elif ((bear_struct or bearish_divergence) and
                  force_now < -MIN_FORCE and
                  smart_money_bias == "BEARISH" and
                  not is_trap):

                bias = "SHORT"
                score = 90 if bearish_divergence else 80
                lane = "🔮 PREDICTION" if bearish_divergence else "🚀 MOMENTUM"

            # COMPRESSION (Watch mode)
            elif abs(force_now) < 0.5:
                bias = "WATCH"
                score = 60
                lane = "🟠 CHARGING"

            # =======================================================================
            # 6. OUTPUTS
            # =======================================================================
            stop = t1 = t2 = t3 = 0.0
            rr = 0.0
            risk_pct = 0.0

            if bias in ["LONG", "SHORT"]:
                direction = 1 if bias == "LONG" else -1
                stop_dist = current_sigma * 1.5
                stop = price - (direction * stop_dist)
                t1 = price + (direction * stop_dist * 2.0)
                t2 = price + (direction * stop_dist * 4.0)
                t3 = price + (direction * stop_dist * 8.0)
                risk_pct = self.BASE_RISK * 1.5
                rr = 2.0

            # =======================================================================
            # 7. VECTORS (Clean Retail Friendly Text)
            # =======================================================================
            regime = "RANGE"
            if abs(force_now) > 1.0: regime = "TREND"
            if abs(force_now) < 0.5: regime = "COMPRESSION"

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

            # 1. Order Book Vector (The Predictor)
            if smart_money_bias != "NEUTRAL":
                desc = "Institutional Buy Walls" if smart_money_bias == "BULLISH" else "Institutional Sell Walls"
                val = calc_pct(obi_score, 0.15, 0.5, 80, 99)
                top_features.append({"desc": desc, "importance": val})

            # 2. Divergence Vector (The Hidden Move)
            if bullish_divergence:
                # REMOVED BRACKETS: "Hidden Buy" -> "Smart Accumulation"
                top_features.append({"desc": "Bullish Divergence Smart Accumulation", "importance": 95})
            elif bearish_divergence:
                # REMOVED BRACKETS: "Hidden Sell" -> "Smart Distribution"
                top_features.append({"desc": "Bearish Divergence Smart Distribution", "importance": 95})

            # 3. Momentum Vector (The Force)
            if abs(force_now) > 1.0:
                desc = "High Velocity Impulse" if force_now > 0 else "High Velocity Dump"
                val = calc_pct(force_now, 1.0, 3.0, 70, 90)
                top_features.append({"desc": desc, "importance": val})

            if not top_features:
                top_features.append({"desc": "Awaiting Smart Money", "importance": 50})

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
                narrative="Predictive Setup Validated" if bias != "HOLD" else "Scanning Order Flow...",
                lifecycle="ACTIVE" if bias != "HOLD" else "WAITING"
            )

        except Exception as e:
            log.error(f"Engine Crash: {e}")
            return self._neutral(price, "System Error")

    def _neutral(self, price, reason):
        return SimpleNamespace(
            bias="HOLD", lane="⚫ HOLD", score=50, price=price,
            entry=0, stop=0, target1=0, target2=0, target3=0, rr_ratio=0, risk_pct=0,
            regime="NEUTRAL", regime_color="gray",
            whale_state="BASELINE", whale_label="---",
            top_features=[{"desc": "Initializing...", "importance": 10}],
            narrative=reason, lifecycle="WAITING"
        )