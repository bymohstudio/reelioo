from __future__ import annotations
from types import SimpleNamespace
import pandas as pd
import numpy as np
import logging
from core.services.marketdata_service import MarketService

log = logging.getLogger(__name__)


class CryptoQuantEngine:
    """
    REELIOO ENGINE v36 – REBALANCED CONFLUENCE CORE

    Changes from v35:
    -----------------
    - Fixed: Threshold vs max-possible-score mismatch (was mathematically impossible)
    - Fixed: min_strength too high for intraday (0.85 → adaptive per style)
    - Fixed: struct_len 200 on 15m candles = 50hrs of lag (now adaptive)
    - Added: Multi-timeframe momentum (fast + slow)
    - Added: Volume climax detection (not just relative volume)
    - Added: Trend consistency score (not just last-bar direction)
    - Added: Weighted confluence with proper scaling
    - Added: VWAP deviation as additional structure layer
    - Kept: All output fields, variable names, frontend contract identical
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
            # 1. ADAPTIVE CONFIG PER TRADE STYLE
            # ==========================================================
            # v36 FIX: Each style has achievable thresholds
            if trade_style == "SCALP":
                struct_len = 21         # EMA 21 (~1.75hrs on 5m)
                struct_slow = 50        # EMA 50 for trend filter
                min_strength = 0.35     # v35 was 0.65 — too high for 5m
                threshold = 1.8         # v35 was 2.2 — unreachable
                stop_mult = 0.8
                mom_fast = 5
                mom_slow = 14
            elif trade_style == "SWING":
                struct_len = 50         # EMA 50
                struct_slow = 200       # EMA 200 for macro
                min_strength = 0.45
                threshold = 2.2
                stop_mult = 1.6
                mom_fast = 8
                mom_slow = 21
            elif trade_style == "POSITION":
                struct_len = 100
                struct_slow = 200
                min_strength = 0.55
                threshold = 2.5
                stop_mult = 2.0
                mom_fast = 14
                mom_slow = 50
            else:  # INTRADAY (default)
                struct_len = 34         # EMA 34 (~8.5hrs on 15m)
                struct_slow = 100       # EMA 100 for trend filter
                min_strength = 0.40     # v35 was 0.85 — nearly impossible
                threshold = 2.0         # v35 was 2.6 — max possible was 2.2!
                stop_mult = 1.2
                mom_fast = 6
                mom_slow = 18

            # ==========================================================
            # 2. VOLATILITY + MULTI-SPEED MOMENTUM
            # ==========================================================
            tr = pd.concat([
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs()
            ], axis=1).max(axis=1)

            sigma = tr.rolling(self.ATR_LEN).mean()
            current_sigma = float(sigma.iloc[-1])

            # Velocity (normalized price change)
            velocity = close.diff() / (sigma + 1e-9)

            # Mass (relative volume)
            vol_ma = volume.rolling(self.MASS_LEN).mean()
            mass = volume / (vol_ma + 1e-9)

            # --- FAST momentum (reactive) ---
            trend_fast = (mass * velocity).ewm(span=mom_fast).mean()
            fast_now = float(trend_fast.iloc[-1])

            # --- SLOW momentum (confirmed) ---
            trend_slow = (mass * velocity).ewm(span=mom_slow).mean()
            slow_now = float(trend_slow.iloc[-1])

            # Combined strength: fast drives the signal, slow confirms
            strength_now = fast_now * 0.6 + slow_now * 0.4

            # ==========================================================
            # 3. STRUCTURE (DUAL EMA + VWAP PROXY)
            # ==========================================================
            eq_fast = close.ewm(span=struct_len, adjust=False).mean()
            eq_slow = close.ewm(span=struct_slow, adjust=False).mean()

            eq_fast_val = float(eq_fast.iloc[-1])
            eq_slow_val = float(eq_slow.iloc[-1])

            stretch_pct = (price - eq_fast_val) / eq_fast_val

            # Structure states
            bull_fast = price > eq_fast_val
            bear_fast = price < eq_fast_val
            bull_slow = price > eq_slow_val
            bear_slow = price < eq_slow_val

            # EMA alignment (fast above slow = bullish structure)
            ema_aligned_bull = eq_fast_val > eq_slow_val
            ema_aligned_bear = eq_fast_val < eq_slow_val

            overstretched = abs(stretch_pct) > self.MAX_STRETCH

            # VWAP proxy (volume-weighted mean as support/resistance)
            typical_price = (high + low + close) / 3
            vwap_proxy = (typical_price * volume).rolling(struct_len).sum() / (volume.rolling(struct_len).sum() + 1e-9)
            vwap_val = float(vwap_proxy.iloc[-1])

            price_above_vwap = price > vwap_val

            # ==========================================================
            # 4. DIVERGENCE (IMPROVED: Multi-bar check)
            # ==========================================================
            # Check divergence over 5 and 10 bars for robustness
            price_chg_5 = close.diff(5).iloc[-1]
            strength_chg_5 = trend_fast.diff(5).iloc[-1]
            price_chg_10 = close.diff(10).iloc[-1]
            strength_chg_10 = trend_fast.diff(10).iloc[-1]

            bullish_div = (
                (price_chg_5 < 0 and strength_chg_5 > 0) or
                (price_chg_10 < 0 and strength_chg_10 > 0)
            )
            bearish_div = (
                (price_chg_5 > 0 and strength_chg_5 < 0) or
                (price_chg_10 > 0 and strength_chg_10 < 0)
            )

            # ==========================================================
            # 5. VOLUME ANALYSIS (CLIMAX + TREND VOLUME)
            # ==========================================================
            vol_std = volume.rolling(self.MASS_LEN).std()
            vol_z_score = float((volume.iloc[-1] - vol_ma.iloc[-1]) / (vol_std.iloc[-1] + 1e-9))

            # Volume climax: massive volume spike (institutional footprint)
            volume_climax = vol_z_score > 2.0

            # Volume trend: is volume increasing with price?
            vol_trend = float(volume.rolling(5).mean().iloc[-1] / (volume.rolling(20).mean().iloc[-1] + 1e-9))
            vol_confirming = vol_trend > 1.15  # Volume expanding

            # ==========================================================
            # 6. TREND CONSISTENCY (New: checks last N bars direction)
            # ==========================================================
            recent_closes = close.tail(6)
            ups = int((recent_closes.diff().dropna() > 0).sum())
            downs = int((recent_closes.diff().dropna() < 0).sum())

            trend_consistent_bull = ups >= 4  # 4 out of 5 bars up
            trend_consistent_bear = downs >= 4

            # ==========================================================
            # 7. ORDER BOOK (SOFT EDGE — kept but improved)
            # ==========================================================
            obi_score = 0.0
            smart_money_bias = "NEUTRAL"

            if symbol:
                try:
                    data = MarketService.get_order_book_snapshot(symbol)
                    if data:
                        bids = np.array(data['bids'], dtype=float)
                        asks = np.array(data['asks'], dtype=float)

                        if len(bids) >= 20 and len(asks) >= 20:
                            bid_vol = np.sum(bids[:20, 1])
                            ask_vol = np.sum(asks[:20, 1])

                            obi_score = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-9)

                            # Tighter threshold for significance
                            if obi_score > 0.08:
                                smart_money_bias = "BULLISH"
                            elif obi_score < -0.08:
                                smart_money_bias = "BEARISH"
                except Exception as e:
                    log.debug(f"Order book fetch failed for {symbol}: {e}")

            # ==========================================================
            # 8. TRAP FILTER (ONLY EXTREME)
            # ==========================================================
            candle_range = high.iloc[-1] - low.iloc[-1]

            wick_u = (high.iloc[-1] - max(close.iloc[-1], open_p.iloc[-1])) / (candle_range + 1e-9)
            wick_l = (min(close.iloc[-1], open_p.iloc[-1]) - low.iloc[-1]) / (candle_range + 1e-9)

            is_trap = wick_u > 0.65 or wick_l > 0.65

            # ==========================================================
            # 9. CONFLUENCE ENGINE (v36 REBALANCED)
            # ==========================================================
            # Key fix: Scores are achievable. Structure + momentum alone
            # CAN reach threshold without requiring divergence or order flow.
            long_score = 0.0
            short_score = 0.0

            # --- Layer 1: Structure (1.0 max per side) ---
            # Fast EMA alignment
            if bull_fast:
                long_score += 0.5
            if bear_fast:
                short_score += 0.5

            # Slow EMA confirms trend (important layer)
            if bull_slow:
                long_score += 0.3
            if bear_slow:
                short_score += 0.3

            # EMA stack alignment (fast above slow = strong trend)
            if ema_aligned_bull:
                long_score += 0.2
            if ema_aligned_bear:
                short_score += 0.2

            # --- Layer 2: Momentum (1.0 max per side) ---
            if abs(strength_now) > min_strength:
                if strength_now > 0:
                    long_score += 0.7
                else:
                    short_score += 0.7

                # Extra credit for strong momentum
                if abs(strength_now) > min_strength * 1.8:
                    if strength_now > 0:
                        long_score += 0.3
                    else:
                        short_score += 0.3

            # --- Layer 3: Volume (0.6 max per side) ---
            if vol_confirming:
                if strength_now > 0:
                    long_score += 0.3
                elif strength_now < 0:
                    short_score += 0.3

            if volume_climax:
                if strength_now > 0:
                    long_score += 0.3
                elif strength_now < 0:
                    short_score += 0.3

            # --- Layer 4: Trend Consistency (0.4 max) ---
            if trend_consistent_bull:
                long_score += 0.4
            if trend_consistent_bear:
                short_score += 0.4

            # --- Layer 5: VWAP Position (0.3 max) ---
            if price_above_vwap and bull_fast:
                long_score += 0.3
            elif not price_above_vwap and bear_fast:
                short_score += 0.3

            # --- Layer 6: Divergence (high alpha bonus) ---
            if bullish_div:
                long_score += 1.0
            if bearish_div:
                short_score += 1.0

            # --- Layer 7: Order Flow (booster) ---
            if smart_money_bias == "BULLISH":
                long_score += 0.5
            if smart_money_bias == "BEARISH":
                short_score += 0.5

            # --- Penalties ---
            # Stretch penalty (softer: scale by how overstretched)
            if overstretched:
                stretch_penalty = min(abs(stretch_pct) / self.MAX_STRETCH - 1.0, 1.0) * 0.5
                if stretch_pct > 0:
                    long_score -= stretch_penalty
                else:
                    short_score -= stretch_penalty

            # Trap penalty (only if trap is against the signal direction)
            if is_trap:
                if wick_u > 0.65:  # Upper wick = bearish trap
                    long_score -= 0.5
                if wick_l > 0.65:  # Lower wick = bullish trap
                    short_score -= 0.5

            # ==========================================================
            # 10. DECISION LOGIC (v36: ACHIEVABLE THRESHOLDS)
            # ==========================================================
            bias = "HOLD"
            lane = "⚫ CONSOLIDATION"
            score = 50

            # Primary signals
            if long_score >= threshold and long_score > short_score + 0.3:
                bias = "LONG"
                lane = "🔥 TREND FOLLOWING"
                score = int(min(75 + (long_score - threshold) * 12, 95))

            elif short_score >= threshold and short_score > long_score + 0.3:
                bias = "SHORT"
                lane = "🔥 TREND FOLLOWING"
                score = int(min(75 + (short_score - threshold) * 12, 95))

            # Watch zone — pressure building but not yet confirmed
            elif max(long_score, short_score) >= threshold * 0.7:
                bias = "WATCH"
                lane = "👀 BUILDING PRESSURE"
                dominant = long_score if long_score > short_score else short_score
                score = int(min(60 + (dominant / threshold) * 10, 74))

            # ==========================================================
            # 11. TARGET SYSTEM (REALISTIC RR)
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
            # 12. MARKET REGIME
            # ==========================================================
            regime = "RANGE"
            if abs(strength_now) > 0.7:
                regime = "TREND"
            elif abs(strength_now) < 0.3:
                regime = "CONSOLIDATION"

            regime_color = "gray"
            if bias == "LONG":
                regime_color = "green"
            elif bias == "SHORT":
                regime_color = "red"
            elif bias == "WATCH":
                regime_color = "violet"

            # ==========================================================
            # 13. FEATURES (UI SAFE)
            # ==========================================================
            top_features = []

            if bullish_div:
                top_features.append({"desc": "Bullish Divergence", "importance": 95})
            elif bearish_div:
                top_features.append({"desc": "Bearish Divergence", "importance": 95})

            if volume_climax:
                desc = "Volume Climax (Institutional)" if vol_z_score > 3 else "Volume Spike"
                top_features.append({"desc": desc, "importance": 90})

            if abs(strength_now) > min_strength:
                desc = "Strong Momentum Up" if strength_now > 0 else "Strong Momentum Down"
                top_features.append({"desc": desc, "importance": 85})

            if ema_aligned_bull and bull_fast:
                top_features.append({"desc": "EMA Stack Bullish", "importance": 82})
            elif ema_aligned_bear and bear_fast:
                top_features.append({"desc": "EMA Stack Bearish", "importance": 82})

            if smart_money_bias != "NEUTRAL":
                desc = "Order Flow Support" if smart_money_bias == "BULLISH" else "Order Flow Resistance"
                top_features.append({"desc": desc, "importance": 80})

            if vol_confirming:
                top_features.append({"desc": "Volume Expanding", "importance": 75})

            if trend_consistent_bull:
                top_features.append({"desc": "Consistent Uptrend", "importance": 73})
            elif trend_consistent_bear:
                top_features.append({"desc": "Consistent Downtrend", "importance": 73})

            if overstretched:
                top_features.append({"desc": "Overextended Move", "importance": 88})

            if is_trap:
                top_features.append({"desc": "Trap Candle Detected", "importance": 85})

            if not top_features:
                top_features.append({"desc": "Neutral Market", "importance": 50})

            # Sort by importance, take top 3
            top_features.sort(key=lambda x: x["importance"], reverse=True)
            top_features = top_features[:3]

            # ==========================================================
            # FINAL OUTPUT (UNCHANGED STRUCTURE — FRONTEND SAFE)
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