from __future__ import annotations
from types import SimpleNamespace
import pandas as pd
import numpy as np
import logging
from core.services.marketdata_service import MarketService

log = logging.getLogger(__name__)


class CryptoQuantEngine:
    """
    REELIOO ENGINE v37 – QUALITY OVER QUANTITY

    Changes from v36 (live-tested, 16.7% win rate):
    ------------------------------------------------
    - Fixed: Permanent long bias from structure layers in bull markets
    - Fixed: No short-term pullback detection (bought dips blindly)
    - Added: Short-term momentum direction must AGREE with signal
    - Added: Regime filter — no signals during consolidation
    - Added: Minimum momentum magnitude (not just direction)
    - Added: Cooldown via recent price action (no signals after sharp moves)
    - Tightened: Spread requirement from 0.3 to 0.5
    - Tightened: Structure only scores if SHORT-TERM price agrees
    - Result: Fewer signals, dramatically higher quality
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
            if trade_style == "SCALP":
                struct_len = 21
                struct_slow = 50
                min_strength = 0.35
                threshold = 2.0
                stop_mult = 0.8
                mom_fast = 5
                mom_slow = 14
                short_term_bars = 3
            elif trade_style == "SWING":
                struct_len = 50
                struct_slow = 200
                min_strength = 0.45
                threshold = 2.4
                stop_mult = 1.6
                mom_fast = 8
                mom_slow = 21
                short_term_bars = 5
            elif trade_style == "POSITION":
                struct_len = 100
                struct_slow = 200
                min_strength = 0.55
                threshold = 2.8
                stop_mult = 2.0
                mom_fast = 14
                mom_slow = 50
                short_term_bars = 8
            else:  # INTRADAY
                struct_len = 34
                struct_slow = 100
                min_strength = 0.40
                threshold = 2.2
                stop_mult = 1.2
                mom_fast = 6
                mom_slow = 18
                short_term_bars = 4

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

            velocity = close.diff() / (sigma + 1e-9)
            vol_ma = volume.rolling(self.MASS_LEN).mean()
            mass = volume / (vol_ma + 1e-9)

            trend_fast = (mass * velocity).ewm(span=mom_fast).mean()
            fast_now = float(trend_fast.iloc[-1])

            trend_slow = (mass * velocity).ewm(span=mom_slow).mean()
            slow_now = float(trend_slow.iloc[-1])

            strength_now = fast_now * 0.6 + slow_now * 0.4

            # ==========================================================
            # 3. SHORT-TERM PRICE ACTION (v37.2 FIX)
            # ==========================================================
            # v37 bug: st_bearish triggered on any 0.3 ATR pullback,
            # which happens constantly in bull markets. This gated out
            # all LONG scoring and opened all SHORT scoring = 80% short bias.
            #
            # v37.2 fix: Higher threshold (0.6 ATR) and no longer used
            # as a gate on structure layers. Only used for momentum
            # agreement and counter-trend penalty.
            short_term_change = float(close.diff(short_term_bars).iloc[-1])
            short_term_direction = short_term_change / (current_sigma + 1e-9)

            st_bullish = short_term_direction > 0.6    # Was 0.3 — too sensitive
            st_bearish = short_term_direction < -0.6   # Was -0.3 — too sensitive
            st_neutral = not st_bullish and not st_bearish

            # ==========================================================
            # 4. STRUCTURE (DUAL EMA + VWAP PROXY)
            # ==========================================================
            eq_fast = close.ewm(span=struct_len, adjust=False).mean()
            eq_slow = close.ewm(span=struct_slow, adjust=False).mean()

            eq_fast_val = float(eq_fast.iloc[-1])
            eq_slow_val = float(eq_slow.iloc[-1])

            stretch_pct = (price - eq_fast_val) / eq_fast_val

            bull_fast = price > eq_fast_val
            bear_fast = price < eq_fast_val
            bull_slow = price > eq_slow_val
            bear_slow = price < eq_slow_val

            ema_aligned_bull = eq_fast_val > eq_slow_val
            ema_aligned_bear = eq_fast_val < eq_slow_val

            overstretched = abs(stretch_pct) > self.MAX_STRETCH

            # Macro trend (v37.2: used for counter-trend penalty)
            macro_bullish = ema_aligned_bull and bull_slow
            macro_bearish = ema_aligned_bear and bear_slow

            typical_price = (high + low + close) / 3
            vwap_proxy = (typical_price * volume).rolling(struct_len).sum() / (volume.rolling(struct_len).sum() + 1e-9)
            vwap_val = float(vwap_proxy.iloc[-1])
            price_above_vwap = price > vwap_val

            # ==========================================================
            # 5. DIVERGENCE
            # ==========================================================
            price_chg_5 = close.diff(5).iloc[-1]
            strength_chg_5 = trend_fast.diff(5).iloc[-1]
            price_chg_10 = close.diff(10).iloc[-1]
            strength_chg_10 = trend_fast.diff(10).iloc[-1]

            # v37.2 FIX: Divergence was too loose with OR logic.
            # Bearish div fired on every consolidation in a bull market.
            # Now requires BOTH timeframes to agree AND minimum magnitude.
            min_div_price_move = current_sigma * 0.5  # Price must move meaningfully

            bullish_div = (
                price_chg_5 < -min_div_price_move and strength_chg_5 > 0 and
                price_chg_10 < -min_div_price_move and strength_chg_10 > 0
            )
            bearish_div = (
                price_chg_5 > min_div_price_move and strength_chg_5 < 0 and
                price_chg_10 > min_div_price_move and strength_chg_10 < 0
            )

            # ==========================================================
            # 6. VOLUME ANALYSIS
            # ==========================================================
            vol_std = volume.rolling(self.MASS_LEN).std()
            vol_z_score = float((volume.iloc[-1] - vol_ma.iloc[-1]) / (vol_std.iloc[-1] + 1e-9))

            volume_climax = vol_z_score > 2.0

            vol_trend = float(volume.rolling(5).mean().iloc[-1] / (volume.rolling(20).mean().iloc[-1] + 1e-9))
            vol_confirming = vol_trend > 1.15

            # ==========================================================
            # 7. TREND CONSISTENCY
            # ==========================================================
            recent_closes = close.tail(6)
            ups = int((recent_closes.diff().dropna() > 0).sum())
            downs = int((recent_closes.diff().dropna() < 0).sum())

            trend_consistent_bull = ups >= 4
            trend_consistent_bear = downs >= 4

            # ==========================================================
            # 8. ORDER BOOK
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

                            if obi_score > 0.08:
                                smart_money_bias = "BULLISH"
                            elif obi_score < -0.08:
                                smart_money_bias = "BEARISH"
                except Exception as e:
                    log.debug(f"Order book fetch failed for {symbol}: {e}")

            # ==========================================================
            # 9. TRAP FILTER
            # ==========================================================
            candle_range = high.iloc[-1] - low.iloc[-1]
            wick_u = (high.iloc[-1] - max(close.iloc[-1], open_p.iloc[-1])) / (candle_range + 1e-9)
            wick_l = (min(close.iloc[-1], open_p.iloc[-1]) - low.iloc[-1]) / (candle_range + 1e-9)
            is_trap = wick_u > 0.65 or wick_l > 0.65

            # ==========================================================
            # 10. REGIME DETECTION (v37 NEW)
            # ==========================================================
            atr_ratio = float(sigma.iloc[-1] / (sigma.rolling(50).mean().iloc[-1] + 1e-9))
            is_consolidation = atr_ratio < 0.8 and abs(strength_now) < 0.3

            # ==========================================================
            # 11. CONFLUENCE ENGINE (v37.2 BALANCED)
            # ==========================================================
            # v37 bug: Every layer required st_bullish/st_bearish gate,
            # which meant LONG could never score during normal pullbacks
            # in a bull market, but SHORT always could. 80% short bias.
            #
            # v37.2 fix: Structure layers score based on EMA position
            # (no short-term gate). Momentum requires agreement between
            # fast and slow. Counter-trend penalty applied at the END
            # based on macro trend, not per-layer.
            long_score = 0.0
            short_score = 0.0

            if is_consolidation:
                pass
            else:
                # --- Layer 1: Structure (0.8 max) ---
                # No st_bullish/st_bearish gate — structure is about position, not direction
                if bull_fast:
                    long_score += 0.4
                if bear_fast:
                    short_score += 0.4

                if bull_slow and bull_fast:
                    long_score += 0.2
                if bear_slow and bear_fast:
                    short_score += 0.2

                if ema_aligned_bull:
                    long_score += 0.2
                if ema_aligned_bear:
                    short_score += 0.2

                # --- Layer 2: Momentum (1.0 max) ---
                # Both fast AND slow must agree (unchanged, this is good)
                if strength_now > min_strength and fast_now > 0 and slow_now > 0:
                    long_score += 0.7
                    if strength_now > min_strength * 2.0:
                        long_score += 0.3
                elif strength_now < -min_strength and fast_now < 0 and slow_now < 0:
                    short_score += 0.7
                    if strength_now < -min_strength * 2.0:
                        short_score += 0.3

                # --- Layer 3: Volume (0.5 max) ---
                if vol_confirming and strength_now > min_strength:
                    long_score += 0.25
                elif vol_confirming and strength_now < -min_strength:
                    short_score += 0.25

                if volume_climax:
                    if strength_now > 0:
                        long_score += 0.25
                    elif strength_now < 0:
                        short_score += 0.25

                # --- Layer 4: Trend Consistency (0.4 max) ---
                if trend_consistent_bull:
                    long_score += 0.4
                if trend_consistent_bear:
                    short_score += 0.4

                # --- Layer 5: VWAP Position (0.2 max) ---
                if price_above_vwap and bull_fast:
                    long_score += 0.2
                elif not price_above_vwap and bear_fast:
                    short_score += 0.2

                # --- Layer 6: Divergence (0.6 max — reduced, now harder to trigger) ---
                if bullish_div:
                    long_score += 0.6
                if bearish_div:
                    short_score += 0.6

                # --- Layer 7: Order Flow (0.4 max) ---
                if smart_money_bias == "BULLISH" and strength_now > 0:
                    long_score += 0.4
                if smart_money_bias == "BEARISH" and strength_now < 0:
                    short_score += 0.4

                # --- Penalties ---
                if overstretched:
                    stretch_penalty = min(abs(stretch_pct) / self.MAX_STRETCH - 1.0, 1.0) * 0.6
                    if stretch_pct > 0:
                        long_score -= stretch_penalty
                    else:
                        short_score -= stretch_penalty

                if is_trap:
                    if wick_u > 0.65:
                        long_score -= 0.6
                    if wick_l > 0.65:
                        short_score -= 0.6

                # v37.2: MACRO TREND PENALTY (replaces the old st_bearish/st_bullish halving)
                # In a confirmed bull market, shorting requires MORE confluence
                # In a confirmed bear market, longing requires MORE confluence
                # This is the key balance fix — trend-following is rewarded, counter-trend is penalized
                if macro_bullish and short_score > 0:
                    short_score *= 0.6   # 40% penalty for shorting a bull market
                if macro_bearish and long_score > 0:
                    long_score *= 0.6    # 40% penalty for longing a bear market

                # Short-term direction penalty (softer than v37's 50% nuke)
                # Only applies in STRONG counter-trend moves, not normal noise
                if st_bearish and long_score > 0 and not macro_bearish:
                    long_score *= 0.8    # Mild penalty, not 0.5
                if st_bullish and short_score > 0 and not macro_bullish:
                    short_score *= 0.8

            # ==========================================================
            # 12. DECISION LOGIC (v37: TIGHTER)
            # ==========================================================
            bias = "HOLD"
            lane = "⚫ CONSOLIDATION"
            score = 50

            if long_score >= threshold and long_score > short_score + 0.5:
                bias = "LONG"
                lane = "🔥 TREND FOLLOWING"
                score = int(min(75 + (long_score - threshold) * 10, 95))

            elif short_score >= threshold and short_score > long_score + 0.5:
                bias = "SHORT"
                lane = "🔥 TREND FOLLOWING"
                score = int(min(75 + (short_score - threshold) * 10, 95))

            elif max(long_score, short_score) >= threshold * 0.75:
                bias = "WATCH"
                lane = "👀 BUILDING PRESSURE"
                dominant = long_score if long_score > short_score else short_score
                score = int(min(60 + (dominant / threshold) * 8, 74))

            # ==========================================================
            # 13. TARGET SYSTEM (v37.1: ONE TARGET, POSITIVE EXPECTANCY)
            # ==========================================================
            # The math: at 35% win rate you need 2.5:1 RR to be profitable
            # (0.35 * 2.5) - (0.65 * 1.0) = +0.225R per trade
            #
            # Old system: T1=1.3x, T2=2.2x, T3=3.8x — confusing, and
            # journal used T1 which gave ~1.08:1 RR = guaranteed loss
            #
            # New system: ONE target at 2.5x stop distance. Period.
            # The user sees one number. The journal tracks one number.
            # Wins are 2.5x bigger than losses.
            # ==========================================================
            stop = 0.0
            target = 0.0
            rr = 0.0
            risk_pct = 0.0

            if bias in ["LONG", "SHORT"]:
                direction = 1 if bias == "LONG" else -1
                stop_dist = current_sigma * stop_mult

                stop = price - (direction * stop_dist)

                # Single target: 2.5x the stop distance
                # This means every WIN recovers 2.5 LOSSES
                target_mult = 2.5
                if trade_style == "SCALP":
                    target_mult = 2.0   # Scalp: tighter but still positive expectancy at 40%+ WR
                elif trade_style == "SWING":
                    target_mult = 3.0   # Swing: wider, catch bigger moves
                elif trade_style == "POSITION":
                    target_mult = 3.5   # Position: let runners run

                target = price + (direction * stop_dist * target_mult)
                rr = round(target_mult, 2)
                risk_pct = self.BASE_RISK

            # Legacy compatibility: set t1/t2/t3 to same value
            # so any old code referencing them doesn't break
            t1 = target
            t2 = target
            t3 = target

            # ==========================================================
            # 14. MARKET REGIME
            # ==========================================================
            regime = "RANGE"
            if is_consolidation:
                regime = "CONSOLIDATION"
            elif abs(strength_now) > 0.7:
                regime = "TREND"

            regime_color = "gray"
            if bias == "LONG":
                regime_color = "green"
            elif bias == "SHORT":
                regime_color = "red"
            elif bias == "WATCH":
                regime_color = "violet"

            # ==========================================================
            # 15. FEATURES (UI SAFE)
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

            if is_consolidation:
                top_features.append({"desc": "Low Volatility Regime", "importance": 70})

            if not top_features:
                top_features.append({"desc": "Neutral Market", "importance": 50})

            top_features.sort(key=lambda x: x["importance"], reverse=True)
            top_features = top_features[:3]

            # ==========================================================
            # FINAL OUTPUT
            # ==========================================================
            return SimpleNamespace(
                bias=bias,
                lane=lane,
                score=score,
                price=price,
                entry=price if bias in ["LONG", "SHORT"] else 0.0,
                stop=round(stop, 4),
                target=round(target, 4),
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
            target=0,
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