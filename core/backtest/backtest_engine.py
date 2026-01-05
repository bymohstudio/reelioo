# core/quant/backtest_engine.py

import numpy as np
import logging
from core.quant.feature_engineering import generate_features

log = logging.getLogger(__name__)


def cap(x, limit=3.0):
    return max(-limit, min(limit, x))


class CryptoBacktestEngine:
    """
    PHYSICS BACKTEST ENGINE (v5.0 - Aligned with Production)
    """

    def __init__(self, df, symbol):
        self.df = df
        self.symbol = symbol
        self.trades = []
        self.SIGMOID_K = 0.45
        self.CONFIRMATION_THRESH = 70.0

    def _sigmoid(self, x):
        return 100 / (1 + np.exp(-self.SIGMOID_K * x))

    def run(self, trade_style="INTRADAY"):
        try:
            if self.df is None or self.df.empty: return self._empty_result()
            try:
                df = generate_features(self.df.copy())
            except:
                return self._empty_result()

            position = None
            entry_price = 0
            stop_loss = 0
            take_profit = 0
            TRADING_FEE_PCT = 0.1

            thresh = self.CONFIRMATION_THRESH - (5 if trade_style == "SCALP" else 0)
            stop_mult, tgt_mult = (1.0, 1.5) if trade_style == "SCALP" else (1.5, 3.0)
            start_idx = 50

            for i in range(start_idx, len(df)):
                curr = df.iloc[i]
                price = curr['close']
                low = curr['low']
                high = curr['high']

                # --- EXIT LOGIC ---
                if position:
                    res = None
                    if position == 'LONG':
                        if low <= stop_loss:
                            res, exit_price = "LOSS", stop_loss
                        elif high >= take_profit:
                            res, exit_price = "WIN", take_profit
                    elif position == 'SHORT':
                        if high >= stop_loss:
                            res, exit_price = "LOSS", stop_loss
                        elif low <= take_profit:
                            res, exit_price = "WIN", take_profit

                    if res:
                        pnl = (exit_price - entry_price) / entry_price * 100
                        if position == 'SHORT': pnl = -pnl
                        self.trades.append({
                            "result": res,
                            "pnl": round(pnl - TRADING_FEE_PCT, 2),
                            "entry": entry_price,
                            "date": str(curr.name)
                        })
                        position = None
                        continue

                # --- ENTRY LOGIC (V5) ---
                if not position:
                    # 1. Regime & Alphas
                    er = float(curr.get("efficiency_ratio", 0.5))
                    regime = "TRENDING" if er > 0.4 else "CHOPPY"

                    ema_diff = cap(curr.get("ema_diff", 0) * 100)
                    rsi_z = cap((curr.get("rsi_14", 50) - 50) / 15)
                    trend = (ema_diff * 1.5 + rsi_z) * (1.0 if regime == "TRENDING" else 0.5)

                    whale = cap(float(curr.get("whale_z", 0)))

                    vwap = cap(curr.get("vwap_dist", 0) * 100)
                    rev = -vwap * (2.0 if regime == "CHOPPY" else 1.2)

                    evt = 0.0
                    if int(curr.get("liq_sweep", 0)) == 1:
                        evt += 2.0
                    elif int(curr.get("liq_sweep", 0)) == -1:
                        evt -= 2.0
                    if int(curr.get("cvd_divergence", 0)) == 1:
                        evt += 1.5
                    elif int(curr.get("cvd_divergence", 0)) == -1:
                        evt -= 1.5

                    # 2. Probability
                    prob = self._sigmoid(cap(trend) + cap(whale) + cap(rev) + cap(evt))

                    # 3. Dampener
                    if float(curr.get("volatility_slope", 0)) > 0.25 and abs(whale) < 1.0:
                        prob = 50 + (prob - 50) * 0.75

                    # 4. Bias & Score
                    bias, score = ("LONG", prob) if prob > 55 else ("SHORT", 100 - prob) if prob < 45 else ("HOLD", 50)
                    score = 50 + (score - 50) * 0.85

                    if score >= thresh and bias != "HOLD":
                        atr = float(curr.get('atr_14', price * 0.01))
                        direction = 1 if bias == 'LONG' else -1
                        position = bias
                        entry_price = price
                        stop_loss = price - direction * atr * stop_mult
                        take_profit = price + direction * atr * tgt_mult

            return self._generate_stats()

        except:
            return self._empty_result()

    def _generate_stats(self):
        total = len(self.trades)
        if total == 0: return self._empty_result()
        wins = [t for t in self.trades if t['pnl'] > 0]
        gross_profit = sum(t['pnl'] for t in wins)
        gross_loss = abs(sum(t['pnl'] for t in self.trades if t['pnl'] < 0))
        pf = (gross_profit / gross_loss) if gross_loss > 0 else 10.0
        return {
            "win_rate": round(len(wins) / total * 100, 1),
            "profit_factor": round(pf, 2),
            "total_trades": total,
            "trades_log": self.trades[-20:]
        }

    def _empty_result(self):
        return {"win_rate": 0, "profit_factor": 0, "total_trades": 0, "trades_log": []}