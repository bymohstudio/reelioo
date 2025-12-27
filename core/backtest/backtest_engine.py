import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import os
import logging
import traceback
import numpy as np
from core.quant.ml_training.feature_engineering import generate_features, FEATURES

log = logging.getLogger(__name__)


class CryptoBacktestEngine:
    def __init__(self, df, symbol):
        self.df = df
        self.symbol = symbol
        self.trades = []
        self.models = {}

        # --- PATH SETUP ---
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            core_dir = os.path.dirname(current_dir)
            self.MODEL_DIR = os.path.join(core_dir, "quant", "ml_models")
        except Exception as e:
            log.error(f"Path Error: {e}")
            self.MODEL_DIR = ""

        self.PATHS = {
            "xgb_long": os.path.join(self.MODEL_DIR, "xgb_long.json"),
            "xgb_short": os.path.join(self.MODEL_DIR, "xgb_short.json"),
            "lgb_long": os.path.join(self.MODEL_DIR, "lgb_long.txt"),
            "lgb_short": os.path.join(self.MODEL_DIR, "lgb_short.txt"),
            "cat_long": os.path.join(self.MODEL_DIR, "cat_long.cbm"),
            "cat_short": os.path.join(self.MODEL_DIR, "cat_short.cbm"),
        }
        self._load_models()

    def _load_models(self):
        try:
            if not os.path.exists(self.MODEL_DIR):
                return

            if os.path.exists(self.PATHS['xgb_long']):
                self.models['xgb_long'] = xgb.Booster(model_file=self.PATHS['xgb_long'])
                self.models['xgb_short'] = xgb.Booster(model_file=self.PATHS['xgb_short'])

            if os.path.exists(self.PATHS['lgb_long']):
                self.models['lgb_long'] = lgb.Booster(model_file=self.PATHS['lgb_long'])
                self.models['lgb_short'] = lgb.Booster(model_file=self.PATHS['lgb_short'])

            if os.path.exists(self.PATHS['cat_long']):
                self.models['cat_long'] = CatBoostClassifier()
                self.models['cat_long'].load_model(self.PATHS['cat_long'])
                self.models['cat_short'] = CatBoostClassifier()
                self.models['cat_short'].load_model(self.PATHS['cat_short'])

        except Exception as e:
            log.error(f"Backtest Model Load Error: {e}")

    def run(self, trade_style="INTRADAY"):
        try:
            if self.df is None or self.df.empty:
                return self._empty_result()

            # 1. Feature Engineering
            try:
                df = generate_features(self.df.copy())
            except Exception as e:
                print(f"❌ Feature Engineering Failed: {e}")
                return self._empty_result()

            if not self.models or 'xgb_long' not in self.models:
                return self._empty_result()

            # 2. Config & Multipliers
            # Updated Multipliers to match "Sniper" Logic
            stop_mult = 1.5
            tgt_mult = 2.5

            if trade_style == "SCALP": stop_mult, tgt_mult = 1.0, 1.5
            if trade_style == "SWING": stop_mult, tgt_mult = 2.0, 4.0

            # 3. Batch Predict (Ensemble Logic)
            try:
                X = df[FEATURES].astype(float)
                dmat = xgb.DMatrix(X)

                # XGBoost
                xl = self.models['xgb_long'].predict(dmat)
                xs = self.models['xgb_short'].predict(dmat)

                # LightGBM
                if 'lgb_long' in self.models:
                    ll = self.models['lgb_long'].predict(X)
                    ls = self.models['lgb_short'].predict(X)
                else:
                    ll, ls = xl, xs

                # CatBoost
                if 'cat_long' in self.models:
                    cl = self.models['cat_long'].predict_proba(X)[:, 1]
                    cs = self.models['cat_short'].predict_proba(X)[:, 1]
                else:
                    cl, cs = xl, xs

                # Ensemble Average (Weights: 33/33/33)
                df['ens_long'] = ((xl + ll + cl) / 3) * 100
                df['ens_short'] = ((xs + ls + cs) / 3) * 100

            except Exception as e:
                print(f"❌ Prediction Error: {e}")
                return self._empty_result()

            # 4. Simulation Loop
            position = None
            entry_price = 0
            stop_loss = 0
            take_profit = 0

            # Fee Simulation (0.05% Maker/Taker avg per side = 0.1% round trip)
            TRADING_FEE_PCT = 0.1

            start_idx = 50  # Skip warmup period

            for i in range(start_idx, len(df)):
                curr = df.iloc[i]
                price = curr['close']

                # Calculate ATR for dynamic stops
                atr = curr.get('atr_14', price * 0.01)

                # --- EXIT LOGIC ---
                if position:
                    res = None
                    raw_pnl = 0

                    if position == 'LONG':
                        if curr['low'] <= stop_loss:
                            res = "LOSS"
                            # Calculate exact loss based on SL hit
                            exit_price = stop_loss
                            raw_pnl = (exit_price - entry_price) / entry_price * 100
                        elif curr['high'] >= take_profit:
                            res = "WIN"
                            exit_price = take_profit
                            raw_pnl = (exit_price - entry_price) / entry_price * 100

                    elif position == 'SHORT':
                        if curr['high'] >= stop_loss:
                            res = "LOSS"
                            exit_price = stop_loss
                            raw_pnl = (entry_price - exit_price) / entry_price * 100
                        elif curr['low'] <= take_profit:
                            res = "WIN"
                            exit_price = take_profit
                            raw_pnl = (entry_price - exit_price) / entry_price * 100

                    if res:
                        # DEDUCT FEES (The Reality Check)
                        net_pnl = raw_pnl - TRADING_FEE_PCT

                        self.trades.append({
                            "result": res,
                            "pnl": round(net_pnl, 2),
                            "entry": entry_price
                        })
                        position = None
                        continue

                # --- ENTRY LOGIC (Sniper Rules) ---
                if not position:

                    # 1. SAFETY VALVE: Volatility Check
                    # Matches your Cron/Live Engine logic
                    last_open = float(curr['open'])
                    last_close = float(curr['close'])
                    move_pct = abs(last_close - last_open) / last_open
                    if move_pct < 0.002:  # Ignore dead candles (<0.2%)
                        continue

                    # 2. Get Scores
                    p_l = curr.get('ens_long', 0)
                    p_s = curr.get('ens_short', 0)

                    # 3. Context Filters
                    rsi = curr.get('rsi_14', 50)
                    vwap_dist = curr.get('vwap_dist', 0)

                    bias = "HOLD"

                    # 4. Thresholds (UPDATED: Matches Retraining)
                    # Use 70% as the "High Conviction" line
                    CONF_THRESH = 70.0

                    # --- LONG LOGIC ---
                    if p_l > CONF_THRESH:
                        # RSI Filter: Don't buy overbought
                        if rsi < 75:
                            # Value Filter: Don't buy if price is >2% above VWAP
                            if vwap_dist < 0.02:
                                bias = "LONG"

                    # --- SHORT LOGIC ---
                    elif p_s > CONF_THRESH:
                        # RSI Filter: Don't sell oversold
                        if rsi > 25:
                            # Value Filter: Don't sell if price is >2% below VWAP
                            if vwap_dist > -0.02:
                                bias = "SHORT"

                    # Execute Trade
                    if bias == 'LONG':
                        position = 'LONG'
                        entry_price = price
                        stop_loss = price - (atr * stop_mult)
                        take_profit = price + (atr * tgt_mult)

                    elif bias == 'SHORT':
                        position = 'SHORT'
                        entry_price = price
                        stop_loss = price + (atr * stop_mult)
                        take_profit = price - (atr * tgt_mult)

            return self._generate_stats()

        except Exception as e:
            print(f"❌ CRITICAL BACKTEST FAILURE: {e}")
            traceback.print_exc()
            return self._empty_result()

    def _generate_stats(self):
        total = len(self.trades)
        if total == 0: return self._empty_result()

        # Calculate Win Rate
        # Note: A "WIN" in result might still be negative PnL if fee > profit (Breakeven trades)
        wins = [t for t in self.trades if t['pnl'] > 0]
        wr = (len(wins) / total * 100)

        # Calculate Profit Factor
        gross_profit = sum(t['pnl'] for t in wins)
        gross_loss = abs(sum(t['pnl'] for t in self.trades if t['pnl'] < 0))

        pf = (gross_profit / gross_loss) if gross_loss > 0 else 10.0

        return {
            "win_rate": round(wr, 1),
            "profit_factor": round(pf, 2),
            "total_trades": total,
            # Return last 20 trades for UI
            "trades_log": self.trades[-20:]
        }

    def _empty_result(self):
        return {"win_rate": 0, "profit_factor": 0, "total_trades": 0, "trades_log": []}