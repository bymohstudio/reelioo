import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import os
import logging
import traceback
from core.quant.ml_training.feature_engineering import generate_features, FEATURES

log = logging.getLogger(__name__)


class CryptoBacktestEngine:
    def __init__(self, df, symbol):
        self.df = df
        self.symbol = symbol
        self.trades = []
        self.models = {}

        # --- PATH FIX ---
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
            stop_mult = 2.0
            tgt_mult = 3.0
            if trade_style == "SCALP": stop_mult, tgt_mult = 1.0, 1.5
            if trade_style == "SWING": stop_mult, tgt_mult = 2.5, 4.0

            # 3. Batch Predict (Raw Intelligence)
            try:
                X = df[FEATURES].astype(float)
                dmat = xgb.DMatrix(X)
                xl = self.models['xgb_long'].predict(dmat)
                xs = self.models['xgb_short'].predict(dmat)

                if 'lgb_long' in self.models:
                    ll = self.models['lgb_long'].predict(X)
                    ls = self.models['lgb_short'].predict(X)
                else:
                    ll, ls = xl, xs

                if 'cat_long' in self.models:
                    cl = self.models['cat_long'].predict_proba(X)[:, 1]
                    cs = self.models['cat_short'].predict_proba(X)[:, 1]
                else:
                    cl, cs = xl, xs

                df['ens_long'] = ((xl + ll + cl) / 3) * 100
                df['ens_short'] = ((xs + ls + cs) / 3) * 100

            except Exception as e:
                print(f"❌ Prediction Error: {e}")
                return self._empty_result()

            # 4. Simulation Loop (THE CASINO LOGIC)
            position = None
            entry_price = 0
            stop_loss = 0
            take_profit = 0
            trades = []

            # Start after rolling windows (50)
            start_idx = 50

            for i in range(start_idx, len(df)):
                curr = df.iloc[i]
                price = curr['close']
                atr = curr.get('atr_14', price * 0.01)

                # --- EXIT LOGIC ---
                if position:
                    res = None
                    pnl = 0
                    if position == 'LONG':
                        if curr['low'] <= stop_loss:
                            res, pnl = "LOSS", -1.0
                        elif curr['high'] >= take_profit:
                            res, pnl = "WIN", (take_profit - entry_price) / entry_price * 100
                    elif position == 'SHORT':
                        if curr['high'] >= stop_loss:
                            res, pnl = "LOSS", -1.0
                        elif curr['low'] <= take_profit:
                            res, pnl = "WIN", (entry_price - take_profit) / entry_price * 100

                    if res:
                        self.trades.append({"result": res, "pnl": round(pnl, 2), "entry": entry_price})
                        position = None
                        continue

                # --- ENTRY LOGIC (THE HOUSE RULES) ---
                if not position:
                    # 1. Get Raw Scores
                    p_l = curr.get('ens_long', 0)
                    p_s = curr.get('ens_short', 0)

                    # 2. Get Context (The Filter)
                    rsi = curr.get('rsi_14', 50)
                    vwap_dist = curr.get('vwap_dist', 0)
                    liq_sweep = curr.get('liq_sweep', 0)
                    cvd_div = curr.get('cvd_divergence', 0)

                    # 3. Apply "House Rules"
                    bias = "HOLD"
                    CONF_THRESH = 65.0  # Base Threshold

                    # --- LONG ---
                    if p_l > CONF_THRESH:
                        # Rule A: Don't Buy Top
                        if rsi < 70:
                            # Rule B: Value Check (Or Momentum Override)
                            if vwap_dist < 0.02:
                                # BONUS: Trap Boost
                                score = p_l
                                if liq_sweep == 1: score += 5
                                if cvd_div == 1: score += 5

                                # Final Trigger
                                if score >= CONF_THRESH:
                                    bias = "LONG"

                    # --- SHORT ---
                    elif p_s > CONF_THRESH:
                        # Rule A: Don't Short Bottom
                        if rsi > 30:
                            # Rule B: Value Check
                            if vwap_dist > -0.02:
                                # BONUS: Trap Boost
                                score = p_s
                                if liq_sweep == -1: score += 5
                                if cvd_div == -1: score += 5

                                # Final Trigger
                                if score >= CONF_THRESH:
                                    bias = "SHORT"

                    # Execute
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
        wins = [t for t in self.trades if t['result'] == 'WIN']
        wr = (len(wins) / total * 100)

        loss_sum = abs(sum(t['pnl'] for t in self.trades if t['pnl'] < 0))
        pf = (sum(t['pnl'] for t in wins) / loss_sum) if loss_sum > 0 else 10.0

        return {
            "win_rate": round(wr, 1),
            "profit_factor": round(pf, 2),
            "total_trades": total,
            "trades_log": self.trades[-20:]
        }

    def _empty_result(self):
        return {"win_rate": 0, "profit_factor": 0, "total_trades": 0, "trades_log": []}