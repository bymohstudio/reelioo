# core/quant/ml_training/feature_engineering.py

from __future__ import annotations
import numpy as np
import pandas as pd

# MASTER LIST: Must match indicators.py
FEATURES = [
    "open", "high", "low", "close", "volume",
    "ret_1", "ret_3", "ret_5", "ret_10",
    "vol_10", "vol_20",
    "ma_10", "ma_20", "ma_50",
    "close_over_ma20", "close_over_ma50",
    "atr_14", "rsi_14",
    "body_to_range", "upper_wick", "lower_wick",
    "vol_zscore_20"
]


class FeatureEngineering:
    DEFAULT_LOOKAHEAD = 5

    @classmethod
    def build(cls, raw_df: pd.DataFrame, horizon: int | None = None) -> pd.DataFrame:
        if raw_df is None or raw_df.empty: return pd.DataFrame()

        horizon = horizon or cls.DEFAULT_LOOKAHEAD
        df = cls._normalize_ohlc(raw_df)
        if df.empty: return pd.DataFrame()

        # 1. Generate
        df = cls._add_features(df)

        # 2. Target
        if "close" in df.columns:
            future_close = df["close"].shift(-horizon)
            df["target"] = (future_close > df["close"]).astype(int)

        # 3. Filter
        cols = [f for f in FEATURES if f in df.columns]
        if "target" in df.columns: cols.append("target")

        return df[cols].dropna()

    @staticmethod
    def _normalize_ohlc(raw_df: pd.DataFrame) -> pd.DataFrame:
        df = raw_df.copy()
        # Flatten
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [str(c[0]).lower() if isinstance(c, tuple) else str(c).lower() for c in df.columns]
        else:
            df.columns = [str(c).lower() for c in df.columns]

        # Rename
        rename = {}
        for c in df.columns:
            if "open" in c:
                rename[c] = "open"
            elif "high" in c:
                rename[c] = "high"
            elif "low" in c:
                rename[c] = "low"
            elif "close" in c:
                rename[c] = "close"
            elif "vol" in c:
                rename[c] = "volume"
        df = df.rename(columns=rename)

        # Numeric
        for c in ["open", "high", "low", "close", "volume"]:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    @classmethod
    def _add_features(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        close = df["close"]

        df["ret_1"] = close.pct_change(1)
        df["ret_3"] = close.pct_change(3)
        df["ret_5"] = close.pct_change(5)
        df["ret_10"] = close.pct_change(10)

        df["vol_10"] = df["ret_1"].rolling(10).std()
        df["vol_20"] = df["ret_1"].rolling(20).std()

        df["ma_10"] = close.rolling(10).mean()
        df["ma_20"] = close.rolling(20).mean()
        df["ma_50"] = close.rolling(50).mean()

        df["close_over_ma20"] = (close / df["ma_20"]) - 1
        df["close_over_ma50"] = (close / df["ma_50"]) - 1

        if "high" in df.columns and "low" in df.columns:
            tr1 = df["high"] - df["low"]
            tr2 = (df["high"] - close.shift(1)).abs()
            tr3 = (df["low"] - close.shift(1)).abs()
            df["atr_14"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()

            rng = (df["high"] - df["low"]).replace(0, np.nan)
            df["body_to_range"] = (close - df["open"]).abs() / rng
            df["upper_wick"] = (df["high"] - df[["open", "close"]].max(axis=1)) / rng
            df["lower_wick"] = (df[["open", "close"]].min(axis=1) - df["low"]) / rng

        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1e-9)
        df["rsi_14"] = 100 - (100 / (1 + rs))

        if "volume" in df.columns:
            df["vol_zscore_20"] = (df["volume"] - df["volume"].rolling(20).mean()) / df["volume"].rolling(20).std()

        return df