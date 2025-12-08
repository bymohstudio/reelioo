# core/quant/ml_training/feature_engineering.py

"""
ML 7.0 Feature Engineering for Reelioo

Goal:
- Take raw OHLCV history (from Yahoo / SmartAPI / WazirX)
- Normalize columns (handle MultiIndex from yfinance)
- Build a rich numeric feature set
- Create a realistic binary target: "will price go up in the next N bars?"

Design:
- Safe: returns a clean DataFrame with ONLY numeric feature columns + `target`
- No datetime columns in features (avoids XGBoost dtype errors)
- No dependency on Django or SmartAPI – pure pandas/numpy logic
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


class FeatureEngineering:
    """
    Central feature engineering for all ML models (equity, futures, options, crypto).

    Public API:
        df_features = FeatureEngineering.build(raw_df, horizon=5)

    Returned DataFrame:
        - index: DatetimeIndex (not used by XGBoost)
        - columns: purely numeric features + `target`
    """

    DEFAULT_LOOKAHEAD = 5  # bars/days ahead for label

    @classmethod
    def build(cls, raw_df: pd.DataFrame, horizon: int | None = None) -> pd.DataFrame:
        """
        Main entry point used by train_equity / train_futures / train_options / train_crypto.

        :param raw_df: Raw OHLCV DataFrame (could be yfinance-style MultiIndex columns).
        :param horizon: How many bars ahead we look to define the target.
        :return: DataFrame with numeric features + 'target' column.
        """
        if raw_df is None or raw_df.empty:
            return pd.DataFrame()

        horizon = horizon or cls.DEFAULT_LOOKAHEAD

        # 1) Normalize columns to a simple, lower-case OHLCV structure
        df = cls._normalize_ohlc(raw_df)

        if df is None or df.empty:
            return pd.DataFrame()

        # 2) Add technical / statistical features
        df = cls._add_features(df)

        # 3) Create binary target (up vs not up) based on future return
        df = cls._add_target(df, horizon=horizon)

        # 4) Drop any rows with NaN after feature/label construction
        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        # 5) Keep ONLY numeric columns + target
        #    (No datetime, no strings – XGBoost requirement)
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c].dtype)]
        if "target" in df.columns and "target" not in numeric_cols:
            # If target is bool, cast to int
            df["target"] = df["target"].astype(int)
            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c].dtype)]

        # Ensure 'target' is present
        if "target" not in df.columns:
            return pd.DataFrame()

        # Final feature matrix
        final_cols = [c for c in numeric_cols if c != "target"] + ["target"]
        df = df[final_cols]

        return df

    # =====================================================================
    # 1. Normalization utilities
    # =====================================================================

    @staticmethod
    def _normalize_ohlc(raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalizes arbitrary OHLCV DataFrames into a standard single-index form.

        Handles:
        - yfinance MultiIndex: ('Price', 'Close'), ('Price', 'Open'), ...
        - Mixed case columns: 'close', 'Close', 'CLOSE'
        - Ensures we end up with: open, high, low, close, volume
        """
        df = raw_df.copy()

        # If MultiIndex columns (typical yfinance), drop top level and keep 2nd level
        if isinstance(df.columns, pd.MultiIndex):
            # Use the last level (Close, Open, High, Low, Volume)
            df.columns = [str(col[-1]).lower() for col in df.columns]
        else:
            df.columns = [str(c).lower() for c in df.columns]

        # Common aliases
        col_map = {}
        for c in df.columns:
            if "open" == c:
                col_map[c] = "open"
            elif "high" == c:
                col_map[c] = "high"
            elif "low" == c:
                col_map[c] = "low"
            elif "close" == c or c == "price":
                col_map[c] = "close"
            elif "volume" == c or c == "vol":
                col_map[c] = "volume"

        df = df.rename(columns=col_map)

        # Keep only the core OHLCV if present
        keep = []
        for k in ["open", "high", "low", "close", "volume"]:
            if k in df.columns:
                keep.append(k)

        if not keep or "close" not in keep:
            # Can't do much without close
            return pd.DataFrame()

        df = df[keep].copy()

        # Coerce everything to numeric (some APIs send strings)
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # Ensure DatetimeIndex if possible
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception:
                # If conversion fails, keep index as-is (not used for XGBoost features)
                pass

        return df

    # =====================================================================
    # 2. Feature construction
    # =====================================================================

    @classmethod
    def _add_features(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build a compact but rich feature set used across all asset classes.

        Features:
            - 1, 3, 5, 10 bar returns
            - Rolling volatility
            - ATR-based volatility
            - Simple moving averages (trend)
            - MA distance (%)
            - RSI-like oscillator
            - Candle body / wick structure
        """
        df = df.copy()

        close = df["close"]

        # ---- Returns ----
        df["ret_1"] = close.pct_change(1)
        df["ret_3"] = close.pct_change(3)
        df["ret_5"] = close.pct_change(5)
        df["ret_10"] = close.pct_change(10)

        # ---- Rolling volatility ----
        df["vol_10"] = df["ret_1"].rolling(10).std()
        df["vol_20"] = df["ret_1"].rolling(20).std()

        # ---- Moving averages ----
        df["ma_10"] = close.rolling(10).mean()
        df["ma_20"] = close.rolling(20).mean()
        df["ma_50"] = close.rolling(50).mean()

        # Distance from MA
        df["close_over_ma20"] = (close - df["ma_20"]) / df["ma_20"]
        df["close_over_ma50"] = (close - df["ma_50"]) / df["ma_50"]

        # ---- ATR-like volatility ----
        if {"high", "low", "close"}.issubset(df.columns):
            high = df["high"]
            low = df["low"]
            prev_close = close.shift(1)

            tr1 = (high - low).abs()
            tr2 = (high - prev_close).abs()
            tr3 = (low - prev_close).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            df["atr_14"] = tr.rolling(14).mean()
        else:
            df["atr_14"] = np.nan

        # ---- RSI-like oscillator ----
        df["rsi_14"] = cls._compute_rsi(close, window=14)

        # ---- Candle structure ----
        if {"open", "high", "low", "close"}.issubset(df.columns):
            body = (df["close"] - df["open"]).abs()
            rng = (df["high"] - df["low"]).replace(0, np.nan)

            df["body_to_range"] = body / rng
            df["upper_wick"] = (df["high"] - df[["close", "open"]].max(axis=1)) / rng
            df["lower_wick"] = (df[["close", "open"]].min(axis=1) - df["low"]) / rng
        else:
            df["body_to_range"] = np.nan
            df["upper_wick"] = np.nan
            df["lower_wick"] = np.nan

        # ---- Volume normalization ----
        if "volume" in df.columns:
            df["vol_zscore_20"] = (df["volume"] - df["volume"].rolling(20).mean()) / (
                df["volume"].rolling(20).std()
            )
        else:
            df["vol_zscore_20"] = np.nan

        return df

    # =====================================================================
    # 3. Target definition
    # =====================================================================

    @staticmethod
    def _add_target(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """
        Create a binary target:
            1 → price goes UP over the next `horizon` bars (net positive move)
            0 → otherwise

        This is intentionally simple and robust:
        - Works for equity, futures, options synthetic, crypto
        - Avoids overfitting to very specific ATR ratios
        """
        df = df.copy()

        if "close" not in df.columns:
            df["target"] = np.nan
            return df

        future_close = df["close"].shift(-horizon)
        future_ret = (future_close / df["close"]) - 1.0

        # Binary upward move label
        df["target"] = (future_ret > 0).astype(int)

        return df

    # =====================================================================
    # 4. Helpers
    # =====================================================================

    @staticmethod
    def _compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
        """
        Simple RSI implementation for ML features (not used as signal directly).
        """
        delta = series.diff()
        gain = (delta.clip(lower=0)).rolling(window).mean()
        loss = (-delta.clip(upper=0)).rolling(window).mean()

        rs = gain / (loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        return rsi
