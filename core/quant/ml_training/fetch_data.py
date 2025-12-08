import pandas as pd
import yfinance as yf

# -------------------------------
# CLEAN YAHOO OHLC DATA (patched)
# -------------------------------
def _clean_yahoo_df(df):
    if df is None or df.empty:
        return None

    df = df.copy()

    # 1. Flatten MultiIndex safely
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join([str(x) for x in col if x]).strip("_")
            for col in df.columns
        ]

    # 2. Lowercase the column names
    df.columns = [str(c).lower() for c in df.columns]

    # 3. If dataframe still contains datetime columns, drop them
    for col in df.columns:
        if "date" in col or df[col].dtype == "datetime64[ns]":
            df = df.drop(columns=[col], errors="ignore")

    # 4. Map the common OHLC names
    rename_map = {}
    for col in df.columns:
        if "open" in col: rename_map[col] = "open"
        if "high" in col: rename_map[col] = "high"
        if "low" in col: rename_map[col] = "low"
        if "close" in col: rename_map[col] = "close"
        if "volume" in col: rename_map[col] = "volume"

    df = df.rename(columns=rename_map)

    # 5. Keep only valid OHLCV
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    if not keep:
        print("[CLEAN] ERROR: No usable OHLC columns after cleaning")
        return None

    df = df[keep].dropna()

    if df.empty:
        return None

    return df.reset_index(drop=True)


# -------------------------------
# YAHOO FETCHERS
# -------------------------------
def fetch_equity_yahoo(symbol: str, days: int = 365):
    ticker = f"{symbol}.NS"
    print(f"[ML] Yahoo fetch → {ticker}")
    df = yf.download(ticker, period=f"{days}d", interval="1d")
    return _clean_yahoo_df(df)


def fetch_index_yahoo(symbol: str, days: int = 365):
    ticker = "^NSEI" if symbol.upper() == "NIFTY" else "^NSEBANK"
    print(f"[ML] Yahoo fetch → {ticker}")
    df = yf.download(ticker, period=f"{days}d", interval="1d")
    return _clean_yahoo_df(df)


def fetch_crypto_yahoo(days: int = 365):
    print("[ML] Yahoo fetch → BTC-INR")
    df = yf.download("BTC-INR", period=f"{days}d", interval="1d")
    return _clean_yahoo_df(df)


# -------------------------------
# Main DataFetcher
# -------------------------------
class DataFetcher:

    @staticmethod
    def fetch(symbol: str, market: str, days: int = 365):
        market = market.upper()

        if market == "EQUITY":
            return fetch_equity_yahoo(symbol, days)

        if market == "FUTURES":
            return fetch_index_yahoo(symbol, days)

        if market == "OPTIONS":
            return fetch_index_yahoo(symbol, days)

        if market == "CRYPTO":
            return fetch_crypto_yahoo(days)

        raise ValueError(f"Unknown market type: {market}")
