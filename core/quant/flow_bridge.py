import requests
import pandas as pd
import numpy as np

BINANCE_LIQ = "https://fapi.binance.com/fapi/v1/allForceOrders"
BINANCE_PREM = "https://fapi.binance.com/fapi/v1/premiumIndex"


def get_btc_flow_snapshot():
    """
    Returns a numpy array [[liq_pressure, funding_z]] for the model.
    """
    try:
        # 1. Fetch Liquidations (Last 15 mins approx)
        r_liq = requests.get(BINANCE_LIQ, params={"symbol": "BTCUSDT", "limit": 99})
        liqs = r_liq.json()

        buy_vol = sum([float(x['origQty']) for x in liqs if x['side'] == "SELL"])  # Liq Sell = Forced Long Close
        sell_vol = sum([float(x['origQty']) for x in liqs if x['side'] == "BUY"])  # Liq Buy = Forced Short Close

        # Net Pressure: Positive = More Shorts dying (Bullish pressure)
        # Negative = More Longs dying (Bearish pressure)
        liq_pressure = sell_vol - buy_vol

        # 2. Fetch Funding
        r_fund = requests.get(BINANCE_PREM, params={"symbol": "BTCUSDT"})
        fund_data = r_fund.json()
        funding_rate = float(fund_data['lastFundingRate'])

        # Normalize Funding (Z-Score approximation based on recent history constants)
        # In production, you might want a moving average, but for speed we use a static baseline
        # Baseline: 0.01% (0.0001) is standard. Std dev approx 0.0002.
        funding_z = (funding_rate - 0.0001) / 0.0002

        return np.array([[liq_pressure, funding_z]])

    except Exception as e:
        print(f"⚠️ Flow Bridge Error: {e}")
        return None