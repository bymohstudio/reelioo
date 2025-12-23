import os
import requests
import pandas as pd
from django.core.management.base import BaseCommand
from django.conf import settings


class Command(BaseCommand):
    help = "Fetches Global crypto symbols and saves to CSV for fast search."

    def handle(self, *args, **kwargs):
        base_dir = settings.BASE_DIR
        global_path = os.path.join(base_dir, "global_symbols.csv")

        self.stdout.write("🌍 Fetching Binance (Global)...")
        self.fetch_binance(global_path)
        self.stdout.write(self.style.SUCCESS("✅ Global symbols updated successfully!"))

    def fetch_binance(self, path):
        try:
            url = "https://api.binance.com/api/v3/exchangeInfo"
            data = requests.get(url, timeout=10).json()
            symbols = []

            for s in data.get("symbols", []):
                if s["status"] == "TRADING" and s["quoteAsset"] == "USDT":
                    symbols.append({
                        "symbol": s["symbol"],
                        "name": s["baseAsset"],
                        "type": "GLOBAL"
                    })

            df = pd.DataFrame(symbols)
            df.to_csv(path, index=False)
            self.stdout.write(f"   Saved {len(df)} Global pairs.")

        except Exception as e:
            self.stdout.write(self.style.ERROR(f"   Binance Failed: {e}"))