from django.test import TestCase, RequestFactory
from rest_framework.test import APIClient
from django.urls import reverse
import pandas as pd
import json
from unittest.mock import patch

# Import Views
from core.api_views import AnalyzeGlobalView

class InstitutionalQuantTest(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.factory = RequestFactory()
        
        # Create Dummy OHLC Dataframe mimicking yfinance
        dates = pd.date_range(start="2024-01-01", periods=100, freq="1D")
        self.mock_df = pd.DataFrame({
            "Open": [150.0] * 100,
            "High": [155.0] * 100,
            "Low": [145.0] * 100,
            "Close": [150.0] * 100,
            "Volume": [1000000] * 100
        }, index=dates)

    @patch("core.services.marketdata_service.MarketService.get_historical_data")
    def test_equity_analysis_pipeline(self, mock_get_data):
        """
        HEAVY TEST: Simulates a full user request for AAPL.
        Verifies:
        1. Data fetching works (Mocked)
        2. Quant Engine runs without crashing
        3. Prediction is saved to DB (No KeyError 'prediction')
        4. Response is 200 OK
        """
        # 1. Setup Mock Data
        mock_get_data.return_value = self.mock_df
        
        # 2. Simulate POST Request
        url = reverse("analyze_global")
        payload = {
            "symbol": "AAPL",
            "market_type": "EQUITY",
            "trade_style": "SWING"
        }
        
        response = self.client.post(url, payload, format='json')
        
        # 3. Debugging Output
        if response.status_code != 200:
            print("\n!!! TEST FAILED WITH ERROR !!!")
            print(response.data)
        
        # 4. Assertions
        self.assertEqual(response.status_code, 200, "API crashed with 500")
        
        data = response.json()
        self.assertIn("score", data)
        self.assertIn("entry", data)
        self.assertIn("time_frame", data) # V2 Key check
        
        # 5. Verify DB Entry
        from core.models import Prediction
        saved = Prediction.objects.last()
        self.assertIsNotNone(saved, "Prediction not saved to DB")
        self.assertEqual(saved.symbol, "AAPL")
        self.assertEqual(saved.market, "EQUITY")

    @patch("core.services.marketdata_service.MarketService.get_historical_data")
    def test_autosuggestion_flow(self, mock_get_data):
        """
        Tests if a symbol search query returns valid structure.
        """
        url = reverse("search_symbol")
        response = self.client.get(url, {"q": "AAP", "market_type": "GLOBAL"})
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Check structure
        if len(data) > 0:
            item = data[0]
            self.assertIn("symbol", item)
            self.assertIn("name", item)
            self.assertIn("market", item)

    def test_missing_symbol_error_handling(self):
        """
        Ensure user gets a clean error message, not a python stacktrace.
        """
        url = reverse("analyze_global")
        response = self.client.post(url, {}, format='json') # No symbol
        
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.data["error"], "Symbol required")