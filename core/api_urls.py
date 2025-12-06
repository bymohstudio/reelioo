from django.urls import path
from . import api_views

urlpatterns = [
    # Analysis
    path("analyze/market/", api_views.AnalyzeMarketView.as_view(), name="analyze_market"),
    path("analyze/backtest/", api_views.BacktestView.as_view(), name="analyze_backtest"),
    
    # Utils
    path("search-symbol/", api_views.SearchSymbolView.as_view(), name="search_symbol"),
    
    # Feedback
    path("feedback/submit/", api_views.SubmitFeedbackView.as_view(), name="submit_feedback"),
]