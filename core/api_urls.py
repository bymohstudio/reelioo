from django.urls import path
from . import views
from .api_views import AnalyzeCryptoView, BacktestCryptoView, SearchCryptoView

urlpatterns = [
    # API
    path("api/analyze/", AnalyzeCryptoView.as_view(), name="analyze_crypto"),
    path("api/backtest/", BacktestCryptoView.as_view(), name="backtest_crypto"),
    path("api/search/", SearchCryptoView.as_view(), name="search_crypto"),

    # Pages
    path("", views.landing_view, name="landing"),
    path("terminal/", views.terminal_view, name="terminal"),
    path("pricing/", views.pricing_view, name="pricing"),
    path("payment/success/", views.payment_success, name="payment_success"),

    # Auth
    path("auth/login/", views.login_view, name="login"),
    path("auth/signup/", views.signup_view, name="signup"),
    path("auth/logout/", views.logout_view, name="logout"),
    path("account/settings/", views.settings_view, name="settings"),

    # Legal
    path("legal/terms/", views.terms_view, name="terms"),
    path("legal/privacy/", views.privacy_view, name="privacy"),
    path("legal/refund/", views.refund_view, name="refund"),
    path("legal/contact/", views.contact_view, name="contact"),
    path("legal/plans/", views.pricing_footer_view, name="plans"),
]