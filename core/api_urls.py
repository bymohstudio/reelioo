from django.urls import path
from . import views
from .api_views import AnalyzeCryptoView, BacktestCryptoView, SearchCryptoView
from .views import robots_view, sitemap_view, cron_scan_trigger

urlpatterns = [
    # API
    path("api/analyze/", AnalyzeCryptoView.as_view(), name="analyze_crypto"),
    path("api/backtest/", BacktestCryptoView.as_view(), name="backtest_crypto"),
    path("api/search/", SearchCryptoView.as_view(), name="search_crypto"),

    # Pages
    path("", views.landing_view, name="landing"),
    path("terminal/", views.terminal_view, name="terminal"),
    path("pricing/", views.pricing_view, name="pricing"),
    path("payment/success/", views.payment_success_view, name="payment_success"),
    path("settings/cancel/", views.cancel_subscription_view, name="cancel_subscription"),
    path('robots.txt', robots_view, name='robots'),
    path('sitemap.xml', sitemap_view, name='sitemap'),

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

# Journal Routes
    path('journal/', views.journal_view, name='journal'),
    path('api/journal/add/', views.add_journal_entry, name='add_journal_entry'),
    path('api/journal/delete/<int:entry_id>/', views.delete_journal_entry, name='delete_journal_entry'),
    path('api/journal/refresh/<int:entry_id>/', views.refresh_journal_entry, name='refresh_journal_entry'),


    path('api/cron/trigger/<str:secret_key>/', cron_scan_trigger),
]