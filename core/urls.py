# core/api_urls.py
from django.urls import path
from . import views
from .views import robots_view, sitemap_view, cron_scan_trigger

urlpatterns = [
    # --- HTMX ENDPOINTS ---
    path('hx/ticker/', views.hx_ticker, name='hx_ticker'),
    path('hx/analyze/', views.hx_analyze, name='hx_analyze'),
    path('hx/backtest/', views.hx_backtest, name='hx_backtest'),
    path('hx/alpha/', views.hx_alpha_scan, name='hx_alpha_scan'),
    path('hx/journal/add/', views.hx_journal_add, name='hx_journal_add'),

    # --- API ENDPOINTS (Now handled by views.py) ---
    path('api/symbols/', views.global_symbols_view, name='global_symbols'),
    path('api/search/', views.search_crypto_view, name='search_crypto'),

    # --- JOURNAL ACTIONS ---
    path('api/journal/add/', views.add_journal_entry, name='add_journal_entry'),
    path('api/journal/delete/<int:entry_id>/', views.delete_journal_entry, name='delete_journal_entry'),
    path('api/journal/refresh/<int:entry_id>/', views.refresh_journal_entry, name='refresh_journal_entry'),

    # --- PAGES ---
    path("", views.landing_view, name="landing"),
    path("terminal/", views.terminal_view, name="terminal"),
    path("pricing/", views.pricing_view, name="pricing"),
    path("payment/success/", views.payment_success_view, name="payment_success"),
    path("settings/cancel/", views.cancel_subscription_view, name="cancel_subscription"),
    path('ops/console/', views.ops_dashboard_view, name='ops_dashboard'),
    path('journal/', views.journal_view, name='journal'),

    # --- AUTH ---
    path("auth/login/", views.login_view, name="login"),
    path("auth/signup/", views.signup_view, name="signup"),
    path("auth/logout/", views.logout_view, name="logout"),
    path("account/settings/", views.settings_view, name="settings"),

    # --- SYSTEM ---
    path('robots.txt', robots_view, name='robots'),
    path('sitemap.xml', sitemap_view, name='sitemap'),
    path('api/cron/trigger/<str:secret_key>/', cron_scan_trigger),

    # --- LEGAL ---
    path("legal/terms/", views.terms_view, name="terms"),
    path("legal/privacy/", views.privacy_view, name="privacy"),
    path("legal/refund/", views.refund_view, name="refund"),
    path("legal/contact/", views.contact_view, name="contact"),
    path("legal/plans/", views.pricing_footer_view, name="plans"),
    path("about/", views.about_view, name="about"),
]