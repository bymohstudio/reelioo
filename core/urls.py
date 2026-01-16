from django.urls import path
from . import views

urlpatterns = [
    # --- HTMX ENDPOINTS (Required for Terminal/Charts) ---
    path('hx/ticker/', views.hx_ticker, name='hx_ticker'),
    path('hx/analyze/', views.hx_analyze, name='hx_analyze'),
    path('hx/backtest/', views.hx_backtest, name='hx_backtest'),
    path('hx/alpha/', views.hx_alpha_scan, name='hx_alpha_scan'),
    path('hx/journal/add/', views.hx_journal_add, name='hx_journal_add'),

    # --- JOURNAL API ---
    path('api/journal/add/', views.add_journal_entry, name='add_journal_entry'),
    path('api/journal/delete/<int:entry_id>/', views.delete_journal_entry, name='delete_journal_entry'),
    path('api/journal/refresh/<int:entry_id>/', views.refresh_journal_entry, name='refresh_journal_entry'),

    # --- MAIN PAGES ---
    path("", views.landing_view, name="landing"),
    path("terminal/", views.terminal_view, name="terminal"),
    path('journal/', views.journal_view, name='journal'),
    path('ops/console/', views.ops_dashboard_view, name='ops_dashboard'),

    # --- AUTH (The Missing Link) ---
    path("get-access/", views.get_access_view, name="get_access"),
    path("auth/logout/", views.logout_view, name="logout"),  # <--- THIS FIXES THE ERROR

    # --- LEGACY REDIRECTS (To prevent broken links) ---
    path("auth/login/", views.get_access_view),
    path("auth/signup/", views.get_access_view),
    path("pricing/", views.pricing_footer_view, name="pricing"),

    # --- API ---
    path('api/symbols/', views.global_symbols_view, name='global_symbols'),
    path('api/search/', views.search_crypto_view, name='search_crypto'),
    path('api/cron/trigger/<str:secret_key>/', views.cron_scan_trigger),

    # --- LEGAL ---
    path("legal/terms/", views.terms_view, name="terms"),
    path("legal/privacy/", views.privacy_view, name="privacy"),
    path("legal/refund/", views.refund_view, name="refund"),
    path("legal/contact/", views.contact_view, name="contact"),
    path("about/", views.about_view, name="about"),

    # --- SYSTEM ---
    path('robots.txt', views.robots_view),
    path('sitemap.xml', views.sitemap_view),
]