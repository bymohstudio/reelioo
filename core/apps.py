from django.apps import AppConfig


class CoreConfig(AppConfig):
    name = 'core'

# core/apps.py
# from django.apps import AppConfig
# import sys
#
# class CoreConfig(AppConfig):
#     default_auto_field = 'django.db.models.BigAutoField'
#     name = 'core'
#
#     def ready(self):
#         # Prevent running twice (reloader) and only run for 'runserver'
#         if 'runserver' in sys.argv:
#             from core.services.marketdata_service import MarketService
#             MarketService.sync_instruments()

