from django.apps import AppConfig
from django.utils.module_loading import autodiscover_modules

class ChattingConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "chatting"

    def ready(self):
        autodiscover_modules('preload.py')