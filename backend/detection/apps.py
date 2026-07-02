import logging
import os

from django.apps import AppConfig


class DetectionConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "detection"

    def ready(self):
        log_level = os.environ.get("DEFAKE_LOG_LEVEL", "INFO").upper()
        logger = logging.getLogger("detection")
        logger.setLevel(log_level)
        formatter = logging.Formatter(
            "[%(levelname)s] %(asctime)s %(name)s - %(message)s",
            datefmt="%H:%M:%S",
        )
        if not any(getattr(handler, "_defake_console", False) for handler in logger.handlers):
            handler = logging.StreamHandler()
            handler._defake_console = True
            logger.addHandler(handler)
        for handler in logger.handlers:
            if getattr(handler, "_defake_console", False):
                handler.setLevel(log_level)
                handler.setFormatter(formatter)
        logger.propagate = False
