"""ConfigManager: Loads config/default.yaml and provides dot-notation access."""
import os
import logging
import yaml

logger = logging.getLogger(__name__)

_INSTANCE = None
_CONFIG_PATH = None


class ConfigManager:
    def __init__(self, config_path: str = "config/default.yaml"):
        global _INSTANCE, _CONFIG_PATH
        if _INSTANCE is not None and config_path == _CONFIG_PATH:
            self.config = _INSTANCE.config
            return

        self.config = {}
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f) or {}
            logger.info(f"Config loaded: {config_path}")
        else:
            logger.warning(f"Config not found: {config_path} — using defaults")

        _INSTANCE = self
        _CONFIG_PATH = config_path

    def get(self, key_path: str, default=None):
        """Dot-notation access: config.get('models.image.width', 832)"""
        keys = key_path.split(".")
        val = self.config
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k)
            else:
                return default
            if val is None:
                return default
        return val
