import yaml
from pathlib import Path
from typing import Any, Dict

class Config:
    """Singleton Configuration manager for WindVision."""
    _instance = None
    _config: Dict[str, Any] = {}

    def __new__(cls, config_path: str = "config.yaml"):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config(config_path)
        return cls._instance

    def _load_config(self, config_path: str):
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(path, "r") as f:
            self._config = yaml.safe_load(f)

    def get(self, key_path: str, default: Any = None) -> Any:
        """Retrieves a value from the config using dot notation (e.g., 'webcam.cam_id')."""
        keys = key_path.split(".")
        val = self._config
        try:
            for k in keys:
                val = val[k]
            return val
        except (KeyError, TypeError):
            return default

    @property
    def raw(self) -> Dict[str, Any]:
        return self._config

config = Config()
