from pathlib import Path

import yaml

from src.core.exceptions import ConfigurationError


ROOT = Path(__file__).resolve().parents[2]


def get_project_root() -> Path:
    return ROOT


def load_config(path=None):
    path = Path(path) if path else ROOT / "configs" / "config.yaml"

    try:
        with path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    except Exception as e:
        raise ConfigurationError(
            f"Cannot load config: {path}: {e}"
        ) from e

    if "model" not in cfg or "registry" not in cfg["model"]:
        raise ConfigurationError(
            "config.yaml must contain model.registry"
        )

    return cfg


def get_model_config(cfg, model_key: str):
    registry = cfg["model"]["registry"]

    if model_key not in registry:
        raise ConfigurationError(
            f"Unknown model '{model_key}'. "
            f"Choices: {', '.join(registry)}"
        )

    return registry[model_key]