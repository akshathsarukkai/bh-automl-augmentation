"""Configuration loading utilities."""

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file.

    Parameters
    ----------
    path:
        Path to a YAML configuration file.

    Returns
    -------
    dict[str, Any]
        Parsed configuration values.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    ValueError
        If the YAML cannot be parsed or does not contain a mapping.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    if not config_path.is_file():
        raise ValueError(f"Config path is not a file: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML config at {config_path}: {exc}") from exc

    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError(
            f"Config file must contain a YAML mapping at top level: {config_path}"
        )

    return config
