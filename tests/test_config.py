"""Tests for configuration loading utilities."""

from pathlib import Path

import pytest

from bh_augmentation.utils.config import load_config


def test_load_config_reads_yaml_mapping(tmp_path: Path) -> None:
    """YAML mappings should load as dictionaries."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "seed: 42\n"
        "dataset:\n"
        "  name: synthetic_bh\n"
        "metrics:\n"
        "  - rmse\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config["seed"] == 42
    assert config["dataset"]["name"] == "synthetic_bh"
    assert config["metrics"] == ["rmse"]


def test_load_config_raises_for_missing_path(tmp_path: Path) -> None:
    """Missing config paths should raise a helpful FileNotFoundError."""
    missing_path = tmp_path / "missing.yaml"

    with pytest.raises(FileNotFoundError, match="Config file does not exist"):
        load_config(missing_path)


def test_load_config_raises_for_invalid_yaml(tmp_path: Path) -> None:
    """Invalid YAML should raise a ValueError with file context."""
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("seed: [42\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid YAML config"):
        load_config(config_path)
