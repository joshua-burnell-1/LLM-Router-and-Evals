"""Paths and YAML config loading shared by every module."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
TASKS_DIR = DATA_DIR / "tasks"


def config_path(name: str) -> Path:
    """Path to a file in config/, overridable per file via ROUTEWISE_<NAME>_CONFIG."""
    env_key = f"ROUTEWISE_{Path(name).stem.upper()}_CONFIG"
    override = os.environ.get(env_key)
    return Path(override) if override else CONFIG_DIR / name


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def default_db_path() -> Path:
    return Path(os.environ.get("ROUTEWISE_DB") or DATA_DIR / "results.db")
