"""Carregamento da configuração do projeto."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


def load_config(path: Path | None = None) -> dict[str, Any]:
    path = path or CONFIG_DIR / "project.yaml"
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ("data_dir", "reports_dir"):
        p = Path(cfg[key])
        cfg[key] = p if p.is_absolute() else PROJECT_ROOT / p
    return cfg
