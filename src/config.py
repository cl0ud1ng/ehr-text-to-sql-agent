from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT_DIR / "configs" / "default.yaml"


def load_config(path: Optional[Union[str, os.PathLike[str]]] = None) -> Dict[str, Any]:
    """Load YAML config when PyYAML is available, otherwise return defaults."""
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    defaults: Dict[str, Any] = {
        "databases": {
            "mimic_iii": {"path": "data/EHRSQL/mimic_iii.sqlite"},
            "eicu": {"path": "data/EHRSQL/eicu.sqlite"},
        },
        "agent": {
            "default_db": "mimic_iii",
            "default_model": "deepseek-v4-flash",
            "default_prompt_version": "fewshot",
            "max_repairs": 2,
            "max_rows": 100,
            "timeout_seconds": 5.0,
            "top_k_tables": 6,
            "top_k_columns": 15,
            "use_cache": True,
        },
    }
    if not config_path.exists():
        return defaults
    try:
        import yaml  # type: ignore
    except Exception:
        return defaults
    with config_path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}
    return _deep_merge(defaults, loaded)


def database_path(db_id: str, config: Optional[Dict[str, Any]] = None) -> Path:
    cfg = config or load_config()
    try:
        raw_path = cfg["databases"][db_id]["path"]
    except KeyError as exc:
        raise ValueError(f"Unknown db_id: {db_id}") from exc
    path = Path(raw_path)
    return path if path.is_absolute() else ROOT_DIR / path


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
