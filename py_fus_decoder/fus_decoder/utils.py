"""Small utility helpers with import-light defaults."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Iterable, List, Sequence


def require_dependency(module_name: str, install_hint: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Missing optional dependency '{module_name}'. Install with: {install_hint}"
        ) from exc


def ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def maybe_load_yaml_or_json(path: Path) -> Any:
    if path.suffix.lower() in {".yaml", ".yml"}:
        yaml = require_dependency("yaml", 'pip install -e ".[io]"')
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    return load_json(path)


def flatten_once(items: Iterable[Sequence[Any]]) -> List[Any]:
    merged: List[Any] = []
    for seq in items:
        merged.extend(seq)
    return merged
