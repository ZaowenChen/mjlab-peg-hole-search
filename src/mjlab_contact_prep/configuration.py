"""Versioned configuration composition for new experiments.

The historical experiment modules keep their frozen configuration paths.  New
runs should resolve a default file, then an experiment file, then explicit
``KEY=VALUE`` overrides and save the returned document beside their outputs.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


CONFIG_FORMAT_VERSION = 1


def _read_json(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"configuration must be a JSON object: {path}")
    version = data.get("format_version", CONFIG_FORMAT_VERSION)
    if version != CONFIG_FORMAT_VERSION:
        raise ValueError(f"unsupported configuration format_version={version}: {path}")
    return data


def deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings without mutating either input."""
    result = copy.deepcopy(dict(base))
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _parse_override_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def parse_typed_override(raw: str, default: Any) -> Any:
    """Parse a CLI value without treating the string 'false' as truthy."""
    value = _parse_override_value(raw)
    if isinstance(default, bool):
        valid = isinstance(value, bool)
    elif isinstance(default, int):
        valid = isinstance(value, int) and not isinstance(value, bool)
    elif isinstance(default, float):
        valid = isinstance(value, (int, float)) and not isinstance(value, bool)
    else:
        valid = isinstance(value, type(default))
    if not valid:
        raise ValueError(f"expected {type(default).__name__} value, got {raw!r}")
    return type(default)(value)


def apply_overrides(config: Mapping[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    """Apply dotted ``KEY=VALUE`` overrides, parsing values as JSON when possible."""
    result = copy.deepcopy(dict(config))
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must be KEY=VALUE: {item}")
        dotted, raw = item.split("=", 1)
        parts = [part for part in dotted.split(".") if part]
        if not parts:
            raise ValueError(f"override has an empty key: {item}")
        target = result
        for part in parts[:-1]:
            existing = target.setdefault(part, {})
            if not isinstance(existing, dict):
                raise ValueError(f"override crosses a non-object key: {dotted}")
            target = existing
        target[parts[-1]] = _parse_override_value(raw)
    return result


def resolve_config(
    default_path: str | Path,
    experiment_path: str | Path | None = None,
    overrides: Iterable[str] = (),
) -> dict[str, Any]:
    """Resolve ``default -> experiment -> command line`` and retain provenance."""
    overrides = list(overrides)
    default = _read_json(default_path)
    experiment = _read_json(experiment_path)
    effective = apply_overrides(deep_merge(default, experiment), overrides)
    if effective.get("format_version", CONFIG_FORMAT_VERSION) != CONFIG_FORMAT_VERSION:
        raise ValueError("unsupported configuration format_version override")
    effective["format_version"] = CONFIG_FORMAT_VERSION
    return {
        "format_version": CONFIG_FORMAT_VERSION,
        "sources": {
            "default": str(Path(default_path).resolve()),
            "experiment": str(Path(experiment_path).resolve()) if experiment_path else None,
            "overrides": overrides,
        },
        "effective": effective,
    }


def write_effective_config(path: str | Path, document: Mapping[str, Any]) -> None:
    """Atomically save the complete resolved configuration used by a run."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(destination)
