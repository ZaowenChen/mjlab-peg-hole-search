"""Versioned access to full simulator/controller state banks.

Version 0 is the legacy dictionary used by the completed 2026-09-23 formal
experiment.  It is upgraded in memory only; the original file is never
rewritten.  Version 1 adds explicit format metadata while preserving all
legacy tensor groups.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch


STATE_BANK_FORMAT_VERSION = 1
SUPPORTED_STATE_BANK_VERSIONS = {0, STATE_BANK_FORMAT_VERSION}
REQUIRED_GROUPS = ("physics", "controller", "term", "probe", "rows")


def normalize_state_bank(raw: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError("state bank must contain a mapping")
    source_version = int(raw.get("format_version", 0))
    if source_version not in SUPPORTED_STATE_BANK_VERSIONS:
        raise ValueError(f"unsupported state-bank format_version={source_version}")
    missing = [name for name in REQUIRED_GROUPS if name not in raw]
    if missing:
        raise ValueError(f"state bank is missing required groups: {', '.join(missing)}")
    upgraded = dict(raw)
    upgraded["format_version"] = STATE_BANK_FORMAT_VERSION
    upgraded.setdefault("format_metadata", {})
    upgraded["format_metadata"] = {
        **dict(upgraded["format_metadata"]),
        "source_format_version": source_version,
    }
    return upgraded


def load_state_bank(path: str | Path, *, map_location: str = "cpu") -> dict[str, Any]:
    return normalize_state_bank(
        torch.load(Path(path), weights_only=False, map_location=map_location)
    )


def save_state_bank(path: str | Path, bank: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = normalize_state_bank(bank)
    payload["format_metadata"] = {
        **payload["format_metadata"],
        "source_format_version": STATE_BANK_FORMAT_VERSION,
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


# The frozen implementation remains the behavior source for the completed
# formal experiment; this is the new responsibility-oriented public entry.
from mjlab_contact_prep.night_run.environment import (  # noqa: E402,F401
    AUDIT_COLUMNS,
    BankEnv,
    prepare_bank,
)

__all__ = [
    "AUDIT_COLUMNS",
    "BankEnv",
    "STATE_BANK_FORMAT_VERSION",
    "load_state_bank",
    "normalize_state_bank",
    "prepare_bank",
    "save_state_bank",
]
