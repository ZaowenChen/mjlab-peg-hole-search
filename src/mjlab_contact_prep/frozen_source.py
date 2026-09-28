"""Resolve historical source bytes against the original experiment hashes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
from typing import Any, Mapping


def frozen_hashes(plan: Mapping[str, Any]) -> dict[str, str]:
    expected: dict[str, str] = {}
    for section in ("physics_frozen", "experiment_source_sha256"):
        for name, digest in plan.get(section, {}).items():
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or not path.parts:
                raise ValueError(f"invalid frozen source path: {name}")
            if name in expected and expected[name] != digest:
                raise ValueError(f"conflicting frozen source hash: {name}")
            expected[name] = digest
    if not expected:
        raise ValueError("experiment plan has no frozen source hashes")
    return expected


def resolve_frozen_source(
    plan: Mapping[str, Any], repository: Path, experiment: Path
) -> dict[str, tuple[bytes, str]]:
    """Use matching run snapshots, archived files, current files or tar members.

    Every file matches its original digest. No manifest is rewritten, and tar
    archives are read without extracting paths or following links.
    """
    repository, experiment = Path(repository), Path(experiment)
    expected = frozen_hashes(plan)
    resolved: dict[str, tuple[bytes, str]] = {}
    archive_manifest = repository / "archive/manifest.json"
    relocated = {}
    if archive_manifest.is_file():
        for row in json.loads(archive_manifest.read_text())["files"]:
            path = PurePosixPath(row["archived_path"])
            if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "archive":
                raise ValueError("invalid archived source path")
            relocated[row["original_path"]] = repository / path

    def accept(name: str, data: bytes, source: str) -> None:
        if hashlib.sha256(data).hexdigest() == expected[name]:
            resolved[name] = (data, source)

    for name in expected:
        candidates = [experiment / "source" / name]
        if name in relocated:
            candidates.append(relocated[name])
        candidates.append(repository / name)
        for candidate in candidates:
            if candidate.is_file():
                accept(name, candidate.read_bytes(), str(candidate))
                if name in resolved:
                    break

    for archive in sorted((repository / "reference").glob("*.tar.gz")):
        if len(resolved) == len(expected):
            break
        with tarfile.open(archive, "r:gz") as stream:
            for member in stream:
                name = member.name.removeprefix("./")
                if name in expected and name not in resolved and member.isfile():
                    payload = stream.extractfile(member)
                    if payload is not None:
                        accept(name, payload.read(), f"{archive}:{member.name}")

    missing = sorted(set(expected) - set(resolved))
    if missing:
        raise ValueError("missing matching frozen source: " + ", ".join(missing))
    return resolved


def materialize_frozen_source(
    files: Mapping[str, tuple[bytes, str]], repository: Path, destination: Path
) -> None:
    """Build a separate source tree; historical evaluation data stays in place."""
    repository, destination = Path(repository), Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    for folder in ("src", "assets", "configs", "scripts", "tests"):
        if (repository / folder).is_dir():
            shutil.copytree(repository / folder, destination / folder,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name, (data, _) in files.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    # Historical PPO code reads this template relative to its source root.
    # Keep it isolated too: a frozen evaluation/config member must never write
    # through a symlink into the original experiment data.
    template = Path("evaluation/gpu_ppo_pilot_v2/amp_0/config.json")
    if (repository / template).is_file() and not (destination / template).exists():
        (destination / template).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repository / template, destination / template)
    (destination / "reference").symlink_to((repository / "reference").resolve(), target_is_directory=True)
