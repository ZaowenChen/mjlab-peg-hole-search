#!/usr/bin/env python3
"""Read-only structural and historical-artifact compatibility audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from mjlab_contact_prep.frozen_source import resolve_frozen_source


ROOT = Path(__file__).resolve().parents[1]
FORMAL = ROOT / "evaluation/night_fullcircle_5mm_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict[str, object]] = []

    def check(self, name: str, condition: bool, detail: str = "") -> None:
        self.checks.append({"name": name, "ok": bool(condition), "detail": detail})

    @property
    def ok(self) -> bool:
        return all(bool(item["ok"]) for item in self.checks)


def audit_workspace(load_payloads: bool = True, compare_working_tree: bool = False) -> Audit:
    audit = Audit()
    required = [
        ROOT / "README.md",
        ROOT / "CURRENT_STATE.md",
        ROOT / "configs/default.json",
        ROOT / "docs/ARCHITECTURE.md",
        ROOT / "evaluation/EXPERIMENT_INDEX.md",
        ROOT / "reference/workspace_origin.json",
    ]
    audit.check("single entry and indexes", all(path.is_file() for path in required))

    try:
        head = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
        root_commit = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-list", "--max-parents=0", "HEAD"],
            text=True,
        ).splitlines()[0]
        origin = json.loads((ROOT / "reference/workspace_origin.json").read_text())
        valid = origin.get("primary_workspace") == "." and bool(root_commit)
        audit.check("independent Git baseline", valid, f"root={root_commit} head={head}")
    except (OSError, subprocess.CalledProcessError, KeyError, json.JSONDecodeError) as exc:
        audit.check("independent Git baseline", False, repr(exc))

    try:
        plan = json.loads((FORMAL / "plan.json").read_text())
        mismatches = []
        for section in ("physics_frozen", "experiment_source_sha256"):
            for relative, expected in plan.get(section, {}).items():
                path = ROOT / relative
                if not path.is_file() or sha256(path) != expected:
                    mismatches.append(relative)
        resolved = resolve_frozen_source(plan, ROOT, FORMAL)
        audit.check("formal frozen source snapshots", True, f"verified={len(resolved)} files")
        if compare_working_tree:
            audit.check("working tree matches formal source", not mismatches,
                        "mismatches=" + ",".join(mismatches))
    except (OSError, ValueError) as exc:
        plan = {}
        audit.check("formal frozen source snapshots", False, repr(exc))

    policies = [
        FORMAL / f"amp_{amp}/seed_{seed}/policy.pt"
        for amp in ("0", "0.05")
        for seed in (7, 17, 27)
    ]
    audit.check("six final policies exist", all(path.is_file() for path in policies))
    if load_payloads and all(path.is_file() for path in policies):
        try:
            valid = True
            for path in policies:
                payload = torch.load(path, weights_only=False, map_location="cpu")
                valid &= all(
                    key in payload
                    for key in ("actor_state_dict", "critic_state_dict", "environment")
                )
            audit.check("six final policies load", valid)
        except Exception as exc:  # A corrupt historical payload must be reported.
            audit.check("six final policies load", False, repr(exc))

    banks = [FORMAL / "training_bank/bank.pt", FORMAL / "test_bank/bank.pt"]
    audit.check("training and test banks exist", all(path.is_file() for path in banks))
    if load_payloads and all(path.is_file() for path in banks):
        try:
            valid = True
            for path in banks:
                bank = torch.load(path, weights_only=False, map_location="cpu")
                valid &= all(key in bank for key in ("physics", "controller", "term", "probe", "rows"))
            audit.check("legacy state banks load", valid)
        except Exception as exc:
            audit.check("legacy state banks load", False, repr(exc))

    trajectories = [
        FORMAL / f"amp_{amp}/seed_{seed}/evaluation/trajectory.npz"
        for amp in ("0", "0.05")
        for seed in (7, 17, 27)
    ]
    try:
        valid = all(path.is_file() for path in trajectories)
        if load_payloads and valid:
            for path in trajectories:
                with np.load(path) as payload:
                    valid &= {"trace", "columns"}.issubset(payload.files)
        audit.check("six audited trajectories read", valid)
    except Exception as exc:
        audit.check("six audited trajectories read", False, repr(exc))

    event_logs = list(FORMAL.glob("amp_*/seed_*/logs/events.out.tfevents.*"))
    audit.check("six TensorBoard logs exist", len(event_logs) == 6, f"count={len(event_logs)}")

    try:
        totals = {0.0: 0, 0.05: 0}
        denominators = {0.0: 0, 0.05: 0}
        for amp in totals:
            for seed in (7, 17, 27):
                result = json.loads(
                    (FORMAL / f"amp_{amp:g}/seed_{seed}/evaluation/results.json").read_text()
                )
                totals[amp] += int(result["successes"])
                denominators[amp] += int(result["total"])
        expected = totals == {0.0: 667, 0.05: 635} and denominators == {0.0: 672, 0.05: 672}
        audit.check("formal aggregate unchanged", expected, f"successes={totals} totals={denominators}")
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        audit.check("formal aggregate unchanged", False, repr(exc))
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-payloads", action="store_true", help="check paths and hashes without torch/NumPy payload loading")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--compare-working-tree", action="store_true",
                        help="also require current development files to match the historical experiment")
    args = parser.parse_args()
    audit = audit_workspace(load_payloads=not args.skip_payloads,
                            compare_working_tree=args.compare_working_tree)
    if args.json:
        print(json.dumps({"ok": audit.ok, "checks": audit.checks}, ensure_ascii=False, indent=2))
    else:
        for item in audit.checks:
            suffix = f" — {item['detail']}" if item["detail"] else ""
            print(f"{'PASS' if item['ok'] else 'FAIL'} {item['name']}{suffix}")
        print(f"RESULT {'PASS' if audit.ok else 'FAIL'}")
    return 0 if audit.ok else 1


if __name__ == "__main__":
    sys.exit(main())
