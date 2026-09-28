#!/usr/bin/env python3
"""Replay a checkpoint with its original frozen source into a new result tag."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from mjlab_contact_prep.frozen_source import materialize_frozen_source, resolve_frozen_source

ROOT = Path(__file__).resolve().parents[1]

WORKER = """
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from mjlab_contact_prep.night_run.jobs import evaluate_job, verify_frozen
experiment, checkpoint, bank = map(Path, sys.argv[2:5])
seed, amplitude, horizon = int(sys.argv[5]), float(sys.argv[6]), int(sys.argv[7])
plan = json.loads((experiment / 'plan.json').read_text())
verify_frozen(plan)
result = evaluate_job(experiment, seed, amplitude, checkpoint=checkpoint,
                      bank=bank, horizon=horizon, tag=sys.argv[8])
print(json.dumps({k:result[k] for k in ('successes','total','audit_passed')}))
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path,
                        default=ROOT / "evaluation/night_fullcircle_5mm_v1")
    parser.add_argument("--seed", type=int, choices=[7, 17, 27], required=True)
    parser.add_argument("--amplitude", type=float, choices=[0.0, 0.05], required=True)
    parser.add_argument("--tag", required=True, help="new result directory under the seed")
    parser.add_argument("--checkpoint", type=Path, help="default: selected seed's policy.pt")
    parser.add_argument("--check-only", action="store_true", help="verify frozen source without GPU or writes")
    args = parser.parse_args()
    if Path(args.tag).name != args.tag or args.tag in {"evaluation", "test", ".", ".."}:
        parser.error("tag must be one new directory name, excluding evaluation and test")
    experiment = args.experiment_root.resolve()
    plan = json.loads((experiment / "plan.json").read_text())
    folder = experiment / f"amp_{args.amplitude:g}/seed_{args.seed}"
    destination = folder / args.tag
    if destination.exists():
        parser.error(f"destination already exists: {destination}")
    files = resolve_frozen_source(plan, ROOT, experiment)
    if args.check_only:
        print(json.dumps({"frozen_source_verified": True, "files": len(files), "writes": False}))
        return
    checkpoint = (args.checkpoint or folder / "policy.pt").resolve()
    bank = experiment / "test_bank/bank.pt"
    for path in (checkpoint, bank):
        if not path.is_file():
            parser.error(f"required local experiment artifact missing: {path}")
    destination.mkdir(parents=True, exist_ok=False)
    workspace = destination / "frozen_source"
    materialize_frozen_source(files, ROOT, workspace)
    provenance = {"experiment": str(experiment), "checkpoint": str(checkpoint),
                  "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  "plan_sha256": hashlib.sha256((experiment / "plan.json").read_bytes()).hexdigest(),
                  "sources": {name: source for name, (_, source) in files.items()},
                  "effective_config": plan.get("effective_config"),
                  "seed": args.seed, "amplitude_deg": args.amplitude}
    (destination / "replay_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    horizon = int(plan.get("evaluation_horizon_steps", plan.get("horizon_steps", 200)))
    subprocess.run([sys.executable, "-c", WORKER, str(workspace / "src"),
                    str(experiment), str(checkpoint), str(bank), str(args.seed),
                    str(args.amplitude), str(horizon), args.tag], cwd=workspace, check=True)


if __name__ == "__main__":
    main()
