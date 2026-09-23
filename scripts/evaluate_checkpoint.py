#!/usr/bin/env python3
"""Re-evaluate one existing formal checkpoint into a new, non-overwriting tag."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mjlab_contact_prep.configuration import resolve_config, write_effective_config
from mjlab_contact_prep.evaluation.formal import evaluate_job
from mjlab_contact_prep.training.formal import verify_frozen


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=ROOT / "evaluation/night_fullcircle_5mm_v1",
    )
    parser.add_argument("--seed", type=int, choices=[7, 17, 27], required=True)
    parser.add_argument("--amplitude", type=float, choices=[0.0, 0.05], required=True)
    parser.add_argument("--tag", required=True, help="new directory name under the selected seed")
    args = parser.parse_args()
    if Path(args.tag).name != args.tag or args.tag in {"evaluation", ".", ".."}:
        parser.error("tag must be one new directory name and cannot be 'evaluation'")

    experiment = args.experiment_root.resolve()
    plan = json.loads((experiment / "plan.json").read_text())
    verify_frozen(plan)
    model_folder = experiment / f"amp_{args.amplitude:g}" / f"seed_{args.seed}"
    destination = model_folder / args.tag
    if destination.exists():
        parser.error(f"destination already exists: {destination}")

    result = evaluate_job(
        experiment,
        args.seed,
        args.amplitude,
        checkpoint=model_folder / "policy.pt",
        bank=experiment / "test_bank/bank.pt",
        horizon=int(plan.get("evaluation_horizon_steps", 200)),
        tag=args.tag,
    )
    resolved = resolve_config(
        ROOT / "configs/default.json",
        ROOT / "configs/experiments/night_fullcircle_5mm_v1.json",
        [
            f"replay.seed={args.seed}",
            f"replay.amplitude_deg={args.amplitude}",
            f"replay.tag={json.dumps(args.tag)}",
        ],
    )
    write_effective_config(destination / "effective_config.json", resolved)
    print(
        json.dumps(
            {
                "destination": str(destination),
                "successes": result["successes"],
                "total": result["total"],
                "audit_passed": result["audit_passed"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
