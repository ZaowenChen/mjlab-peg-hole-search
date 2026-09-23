#!/usr/bin/env python3
"""Run a tiny save/restore/evaluate smoke through the responsibility APIs."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
from rsl_rl.runners import OnPolicyRunner

from mjlab_contact_prep.configuration import resolve_config, write_effective_config
from mjlab_contact_prep.data.state_bank import BankEnv
from mjlab_contact_prep.evaluation.formal import evaluate_job
from mjlab_contact_prep.training.formal import ppo_config, train_job, verify_frozen


ROOT = Path(__file__).resolve().parents[1]
FORMAL = ROOT / "evaluation/night_fullcircle_5mm_v1"


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    plan = json.loads((FORMAL / "plan.json").read_text())
    verify_frozen(plan)
    plan.update(
        num_envs=8,
        iterations=2,
        samples_per_seed=64,
        queue_order=[{"amplitude": 0.0, "seed": 7}],
        smoke=True,
        smoke_source="DES-005 responsibility API save/restore/evaluate check",
    )
    write_json(output / "plan.json", plan)

    training_bank = FORMAL / "preflight/bank/bank.pt"
    test_bank = FORMAL / "preflight/queue_smoke/test_bank/bank.pt"
    trained = train_job(
        output,
        7,
        0.0,
        bank=training_bank,
        n=8,
        iterations=2,
        steps=4,
        resume=True,
    )

    environment = BankEnv(training_bank, 8, 0.0, 7)
    try:
        runner = OnPolicyRunner(
            environment,
            copy.deepcopy(ppo_config(7, 2, 4)),
            None,
            device=environment.device,
        )
        checkpoint = torch.load(
            output / "amp_0/seed_7/policy.pt",
            weights_only=False,
            map_location="cpu",
        )
        runner.alg.load(checkpoint, None, True)
        environment.restore_snapshot(checkpoint["environment"])
        expected = checkpoint["environment"]["buffers"]["history"].flatten(1)
        observation_restored = torch.equal(
            environment.get_observations()["actor"].cpu(), expected
        )
        if not observation_restored:
            raise RuntimeError("checkpoint restore changed the policy observation")
        with torch.inference_mode():
            environment.step(runner.get_inference_policy()(environment.get_observations()))
    finally:
        environment.close()

    evaluated = evaluate_job(
        output,
        7,
        0.0,
        bank=test_bank,
        checkpoint=output / "amp_0/seed_7/policy.pt",
        horizon=4,
    )
    effective = resolve_config(
        ROOT / "configs/default.json",
        ROOT / "configs/experiments/night_fullcircle_5mm_v1.json",
        [
            "smoke.num_envs=8",
            "smoke.iterations=2",
            "smoke.steps_per_update=4",
            "smoke.seed=7",
            "smoke.amplitude_deg=0.0",
        ],
    )
    write_effective_config(output / "effective_config.json", effective)
    result = {
        "passed": bool(observation_restored and evaluated["audit_passed"]),
        "training": trained,
        "checkpoint_restore_observation_identity": observation_restored,
        "evaluation": {
            "total": evaluated["total"],
            "successes": evaluated["successes"],
            "audit_passed": evaluated["audit_passed"],
        },
        "formal_training_samples": 0,
    }
    write_json(output / "smoke_result.json", result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
