"""Evaluate afternoon and overnight 0.2 mm/s PPO weights on one frozen bank."""

import copy
import hashlib
import json
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from mjlab_contact_prep.config import ContactConfig
from mjlab_contact_prep.night_run.environment import BankEnv, save_json
from mjlab_contact_prep.night_run.jobs import ppo_config
from evaluate_xy_handoff import COLS, HandoffProbe, summarize_case


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "evaluation/spike011_small_1mms_v1/test"
OUT = ROOT / "evaluation/low_speed_budget_comparison_v1"
DT = .002


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate(family, seed, rows):
    if family == "afternoon":
        checkpoint = ROOT / f"evaluation/gpu_expanded_ab_v3/seed_{seed}_amp_0/policy.pt"
    elif family == "overnight_early":
        checkpoint = ROOT / f"evaluation/night_fullcircle_5mm_v1/amp_0/seed_{seed}/checkpoint_0016.pt"
    else:
        checkpoint = ROOT / f"evaluation/night_fullcircle_5mm_v1/amp_0/seed_{seed}/policy.pt"
    destination = OUT / f"{family}_seed_{seed}"
    destination.mkdir(parents=True, exist_ok=False)
    env = BankEnv(SOURCE / "bank/bank.pt", len(rows), 0., seed,
                  autoreset=False, audit=False, horizon=203)
    config = replace(ContactConfig(), xy_speed=.0002)
    env.term.cfg.contact = config
    env.term.controller.cfg = config
    ids = torch.arange(len(rows), device=env.device)
    env.restore(ids, ids)
    radius = env.metrics()[0].detach().cpu().numpy()
    probe = HandoffProbe(env, "A", radius, slow_cap_mm_s=.2)
    invalid = torch.tensor([r["prep_class"] == "failed" for r in rows],
                           dtype=torch.bool, device=env.device)
    prepared_capture = torch.tensor([r["prep_class"] == "captured" for r in rows],
                                    dtype=torch.bool, device=env.device)
    probe.finished[invalid] = True
    runner = OnPolicyRunner(env, copy.deepcopy(ppo_config(seed, 1)), None, device=env.device)
    runner.load(str(checkpoint), map_location=env.device)
    policy = runner.get_inference_policy()
    start = time.monotonic()
    try:
        with torch.inference_mode():
            for _ in range(203):
                probe.decision_start()
                action = policy(env.get_observations()).clamp(-1, 1)
                action[probe.finished | prepared_capture] = 0
                env.step(action)
                newly = (env.term.controller.reason != 0) & ~probe.finished
                probe.finished |= newly
                probe.finish_tick[newly] = env.ticks[newly] - 1
                if bool(probe.finished.all()):
                    break
        trace = np.stack(probe.trace)
        if not np.isfinite(trace).all():
            raise FloatingPointError("non-finite evaluation trajectory")
        assert np.allclose(trace[:, 0, 0], np.arange(len(trace)) * DT, atol=5e-6)
        req = np.linalg.norm(trace[:, :, [COLS.index("requested_vx_m_s"),
                                           COLS.index("requested_vy_m_s")]], axis=-1)
        assert float(req.max()) <= .000201
        cases = [summarize_case(trace[:, i], row, probe, i, "A")
                 for i, row in enumerate(rows)]
        for case, row in zip(cases, rows):
            case["ppo_success"] = bool(case["hold_passed"] and row["prep_class"] == "ready")
            case["pipeline_success"] = bool(case["hold_passed"] and row["prep_class"] != "failed")
            if case["first_capture_s"] is not None and case["first_capture_s"] > 40.:
                case["ppo_success"] = case["pipeline_success"] = False
                case["failure"] = "near_timeout"
                case["capped_capture_s"] = 40.
            if row["prep_class"] == "failed":
                case["failure"] = "preparation_failed"
                case["capped_capture_s"] = 40.
        save_json(destination / "results.json", {
            "family": family, "seed": seed, "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha(checkpoint), "requested_cap_mm_s": .2,
            "bank_sha256": sha(SOURCE / "bank/bank.pt"),
            "evaluator_sha256": sha(__file__), "wall_s": time.monotonic() - start,
            "cases": cases,
        })
        print(f"{family} seed {seed}: {sum(c['pipeline_success'] for c in cases)}/{len(cases)} "
              f"in {time.monotonic() - start:.1f}s", flush=True)
    finally:
        env.close()


def summarize(rows):
    successes = [r for r in rows if r["pipeline_success"]]
    ready = [r for r in rows if r["prep_class"] == "ready"]
    return {
        "success": len(successes), "total": len(rows),
        "ppo_success": sum(r["ppo_success"] for r in rows), "ready": len(ready),
        "mean_capped_s": float(np.mean([r["capped_capture_s"] for r in rows])),
        "median_success_s": float(np.median([r["first_capture_s"] for r in successes])),
        "failures": dict(Counter(r["failure"] for r in rows if not r["pipeline_success"])),
    }


def report(rows):
    cases = {(family, seed): json.loads((OUT / f"{family}_seed_{seed}/results.json").read_text())["cases"]
             for family in ("afternoon", "overnight") for seed in (7, 17, 27)}
    ids = [r["id"] for r in rows]
    assert all([r["id"] for r in value] == ids for value in cases.values())
    records = []
    for seed in (7, 17, 27):
        for subset_name, select in (
            ("all", lambda r: True),
            ("near_original_sectors", lambda r: 55 <= r["angle_deg"] <= 125 or 235 <= r["angle_deg"] <= 305),
            ("other_sectors", lambda r: not (55 <= r["angle_deg"] <= 125 or 235 <= r["angle_deg"] <= 305)),
            ("nominal_0p5mm", lambda r: r["kind"] == "grid" and r["radius_mm"] == .5),
            ("nominal_1mm", lambda r: r["kind"] == "grid" and r["radius_mm"] == 1.),
            ("nominal_2mm", lambda r: r["kind"] == "grid" and r["radius_mm"] == 2.),
            ("nominal_5mm", lambda r: r["kind"] == "grid" and r["radius_mm"] == 5.),
        ):
            a = [r for r in cases["afternoon", seed] if select(r)]
            b = [r for r in cases["overnight", seed] if select(r)]
            assert [r["id"] for r in a] == [r["id"] for r in b]
            common = [(x, y) for x, y in zip(a, b)
                      if x["pipeline_success"] and y["pipeline_success"]]
            records.append({
                "seed": seed, "subset": subset_name,
                "afternoon": summarize(a), "overnight": summarize(b),
                "both_success_n": len(common),
                "mean_paired_overnight_minus_afternoon_s":
                    float(np.mean([y["first_capture_s"] - x["first_capture_s"] for x, y in common]))
                    if common else None,
                "median_paired_overnight_minus_afternoon_s":
                    float(np.median([y["first_capture_s"] - x["first_capture_s"] for x, y in common]))
                    if common else None,
                "overnight_faster_n": sum(y["first_capture_s"] < x["first_capture_s"] for x, y in common),
                "afternoon_faster_n": sum(y["first_capture_s"] > x["first_capture_s"] for x, y in common),
            })
    save_json(OUT / "summary.json", {
        "test_bank": str(SOURCE / "bank/bank.pt"),
        "test_bank_sha256": sha(SOURCE / "bank/bank.pt"),
        "preparation": dict(Counter(r["prep_class"] for r in rows)),
        "afternoon_samples": 147456, "overnight_samples": 2097152,
        "records": records,
    })
    print(json.dumps([r for r in records if r["subset"] in ("all", "near_original_sectors", "other_sectors")],
                     ensure_ascii=False, indent=2), flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = json.loads((SOURCE / "preparation_audit.json").read_text())
    assert len(rows) == 48
    assert sha(SOURCE / "bank/bank.pt") == json.loads((SOURCE / "manifest.json").read_text())["bank_sha256"]
    for family in ("afternoon", "overnight"):
        for seed in (7, 17, 27):
            evaluate(family, seed, rows)
    report(rows)


if __name__ == "__main__":
    main()
