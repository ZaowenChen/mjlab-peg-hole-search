"""Detached, single-GPU SPIKE-013 256-env training and validation queue."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from mjlab_contact_prep.data.state_bank import load_state_bank
from mjlab_contact_prep.night_run.jobs import evaluate_job, train_job, verify_frozen
from mjlab_contact_prep.night_run.spike013 import ROOT
from spike013_v02_v03 import create_plan

SOURCE = ROOT / "evaluation/spike013_1mms_10mm/20260924_ppo_small_v1"
BASE = ROOT / "evaluation/spike013_1mms_10mm"
UPDATES = (64, 128, 192, 256)
SEEDS = (7, 17, 27)
RADII = (1, 2, 5, 7.5, 10)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def put(path: Path, data: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def prepare(out: Path) -> None:
    if out.exists():
        raise FileExistsError(f"output already exists: {out}")
    old = json.loads((SOURCE / "plan.json").read_text())
    verify_frozen(old)
    for group in ("train", "validation"):
        bank = SOURCE / f"{group}_bank/bank.pt"
        raw = load_state_bank(bank)
        expected = old[f"{'training' if group == 'train' else group}_fingerprint"]
        if raw["fingerprint"] != expected or len(raw["rows"]) != len(old[f"{group}_cases"]):
            raise ValueError(f"source {group} bank identity mismatch")
        if [r["id"] for r in raw["rows"]] != [r["id"] for r in old[f"{group}_cases"]]:
            raise ValueError(f"source {group} case list mismatch")
    if sum(r["eligible"] and not r.get("preparation_captured", False)
           for r in load_state_bank(SOURCE / "train_bank/bank.pt")["rows"]) != 2390:
        raise ValueError("training bank eligibility changed")
    validation = load_state_bank(SOURCE / "validation_bank/bank.pt")["rows"]
    if len(validation) != 40 or not all(r["eligible"] and not r.get("preparation_captured", False) for r in validation):
        raise ValueError("validation bank eligibility changed")
    free = shutil.disk_usage(BASE).free
    if free < 100 * 1024**3:
        raise OSError(f"less than 100 GiB free: {free}")

    plan = create_plan(out, ())
    for key in ("training_fingerprint", "validation_fingerprint", "test_fingerprint"):
        if plan[key] != old[key]:
            raise ValueError(f"new {key} differs from source; refusing to reuse bank")
    for group in ("train", "validation", "test"):
        if plan[f"{group}_cases"] != old[f"{group}_cases"]:
            raise ValueError(f"new {group} case list differs from source")
    plan.update(num_envs=256, iterations=256, steps_per_update=32,
                samples_per_seed=256 * 256 * 32,
                queue_order=[{"amplitude": 0., "seed": seed} for seed in SEEDS],
                validation_updates=list(UPDATES))
    put(out / "plan.json", plan)
    copied = {}
    for group in ("train", "validation"):
        dst = out / f"{group}_bank"
        dst.mkdir()
        for name in ("bank.pt", "summary.json", "preparation.json", "progress.json"):
            src_file = SOURCE / f"{group}_bank" / name
            if src_file.exists():
                shutil.copy2(src_file, dst / name)
                if digest(src_file) != digest(dst / name):
                    raise IOError(f"copy hash mismatch: {src_file}")
                copied[f"{group}_bank/{name}"] = digest(src_file)
    manifest = json.loads((out / "manifest.json").read_text())
    manifest.update(queue_script_sha256=digest(Path(__file__)), source_experiment=str(SOURCE),
                    copied_file_sha256=copied, num_envs=256, iterations=256,
                    steps_per_update=32, queue_order=plan["queue_order"],
                    validation_updates=list(UPDATES), prepared_at=now(),
                    launch_command=f"bash scripts/run_local.sh python scripts/spike013_overnight.py run {out}")
    put(out / "manifest.json", manifest)
    put(out / "queue_status.json", {"state": "prepared", "updated_at": now(), "stages": [], "worker_pid": None})


def summarize(out: Path) -> None:
    runs = []
    for seed in SEEDS:
        for update in UPDATES:
            folder = out / f"amp_0/seed_{seed}/validation_u{update:04d}"
            result = json.loads((folder / "results.json").read_text())
            if result["total"] != 40 or result["eligible_nonprepcapture"] != 40 or not result["audit_passed"]:
                raise ValueError(f"validation audit or denominator failed: {folder}")
            data = np.load(folder / "trajectory.npz")
            trace, columns = data["trace"], list(data["columns"])
            radial = columns.index("radial_m")
            classes = Counter()
            for index, row in enumerate(result["rows"]):
                samples = row["terminal_samples"]
                closest = float(trace[:samples, index, radial].min() * 1000) if samples else float("inf")
                if row["outcome"] == "success":
                    category = "success"
                elif not row["prep_eligible"]:
                    category = "preparation_failed"
                elif row.get("preparation_captured"):
                    category = "preparation_capture_incomplete"
                elif row.get("capture_count", 0) or row.get("hold_break_count", 0):
                    category = "capture_then_lost"
                elif row["outcome"] == "fault" and closest <= 1:
                    category = "near_hole_fault"
                elif row["outcome"] == "timeout":
                    category = "timeout_near_hole" if closest <= 1 else "timeout_far"
                else:
                    category = "fault_far"
                classes[category] += 1
            runs.append({"seed": seed, "update": update, "successes": result["ppo_successes"],
                         "total": 40, "eligible_nonprepcapture": 40, "faults": result["faults"],
                         "timeouts": sum(r["outcome"] == "timeout" for r in result["rows"]),
                         "mean_capped_time_s": result["mean_capped_time_s"],
                         "preparation_failures": result["preparation_failures"],
                         "preparation_captures": result["preparation_captures"],
                         "categories": dict(classes),
                         "by_radius": {str(radius): {"successes": sum(r["outcome"] == "success" for r in result["rows"] if r["radius_mm"] == radius), "total": 8} for radius in RADII},
                         "policy_request_peak_mps": max(r.get("policy_request_peak_mps", 0) for r in result["rows"]),
                         "measured_xy_speed_peak_mps": max(r.get("measured_xy_speed_peak_mps", 0) for r in result["rows"]),
                         "audit_passed": True})
    ranked = sorted(runs, key=lambda r: (-r["successes"], r["faults"], r["mean_capped_time_s"], r["update"]))
    put(out / "summary.json", {"complete": True, "runs": runs, "exploratory_choice": {"seed": ranked[0]["seed"], "update": ranked[0]["update"]}, "independent_280_case_test_run": False, "combined_vx_vy_valid": False})
    lines = ["# SPIKE-013 256 环境三种子验证汇总", "", "12 次验证均为同一 40 案，在线结果已与 2 ms 轨迹复算。280 案独立最终测试未运行。", "", "combined_vx/vy 记录语义无效；策略请求与实测速度分别见 summary.json。", "", "| 种子 | 更新 | 成功/40 | 故障 | 超时 | 平均封顶耗时 s | 1/2/5/7.5/10 mm 成功 |", "|---:|---:|---:|---:|---:|---:|---|"]
    for r in runs:
        radius = "/".join(str(r["by_radius"][str(x)]["successes"]) for x in RADII)
        lines.append(f"| {r['seed']} | {r['update']} | {r['successes']}/40 | {r['faults']} | {r['timeouts']} | {r['mean_capped_time_s']:.2f} | {radius} |")
    lines += ["", f"探索性选择：种子 {ranked[0]['seed']}，第 {ranked[0]['update']} 次更新。按成功数、故障数、失败计 40 s 的平均耗时、较早检查点排序。", "", "各验证目录保留逐案结果、故障与保持分类所需字段及轨迹；summary.json 包含分类计数。", ""]
    (out / "REPORT.md").write_text("\n".join(lines))


def run(out: Path) -> None:
    lock = (out / "queue.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = out / "queue_status.json"
    status = json.loads(status_path.read_text())
    if status["state"] != "prepared":
        raise ValueError(f"queue cannot start in state {status['state']}")
    plan = json.loads((out / "plan.json").read_text())
    verify_frozen(plan)
    status.update(state="running", queue_pid=os.getpid(), started_at=now(), updated_at=now())
    put(status_path, status)
    try:
        for seed in SEEDS:
            stages = [("train", None)] + [("evaluate", update) for update in UPDATES]
            for kind, update in stages:
                label = f"seed_{seed}_{kind}" + (f"_u{update:04d}" if update else "")
                cmd = [sys.executable, str(Path(__file__).resolve()), "worker", str(out), kind, str(seed)]
                if update:
                    cmd.append(str(update))
                log = out / f"{label}.log"
                record = {"label": label, "command": cmd, "log": str(log), "started_at": now(), "state": "running"}
                status["stages"].append(record)
                with log.open("w") as stream:
                    child = subprocess.Popen(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                    record["worker_pid"] = child.pid
                    status.update(worker_pid=child.pid, current_stage=label, updated_at=now())
                    put(status_path, status)
                    code = child.wait()
                record.update(exit_code=code, finished_at=now(), state="complete" if code == 0 else "failed")
                log_text = log.read_text(errors="replace")
                record["epa_warning_count"] = log_text.count("EPA horizon overflow")
                status.update(worker_pid=None, updated_at=now())
                put(status_path, status)
                if code:
                    raise RuntimeError(f"{label} exited {code}; see {log}")
        summarize(out)
        status.update(state="complete", finished_at=now(), current_stage=None, updated_at=now())
        put(status_path, status)
    except BaseException as exc:
        status.update(state="failed", error=repr(exc), traceback=traceback.format_exc(),
                      worker_pid=None, finished_at=now(), updated_at=now())
        put(status_path, status)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run", "worker"))
    parser.add_argument("out", type=Path)
    parser.add_argument("kind", nargs="?")
    parser.add_argument("seed", nargs="?", type=int)
    parser.add_argument("update", nargs="?", type=int)
    args = parser.parse_args()
    out = args.out.resolve()
    if args.action == "prepare":
        prepare(out)
    elif args.action == "run":
        run(out)
    elif args.kind == "train":
        train_job(out, args.seed, 0., bank=out / "train_bank/bank.pt", n=256,
                  iterations=256, steps=32, resume=False)
    elif args.kind == "evaluate" and args.update in UPDATES:
        evaluate_job(out, args.seed, 0., bank=out / "validation_bank/bank.pt",
                     checkpoint=out / f"amp_0/seed_{args.seed}/checkpoint_{args.update:04d}.pt",
                     tag=f"validation_u{args.update:04d}")
    else:
        parser.error("invalid worker arguments")


if __name__ == "__main__":
    main()
