"""Build and run the explicit SPIKE-013 V02/V03 experiment.

Usage: python scripts/spike013_v02_v03.py plan OUT [KEY=VALUE ...]
       python scripts/spike013_v02_v03.py prepare OUT {train,validation,test}
       python scripts/spike013_v02_v03.py train OUT SEED [--envs N] [--iterations N]
       python scripts/spike013_v02_v03.py evaluate OUT SEED {validation,test} [--checkpoint PATH]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
from pathlib import Path

from mjlab_contact_prep.config import ContactConfig
from mjlab_contact_prep.configuration import write_effective_config
from mjlab_contact_prep.night_run.environment import prepare_bank, save_json
from mjlab_contact_prep.night_run.jobs import train_job, evaluate_job, verify_frozen
from mjlab_contact_prep.night_run.shallow_hold import PROTOCOL_ID
from mjlab_contact_prep.night_run.spike013 import ROOT, cases, fingerprint, load_experiment


def source_hashes():
    paths = sorted([*ROOT.glob("src/mjlab_contact_prep/**/*.py"),
                    *ROOT.glob("assets/**/*"), *ROOT.glob("configs/*.json"),
                    ROOT / "scripts/spike013_v02_v03.py"])
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths if path.is_file()}


def create_plan(out, overrides):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    document, contact = load_experiment(overrides)
    effective = document["effective"]
    groups = cases(effective)
    hashes = source_hashes()
    for name, rows in groups.items():
        save_json(out / f"{name}_cases.json", rows)
    write_effective_config(out / "effective_config.json", document)
    source_dir = out / "source"
    for relative in hashes:
        dest = source_dir / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, dest)
    git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    diff = subprocess.run(["git", "diff", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    (out / "tracked.diff").write_text(diff)
    plan = dict(protocol_id=PROTOCOL_ID, effective_config=effective,
                experiment_source_sha256=hashes, physics_frozen={},
                head=git_head, tracked_diff_sha256=hashlib.sha256(diff.encode()).hexdigest(),
                training_fingerprint=fingerprint(effective, groups["train"], hashes),
                validation_fingerprint=fingerprint(effective, groups["validation"], hashes),
                test_fingerprint=fingerprint(effective, groups["test"], hashes),
                num_envs=64, iterations=1, horizon_steps=round(effective["search_s"] / (contact.dt * 20 * 5)),
                preparation_s=effective["preparation_s"], steps_per_update=32,
                train_cases=groups["train"], validation_cases=groups["validation"], test_cases=groups["test"],
                queue_order=[dict(amplitude=0., seed=7)])
    save_json(out / "plan.json", plan)
    dependencies={name:importlib.metadata.version(name) for name in
                  ("mjlab","mujoco","mujoco-warp","torch","rsl-rl-lib","numpy")}
    save_json(out / "manifest.json", dict(protocol_id=PROTOCOL_ID, head=git_head,
              tracked_diff_sha256=plan["tracked_diff_sha256"], source_hashes=hashes,
              dependencies=dependencies,physics_backend="contact-fix",asset_hashes={k:v for k,v in hashes.items() if k.startswith("assets/")},
              seeds={"train":effective["train_seed"],"test":effective["test_seed"]},
              fingerprints={key:plan[key] for key in ("training_fingerprint","validation_fingerprint","test_fingerprint")},
              command=" ".join(__import__("sys").argv)))
    return plan


def checked_plan(out):
    plan = json.loads((out / "plan.json").read_text())
    if plan.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("not a SPIKE-013 V02/V03 plan")
    verify_frozen(plan)
    return plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("plan", "prepare", "train", "evaluate", "report"))
    parser.add_argument("out", type=Path)
    parser.add_argument("arguments", nargs="*")
    parser.add_argument("--envs", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if args.stage == "plan":
        create_plan(args.out, args.arguments)
        return
    plan = checked_plan(args.out)
    if args.stage == "report":
        evaluations=sorted(args.out.glob("amp_0/seed_*/test/results.json"))
        lines=["# SPIKE-013 V02/V03 结果", "", f"协议：{PROTOCOL_ID}", "",
               f"配置：1 mm/s 策略 XY 请求，五桶 0.5–10 mm，{plan['effective_config']['search_s']:g} s 寻孔预算。", ""]
        if not evaluations:
            lines.append("最终测试尚未运行；不得宣称 10 mm 策略达标。")
        for path in evaluations:
            result=json.loads(path.read_text())
            lines += [f"## {path.parent.parent.name}", "",
              f"全分母 PPO 稳定成功：{result['ppo_successes']}/{result['total']}；合格且非准备捕获分母：{result['ppo_successes']}/{result['eligible_nonprepcapture']}。",
              f"准备失败：{result['preparation_failures']}；准备提前捕获：{result['preparation_captures']}；准备捕获后保持成功：{result['preparation_capture_hold_successes']}。", "",
              "| 名义半径 mm | PPO 成功/全部 | 实际接手半径范围 mm |", "|---:|---:|---:|"]
            for radius in (1,2,5,7.5,10):
                rows=[x for x in result['rows'] if x['kind']=='grid' and x['radius_mm']==radius]
                actual=[x['actual_handoff_radius_mm'] for x in rows if 'actual_handoff_radius_mm' in x]
                span=f"{min(actual):.3f}–{max(actual):.3f}" if actual else "未记录"
                lines.append(f"| {radius:g} | {sum(x['outcome']=='success' for x in rows)}/{len(rows)} | {span} |")
            lines.append("")
        (args.out / "REPORT.md").write_text("\n".join(lines)+"\n")
        return
    if args.stage == "prepare":
        if len(args.arguments) != 1 or args.arguments[0] not in ("train", "validation", "test"):
            parser.error("prepare requires train, validation or test")
        group = args.arguments[0]
        key = {"train":"training", "validation":"validation", "test":"test"}[group] + "_fingerprint"
        destination = args.out / f"{group}_bank"
        prepare_bank(plan[f"{group}_cases"], destination, contact=ContactConfig(**plan["effective_config"]["contact"]),
                     preparation_s=plan["preparation_s"], fingerprint=plan[key])
        return
    if args.stage == "train":
        if len(args.arguments) != 1:parser.error("train requires SEED")
        train_job(args.out, int(args.arguments[0]), 0., bank=args.out / "train_bank/bank.pt",
                  n=args.envs, iterations=args.iterations)
        return
    if len(args.arguments) != 2 or args.arguments[1] not in ("validation", "test"):
        parser.error("evaluate requires SEED and validation or test")
    seed = int(args.arguments[0]);group = args.arguments[1]
    evaluate_job(args.out, seed, 0., bank=args.out / f"{group}_bank/bank.pt",
                 checkpoint=args.checkpoint, tag=group)


if __name__ == "__main__":
    main()
