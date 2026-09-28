#!/usr/bin/env python3
"""Paired A/B summary for the one guarded SPIKE-012 parking candidate."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("batch", type=Path)
    args = p.parse_args()
    rows = []
    for folder in sorted(args.batch.glob("*_W1_r*_[AB]")):
        m = json.loads((folder / "metrics.json").read_text())
        x = np.load(folder / "trace.npz")
        fz = x["wrench_target_pre"][:, 0, 2]
        radial = np.linalg.norm(x["wrench_target_pre"][:, 0, :2], axis=-1)
        force_moment = np.linalg.norm(x["wrench_target_pre"][:, 0, 3:6], axis=-1)
        geom = x["geometry_tip_pre_m"][:, 0, :]
        rows.append(dict(run_id=folder.name, arm=folder.name[-1],
            scenario=folder.name.split("_")[0], stage=folder.name.split("_")[1],
            direction=folder.name.split("_")[2], fault=m["fault_reason"],
            dmax_mm=[s["max_residual_mm"] for s in m["stops"]],
            fz_min_N=float(fz.min()), fz_max_N=float(fz.max()),
            below5_s=float((fz < 5).sum() * .002),
            radial_peak_N=float(radial.max()), moment_peak_Nm=float(force_moment.max()),
            z_range_mm=float(np.ptp(geom[:, 2]) * 1000)))
    (args.batch / "candidate_analysis.json").write_text(json.dumps(rows, indent=2) + "\n")
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["scenario"], r["stage"], r["direction"], r["arm"])].append(r)
    lines = ["# SPIKE-012 有限停车目标候选 A/B", "",
        "A 为原执行链；B 仅启用诊断性有限 XY 停车目标。每次从同一快照恢复，A/B 顺序交替。", "",
        "| 场景 / 层次 / 方向 / 组 | 回合 | Dmax 中位 mm | Dmax 范围 mm | Fz 峰值最大 N | Fz<5N 最长 s | 故障数 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for key, group in sorted(grouped.items()):
        values = [v for r in group for v in r["dmax_mm"]]
        lines.append(f"| {' / '.join(key)} | {len(group)} | {np.median(values):.3f} | {min(values):.3f}–{max(values):.3f} | {max(r['fz_max_N'] for r in group):.2f} | {max(r['below5_s'] for r in group):.3f} | {sum(r['fault'] != 0 for r in group)} |")
    lines += ["", "候选准入建议：1 mm/s Dmax ≤0.05 mm，正常移动不降速，载荷不超过原工作与保护限制且无新增支撑退化。未同时满足时保留诊断代码默认关闭。"]
    (args.batch / "CANDIDATE_SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("runs", len(rows))


if __name__ == "__main__":
    main()
