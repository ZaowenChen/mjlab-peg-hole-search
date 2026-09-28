#!/usr/bin/env python3
"""Analyze SPIKE-012 W2 reversal and E2 stop-gate controls."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def reversal(folder):
    data = np.load(folder / "trace.npz")
    axis = 0 if "_X_" in folder.name else 1
    p = data["geometry_tip_pre_m"][:, 0, axis]
    # 20 ms centered difference suppresses float32 position quantization.
    half = 5
    v = np.full(len(p), np.nan)
    v[half:-half] = (p[2 * half:] - p[:-2 * half]) / (.004 * half)
    sign = np.sign(np.nanmean(v[1250:1500]))
    if sign == 0:
        raise ValueError(folder)
    k = 1500  # W2 reverses at 3.0 s.
    reversed_speed = sign * v[k:2500] < -.00008
    stable = np.convolve(reversed_speed.astype(int), np.ones(100, int), "valid")
    index = np.flatnonzero(stable == 100)
    delay = float(index[0] * .002) if len(index) else None
    extra = float(max(0, np.max(sign * (p[k:2500] - p[k])) * 1000))
    return dict(run_id=folder.name, reversal_delay_s=delay,
                reversal_censored=delay is None, old_direction_extra_mm=extra)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("batch", type=Path)
    args = parser.parse_args()
    rows = [reversal(p) for p in sorted(args.batch.glob("*_W2_r*"))]
    gates = []
    for gated in sorted(args.batch.glob("plane_E2_gate_*_W1_r*")):
        base = args.batch / gated.name.replace("E2_gate", "E2")
        if not base.is_dir():
            raise FileNotFoundError(base)
        ga = json.loads((gated / "metrics.json").read_text())
        ba = json.loads((base / "metrics.json").read_text())
        for j, (a, b) in enumerate(zip(ga["stops"], ba["stops"])):
            gates.append(dict(gated=gated.name, base=base.name, stop_index=j,
                gate_dmax_mm=a["max_residual_mm"], base_dmax_mm=b["max_residual_mm"],
                gate_minus_base_mm=a["max_residual_mm"] - b["max_residual_mm"]))
    (args.batch / "extension_analysis.json").write_text(json.dumps(dict(reversal=rows, gate=gates), indent=2) + "\n")
    grouped = defaultdict(list)
    for r in rows:
        key = " / ".join(r["run_id"].split("_")[:4])
        grouped[key].append(r)
    lines = ["# SPIKE-012 反向和关 XY 通路对照", "",
        "反向时间按 20 ms 中心差分速度连续 0.2 s 低于反向 0.08 mm/s 判定；阈值高于 E0 平面噪声。", "",
        "| W2 场景 / 层次 / 方向 / 速度 | 反向延迟中位 s | 原方向额外位移中位 mm |",
        "|---|---:|---:|"]
    for key, group in sorted(grouped.items()):
        delays = [x["reversal_delay_s"] for x in group if x["reversal_delay_s"] is not None]
        lines.append(f"| {key} | {np.median(delays):.3f} | {np.median([x['old_direction_extra_mm'] for x in group]):.3f} |" if delays
                     else f"| {key} | >2.000 | {np.median([x['old_direction_extra_mm'] for x in group]):.3f} |")
    lines += ["", "| 平面 E2 关通路方向 | 基线 Dmax 中位 mm | 关通路 Dmax 中位 mm |", "|---|---:|---:|"]
    for direction in ("X", "Y"):
        subset = [x for x in gates if f"_{direction}_" in x["gated"]]
        if subset:
            lines.append(f"| {direction} | {np.median([x['base_dmax_mm'] for x in subset]):.3f} | {np.median([x['gate_dmax_mm'] for x in subset]):.3f} |")
    (args.batch / "EXTENSION_SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("W2", len(rows), "gate stops", len(gates))


if __name__ == "__main__":
    main()
