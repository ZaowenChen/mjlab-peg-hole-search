#!/usr/bin/env python3
"""Summarize the read-only SPIKE-012 E0/E1/E2 GPU traces."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def summarize_run(folder):
    meta = json.loads((folder / "run.json").read_text())
    m = json.loads((folder / "metrics.json").read_text())
    d = np.load(folder / "trace.npz")
    p = d["geometry_tip_pre_m"][:, 0, :]
    q = d["measured_q"][:, 0, :]
    cmd = d["command"][:, 0, :]
    t = d["time_pre_s"]
    velocity = np.gradient(p[:, :2], .002, axis=0)
    speed = np.linalg.norm(velocity, axis=1)
    result = dict(run_id=folder.name, **meta, fault_reason=m["fault_reason"],
        peak_axial_N=float(np.max(np.abs(d["wrench_target_pre"][:, 0, 2]))),
        peak_radial_N=float(np.max(np.linalg.norm(d["wrench_target_pre"][:, 0, :2], axis=-1))),
        ready_fraction=float(d["ready_post"].mean()),
        xy_range_mm=(np.ptp(p[:, :2], axis=0) * 1000).tolist(),
        zero_speed_p95_mm_s=float(np.percentile(speed, 95) * 1000) if meta["waveform"] == "E0" else None,
        stops=[])
    for stop in m["stops"]:
        k = round(stop["stop_t0_s"] / .002)
        end = min(k + 1000, len(t) - 1)
        before = max(0, k - 250)
        ref = d["reference_post_m"][k, 0, :2]
        initial = p[k, :2]
        move = velocity[before:k]
        request = d["final_twist"][k, 0, :2]
        lead = d["lead"][k, 0, :2]
        result["stops"].append(dict(**stop,
            reference_error_t0_mm=float(np.linalg.norm(ref - initial) * 1000),
            final_xy_request_t0_mm_s=float(np.linalg.norm(request) * 1000),
            command_lead_t0_mm=float(np.linalg.norm(lead) * 1000),
            command_change_next_0p5s_rad=float(np.linalg.norm(cmd[min(k + 250, len(t)-1)] - cmd[k])),
            actual_joint_change_next_0p5s_rad=float(np.linalg.norm(q[min(k + 250, len(t)-1)] - q[k])),
            moving_speed_mean_last_0p5s_mm_s=float(np.linalg.norm(move.mean(axis=0)) * 1000),
            speed_at_t0_mm_s=float(speed[k] * 1000),
            end_k=end))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("batch", type=Path)
    args = p.parse_args()
    rows = [summarize_run(folder) for folder in sorted(args.batch.glob("*_r*"))
            if (folder / "run.json").is_file()]
    write(args.batch / "analysis.json", rows)
    by = defaultdict(list)
    for row in rows:
        if row["waveform"] != "W1":
            continue
        key = (row["scenario"], row["stage"], row["direction"], row["speed_mm_s"])
        by[key].extend(row["stops"])
    lines = ["# SPIKE-012 执行链筛查摘要", "", "原始轨迹为 GPU 后端每 2 ms 的采样；采样值是积分前状态及当前新写入命令。", "",
             "| 场景 | 层次 | 方向 | 输入 mm/s | 停止窗数 | Dmax 中位 mm | Dmax 范围 mm | 停前实际均速中位 mm/s | t0 最终横向请求中位 mm/s | t0 命令超前中位 mm |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key, stops in sorted(by.items()):
        med = lambda n: float(np.median([x[n] for x in stops]))
        dm = [x["max_residual_mm"] for x in stops]
        lines.append(f"| {' / '.join(map(str,key[:3]))} | {key[3]:g} | {len(stops)} | {np.median(dm):.3f} | {min(dm):.3f}–{max(dm):.3f} | {med('moving_speed_mean_last_0p5s_mm_s'):.3f} | {med('final_xy_request_t0_mm_s'):.3f} | {med('command_lead_t0_mm'):.3f} |")
    # Build a correctly labelled compact table; the preceding loop creates the
    # numeric rows without converting raw trajectories or hiding failed runs.
    lines[4] = "| 场景 / 层次 / 方向 | 输入 mm/s | 停止窗数 | Dmax 中位 mm | Dmax 范围 mm | 停前实际均速中位 mm/s | t0 最终横向请求中位 mm/s | t0 命令超前中位 mm |"
    lines[5] = "|---|---:|---:|---:|---:|---:|---:|---:|"
    (args.batch / "SCREEN_SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("runs", len(rows), "W1 groups", len(by))


if __name__ == "__main__":
    main()
