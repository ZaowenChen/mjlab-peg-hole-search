#!/usr/bin/env python3
"""Versioned SPIKE-012 first-stop analysis; never rewrites c/d/e summaries."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from scripts.spike012_execution import OUT, sha, write_json

DT = .002


def low_force_segments(fz, threshold=5., dt=DT):
    bad = np.asarray(fz) < threshold
    starts = np.flatnonzero(bad & ~np.r_[False, bad[:-1]])
    ends = np.flatnonzero(bad & ~np.r_[bad[1:], False]) + 1
    segments = [dict(start_sample=int(s), end_sample_exclusive=int(e), duration_s=float((e-s)*dt),
                     left_truncated=bool(s == 0), right_truncated=bool(e == len(bad)))
                for s, e in zip(starts, ends)]
    return dict(cumulative_s=float(bad.sum()*dt),
                longest_continuous_s=float(max((x["duration_s"] for x in segments), default=0.)),
                segment_count=len(segments), segments=segments)


def tip_velocity(x):
    control = x["control_tip_pre_m"][:, 0]
    geom = x["geometry_tip_pre_m"][:, 0]
    # The existing trace's executed_twist is J(q) qdot at the control point.
    # Rigid-body velocity transfers that velocity to the geometry tip.
    executed = x["executed_twist_pre"][:, 0]
    kinematic = executed[:, :3] + np.cross(executed[:, 3:6], geom-control)
    finite_difference = np.gradient(geom, DT, axis=0)
    return np.linalg.norm(kinematic[:, :2], axis=1)*1000, np.linalg.norm(finite_difference[:, :2], axis=1)*1000


def difference_speed_20ms(x):
    geom = x["geometry_tip_pre_m"][:, 0, :2]
    n = 10
    v = np.linalg.norm((geom[n:]-geom[:-n])/(n*DT), axis=1)*1000
    return np.r_[np.repeat(v[0], n), v]


def stop_time(x, e0):
    kin, _ = tip_velocity(x)
    diff = difference_speed_20ms(x)
    kin_floor = e0["kinematic_p99_mm_s"]
    diff_floor = e0["difference_20ms_p99_mm_s"]
    if max(kin_floor, diff_floor) >= .02:
        return dict(time_s=None, status="unresolvable_E0_floor",
                    E0_kinematic_p99_mm_s=kin_floor, E0_difference_20ms_p99_mm_s=diff_floor)
    dwell = 100
    good = (kin < .02) & (diff < .02)
    continuous = np.convolve(good.astype(int), np.ones(dwell, dtype=int), mode="valid")
    found = np.flatnonzero(continuous == dwell)
    return dict(time_s=float(found[0]*DT) if len(found) else None,
                status="confirmed" if len(found) else "not_confirmed_in_2s",
                E0_kinematic_p99_mm_s=kin_floor, E0_difference_20ms_p99_mm_s=diff_floor)


def peak(values, t, mode="max"):
    idx = int(np.argmax(values) if mode == "max" else np.argmin(values))
    return dict(value=float(values[idx]), time_s=float(t[idx]))


def window_summary(x):
    t = x["time_pre_s"]
    geom = x["geometry_tip_pre_m"][:, 0]
    delta = (geom-geom[0])*1000
    f = x["wrench_target_pre"][:, 0]
    radial = np.linalg.norm(f[:, :2], axis=1)
    moment = np.linalg.norm(f[:, 3:6], axis=1)
    rot = x["geometry_rot_pre"][:, 0].reshape(-1, 3, 3)
    rel = np.einsum("ij,tjk->tik", rot[0].T, rot)
    angle = np.rad2deg(np.arccos(np.clip((np.trace(rel, axis1=1, axis2=2)-1)/2, -1, 1)))
    result = dict(samples=len(t), start_s=float(t[0]), end_exclusive_s=float(t[-1]+DT),
        Dmax_mm=float(np.linalg.norm(delta[:, :2], axis=1).max()),
        net_xy_mm=delta[-1, :2].tolist(), net_xy_norm_mm=float(np.linalg.norm(delta[-1, :2])),
        path_mm=float(np.linalg.norm(np.diff(geom[:, :2], axis=0), axis=1).sum()*1000),
        xy_range_mm=(np.ptp(geom[:, :2], axis=0)*1000).tolist(),
        z_net_mm=float(delta[-1, 2]), z_range_mm=float(np.ptp(geom[:, 2])*1000),
        attitude_max_deg=float(angle.max()),
        low_force=low_force_segments(f[:, 2]), fz_min_N=peak(f[:, 2], t, "min"),
        fz_max_N=peak(f[:, 2], t), radial_peak_N=peak(radial, t),
        moment_peak_Nm=peak(moment, t),
        fault_count=int(np.count_nonzero(x["reason_post"][:, 0])),
        accel_limited_fraction=float(np.mean(x["accel_limited"][:, 0])),
        energy_limited_fraction=float(np.mean(x["energy_limited"][:, 0])),
        reference_delta_mm=((x["reference_post_m"][-1, 0]-x["reference_pre_m"][0, 0])*1000).tolist(),
        command_joint_delta_rad=(x["command"][-1, 0]-x["command_pre_rad"][0, 0]).tolist(),
        ctrl_joint_delta_rad=(x["ctrl_written"][-1, 0]-x["ctrl_pre"][0, 0]).tolist(),
        lead_xy_start_mm=(x["lead"][0, 0, :2]*1000).tolist(),
        lead_xy_end_mm=(x["lead"][-1, 0, :2]*1000).tolist())
    kin, fd = tip_velocity(x)
    result["speed_kinematic_mm_s"] = dict(start=float(kin[0]), median=float(np.median(kin)),
                                           p95=float(np.percentile(kin, 95)), end=float(kin[-1]))
    result["speed_difference_mm_s"] = dict(start=float(fd[0]), median=float(np.median(fd)),
                                          p95=float(np.percentile(fd, 95)), end=float(fd[-1]))
    fd20 = difference_speed_20ms(x)
    result["speed_difference_20ms_mm_s"] = dict(start=float(fd20[0]), median=float(np.median(fd20)),
                                                 p95=float(np.percentile(fd20, 95)), end=float(fd20[-1]))
    return result


def calibrate(batch):
    result = dict(protocol="spike012-stop-v2", source=str(batch), scenarios={})
    for scenario in ("free", "plane"):
        path = batch / f"{scenario}_E0_v2" / "trace.npz"
        x = np.load(path)
        kin, fd = tip_velocity(x)
        result["scenarios"][scenario] = dict(trace_sha256=sha(path),
            kinematic_p95_mm_s=float(np.percentile(kin, 95)),
            kinematic_p99_mm_s=float(np.percentile(kin, 99)),
            difference_p95_mm_s=float(np.percentile(fd, 95)),
            difference_p99_mm_s=float(np.percentile(fd, 99)),
            difference_20ms_p99_mm_s=float(np.percentile(difference_speed_20ms(x), 99)),
            legacy_threshold_mm_s=.02,
            interpretation="Binary 0.02 mm/s stop time is unresolved if the E0 noise floor exceeds it; report speed and displacement instead.")
    write_json(batch / "E0_calibration_v2.json", result)
    print(json.dumps(result, indent=2))


def analyze(batch):
    if not (batch / "E0_calibration_v2.json").exists():
        raise RuntimeError("Calibrate E0 before analyzing B")
    calibration = json.loads((batch / "E0_calibration_v2.json").read_text())
    # The measurement rule was frozen in stop_measurement_pre_F1.json before F1.
    for scenario in ("free", "plane"):
        e0 = np.load(batch / f"{scenario}_E0_v2" / "trace.npz")
        calibration["scenarios"][scenario]["difference_20ms_p99_mm_s"] = float(np.percentile(difference_speed_20ms(e0), 99))
    rows = []
    for folder in sorted(batch.glob("*_E*")):
        if not folder.is_dir() or not (folder / "prefix" / "trace.npz").exists():
            continue
        scenario, stage = folder.name.split("_")[:2]
        prefix_file = folder / "prefix" / "trace.npz"
        prefix = np.load(prefix_file)
        prefix_last = np.flatnonzero(prefix["time_pre_s"] >= 2.5)
        pkin, pfd = tip_velocity(prefix)
        prefix_speed = dict(kinematic_mean_mm_s=float(np.mean(pkin[prefix_last])),
                            difference_mean_mm_s=float(np.mean(pfd[prefix_last])),
                            requested_mm_s=1., within_10pct=bool(.9 <= np.mean(pkin[prefix_last]) <= 1.1))
        for run in sorted(folder.glob("r*_[ABG]")):
            file = run / "trace.npz"
            x = np.load(file)
            # Recreate the complete force window directly; other full-run fields remain separately auditable.
            full_fz = np.concatenate((prefix["wrench_target_pre"][:, 0, 2], x["wrench_target_pre"][:, 0, 2]))
            full_t = np.concatenate((prefix["time_pre_s"], x["time_pre_s"]))
            full_force = np.concatenate((prefix["wrench_target_pre"][:, 0], x["wrench_target_pre"][:, 0]))
            stop = window_summary(x)
            stop["stop_time"] = stop_time(x, calibration["scenarios"][scenario])
            full_radial = np.linalg.norm(full_force[:, :2], axis=1)
            full_moment = np.linalg.norm(full_force[:, 3:6], axis=1)
            whole_force = dict(low_force=low_force_segments(full_fz), fz_min_N=peak(full_fz, full_t, "min"),
                               fz_max_N=peak(full_fz, full_t), radial_peak_N=peak(full_radial, full_t),
                               moment_peak_Nm=peak(full_moment, full_t))
            events = json.loads((run / "events.json").read_text())
            row = dict(run_id=run.name, scenario=scenario, stage=stage, arm=run.name[-1],
                       prefix_sha256=sha(prefix_file), trace_sha256=sha(file),
                       prestop_sha256=json.loads((run / "branch_audit.json").read_text())["prestop_sha256"],
                       prefix_tracking=prefix_speed, stop=stop, full_force=whole_force,
                       stop_events=[e for e in events if e["event"] == "stop_requested"])
            rows.append(row)
    write_json(batch / "analysis_v2.json", dict(version="spike012-stop-v2", analysis_sha256=sha(__file__),
                                            calibration_sha256=sha(batch / "E0_calibration_v2.json"), runs=rows))
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["scenario"], row["stage"], row["arm"])].append(row)
    lines = ["# SPIKE-012 F0/F1 同停车前状态复验", "", "每个分支恢复同一 t=3 s 快照，仅分析 [3,5) s 首次停车。", "",
             "| 场景 | 链路 | 组 | n | Dmax 中位/范围 mm | 停车低力最长连续 s | Fz 峰值 N | 前缀跟踪 mm/s |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for key, group in sorted(grouped.items()):
        ds = [r["stop"]["Dmax_mm"] for r in group]
        ls = [r["stop"]["low_force"]["longest_continuous_s"] for r in group]
        fs = [r["stop"]["fz_max_N"]["value"] for r in group]
        v = group[0]["prefix_tracking"]["kinematic_mean_mm_s"]
        lines.append(f"| {' | '.join(key)} | {len(group)} | {np.median(ds):.3f} / {min(ds):.3f}–{max(ds):.3f} | {max(ls):.3f} | {max(fs):.2f} | {v:.3f} |")
    lines += ["", "Dmax、Fz 和最长连续低力只作诊断；完整逐样本数据及全部指标见 analysis_v2.json。",
              "停稳时间沿用 0.02 mm/s 历史口径作可辨识性判断，E0 校准见 E0_calibration_v2.json。"]
    (batch / "REPORT_v2.md").write_text("\n".join(lines)+"\n")
    print("analyzed", len(rows))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=("calibrate", "analyze"))
    p.add_argument("--batch", required=True, type=Path)
    a = p.parse_args()
    (calibrate if a.phase == "calibrate" else analyze)(a.batch)
