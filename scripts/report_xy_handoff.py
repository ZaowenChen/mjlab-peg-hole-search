"""Audit and report the frozen SPIKE-010 paired comparison."""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ARMS = ("A", "B", "C")
SEEDS = (7, 17, 27)
RADII = (1, 2, 5)


def fmt(value, places=2):
    return "—" if value is None else f"{value:.{places}f}"


def mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def aggregate(rows, arm):
    subset = [r for r in rows if r["arm"] == arm]
    outside = [r for r in subset if not r["initially_near"]]
    arrived = [r for r in outside if r["first_near_s"] is not None]
    high = [r for r in subset if r["high_speed_arrived"]]
    return {
        "total": len(subset), "hold_passed": sum(r["hold_passed"] for r in subset),
        "first_capture": sum(r["first_capture"] for r in subset),
        "outside": len(outside), "arrived": len(arrived),
        "high_arrival_total": len(high), "high_arrival_hold": sum(r["hold_passed"] for r in high),
        "brake_duration_mean_s": mean(r["brake_duration_s"] for r in subset),
        "brake_completed": sum(r["brake_done_s"] is not None for r in subset),
        "post_switch_0p2s_path_mean_mm": mean(r["post_switch_0p2s_path_mm"] for r in subset),
        "switch_to_low_speed_path_mean_mm": mean(r["switch_to_low_speed_path_mm"] for r in subset),
        "low_speed_not_reached": sum(r["switch_s"] is not None and r["first_low_speed_s"] is None for r in subset),
        "brake_incomplete": sum(r["brake_failed"] for r in subset),
        "mean_capped_capture_s": mean(r["capped_capture_s"] for r in subset),
        "mean_hold_duration_s": mean(r["hold_duration_s"] for r in subset if r["hold_passed"]),
        "failure_reasons": dict(sorted(Counter(r["failure"] for r in subset if r["failure"] is not None).items())),
    }


def audit_trace(path, cases):
    archive = np.load(path)
    trace = archive["trace"]
    ci = {str(name): i for i, name in enumerate(archive["columns"])}
    required = ("time_s", "radial_m", "depth_m", "reason", "requested_vx_m_s", "requested_vy_m_s", "capture_hold_gate", "brake_gate")
    assert all(name in ci for name in required)
    assert trace.shape[1] == len(cases)
    assert np.isfinite(trace).all()
    assert np.allclose(np.diff(trace[:, 0, ci["time_s"]]), .002, atol=5e-6)
    for i, row in enumerate(cases):
        stop = int(round(row["hold_end_s"] / .002)) + 1 if row["hold_end_s"] is not None else len(trace)
        x = trace[:stop, i]
        good = (x[:, ci["radial_m"]] <= .00015) & (x[:, ci["depth_m"]] >= .0001) & (x[:, ci["reason"]] == 0)
        rolling = np.convolve(good.astype(np.int16), np.ones(101, dtype=np.int16), mode="valid") if len(good) >= 101 else np.array([])
        first = int(np.flatnonzero(rolling == 101)[0] + 100) if (rolling == 101).any() else None
        if row["first_capture_s"] is not None:
            tick = int(round(row["first_capture_s"] / .002))
            assert first == tick, (row["id"], row["seed"], row["arm"], first, tick)
            assert x[tick, ci["capture_hold_gate"]] == 1
            if row["hold_passed"]:
                end = int(round(row["hold_end_s"] / .002))
                assert end - tick == 250
                assert good[tick:end+1].all(), (row["id"], row["seed"], row["arm"], "hold not continuous")
        else:
            assert first is None, (row["id"], row["seed"], row["arm"], "missed capture")
        requested = np.linalg.norm(x[:, [ci["requested_vx_m_s"], ci["requested_vy_m_s"]]], axis=1) * 1000
        assert requested.max() <= (0.2 if row["arm"] == "A" else 1.) + .001
        assert (requested[x[:, ci["capture_hold_gate"]] == 1] <= .001).all()
        assert (requested[x[:, ci["brake_gate"]] == 1] <= .001).all()
        if row["arm"] in ("B", "C") and row["switch_s"] is not None:
            tick = int(round(row["switch_s"] / .002))
            assert requested[tick:].max() <= .201, (row["id"], row["seed"], row["arm"], "late speed switch")
    return trace.shape[0]


def load_formal(out):
    rows = []
    trace_steps = {}
    for seed in SEEDS:
        for arm in ARMS:
            folder = out / "formal" / f"seed_{seed}" / arm
            item = json.loads((folder / "results.json").read_text())
            assert item["seed"] == seed and item["arm"] == arm and not item["legacy"]
            cases = item["cases"]
            assert len(cases) == 24
            for case in cases:
                case["seed"] = seed
                assert case["arm"] == arm
            trace_steps[f"{seed}_{arm}"] = audit_trace(folder / "trajectory.npz", cases)
            rows.extend(cases)
    assert len(rows) == 216
    keys = Counter((r["id"], r["seed"], r["arm"]) for r in rows)
    assert all(n == 1 for n in keys.values()) and len(keys) == 216
    return rows, trace_steps


def paired(rows, lhs, rhs):
    a = {(r["id"], r["seed"]): r for r in rows if r["arm"] == lhs}
    b = {(r["id"], r["seed"]): r for r in rows if r["arm"] == rhs}
    assert a.keys() == b.keys()
    gained = [k for k in a if not a[k]["hold_passed"] and b[k]["hold_passed"]]
    lost = [k for k in a if a[k]["hold_passed"] and not b[k]["hold_passed"]]
    return {"gained": gained, "lost": lost, "unchanged_success": sum(a[k]["hold_passed"] and b[k]["hold_passed"] for k in a)}


def replay_summary(out):
    result = {}
    for seed in (7, 17):
        for arm in ("old", "A"):
            folder = out / "replay" / f"seed_{seed}" / arm
            data = json.loads((folder / "results.json").read_text())
            case = next(r for r in data["cases"] if r["id"] == "speed_1_04")
            archive = np.load(folder / "trajectory.npz")
            trace = archive["trace"]
            ci = {str(name): i for i, name in enumerate(archive["columns"])}
            idx = next(i for i, r in enumerate(data["cases"]) if r["id"] == "speed_1_04")
            x = trace[:, idx]
            good = (x[:, ci["radial_m"]] <= .00015) & (x[:, ci["depth_m"]] >= .0001) & (x[:, ci["reason"]] == 0)
            rolls = np.convolve(good.astype(np.int16), np.ones(101, dtype=np.int16), mode="valid")
            first = int(np.flatnonzero(rolls == 101)[0] + 100) if (rolls == 101).any() else None
            result[f"{seed}_{arm}"] = {
                "first_geometric_capture_s": round(first * .002, 3) if first is not None else None,
                "recorded_capture_s": case["first_capture_s"], "hold_passed": case["hold_passed"],
                "first_fault_reason": case["first_fault_reason"], "terminal_s": case["hold_end_s"],
                "legacy_decision_success": arm == "old" and case["first_fault_reason"] == 0 and case["hold_end_s"] < 40,
                "xy_cap_mm_s": data["xy_cap_mm_s"],
            }
    return result


def main(out, replay_source=None):
    replay_source = replay_source or out
    manifest = json.loads((out / "manifest.json").read_text())
    rows, trace_steps = load_formal(out)
    summary = {arm: aggregate(rows, arm) for arm in ARMS}
    pairs = {f"{a}_to_{b}": paired(rows, a, b) for a, b in (("A", "B"), ("A", "C"), ("B", "C"))}
    replay = replay_summary(replay_source)
    original_events = json.loads((Path(__file__).resolve().parents[1] / "evaluation/xy_speed_formal_v1/transient_capture_audit.json").read_text())
    smoke = json.loads((out / "replay/seed_7/C/results.json").read_text())["cases"]
    result = {"complete": True, "formal_cases": len(rows), "trace_steps": trace_steps, "aggregate": summary,
              "paired": pairs, "replay": replay, "original_transient_events": original_events,
              "replay_source": str(replay_source), "smoke": smoke,
              "manifest_sha256": __import__("hashlib").sha256((out / "manifest.json").read_bytes()).hexdigest()}
    (out / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    keys = ["id", "seed", "arm", "radius_mm", "angle_deg", "actual_handoff_radius_mm", "initially_near", "first_near_s", "switch_s", "brake_done_s", "first_low_speed_s", "first_capture_s", "hold_end_s", "hold_passed", "failure", "first_fault_reason", "brake_duration_s", "post_switch_0p2s_path_mm", "switch_to_low_speed_path_mm", "capped_capture_s", "hold_duration_s", "actual_xy_mean_mm_s", "requested_xy_max_mm_s"]
    with (out / "cases.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# SPIKE-010 高速近孔减速与 PPO 交接对照", "",
             "## 固定条件", "",
             f"开发诊断集：24 个位置 × 3 个冻结 PPO 种子 × A/B/C = {len(rows)} 回合。名义偏差 1/2/5 mm，每档 8 个方向。",
             "A 全程 0.2 mm/s；B 首次实测孔距 ≤1 mm 后从 1.0 降为 0.2 mm/s；C 同 B 并在交接时制动。真实孔距只用于评估开关，未加入 PPO 观测。",
             "20 N、2 ms 物理步、0.2 s PPO 决策、32 帧历史、40 s 首次捕获预算、原保护和权重。全部回合从同一批准备状态配对起步。",
             f"仓库 HEAD `{manifest['git_head']}`，评估脚本 SHA256 `{manifest['evaluation_script_sha256']}`。", "",
             "## 主表", "",
             "| 组别 | 保持通过/总数 | 初始在外且曾进入/初始在外 | 高速到达后通过/高速到达 | 制动时间均值 s | 切换后 0.2 s 余移均值 mm | 至首次低速阈值余移均值 mm | 未达阈值 | 失败计 40 s 平均捕获耗时 s | 失败原因 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for arm in ARMS:
        a = summary[arm]
        high_text = f"{a['high_arrival_hold']}/{a['high_arrival_total']}" if arm != "A" else "不适用"
        low_missing = a["low_speed_not_reached"] if arm != "A" else "不适用"
        brake_text = f"{fmt(a['brake_duration_mean_s'], 3)}（{a['brake_completed']} 次完成）" if arm == "C" else "—"
        lines.append(f"| {arm} | {a['hold_passed']}/{a['total']} | {a['arrived']}/{a['outside']} | {high_text} | {brake_text} | {fmt(a['post_switch_0p2s_path_mean_mm'], 3)} | {fmt(a['switch_to_low_speed_path_mean_mm'], 3)} | {low_missing} | {fmt(a['mean_capped_capture_s'])} | {a['failure_reasons']} |")
    all_captures = sum(a["first_capture"] for a in summary.values())
    all_held = sum(a["hold_passed"] for a in summary.values())
    lines += ["", f"首次捕获要求连续 101 个物理步满足径向误差 ≤0.15 mm、深度 ≥0.1 mm 且无故障。其后 XY 保持 0.5 s，几何条件须持续满足且无故障，才计保持通过。本轮 {all_captures} 次首次捕获中 {all_held} 次保持通过。失败计 40 s；准备阶段捕获不算 PPO 成功。", "",
              "## 半径与模型明细", "",
              "| 半径 mm | 种子 | A 保持/8 | B 保持/8 | C 保持/8 | A/B/C 首次捕获 | A/B/C 未进入近孔 |",
              "|---:|---:|---:|---:|---:|---|---|"]
    for radius in RADII:
        for seed in SEEDS:
            groups = {arm: [r for r in rows if r["radius_mm"] == radius and r["seed"] == seed and r["arm"] == arm] for arm in ARMS}
            held = [sum(r["hold_passed"] for r in groups[arm]) for arm in ARMS]
            captured = [sum(r["first_capture"] for r in groups[arm]) for arm in ARMS]
            no_near = [sum(r["failure"] == "never_near" for r in groups[arm]) for arm in ARMS]
            lines.append(f"| {radius} | {seed} | {held[0]}/8 | {held[1]}/8 | {held[2]}/8 | {'/'.join(map(str, captured))} | {'/'.join(map(str, no_near))} |")
    lines += ["", "## 配对变化", ""]
    for label, pair in pairs.items():
        lines.append(f"- {label}：失败转通过 {len(pair['gained'])}，通过转失败 {len(pair['lost'])}；共同通过 {pair['unchanged_success']}。逐案见 `cases.csv`。")
    lines += ["", "## 成功事件重放与制动冒烟", ""]
    for seed in (7, 17):
        old, new = replay[f"{seed}_old"], replay[f"{seed}_A"]
        lines.append(f"- 种子 {seed}，0.5 mm/s，speed_1_04：重放旧逻辑首次几何捕获 {fmt(old['first_geometric_capture_s'], 3)} s，决策末 {'成功' if old['legacy_decision_success'] else '失败'}；重放新逻辑首次捕获 {fmt(new['recorded_capture_s'], 3)} s，保持 {'通过' if new['hold_passed'] else '失败'}，故障码 {new['first_fault_reason']}。")
    lines.append("- SPIKE-009 原档案中这两例分别在 3.214 s、3.234 s 连续达标，随后 3.346 s、3.344 s 超载。本轮重新创建的仿真批次未逐步复现原故障，且旧/新重放在干预前已有数值分岔；这些重放结果不能证明提前锁存单独修复了原两例。")
    lines.append(f"- 重放来源：`{replay_source}`。其中 A/旧逻辑重放的代码不受后续 B/C 低速向量限幅修正影响；被中止轮次的正式 B/C 结果已作废。")
    for case in smoke:
        lines.append(f"- C 冒烟 {case['id']}：切换 {fmt(case['switch_s'], 3)} s，停稳 {fmt(case['brake_done_s'], 3)} s，保持 {'通过' if case['hold_passed'] else '失败'}，终止原因为 {case['failure'] or '无'}。")
    b_success_high = [r for r in rows if r["arm"] == "B" and r["high_speed_arrived"] and r["hold_passed"]]
    a_by_key = {(r["id"], r["seed"]): r for r in rows if r["arm"] == "A"}
    high_time_gain = mean(a_by_key[(r["id"], r["seed"])]["first_capture_s"] - r["first_capture_s"] for r in b_success_high)
    b_by_key = {(r["id"], r["seed"]): r for r in rows if r["arm"] == "B"}
    c_by_key = {(r["id"], r["seed"]): r for r in rows if r["arm"] == "C"}
    both_switched = [(b_by_key[key], c_by_key[key]) for key in b_by_key if b_by_key[key]["switch_s"] is not None and c_by_key[key]["switch_s"] is not None]
    paired_path_reduction = mean(b["post_switch_0p2s_path_mm"] - c["post_switch_0p2s_path_mm"] for b, c in both_switched)
    lines += ["", "## 结论与边界", "",
              f"- 本候选交接未达到稳定浅入孔：B 高速到达后的保持通过 {summary['B']['high_arrival_hold']}/{summary['B']['high_arrival_total']}，C 为 {summary['C']['high_arrival_hold']}/{summary['C']['high_arrival_total']}；整体 A/B/C 为 {summary['A']['hold_passed']}/{summary['A']['total']}、{summary['B']['hold_passed']}/{summary['B']['total']}、{summary['C']['hold_passed']}/{summary['C']['total']}。",
              f"- B 有 {summary['B']['outside'] - summary['B']['arrived']}/{summary['B']['outside']} 个初始在外案例未进入 1 mm；已经到达的 {summary['B']['high_arrival_total']} 个案例中有 {summary['B']['failure_reasons'].get('contact_recovery_failed', 0)} 个接触恢复失败。单纯进入 1 mm 后降到 0.2 mm/s 不够。",
              f"- C 在双方都发生切换的 {len(both_switched)} 个配对案例中，切换后 0.2 s 实际余移平均减少 {fmt(paired_path_reduction, 3)} mm；但 {summary['C']['brake_incomplete']}/{summary['C']['high_arrival_total']} 个高速到达案例在 1 s 内未连续达到制动阈值。C 相比 B 挽回 {len(pairs['B_to_C']['gained'])} 案、损失 {len(pairs['B_to_C']['lost'])} 案；当前制动判据/执行交接不能作为稳定方案。",
              f"- B 高速到达且最终通过的 {len(b_success_high)} 案，比同案 A 的首次捕获平均早 {fmt(high_time_gain)} s；这是成功子集的条件比较。计入失败后，B 的平均捕获耗时 {fmt(summary['B']['mean_capped_capture_s'])} s，高于 A 的 {fmt(summary['A']['mean_capped_capture_s'])} s。",
              "- 24 个位置属于看过旧结果的开发诊断集，三个模型共享这些位置。真实孔距只触发本次诊断开关；本结果不能证明机器人能用力觉自主判断何时减速，也不能外推到完整插入或新位置。",
              "- 下一步应先核查高速到达时的接触状态、命令超前量和实际横移，改进可验证的 XY 制动与交接；另行验证远场覆盖/高速策略适应，处理未进入 1 mm 的案例。随后才用可观测信号触发减速，并在独立新位置复测。",
              "",
              "原始 2 ms 轨迹位于各 `formal/seed_*/A|B|C/trajectory.npz`；列名见 NPZ `columns`。`summary.json` 与 `cases.csv` 保存汇总和逐案结果。旧 SPIKE-009 分数未回写。", ""]
    (out / "REPORT.md").write_text("\n".join(lines))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--replay-source")
    args = parser.parse_args()
    main(Path(args.output).resolve(), Path(args.replay_source).resolve() if args.replay_source else None)
