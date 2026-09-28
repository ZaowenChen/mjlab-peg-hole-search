"""SPIKE-010: frozen-policy, paired XY deceleration and braking experiment.

The true hole radius is used only by this evaluation harness. It is never
inserted into the policy observation. All interventions act on the XY channel.
"""
import argparse
import copy
import hashlib
import json
import math
import subprocess
import time
from collections import deque
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from mjlab_contact_prep.config import ContactConfig
from mjlab_contact_prep.night_run.environment import BankEnv, save_json
from mjlab_contact_prep.night_run.jobs import ppo_config, verify_frozen


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "evaluation/xy_speed_formal_v1"
WEIGHTS = ROOT / "evaluation/night_fullcircle_5mm_v1"
DT = .002
SEARCH_S = 40.
HOLD_STEPS = 250
BRAKE_MAX_STEPS = 500
BRAKE_STABLE_STEPS = 25
BRAKE_SPEED_M_S = .00005
SWITCH_M = .001
COLS = [
    "time_s", "radial_m", "depth_m", "tip_x_m", "tip_y_m", "tip_z_m",
    "actual_vx_m_s", "actual_vy_m_s", "actual_vz_m_s", "requested_vx_m_s",
    "requested_vy_m_s", "xy_reference_x_m", "xy_reference_y_m",
    "xy_reference_error_x_m", "xy_reference_error_y_m", "command_lead_x_m",
    "command_lead_y_m", "command_lead_z_m", "requested_z_m_s",
    "executed_vx_m_s", "executed_vy_m_s", "executed_vz_m_s",
    "Fx_N", "Fy_N", "Fz_N", "Mx_Nm", "My_Nm", "Mz_Nm",
    "filtered_force_N", "controller_state", "reason", "ready",
    "recovery_count", "xy_action_x", "xy_action_y", "brake_gate",
    "capture_hold_gate", "speed_cap_mm_s", "accel_limited", "energy_limited",
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selected_ids():
    return [f"speed_{radius}_{j:02}" for radius in (1, 2, 5) for j in range(0, 16, 2)]


def source_categories(case_id, old_rows):
    matching = [r for r in old_rows if r["id"] == case_id]
    categories = set()
    for row in matching:
        if row["outcome"] == "success":
            categories.add("high_speed_success")
        if row["outcome"] == "fault" and row["search_reason"] in (1, 2, 7):
            categories.add("near_hole_or_contact_fault")
        if row["outcome"] == "timeout" and row["min_radial_mm"] > 1:
            categories.add("never_near_timeout")
    return sorted(categories)


def make_manifest(out):
    old_plan = json.loads((SOURCE / "plan.json").read_text())
    verify_frozen(json.loads((WEIGHTS / "plan.json").read_text()))
    old_preparation = json.loads((SOURCE / "preparation_audit.json").read_text())
    by_id = {row["id"]: row for row in old_preparation}
    old_rows = [r for seed in (7, 17, 27) for r in json.loads((SOURCE / f"seed_{seed}/speed_1/results.json").read_text())["rows"]]
    cases = []
    for case_id in selected_ids():
        row = by_id[case_id]
        cases.append({key: row[key] for key in ("id", "radius_mm", "angle_deg", "dx_mm", "dy_mm", "bucket", "prep_class", "eligible", "actual_handoff_radius_mm", "actual_handoff_depth_mm")})
        cases[-1]["old_categories"] = source_categories(case_id, old_rows)
    present = {c for case in cases for c in case["old_categories"]}
    required = {"high_speed_success", "near_hole_or_contact_fault", "never_near_timeout"}
    if not required <= present:
        raise RuntimeError(f"Representative old-result categories missing: {required - present}")
    raw = torch.load(SOURCE / "bank/bank.pt", weights_only=False, map_location="cpu")
    indices = [next(i for i, row in enumerate(raw["rows"]) if row["id"] == cid) for cid in selected_ids()]
    bank = {group: {key: tensor[indices].clone() for key, tensor in raw[group].items()} for group in ("physics", "controller", "term", "probe")}
    bank["rows"] = [raw["rows"][i] for i in indices]
    torch.save(bank, out / "bank.pt")
    del bank, raw
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    git_status = subprocess.run(["git", "status", "--short"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    manifest = {
        "design": "MPHS SPIKE-010", "git_head": git, "git_status_at_start": git_status,
        "source_bank": str(SOURCE / "bank/bank.pt"), "source_bank_sha256": digest(SOURCE / "bank/bank.pt"),
        "source_plan_sha256": digest(SOURCE / "plan.json"), "source_preparation_audit_sha256": digest(SOURCE / "preparation_audit.json"),
        "evaluation_script": str(Path(__file__).resolve()), "evaluation_script_sha256": digest(__file__),
        "contact_config": asdict(ContactConfig()), "source_physics": old_plan["frozen_physics"],
        "weights": {str(s): {"path": str(WEIGHTS / f"amp_0/seed_{s}/policy.pt"), "sha256": digest(WEIGHTS / f"amp_0/seed_{s}/policy.pt")} for s in (7, 17, 27)},
        "cases": cases, "seeds": [7, 17, 27], "arms": ["A", "B", "C"],
        "switch_radius_mm": 1., "fast_xy_cap_mm_s": 1., "slow_xy_cap_mm_s": .2,
        "brake_speed_threshold_mm_s": .05, "brake_stable_s": .05, "brake_timeout_s": 1.,
        "brake_speed_estimator": "20 ms tip XY displacement / 0.02 s, sampled each 2 ms",
        "capture": "101 consecutive 2 ms physical samples, radius <=0.15 mm, depth >=0.1 mm, reason=0",
        "hold": "250 further physical samples (0.5 s), same geometry and no fault, same XY hold for all arms",
        "timing": "40 s first-capture budget; brake time included; at most 0.5 s added hold",
        "policy_observation": "Unmodified 32x20 force/proprioceptive history; true radius used only by evaluation control",
    }
    save_json(out / "manifest.json", manifest)
    return manifest


class HandoffProbe:
    def __init__(self, env, arm, initial_radius, legacy=False, slow_cap_mm_s=.2):
        self.env = env
        self.arm = arm
        self.legacy = legacy
        self.slow_cap_mm_s = slow_cap_mm_s
        self.n = env.num_envs
        self.initial_radius = np.asarray(initial_radius, dtype=float)
        self.near_initial = self.initial_radius <= SWITCH_M
        self.switched = torch.tensor(self.near_initial if arm != "A" else [False] * self.n, device=env.device)
        self.braking = torch.zeros(self.n, dtype=torch.bool, device=env.device)
        self.brake_done = torch.zeros_like(self.braking)
        self.brake_failed = torch.zeros_like(self.braking)
        self.finished = torch.zeros_like(self.braking)
        self.captured = torch.zeros_like(self.braking)
        self.hold_passed = torch.zeros_like(self.braking)
        self.hold_failed = torch.zeros_like(self.braking)
        self.first_near_tick = torch.full((self.n,), -1, dtype=torch.long, device=env.device)
        self.switch_tick = torch.full_like(self.first_near_tick, -1)
        self.brake_done_tick = torch.full_like(self.first_near_tick, -1)
        self.first_low_speed_tick = torch.full_like(self.first_near_tick, -1)
        self.capture_tick = torch.full_like(self.first_near_tick, -1)
        self.finish_tick = torch.full_like(self.first_near_tick, -1)
        self.brake_stable = torch.zeros_like(self.first_near_tick)
        self.brake_elapsed = torch.zeros_like(self.first_near_tick)
        self.terminal_reason = torch.zeros(self.n, dtype=torch.long, device=env.device)
        self.pos_history = deque(maxlen=11)
        self.trace = []
        self.previous_pos = None
        self.raw_action = torch.zeros_like(env.term.xy_action)
        self.original_process = env.term.process_actions
        def record_policy_action(actions):
            self.original_process(actions)
            self.raw_action[:] = env.term.xy_action
        env.term.process_actions = record_policy_action
        self.original = env.term.apply_actions
        env.term.apply_actions = self.apply

    def decision_start(self):
        # A completed brake remains gated through the end of its current PPO
        # interval. The next action is computed from the updated history.
        self.braking &= ~self.brake_done

    def reanchor_xy_command(self, mask, pos):
        """Cancel the XY task-space lead while retaining Z and angular intent."""
        term = self.env.term
        term._point_torch[:] = pos
        term._compute_jacobian()
        jac = torch.cat((term._jacp_torch[:, :, term._joint_dof_ids], term._jacr_torch[:, :, term._joint_dof_ids]), dim=1)
        actual = term._entity.data.joint_pos[:, term._joint_ids]
        lead = (jac @ (term.command - actual.double()).float()[:, :, None]).squeeze(-1)
        target = torch.zeros_like(lead)
        target[:, :2] = lead[:, :2]
        matrix = jac.transpose(1, 2) @ jac
        matrix.diagonal(dim1=-2, dim2=-1).add_(term.cfg.damping ** 2)
        correction = torch.linalg.solve(matrix, (jac.transpose(1, 2) @ target[:, :, None]).squeeze(-1))
        term.command[mask] -= correction[mask].double()
        velocity = (jac @ term.qvelocity[:, :, None]).squeeze(-1)
        target.zero_()
        target[:, :2] = velocity[:, :2]
        correction_v = torch.linalg.solve(matrix, (jac.transpose(1, 2) @ target[:, :, None]).squeeze(-1))
        term.qvelocity[mask] -= correction_v[mask]

    def apply(self):
        env, term = self.env, self.env.term
        tick = env.ticks.clone()
        term.xy_action[:] = self.raw_action
        pos = env.env.sim.data.site_xpos[:, term.geometry_id].clone()
        radial, depth = env.metrics()
        healthy = term.controller.reason == 0
        finite = torch.isfinite(radial) & torch.isfinite(depth) & torch.isfinite(pos).all(-1)
        near = (radial <= SWITCH_M) & healthy & finite
        first = near & (self.first_near_tick < 0)
        self.first_near_tick[first] = tick[first]
        if self.arm in ("B", "C"):
            crossing = near & ~self.switched & ~self.finished
            self.switched |= crossing
            self.switch_tick[crossing] = tick[crossing]
            if self.arm == "C":
                self.braking |= crossing
                if bool(crossing.any()):
                    self.reanchor_xy_command(crossing, pos)

        inside = (radial <= .00015) & (depth >= .0001) & healthy & finite
        impending = inside & (env.inside_samples >= 100) & ~self.captured & ~self.finished
        if self.legacy:
            impending[:] = False
        if bool(impending.any()):
            self.reanchor_xy_command(impending, pos)
        # Gate on the qualifying physical sample itself; a fault raised by
        # controller.step below cancels the provisional event.
        hold_gate = self.captured | impending
        brake_gate = self.braking & ~hold_gate
        gate = hold_gate | brake_gate | self.finished
        if bool(gate.any()):
            term.xy_action[gate] = 0
            term.controller.xy_reference[gate, :2] = pos[gate, :2]
        if self.arm in ("B", "C"):
            low = self.switched & ~gate
            if bool(low.any()):
                action = term.xy_action[low]
                term.xy_action[low] = action / action.norm(dim=-1, keepdim=True).clamp_min(1.) * .2
        self.original()

        actual_reason = term.controller.reason
        good = (radial <= .00015) & (depth >= .0001) & (actual_reason == 0) & finite
        if not self.legacy:
            new_capture = impending & good & (env.inside_samples >= 101)
            self.captured |= new_capture
            self.capture_tick[new_capture] = tick[new_capture]
            holding = self.captured & ~self.finished
            broken = holding & ~good
            self.hold_failed |= broken
            elapsed = tick - self.capture_tick
            passed = holding & ~broken & (elapsed >= HOLD_STEPS)
            self.hold_passed |= passed
            completed = broken | passed
            self.finished |= completed
            self.finish_tick[completed] = tick[completed]

        self.pos_history.append(pos[:, :2].clone())
        if len(self.pos_history) == 11:
            smooth_v = (pos[:, :2] - self.pos_history[0]).norm(dim=-1) / .02
            first_low = (self.switch_tick >= 0) & (self.first_low_speed_tick < 0) & ~self.finished & (smooth_v <= BRAKE_SPEED_M_S)
            self.first_low_speed_tick[first_low] = tick[first_low]
        else:
            smooth_v = torch.full_like(radial, float("inf"))
        if self.arm == "C":
            self.brake_elapsed += self.braking.long()
            quiet = smooth_v <= BRAKE_SPEED_M_S
            self.brake_stable = torch.where(self.braking & quiet, self.brake_stable + 1, 0)
            completed = self.braking & (self.brake_stable >= BRAKE_STABLE_STEPS) & ~self.brake_done & ~self.finished
            self.brake_done |= completed
            self.brake_done_tick[completed] = tick[completed]
            timed_out = self.braking & (self.brake_elapsed >= BRAKE_MAX_STEPS) & ~self.brake_done & ~self.finished & (actual_reason == 0)
            self.brake_failed |= timed_out
            self.finished |= timed_out
            self.finish_tick[timed_out] = tick[timed_out]
        if not self.legacy:
            fault = (actual_reason != 0) & ~self.finished
            self.terminal_reason[fault] = actual_reason[fault]
            self.finished |= fault
            self.finish_tick[fault] = tick[fault]

        if self.previous_pos is None:
            measured = torch.zeros_like(pos)
        else:
            measured = (pos - self.previous_pos) / DT
        self.previous_pos = pos
        ctrl = term.controller
        actual_q = term._entity.data.joint_pos[:, term._joint_ids]
        jac = torch.cat((term._jacp_torch[:, :, term._joint_dof_ids], term._jacr_torch[:, :, term._joint_dof_ids]), dim=1)
        lead = (jac @ (term.command - actual_q.double()).float()[:, :, None]).squeeze(-1)
        ref = ctrl.xy_reference[:, :2]
        caps = torch.full((self.n,), self.slow_cap_mm_s if self.arm == "A" else 1., device=env.device)
        if self.arm != "A":
            caps = torch.where(self.switched, .2, caps)
        fields = [
            tick.float() * DT, radial, depth, *pos.unbind(-1), *measured.unbind(-1),
            *ctrl.last_xy[:, :2].unbind(-1), *ref.unbind(-1), *((ref - pos[:, :2]).unbind(-1)),
            *lead[:, :3].unbind(-1), ctrl.velocity, *term.executed_twist[:, :3].unbind(-1),
            *term.wrench.unbind(-1), ctrl.filtered_force, ctrl.state.float(), actual_reason.float(),
            ctrl.ready.float(), ctrl.recoveries.float(), *term.xy_action.unbind(-1), brake_gate.float(),
            hold_gate.float(), caps, term.accel_limited.float(), term.energy_limited.float(),
        ]
        record = torch.stack(fields, dim=-1).detach().cpu().numpy().astype(np.float32)
        assert record.shape[-1] == len(COLS), (record.shape, len(COLS))
        self.trace.append(record)


def summarize_case(x, row, probe, i, arm):
    ci = {name: j for j, name in enumerate(COLS)}
    case = dict(row)
    case.update(arm=arm, initial_radius_mm=float(probe.initial_radius[i] * 1000),
                initially_near=bool(probe.near_initial[i]))
    def when(v):
        return None if v < 0 else round(float(v * DT), 3)
    for field, tensor in (("first_near_s", probe.first_near_tick), ("switch_s", probe.switch_tick),
                          ("brake_done_s", probe.brake_done_tick), ("first_low_speed_s", probe.first_low_speed_tick),
                          ("first_capture_s", probe.capture_tick),
                          ("hold_end_s", probe.finish_tick)):
        case[field] = when(int(tensor[i]))
    case["brake_failed"] = bool(probe.brake_failed[i])
    case["first_capture"] = bool(probe.captured[i])
    case["hold_passed"] = bool(probe.hold_passed[i])
    case["hold_failed"] = bool(probe.hold_failed[i])
    if case["brake_done_s"] is not None:
        case["brake_duration_s"] = round(case["brake_done_s"] - case["switch_s"], 3)
    else:
        case["brake_duration_s"] = None
    n = min(len(x), int(probe.finish_tick[i]) + 1 if int(probe.finish_tick[i]) >= 0 else len(x))
    x = x[:n]
    path = np.linalg.norm(np.diff(x[:, [ci["tip_x_m"], ci["tip_y_m"]]], axis=0), axis=1)
    case["actual_xy_mean_mm_s"] = float(path.sum() * 1000 / max((len(x)-1)*DT, DT))
    case["actual_xy_path_mm"] = float(path.sum() * 1000)
    case["requested_xy_max_mm_s"] = float(np.linalg.norm(x[:, [ci["requested_vx_m_s"], ci["requested_vy_m_s"]]], axis=1).max() * 1000)
    case["min_radius_mm"] = float(x[:, ci["radial_m"]].min() * 1000)
    case["peak_axial_N"] = float(x[:, ci["Fz_N"]].max())
    case["peak_radial_N"] = float(np.linalg.norm(x[:, [ci["Fx_N"], ci["Fy_N"]]], axis=1).max())
    case["peak_moment_Nm"] = float(np.linalg.norm(x[:, [ci["Mx_Nm"], ci["My_Nm"], ci["Mz_Nm"]]], axis=1).max())
    switch = int(probe.switch_tick[i])
    done = int(probe.first_low_speed_tick[i])
    if switch >= 0:
        j = min(switch + 100, len(x) - 1)
        case["post_switch_0p2s_path_mm"] = float(path[switch:j].sum() * 1000)
        case["switch_to_low_speed_path_mm"] = float(path[switch:min(done, len(path))].sum() * 1000) if done >= 0 else None
    else:
        case["post_switch_0p2s_path_mm"] = None
        case["switch_to_low_speed_path_mm"] = None
    fault_at = np.flatnonzero(x[:, ci["reason"]] != 0)
    reason = int(x[fault_at[0], ci["reason"]]) if len(fault_at) else 0
    case["first_fault_reason"] = reason
    if row["prep_class"] == "failed":
        failure = "preparation_failed"
    elif row["prep_class"] == "captured":
        failure = None
    elif case["brake_failed"]:
        failure = "brake_incomplete"
    elif case["hold_failed"]:
        failure = "post_capture_hold_failed"
    elif reason in (1, 2, 3):
        failure = "overload"
    elif reason == 7:
        failure = "contact_recovery_failed"
    elif reason:
        failure = "other_fault"
    elif case["first_near_s"] is None and not case["initially_near"]:
        failure = "never_near"
    else:
        failure = "near_timeout"
    case["failure"] = None if case["hold_passed"] else failure
    case["capped_capture_s"] = case["first_capture_s"] if case["hold_passed"] else SEARCH_S
    case["hold_duration_s"] = None if case["hold_end_s"] is None or case["first_capture_s"] is None else round(case["hold_end_s"] - case["first_capture_s"], 3)
    case["high_speed_arrived"] = bool(arm in ("B", "C") and not case["initially_near"] and case["first_near_s"] is not None)
    return case


def evaluate(out, manifest, arm, seed, case_ids=None, legacy=False, speed_override_mm_s=None, full_batch=False):
    all_cases = json.loads((SOURCE / "preparation_audit.json").read_text()) if full_batch else manifest["cases"]
    indices = list(range(len(all_cases))) if full_batch else [i for i, c in enumerate(all_cases) if case_ids is None or c["id"] in case_ids]
    if not indices:
        raise ValueError("No selected cases")
    bank_file = SOURCE / "bank/bank.pt" if full_batch else out / "bank.pt"
    if len(indices) != len(all_cases):
        raw = torch.load(bank_file, weights_only=False, map_location="cpu")
        bank = {g: {k: v[indices] for k, v in raw[g].items()} for g in ("physics", "controller", "term", "probe")}
        bank["rows"] = [raw["rows"][i] for i in indices]
        subset = out / "smoke_bank.pt"
        torch.save(bank, subset)
        bank_file = subset
    env = BankEnv(bank_file, len(indices), 0, 7, autoreset=False, audit=False, horizon=203)
    slow_cap = .2 if speed_override_mm_s is None else speed_override_mm_s
    config = replace(ContactConfig(), xy_speed=(slow_cap * .001 if arm == "A" else .001))
    env.term.cfg.contact = config
    env.term.controller.cfg = config
    ids = torch.arange(len(indices), device=env.device)
    env.restore(ids, ids)
    for group, obj in (("physics", env.env.sim.data), ("controller", env.term.controller), ("term", env.term), ("probe", env.term.probe)):
        for key, value in env.pool[group].items():
            if not torch.equal(getattr(obj, key), value):
                raise AssertionError(("restore mismatch", group, key))
    initial = env.metrics()[0].detach().cpu().numpy()
    probe = HandoffProbe(env, arm, initial, legacy=legacy, slow_cap_mm_s=slow_cap)
    runner = OnPolicyRunner(env, copy.deepcopy(ppo_config(7, 1)), None, device=env.device)
    runner.load(manifest["weights"][str(seed)]["path"], map_location=env.device)
    policy = runner.get_inference_policy()
    start = time.monotonic()
    rows = [all_cases[i] for i in indices]
    try:
        with torch.inference_mode():
            for decision in range(203):
                probe.decision_start()
                actions = policy(env.get_observations()).clamp(-1, 1)
                actions[probe.finished] = 0
                env.step(actions)
                fault = env.term.controller.reason != 0
                newly = fault & ~probe.finished
                probe.finished |= newly
                probe.finish_tick[newly] = env.ticks[newly] - 1
                if legacy:
                    old_success = env.last_success & ~probe.finished
                    probe.finished |= old_success
                    probe.finish_tick[old_success] = env.ticks[old_success] - 1
                if decision % 10 == 9:
                    save_json(out / "progress.json", {"stage": "evaluation", "arm": arm, "seed": seed, "decision": decision + 1, "active": int((~probe.finished).sum()), "elapsed_s": time.monotonic() - start})
                if bool(probe.finished.all()):
                    break
        trace = np.stack(probe.trace)
        expected = (np.arange(len(trace)) * DT).astype(np.float32)
        if not np.allclose(trace[:, 0, 0], expected, atol=5e-6):
            raise AssertionError("Physical trace has missing steps")
        if not np.isfinite(trace).all():
            raise FloatingPointError("Nonfinite trajectory")
        maximum = np.linalg.norm(trace[:, :, [COLS.index("requested_vx_m_s"), COLS.index("requested_vy_m_s")]], axis=-1)
        if float(maximum.max()) > (slow_cap * .001 if arm == "A" else .001) + 1e-6:
            raise AssertionError("XY request cap exceeded")
        results = [summarize_case(trace[:, i, :], row, probe, i, arm) for i, row in enumerate(rows)]
        for result in results:
            if result["first_capture_s"] is not None and result["first_capture_s"] > SEARCH_S:
                result["hold_passed"] = False
                result["failure"] = "near_timeout"
                result["capped_capture_s"] = SEARCH_S
        dest = out / ("replay" if legacy or case_ids is not None else "formal") / f"seed_{seed}" / arm
        if legacy:
            dest = out / "replay" / f"seed_{seed}" / "old"
        dest.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dest / "trajectory.npz", trace=trace, columns=COLS)
        save_json(dest / "results.json", {"arm": arm, "seed": seed, "legacy": legacy, "xy_cap_mm_s": slow_cap if arm == "A" else 1., "wall_s": time.monotonic() - start, "cases": results})
        print("RESULT", dest, "hold", sum(r["hold_passed"] for r in results), "capture", sum(r["first_capture"] for r in results), "wall", round(time.monotonic() - start, 1), flush=True)
        return results
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--phase", choices=("prepare", "replay", "smoke", "formal"), required=True)
    args = parser.parse_args()
    out = Path(args.output).resolve()
    if args.phase == "prepare":
        out.mkdir(parents=True, exist_ok=False)
        manifest = make_manifest(out)
        print("PREPARED", len(manifest["cases"]), out, flush=True)
        return
    manifest = json.loads((out / "manifest.json").read_text())
    if digest(__file__) != manifest["evaluation_script_sha256"]:
        raise RuntimeError("Evaluation script changed after manifest was frozen")
    for item in manifest["weights"].values():
        if digest(item["path"]) != item["sha256"]:
            raise RuntimeError("Model changed after manifest was frozen")
    if args.phase == "replay":
        for seed in (7, 17):
            evaluate(out, manifest, "A", seed, ["speed_1_04"], legacy=True, speed_override_mm_s=.5, full_batch=True)
            evaluate(out, manifest, "A", seed, ["speed_1_04"], speed_override_mm_s=.5, full_batch=True)
    elif args.phase == "smoke":
        evaluate(out, manifest, "C", 7, ["speed_2_00", "speed_2_02"])
    else:
        for seed in manifest["seeds"]:
            for arm in manifest["arms"]:
                evaluate(out, manifest, arm, seed)
    save_json(out / "progress.json", {"stage": f"{args.phase}_complete", "time": time.time()})


if __name__ == "__main__":
    main()
