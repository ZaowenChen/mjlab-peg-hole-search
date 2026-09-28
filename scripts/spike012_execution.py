#!/usr/bin/env python3
"""SPIKE-012: paired, opt-in execution-chain experiments on the GPU backend.

This script never loads a policy for E0/E1/E2.  The 0.2 mm/s setting is a
diagnostic input; the frozen 1 mm/s PPO checkpoint is reserved for E3.
"""
import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
import math
import platform
import random
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
import mujoco
from mjlab.envs import ManagerBasedRlEnv

from mjlab_contact_prep.backend import backend_metadata
from mjlab_contact_prep.config import ContactConfig
from mjlab_contact_prep.environment import make_env
from mjlab_contact_prep.probe import ProbeConfig

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evaluation/spike012_execution_v1"
WEIGHT = ROOT / "evaluation/spike011_small_1mms_v1/amp_0/seed_7/policy.pt"
BANK = ROOT / "evaluation/spike011_small_1mms_v1/test/bank/bank.pt"
PHYSICS = ("time", "qpos", "qvel", "act", "ctrl", "qacc_warmstart", "mocap_pos",
           "mocap_quat", "qfrc_applied", "xfrc_applied")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def source_freeze(folder, cfg):
    folder.mkdir(parents=True, exist_ok=False)
    source = folder / "source"
    source.mkdir()
    files = [p for root in (ROOT / "src", ROOT / "scripts", ROOT / "assets")
             for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    hashes = {}
    for p in files:
        relative = p.relative_to(ROOT)
        hashes[str(relative)] = sha(p)
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
    for p in (WEIGHT, BANK):
        if not p.is_file():
            raise FileNotFoundError(p)
    if sha(WEIGHT) != "dee6860c8ca1071d90772b45ada52022bd3e5d638b62382bdce78ea4223cb2d5":
        raise ValueError("SPIKE-011 policy SHA256 mismatch")
    if sha(BANK) != "615e557f3a16fd34460703423485c71f8ad9f0c588872e876c0f57e782985719":
        raise ValueError("SPIKE-011 bank SHA256 mismatch")
    git = lambda *args: subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                                        text=True, check=True).stdout
    (folder / "working_tree.diff").write_text(git("diff", "--binary"))
    (folder / "git_status.txt").write_text(git("status", "--short"))
    write_json(folder / "manifest.json", dict(design="MPHS SPIKE-012 P0", created=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        git_head=git("rev-parse", "HEAD").strip(), git_branch=git("branch", "--show-current").strip(),
        source_sha256=hashes, checkpoint_sha256=sha(WEIGHT), bank_sha256=sha(BANK),
        effective_contact=asdict(cfg), physics_backend=backend_metadata("contact-fix"),
        collision_model="partitioned", python=sys.version, platform=platform.platform(),
        torch=torch.__version__, mujoco=mujoco.__version__, gpu=torch.cuda.get_device_name(0),
        physics_dt_s=cfg.dt, action_decimation=20, ppo_repeat=5, ppo_decision_dt_s=.2,
        command=" ".join(sys.argv)))
    return folder


def snapshot(env, term):
    return dict(physics={n: getattr(env.sim.data, n).detach().cpu().clone() for n in PHYSICS},
        controller={n: v.detach().cpu().clone() for n, v in vars(term.controller).items()
                    if isinstance(v, torch.Tensor)},
        term={n: getattr(term, n).detach().cpu().clone() for n in ("command", "qvelocity", "xy_action", "executed_twist")},
        probe={n: v.detach().cpu().clone() for n, v in vars(term.probe).items()
               if isinstance(v, torch.Tensor)},
        diagnostics={n: getattr(term, n).detach().cpu().clone() for n in
                     ("diagnostic_parking_active", "diagnostic_parking_goal")},
        optional_diagnostics={n: (None if getattr(term, n) is None else getattr(term, n).detach().cpu().clone())
                              for n in ("diagnostic_lateral", "diagnostic_parking_mask")},
        parking_speed=term.diagnostic_parking_speed,
        env={n: v.detach().cpu().clone() for n, v in vars(env).items()
             if isinstance(v, torch.Tensor) and n in ("episode_length_buf", "reset_terminated", "reset_time_outs")},
        rng=dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all(),
                 numpy=np.random.get_state(), python=random.getstate()))


def restore(env, term, state):
    for n, v in state["physics"].items():
        getattr(env.sim.data, n)[:] = v.to(env.device)
    for group, obj in (("controller", term.controller), ("term", term), ("probe", term.probe)):
        for n, v in state[group].items():
            getattr(obj, n)[:] = v.to(env.device)
    for n, v in state.get("diagnostics", {}).items():
        getattr(term, n)[:] = v.to(env.device)
    for n, v in state.get("optional_diagnostics", {}).items():
        setattr(term, n, None if v is None else v.to(env.device).clone())
    term.diagnostic_parking_speed = state.get("parking_speed")
    for n, v in state.get("env", {}).items():
        current = getattr(env, n, None)
        if isinstance(current, torch.Tensor):
            current[:] = v.to(env.device)
        else:
            setattr(env, n, v.to(env.device).clone())
    env.sim.forward()
    term.read_wrench()
    if "rng" in state:
        torch.set_rng_state(state["rng"]["torch"])
        torch.cuda.set_rng_state_all(state["rng"]["cuda"])
        np.random.set_state(state["rng"]["numpy"])
        random.setstate(state["rng"]["python"])


class StopEdge:
    """Diagnostic request edge, evaluated once per environment action.

    `moving` denotes a nonzero XY request. `channel_open` allows that request
    through the final transverse path. The initial zero request is not a stop.
    """
    def __init__(self):
        self.previous_moving = False
        self.previous_channel_open = True
        self.active = False

    def update(self, moving, channel_open=True):
        moving, channel_open = bool(moving), bool(channel_open)
        request_stop = self.previous_moving and not moving
        gate_stop = self.previous_moving and self.previous_channel_open and not channel_open
        source = ("channel_closed_while_moving" if moving else "channel_closed_on_stop") if gate_stop else \
                 "request_nonzero_to_zero" if request_stop else "none"
        if request_stop or gate_stop:
            self.active = True
        elif moving and channel_open:
            self.active = False
        self.previous_moving = moving
        self.previous_channel_open = channel_open
        return source != "none", self.active, source

    def reset(self):
        self.__init__()


def model_metadata(model):
    ids = [model.actuator(f"robot/joint_{i}").id for i in range(1, 7)]
    return dict(nq=model.nq, nv=model.nv, nu=model.nu, timestep=float(model.opt.timestep),
        solver=int(model.opt.solver), iterations=int(model.opt.iterations),
        ccd_iterations=int(model.opt.ccd_iterations),
        gravity=np.asarray(model.opt.gravity).tolist(),
        actuator=[dict(name=model.actuator(i).name, actadr=int(model.actuator_actadr[i]),
            gaingar=int(model.actuator_gaintype[i]), biastype=int(model.actuator_biastype[i]),
            dyntype=int(model.actuator_dyntype[i]), gainprm=model.actuator_gainprm[i].tolist(),
            biasprm=model.actuator_biasprm[i].tolist(), dynprm=model.actuator_dynprm[i].tolist(),
            ctrlrange=model.actuator_ctrlrange[i].tolist(), forcerange=model.actuator_forcerange[i].tolist())
            for i in ids],
        dof_damping=np.asarray(model.dof_damping).tolist(),
        dof_armature=np.asarray(model.dof_armature).tolist())


class Trace:
    """Record the sampled state and newly written command at each physics tick."""
    def __init__(self, env, scenario, stage, direction, speed, waveform, gate_stop=False, parking_speed=None,
                 time_offset_s=0.):
        self.env, self.scenario, self.stage = env, scenario, stage
        self.direction, self.speed, self.waveform = direction, speed, waveform
        self.data = {}
        self.contacts = []
        self.events = []
        self.tick = 0
        self.stop_gate = False
        self.stop_source = "none"
        self.gate_stop = gate_stop
        self.parking_speed = parking_speed
        self.time_offset_s = time_offset_s
        self.stop_active = False

    def put(self, name, value):
        if isinstance(value, torch.Tensor):
            item = value.detach().cpu().numpy().copy()
        elif hasattr(value, "numpy"):
            try:
                item = value.numpy().copy()
            except TypeError:
                item = value.to("cpu").numpy().copy()
        else:
            item = np.asarray(value).copy()
        self.data.setdefault(name, []).append(item)

    def before(self, term, pos, rot):
        c, d = term.controller, self.env.sim.data
        self.put("time_pre_s", self.time_offset_s + self.tick * term.cfg.contact.dt)
        self.put("q_pre_rad", term._entity.data.joint_pos[:, term._joint_ids])
        self.put("qd_pre_rad_s", term._entity.data.joint_vel[:, term._joint_ids])
        for n in ("act", "ctrl", "qfrc_actuator", "qfrc_constraint", "actuator_force"):
            if hasattr(d, n):
                self.put(n + "_pre", getattr(d, n))
        self.put("command_pre_rad", term.command)
        self.put("qvelocity_memory_pre_rad_s", term.qvelocity)
        self.put("reference_pre_m", c.xy_reference)
        self.put("ready_pre", c.ready)
        self.put("state_pre", c.state)
        self.put("wrench_raw_pre", term.raw_wrench)
        self.put("wrench_compensated_pre", term.sensor_wrench)
        self.put("wrench_target_pre", term.wrench)
        self.put("control_tip_pre_m", pos)
        self.put("control_rot_pre", rot)
        self.put("geometry_tip_pre_m", d.site_xpos[:, term.geometry_id])
        self.put("geometry_rot_pre", d.site_xmat[:, term.geometry_id])
        self.put("hole_tip_pre_m", d.site_xpos[:, term.hole_id])
        self.put("hole_rot_pre", d.site_xmat[:, term.hole_id])
        self.put("raw_action", term.xy_action)
        self.put("scaled_pre_gate_m_s", term.xy_action * term.cfg.contact.xy_speed)
        self.put("injected_lateral_m_s", term.diagnostic_lateral if term.diagnostic_lateral is not None else torch.zeros_like(pos))
        self.put("stop_event", self.stop_gate)
        self.put("stop_event_source", self.stop_source)
        self.put("stop_active", self.stop_active)
        self._contacts()

    def _contacts(self):
        sim = self.env.sim
        co = sim.wp_data.contact
        n = int(sim.wp_data.nacon.numpy()[0])
        world = co.worldid.numpy()[:n]
        geoms = co.geom.numpy()[:n]
        positions = co.pos.numpy()[:n]
        normals = co.frame.numpy()[:n]
        distances = co.dist.numpy()[:n]
        addresses = co.efc_address.numpy()[:n]
        solver_force = sim.wp_data.efc.force.numpy()[0]
        self.put("total_contact_count_pre", n)
        for j in range(n):
            if int(world[j]) != 0:
                continue
            self.contacts.append(dict(tick=self.tick, geom_ids=[int(x) for x in geoms[j]],
                position_world_m=positions[j].tolist(), normal_world=normals[j, 0].tolist(),
                distance_m=float(distances[j]),
                constraint_addresses=[int(a) for a in addresses[j] if 0 <= a < len(solver_force)],
                constraint_coefficients=[float(solver_force[a]) for a in addresses[j] if 0 <= a < len(solver_force)]))

    def after(self, term, values):
        c, d = term.controller, self.env.sim.data
        for n, v in values.items():
            self.put(n, v)
        self.put("reference_post_m", c.xy_reference)
        self.put("ready_post", c.ready)
        self.put("state_post", c.state)
        self.put("reason_post", c.reason)
        self.put("recoveries_post", c.recoveries)
        self.put("travel_post_m", c.travel)
        self.put("filtered_force_post_N", c.filtered_force)
        self.put("force_target_post_N", c.target)
        self.put("force_integral_post", c.integral)
        self.put("axial_velocity_request_post_m_s", c.velocity)
        self.put("requested_world_post_m_s", c.last_xy)
        self.put("reanchor_event", c.reanchor_event)
        self.put("accel_limited", term.accel_limited)
        self.put("energy_limited", term.energy_limited)
        self.put("ctrl_written", d.ctrl)
        self.put("act_pre_integration", d.act)
        self.put("parking_mask", term.diagnostic_parking_mask if term.diagnostic_parking_mask is not None
                 else torch.zeros_like(c.ready))
        self.put("parking_goal_world_m", term.diagnostic_parking_goal)
        self.put("parking_goal_latched", term.diagnostic_parking_active)
        self.put("executed_twist_pre", term.executed_twist)
        if self.tick:
            previous_ready = bool(self.data["ready_post"][-2][0])
            current_ready = bool(self.data["ready_post"][-1][0])
            previous_recovery = int(self.data["recoveries_post"][-2][0])
            current_recovery = int(self.data["recoveries_post"][-1][0])
            previous_reason = int(self.data["reason_post"][-2][0])
            current_reason = int(self.data["reason_post"][-1][0])
            for condition, name in ((previous_ready and not current_ready, "ready_fell"),
                                    (current_recovery > previous_recovery, "recovery_started"),
                                    (previous_reason == 0 and current_reason != 0, "fault")):
                if condition:
                    self.events.append(dict(tick=self.tick, time_s=self.time_offset_s + self.tick * term.cfg.contact.dt, event=name))
        if self.stop_gate:
            self.events.append(dict(tick=self.tick, time_s=self.time_offset_s + self.tick * term.cfg.contact.dt,
                                    event="stop_requested", source=self.stop_source))
        self.stop_gate = False
        self.stop_source = "none"
        if self.tick and self.tick % 500 == 0:
            self.events.append(dict(tick=self.tick, event="progress", time_s=self.time_offset_s + self.tick * term.cfg.contact.dt))
        self.tick += 1

    def save(self, out, term):
        out.mkdir(parents=True, exist_ok=False)
        arrays = {n: np.asarray(v) for n, v in self.data.items()}
        for n, a in arrays.items():
            if np.issubdtype(a.dtype, np.number) and not np.isfinite(a).all():
                raise FloatingPointError(f"non-finite {n}")
        arrays["step"] = np.arange(self.tick)
        arrays["run_id"] = np.full(self.tick, out.name)
        arrays["case_id"] = np.full(self.tick, self.scenario)
        arrays["pre_or_post"] = np.full(self.tick, "pre_state_plus_new_command")
        np.savez_compressed(out / "trace.npz", **arrays)
        model = self.env.sim.mj_model
        for x in self.contacts:
            x["geoms"] = [model.geom(i).name for i in x["geom_ids"]]
        write_json(out / "contacts.json", dict(backend="MuJoCo-Warp GPU", pre_integration=True,
            sampled_all_contacts=True, constraint_coefficients_are_raw_solver_values=True,
            total_records=len(self.contacts), entries=self.contacts))
        write_json(out / "events.json", self.events)
        write_json(out / "columns.json", dict(sample="pre-integration state plus newly written ctrl; row k ctrl affects row k+1 state",
            units="raw SI; names carry units where practical", position_frame="world",
            positive_axial="along current control axis, downward", contact_backend="GPU"))
        write_json(out / "run.json", dict(scenario=self.scenario, stage=self.stage,
            direction=self.direction, speed_mm_s=self.speed, waveform=self.waveform,
            samples=self.tick, dt_s=term.cfg.contact.dt, time_offset_s=self.time_offset_s,
            interventions=(["replace final transverse twist"] if self.stage == "E1" else []) +
            (["close transverse execution path on zero segments"] if self.gate_stop else []) +
            (["fixed finite XY parking goal; gain 20/s; 25 ms velocity projection; parking speed cap"]
             if self.parking_speed is not None else []) +
            (["diagnostic free-space XY permission and axial hold"] if self.scenario == "free" else [])))
        if self.parking_speed is not None:
            run = json.loads((out / "run.json").read_text())
            run["parking_speed_m_s"] = self.parking_speed
            write_json(out / "run.json", run)


def segments(waveform):
    if waveform == "W1":
        return [(1., 0.), (2., 1.), (2., 0.), (2., -1.), (2., 0.)]
    if waveform == "W2":
        return [(1., 0.), (2., 1.), (2., -1.), (2., 0.)]
    return [(2., 0.)]


def metrics(trace, direction, speed):
    x = np.asarray(trace.data["geometry_tip_pre_m"])[:, 0, :]
    axis = np.asarray(trace.data["axis"])[:, 0, :]
    t = np.asarray(trace.data["time_pre_s"])
    v = np.gradient(x, t, axis=0)
    # For the initial horizontal rig, the command basis is the frozen control frame.
    xy = x[:, :2]
    vel = v[:, :2]
    speed_xy = np.linalg.norm(vel, axis=1)
    rows = []
    cursor = 0.
    for duration, sign in segments(trace.waveform):
        end = cursor + duration
        if sign == 0 and cursor > 0:
            at = int(round(cursor / .002))
            finish = min(len(t), int(round(end / .002)))
            if at >= finish - 1:
                cursor = end
                continue
            disp = (xy[at:finish] - xy[at]) * 1000
            stable = np.convolve((speed_xy[at:finish] < .00002).astype(int), np.ones(100, dtype=int), "valid")
            stopped = float(np.flatnonzero(stable == 100)[0] * .002) if np.any(stable == 100) else None
            rows.append(dict(stop_t0_s=cursor, stop_time_s=stopped, stop_censored=stopped is None,
                max_residual_mm=float(np.linalg.norm(disp, axis=1).max()),
                path_mm=float(np.linalg.norm(np.diff(xy[at:finish], axis=0), axis=1).sum() * 1000),
                net_x_mm=float(disp[-1, 0]), net_y_mm=float(disp[-1, 1])))
        cursor = end
    return rows


def prepare_environment(scenario, cfg, out):
    env = ManagerBasedRlEnv(cfg=make_env([dict(id=scenario, dx_mm=0., dy_mm=0.)], cfg,
        plane=True, probe=ProbeConfig(amplitude_deg=0), seconds=15.), device="cuda:0")
    env.reset()
    term = env.action_manager.get_term("contact")
    if scenario == "free":
        term.controller.diagnostic_free_hold = True
        term.controller.diagnostic_xy_permission = torch.ones(1, dtype=torch.bool, device=env.device)
    else:
        # The first ready sample can still carry a large XY drive transient.
        # Keep the zero-input preparation running before freezing the state.
        for _ in range(250):
            env.step(torch.zeros(1, 2, device=env.device))
        if not bool(term.controller.ready[0]):
            raise RuntimeError("plane not ready after 10 s zero-input settling")
    state = snapshot(env, term)
    torch.save(state, out / f"{scenario}_initial.pt")
    write_json(out / f"{scenario}_model.json", model_metadata(env.sim.mj_model))
    return env, term, state


def run_one(env, term, state, out, scenario, stage, direction, speed, waveform, repeat,
            gate_stop=False, parking_speed=None):
    restore(env, term, state)
    term.controller.diagnostic_xy_permission = (torch.ones(1, dtype=torch.bool, device=env.device)
        if scenario == "free" else None)
    term.diagnostic_lateral = None
    term.diagnostic_parking_speed = parking_speed
    term.diagnostic_parking_mask = None
    term.diagnostic_parking_active.zero_()
    trace = Trace(env, scenario, stage, direction, speed, waveform, gate_stop, parking_speed)
    term.diagnostic_hook = trace
    dt = term.cfg.contact.dt
    tick = 0
    edge = StopEdge()
    for duration, sign in segments(waveform):
        count = round(duration / (20 * dt))
        for segment_step in range(count):
            action = torch.zeros(1, 2, device=env.device)
            action[0, 0 if direction == "X" else 1] = sign * speed
            moving = abs(sign * speed) > 1e-12
            channel_open = not (gate_stop and not moving and edge.previous_moving)
            trace.stop_gate, parking_active, trace.stop_source = edge.update(moving, channel_open)
            trace.stop_active = parking_active
            term.diagnostic_parking_mask = (torch.ones(1, dtype=torch.bool, device=env.device)
                if parking_speed is not None and parking_active
                else (torch.zeros(1, dtype=torch.bool, device=env.device) if parking_speed is not None else None))
            if stage == "E1":
                lateral = torch.zeros(1, 3, device=env.device)
                lateral[0, 0 if direction == "X" else 1] = sign * speed * .001
                term.diagnostic_lateral = lateral
                action.zero_()
            elif gate_stop and parking_active:
                term.diagnostic_lateral = torch.zeros(1, 3, device=env.device)
            else:
                term.diagnostic_lateral = None
            env.step(action)
            tick += 1
            if int(term.controller.reason[0]) != 0:
                trace.events.append(dict(tick=trace.tick, event="fault", reason=int(term.controller.reason[0])))
                break
        if int(term.controller.reason[0]) != 0:
            break
    term.diagnostic_hook = None
    term.diagnostic_lateral = None
    term.diagnostic_parking_speed = None
    term.diagnostic_parking_mask = None
    trace.save(out, term)
    result = metrics(trace, direction, speed)
    write_json(out / "metrics.json", dict(stops=result, fault_reason=int(term.controller.reason[0]),
        samples=trace.tick, expected_samples=round(sum(x[0] for x in segments(waveform)) / dt)))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("freeze", "E0", "screen", "extend", "candidate"))
    p.add_argument("--batch", required=True, help="new immutable batch ID")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--pilot", action="store_true", help="one 1 mm/s X pair per scenario")
    args = p.parse_args()
    if not 1 <= args.repeats <= 10:
        p.error("repeats must be 1..10")
    batch = OUT / args.batch
    cfg = replace(ContactConfig(), xy_speed=.001)
    if args.stage == "freeze":
        source_freeze(batch, cfg)
        print("FROZEN", batch, flush=True)
        return
    if not (batch / "manifest.json").is_file():
        p.error("run freeze first")
    results = []
    for scenario in ("free", "plane"):
        env = None
        try:
            env, term, state = prepare_environment(scenario, cfg, batch)
            if args.stage == "E0":
                grid = [("E2", "X", 0., "E0", 0)]
            elif args.stage == "candidate":
                directions = ("X",) if args.pilot else ("X", "Y")
                grid = [(stage, direction, 1., "W1", repeat, arm)
                    for direction in directions for stage in ("E1", "E2")
                    for repeat in range(args.repeats)
                    for arm in (("A", "B") if repeat % 2 == 0 else ("B", "A"))]
            elif args.pilot:
                grid = [(stage, "X", 1., "W1", 0) for stage in ("E1", "E2")]
            elif args.stage == "extend":
                grid = [(stage, direction, 1., "W2", repeat)
                    for stage in ("E1", "E2") for direction in ("X", "Y")
                    for repeat in range(args.repeats)]
                if scenario == "plane":
                    grid += [(stage, direction, 1., "W1", repeat)
                        for stage in ("E2", "E2_gate") for direction in ("X", "Y")
                        for repeat in range(args.repeats)]
            else:
                grid = [(stage, direction, speed, "W1", repeat)
                    for stage in ("E1", "E2") for direction in ("X", "Y")
                    for speed in (.2, 1.) for repeat in range(args.repeats)]
            for item in grid:
                stage, direction, speed, waveform, repeat = item[:5]
                arm = item[5] if len(item) > 5 else None
                run_id = f"{scenario}_{stage}_{direction}_{speed:g}_{waveform}_r{repeat}" + (f"_{arm}" if arm else "")
                destination = batch / run_id
                print("RUN", run_id, flush=True)
                stops = run_one(env, term, state, destination, scenario, stage, direction, speed, waveform, repeat,
                                gate_stop=stage == "E2_gate", parking_speed=.01 if arm == "B" else None)
                for index, row in enumerate(stops):
                    results.append(dict(run_id=run_id, stop_index=index, **row))
        finally:
            if env is not None:
                env.close()
    with (batch / f"{args.stage}_metrics.csv").open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]) if results else ["run_id"])
        writer.writeheader()
        writer.writerows(results)
    print("DONE", args.stage, len(results), flush=True)


if __name__ == "__main__":
    main()
