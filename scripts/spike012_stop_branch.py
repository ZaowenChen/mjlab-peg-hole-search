#!/usr/bin/env python3
"""SPIKE-012 F0/F1: restore the identical pre-stop state for every branch."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from mjlab_contact_prep.config import ContactConfig
from scripts.spike012_execution import (OUT, StopEdge, Trace, prepare_environment,
                                        restore, snapshot, source_freeze, write_json, sha)

DT = .002
ACTION_STEPS = 20
SPEED = .001
SCENARIOS = ("free", "plane")
STAGES = ("E1", "E2")


def tensor_digest(state):
    h = hashlib.sha256()
    for group in ("physics", "controller", "term", "probe", "diagnostics", "env"):
        for name, value in sorted(state[group].items()):
            h.update(f"{group}/{name}:{value.dtype}:{tuple(value.shape)}".encode())
            h.update(value.contiguous().numpy().tobytes())
    for name, value in sorted(state["optional_diagnostics"].items()):
        h.update(f"optional/{name}".encode())
        h.update(b"None" if value is None else value.contiguous().numpy().tobytes())
    h.update(repr(state["parking_speed"]).encode())
    return h.hexdigest()


def tensor_difference(a, b):
    errors = {}
    for group in ("physics", "controller", "term", "probe", "diagnostics", "env"):
        for name in a[group]:
            x, y = a[group][name], b[group][name]
            if not torch.equal(x, y):
                errors[f"{group}/{name}"] = float((x.double()-y.double()).abs().max())
    return errors


def observable(env, term):
    d = env.sim.data
    term.read_wrench()
    return dict(wrench=term.wrench.detach().cpu().clone(), raw=term.raw_wrench.detach().cpu().clone(),
                tip=d.site_xpos[:, term.geometry_id].detach().cpu().clone(),
                hole=d.site_xpos[:, term.hole_id].detach().cpu().clone(),
                sensor=d.sensordata.detach().cpu().clone(),
                constraint=d.qfrc_constraint.detach().cpu().clone())


def observation_difference(a, b):
    return {n: float((a[n].double()-b[n].double()).abs().max()) for n in a}


def action_for(env, moving):
    a = torch.zeros(1, 2, device=env.device)
    if moving:
        a[0, 0] = 1.
    return a


def apply_segment(env, term, trace, edge, stage, duration_s, moving, arm="A"):
    for _ in range(round(duration_s / (ACTION_STEPS*DT))):
        channel_open = arm != "G"
        event, active, source = edge.update(moving, channel_open)
        trace.stop_gate, trace.stop_active, trace.stop_source = event, active, source
        term.diagnostic_parking_speed = .01 if arm == "B" else None
        term.diagnostic_parking_mask = torch.tensor([active], device=env.device) if arm == "B" else None
        action = action_for(env, moving)
        if stage == "E1":
            lateral = torch.zeros(1, 3, device=env.device)
            lateral[0, 0] = SPEED if moving else 0.
            term.diagnostic_lateral = lateral
            action.zero_()
        elif arm == "G":
            term.diagnostic_lateral = torch.zeros(1, 3, device=env.device)
        else:
            term.diagnostic_lateral = None
        env.step(action)
        if int(term.controller.reason[0]):
            break
    return int(term.controller.reason[0])


def prepare_prefix(env, term, base, scenario, stage, out):
    restore(env, term, base)
    if scenario == "free":
        term.controller.diagnostic_xy_permission = torch.ones(1, dtype=torch.bool, device=env.device)
    edge = StopEdge()
    trace = Trace(env, scenario, stage, "X", 1., "prefix")
    term.diagnostic_hook = trace
    try:
        if apply_segment(env, term, trace, edge, stage, 1., False):
            raise RuntimeError("fault during zero prefix")
        if apply_segment(env, term, trace, edge, stage, 2., True):
            raise RuntimeError("fault during moving prefix")
    finally:
        term.diagnostic_hook = None
    if trace.tick != 1500:
        raise AssertionError(trace.tick)
    if any(bool(x) for x in trace.data["stop_event"]) or any(bool(x) for x in trace.data["stop_active"]):
        raise AssertionError("prefix contains premature stop")
    state = snapshot(env, term)
    edge_state = vars(edge).copy()
    if not edge_state["previous_moving"] or edge_state["active"]:
        raise AssertionError(edge_state)
    before = observable(env, term)
    restore(env, term, state)
    after = observable(env, term)
    audit = dict(snapshot_sha256=tensor_digest(state), state_difference=tensor_difference(state, snapshot(env, term)),
                 observable_difference=observation_difference(before, after), edge_state=edge_state,
                 stop_event_count=0, initial_mask_count=0)
    if audit["state_difference"]:
        raise AssertionError(audit["state_difference"])
    if max(audit["observable_difference"].values()) > 1e-5:
        raise AssertionError(audit["observable_difference"])
    torch.save(dict(state=state, edge=edge_state), out / "prestop.pt")
    trace.save(out / "prefix", term)
    write_json(out / "prefix_audit.json", audit)
    return state, edge_state, audit


def branch(env, term, state, edge_state, scenario, stage, arm, destination):
    restore(env, term, state)
    check = tensor_difference(state, snapshot(env, term))
    if check:
        raise AssertionError(f"precommand state differs: {check}")
    edge = StopEdge()
    vars(edge).update(edge_state)
    trace = Trace(env, scenario, stage, "X", 1., "first_stop", gate_stop=arm == "G",
                  parking_speed=.01 if arm == "B" else None, time_offset_s=3.)
    term.diagnostic_hook = trace
    try:
        fault = apply_segment(env, term, trace, edge, stage, 2., False, arm)
    finally:
        term.diagnostic_hook = None
        term.diagnostic_lateral = None
        term.diagnostic_parking_mask = None
        term.diagnostic_parking_speed = None
    if trace.tick != 1000 and not fault:
        raise AssertionError(trace.tick)
    events = [x for x in trace.events if x["event"] == "stop_requested"]
    if len(events) != 1 or abs(events[0]["time_s"]-3.) > 1e-10:
        raise AssertionError(events)
    if arm == "B":
        mask = np.asarray(trace.data["parking_mask"])[:, 0]
        latch = np.asarray(trace.data["parking_goal_latched"])[:, 0]
        if not mask.all() or not latch.all():
            raise AssertionError("parking mask/goal not latched")
    trace.save(destination, term)
    write_json(destination / "branch_audit.json", dict(prestop_sha256=tensor_digest(state),
               precommand_difference=check, edge_state=edge_state, stop_events=events,
               fault_reason=fault, arm=arm, trace_sha256=sha(destination / "trace.npz")))
    return sha(destination / "trace.npz")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=("freeze", "F0", "F1"))
    p.add_argument("--batch", required=True)
    a = p.parse_args()
    batch = OUT / a.batch
    cfg = replace(ContactConfig(), xy_speed=.001)
    if a.phase == "freeze":
        source_freeze(batch, cfg)
        write_json(batch / "analysis_protocol.json", dict(version="spike012-stop-v2", request_zero_tolerance=1e-12,
            stop_window_s=[3., 5.], sample_dt_s=DT, low_force_N=5., original_stop_threshold_mm_s=.02,
            new_stop_estimator="geometric tip finite difference plus Jacobian x measured qvel; threshold assessed from E0 before B inspection",
            stop_dwell_s=.2, Dmax_goal_mm=.05, force_working_upper_N=30., force_fault_N=40.))
        return
    if not (batch / "manifest.json").exists():
        p.error("freeze first")
    if a.phase == "F0":
        for scenario in SCENARIOS:
            env = None
            try:
                env, term, base = prepare_environment(scenario, cfg, batch)
                restore(env, term, base)
                if scenario == "free":
                    term.controller.diagnostic_xy_permission = torch.ones(1, dtype=torch.bool, device=env.device)
                e0 = Trace(env, scenario, "E2", "X", 0., "E0_v2")
                term.diagnostic_hook = e0
                try:
                    if apply_segment(env, term, e0, StopEdge(), "E2", 2., False):
                        raise RuntimeError("fault during E0")
                finally:
                    term.diagnostic_hook = None
                e0.save(batch / f"{scenario}_E0_v2", term)
                for stage in STAGES:
                    out = batch / f"{scenario}_{stage}"
                    out.mkdir(exist_ok=False)
                    print("PREFIX", scenario, stage, flush=True)
                    state, edge, audit = prepare_prefix(env, term, base, scenario, stage, out)
                    for repeat in range(2):
                        print("A/A", scenario, stage, repeat, flush=True)
                        branch(env, term, state, edge, scenario, stage, "A", out / f"AA_{repeat}")
                    x = [np.load(out / f"AA_{i}" / "trace.npz") for i in range(2)]
                    write_json(out / "aa_audit.json", dict(max_position_difference_mm=float(np.max(np.abs(
                        x[0]["geometry_tip_pre_m"]-x[1]["geometry_tip_pre_m"])) * 1000),
                        max_fz_difference_N=float(np.max(np.abs(x[0]["wrench_target_pre"][:,:,2]-x[1]["wrench_target_pre"][:,:,2]))),
                        numeric_traces_identical=all(np.array_equal(x[0][k], x[1][k]) for k in x[0].files
                                                     if np.issubdtype(x[0][k].dtype, np.number))))
            finally:
                if env is not None: env.close()
        write_json(batch / "F0_done.json", dict(completed=time.strftime("%Y-%m-%dT%H:%M:%S%z")))
    else:
        if not (batch / "F0_done.json").exists():
            p.error("F0 must pass first")
        (batch / "f1_initial").mkdir(exist_ok=False)
        for scenario in SCENARIOS:
            env = None
            try:
                env, term, base = prepare_environment(scenario, cfg, batch / "f1_initial")
                for stage in STAGES:
                    out = batch / f"{scenario}_{stage}"
                    frozen = torch.load(out / "prestop.pt", weights_only=False, map_location="cpu")
                    state, edge = frozen["state"], frozen["edge"]
                    for repeat in range(3):
                        order = ("A", "B") if repeat % 2 == 0 else ("B", "A")
                        if stage == "E2": order += ("G",)
                        for arm in order:
                            dest = out / f"r{repeat}_{arm}"
                            print("BRANCH", scenario, stage, repeat, arm, flush=True)
                            branch(env, term, state, edge, scenario, stage, arm, dest)
            finally:
                if env is not None: env.close()
        write_json(batch / "F1_done.json", dict(completed=time.strftime("%Y-%m-%dT%H:%M:%S%z")))


if __name__ == "__main__":
    main()
