#!/usr/bin/env python3
"""Four paired ordinary-path W1 runs from existing frozen SPIKE-012 states."""
import argparse
from dataclasses import asdict, replace
from pathlib import Path
import shutil

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab_contact_prep.config import ContactConfig
from mjlab_contact_prep.environment import make_env
from mjlab_contact_prep.probe import ProbeConfig
from scripts.spike012_execution import ROOT, OUT, run_one, sha, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True, help='new output directory')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cfg = replace(ContactConfig(), xy_speed=.001, parking_enabled=False, xy_tracking_limits_enabled=True)
    files = ['src/mjlab_contact_prep/'+name+'.py' for name in
             ('config', 'controller', 'execution', 'parking', 'xy_tracking', 'environment')]
    files += ['scripts/validate_xy_tracking.py', 'scripts/spike012_execution.py']
    for name in files:
        dest = out/'source'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, dest)
    manifest = dict(config=asdict(cfg), source_sha256={f: sha(ROOT/f) for f in files},
                    snapshots={}, protocol='E2 W1 X: zero 1s, +1mm/s 2s, zero 2s, -1mm/s 2s, zero 2s; '
                    'same frozen state per pair; only xy_tracking_limits_enabled differs; no PPO')
    rows = []
    for scenario in ('free', 'plane'):
        source = OUT/'20260924_i'/f'{scenario}_initial.pt'
        manifest['snapshots'][scenario] = dict(path=str(source), sha256=sha(source))
        state = torch.load(source, map_location='cpu', weights_only=False)
        env = ManagerBasedRlEnv(cfg=make_env([dict(id=scenario, dx_mm=0., dy_mm=0.)], cfg,
            plane=True, probe=ProbeConfig(amplitude_deg=0), seconds=15.), device='cuda:0')
        try:
            env.reset()
            term = env.action_manager.get_term('contact')
            term.controller.diagnostic_free_hold = scenario == 'free'
            # Old free-space banks serialized this optional tensor. Allocate
            # its destination before the historical in-place restore helper.
            if 'diagnostic_xy_permission' in state['controller']:
                term.controller.diagnostic_xy_permission = state['controller']['diagnostic_xy_permission'].to(env.device).clone()
            for enabled in (False, True):
                arm = 'fixed' if enabled else 'legacy'
                effective = replace(cfg, xy_tracking_limits_enabled=enabled)
                term.cfg.contact = term.controller.cfg = term.parking.cfg = effective
                term.parking.reset()
                folder = out/f'{scenario}_{arm}'
                run_one(env, term, state, folder, scenario, 'E2', 'X', 1., 'W1', 0)
                write_json(folder/'effective_config.json', asdict(effective))
                with np.load(folder/'trace.npz') as x:
                    axis = x['axis'][:, 0]
                    project = lambda v: v-(v*axis).sum(-1, keepdims=True)*axis
                    command_speed = np.linalg.norm(project(x['final_twist'][:, 0, :3]), axis=-1)*1000
                    lag = np.linalg.norm(project(x['reference_post_m'][:, 0]-x['pos'][:, 0]), axis=-1)*1000
                    actual_speed = np.linalg.norm(project(x['executed_twist_pre'][:, 0, :3]), axis=-1)*1000
                    force = x['wrench_target_pre'][:, 0, 2]
                    low = force < 5
                    edges = np.diff(np.r_[False, low, False].astype(int))
                    spans = np.flatnonzero(edges == -1)-np.flatnonzero(edges == 1)
                    row = dict(scenario=scenario, arm=arm, samples=len(force),
                        max_request_mm_s=float(command_speed.max()), max_reference_lag_mm=float(lag.max()),
                        max_actual_control_speed_mm_s=float(actual_speed.max()),
                        peak_Fz_N=float(force.max()), low_total_s=float(low.sum()*cfg.dt),
                        low_longest_s=float(spans.max()*cfg.dt) if len(spans) else 0.,
                        fault_reason=int(term.controller.reason[0]))
                    if enabled:
                        assert row['max_request_mm_s'] <= 1.00001, row
                    rows.append(row)
                    print(row, flush=True)
        finally:
            env.close()
    write_json(out/'manifest.json', manifest)
    write_json(out/'summary.json', rows)


if __name__ == '__main__':
    main()
