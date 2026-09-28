#!/usr/bin/env python3
"""SPIKE-012 merged TRACK/parking W1 stop and short-resume validation."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv

from mjlab_contact_prep.config import ContactConfig
from mjlab_contact_prep.environment import make_env
from mjlab_contact_prep.probe import ProbeConfig
from scripts.analyze_spike012_stop_v2 import low_force_segments, tip_velocity, window_summary
from scripts.spike012_execution import OUT, ROOT, Trace, run_one, sha, write_json

SOURCE = OUT / '20260924_i'
FILES = ['src/mjlab_contact_prep/' + n + '.py' for n in
         ('config', 'controller', 'execution', 'parking', 'xy_tracking', 'environment')]
FILES += ['scripts/validate_spike012_merged.py', 'scripts/spike012_execution.py',
          'scripts/analyze_spike012_stop_v2.py', 'tests/test_xy_tracking.py',
          'tests/test_lateral_parking.py']
DT = .002


def tip_speed(x):
    return tip_velocity(x)[0]


def stats(x):
    f = x['wrench_target_pre'][:, 0]
    radial = np.linalg.norm(f[:, :2], axis=-1)
    moment = np.linalg.norm(f[:, 3:6], axis=-1)
    geom = x['geometry_tip_pre_m'][:, 0]
    rot = x['geometry_rot_pre'][:, 0].reshape(-1, 3, 3)
    relative = np.einsum('ij,tjk->tik', rot[0].T, rot)
    angle = np.rad2deg(np.arccos(np.clip((np.trace(relative, axis1=1, axis2=2)-1)/2, -1, 1)))
    return dict(fz_max_N=float(f[:, 2].max()), fz_min_N=float(f[:, 2].min()),
                low_force=low_force_segments(f[:, 2]), radial_peak_N=float(radial.max()),
                moment_peak_Nm=float(moment.max()), z_range_mm=float(np.ptp(geom[:, 2])*1000),
                attitude_change_peak_deg=float(angle.max()),
                accel_limit_fraction=float(x['accel_limited'][:, 0].mean()),
                energy_limit_fraction=float(x['energy_limited'][:, 0].mean()),
                parking_speed_limit_fraction=float(x['parking_speed_limited'][:, 0].mean()),
                parking_acceleration_limit_fraction=float(x['parking_acceleration_limited'][:, 0].mean()),
                parking_lead_limit_fraction=float(x['parking_lead_limited'][:, 0].mean()),
                parking_load_limit_fraction=float((x['parking_load_fraction'][:, 0] < 1).mean()),
                parking_downstream_limit_fraction=float(x['parking_downstream_limited'][:, 0].mean()))


def summarize_run(path, scenario, direction, repeat):
    with np.load(path) as trace:
        x = {k: trace[k] for k in trace.files}
    n = len(x['time_pre_s'])
    initial_parking = bool(np.any(x['parking_event'][:500, 0] == 1))
    active = x['tracking_limits_active'][:, 0].astype(bool)
    parked = x['parking_state'][:, 0] != 0
    # The two scripted nonzero-to-zero transitions are exactly 3 and 7 s.
    stops = []
    for number, start in enumerate((1500, 3500), 1):
        end = min(start+1000, n)
        if end-start < 2:
            stops.append(dict(number=number, status='not_observed', samples=max(0, end-start)))
            continue
        w = {k: v[start:end] for k, v in x.items() if len(v) == n}
        summary = window_summary(w)
        brake_events = np.flatnonzero(x['parking_event'][:end, 0] == 1)
        prior = brake_events[brake_events <= start]
        brake_start = int(prior[-1]) if len(prior) else None
        event_offset = None
        if brake_start is not None and brake_start < start:
            event_geom = x['geometry_tip_pre_m'][brake_start:end, 0, :2]
            event_offset = float(np.linalg.norm(event_geom-event_geom[0], axis=-1).max()*1000)
        summary.update(number=number, status='complete' if end == start+1000 else 'truncated',
                       scripted_stop_s=start*DT, state_at_scripted_stop=int(x['parking_state'][start, 0]),
                       brake_event_s=None if brake_start is None else brake_start*DT,
                       Dmax_from_earlier_brake_event_mm=event_offset,
                       start_reference_error_xy_mm=float(np.linalg.norm(
                           x['reference_pre_m'][start, 0, :2]-x['control_tip_pre_m'][start, 0, :2])*1000),
                       end_goal_error_xy_mm=float(np.linalg.norm(
                           x['parking_goal'][end-1, 0, :2]-x['control_tip_pre_m'][end-1, 0, :2])*1000),
                       end_state=int(x['parking_state'][end-1, 0]),
                       tracking_active_samples=int(active[start:end].sum()),
                       limits=stats(w))
        stops.append(summary)
    motion = []
    for start, end in ((500, 1500), (2500, 3500)):
        if n <= start:
            continue
        end = min(end, n)
        speed = tip_speed({k: v[start:end] for k, v in x.items() if len(v) == n})
        requested = np.linalg.norm(x['tracking_desired'][start:end, 0, :2], axis=-1)*1000
        commanded = np.linalg.norm(x['final_twist'][start:end, 0, :2], axis=-1)*1000
        motion.append(dict(samples=end-start, raw_combined_peak_mm_s=float(requested.max()),
                           limited_combined_peak_mm_s=float(commanded.max()),
                           measured_mean_mm_s=float(speed.mean()), measured_peak_mm_s=float(speed.max())))
    return dict(scenario=scenario, direction=direction, repeat=repeat, trace_sha256=sha(path),
                samples=n, expected_samples=4500, fault_reason=int(x['reason_post'][-1, 0]),
                initial_zero_triggered_parking=initial_parking,
                track_active_samples=int(active.sum()), parked_samples=int(parked.sum()),
                track_limit_max_mm_s=float(np.linalg.norm(
                    x['final_twist'][active, 0, :2], axis=-1).max()*1000) if active.any() else None,
                parking_event_counts={str(e): int(np.sum(x['parking_event'][:, 0] == e)) for e in range(1, 6)},
                full=stats(x), motion=motion, stops=stops)


def resume(env, term, out, scenario):
    previous = term.parking.last_output[0].detach().cpu().numpy().copy()
    reference = term.controller.xy_reference[0].detach().cpu().numpy().copy()
    trace = Trace(env, scenario, 'E2', 'X', 1., 'resume', time_offset_s=9.)
    term.diagnostic_hook = trace
    try:
        for _ in range(25):
            env.step(torch.tensor([[1., 0.]], device=env.device))
            if int(term.controller.reason[0]):
                break
    finally:
        term.diagnostic_hook = None
    trace.save(out, term)
    with np.load(out/'trace.npz') as loaded:
        x = {k: loaded[k] for k in loaded.files}
    lateral = x['final_twist'][:, 0, :2]
    reference_series = x['reference_post_m'][:, 0, :2]
    return dict(trace_sha256=sha(out/'trace.npz'), samples=len(lateral),
                permitted_samples=int(x['ready_post'][:, 0].sum()) if scenario == 'plane' else
                int(x['requested_world_post_m_s'][:, 0, :2].any(axis=-1).sum()),
                resume_events=int(np.sum(x['parking_event'][:, 0] == 3)),
                final_state=int(x['parking_state'][-1, 0]),
                initial_reference_m=reference.tolist(),
                first_reference_jump_mm=float(np.linalg.norm(reference_series[0]-reference[:2])*1000),
                first_output_change_mm_s=float(np.linalg.norm(lateral[0]-previous[:2])*1000),
                max_step_output_change_mm_s=float(np.linalg.norm(np.diff(lateral, axis=0), axis=-1).max()*1000)
                if len(lateral) > 1 else None,
                final_fault_reason=int(x['reason_post'][-1, 0]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if not out.is_dir() or (out/'manifest.json').exists():
        raise ValueError('output must be a precreated, unused validation directory')
    cfg = replace(ContactConfig(), xy_speed=.001, parking_enabled=True,
                  xy_tracking_limits_enabled=True)
    for file in FILES:
        dest = out/'source_after'/file
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/file, dest)
    manifest = dict(config=asdict(cfg), source_sha256={f: sha(ROOT/f) for f in FILES},
                    snapshots={}, command='bash scripts/run_local.sh python -m scripts.validate_spike012_merged '
                    f'--output {out}', physics_dt_s=DT, protocol='W1: 1s zero, 2s +1 mm/s, 2s zero, '
                    '2s -1 mm/s, 2s zero; 3 repeats per scenario/direction; 0.5s resume per scenario; '
                    'ordinary gated path, no E1 injection')
    rows, resumes = [], {}
    write_json(out/'manifest.json', manifest)
    for scenario in ('free', 'plane'):
        source = SOURCE/f'{scenario}_initial.pt'
        manifest['snapshots'][scenario] = dict(path=str(source), sha256=sha(source))
        state = torch.load(source, map_location='cpu', weights_only=False)
        env = ManagerBasedRlEnv(cfg=make_env([dict(id=scenario, dx_mm=0., dy_mm=0.)], cfg,
            plane=True, probe=ProbeConfig(amplitude_deg=0), seconds=15.), device='cuda:0')
        try:
            env.reset()
            term = env.action_manager.get_term('contact')
            term.controller.diagnostic_free_hold = scenario == 'free'
            if 'diagnostic_xy_permission' in state['controller']:
                term.controller.diagnostic_xy_permission = state['controller']['diagnostic_xy_permission'].to(env.device).clone()
            for direction in ('X', 'Y'):
                for repeat in range(1, 4):
                    term.parking.reset()
                    folder = out/f'{scenario}_{direction}_{repeat}'
                    print('RUN', folder.name, flush=True)
                    run_one(env, term, state, folder, scenario, 'E2', direction, 1., 'W1', repeat)
                    row = summarize_run(folder/'trace.npz', scenario, direction, repeat)
                    rows.append(row)
                    write_json(out/'summary.json', rows)
                    print(json.dumps(dict(run=folder.name, fault=row['fault_reason'],
                        Dmax_mm=[s.get('Dmax_mm') for s in row['stops']],
                        Fz_max_N=row['full']['fz_max_N'])), flush=True)
                    if direction == 'X' and repeat == 1:
                        resumes[scenario] = resume(env, term, out/f'{scenario}_resume', scenario)
                        write_json(out/'resume_summary.json', resumes)
        finally:
            env.close()
        write_json(out/'manifest.json', manifest)
    write_json(out/'summary.json', rows)
    write_json(out/'resume_summary.json', resumes)


if __name__ == '__main__':
    main()
