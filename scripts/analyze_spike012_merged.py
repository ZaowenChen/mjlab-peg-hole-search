#!/usr/bin/env python3
"""Audit merged SPIKE-012 phase routing and resume transitions from saved traces."""
import argparse
import json
from pathlib import Path

import numpy as np


def check_run(path):
    with np.load(path) as loaded:
        x = {k: loaded[k] for k in loaded.files}
    state = x['parking_state'][:, 0]
    event = x['parking_event'][:, 0]
    active = x['tracking_limits_active'][:, 0].astype(bool)
    increment = x['reference_increment'][:, 0]
    rollback = x['reference_rollback'][:, 0]
    reference_used = x['reference_used'][:, 0]
    reference_pre = x['reference_pre_m'][:, 0]
    reference_post = x['reference_post_m'][:, 0]
    parked = state != 0
    ordinary_track = active & (event != 3)
    stop_indices = np.flatnonzero(event == 1)
    fixed_goal_error = 0.
    for start in stop_indices:
        following = np.flatnonzero(event[start+1:] == 3)
        end = start+1+int(following[0]) if len(following) else len(event)
        if end > start+1:
            fixed_goal_error = max(fixed_goal_error, float(np.max(np.abs(
                x['parking_goal'][start:end, 0, :2]-x['parking_goal'][start, 0, :2]))))
    return dict(samples=len(state), stop_indices=stop_indices.tolist(),
                resume_indices=np.flatnonzero(event == 3).tolist(),
                initial_zero_stop_events=int(np.sum(event[:500] == 1)),
                track_samples=int(active.sum()), parked_samples=int(parked.sum()),
                track_and_park_overlap=int(np.sum(active & parked)),
                parked_increment_max_m=float(np.abs(increment[parked]).max(initial=0)),
                parked_rollback_max_m=float(np.abs(rollback[parked]).max(initial=0)),
                ordinary_track_single_advance_residual_max_m=float(np.abs(
                    reference_used[ordinary_track]-reference_pre[ordinary_track]-
                    increment[ordinary_track]).max(initial=0)),
                reference_feedback_residual_max_m=float(np.abs(
                    reference_post-reference_used+rollback).max()),
                goal_change_between_stop_and_resume_max_m=fixed_goal_error,
                track_twist_peak_mm_s=float(np.linalg.norm(
                    x['final_twist'][active, 0, :2], axis=-1).max()*1000),
                fault_count=int(np.sum(x['reason_post'][:, 0] != 0)))


def check_resume(path):
    with np.load(path) as loaded:
        x = {k: loaded[k] for k in loaded.files}
    event = x['parking_event'][:, 0]
    output = x['final_twist'][:, 0, :2]
    reference_pre = x['reference_pre_m'][:, 0, :2]
    reference_post = x['reference_post_m'][:, 0, :2]
    transitions = []
    for i in np.flatnonzero(event == 3):
        transitions.append(dict(tick=int(i), time_s=float(x['time_pre_s'][i]),
                                reference_handoff_mm=float(np.linalg.norm(
                                    reference_post[i]-reference_pre[i])*1000),
                                output_step_change_mm_s=float(np.linalg.norm(
                                    output[i]-output[i-1])*1000) if i else None,
                                state_after=int(x['parking_state'][i, 0])))
    changes = np.linalg.norm(np.diff(output, axis=0), axis=-1)*1000
    peak = int(np.argmax(changes))+1 if len(changes) else None
    return dict(resume=transitions, stop_events=int(np.sum(event == 1)),
                max_output_step_mm_s=float(changes.max()) if len(changes) else None,
                max_output_step_tick=peak,
                max_output_step_is_handoff=bool(peak in [v['tick'] for v in transitions]),
                final_fault_reason=int(x['reason_post'][-1, 0]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=Path, required=True)
    args = parser.parse_args()
    out = args.batch.resolve()
    runs = {p.parent.name: check_run(p) for p in sorted(out.glob('*_?_*/trace.npz'))}
    resumes = {scenario: check_resume(out/f'{scenario}_resume/trace.npz')
               for scenario in ('free', 'plane')}
    checks = dict(runs=runs, resumes=resumes,
                  all_12_runs=bool(len(runs) == 12),
                  all_full_9s=all(v['samples'] == 4500 for v in runs.values()),
                  all_track_and_park_separate=all(v['track_and_park_overlap'] == 0 for v in runs.values()),
                  all_goals_fixed=all(v['goal_change_between_stop_and_resume_max_m'] == 0 for v in runs.values()),
                  all_no_fault=all(v['fault_count'] == 0 for v in runs.values()))
    (out/'validation_checks.json').write_text(json.dumps(checks, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({k: checks[k] for k in checks if k.startswith('all_')}, indent=2))


if __name__ == '__main__':
    main()
