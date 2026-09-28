#!/usr/bin/env python3
"""One old/new E2 stop branch per i-batch prestop snapshot and short resume."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import torch

from mjlab_contact_prep.config import ContactConfig
from scripts.spike012_execution import OUT, Trace, prepare_environment, restore, sha, write_json
from scripts.analyze_spike012_stop_v2 import window_summary

SOURCE = OUT / '20260924_i'


def branch(scenario, enabled, folder):
    folder.mkdir(parents=True, exist_ok=False)
    cfg = replace(ContactConfig(), xy_speed=.001, parking_enabled=enabled)
    env = None
    try:
        env, term, _ = prepare_environment(scenario, cfg, folder)
        frozen = torch.load(SOURCE / f'{scenario}_E2/prestop.pt', weights_only=False, map_location='cpu')
        restore(env, term, frozen['state'])
        term.parking.reset()
        # The frozen i-batch snapshot predates the parking buffers. Reconstruct
        # the immediately preceding output and edge from its saved prefix trace.
        prefix = np.load(SOURCE / f'{scenario}_E2/prefix/trace.npz')
        preceding = torch.as_tensor(prefix['final_twist'][-1, 0, :3].copy(), device=env.device)
        axis = term.control_rotation[0, :, 2]
        term.parking.last_output[0] = preceding - (preceding*axis).sum()*axis
        term.parking.tracked[0] = True
        term.parking.was_moving[0] = True
        term.parking.was_permitted[0] = True
        term.parking.request_active[0] = True
        if scenario == 'free':
            term.controller.diagnostic_xy_permission = torch.ones(1, dtype=torch.bool, device=env.device)
        trace = Trace(env, scenario, 'E2', 'X', 1., 'first_stop', time_offset_s=3.)
        term.diagnostic_hook = trace
        for _ in range(50):
            env.step(torch.zeros(1, 2, device=env.device))
            if int(term.controller.reason[0]):
                break
        term.diagnostic_hook = None
        trace.save(folder / 'stop', term)
        x = np.load(folder / 'stop/trace.npz')
        stop = window_summary(x)
        states = x['parking_state'][:, 0].astype(int)
        events = x['parking_event'][:, 0].astype(int)
        result = dict(config=asdict(cfg), source_snapshot_sha256=sha(SOURCE / f'{scenario}_E2/prestop.pt'),
                      trace_sha256=sha(folder / 'stop/trace.npz'), stop=stop,
                      parking_state_counts={str(i): int(np.count_nonzero(states == i)) for i in range(3)},
                      parking_events={str(i): int(np.count_nonzero(events == i)) for i in range(1, 6)},
                      load_constrained_samples=int(np.count_nonzero(x['parking_load_fraction'][:, 0] < 1)),
                      lead_limited_samples=int(np.count_nonzero(x['parking_lead_limited'][:, 0])),
                      downstream_limited_samples=int(np.count_nonzero(x['parking_downstream_limited'][:, 0])))
        if enabled and not int(term.controller.reason[0]):
            previous = term.parking.last_output[0].clone()
            resume_trace = Trace(env, scenario, 'E2', 'X', 1., 'resume', time_offset_s=5.)
            term.diagnostic_hook = resume_trace
            for _ in range(5):
                action = torch.tensor([[1., 0.]], device=env.device)
                env.step(action)
                if int(term.controller.reason[0]):
                    break
            term.diagnostic_hook = None
            resume_trace.save(folder / 'resume', term)
            rx = np.load(folder / 'resume/trace.npz')
            outputs = rx['final_twist'][:, 0, :2]
            result['resume'] = dict(events=int(np.count_nonzero(rx['parking_event'][:, 0] == 3)),
                                    state=int(term.parking.state[0]),
                                    first_output_change_m_s=float(np.linalg.norm(outputs[0]-previous[:2].cpu().numpy())),
                                    maximum_step_change_m_s=float(np.linalg.norm(np.diff(outputs, axis=0), axis=1).max()),
                                    reference_world_m=term.controller.xy_reference[0].tolist(),
                                    fault_reason=int(term.controller.reason[0]))
        write_json(folder / 'result.json', result)
        return result
    finally:
        if env is not None:
            env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', required=True)
    parser.add_argument('--new-only', action='store_true')
    args = parser.parse_args()
    dest = OUT / args.batch
    dest.mkdir(parents=True, exist_ok=False)
    results = {}
    for scenario in ('free', 'plane'):
        for enabled in ((True,) if args.new_only else (False, True)):
            name = f'{scenario}_{"new" if enabled else "old"}'
            print('RUN', name, flush=True)
            results[name] = branch(scenario, enabled, dest / name)
            write_json(dest / 'summary.json', results)
    print(json.dumps({n: dict(Dmax_mm=r['stop']['Dmax_mm'], fz_max_N=r['stop']['fz_max_N']['value'],
                             low_force_cumulative_s=r['stop']['low_force']['cumulative_s'],
                             low_force_longest_s=r['stop']['low_force']['longest_continuous_s'])
                      for n, r in results.items()}, indent=2))


if __name__ == '__main__':
    main()
