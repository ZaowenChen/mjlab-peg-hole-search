#!/usr/bin/env python3
"""Paired logging-on/off E0 check from an identical full GPU state snapshot."""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from spike012_execution import Trace, prepare_environment, restore
from mjlab_contact_prep.config import ContactConfig
from dataclasses import replace


def state(env, term):
    d = env.sim.data
    term.read_wrench()
    return dict(tip=d.site_xpos[:, term.geometry_id].detach().cpu().numpy().copy(),
        qpos=d.qpos.detach().cpu().numpy().copy(),
        qvel=d.qvel.detach().cpu().numpy().copy(),
        Fz=term.wrench[:, 2].detach().cpu().numpy().copy())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("batch", type=Path)
    a = p.parse_args()
    cfg = replace(ContactConfig(), xy_speed=.001)
    results = []
    for scenario in ("free", "plane"):
        env = None
        try:
            with tempfile.TemporaryDirectory() as temporary:
                env, term, _ = prepare_environment(scenario, cfg, Path(temporary))
            prepared = torch.load(a.batch / f"{scenario}_initial.pt", weights_only=False)
            for repeat in range(3):
                final = []
                for logging in (True, False):
                    restore(env, term, prepared)
                    hook = Trace(env, scenario, "E0", "X", 0., "E0") if logging else None
                    term.diagnostic_hook = hook
                    for _ in range(50):
                        env.step(torch.zeros(1, 2, device=env.device))
                    term.diagnostic_hook = None
                    final.append(state(env, term))
                results.append(dict(scenario=scenario, repeat=repeat,
                    tip_delta_um=float(np.linalg.norm(final[0]["tip"] - final[1]["tip"]) * 1e6),
                    qpos_max_delta_rad=float(np.max(np.abs(final[0]["qpos"] - final[1]["qpos"]))),
                    qvel_max_delta_rad_s=float(np.max(np.abs(final[0]["qvel"] - final[1]["qvel"]))),
                    Fz_delta_N=float(np.abs(final[0]["Fz"] - final[1]["Fz"]).max())))
        finally:
            if env is not None:
                env.close()
    (a.batch / "e0_logging_impact.json").write_text(json.dumps(results, indent=2) + "\n")
    print(results)


if __name__ == "__main__":
    main()
