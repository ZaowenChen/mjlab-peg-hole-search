#!/usr/bin/env python3
"""Create layered request/reference/drive/response figures from SPIKE-012 traces."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def draw(folder):
    x = np.load(folder / "trace.npz")
    t = x["time_pre_s"]
    direction = folder.name.split("_")[2]
    sign = 0 if direction == "X" else 1
    p = x["geometry_tip_pre_m"][:, 0, sign] * 1000
    ref = x["reference_post_m"][:, 0, sign] * 1000
    vel = np.gradient(p, .002)
    request = x["final_twist"][:, 0, sign] * 1000
    lead = x["lead"][:, 0, sign] * 1000
    force = x["wrench_target_pre"][:, 0, 2]
    ctrl = x["ctrl_written"][:, 0, 1]
    act = x["act_pre"][:, 0, 1]
    q = x["measured_q"][:, 0, 1]
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True, constrained_layout=True)
    axes[0].plot(t, p - p[0], label="geometric tip displacement")
    axes[0].plot(t, ref - ref[0], label="XY reference displacement")
    axes[0].set_ylabel("mm")
    axes[1].plot(t, request, label="final transverse request")
    axes[1].plot(t, vel, label="measured tip velocity", alpha=.8)
    axes[1].set_ylabel("mm/s")
    axes[2].plot(t, ref - p, label="reference minus tip")
    axes[2].plot(t, lead, label="command lead, Jacobian projection")
    axes[2].set_ylabel("mm")
    axes[3].plot(t, (ctrl - q) * 1000, label="ctrl minus q, joint 2")
    axes[3].plot(t, (act - q) * 1000, label="act minus q, joint 2")
    ax2 = axes[3].twinx()
    ax2.plot(t, force, color="tab:green", alpha=.5, label="Fz")
    axes[3].set_ylabel("mrad")
    ax2.set_ylabel("N")
    axes[3].set_xlabel("s, sampled before physics integration")
    for ax in axes:
        for stop in (3, 7):
            ax.axvline(stop, color="black", linestyle="--", alpha=.55)
        ax.grid(alpha=.25)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle(folder.name)
    plots = folder / "plots"
    plots.mkdir(exist_ok=True)
    fig.savefig(plots / "chain.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("folders", nargs="+", type=Path)
    args = p.parse_args()
    for folder in args.folders:
        draw(folder)
