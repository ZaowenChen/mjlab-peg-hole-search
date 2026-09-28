"""Record and play the SPIKE-013 validation cases in MuJoCo Viewer.

Recording reruns the frozen preparation cases and the selected PPO checkpoint.
It does not modify the training experiment or its validation results. The saved
model and poses let `play` run without MJLab, Torch, or a GPU.
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import time
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "evaluation/spike013_1mms_10mm/20260924_ppo_small_v1"
DEFAULT_OUTPUT = EXPERIMENT / "viewer_replay_v1"
FIELDS = ("qpos", "qvel", "mocap_pos", "mocap_quat")
PHYSICS_DT = 0.002
RECORD_EVERY = 5  # One visual pose every 10 ms; physics still runs at 2 ms.


def capture(sim, frames):
    for name in FIELDS:
        frames[name].append(getattr(sim.data, name).detach().clone())


def archive(frames, path):
    import torch

    arrays = {name: np.asarray(torch.stack(values).cpu().numpy())
              for name, values in frames.items()}
    np.savez_compressed(path, **arrays)
    return {name: list(value.shape) for name, value in arrays.items()}


def record(output: Path, checkpoint: int, limit: int) -> None:
    import torch
    from dataclasses import replace
    from mjlab.envs import ManagerBasedRlEnv
    from rsl_rl.runners import OnPolicyRunner
    from mjlab_contact_prep.config import ContactConfig
    from mjlab_contact_prep.environment import make_env
    from mjlab_contact_prep.probe import ProbeConfig
    from mjlab_contact_prep.night_run.environment import BankEnv
    from mjlab_contact_prep.night_run.jobs import ppo_config, verify_frozen
    from mjlab_contact_prep.night_run.shallow_hold import PROTOCOL_ID

    output.mkdir(parents=True, exist_ok=False)
    plan = json.loads((EXPERIMENT / "plan.json").read_text())
    verify_frozen(plan)
    cases = plan["validation_cases"][:limit]
    contact = ContactConfig(**plan["effective_config"]["contact"])
    reference = json.loads((EXPERIMENT / f"amp_0/seed_7/validation_u{checkpoint:04d}/results.json").read_text())
    reference_by_id = {row["id"]: row for row in reference["rows"]}
    bank = torch.load(EXPERIMENT / "validation_bank/bank.pt", weights_only=False, map_location="cpu")
    assert bank["fingerprint"] == plan["validation_fingerprint"]
    assert [row["id"] for row in bank["rows"][:limit]] == [row["id"] for row in cases]

    # Reconstruct the full contact preparation from its registered cases. The
    # original validation bank begins only after these six seconds.
    prep = ManagerBasedRlEnv(
        cfg=make_env(cases, contact, probe=ProbeConfig(amplitude_deg=0)), device="cuda:0"
    )
    prep_frames = {name: [] for name in FIELDS}
    try:
        prep.reset()
        capture(prep.sim, prep_frames)
        original_step = prep.sim.step
        ticks = 0

        def sampled_prep_step():
            nonlocal ticks
            original_step()
            ticks += 1
            if ticks % RECORD_EVERY == 0:
                capture(prep.sim, prep_frames)

        prep.sim.step = sampled_prep_step
        zero = torch.zeros(len(cases), 2, device="cuda:0")
        with torch.no_grad():
            for _ in range(round(plan["preparation_s"] / prep.step_dt)):
                prep.step(zero)
        assert ticks == round(plan["preparation_s"] / PHYSICS_DT)
        preparation_shapes = archive(prep_frames, output / "preparation_states.npz")
        prep_end = {name: getattr(prep.sim.data, name).detach().cpu().numpy().copy()
                    for name in FIELDS}
    finally:
        prep.close()

    # Rerun the exact 40-environment validation protocol and record poses after
    # each fifth 2 ms integration. Finished cases are cut at their first done.
    env = BankEnv(
        EXPERIMENT / "validation_bank/bank.pt", len(cases), 0., 7,
        autoreset=False, horizon=plan["horizon_steps"], contact=contact,
        protocol_id=PROTOCOL_ID, expected_fingerprint=plan["validation_fingerprint"],
        bucket_weights=plan["effective_config"]["bucket_weights"],
    )
    search_frames = {name: [] for name in FIELDS}
    outcomes = []
    try:
        mujoco.mj_saveModel(env.env.sim.mj_model, str(output / "model.mjb"))
        capture(env.env.sim, search_frames)
        original_step = env.env.sim.step
        ticks = 0

        def sampled_search_step():
            nonlocal ticks
            original_step()
            ticks += 1
            if ticks % RECORD_EVERY == 0:
                capture(env.env.sim, search_frames)

        env.env.sim.step = sampled_search_step
        runner = OnPolicyRunner(env, ppo_config(7, 1), None, device=env.device)
        runner.load(str(EXPERIMENT / f"amp_0/seed_7/checkpoint_{checkpoint:04d}.pt"), map_location=env.device)
        policy = runner.get_inference_policy()
        active = torch.ones(len(cases), dtype=torch.bool, device=env.device)
        ended = {}
        with torch.inference_mode():
            for decision in range(1, plan["horizon_steps"] + 1):
                if not bool(active.any()):
                    break
                actions = policy(env.get_observations()).clamp(-1, 1)
                actions[~active] = 0
                _, _, done, _ = env.step(actions)
                for i in torch.nonzero(active & done.bool()).flatten().tolist():
                    ended[i] = {"decision": decision,
                                "success": bool(env.last_success[i]),
                                "fault": bool(env.last_fault[i]),
                                "first_capture_s": (None if env.shallow_hold.first_capture_tick[i] < 0
                                                    else float(env.shallow_hold.first_capture_tick[i]) * PHYSICS_DT)}
                active &= ~done.bool()
        assert len(ended) == len(cases)
        search_shapes = archive(search_frames, output / "search_states.npz")
    finally:
        env.close()

    for i, case in enumerate(cases):
        original = reference_by_id[case["id"]]
        actual = ended[i]
        outcome = "success" if actual["success"] else "fault" if actual["fault"] else "timeout"
        outcomes.append({
            "index": i, "id": case["id"], "radius_mm": case["radius_mm"],
            "angle_deg": case["angle_deg"],
            "first_touch_s": bank["rows"][i]["stats"]["first_touch_s"],
            "preparation_eligible": bool(bank["rows"][i]["eligible"]),
            "outcome": outcome, "search_s": actual["decision"] * .2,
            "first_capture_s": actual["first_capture_s"],
            "reference_outcome": original["outcome"],
            "reference_search_s": original["search_s"],
            "matches_validation": (outcome == original["outcome"] and
                                   math.isclose(actual["decision"] * .2, original["search_s"], abs_tol=1e-6)),
        })

    handoff = {
        "max_abs_qpos": float(np.max(np.abs(prep_end["qpos"] - bank["physics"]["qpos"][:limit].numpy()))),
        "max_abs_mocap_pos_m": float(np.max(np.abs(prep_end["mocap_pos"] - bank["physics"]["mocap_pos"][:limit].numpy()))),
    }
    metadata = {
        "source_experiment": str(EXPERIMENT), "checkpoint_update": checkpoint,
        "source_validation_bank_fingerprint": plan["validation_fingerprint"],
        "physics_dt": PHYSICS_DT, "sample_dt": PHYSICS_DT * RECORD_EVERY,
        "preparation_shapes": preparation_shapes, "search_shapes": search_shapes,
        "handoff_difference": handoff, "cases": outcomes,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(f"Recorded {len(cases)} cases in {output}", flush=True)
    print(f"Validation matches: {sum(x['matches_validation'] for x in outcomes)}/{len(outcomes)}", flush=True)
    print(f"Preparation-to-bank differences: {handoff}", flush=True)


def play(output: Path, speed: float, fps: int, start: int) -> None:
    import mujoco.viewer

    metadata = json.loads((output / "metadata.json").read_text())
    cases = metadata["cases"]
    if not 1 <= start <= len(cases):
        raise ValueError(f"--start must be between 1 and {len(cases)}")
    model = mujoco.MjModel.from_binary_path(str(output / "model.mjb"))
    data = mujoco.MjData(model)
    with np.load(output / "preparation_states.npz") as saved:
        prep = {name: saved[name] for name in FIELDS}
    with np.load(output / "search_states.npz") as saved:
        search = {name: saved[name] for name in FIELDS}
    dt = float(metadata["sample_dt"])
    controls = queue.SimpleQueue()

    def on_key(keycode):
        try:
            controls.put(chr(keycode).lower())
        except ValueError:
            pass

    index = start - 1
    current_time = 0.0
    paused = False
    next_frame = time.monotonic()
    print("Controls: Space pause/resume | N next | B previous | R restart | +/- speed | Q quit", flush=True)

    def show_case():
        c = cases[index]
        exact = "matched" if c["matches_validation"] else "fresh rerun differs"
        print(f"[{index+1:02d}/{len(cases)}] {c['id']} | {c['radius_mm']:g} mm, "
              f"{c['angle_deg']:g}° | {c['outcome']} at {c['search_s']:.1f} s | {exact}", flush=True)

    show_case()
    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        with viewer.lock():
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.lookat[:] = (0.85, 0.0, 0.10)
            viewer.cam.distance = 0.35
            viewer.cam.azimuth = 115
            viewer.cam.elevation = -25
            label = viewer.user_scn.geoms[0]
            mujoco.mjv_initGeom(
                label, mujoco.mjtGeom.mjGEOM_LABEL,
                np.zeros(3), np.array([0.85, 0.0, 0.17]), np.eye(3).ravel(),
                np.array([1., 1., 1., 1.], dtype=np.float32),
            )
            viewer.user_scn.ngeom = 1
        while viewer.is_running():
            while not controls.empty():
                key = controls.get_nowait()
                if key == "q":
                    return
                if key == " ":
                    paused = not paused
                elif key == "n":
                    index = (index + 1) % len(cases)
                    current_time = 0.0
                    show_case()
                elif key == "b":
                    index = (index - 1) % len(cases)
                    current_time = 0.0
                    show_case()
                elif key == "r":
                    current_time = 0.0
                elif key in ("+", "="):
                    speed = min(32.0, speed * 2)
                    print(f"Playback speed: {speed:g}x", flush=True)
                elif key == "-":
                    speed = max(0.25, speed / 2)
                    print(f"Playback speed: {speed:g}x", flush=True)

            case = cases[index]
            prep_duration = (len(prep["qpos"]) - 1) * dt
            duration = prep_duration + case["search_s"]
            if not paused and current_time >= duration + speed * .75:
                index = (index + 1) % len(cases)
                current_time = 0.0
                show_case()
                case = cases[index]
            if current_time <= prep_duration:
                states = prep
                phase_time = current_time
                stage = "APPROACH" if current_time < (case["first_touch_s"] or prep_duration) else "CONTACT"
            else:
                states = search
                phase_time = current_time - prep_duration
                capture_time = case["first_capture_s"]
                stage = "SEARCH" if capture_time is None or phase_time < capture_time else "CAPTURE/HOLD"
                if phase_time >= case["search_s"]:
                    stage = "FINISHED"
            frame = min(round(phase_time / dt), len(states["qpos"]) - 1)
            for field in FIELDS:
                getattr(data, field)[:] = states[field][frame, index]
            data.time = current_time
            with viewer.lock():
                difference = "  DIFF FROM ORIGINAL" if case["outcome"] != case["reference_outcome"] else ""
                label.label = (f"{index+1:02d}/{len(cases)}  {case['radius_mm']:g}mm "
                               f"{case['angle_deg']:g}deg  {stage}  {phase_time:.1f}s  "
                               f"REPLAY {case['outcome'].upper()}  {speed:g}x{difference}")
            viewer.sync(state_only=True)
            if not paused:
                current_time += speed / fps
            next_frame += 1 / fps
            delay = next_frame - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_frame = time.monotonic()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    rec = sub.add_parser("record", help="Rerun and save visualization states")
    rec.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    rec.add_argument("--checkpoint", type=int, choices=(16, 64, 128), default=128)
    rec.add_argument("--limit", type=int, choices=range(1, 41), default=40)
    view = sub.add_parser("play", help="Play saved cases in one MuJoCo window")
    view.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    view.add_argument("--speed", type=float, default=4.0)
    view.add_argument("--fps", type=int, default=30)
    view.add_argument("--start", type=int, default=1, help="One-based case number")
    args = parser.parse_args()
    if args.command == "record":
        record(args.output, args.checkpoint, args.limit)
    else:
        if args.speed <= 0 or args.fps <= 0:
            parser.error("--speed and --fps must be positive")
        play(args.output, args.speed, args.fps, args.start)


if __name__ == "__main__":
    main()
