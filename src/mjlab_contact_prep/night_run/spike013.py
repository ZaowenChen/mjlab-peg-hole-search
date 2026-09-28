"""SPIKE-013 configuration, fixed case lists and cache identity."""
from __future__ import annotations

from dataclasses import asdict, fields, replace
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from mjlab_contact_prep.config import ContactConfig
from mjlab_contact_prep.configuration import resolve_config
from .shallow_hold import PROTOCOL_ID

ROOT = Path(__file__).resolve().parents[3]


def load_experiment(overrides=()):
    document = resolve_config(ROOT / "configs/spike013_default.json",
                              ROOT / "configs/spike013_1mms_10mm_v1.json", overrides)
    effective = document["effective"]
    allowed = {f.name for f in fields(ContactConfig)}
    contact_values = effective["contact"]
    unknown = set(contact_values) - allowed
    if unknown:
        raise ValueError(f"unknown contact configuration: {sorted(unknown)}")
    contact = replace(ContactConfig(), **contact_values)
    if effective["protocol_id"] != PROTOCOL_ID:
        raise ValueError("SPIKE-013 requires autonomous_shallow_hold_v1")
    if (not math.isclose(contact.dt, .002) or
        not math.isclose(effective["physics_dt"], contact.dt) or
        effective["decimation"] != 20 or effective["repeat"] != 5 or
        effective["history_steps"] != 32 or effective["probe_amplitude_deg"] != 0 or
        not math.isclose(effective["capture_s"], .2) or not math.isclose(effective["hold_s"], .5) or
        not math.isclose(effective["preparation_s"], 6.) or
        not math.isclose(contact.xy_speed, .001) or contact.parking_enabled or
        contact.xy_tracking_limits_enabled):
        raise ValueError("SPIKE-013 runtime configuration violates the frozen first-version protocol")
    if not math.isclose(effective["search_s"] / (contact.dt * effective["decimation"] * effective["repeat"]), 200, abs_tol=1e-9):
        raise ValueError("search budget must be 40 s at 0.2 s per decision")
    bands = effective["bands_mm"]
    weights = effective["bucket_weights"]
    if (len(bands) != 5 or len(weights) != 5 or
        any(len(b) != 2 or b[0] >= b[1] for b in bands) or
        bands[0][0] != .5 or bands[-1][1] != 10 or
        any(bands[i][1] != bands[i + 1][0] for i in range(4)) or
        any(w <= 0 for w in weights) or not math.isclose(sum(weights), 1)):
        raise ValueError("invalid five-bucket distribution")
    effective["contact"] = asdict(contact)
    return document, contact


def case(radius, angle, index, bucket, kind):
    theta = math.radians(angle)
    return dict(id=f"{kind}_{index:05d}", radius_mm=float(radius), angle_deg=float(angle),
                dx_mm=float(radius * math.cos(theta)), dy_mm=float(radius * math.sin(theta)),
                bucket=bucket, kind=kind)


def bucket_of(radius, bands):
    for i, (lo, hi) in enumerate(bands):
        if lo <= radius < hi or (i == len(bands) - 1 and radius == hi):
            return i
    raise ValueError(f"radius outside configured bands: {radius}")


def cases(effective):
    bands = effective["bands_mm"]
    train_rng = np.random.default_rng(effective["train_seed"])
    test_rng = np.random.default_rng(effective["test_seed"])
    train = [case(train_rng.uniform(*bands[b]), 30 * s + train_rng.uniform(0, 30),
                  b * 480 + s * 40 + k, b, "train")
             for b in range(5) for s in range(12) for k in range(40)]
    radii = (1, 2, 5, 7.5, 10)
    validation = [case(r, 45 * k, j * 8 + k, bucket_of(r, bands), "validation")
                  for j, r in enumerate(radii) for k in range(8)]
    grid = [case(r, 5.625 + 11.25 * k, j * 32 + k, bucket_of(r, bands), "grid")
            for j, r in enumerate(radii) for k in range(32)]
    random_test = [case(test_rng.uniform(*bands[b]), 30 * s + test_rng.uniform(0, 30),
                        b * 24 + s * 2 + k, b, "heldout_random")
                   for b in range(5) for s in range(12) for k in range(2)]
    all_cases = train + validation + grid + random_test
    coordinates = [(x["dx_mm"], x["dy_mm"]) for x in all_cases]
    if len(set(coordinates)) != len(coordinates):
        raise ValueError("duplicate train/validation/test coordinate")
    return dict(train=train, validation=validation, test=grid + random_test)


def fingerprint(effective, case_list, source_hashes):
    from mjlab_contact_prep.data.state_bank import STATE_BANK_FORMAT_VERSION
    from mjlab_contact_prep.parking import LateralParking
    payload = dict(config=effective, cases=case_list, sources=source_hashes,
                   state_bank_format_version=STATE_BANK_FORMAT_VERSION,
                   parking_schema_version=LateralParking.STATE_SCHEMA_VERSION)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
