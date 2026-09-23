"""Reset-pose randomization with disk-bounded translation and tilt."""

from __future__ import annotations

import math

import torch


def safe_hole_mouth_height(
    tip_height: torch.Tensor,
    lateral_xy: torch.Tensor,
    tilt_xy: torch.Tensor,
    axial_gap: float,
    peg_radius: float,
    peg_face_slope_xy: tuple[float, float],
) -> torch.Tensor:
    """Return a safe tilted-mouth Z while preserving the nominal face gap.

    ``lateral_xy`` is the mouth-center displacement from the peg-tip center.
    The first correction accounts for the tilted mouth plane at the peg XY
    position.  The second accounts for the finite peg-front radius and its
    small calibrated start-pose tilt.  At zero hole tilt this deliberately
    reduces to ``tip_height - axial_gap`` so existing 5 mm resets are unchanged.
    """
    if lateral_xy.ndim != 2 or lateral_xy.shape[1] != 2:
        raise ValueError("lateral_xy must have shape (N, 2)")
    if tilt_xy.shape != lateral_xy.shape:
        raise ValueError("tilt_xy must have the same shape as lateral_xy")
    if tip_height.ndim != 1 or tip_height.shape[0] != lateral_xy.shape[0]:
        raise ValueError("tip_height must have shape (N,)")
    if axial_gap < 0.0 or peg_radius < 0.0:
        raise ValueError("axial_gap and peg_radius must be non-negative")

    angle = torch.linalg.vector_norm(tilt_xy, dim=-1)
    safe_angle = torch.clamp(angle, min=torch.finfo(tilt_xy.dtype).eps)
    sine_scale = torch.sin(angle) / safe_angle
    # Rodrigues rotation of the local +Z mouth normal by [rx, ry, 0].
    normal_xy = torch.stack(
        (tilt_xy[:, 1] * sine_scale, -tilt_xy[:, 0] * sine_scale), dim=-1
    )
    normal_z = torch.cos(angle)
    if torch.any(normal_z <= 0.0):
        raise ValueError("tilt must keep the hole mouth normal above horizontal")
    mouth_slope = normal_xy / normal_z.unsqueeze(-1)

    peg_slope = torch.as_tensor(
        peg_face_slope_xy, device=tilt_xy.device, dtype=tilt_xy.dtype
    )
    nominal_edge_drop = float(peg_radius) * torch.linalg.vector_norm(peg_slope)
    tilted_edge_drop = float(peg_radius) * torch.linalg.vector_norm(
        mouth_slope - peg_slope, dim=-1
    )
    extra_edge_clearance = torch.clamp(
        tilted_edge_drop - nominal_edge_drop, min=0.0
    )
    surface_height_at_tip = torch.sum(mouth_slope * lateral_xy, dim=-1)
    return (
        tip_height
        - float(axial_gap)
        - surface_height_at_tip
        - extra_edge_clearance
    )


def sample_uniform_disk(
    count: int,
    radius: float,
    *,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample ``count`` XY vectors uniformly by area inside a disk."""
    if count < 0:
        raise ValueError("count must be non-negative")
    if radius < 0.0:
        raise ValueError("radius must be non-negative")
    uniform = torch.rand((count, 2), device=device, generator=generator)
    sample_radius = float(radius) * torch.sqrt(uniform[:, 0])
    phase = 2.0 * math.pi * uniform[:, 1]
    return torch.stack(
        (sample_radius * torch.cos(phase), sample_radius * torch.sin(phase)),
        dim=-1,
    )


def sample_uniform_annulus(
    count: int,
    inner_radius: float,
    outer_radius: float,
    *,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample XY vectors uniformly by area inside a closed annulus."""
    if count < 0:
        raise ValueError("count must be non-negative")
    if inner_radius < 0.0 or outer_radius < 0.0:
        raise ValueError("annulus radii must be non-negative")
    if inner_radius > outer_radius:
        raise ValueError("inner_radius must not exceed outer_radius")
    uniform = torch.rand((count, 2), device=device, generator=generator)
    radius_squared = (
        float(inner_radius) ** 2
        + uniform[:, 0]
        * (float(outer_radius) ** 2 - float(inner_radius) ** 2)
    )
    sample_radius = torch.sqrt(radius_squared)
    phase = 2.0 * math.pi * uniform[:, 1]
    return torch.stack(
        (sample_radius * torch.cos(phase), sample_radius * torch.sin(phase)),
        dim=-1,
    )


def sample_lateral_and_tilt(
    count: int,
    max_lateral_offset: float,
    max_tilt: float,
    *,
    lateral_outer_inner_radius: float = 0.0,
    lateral_outer_fraction: float = 0.0,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample disk-bounded pose vectors with optional lateral outer-ring mixing.

    ``lateral_outer_fraction`` of each batch is drawn uniformly by area from
    ``[lateral_outer_inner_radius, max_lateral_offset]``.  The remainder is
    drawn from the old-range disk ending at ``lateral_outer_inner_radius``.
    This gives every new curriculum ring explicit coverage while retaining
    easier states from already learned ranges.
    """
    if not 0.0 <= lateral_outer_fraction <= 1.0:
        raise ValueError("lateral_outer_fraction must lie in [0, 1]")
    if lateral_outer_inner_radius < 0.0:
        raise ValueError("lateral_outer_inner_radius must be non-negative")
    if lateral_outer_inner_radius > max_lateral_offset:
        raise ValueError(
            "lateral_outer_inner_radius must not exceed max_lateral_offset"
        )
    if lateral_outer_fraction > 0.0 and max_lateral_offset == 0.0:
        raise ValueError("a non-zero outer-ring fraction requires a non-zero radius")

    if lateral_outer_fraction == 0.0:
        lateral = sample_uniform_disk(
            count, max_lateral_offset, device=device, generator=generator
        )
    else:
        outer_count = int(math.ceil(count * lateral_outer_fraction))
        old_count = count - outer_count
        old = sample_uniform_disk(
            old_count,
            lateral_outer_inner_radius,
            device=device,
            generator=generator,
        )
        outer = sample_uniform_annulus(
            outer_count,
            lateral_outer_inner_radius,
            max_lateral_offset,
            device=device,
            generator=generator,
        )
        lateral = torch.cat((old, outer), dim=0)
        if count > 1:
            order = torch.randperm(count, device=device, generator=generator)
            lateral = lateral[order]
    tilt = sample_uniform_disk(count, max_tilt, device=device, generator=generator)
    return lateral, tilt
