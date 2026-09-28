"""Ordinary XY tracking: bounded reference advance and command anti-windup.

Applied command velocity is not measured tip velocity. The reference leash uses
the measured position; feedback separately removes this step's unapplied advance.
No function reanchors a stopped reference or changes the axial reference.
"""
import torch


def project(value, axis):
    return value - (value * axis).sum(-1, keepdim=True) * axis


def advance_reference(reference, position, axis, request, dt, max_lead):
    """Advance along request only, stopping at the measured-position leash.

    If a restored reference is already outside the leash, allow inward motion
    without jumping it back or increasing its current distance. Zero request
    leaves the reference exactly unchanged, including after an external push.
    """
    # Float64 avoids cancellation close to the boundary at micrometre steps.
    offset = project(reference.double()-position.double(), axis.double())
    delta = project(request.double(), axis.double()) * dt
    distance2 = offset.square().sum(-1, keepdim=True)
    radius2 = distance2.clamp_min(max_lead**2)
    step2 = delta.square().sum(-1, keepdim=True)
    along = (offset*delta).sum(-1, keepdim=True)
    root = (along.square()+step2*(radius2-distance2)).clamp_min(0).sqrt()
    fraction = ((-along+root)/step2.clamp_min(1e-30)).clamp(0, 1)
    increment = (fraction*delta).to(reference.dtype)
    updated = reference + increment
    # Return the representable increment actually added to the reference.
    return updated, updated-reference


def limit_request(request, axis, speed):
    """Norm cap in the control plane, after feedforward and correction combine."""
    lateral = project(request, axis)
    return lateral * (speed/lateral.norm(dim=-1, keepdim=True).clamp_min(1e-12)).clamp(max=1)


def unapplied_advance(increment, desired, applied, axis, dt):
    """Undo only newly integrated advance lost to command limits / DLS.

    Neither extra achieved motion nor opposite-direction corrections accumulate
    credit. Stops have no new increment, so this cannot move a holding target.
    """
    length = increment.norm(dim=-1, keepdim=True)
    direction = increment/length.clamp_min(1e-12)
    deficit = (project(desired-applied, axis)*direction).sum(-1, keepdim=True)*dt
    return direction * torch.minimum(deficit.clamp_min(0), length)
