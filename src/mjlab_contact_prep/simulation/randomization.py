"""Stable sampling API; historical imports remain supported."""

from mjlab_contact_prep.randomization import (  # noqa: F401
    safe_hole_mouth_height,
    sample_lateral_and_tilt,
    sample_uniform_annulus,
    sample_uniform_disk,
)

__all__ = [
    "safe_hole_mouth_height",
    "sample_lateral_and_tilt",
    "sample_uniform_annulus",
    "sample_uniform_disk",
]
