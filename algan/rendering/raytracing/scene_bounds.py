"""Shared host facts derived once from an immutable merged triangle table."""

from __future__ import annotations

from typing import NamedTuple

import torch


class TriangleBounds(NamedTuple):
    lower: tuple[float, float, float]
    upper: tuple[float, float, float]
    diagonal: float


def triangle_scene_bounds(merged):
    """Return cached extrema and the original tensor-precision diagonal.

    The input is the merged renderer's float32 triangle-position table, over
    all of its stored frames. Do not replace the torch norm with a host
    double-precision norm: the diagonal feeds the shadow acceptance threshold.
    A newly prepared scene owns a new dictionary; mutating tri_pos in place
    after preparation is outside the immutable-batch contract.
    """
    bounds = merged.get("_triangle_scene_bounds")
    if bounds is None:
        positions = merged["tri_pos"]
        if positions.numel():
            vertices = positions.reshape(-1, 3)
            lo, hi = vertices.amin(0), vertices.amax(0)
            values = torch.cat((lo, hi, (hi - lo).norm().reshape(1))).tolist()
            bounds = TriangleBounds(tuple(values[:3]), tuple(values[3:6]), values[6])
        else:
            # Preserve the Morton quantizer's empty-box fallback. Shadows
            # retain a zero scale and therefore use their minimum-hit floor.
            bounds = TriangleBounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 0.0)
        merged["_triangle_scene_bounds"] = bounds
    return bounds
