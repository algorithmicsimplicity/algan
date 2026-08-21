"""Check that ``_clipped_screen_extents`` bounds every reachable hit pixel.

The straddler clip only removes candidate pixels; that is output-neutral only
if it never removes a pixel a primary ray could actually hit. This brute-forces
that property: for random triangles and boxes straddling the camera plane, it
densely samples the primitive, keeps the samples in front of the camera, and
asserts every one of their projections falls inside the reported extent (plus
the one-pixel pad the bbox builder adds), or that the primitive was reported
unbounded.

Each primitive gets its own random camera, so it occupies its own frame slot
in the ``[frames, primitives, ...]`` layout the extent helper expects (one
primitive per frame). The dense sampling is the expensive part -- ``primitives
x samples x 3`` floats, several times over inside the reference projection --
so it runs in primitive chunks sized to keep each pass to a few tens of MB.

    .venv/Scripts/python.exe benchmarks/_raster_clip_extents_check.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch  # noqa: E402

from algan.rendering.raytracing.raster_pipeline import (  # noqa: E402
    _BOX_EDGES,
    _TRI_EDGES,
    _aabb_corners,
    _clipped_screen_extents,
)

HALF_W, HALF_H = 960.0, 540.0
PAD = 1.0
NUM_PRIMITIVES = 4000
NUM_SAMPLES = 4000
CHUNK = 100


def _project(points, ro, sp, pbx, pby):
    """Reference projection, matching precompute_triangle_projection."""
    nvec = torch.linalg.cross(pbx, pby)
    big_d = ((sp - ro) * nvec).sum(-1)
    d = points - ro[:, None, :]
    denom = (d * nvec[:, None, :]).sum(-1)
    t = big_d[:, None] / denom
    hit = ro[:, None, :] + t[..., None] * d
    rel = hit - sp[:, None, :]
    n2 = (nvec * nvec).sum(-1)
    u = (
        torch.linalg.cross(rel, pby[:, None, :].expand_as(rel)) * nvec[:, None, :]
    ).sum(-1) / n2[:, None]
    v = (
        torch.linalg.cross(pbx[:, None, :].expand_as(rel), rel) * nvec[:, None, :]
    ).sum(-1) / n2[:, None]
    sign = torch.where(big_d >= 0, 1.0, -1.0)
    return (u * HALF_H + HALF_W, v * HALF_H + HALF_H, denom * sign[:, None] > 0)


def _sample_extents(points, ro, sp, pbx, pby):
    """Screen extent of the samples a primary ray could actually return.

    Restricted to samples in front of the camera that also land on screen:
    those are the only ones the candidate bbox has to cover, since the caller
    clamps it to the frame anyway.  Without that restriction a single sample
    projecting to x = 1e8 would dominate the extent and every straddler would
    look like a miss.
    """
    px, py, front = _project(points, ro, sp, pbx, pby)
    visible = (
        front
        & (px >= -PAD)
        & (px <= 2 * HALF_W + PAD)
        & (py >= -PAD)
        & (py <= 2 * HALF_H + PAD)
    )
    inf = torch.tensor(float("inf"))
    return (
        torch.where(visible, px, inf).amin(-1),
        torch.where(visible, px, -inf).amax(-1),
        torch.where(visible, py, inf).amin(-1),
        torch.where(visible, py, -inf).amax(-1),
    )


def _barycentric(n):
    a = torch.rand(n)
    b = torch.rand(n)
    flip = a + b > 1
    a = torch.where(flip, 1 - a, a)
    b = torch.where(flip, 1 - b, b)
    return torch.stack((a, b, 1 - a - b), -1)


def _camera(num):
    ro = torch.randn(num, 3) * 2.0
    forward = torch.nn.functional.normalize(torch.randn(num, 3), dim=-1)
    up = torch.nn.functional.normalize(
        torch.linalg.cross(forward, torch.randn(num, 3)), dim=-1
    )
    right = torch.linalg.cross(up, forward)
    dist = 1.0 + torch.rand(num, 1) * 3.0
    sp = ro + forward * dist
    return ro, sp, right * dist * 0.9, up * dist * 0.5


def _chunks(num):
    for start in range(0, num, CHUNK):
        yield start, min(start + CHUNK, num)


def check_triangles(seed=0):
    torch.manual_seed(seed)
    ro, sp, pbx, pby = _camera(NUM_PRIMITIVES)
    # Centre the triangles on the camera so a good fraction straddle its plane.
    verts = ro[:, None, :] + torch.randn(NUM_PRIMITIVES, 3, 3) * 2.5
    extents = _clipped_screen_extents(
        verts.unsqueeze(1), _TRI_EDGES, ro, sp, pbx, pby, HALF_W, HALF_H
    )
    weights = _barycentric(NUM_SAMPLES)

    sampled = []
    for a, b in _chunks(NUM_PRIMITIVES):
        points = torch.einsum("sk,nkc->nsc", weights, verts[a:b])
        sampled.append(
            torch.stack(_sample_extents(points, ro[a:b], sp[a:b], pbx[a:b], pby[a:b]))
        )
    return _report("triangles", extents, torch.cat(sampled, -1))


def check_boxes(seed=1):
    torch.manual_seed(seed)
    ro, sp, pbx, pby = _camera(NUM_PRIMITIVES)
    centre = ro + torch.randn(NUM_PRIMITIVES, 3) * 2.0
    half = torch.rand(NUM_PRIMITIVES, 3) * 1.5 + 0.1
    lo = centre - half
    hi = centre + half
    corners = _aabb_corners(lo, hi)
    extents = _clipped_screen_extents(
        corners.unsqueeze(1), _BOX_EDGES, ro, sp, pbx, pby, HALF_W, HALF_H
    )

    sampled = []
    for a, b in _chunks(NUM_PRIMITIVES):
        frac = torch.rand(b - a, NUM_SAMPLES, 3)
        points = lo[a:b, None, :] + frac * (hi - lo)[a:b, None, :]
        sampled.append(
            torch.stack(_sample_extents(points, ro[a:b], sp[a:b], pbx[a:b], pby[a:b]))
        )
    return _report("boxes", extents, torch.cat(sampled, -1))


def _report(name, extents, sampled):
    # One primitive per frame, so the primitive axis is a singleton.
    x0, x1, y0, y1, bounded = (value[:, 0] for value in extents)
    lo_x, hi_x, lo_y, hi_y = sampled
    tested = bounded & torch.isfinite(lo_x)
    escapes = tested & (
        (lo_x < x0 - PAD) | (hi_x > x1 + PAD) | (lo_y < y0 - PAD) | (hi_y > y1 + PAD)
    )
    print(
        f"{name}: {int(bounded.numel())} primitives, "
        f"{int(tested.sum())} bounded with visible samples, "
        f"{int((~bounded).sum())} reported unbounded, "
        f"{int(escapes.sum())} escaping"
    )
    return int(escapes.sum())


if __name__ == "__main__":
    bad = check_triangles() + check_boxes()
    print("FAIL" if bad else "PASS")
    sys.exit(1 if bad else 0)
