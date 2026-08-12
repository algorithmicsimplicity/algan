"""Analytic AA: the exact fixed-point sample partition, checked in isolation.

The triangle coverage path stands or falls on one property: two triangles that
share an edge must partition the sub-pixel samples along it -- every sample
claimed by exactly one of them. Claimed by both dilates silhouettes into a
halo; claimed by neither leaves a dark notch on every internal mesh edge.

This replicates ``_ss_pixel``'s integer test in plain Python (where the
arithmetic is unambiguously exact) and checks the property directly on randomly
generated shared edges. It runs in a second and needs no GPU, so it isolates
"is the rule right" from "is the renderer right" -- worth having, because when
the two disagreed the cause was a Taichi type collision that silently ran the
whole computation in float32, not the rule.

Keep SHIFT and SAMPLES in step with raster_taichi's _AA_FIXED_SHIFT and
_AA_SAMPLES.

Run: .venv/Scripts/python.exe benchmarks/_analytic_aa_fillrule_check.py
"""

from __future__ import annotations

import random
import sys

SHIFT = 12
SCALE = 1 << SHIFT
HALF = 1 << (SHIFT - 1)
SAMPLES = [
    (x * SCALE // 16, y * SCALE // 16)
    for x, y in ((1, -3), (-1, 3), (5, 1), (-3, -5), (-5, 5), (-7, -1), (3, 7), (7, -7))
]

FLOAT_MISMATCH = [0]


def snap(v):
    return int(round(v * SCALE))


def mask(tri, px, py):
    """The kernel's exact test: returns (sample mask, orientation)."""
    (sx0, sy0), (sx1, sy1), (sx2, sy2) = tri
    fx0, fy0 = snap(sx0), snap(sy0)
    fx1, fy1 = snap(sx1), snap(sy1)
    fx2, fy2 = snap(sx2), snap(sy2)
    qx = (px << SHIFT) + HALF
    qy = (py << SHIFT) + HALF
    ex0, ey0 = fx2 - fx1, fy2 - fy1
    ex1, ey1 = fx0 - fx2, fy0 - fy2
    ex2, ey2 = fx1 - fx0, fy1 - fy0
    r0 = ex0 * (qy - fy1) - ey0 * (qx - fx1)
    r1 = ex1 * (qy - fy2) - ey1 * (qx - fx2)
    r2 = ex2 * (qy - fy0) - ey2 * (qx - fx0)
    # Orientation from the EXACT integer sum (twice the lattice signed area).
    # The float sum is the same quantity built from large cancelling products;
    # two neighbours disagreeing on its SIGN is what breaks the partition.
    oi = 1 if (r0 + r1 + r2) >= 0 else -1
    e0 = (sx2 - sx1) * (py + 0.5 - sy1) - (sy2 - sy1) * (px + 0.5 - sx1)
    e1 = (sx0 - sx2) * (py + 0.5 - sy2) - (sy0 - sy2) * (px + 0.5 - sx2)
    e2 = (sx1 - sx0) * (py + 0.5 - sy0) - (sy1 - sy0) * (px + 0.5 - sx0)
    if (1 if (e0 + e1 + e2) >= 0 else -1) != oi:
        FLOAT_MISMATCH[0] += 1
    ec0, ec1, ec2 = oi * r0, oi * r1, oi * r2
    gx0, gy0 = oi * ex0, oi * ey0
    gx1, gy1 = oi * ex1, oi * ey1
    gx2, gy2 = oi * ex2, oi * ey2

    def top_left(gx, gy):
        return 1 if (gy > 0 or (gy == 0 and gx < 0)) else 0

    t0, t1, t2 = (top_left(gx0, gy0), top_left(gx1, gy1), top_left(gx2, gy2))
    m = 0
    for k, (ox, oy) in enumerate(SAMPLES):
        q0 = ec0 + gx0 * oy - gy0 * ox + t0
        q1 = ec1 + gx1 * oy - gy1 * ox + t1
        q2 = ec2 + gx2 * oy - gy2 * ox + t2
        if min(q0, q1, q2) > 0:
            m |= 1 << k
    return m, oi


def inside_by(quad, px, py, margin):
    """Is the pixel centre at least ``margin`` pixels inside the convex quad?"""
    cx, cy = px + 0.5, py + 0.5
    n = len(quad)
    # Orientation of the quad, so the test works either way round.
    area2 = sum(
        quad[i][0] * quad[(i + 1) % n][1] - quad[(i + 1) % n][0] * quad[i][1]
        for i in range(n)
    )
    sgn = 1.0 if area2 >= 0 else -1.0
    for i in range(n):
        ax, ay = quad[i]
        bx, by = quad[(i + 1) % n]
        ex, ey = bx - ax, by - ay
        length = (ex * ex + ey * ey) ** 0.5
        if length < 1e-9:
            return False
        if sgn * (ex * (cy - ay) - ey * (cx - ax)) / length < margin:
            return False
    return True


def main():
    random.seed(7)
    both = neither = trials = flipped = 0
    for _ in range(4000):
        # A quad split along the diagonal V0-V2: the canonical shared edge.
        # Shear stays small so the quad is convex and the two triangles wind
        # the same way, which is the case the rule is for (a genuine winding
        # flip means the surface folds, and the two are not neighbours in the
        # tiling sense).
        ox_, oy_ = random.uniform(0, 4), random.uniform(0, 4)
        w, h = random.uniform(1.5, 6), random.uniform(1.5, 6)
        sh = random.uniform(-0.4, 0.4)
        v0 = (ox_, oy_)
        v1 = (ox_ + w, oy_ + sh)
        v2 = (ox_ + w + sh, oy_ + h)
        v3 = (ox_ + sh, oy_ + h)
        triA, triB = (v0, v1, v2), (v0, v2, v3)
        quad = (v0, v1, v2, v3)
        for px in range(8):
            for py in range(8):
                ma, oa = mask(triA, px, py)
                mb, ob = mask(triB, px, py)
                trials += 1
                if oa != ob:
                    flipped += 1
                    continue
                if ma & mb:
                    both += 1
                # The partition is only owed where the pixel lies WHOLLY
                # inside the quad. Elsewhere samples outside it are correctly
                # claimed by neither triangle. One pixel of margin from every
                # quad edge puts all eight samples strictly inside.
                if inside_by(quad, px, py, 1.0) and (ma | mb) != 0xFF:
                    neither += 1

    print(f"pixel tests:                         {trials}")
    print(f"  winding-flipped pairs (skipped):   {flipped}")
    print(f"  samples claimed by BOTH:           {both}")
    print(f"  samples claimed by NEITHER:        {neither}")
    print(f"float-vs-exact orientation mismatch: {FLOAT_MISMATCH[0]} of {trials * 2}")
    ok = both == 0 and neither == 0
    print("\nFILL_RULE_OK:", ok)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
