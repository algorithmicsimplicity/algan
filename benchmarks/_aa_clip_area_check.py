"""Unit checks for the exact pixel-coverage primitives.

``_pixel_clip_area`` -- the area of (triangle n pixel) -- has three properties,
each of which the continuous product-of-edge-distances form it replaces does NOT
have (DESIGN_analytic_aa.md ss15.3):

  1. AGREEMENT with a brute-force reference (dense point-in-triangle sampling of
     the pixel square), on random triangles of every scale from far larger than
     the pixel down to slivers thinner than 1/1000 of it.
  2. ZERO outside. A triangle disjoint from the pixel square must contribute
     nothing; the product form spreads coverage half a pixel past the geometry.
  3. SUMS OVER A TILING. A polygon fan-triangulated into pieces must have its
     piece areas sum to the whole's area, however the pieces are cut -- this is
     the property that makes a silhouette rim of foreshortened triangles sum to
     the band they cover instead of to a halo.

``_halfplane_clip_area`` -- the area of (half-plane n pixel), the one-crossing-
edge case in closed form -- adds four more:

  4. AGREEMENT with brute force over every orientation and offset.
  5. AGREEMENT WITH ``_pixel_clip_area`` on a triangle wide enough that only one
     of its edges reaches the pixel. The two are dispatched between per fragment,
     so they have to meet where they overlap.
  6. COLLAPSE to ``clamp(d + 0.5, 0, 1)``, exactly, for an axis-aligned
     boundary -- i.e. the box filter the circuit path currently applies at every
     orientation is this formula's b == 0 special case. The same sweep reports
     how wrong that filter is elsewhere.
  7. COMPLEMENT: ``A(n, d) + A(-n, -d) == 1``. The two sides of one boundary sum
     to the whole pixel, which is the seam property in its smallest form.

Run: .venv/Scripts/python.exe benchmarks/_aa_clip_area_check.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
import taichi as ti  # noqa: E402

ti.init(arch=ti.cpu)

# Benchmarks must never be measured inside a warm daemon: it keeps adaptive
# renderer state (the memory model's batch-size fit) across runs, so one
# benchmark would be timed against whatever ran before it.
os.environ.setdefault("ALGAN_USE_DAEMON", "0")

from algan.rendering.raytracing.raster_taichi import (  # noqa: E402
    _halfplane_clip_area,
    _pixel_clip_area,
)


@ti.kernel
def clip_areas(v: ti.types.ndarray(), out: ti.types.ndarray()):
    for i in range(v.shape[0]):
        out[i] = _pixel_clip_area(
            ti.math.vec3(v[i, 0], v[i, 2], v[i, 4]),
            ti.math.vec3(v[i, 1], v[i, 3], v[i, 5]),
        )


@ti.kernel
def halfplane_areas(h: ti.types.ndarray(), out: ti.types.ndarray()):
    for i in range(h.shape[0]):
        out[i] = _halfplane_clip_area(h[i, 0], h[i, 1], h[i, 2])


def _unit(nx, ny):
    """The routine reads only the normal's DIRECTION; ``d`` is already a distance."""
    length = np.hypot(float(nx), float(ny))
    if length == 0.0:
        return 0.0, 0.0
    return float(nx) / length, float(ny) / length


def ref_halfplane(nx, ny, d):
    """Reference area by float64 polygon clipping -- Sutherland-Hodgman + shoelace.

    Independent of the routine under test, which is a case analysis over branch
    regions rather than a clip, and exact to float64 rather than to a sampling
    grid. ``brute_halfplane`` keeps the cruder count-the-points arm on a subset.
    """
    ux, uy = _unit(nx, ny)
    poly = [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]
    out = []
    for i, (ax, ay) in enumerate(poly):
        bx, by = poly[(i + 1) % len(poly)]
        fa = ux * ax + uy * ay + d
        fb = ux * bx + uy * by + d
        if fa >= 0.0:
            out.append((ax, ay))
        if (fa >= 0.0) != (fb >= 0.0):
            s = fa / (fa - fb)
            out.append((ax + s * (bx - ax), ay + s * (by - ay)))
    acc = 0.0
    for i, (ax, ay) in enumerate(out):
        bx, by = out[(i + 1) % len(out)]
        acc += ax * by - bx * ay
    return abs(acc) * 0.5


def brute_halfplane(nx, ny, d, n=1024):
    """Reference area by dense uniform sampling of the pixel square."""
    ux, uy = _unit(nx, ny)
    g = (np.arange(n) + 0.5) / n - 0.5
    qx, qy = np.meshgrid(g, g)
    return float(((ux * qx + uy * qy + d) >= 0.0).sum()) / (n * n)


def brute(tri, n=2048):
    """Reference area by dense uniform sampling of the pixel square."""
    g = (np.arange(n) + 0.5) / n - 0.5
    qx, qy = np.meshgrid(g, g)
    q = np.stack([qx.ravel(), qy.ravel()], 1)
    inside = np.ones(q.shape[0], dtype=bool)
    sgn = 0.0
    for k in range(3):
        a, b = tri[k], tri[(k + 1) % 3]
        e = (b[0] - a[0]) * (q[:, 1] - a[1]) - (b[1] - a[1]) * (q[:, 0] - a[0])
        if k == 0:
            area2 = (tri[1][0] - tri[0][0]) * (tri[2][1] - tri[0][1]) - (
                tri[1][1] - tri[0][1]
            ) * (tri[2][0] - tri[0][0])
            sgn = 1.0 if area2 >= 0 else -1.0
        inside &= (sgn * e) >= 0
    return inside.sum() / q.shape[0]


def main():
    rng = np.random.default_rng(7)
    ok = True

    # 1. Agreement with the brute-force reference across scales.
    worst = 0.0
    worst_case = None
    cases = []
    for scale in (0.05, 0.2, 1.0, 5.0, 50.0):
        for _ in range(60):
            c = rng.uniform(-1.0, 1.0, 2)
            t = c[None, :] + rng.uniform(-scale, scale, (3, 2))
            cases.append(t)
    # Deliberate slivers: three near-collinear points.
    for _ in range(120):
        a = rng.uniform(-2.0, 2.0, 2)
        d = rng.normal(size=2)
        d /= np.linalg.norm(d)
        nrm = np.array([-d[1], d[0]])
        w = 10.0 ** rng.uniform(-4, -1)
        t = np.stack(
            [a, a + d * rng.uniform(0.5, 6.0), a + d * rng.uniform(0.0, 6.0) + nrm * w]
        )
        cases.append(t)
    v = np.array(
        [[t[0, 0], t[0, 1], t[1, 0], t[1, 1], t[2, 0], t[2, 1]] for t in cases],
        dtype=np.float32,
    )
    out = np.zeros(len(cases), dtype=np.float32)
    clip_areas(v, out)
    for t, got in zip(cases, out):
        ref = brute(t)
        # The reference itself quantizes at 1/n per boundary row, so 3e-3 is the
        # floor of what this comparison can resolve.
        if abs(got - ref) > 3e-3 and abs(got - ref) > worst:
            worst = abs(got - ref)
            worst_case = (t, got, ref)
    if worst_case is not None:
        print(
            f"FAIL agreement: max|d| {worst:.5f} on {worst_case[0].tolist()} "
            f"got {worst_case[1]:.5f} ref {worst_case[2]:.5f}"
        )
        ok = False
    else:
        print(f"agreement: {len(cases)} triangles, all within 3e-3 of brute force")

    # 2. Zero outside. Triangles pushed clear of the square on one side.
    far = []
    for _ in range(200):
        t = rng.uniform(-3.0, 3.0, (3, 2))
        ax = rng.integers(0, 2)
        sgn = rng.choice([-1.0, 1.0])
        # Move the whole triangle to one side of the square with a margin.
        t[:, ax] = sgn * (np.abs(t[:, ax]) + 0.5001)
        far.append(t)
    v = np.array(
        [[t[0, 0], t[0, 1], t[1, 0], t[1, 1], t[2, 0], t[2, 1]] for t in far],
        dtype=np.float32,
    )
    out = np.zeros(len(far), dtype=np.float32)
    clip_areas(v, out)
    if out.max() > 1e-6:
        print(
            f"FAIL outside: max area {out.max():.3e} for a triangle disjoint "
            f"from the pixel"
        )
        ok = False
    else:
        print(f"outside: {len(far)} disjoint triangles, all exactly zero")

    # 3. Sums over a tiling. Fan-triangulate a random convex polygon and check
    #    the pieces sum to the whole, including when the fan produces slivers.
    worst_sum = 0.0
    for _ in range(300):
        n = int(rng.integers(3, 8))
        # A convex polygon, so that fanning it really is a tiling: points on a
        # circle in angle order, pushed through a random affine map (which
        # preserves convexity) to get every aspect ratio including near-slivers.
        ang = np.sort(rng.uniform(0, 2 * np.pi, n))
        unit = np.stack([np.cos(ang), np.sin(ang)], 1)
        m = rng.normal(size=(2, 2)) * 10.0 ** rng.uniform(-1.5, 0.5)
        if np.linalg.det(m) < 0:  # keep the winding CCW
            m = m[::-1]
        poly = rng.uniform(-1.0, 1.0, 2)[None, :] + unit @ m
        pieces = [np.stack([poly[0], poly[k], poly[k + 1]]) for k in range(1, n - 1)]
        v = np.array(
            [[t[0, 0], t[0, 1], t[1, 0], t[1, 1], t[2, 0], t[2, 1]] for t in pieces],
            dtype=np.float32,
        )
        out = np.zeros(len(pieces), dtype=np.float32)
        clip_areas(v, out)
        # Reference: the polygon's own clipped area, by dense sampling.
        g = (np.arange(1024) + 0.5) / 1024 - 0.5
        qx, qy = np.meshgrid(g, g)
        q = np.stack([qx.ravel(), qy.ravel()], 1)
        inside = np.ones(q.shape[0], dtype=bool)
        for k in range(n):
            a, b = poly[k], poly[(k + 1) % n]
            e = (b[0] - a[0]) * (q[:, 1] - a[1]) - (b[1] - a[1]) * (q[:, 0] - a[0])
            inside &= e >= 0
        ref = inside.sum() / q.shape[0]
        worst_sum = max(worst_sum, abs(float(out.sum()) - ref))
    if worst_sum > 5e-3:
        print(f"FAIL tiling: piece areas sum to within {worst_sum:.4f} of the whole")
        ok = False
    else:
        print(
            f"tiling: 300 fans, piece sums within {worst_sum:.4f} of the "
            f"whole polygon's clipped area"
        )

    # 4. Half-plane: agreement with brute force over orientation and offset.
    #    Normals are deliberately NOT unit length -- the routine reads only
    #    their direction, and a caller with a closest-point vector or an edge
    #    normal has no reason to have normalized it.
    hp = []
    for _ in range(400):
        ang = rng.uniform(0.0, 2 * np.pi)
        scale = 10.0 ** rng.uniform(-3, 3)
        hp.append((np.cos(ang) * scale, np.sin(ang) * scale, rng.uniform(-0.8, 0.8)))
    # Plus a systematic sweep, which is where the branch boundaries live: the
    # trapezoid/corner switch at |d| = (a-b)/2 and the clears-the-square switch
    # at |d| = (a+b)/2 both move with the angle.
    for i in range(37):
        ang = i * (np.pi / 72.0)
        for d in np.linspace(-0.75, 0.75, 41):
            hp.append((np.cos(ang), np.sin(ang), float(d)))
    h = np.array(hp, dtype=np.float32)
    out = np.zeros(len(hp), dtype=np.float32)
    halfplane_areas(h, out)
    worst_hp = 0.0
    worst_hp_case = None
    for (nx, ny, d), got in zip(hp, out):
        ref = ref_halfplane(nx, ny, d)
        if abs(got - ref) > worst_hp:
            worst_hp = abs(got - ref)
            worst_hp_case = (nx, ny, d, got, ref)
    if worst_hp > 3e-6:
        nx, ny, d, got, ref = worst_hp_case
        print(
            f"FAIL halfplane: max|d| {worst_hp:.3e} at n=({nx:.4f}, {ny:.4f}) "
            f"d={d:.4f} got {got:.6f} ref {ref:.6f}"
        )
        ok = False
    else:
        print(
            f"halfplane: {len(hp)} (normal, offset) pairs, max deviation from "
            f"an f64 polygon clip {worst_hp:.2e}"
        )

    #    ...and the same against counting points, on a subset. Slower per case by
    #    four orders of magnitude, but it shares no reasoning with either the
    #    routine or the clip above.
    worst_bf = 0.0
    for i in range(0, len(hp), max(1, len(hp) // 48)):
        nx, ny, d = hp[i]
        worst_bf = max(worst_bf, abs(float(out[i]) - brute_halfplane(nx, ny, d)))
    if worst_bf > 3e-3:
        print(f"FAIL halfplane brute force: max|d| {worst_bf:.5f}")
        ok = False
    else:
        print(f"halfplane: 48 of them within {worst_bf:.5f} of brute force too")

    # 5. Half-plane against _pixel_clip_area. Build a triangle whose base lies
    #    on the boundary and whose other two edges are far enough out that only
    #    the base reaches the pixel: the closed form and the general clip are
    #    dispatched between per fragment, so they must agree where they overlap.
    k = 8.0
    tris = []
    ds = []
    for i in range(24):
        ang = i * (np.pi / 12.0)
        n = np.array([np.cos(ang), np.sin(ang)])
        t = np.array([-n[1], n[0]])
        for d in np.linspace(-0.8, 0.8, 33):
            p0 = -d * n
            tris.append(np.stack([p0 - k * t, p0 + k * t, p0 + k * n]))
            ds.append((n[0], n[1], float(d)))
    v = np.array(
        [[t[0, 0], t[0, 1], t[1, 0], t[1, 1], t[2, 0], t[2, 1]] for t in tris],
        dtype=np.float32,
    )
    got_tri = np.zeros(len(tris), dtype=np.float32)
    clip_areas(v, got_tri)
    got_hp = np.zeros(len(ds), dtype=np.float32)
    halfplane_areas(np.array(ds, dtype=np.float32), got_hp)
    worst_pair = float(np.abs(got_tri - got_hp).max())
    if worst_pair > 3e-4:
        i = int(np.abs(got_tri - got_hp).argmax())
        print(
            f"FAIL halfplane/clip: max|d| {worst_pair:.6f} at n=({ds[i][0]:.4f},"
            f" {ds[i][1]:.4f}) d={ds[i][2]:.4f} clip {got_tri[i]:.6f} "
            f"halfplane {got_hp[i]:.6f}"
        )
        ok = False
    else:
        print(
            f"halfplane/clip: {len(tris)} single-edge triangles agree with the "
            f"closed form to {worst_pair:.2e}"
        )

    # 6. Axis-aligned collapse to the box filter -- and how far the box filter is
    #    from exact everywhere else, which is the defect the circuit path has.
    axis = []
    for d in np.linspace(-0.9, 0.9, 181):
        axis.append((1.0, 0.0, float(d)))
        axis.append((0.0, 1.0, float(d)))
        axis.append((-1.0, 0.0, float(d)))
    got_axis = np.zeros(len(axis), dtype=np.float32)
    halfplane_areas(np.array(axis, dtype=np.float32), got_axis)
    box_axis = np.clip(
        np.array([d for _, _, d in axis], dtype=np.float32) + np.float32(0.5), 0, 1
    )
    worst_axis = float(np.abs(got_axis - box_axis).max())
    if worst_axis > 1e-7:
        print(f"FAIL axis-aligned: differs from the box filter by {worst_axis:.2e}")
        ok = False
    else:
        print(f"axis-aligned: identical to clamp(d + 0.5, 0, 1) ({worst_axis:.1e})")

    sweep = []
    for i in range(46):
        ang = i * (np.pi / 90.0)
        for d in np.linspace(-0.75, 0.75, 61):
            sweep.append((np.cos(ang), np.sin(ang), float(d)))
    got_sweep = np.zeros(len(sweep), dtype=np.float32)
    halfplane_areas(np.array(sweep, dtype=np.float32), got_sweep)
    box_sweep = np.clip(
        np.array([d for _, _, d in sweep], dtype=np.float32) + np.float32(0.5), 0, 1
    )
    err = np.abs(got_sweep - box_sweep).reshape(46, 61)
    peak_ang = int(err.max(axis=1).argmax()) * 2.0
    print(
        f"box filter error: peak {err.max():.4f} coverage at {peak_ang:.0f} deg, "
        f"{err[22].max():.4f} at 44 deg, {err[0].max():.1e} axis-aligned"
    )

    # 7. Complement: the two sides of one boundary sum to the whole pixel.
    comp = []
    for i in range(90):
        ang = i * (np.pi / 45.0)
        for d in np.linspace(-0.8, 0.8, 41):
            comp.append((np.cos(ang), np.sin(ang), float(d)))
            comp.append((-np.cos(ang), -np.sin(ang), -float(d)))
    got_comp = np.zeros(len(comp), dtype=np.float32)
    halfplane_areas(np.array(comp, dtype=np.float32), got_comp)
    sums = got_comp.reshape(-1, 2).sum(axis=1)
    worst_comp = float(np.abs(sums - 1.0).max())
    if worst_comp > 1e-6:
        print(f"FAIL complement: A(n, d) + A(-n, -d) off by {worst_comp:.2e}")
        ok = False
    else:
        exact = int((sums == np.float32(1.0)).sum())
        print(
            f"complement: {len(sums)} boundaries sum to 1 within {worst_comp:.1e} "
            f"({exact}/{len(sums)} bit-exactly)"
        )

    print("\nCLIP_AREA_OK:", ok)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
