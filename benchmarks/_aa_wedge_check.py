"""Standalone validation of the oriented two-wall (wedge) coverage model.

Rebuilt and extended for DESIGN_analytic_aa_v2.md ss5 (the ss21.6 original was
lost to a truncated write). Validates, without a render:

  1. ``_two_halfplane_area`` against float64 polygon clipping over random wall
     pairs (the primitive the wedge composes; ss21.6 validated it to 0.0115).
  2. THE SELECTION RULE: coverage of a pixel near a polygon CORNER -- convex
     and CONCAVE, every orientation -- computed from the two nearest walls with
     STORED inward normals via the parity-agreement +/- turn rule (the exact
     logic ``_bez_pixel_hit``'s wedge branch applies), against brute-force
     point sampling of the true polygon. This is the configuration the ss21.6
     handedness calibration failed on (plain square 0.1093 -> 0.2467).
  3. SIGMA: ``_circuit_edge_inward_signs`` on synthetic contours -- squares of
     both windings, a ring with a hole (both winding conventions), sub-pixel
     stems, and a self-intersecting bowtie -- asserting the definitional
     invariant (the drawn side is the one the sign points to, verified by
     high-resolution parity) and the hole property (signs point out of holes
     regardless of winding).

Run: .venv/Scripts/python.exe benchmarks/_aa_wedge_check.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
import taichi as ti  # noqa: E402
import torch  # noqa: E402

ti.init(arch=ti.cpu, default_fp=ti.f32)

# Benchmarks must never be measured inside a warm daemon: it keeps adaptive
# renderer state (the memory model's batch-size fit) across runs, so one
# benchmark would be timed against whatever ran before it.
os.environ.setdefault("ALGAN_USE_DAEMON", "0")

from algan.rendering.raytracing.primitives import (  # noqa: E402
    _circuit_edge_inward_signs,
)
from algan.rendering.raytracing.raster_taichi import (  # noqa: E402
    _halfplane_clip_area,
    _two_halfplane_area,
)

FAILS = []


def check(ok, msg):
    print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
    if not ok:
        FAILS.append(msg)


@ti.kernel
def wedge_areas(h: ti.types.ndarray(), out: ti.types.ndarray()):
    for i in range(h.shape[0]):
        a1 = _halfplane_clip_area(h[i, 0], h[i, 1], h[i, 2])
        a2 = _halfplane_clip_area(h[i, 3], h[i, 4], h[i, 5])
        out[i, 0] = a1
        out[i, 1] = a2
        out[i, 2] = _two_halfplane_area(
            h[i, 0], h[i, 1], h[i, 2], h[i, 3], h[i, 4], h[i, 5], a1, a2
        )


def brute_halfplanes(n1, d1, n2, d2, res=192):
    """Fraction of the pixel square inside both half-planes, by sampling."""
    t = (np.arange(res) + 0.5) / res - 0.5
    x, y = np.meshgrid(t, t)
    return float(
        ((n1[0] * x + n1[1] * y + d1 >= 0) & (n2[0] * x + n2[1] * y + d2 >= 0)).mean()
    )


def part1():
    print("1. _two_halfplane_area vs float64 sampling")
    rng = np.random.default_rng(7)
    worst = 0.0
    rows = []
    for _ in range(4000):
        a1, a2 = rng.uniform(0, 2 * np.pi, 2)
        d1, d2 = rng.uniform(-0.9, 0.9, 2)
        rows.append((np.cos(a1), np.sin(a1), d1, np.cos(a2), np.sin(a2), d2))
    h = np.asarray(rows, np.float32)
    out = np.zeros((len(rows), 3), np.float32)
    wedge_areas(h, out)
    for i, r in enumerate(rows):
        ref = brute_halfplanes(r[0:2], r[2], r[3:5], r[5])
        worst = max(worst, abs(out[i, 2] - ref))
    check(worst < 0.06, f"random wall pairs, worst |err| {worst:.4f}")


def _corner_polygon(apex, d_in, d_out):
    """A big simple polygon whose only near-pixel feature is one corner.

    The incoming edge arrives at ``apex`` along ``d_in``, the outgoing one
    leaves along ``d_out``; the contour closes through a far CCW arc, which
    fixes which side is interior. Whether the corner comes out convex or
    reflex follows from the wall angles -- the caller classifies it by a
    point test, it does not choose.
    """
    L = 50.0
    p_prev = apex - d_in * L
    p_next = apex + d_out * L
    t0 = np.arctan2(d_out[1], d_out[0])
    t1 = np.arctan2(-d_in[1], -d_in[0])
    while t1 <= t0:
        t1 += 2 * np.pi
    arc = [
        apex + 4 * L * np.array([np.cos(t), np.sin(t)])
        for t in np.linspace(t0, t1, 10)[1:-1]
    ]
    return np.array([p_prev, apex, p_next, *arc])


def _point_in_poly(pts, poly):
    x, y = pts[..., 0], pts[..., 1]
    inside = np.zeros(x.shape, bool)
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        straddle = (y0 > y) != (y1 > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xc = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
        inside ^= straddle & (xc > x)
    return inside


def wedge_model(n1, cp1, d1, n2, cp2, d2, centre_inside, ti_area):
    """The kernel's selection rule, mirrored in python (query at the origin).

    CONVEX vs REFLEX is a property of WHICH RAY of its line each wall segment
    occupies: the boundary of the intersection region is the pair of rays on
    which the OTHER constraint is satisfied, the boundary of the union the
    pair on which it is violated. So the side test ``s1 = n2 . (q1 - apex)``
    (``q1`` the closest point on wall 1, ``apex`` the line intersection)
    reads convex (>0) or reflex (<0) directly -- and when the closest point
    IS the apex (the query sits past the corner), the clamped-endpoint sign
    ``cp . d`` says which way the segment leaves it instead. Parity agreement
    remains the arbiter when both side tests are degenerate, and near-parallel
    walls are a strip: intersection unconditionally (ss21.6).

    Returns (coverage, tag) with tag in {"inter", "union", "single"}.
    """
    sd1 = -np.dot(n1, cp1)
    sd2 = -np.dot(n2, cp2)
    a1, a2, inter = ti_area(n1, sd1, n2, sd2)
    uni = min(max(a1 + a2 - inter, 0.0), 1.0)
    cross = n1[0] * n2[1] - n1[1] * n2[0]
    if abs(cross) < 0.2:
        return inter, "inter"
    b1 = np.dot(n1, cp1)
    b2 = np.dot(n2, cp2)
    apex = np.array(
        [(b1 * n2[1] - b2 * n1[1]) / cross, (n1[0] * b2 - n2[0] * b1) / cross]
    )
    scale = abs(cp1).sum() + abs(cp2).sum() + 1e-6

    def side(cp, d, n_other):
        r = cp - apex
        if np.linalg.norm(r) > 1e-4 * scale:
            return np.dot(n_other, r)
        t = np.dot(cp, d)
        if abs(t) <= 1e-9 * scale:
            return 0.0
        return np.sign(t) * np.dot(n_other, d)

    s1 = side(cp1, d1, n2)
    s2 = side(cp2, d2, n1)
    if s1 > 0 and s2 > 0:
        return inter, "inter"
    if s1 < 0 and s2 < 0:
        return uni, "union"
    in_i = (sd1 > 0) and (sd2 > 0)
    in_u = (sd1 > 0) or (sd2 > 0)
    if in_i == centre_inside and in_u != centre_inside:
        return inter, "inter"
    if in_u == centre_inside and in_i != centre_inside:
        return uni, "union"
    return inter, "inter"


def part2():
    print("2. corner coverage: parity-agreement wedge vs true polygon")
    rng = np.random.default_rng(11)
    h_one = np.zeros((1, 6), np.float32)
    out_one = np.zeros((1, 3), np.float32)

    def ti_area(n1, sd1, n2, sd2):
        h_one[0] = (n1[0], n1[1], sd1, n2[0], n2[1], sd2)
        wedge_areas(h_one, out_one)
        return float(out_one[0, 0]), float(out_one[0, 1]), float(out_one[0, 2])

    res = 128
    t = (np.arange(res) + 0.5) / res - 0.5
    gx, gy = np.meshgrid(t, t)
    grid = np.stack((gx, gy), -1)

    worst = {"convex": 0.0, "reflex": 0.0}
    fell_back = 0
    n_class = {"convex": 0, "reflex": 0}
    for trial in range(600):
        ang = rng.uniform(0, 2 * np.pi)
        # Turn angle between the wall directions, kept away from parallel and
        # antiparallel (the nd/near-parallel gates take those in production).
        turn = np.deg2rad(rng.uniform(30, 150)) * (1 if trial % 2 else -1)
        d_in = np.array([np.cos(ang), np.sin(ang)])
        rot = np.array([[np.cos(turn), -np.sin(turn)], [np.sin(turn), np.cos(turn)]])
        d_out = rot @ d_in
        apex = rng.uniform(-0.45, 0.45, 2)
        poly = _corner_polygon(apex, d_in, d_out)
        inside = _point_in_poly(grid, poly)
        true_cov = float(inside.mean())
        # Classify by the point just inside the wedge between the wall rays.
        w = -d_in + d_out
        w = w / np.linalg.norm(w)
        reflex = not bool(_point_in_poly((apex + 0.05 * w)[None], poly)[0])

        # The two walls as the kernel sees them: contour directions, stored
        # inward signs from a probe just off each wall's midpoint.
        walls = []
        for p0, p1 in ((poly[0], poly[1]), (poly[1], poly[2])):
            d = p1 - p0
            d = d / np.linalg.norm(d)
            left = np.array([-d[1], d[0]])
            mid = (p0 + p1) / 2
            sg = 0.0
            for eps in (0.05, 0.025, 0.0125):
                pl = _point_in_poly((mid + eps * left)[None], poly)[0]
                pr = _point_in_poly((mid - eps * left)[None], poly)[0]
                if pl != pr:
                    sg = 1.0 if pl else -1.0
                    break
            n = left * sg
            # Closest point on the segment from the pixel centre (the query).
            seg = p1 - p0
            tt = np.clip(np.dot(-p0, seg) / np.dot(seg, seg), 0, 1)
            cp = p0 + tt * seg  # centre is the origin
            sd = -np.dot(n, cp)
            walls.append((n, sd, d, cp, sg))
        (n1, sd1, d1, cp1, sg1), (n2, sd2, d2, cp2, sg2) = walls
        if sg1 == 0.0 or sg2 == 0.0:
            fell_back += 1
            continue
        centre_inside = bool(_point_in_poly(np.zeros((1, 2)), poly)[0])
        cov, tag = wedge_model(n1, cp1, d1, n2, cp2, d2, centre_inside, ti_area)
        key = "reflex" if reflex else "convex"
        n_class[key] += 1
        err = abs(cov - true_cov)
        if err > worst[key]:
            worst[key] = err
            if err > 0.2:
                print(
                    f"     worst[{key}] tr{trial}: cov {cov:.3f} ({tag}) "
                    f"true {true_cov:.3f} apex {apex.round(3)} "
                    f"turn {np.rad2deg(turn):.0f} sd ({sd1:.3f},{sd2:.3f}) "
                    f"sg ({sg1},{sg2}) cin {centre_inside}"
                )
    check(worst["convex"] < 0.08, f"convex corners, worst |err| {worst['convex']:.4f}")
    check(worst["reflex"] < 0.08, f"reflex corners, worst |err| {worst['reflex']:.4f}")
    print(
        f"     ({n_class['convex']} convex / {n_class['reflex']} reflex,"
        f" fell back on {fell_back})"
    )


def _parity_torch(edges, q):
    x0, y0, x1, y1 = edges[:, 0], edges[:, 1], edges[:, 2], edges[:, 3]
    qx, qy = q
    straddle = (y0 > qy) != (y1 > qy)
    denom = np.where(y1 - y0 == 0, 1.0, y1 - y0)
    xc = x0 + (qy - y0) * (x1 - x0) / denom
    return int((straddle & (xc > qx)).sum()) % 2 == 1


def _sigma_case(name, loops, expect_holes=False):
    """Build [1, V, 5] edges from closed loops; validate sigma's invariant."""
    rows = []
    circ = []
    for pts in loops:
        pts = np.asarray(pts, np.float64)
        for i in range(len(pts)):
            p0, p1 = pts[i], pts[(i + 1) % len(pts)]
            rows.append((p0[0], p0[1], p1[0], p1[1], 1.0))
            circ.append(0)
    edges = torch.tensor(np.asarray(rows, np.float32)).unsqueeze(0)
    vert_circuit = torch.tensor(circ, dtype=torch.long)
    sigma = _circuit_edge_inward_signs(edges, vert_circuit)[0].numpy()
    e = edges[0].numpy().astype(np.float64)
    bad = 0
    zero = 0
    for i in range(len(e)):
        d = e[i, 2:4] - e[i, 0:2]
        ln = np.linalg.norm(d)
        if ln < 1e-12:
            continue
        left = np.array([-d[1], d[0]]) / ln
        mid = (e[i, 0:2] + e[i, 2:4]) / 2
        if sigma[i] == 0:
            zero += 1
            continue
        # Reference parity at a tiny offset in float64.
        got = None
        for eps in (0.02 * ln, 0.005 * ln, 0.001 * ln):
            pl = _parity_torch(e, mid + eps * left)
            pr = _parity_torch(e, mid - eps * left)
            if pl != pr:
                got = 1.0 if pl else -1.0
                break
        if got is not None and got != sigma[i]:
            bad += 1
    check(bad == 0, f"sigma[{name}]: {bad} wrong of {len(e)} ({zero} unresolved)")


def part3():
    print("3. flatten-time inward signs on synthetic contours")
    sq = [(0, 0), (2, 0), (2, 2), (0, 2)]
    _sigma_case("square ccw", [sq])
    _sigma_case("square cw", [sq[::-1]])
    ring_o = [(0, 0), (4, 0), (4, 4), (0, 4)]
    ring_i = [(1, 1), (3, 1), (3, 3), (1, 3)]
    _sigma_case("ring hole same-winding", [ring_o, ring_i])
    _sigma_case("ring hole opposite-winding", [ring_o, ring_i[::-1]])
    stem = [(0, 0), (0.02, 0), (0.02, 3), (0, 3)]
    _sigma_case("sub-pixel stem", [stem])
    bow = [(0, 0), (2, 2), (2, 0), (0, 2)]
    _sigma_case("self-intersecting bowtie", [bow])


def main():
    part1()
    part2()
    part3()
    print()
    if FAILS:
        print(f"{len(FAILS)} FAILURES")
        sys.exit(1)
    print("all wedge checks passed")


if __name__ == "__main__":
    main()
