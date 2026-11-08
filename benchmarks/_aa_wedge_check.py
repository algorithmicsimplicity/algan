"""Standalone validation of the oriented two-wall (wedge) coverage model.

Rebuilt and extended for DESIGN_analytic_aa_v2.md ss5 (the ss21.6 original was
lost to a truncated write). Validates, without a render:

  1. ``_two_halfplane_area`` against float64 polygon clipping over random wall
     pairs (the primitive the wedge composes; ss21.6 validated it to 0.0115).
  2. THE SELECTION RULE: coverage of a pixel near a polygon CORNER -- convex
     and CONCAVE, every orientation -- computed from the two nearest walls with
     STORED inward normals by the rule ``_bez_pixel_hit``'s wedge branch
     applies (a corner's call is its contour's turn), against brute-force
     point sampling of the true polygon. This is the configuration the ss21.6
     handedness calibration failed on (plain square 0.1093 -> 0.2467). 2b
     adds right-angle corners whose pixel centre is on one wall's line, where
     the other wall's side test abstains and parity is a rounding; 2c corners
     whose apex is ON the pixel centre, where both side tests abstain; 2d
     hairpins, within the near-parallel cutoff; 2e two walls that do not meet:
     a stem's two sides, a slit's, and a wall that ends beside the pixel,
     which the wedge must leave out; 2f stored sides gone stale, where another
     contour crosses the walls between their midpoints and the pixel.
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
import torch  # noqa: E402

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
from algan.taichi_compat import ti  # noqa: E402
from algan.rendering.taichi_runtime import init_taichi  # noqa: E402

# Algan's own Taichi configuration, not Taichi's defaults: a check of a shipped
# kernel has to compile it the way the renderer does. A bare ``ti.init`` differs
# in ``advanced_optimization`` alone, and that one is enough to change what a
# kernel computes (see tests/unit_tests/test_taichi_runtime_config.py).
# Idempotent -- the imports above have already brought Taichi up.
init_taichi()

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


#: The box filter's reach from the pixel centre, in pixels: half the diagonal
#: (``raster_taichi._AA_FILTER_RADIUS``).
REACH = 0.7071067811865476

#: Within this many pixels of a wall's line which side the centre is on is a
#: rounding (``raster_taichi._AA_ON_BOUNDARY``).
ON = 0.01


def _inward_sign(n, d):
    """The stored inward sign a wall's normal carries: ``n = sg * left(d)``."""
    return 1.0 if (n[1] * d[0] - n[0] * d[1]) > 0 else -1.0


def _half_plane(cp1, centre_inside, ti_area):
    """The nearest wall's coverage alone, as the kernel's single-wall path has it.

    Oriented by the parity at the centre and the closest-point vector, never
    by a stored side.
    """
    dist = float(np.linalg.norm(cp1))
    if dist < 1e-12:
        return 0.5
    g = -cp1 / dist if centre_inside else cp1 / dist
    sd = dist if centre_inside else -dist
    return ti_area(g, sd, g, sd)[0]


def wedge_model(
    n1, cp1, d1, n2, cp2, d2, centre_inside, ti_area, meet=0, crosses=True, guards=True
):
    """The kernel's selection rule, mirrored in python (query at the origin).

    ``n1``/``n2`` carry the walls' STORED inward sides. ``meet`` is how the
    two segments meet, as ``raster_taichi._walls_meet`` reads it off the
    outline: 1 when wall 1 ends where wall 2 starts, -1 the other way round,
    0 when they share no vertex.

    A pair whose stored sides cannot both hold at the pixel keeps the nearest
    wall's half-plane (``_half_plane``): a corner whose two walls disagree,
    or a nearest wall whose side the parity contradicts where its closest
    point lies inside it.

    A CORNER's two walls are convex where the contour turns toward its inside,
    and the turn is the cross product of the walls' normals taken in the order
    the contour runs them. They are folded together only where the second is
    within the filter's ``REACH``; past it the nearest wall's half-plane
    stands alone.

    Two walls that do not meet are combined only where the second crosses the
    pixel with the centre's foot inside it (``crosses``, mirroring
    ``raster_taichi._wall_in_pixel``); otherwise the nearest wall's half-plane
    stands alone. Antiparallel (|cross| < 0.2) they are a stem, intersected,
    when the second wall lies on the first's inner side, and a slit, united,
    when it lies on its outer side. Otherwise CONVEX vs REFLEX is a property of
    WHICH RAY of its line each wall segment occupies: the boundary of the
    intersection region is the pair of rays on which the OTHER constraint is
    satisfied, the boundary of the union the pair on which it is violated. So
    the side test ``s1 = n2 . (q1 - apex)`` (``q1`` the closest point on wall
    1, ``apex`` the line intersection) reads convex (>0) or reflex (<0)
    directly -- and when the closest point IS the apex (the query sits past
    the corner), the clamped-endpoint sign ``cp . d`` says which way the
    segment leaves it instead. A side test that cannot tell (its closest point
    is the apex and the query is square to its wall there) abstains with 0,
    and the other decides alone; parity agreement is the arbiter only when
    both abstain or they disagree.

    Whatever the call, the wedge's verdict on the centre (inside both walls for
    the intersection, either for the union) must be the parity's, or the
    nearest wall's half-plane stays. A wall within ``ON`` of the centre has no
    say, and the verdict is asked wherever the other settles it.
    ``guards=False`` drops this check and the stale-side ones, trusting the
    stored sides as they are.

    Returns (coverage, tag) with tag in {"inter", "union", "single"}.
    """
    sd1 = -np.dot(n1, cp1)
    sd2 = -np.dot(n2, cp2)
    single = _half_plane(cp1, centre_inside, ti_area)
    sg1 = _inward_sign(n1, d1)
    if guards and meet != 0 and sg1 != _inward_sign(n2, d2):
        return single, "single"
    if (
        guards
        and abs(sd1) > ON
        and abs(np.dot(cp1, d1)) <= 1e-3 * abs(sd1)
        and (sd1 > 0) != centre_inside
    ):
        return single, "single"
    if np.dot(n1, n2) >= 0.9:
        return single, "single"
    a1, a2, inter = ti_area(n1, sd1, n2, sd2)
    uni = min(max(a1 + a2 - inter, 0.0), 1.0)
    cross = n1[0] * n2[1] - n1[1] * n2[0]
    in_i = (sd1 > 0) and (sd2 > 0)
    in_u = (sd1 > 0) or (sd2 > 0)
    if meet != 0:
        if np.linalg.norm(cp2) > REACH:
            return single, "single"
        convex = meet * sg1 * cross >= 0
    elif not crosses:
        return single, "single"
    elif abs(cross) < 0.2:
        convex = np.dot(n1, cp2 - cp1) >= 0
    else:
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
        convex = True
        if s1 <= 0 and s2 <= 0 and s1 + s2 < 0:
            convex = False
        elif not (s1 >= 0 and s2 >= 0 and s1 + s2 > 0):
            if in_u == centre_inside and in_i != centre_inside:
                convex = False
    # The verdict where the walls clear of the centre settle it: one wall the
    # centre is clearly outside of puts it outside an intersection, one it is
    # clearly inside of puts it inside a union.
    outs = (sd1 < -ON, sd2 < -ON)
    ins = (sd1 > ON, sd2 > ON)
    if convex:
        wrong = (centre_inside and any(outs)) or (not centre_inside and all(ins))
    else:
        wrong = (not centre_inside and any(ins)) or (centre_inside and all(outs))
    if guards and wrong:
        return single, "single"
    return (inter, "inter") if convex else (uni, "union")


def _crosses_pixel(p0, p1, h=0.5):
    """``raster_taichi._wall_in_pixel``'s answer, the centre at the origin."""
    d = p1 - p0
    foot = -np.dot(p0, d) / max(np.dot(d, d), 1e-12)
    lo = np.minimum(p0, p1)
    hi = np.maximum(p0, p1)
    return bool(
        0.0 < foot < 1.0
        and (lo <= h).all()
        and (hi >= -h).all()
        and abs(d[0] * p0[1] - d[1] * p0[0]) <= h * (abs(d[0]) + abs(d[1]))
    )


def _area_probe():
    """``ti_area`` for ``wedge_model``: the kernel's clip areas, one pair a call."""
    h_one = np.zeros((1, 6), np.float32)
    out_one = np.zeros((1, 3), np.float32)

    def ti_area(n1, sd1, n2, sd2):
        h_one[0] = (n1[0], n1[1], sd1, n2[0], n2[1], sd2)
        wedge_areas(h_one, out_one)
        return float(out_one[0, 0]), float(out_one[0, 1]), float(out_one[0, 2])

    return ti_area


def _pixel_samples(res=128):
    """Sample points over the pixel square, centred on the origin."""
    t = (np.arange(res) + 0.5) / res - 0.5
    gx, gy = np.meshgrid(t, t)
    return np.stack((gx, gy), -1)


def _corner_walls(poly):
    """The corner's two walls as the kernel sees them, from the pixel centre.

    Contour directions, stored inward signs from a probe just off each wall's
    midpoint, and the closest point to the centre (the origin): one
    ``(n, sd, d, cp, sg)`` per wall, the nearer first as the query ranks them,
    then how they meet as ``raster_taichi._walls_meet`` reads it (1 when the
    first is the incoming wall, which ends where the outgoing one starts).
    """
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
    if np.linalg.norm(walls[1][3]) < np.linalg.norm(walls[0][3]):
        return walls[1], walls[0], -1
    return walls[0], walls[1], 1


def _is_reflex(apex, d_in, d_out, poly):
    """Classify by the point just inside the wedge between the wall rays."""
    w = -d_in + d_out
    w = w / np.linalg.norm(w)
    return not bool(_point_in_poly((apex + 0.05 * w)[None], poly)[0])


def part2():
    print("2. corner coverage: the corner's turn vs true polygon")
    rng = np.random.default_rng(11)
    ti_area = _area_probe()
    grid = _pixel_samples()

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
        reflex = _is_reflex(apex, d_in, d_out, poly)
        (n1, sd1, d1, cp1, sg1), (n2, sd2, d2, cp2, sg2), meet = _corner_walls(poly)
        if sg1 == 0.0 or sg2 == 0.0:
            fell_back += 1
            continue
        centre_inside = bool(_point_in_poly(np.zeros((1, 2)), poly)[0])
        cov, tag = wedge_model(
            n1, cp1, d1, n2, cp2, d2, centre_inside, ti_area, meet=meet
        )
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


def part2b():
    print("2b. right-angle corners with the pixel centre on one wall's line")
    # The other wall's closest point is then the apex, square to the centre, so
    # its side test abstains; the wall the centre is on reads the corner alone.
    # Parity at the centre is a rounding on that line, so each corner runs with
    # the centre counted inside and outside, and the answer may not depend on
    # it. A square whose edge ran through pixel centres drew the pixels beside
    # its corners fully covered where they are half while parity arbitrated.
    # The kernel now calls a corner by its turn; the side tests, which still
    # call two walls that do not meet, are checked on these corners as well.
    ti_area = _area_probe()
    grid = _pixel_samples()
    worst = {}
    for ang in np.linspace(0.0, 2 * np.pi, 24, endpoint=False):
        d_in = np.array([np.cos(ang), np.sin(ang)])
        for turn in (90.0, -90.0):
            c, s = np.cos(np.deg2rad(turn)), np.sin(np.deg2rad(turn))
            d_out = np.array([[c, -s], [s, c]]) @ d_in
            # How far the apex is from the centre, along the wall it is on:
            # 0.75 and 1.75 are the two pixels beside a corner on that line.
            for ahead in (0.25, 0.75, 1.75):
                # The centre on the incoming wall, then on the outgoing one.
                for apex in (ahead * d_in, -ahead * d_out):
                    poly = _corner_polygon(apex, d_in, d_out)
                    true_cov = float(_point_in_poly(grid, poly).mean())
                    key = "reflex" if _is_reflex(apex, d_in, d_out, poly) else "convex"
                    (n1, _, d1, cp1, _), (n2, _, d2, cp2, _), order = _corner_walls(
                        poly
                    )
                    for meet, rule in ((order, "turn"), (0, "side tests")):
                        for centre_inside in (False, True):
                            cov, _ = wedge_model(
                                n1,
                                cp1,
                                d1,
                                n2,
                                cp2,
                                d2,
                                centre_inside,
                                ti_area,
                                meet=meet,
                            )
                            k = (rule, key)
                            worst[k] = max(worst.get(k, 0.0), abs(cov - true_cov))
    for (rule, key), err in sorted(worst.items()):
        check(err < 0.08, f"{key} corners by the {rule}, worst |err| {err:.4f}")


def part2c():
    print("2c. corners whose apex is on the pixel centre")
    # Both closest points are the apex, square to their walls, so both side
    # tests abstain and parity at the centre -- on the vertex itself, a
    # rounding -- chose: three corners of a grid of squares at 320x180 drew
    # 0.75 for 0.25. The corner's turn does not depend on where it falls.
    ti_area = _area_probe()
    grid = _pixel_samples()
    rng = np.random.default_rng(5)
    worst = {}
    turns = [90.0, -90.0] + list(rng.uniform(30, 150, 40) * rng.choice((-1, 1), 40))
    for turn in turns:
        ang = rng.uniform(0, 2 * np.pi)
        d_in = np.array([np.cos(ang), np.sin(ang)])
        c, s = np.cos(np.deg2rad(turn)), np.sin(np.deg2rad(turn))
        d_out = np.array([[c, -s], [s, c]]) @ d_in
        apex = np.zeros(2)
        poly = _corner_polygon(apex, d_in, d_out)
        true_cov = float(_point_in_poly(grid, poly).mean())
        key = "reflex" if _is_reflex(apex, d_in, d_out, poly) else "convex"
        (n1, _, d1, cp1, _), (n2, _, d2, cp2, _), order = _corner_walls(poly)
        for meet, rule in ((order, "turn"), (0, "side tests")):
            for centre_inside in (False, True):
                cov, _ = wedge_model(
                    n1, cp1, d1, n2, cp2, d2, centre_inside, ti_area, meet=meet
                )
                k = (rule, key)
                worst[k] = max(worst.get(k, 0.0), abs(cov - true_cov))
    for key in ("convex", "reflex"):
        check(
            worst[("turn", key)] < 0.08,
            f"{key} corners by the turn, worst |err| {worst[('turn', key)]:.4f}"
            f" (by the side tests and parity: {worst[('side tests', key)]:.4f})",
        )


def part2d():
    print("2d. hairpin corners, within the near-parallel cutoff")
    # A turn within 11.5 degrees of reversing: a spike's tip (convex) or a
    # slit's end (reflex). The side tests take no part below |cross| = 0.2,
    # where the call was the intersection unconditionally, which drew the
    # pixel at a slit's end empty.
    ti_area = _area_probe()
    grid = _pixel_samples()
    rng = np.random.default_rng(13)
    worst = {"convex": 0.0, "reflex": 0.0}
    for trial in range(200):
        ang = rng.uniform(0, 2 * np.pi)
        turn = np.deg2rad(rng.uniform(169.0, 179.0)) * (1 if trial % 2 else -1)
        d_in = np.array([np.cos(ang), np.sin(ang)])
        rot = np.array([[np.cos(turn), -np.sin(turn)], [np.sin(turn), np.cos(turn)]])
        d_out = rot @ d_in
        apex = rng.uniform(-0.45, 0.45, 2)
        poly = _corner_polygon(apex, d_in, d_out)
        true_cov = float(_point_in_poly(grid, poly).mean())
        key = "reflex" if _is_reflex(apex, d_in, d_out, poly) else "convex"
        (n1, _, d1, cp1, sg1), (n2, _, d2, cp2, sg2), meet = _corner_walls(poly)
        if sg1 == 0.0 or sg2 == 0.0:
            continue
        centre_inside = bool(_point_in_poly(np.zeros((1, 2)), poly)[0])
        cov, _ = wedge_model(
            n1, cp1, d1, n2, cp2, d2, centre_inside, ti_area, meet=meet
        )
        worst[key] = max(worst[key], abs(cov - true_cov))
        cov, _ = wedge_model(n1, cp1, d1, n2, cp2, d2, centre_inside, ti_area)
        worst[key + " before"] = max(
            worst.get(key + " before", 0.0), abs(cov - true_cov)
        )
    for key in ("convex", "reflex"):
        check(
            worst[key] < 0.08,
            f"{key} hairpins, worst |err| {worst[key]:.4f}"
            f" (as the intersection: {worst[key + ' before']:.4f})",
        )


def _even_odd(points, contours):
    return sum(_point_in_poly(points, c) for c in contours) % 2 == 1


def _wall(p0, p1, contours):
    """One wall as the kernel sees it from the centre: ``(n, d, cp, (p0, p1))``."""
    d = (p1 - p0) / np.linalg.norm(p1 - p0)
    left = np.array([-d[1], d[0]])
    mid = (p0 + p1) / 2
    sg = 1.0 if _even_odd((mid + 0.0125 * left)[None], contours)[0] else -1.0
    seg = p1 - p0
    tt = np.clip(np.dot(-p0, seg) / np.dot(seg, seg), 0, 1)
    return left * sg, d, p0 + tt * seg, (p0, p1)


def part2e():
    print("2e. two walls that do not meet")
    ti_area = _area_probe()
    grid = _pixel_samples()
    rng = np.random.default_rng(17)

    # A stem narrower than a pixel through the pixel: its two sides are the
    # two walls, the second's foot inside it, and the pixel is their strip.
    worst = 0.0
    for _ in range(200):
        ang = rng.uniform(0, np.pi)
        d = np.array([np.cos(ang), np.sin(ang)])
        left = np.array([-d[1], d[0]])
        w = rng.uniform(0.2, 0.9)
        off = rng.uniform(-0.3, 0.3)
        a, b = (off - w / 2) * left, (off + w / 2) * left
        stem = np.array([a - 30 * d, a + 30 * d, b + 30 * d, b - 30 * d])
        walls = [_wall(stem[0], stem[1], [stem]), _wall(stem[2], stem[3], [stem])]
        walls.sort(key=lambda wl: np.linalg.norm(wl[2]))
        (n1, d1, cp1, _), (n2, d2, cp2, seg2) = walls
        centre_inside = bool(_even_odd(np.zeros((1, 2)), [stem])[0])
        cov, _ = wedge_model(
            n1,
            cp1,
            d1,
            n2,
            cp2,
            d2,
            centre_inside,
            ti_area,
            crosses=_crosses_pixel(*seg2),
        )
        worst = max(worst, abs(cov - float(_even_odd(grid, [stem]).mean())))
    check(worst < 0.08, f"a sub-pixel stem's two sides, worst |err| {worst:.4f}")

    # A slit narrower than a pixel through the pixel, cut into a shape: the
    # same two walls with their inward sides turned away from each other, and
    # the pixel is their union. Intersected as a stem, it drew empty (a slit
    # 0.08 px wide in a glyph in text_and_media, 0.00 for 0.92).
    worst = {"by side": 0.0, "as a stem": 0.0}
    for _ in range(200):
        ang = rng.uniform(0, np.pi)
        d = np.array([np.cos(ang), np.sin(ang)])
        left = np.array([-d[1], d[0]])
        w = rng.uniform(0.2, 0.9)
        off = rng.uniform(-0.3, 0.3)
        a, b = (off - w / 2) * left, (off + w / 2) * left
        slit = np.array([a - 30 * d, a + 30 * d, b + 30 * d, b - 30 * d])
        body = np.array([(-40.0, -40.0), (40.0, -40.0), (40.0, 40.0), (-40.0, 40.0)])
        contours = [body, slit]
        walls = [_wall(slit[0], slit[1], contours), _wall(slit[2], slit[3], contours)]
        walls.sort(key=lambda wl: np.linalg.norm(wl[2]))
        (n1, d1, cp1, _), (n2, d2, cp2, seg2) = walls
        true_cov = float(_even_odd(grid, contours).mean())
        centre_inside = bool(_even_odd(np.zeros((1, 2)), contours)[0])
        cov, _ = wedge_model(
            n1,
            cp1,
            d1,
            n2,
            cp2,
            d2,
            centre_inside,
            ti_area,
            crosses=_crosses_pixel(*seg2),
        )
        worst["by side"] = max(worst["by side"], abs(cov - true_cov))
        _, _, inter = ti_area(n1, -np.dot(n1, cp1), n2, -np.dot(n2, cp2))
        worst["as a stem"] = max(worst["as a stem"], abs(inter - true_cov))
    check(
        worst["by side"] < 0.08,
        f"a sub-pixel slit's two sides, worst |err| {worst['by side']:.4f}"
        f" (as a stem: {worst['as a stem']:.4f})",
    )

    # A long edge across the pixel, and another contour's tip just outside the
    # pixel on the empty side: the tip's wall is the second nearest segment,
    # closest at its end, and its line runs across the pixel where the outline
    # does not. Combined anyway, the wedge cut the pixel along that line (a
    # pixel by a corner in manim_compat_and_plots drew 0.34 for 0.90).
    worst = {"guarded": 0.0, "combined": 0.0}
    cases = 0
    while cases < 300:
        ang = rng.uniform(0, 2 * np.pi)
        d = np.array([np.cos(ang), np.sin(ang)])
        left = np.array([-d[1], d[0]])
        off = rng.uniform(-0.4, 0.4)
        p, q = off * left - 40 * d, off * left + 40 * d
        body = np.array([p, q, q + 40 * left, p + 40 * left])
        tip = rng.uniform(0.8, 1.4) * -left + rng.uniform(-0.6, 0.6) * d
        if np.abs(tip).max() <= 0.55 or np.dot(tip, left) > off - 0.3:
            continue
        # The spike points at the pixel and widens away from it, all of it on
        # the body's empty side.
        out = tip / np.linalg.norm(tip)
        side = np.array([-out[1], out[0]])
        spread = rng.uniform(0.3, 1.2)
        spike = np.array(
            [tip, tip + 6 * (out + spread * side), tip + 6 * (out - spread * side)]
        )
        if (spike @ left).max() > off - 0.05:
            continue
        cases += 1
        contours = [body, spike]
        first = _wall(body[0], body[1], contours)
        second = min(
            (_wall(spike[0], spike[1], contours), _wall(spike[2], spike[0], contours)),
            key=lambda wl: np.linalg.norm(wl[2]),
        )
        (n1, d1, cp1, _), (n2, d2, cp2, seg2) = first, second
        true_cov = float(_even_odd(grid, contours).mean())
        centre_inside = bool(_even_odd(np.zeros((1, 2)), contours)[0])
        for crosses, key in ((_crosses_pixel(*seg2), "guarded"), (True, "combined")):
            cov, _ = wedge_model(
                n1, cp1, d1, n2, cp2, d2, centre_inside, ti_area, crosses=crosses
            )
            worst[key] = max(worst[key], abs(cov - true_cov))
    check(
        worst["guarded"] < 0.08,
        f"a wall ending beside the pixel, worst |err| {worst['guarded']:.4f}"
        f" (combined regardless: {worst['combined']:.4f})",
    )

    # A pixel deep inside a shape whose nearest wall, 3-5 px away, runs into
    # a flattening's sliver: a 0.2 px chord doubling back off its end. The two
    # meet, so they are a corner, and the sliver's line runs through the
    # shape: folded in, the corner zeroed pixels deep in a circle's 5 px
    # border (DESIGN_analytic_aa.md ss22.3). Past the filter's reach the
    # nearest wall alone says the pixel is covered.
    worst = {"within reach": 0.0, "regardless": 0.0}
    for _ in range(100):
        depth = rng.uniform(3.0, 5.0)
        along = rng.uniform(-2.0, 2.0)
        back = np.deg2rad(rng.uniform(140.0, 170.0))
        ang = rng.uniform(0, 2 * np.pi)
        rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        corner = np.array([along, -depth])
        sliver = corner + 0.2 * np.array([np.cos(back), np.sin(back)])
        local = np.array(
            [
                (-40.0, -depth),
                corner,
                sliver,
                (40.0, -depth),
                (40.0, 40.0),
                (-40.0, 40.0),
            ]
        )
        poly = local @ rot.T
        (n1, d1, cp1, _), (n2, d2, cp2, _) = (
            _wall(poly[0], poly[1], [poly]),
            _wall(poly[1], poly[2], [poly]),
        )
        true_cov = float(_even_odd(grid, [poly]).mean())
        cov, _ = wedge_model(n1, cp1, d1, n2, cp2, d2, True, ti_area, meet=1)
        worst["within reach"] = max(worst["within reach"], abs(cov - true_cov))
        # The corner folded in at any distance, by its turn.
        a1, a2, inter = ti_area(n1, -np.dot(n1, cp1), n2, -np.dot(n2, cp2))
        convex = _inward_sign(n1, d1) * (n1[0] * n2[1] - n1[1] * n2[0]) > 0
        folded = inter if convex else min(max(a1 + a2 - inter, 0.0), 1.0)
        worst["regardless"] = max(worst["regardless"], abs(folded - true_cov))
    check(
        worst["within reach"] < 0.08,
        f"a far sliver off the nearest wall, worst |err| {worst['within reach']:.4f}"
        f" (folded in regardless: {worst['regardless']:.4f})",
    )


def _far_square(u, r):
    """A big square beyond the line ``p . u = r``, ``u`` a unit vector."""
    v = np.array([-u[1], u[0]])
    return np.array(
        [
            r * u - 200 * v,
            (r + 400) * u - 200 * v,
            (r + 400) * u + 200 * v,
            r * u + 200 * v,
        ]
    )


def part2f():
    print("2f. stored sides gone stale")
    # A wall's stored side is its filled side at its midpoint, and another
    # contour crossing the wall flips it beyond the crossing (even-odd). Here a
    # far square covers the walls' midpoints and not the pixel, so at the pixel
    # every stored side it covers is wrong: both of a corner's walls, one of
    # them, or both sides of a stem. Trusted, the wedge turns the pixel inside
    # out; every such pair must be left to the nearest wall's half-plane, which
    # the parity orients, except where the centre is within ON of the nearest
    # wall's line, which says nothing there.
    ti_area = _area_probe()
    grid = _pixel_samples()
    rng = np.random.default_rng(19)
    kinds = ("both walls of a corner", "one wall of a corner", "both sides of a stem")
    stats = {
        k: {"cases": 0, "single": 0, "on": 0, "guarded": 0.0, "trusted": 0.0}
        for k in kinds
    }
    for trial in range(600):
        kind = kinds[trial % 3]
        ang = rng.uniform(0, 2 * np.pi)
        r = rng.uniform(1.5, 5.0)
        if kind == "both sides of a stem":
            d = np.array([np.cos(ang), np.sin(ang)])
            left = np.array([-d[1], d[0]])
            w = rng.uniform(0.2, 0.9)
            off = rng.uniform(-0.3, 0.3)
            a, b = (off - w / 2) * left, (off + w / 2) * left
            shape = np.array([a - 10 * d, a + 60 * d, b + 60 * d, b - 10 * d])
            contours = [shape, _far_square(d, r)]
            walls = [
                _wall(shape[0], shape[1], contours),
                _wall(shape[2], shape[3], contours),
            ]
            walls.sort(key=lambda wl: np.linalg.norm(wl[2]))
            (n1, d1, cp1, _), (n2, d2, cp2, seg2) = walls
            meet, crosses = 0, _crosses_pixel(*seg2)
        else:
            # Under 80 degrees the far square can cover the outgoing wall's
            # midpoint without the incoming one's.
            top = 150.0 if kind == "both walls of a corner" else 80.0
            turn = np.deg2rad(rng.uniform(30, top)) * (1 if trial % 2 else -1)
            d_in = np.array([np.cos(ang), np.sin(ang)])
            rot = np.array(
                [[np.cos(turn), -np.sin(turn)], [np.sin(turn), np.cos(turn)]]
            )
            d_out = rot @ d_in
            apex = rng.uniform(-0.45, 0.45, 2)
            poly = _corner_polygon(apex, d_in, d_out)
            u = d_out
            if kind == "both walls of a corner":
                u = (d_out - d_in) / np.linalg.norm(d_out - d_in)
            contours = [poly, _far_square(u, r)]
            w_in = _wall(poly[0], poly[1], contours)
            w_out = _wall(poly[1], poly[2], contours)
            meet = 1
            if np.linalg.norm(w_out[2]) < np.linalg.norm(w_in[2]):
                w_in, w_out, meet = w_out, w_in, -1
            (n1, d1, cp1, _), (n2, d2, cp2, _) = w_in, w_out
            crosses = True
        true_cov = float(_even_odd(grid, contours).mean())
        centre_inside = bool(_even_odd(np.zeros((1, 2)), contours)[0])
        st = stats[kind]
        st["cases"] += 1
        for guards, key in ((True, "guarded"), (False, "trusted")):
            cov, tag = wedge_model(
                n1,
                cp1,
                d1,
                n2,
                cp2,
                d2,
                centre_inside,
                ti_area,
                meet=meet,
                crosses=crosses,
                guards=guards,
            )
            st[key] = max(st[key], abs(cov - true_cov))
            if guards and tag == "single":
                st["single"] += 1
            elif guards and abs(np.dot(n1, cp1)) <= ON:
                st["on"] += 1
    for kind in kinds:
        st = stats[kind]
        check(
            st["single"] + st["on"] == st["cases"],
            f"{kind}: {st['single']} of {st['cases']} left to the half-plane"
            f" ({st['on']} more on the nearest wall's line), worst |err|"
            f" {st['guarded']:.4f} (trusted: {st['trusted']:.4f})",
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
    part2b()
    part2c()
    part2d()
    part2e()
    part2f()
    part3()
    print()
    if FAILS:
        print(f"{len(FAILS)} FAILURES")
        sys.exit(1)
    print("all wedge checks passed")


if __name__ == "__main__":
    main()
