"""Sanity check on anti-aliasing: bezier-circuit lines vs triangle lines.

Draws one straight line across the frame at a series of slopes, four ways -- as
a :class:`~algan.mobs.shapes_2d.Line` (a cubic bezier circuit), as a thin flat
quad of two triangles, and as a thin :class:`~algan.mobs.shapes_3d.Cylinder`
at two tessellation densities (a prism, also flat triangles) -- and compares
every rendered pixel against the EXACT area of ``strip n pixel square``.

The comparison is analytic, not a supersampled reference. A line's silhouette
is an infinite strip ``{q : |n.q - c| <= h}``, and the area of a half-plane
clipped to a unit square has a closed form (:func:`_halfplane_area`), so the
exact anti-aliased coverage of every pixel is available in closed form once
``(n, c, h)`` are known. Those are FITTED per 128-pixel segment of each line
(with a linear taper, see :func:`fit_segment`), which makes the check
insensitive to where the renderer places a line or how wide it draws it, and
sensitive only to the SHAPE of the coverage ramp -- which is what
anti-aliasing is.

Two readings are reported per line, on rendered pixels converted back to
linear coverage through a measured transfer LUT (:func:`build_lut`):

ink wobble
    THE PRIMARY METRIC, and parameter-free. The coverage summed down each
    cross-section of a constant-width strip is a constant, independent of
    where the strip's edges fall between pixel centres; a line sweeps through
    every sub-pixel phase as it advances, so any failure to resolve coverage
    continuously shows up as ink gained and lost from column to column. See
    :func:`cross_section_ink`.
coverage rms
    Root-mean-square error against the exact analytic coverage.

Both are also computed on synthetic strips drawn by ideal box-filtered
supersamplers and by no anti-aliasing at all (:func:`reference_errors`), and
those land beside every measurement as the yardstick.

Measured on this repo at ``--res md`` (2026-08, CPU render device): the flat
quad scores 0.014 px wobble / 0.0026 rms and the bezier Line 0.017 / 0.0084,
against 0.10 / 0.036 for an ideal 2x2 box filter and 0.39 / 0.095 for no
anti-aliasing -- so both are resolving coverage continuously, not
supersampling. A tessellated Cylinder is the weakest of the four (0.057 /
0.0094, worse the more finely it is diced) because its silhouette pixels are
contended by several triangles and fall back to the 8-sub-pixel-sample masks;
it is still better than an ideal 2x2 supersampler would be.

Renders go through the ordinary public API at default quality settings, so
whatever route the renderer picks for the scene is the route under test, and
``--routes`` reports that route for a set of representative scenes.
``--no-analytic-aa`` is the harness's own self-check: with analytic coverage
disabled every measurement should land on the ``box2`` reference column, which
is what validates the measurement chain.

Run:  <venv-python> benchmarks/_aa_line_check.py [--res md|hd|ld] [--routes]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# --------------------------------------------------------------------------
# Exact analytic coverage
# --------------------------------------------------------------------------
def _halfplane_area(a, b, t):
    """Area of ``{(x, y) in [-1/2, 1/2]^2 : a*x + b*y <= t}``.

    ``(a, b)`` is a unit normal. The unit square is symmetric under
    ``x -> -x`` and ``y -> -y``, so the signs of ``a`` and ``b`` may be folded
    away and the result depends only on their magnitudes. The clipped region is
    then empty, a corner triangle, a trapezoid spanning the square, the
    complement of a corner triangle, or the whole square, in that order as
    ``t`` sweeps from ``-(a+b)/2`` to ``(a+b)/2``.
    """
    a = np.abs(np.asarray(a, dtype=np.float64))
    b = np.abs(np.asarray(b, dtype=np.float64))
    t = np.asarray(t, dtype=np.float64)

    hi = (a + b) / 2.0  # projection of the square's far corner
    lo = np.abs(a - b) / 2.0  # projection of the square's near corner
    big = np.maximum(a, b)
    two_ab = np.maximum(2.0 * a * b, 1e-300)

    # Trapezoid band: the cut crosses two opposite sides, and the area is
    # linear in t with slope 1/max(a, b) (independent of the smaller one).
    band = np.clip(t / np.maximum(big, 1e-300) + 0.5, 0.0, 1.0)
    corner_lo = np.square(np.maximum(t + hi, 0.0)) / two_ab
    corner_hi = 1.0 - np.square(np.maximum(hi - t, 0.0)) / two_ab

    area = np.where(t < -lo, corner_lo, np.where(t > lo, corner_hi, band))
    area = np.where(t <= -hi, 0.0, area)
    area = np.where(t >= hi, 1.0, area)
    return area


def strip_coverage(phi, c, h, px, py):
    """Exact fraction of each pixel square covered by an infinite strip.

    The strip is ``{q : |n.q - c| <= h}`` with ``n = (cos phi, sin phi)``;
    ``px``/``py`` are pixel CENTRES. Coverage is the difference of the two
    bounding half-planes' clipped areas.
    """
    a, b = math.cos(phi), math.sin(phi)
    d = a * px + b * py - c
    return _halfplane_area(a, b, h - d) - _halfplane_area(a, b, -h - d)


def strip_coverage_varying(phi, c, h, px, py):
    """:func:`strip_coverage` with per-pixel ``c`` and ``h`` arrays."""
    a, b = math.cos(phi), math.sin(phi)
    d = a * px + b * py - c
    return _halfplane_area(a, b, h - d) - _halfplane_area(a, b, -h - d)


# --------------------------------------------------------------------------
# Scene construction
# --------------------------------------------------------------------------
#: Angles of the test lines, in degrees, measured from the +x axis. Includes
#: both axis-aligned cases, the exact diagonal, and slopes whose run/rise is
#: deliberately not a small rational.
ANGLES = (0.0, 5.0, 11.0, 18.0, 26.565, 33.0, 44.0, 45.0, 63.435, 71.0, 79.0, 90.0)

#: Angles at which the cross-section ink metric is DEGENERATE and reported but
#: not aggregated. At a multiple of 45 degrees the line advances a whole number
#: of pixels per cross-section, so it never sweeps through sub-pixel phases and
#: even a completely un-antialiased line has zero wobble.
DEGENERATE_WOBBLE_ANGLES = (0.0, 45.0, 90.0, 135.0, 180.0)

#: How far past the frame each line runs, in world units. Long enough that the
#: line's ends (and any cap or open tube mouth) are off-screen at every angle,
#: so only the infinite-strip part of it is ever measured.
LINE_HALF_LENGTH = 40.0

#: Cylinder radius, chosen so its drawn width lands in the same few-pixel range
#: as the default Line stroke.
CYLINDER_RADIUS = 0.045

#: Half-thickness of the flat two-triangle quad, matching the cylinder.
QUAD_HALF_WIDTH = CYLINDER_RADIUS

KINDS = ("bez", "quad", "cyl", "cyl_fine")
KIND_LABELS = {
    "bez": "Line (cubic bezier circuit)",
    "quad": "flat quad (2 triangles)",
    "cyl": "Cylinder (default tessellation)",
    "cyl_fine": "Cylinder (resolution=(256, 2))",
}


def _new_scene(video_settings):
    """A fresh Scene, pushed as the active one."""
    from algan.scene import Scene

    return Scene(video_settings=video_settings)


def build_line(kind, angle_deg, scene):
    """Spawn one full-frame line of the requested kind into ``scene``."""
    import torch

    from algan.constants.color import WHITE
    from algan.mobs.shapes_2d import Line, TriangleTriangulated
    from algan.mobs.shapes_3d import Cylinder
    from algan.rendering.shaders.materials import MeshBasicMaterial

    phi = math.radians(angle_deg)
    direction = torch.tensor([math.cos(phi), math.sin(phi), 0.0])
    # In-plane perpendicular; the frame is the z = 0 plane facing the camera.
    perp = torch.tensor([-math.sin(phi), math.cos(phi), 0.0])
    start = -direction * LINE_HALF_LENGTH
    end = direction * LINE_HALF_LENGTH

    if kind == "bez":
        mob = Line(start, end, color=WHITE, scene=scene)
    elif kind == "quad":
        a = start + perp * QUAD_HALF_WIDTH
        b = start - perp * QUAD_HALF_WIDTH
        c = end - perp * QUAD_HALF_WIDTH
        d = end + perp * QUAD_HALF_WIDTH
        # Two triangles sharing the a-c diagonal, wound so both face +z.
        corners = torch.stack([a, b, c, a, c, d]).view(2, 3, 3)
        mob = TriangleTriangulated(corners, color=WHITE, scene=scene)
        mob.set_material(MeshBasicMaterial(color=WHITE))
    elif kind in ("cyl", "cyl_fine"):
        extra = {"resolution": (256, 2)} if kind == "cyl_fine" else {}
        mob = Cylinder(
            radius=CYLINDER_RADIUS,
            height=2 * LINE_HALF_LENGTH,
            direction=direction,
            color=WHITE,
            scene=scene,
            **extra,
        )
        mob.set_material(MeshBasicMaterial(color=WHITE))
    else:
        raise ValueError(f"unknown kind {kind!r}")

    mob.spawn()
    return mob


#: Calibration patch grid, in world units. Sized to fit the default camera
#: frame (roughly 12.4 x 7.0 units) with room to spare, so every patch is tens
#: of pixels across and its centre can be sampled without touching an edge.
CALIB_COLS = 13
CALIB_PITCH_X = 0.85
CALIB_PITCH_Y = 0.80
CALIB_HALF_X = 0.34
CALIB_HALF_Y = 0.30


def build_calibration(scene, levels):
    """Spawn a grid of large flat patches of known linear grey, for the LUT."""
    import torch

    from algan.constants.color import Color
    from algan.mobs.shapes_2d import TriangleTriangulated
    from algan.rendering.shaders.materials import MeshBasicMaterial

    rows = math.ceil(len(levels) / CALIB_COLS)
    mobs = []
    for i, level in enumerate(levels):
        cx = (i % CALIB_COLS - (CALIB_COLS - 1) / 2) * CALIB_PITCH_X
        cy = ((i // CALIB_COLS) - (rows - 1) / 2) * CALIB_PITCH_Y
        a = torch.tensor([cx - CALIB_HALF_X, cy - CALIB_HALF_Y, 0.0])
        b = torch.tensor([cx + CALIB_HALF_X, cy - CALIB_HALF_Y, 0.0])
        c = torch.tensor([cx + CALIB_HALF_X, cy + CALIB_HALF_Y, 0.0])
        d = torch.tensor([cx - CALIB_HALF_X, cy + CALIB_HALF_Y, 0.0])
        corners = torch.stack([a, b, c, a, c, d]).view(2, 3, 3)
        color = Color((float(level), float(level), float(level)))
        mob = TriangleTriangulated(corners, color=color, scene=scene)
        mob.set_material(MeshBasicMaterial(color=color))
        mob.spawn()
        mobs.append((level, cx, cy))
    return mobs


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
#: Filled by :func:`instrument_route` with what the renderer decided for the
#: most recent render: whether analytic coverage was used, and at what
#: supersample level the frame was actually rendered.
ROUTE = {}


def instrument_route():
    """Record the renderer's own AA route decision for each rendered frame.

    Which anti-aliasing a frame gets is a per-batch decision inside the tracer
    (``analytic_raster_route_active`` / ``effective_anti_alias_level``), not a
    setting, so the only honest way to say which path a measurement describes
    is to ask the renderer as it renders. Wrapping the two functions leaves
    their behaviour untouched.
    """
    from algan.rendering.raytracing import tracer

    if getattr(tracer, "_aa_check_instrumented", False):
        return
    original_route = tracer.analytic_raster_route_active
    original_level = tracer.effective_anti_alias_level

    def route(merged, **kwargs):
        active = original_route(merged, **kwargs)
        ROUTE["analytic"] = bool(active)
        ROUTE["num_pn"] = int(merged.get("num_pn", 0))
        ROUTE["num_triangles"] = int(merged.get("num_triangles", 0))
        ROUTE["num_circuits"] = int(merged.get("num_circuits", 0))
        return active

    def level(merged, requested, **kwargs):
        effective = original_level(merged, requested, **kwargs)
        ROUTE["anti_alias_level"] = int(effective)
        return effective

    tracer.analytic_raster_route_active = route
    tracer.effective_anti_alias_level = level
    tracer._aa_check_instrumented = True


def render(scene, path, video_settings):
    """Render one still to ``path`` on a black background and read it back."""
    import cv2

    from algan.constants.color import BLACK

    ROUTE.clear()
    scene.save_frame(str(path), video_settings, background_color=BLACK, overwrite=True)
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"could not read back {path}")
    return image[..., :3].astype(np.float64)


def _measure_ramp(out_dir, video_settings, levels, tag):
    """Render flat patches of known linear grey and read back their 8-bit value."""
    scene = _new_scene(video_settings)
    patches = build_calibration(scene, levels)
    image = render(scene, out_dir / f"calib_{tag}.png", video_settings)
    h, w = image.shape[:2]
    scale = _world_scale(image)
    measured = []
    for _level, cx, cy in patches:
        px = int(round(w / 2 + cx * scale))
        py = int(round(h / 2 - cy * scale))
        block = image[py - 3 : py + 4, px - 3 : px + 4]
        if block.size == 0:
            raise RuntimeError("calibration patch fell outside the frame")
        measured.append(block.reshape(-1, 3).mean(0))
    measured = np.asarray(measured)
    if (measured.max(1) - measured.min(1)).max() > 1.5:
        raise RuntimeError("calibration patches are not neutral grey")
    return measured.mean(1)


def build_lut(out_dir, video_settings):
    """Map displayed 8-bit value -> linear radiance, by rendering known greys.

    Everything downstream of the resolve -- tonemapping, the transfer curve,
    the 8-bit write -- is a fixed monotone per-pixel function of the composited
    LINEAR value, and compositing is what anti-aliasing happens in. Measuring
    that function on flat patches of known radiance lets coverage be recovered
    from a rendered pixel exactly, with no assumption that the curve is a gamma
    or that full coverage reads 255.

    A second, interleaved ramp is rendered purely to VALIDATE the first: its
    levels are the midpoints of the first ramp's, so inverting the LUT on them
    must return their nominal linear values. That bounds the interpolation
    error of the LUT itself, which would otherwise be charged to the renderer.
    """
    # Denser near black: the curve has a slight toe there, so evenly spaced
    # levels collapse onto the same 8-bit output and the inverse loses
    # resolution exactly where a faint edge pixel lives.
    levels = np.concatenate(
        [np.linspace(0.0, 0.12, 49), np.linspace(0.12, 1.0, 49)[1:]]
    )
    curve = _measure_ramp(out_dir, video_settings, levels, "main")
    if np.any(np.diff(curve) < -0.5):
        raise RuntimeError("calibration curve is not monotone")

    # The output is 8-bit, so several input levels can share one output value.
    # Keep the FIRST level of each plateau: that is the only choice under which
    # the background (output 0) inverts to exactly zero coverage, which the
    # whole measurement depends on. The residual bias is at most one plateau
    # wide and is reported as the LUT's round-trip error below.
    curve, first = np.unique(curve, return_index=True)
    levels = levels[first]

    check_levels = np.linspace(0.01, 0.99, 96)
    check_curve = _measure_ramp(out_dir, video_settings, check_levels, "check")
    recovered = np.interp(check_curve, curve, levels)
    lut_error = float(np.max(np.abs(recovered - check_levels)))
    return levels, curve, lut_error


def _world_scale(image):
    """Pixels per world unit, from the outermost lit calibration patches."""
    lit = image.max(-1) > 8
    cols = np.where(lit.any(0))[0]
    if cols.size < 2:
        raise RuntimeError("no calibration patches were rendered")
    span_px = cols.max() - cols.min() + 1
    span_world = (CALIB_COLS - 1) * CALIB_PITCH_X + 2 * CALIB_HALF_X
    return span_px / span_world


def to_coverage(image, levels, curve):
    """Recover linear coverage in [0, 1] from a rendered white-on-black frame."""
    value = image.mean(-1)
    return np.interp(value, curve, levels)


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------
def _normal_angle(angle_deg):
    """Image-space angle of the strip normal for a world-space line angle."""
    return math.radians(90.0 - angle_deg)


def _initial_params(cov, phi, xs, ys):
    """Rough ``(c, h)`` for a strip of known orientation from its coverage."""
    a, b = math.cos(phi), math.sin(phi)
    weight = cov.sum()
    if weight <= 0:
        return None
    c = float((cov * (a * xs + b * ys)).sum() / weight)
    # Sum of coverage down a cross-section perpendicular to the dominant axis
    # is 2h / |sin phi| for columns, 2h / |cos phi| for rows.
    if abs(b) >= abs(a):
        n_sections = np.unique(xs[cov > 0]).size
        h = 0.5 * abs(b) * weight / max(n_sections, 1)
    else:
        n_sections = np.unique(ys[cov > 0]).size
        h = 0.5 * abs(a) * weight / max(n_sections, 1)
    return c, max(float(h), 0.05)


def fit_segment(cov, xs, ys, phi0, along):
    """Least-squares fit of a locally linearly-tapered strip to one segment.

    Parameters are ``(phi, c0, c1, h0, h1)``: the strip's centre offset and
    half-width each vary linearly in ``along``, the position along the line
    normalised to [-1, 1] over the segment. The taper terms are not cosmetic --
    a drawn line's width genuinely drifts across the frame (see the module
    docstring's note on the transfer of a stroke's width off-axis), and
    charging that smooth drift to anti-aliasing would swamp the measurement.
    Within one pixel the taper is negligible, so evaluating the untapered
    closed form at each pixel's own ``(c, h)`` is exact to ~1e-3.
    """
    from scipy.optimize import least_squares

    init = _initial_params(cov, phi0, xs, ys)
    if init is None:
        return None
    c0, h0 = init

    def residual(p):
        phi, ca, cb, ha, hb = p
        return (
            strip_coverage_varying(phi, ca + cb * along, ha + hb * along, xs, ys) - cov
        )

    result = least_squares(
        residual,
        x0=np.array([phi0, c0, 0.0, h0, 0.0]),
        bounds=(
            np.array([phi0 - 0.25, c0 - 30.0, -10.0, 1e-3, -5.0]),
            np.array([phi0 + 0.25, c0 + 30.0, 10.0, 200.0, 5.0]),
        ),
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
    )
    return result


def analyse_line(cov, angle_deg, segment=128):
    """Fit the strip locally along its length and report coverage error.

    The fit is per segment so that perspective convergence of a prism's two
    silhouette edges, and the switch of a tessellated cylinder's silhouette
    from one lateral edge to the next, are absorbed as geometry rather than
    charged to anti-aliasing.
    """
    h, w = cov.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    # Pixel centres.
    xs += 0.5
    ys += 0.5

    lit = cov > 0.002
    if not lit.any():
        return None
    # Include the unlit ring around the line: a renderer that spills coverage
    # outside the true silhouette must be charged for it.
    band = lit.copy()
    for _ in range(3):
        grown = band.copy()
        grown[1:, :] |= band[:-1, :]
        grown[:-1, :] |= band[1:, :]
        grown[:, 1:] |= band[:, :-1]
        grown[:, :-1] |= band[:, 1:]
        band = grown

    # Strip normal, in IMAGE coordinates. A line at +angle_deg in world space
    # descends in image space (rows increase downward), so its normal is at
    # 90 - angle_deg, not angle_deg + 90.
    phi0 = _normal_angle(angle_deg)
    steep = abs(math.sin(math.radians(angle_deg))) > abs(
        math.cos(math.radians(angle_deg))
    )
    # Segment along the line's dominant axis.
    axis = ys if steep else xs
    extent = h if steep else w
    errors = []
    widths = []
    angles = []
    n_pixels = 0
    for lo in range(0, extent, segment):
        hi = min(lo + segment, extent)
        if hi - lo < segment // 2:
            continue
        sel = band & (axis >= lo) & (axis < hi)
        if sel.sum() < 40 or cov[sel].max() < 0.5:
            continue
        mid = (lo + hi) / 2.0
        along = (axis[sel] - mid) / max((hi - lo) / 2.0, 1.0)
        result = fit_segment(cov[sel], xs[sel], ys[sel], phi0, along)
        if result is None:
            continue
        errors.append(result.fun)
        widths.append(2 * result.x[3])
        angles.append(math.degrees(result.x[0] - phi0))
        n_pixels += int(sel.sum())

    if not errors:
        return None
    err = np.concatenate(errors)
    return {
        "rms": float(np.sqrt(np.mean(err**2))),
        "max": float(np.max(np.abs(err))),
        "p999": float(np.quantile(np.abs(err), 0.999)),
        "width_px": float(np.mean(widths)),
        "width_spread": float(np.max(widths) - np.min(widths)),
        "angle_err_deg": float(np.mean(angles)),
        "n_pixels": n_pixels,
        "n_segments": len(errors),
    }


#: Width of the moving average that separates a line's smooth width drift from
#: the pixel-scale wobble anti-aliasing is responsible for. Long enough that a
#: sub-pixel sweep (which repeats every few pixels at any non-degenerate slope)
#: averages out, short enough to track the slow drift.
DETREND_WINDOW = 101


def cross_section_ink(cov, angle_deg):
    """Pixel-scale wobble in the coverage carried by each cross-section.

    THE PRIMARY METRIC, and it needs no fit at all. For a strip of locally
    constant width, the sum of coverage down each column crossing it is a
    constant -- ``2h / |sin phi|`` -- independent of where the strip's edges
    happen to fall between pixel centres. As the line advances it sweeps
    through every sub-pixel phase, so any failure to resolve coverage
    continuously shows up directly as ink the renderer gains and loses from
    column to column: ropiness, in one number.

    A drawn line's width also drifts slowly across the frame for reasons that
    have nothing to do with anti-aliasing, so the sums are detrended with a
    moving average and only the residual is reported. Cross-sections the strip
    does not fully cross (where it leaves through the top or bottom of the
    frame) are dropped.
    """
    steep = abs(math.sin(math.radians(angle_deg))) > abs(
        math.cos(math.radians(angle_deg))
    )
    axis_sum = cov.sum(0) if not steep else cov.sum(1)
    lit = cov > 0.002
    touches = lit[0, :] | lit[-1, :] if not steep else lit[:, 0] | lit[:, -1]
    valid = (~touches) & (axis_sum > 0.05)

    # Keep only the longest run of valid cross-sections, so the moving average
    # never straddles a gap.
    runs, start = [], None
    for i, ok in enumerate(np.append(valid, False)):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            runs.append((start, i))
            start = None
    if not runs:
        return None
    lo, hi = max(runs, key=lambda r: r[1] - r[0])
    sums = axis_sum[lo:hi]
    if sums.size < DETREND_WINDOW + 32:
        return None

    kernel = np.ones(DETREND_WINDOW) / DETREND_WINDOW
    trend = np.convolve(sums, kernel, mode="valid")
    core = sums[DETREND_WINDOW // 2 : DETREND_WINDOW // 2 + trend.size]
    wobble = core - trend
    mean = float(core.mean())
    return {
        "mean": mean,
        "raw_std": float(sums.std()),
        "wobble_std": float(wobble.std()),
        "wobble_max": float(np.max(np.abs(wobble))),
        "rel_wobble": float(wobble.std() / mean) if mean else float("nan"),
        "n": int(core.size),
    }


def level_stats(cov):
    """How many distinct coverage levels appear on the line's edges.

    A 2x2 box filter can only produce five; an ideal analytic coverage produces
    as many as 8-bit output allows.
    """
    edge = cov[(cov > 0.01) & (cov < 0.99)]
    if edge.size == 0:
        return {"n_levels": 0, "n_edge_px": 0}
    return {
        "n_levels": int(np.unique(np.round(edge * 255).astype(int)).size),
        "n_edge_px": int(edge.size),
    }


# --------------------------------------------------------------------------
# Reference renderers, for context on what the numbers mean
# --------------------------------------------------------------------------
def _synthetic_strip(angle_deg, width_px, shape, k, offset):
    """A strip drawn by an ideal ``k x k`` box-filtered supersampler."""
    h, w = shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    xs += 0.5
    ys += 0.5
    phi = _normal_angle(angle_deg)
    c = offset + math.cos(phi) * w / 2 + math.sin(phi) * h / 2
    exact = strip_coverage(phi, c, width_px / 2, xs, ys)
    if k <= 0:
        return exact, exact
    got = np.zeros_like(exact)
    for i in range(k):
        for j in range(k):
            sx = xs + (i + 0.5) / k - 0.5
            sy = ys + (j + 0.5) / k - 0.5
            d = math.cos(phi) * sx + math.sin(phi) * sy - c
            got += (np.abs(d) <= width_px / 2).astype(np.float64)
    return got / (k * k), exact


def reference_errors(angle_deg, width_px, shape, rng, trials=2):
    """What ideal box-filtered and un-antialiased renderings would score.

    Computed against the same exact analytic coverage the renderer is measured
    against, at the same angle and width, and passed through the SAME two
    metrics -- so the yardstick is measured the way the subject is. An
    ``anti_alias_level = k`` supersampler cannot beat ``box(k)``, and ``none``
    is what no anti-aliasing at all scores.
    """
    out = {}
    for label, k in (("none", 1), ("box2", 2), ("box4", 4), ("exact", 0)):
        rms, wobble = [], []
        for _ in range(trials):
            got, exact = _synthetic_strip(
                angle_deg, width_px, shape, k, float(rng.uniform(-0.5, 0.5))
            )
            band = exact > 0.002
            for _ in range(3):
                grown = band.copy()
                grown[1:, :] |= band[:-1, :]
                grown[:-1, :] |= band[1:, :]
                grown[:, 1:] |= band[:, :-1]
                grown[:, :-1] |= band[:, 1:]
                band = grown
            rms.append(np.sqrt(np.mean((got[band] - exact[band]) ** 2)))
            ink = cross_section_ink(got, angle_deg)
            if ink:
                wobble.append(ink["wobble_std"])
        out[label] = float(np.mean(rms))
        out[label + "_wobble"] = float(np.mean(wobble)) if wobble else float("nan")
    return out


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def _video_settings(name):
    from algan.settings.video_settings import HD, LD, MD

    return {"ld": LD, "md": MD, "hd": HD}[name]


def report_routes(video_settings):
    """Which representative scenes keep analytic coverage, and which don't.

    Anti-aliasing is chosen per render batch, not per setting: a scene whose
    features the raster frontend cannot honor keeps the requested supersample
    level instead. Measuring one line tells you nothing about a scene that
    routes elsewhere, so this enumerates the common cases and asks the
    renderer directly.
    """
    import torch

    from algan.constants.color import BLACK, BLUE, WHITE
    from algan.mobs.shapes_2d import Square
    from algan.mobs.shapes_3d import Sphere
    from algan.mobs.text import Text
    from algan.rendering.raytracing import settings as rt_settings
    from algan.rendering.shaders.materials import (
        MeshPhysicalMaterial,
        MeshStandardMaterial,
    )

    instrument_route()

    def sphere(scene, material=None):
        mob = Sphere(radius=1.2, color=BLUE, scene=scene)
        if material is not None:
            mob.set_material(material)
        return mob.spawn()

    def with_shadows(scene):
        rt_settings.set_ray_traced_shadows(True)
        sphere(scene)
        Square(side_length=6, color=WHITE, scene=scene).spawn()

    def with_spp(scene):
        rt_settings.SAMPLES_PER_PIXEL = 4
        sphere(scene)

    cases = [
        ("lit sphere (triangles)", sphere, None),
        (
            "sphere + Text (triangles + circuits)",
            lambda s: (sphere(s), Text("hi", scene=s).spawn()),
            None,
        ),
        (
            "ray-traced shadows on",
            with_shadows,
            lambda: rt_settings.set_ray_traced_shadows(False),
        ),
        (
            "polished metal",
            lambda s: sphere(
                s, MeshStandardMaterial(color=BLUE, metalness=1.0, roughness=0.05)
            ),
            None,
        ),
        (
            "refracting glass",
            lambda s: sphere(
                s, MeshPhysicalMaterial(color=WHITE, transmission=1.0, ior=1.5)
            ),
            None,
        ),
        (
            "samples_per_pixel = 4 (path tracer)",
            with_spp,
            lambda: setattr(rt_settings, "SAMPLES_PER_PIXEL", 1),
        ),
    ]

    del torch
    print("route taken per scene (analytic coverage vs supersampled fallback)")
    for name, make, teardown in cases:
        ROUTE.clear()
        scene = _new_scene(video_settings)
        try:
            make(scene)
            scene.save_frame(
                str(Path(SCRATCH) / f"route_{abs(hash(name))}"),
                video_settings,
                background_color=BLACK,
            )
        finally:
            if teardown:
                teardown()
        print(
            f"  {name:<38} analytic={str(ROUTE.get('analytic')):<5} "
            f"rendered_at_aa={ROUTE.get('anti_alias_level')}"
        )
    print()


SCRATCH = "/tmp"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--res", default="md", choices=("ld", "md", "hd"))
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--kinds", default=",".join(KINDS), help="comma-separated subset of kinds"
    )
    parser.add_argument(
        "--angles",
        default=None,
        help="comma-separated angles in degrees (default: the standard sweep)",
    )
    parser.add_argument(
        "--routes",
        action="store_true",
        help="also report which representative scenes keep analytic coverage",
    )
    parser.add_argument(
        "--no-analytic-aa",
        action="store_true",
        help=(
            "disable analytic coverage, forcing the supersampled fallback. "
            "SELF-CHECK: the measured numbers should then land on the box2 "
            "reference column, which is what validates the harness."
        ),
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out) if args.out else REPO_ROOT / "algan_outputs" / "aa_check"
    out_dir.mkdir(parents=True, exist_ok=True)

    video_settings = _video_settings(args.res)
    kinds = tuple(k.strip() for k in args.kinds.split(",") if k.strip())
    angles = tuple(float(a) for a in args.angles.split(",")) if args.angles else ANGLES

    from algan.rendering.raytracing import settings as rt_settings

    if args.no_analytic_aa:
        rt_settings.set_analytic_aa(False)
    instrument_route()

    print(f"resolution   {video_settings.resolution}")
    print(f"anti_alias   {video_settings.anti_alias_level} (requested)")
    print(
        f"analytic AA  master={rt_settings.ANALYTIC_AA} "
        f"bez={rt_settings.analytic_aa_bez_active()} "
        f"tri={rt_settings.analytic_aa_tri_active()}"
    )
    print()

    global SCRATCH
    SCRATCH = str(out_dir)
    if args.routes:
        report_routes(video_settings)

    levels, curve, lut_error = build_lut(out_dir, video_settings)
    print(
        f"transfer LUT: linear 0 -> {curve[0]:.1f}, linear 1 -> {curve[-1]:.1f}; "
        f"round-trip error on held-out greys {lut_error:.4f}"
    )
    print()
    print(
        f"{'kind':>5} {'angle':>7}  {'width':>7}  {'ink wobble (px)':>17}  "
        f"{'coverage rms':>13}  {'levels':>6}"
    )

    rng = np.random.default_rng(0)
    rows = []
    for kind in kinds:
        for angle in angles:
            scene = _new_scene(video_settings)
            build_line(kind, angle, scene)
            path = out_dir / f"line_{kind}_{angle:g}.png"
            image = render(scene, path, video_settings)
            route = dict(ROUTE)
            cov = to_coverage(image, levels, curve)
            fit = analyse_line(cov, angle)
            cross = cross_section_ink(cov, angle)
            lev = level_stats(cov)
            ref = (
                reference_errors(angle, fit["width_px"], cov.shape, rng) if fit else {}
            )
            rows.append(
                {
                    "kind": kind,
                    "angle": angle,
                    "fit": fit,
                    "cross": cross,
                    "levels": lev,
                    "ref": ref,
                    "route": route,
                }
            )
            _print_row(rows[-1])

    print()
    _print_summary(rows, kinds)
    return rows


def _print_row(row):
    fit, cross, lev, ref = row["fit"], row["cross"], row["levels"], row["ref"]
    if fit is None:
        print(f"{row['kind']:>5} {row['angle']:>5.1f} deg   NO LINE FOUND")
        return
    if cross:
        cross_txt = f"{cross['wobble_std']:.4f} (box2 {ref.get('box2_wobble', float('nan')):.4f})"
    else:
        cross_txt = "n/a (axis-aligned)"
    print(
        f"{row['kind']:>5} {row['angle']:>5.1f} deg  "
        f"{fit['width_px']:5.2f}px  "
        f"{cross_txt:>17}  "
        f"{fit['rms']:.4f} (box2 {ref.get('box2', float('nan')):.4f})  "
        f"{lev['n_levels']:6d}"
    )


def _print_summary(rows, kinds):
    print("=" * 88)
    print("Metric 1 -- pixel-scale ink wobble per cross-section, in pixels.")
    print("  Parameter-free. Ideal analytic coverage scores ~0; the numbers in")
    print("  brackets are what ideal supersamplers and no AA at all score.")
    print("Metric 2 -- rms coverage error against exact analytic coverage.")
    print("=" * 88)
    for kind in kinds:
        got = [r for r in rows if r["kind"] == kind and r["fit"]]
        if not got:
            continue
        with_sweep = [
            r for r in got if r["cross"] and r["angle"] not in DEGENERATE_WOBBLE_ANGLES
        ]
        if not with_sweep:
            continue
        rms = np.array([r["fit"]["rms"] for r in got])
        wob = np.array([r["cross"]["wobble_std"] for r in with_sweep])
        refs = {
            key: np.array([r["ref"][key] for r in with_sweep])
            for key in ("none_wobble", "box2_wobble", "box4_wobble", "exact_wobble")
        }
        ref_rms = {
            key: np.array([r["ref"][key] for r in got])
            for key in ("none", "box2", "box4", "exact")
        }
        routes = {
            (r["route"].get("analytic"), r["route"].get("anti_alias_level"))
            for r in got
        }
        route_txt = ", ".join(
            f"analytic={a} rendered_at_aa={lv}" for a, lv in sorted(routes, key=str)
        )
        print()
        print(f"{KIND_LABELS[kind]}   [{route_txt}]")
        print(
            f"    ink wobble   {wob.mean():.4f} px  (worst {wob.max():.4f})"
            f"   [none {refs['none_wobble'].mean():.4f} |"
            f" box2 {refs['box2_wobble'].mean():.4f} |"
            f" box4 {refs['box4_wobble'].mean():.4f} |"
            f" exact {refs['exact_wobble'].mean():.4f}]"
        )
        print(
            f"    coverage rms {rms.mean():.4f}     (worst {rms.max():.4f})"
            f"   [none {ref_rms['none'].mean():.4f} |"
            f" box2 {ref_rms['box2'].mean():.4f} |"
            f" box4 {ref_rms['box4'].mean():.4f} |"
            f" exact {ref_rms['exact'].mean():.4f}]"
        )


if __name__ == "__main__":
    main()
