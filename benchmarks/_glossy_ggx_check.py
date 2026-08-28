"""GROUND TRUTH for the deterministic glossy reflection lobe.

The `_aa_match_aa2.py` gate cannot validate this feature: every one of its
reflective configs authors roughness 0.0-0.05, and its aa=4 reference is
rendered by the same engine, which is to say by a renderer that ignores
roughness on the bounce. It is a SHARP-reflection reference. So the lobe needs
its own ground truth, and this is it -- two independent checks:

PART A -- the sampler against the analytic GGX NDF (no renderer involved).
    Reimplements the tap construction of ``raster_taichi._glossy_reflect`` in
    numpy and compares the distribution of microfacet normals it produces
    against the closed-form GGX / Trowbridge-Reitz density

        p(theta) = D(theta) * cos(theta) * sin(theta) * 2*pi,
        D = a^2 / (pi * (cos^2 (a^2 - 1) + 1)^2),     a = roughness^2

    which is the SAME D that ``shading_taichi._ggx_distribution`` evaluates for
    the direct highlight -- the point of the exercise being that the reflected
    image and the highlight beside it describe one material. Reported as the
    Kolmogorov-Smirnov distance between the empirical and analytic CDFs of the
    half-angle over the taps of a 4x4 pixel block (4 taps x 16 Bayer rotations).

    It also scores the Monte Carlo megakernel's normal-perturbation lobe
    (``rd + roughness * random_unit``) against the same analytic CDF, which is
    what makes this a discriminating test rather than a self-consistency one:
    that model is a different lobe, ~3x wider, and it fails where GGX passes.

PART B -- rendered blur width against the r^2 prediction.
    Renders a flat mirror reflecting a straight bright edge and measures the
    10-90% rise width of the reflected edge across a roughness sweep. GGX with
    alpha = roughness^2 predicts the screen-space blur width to be PROPORTIONAL
    TO ROUGHNESS^2; a normal-perturbation lobe would give roughness^1. Fitting
    the exponent therefore tells which lobe is actually wired in, using only
    rendered pixels and no knowledge of the camera.

Run: .venv/Scripts/python.exe benchmarks/_glossy_ggx_check.py [--part a|b|ab]
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("ALGAN_PREFETCH_BATCHES", "0")

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

GOLDEN_ANGLE = 2.3999632297286533
OUT_DIR = os.path.join(os.path.dirname(__file__), "_rt2_out")
os.makedirs(OUT_DIR, exist_ok=True)

# Sorted analytic lobe deflections, one array per roughness (see
# ``lobe_deflections``); filled in by ``part_b``.
DEFL = {}
# 10-90% rise of the analytic ESF, in radians, per roughness.
ANALYTIC_W = {}


# ---------------------------------------------------------------------------
# PART A -- the sampler, in numpy, against the closed-form NDF
# ---------------------------------------------------------------------------


def bayer4(px, py):
    """Mirror of ``raster_taichi._bayer4``."""

    def m2(a, b):
        return 2 * (a ^ b) + a

    return 4 * m2(py & 1, px & 1) + m2((py >> 1) & 1, (px >> 1) & 1)


def tap_half_angles(roughness, k, interleave=True):
    """Half-angles ``theta_h`` the kernel's taps produce over a 4x4 block.

    Mirrors ``_glossy_rotation`` + the radial half of ``_glossy_reflect``. The
    azimuth is irrelevant to the marginal distribution of theta, so only the
    radial coordinate is reproduced here.
    """
    a = roughness * roughness
    out = []
    for py in range(4):
        for px in range(4):
            if interleave:
                b = bayer4(px, py)
                r_off = (b + 0.5) / 16.0
            else:
                r_off = 0.5
            for j in range(k):
                u1 = (j + r_off) / k
                tan2 = (a * a) * u1 / max(1.0 - u1, 1e-6)
                out.append(np.arctan(np.sqrt(tan2)))
    return np.array(out)


def ggx_theta_cdf(theta, roughness):
    """Analytic CDF of the half-angle under ``p(h) = D(h) cos(theta)``.

    The standard NDF-sampling density integrates in closed form to
    ``F(theta) = tan^2 / (a^2 + tan^2)``; stated independently of the sampler so
    agreement is evidence rather than tautology.
    """
    a = roughness * roughness
    t2 = np.tan(theta) ** 2
    return t2 / (a * a + t2)


def mc_lobe_half_angles(roughness, n=200000, seed=0):
    """The Monte Carlo megakernel's lobe, as HALF-angles, for comparison.

    ``rd_new = normalize(mirror + roughness * random_unit)``; the half-vector
    between ``-rd`` and ``rd_new`` sits at half the deflection, so the angles
    are directly comparable to the GGX half-angles above.
    """
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    mirror = np.array([0.0, 0.0, 1.0])
    d = mirror + roughness * v
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    # Reject the (rare) below-horizon half, as the kernel does.
    below = d[:, 2] < 0.0
    d[below, 2] = -d[below, 2]
    return 0.5 * np.arccos(np.clip(d[:, 2], -1.0, 1.0))


def ks_distance(samples, cdf):
    s = np.sort(samples)
    n = len(s)
    f = cdf(s)
    return float(
        np.max(
            np.abs(np.concatenate([f - np.arange(n) / n, (np.arange(n) + 1) / n - f]))
        )
    )


def part_a():
    print(
        "PART A -- tap distribution vs the analytic GGX NDF "
        "(KS distance, lower better)\n"
    )
    print(
        f"{'roughness':>9s} {'alpha':>8s} {'K':>3s} {'GGX taps':>9s} "
        f"{'no-interleave':>14s} {'MC lobe':>9s} {'median deg':>11s}"
    )
    rows = []
    for r in (0.05, 0.18, 0.35, 0.6, 1.0):
        cdf = lambda t, r=r: ggx_theta_cdf(t, r)  # noqa: E731
        for k in (4, 8):
            th = tap_half_angles(r, k, True)
            th_ni = tap_half_angles(r, k, False)
            ks = ks_distance(th, cdf)
            ks_ni = ks_distance(th_ni, cdf)
            ks_mc = ks_distance(mc_lobe_half_angles(r), cdf)
            print(
                f"{r:9.2f} {r * r:8.4f} {k:3d} {ks:9.3f} {ks_ni:14.3f} "
                f"{ks_mc:9.3f} {np.degrees(np.median(th)):11.2f}"
            )
            rows.append((r, k, ks, ks_ni, ks_mc))
    print(
        "\nWith the rotation a tap set is a K*16-point deterministic "
        "quadrature, whose\nbest possible KS distance is 1/(2*K*16): 0.0078 "
        "at K=4, 0.0039 at K=8. Without\nit every pixel repeats the same K "
        "strata, so the floor is 1/(2*K) instead.\nA value at the floor means "
        "the taps are an optimally stratified sample of the\nanalytic lobe; "
        "anything much larger means the SHAPE is wrong."
    )
    ggx_worst = max(r[2] for r in rows)
    ni_worst = max(r[3] for r in rows)
    mc_worst = max(r[4] for r in rows)
    print(
        f"\nworst KS  ggx-taps {ggx_worst:.3f}   no-interleave {ni_worst:.3f} "
        f"  MC normal-perturbation {mc_worst:.3f}"
    )
    return ggx_worst, ni_worst, mc_worst


# ---------------------------------------------------------------------------
# PART B -- rendered edge width against the roughness^2 prediction
# ---------------------------------------------------------------------------

MIRROR_W, MIRROR_H = 640, 360


def _build_edge_scene(roughness):
    """A frame-filling flat mirror reflecting one straight bright edge.

    The camera sits at z = -7 looking along +z, so:

      * the MIRROR is a huge triangulated quad at z = +3 facing the camera. It
        covers every pixel, which is what makes the measurement unambiguous --
        no mirror-region mask to get wrong, every pixel IS the reflected image.
      * the bright PANEL is a half-plane BEHIND the camera (z = -12, spanning
        y >= +3). A mirror facing the camera reflects the space behind it, so
        the panel never appears directly; only its straight lower edge does,
        as one horizontal step part-way up the mirror.

    Nothing else is in the scene, so the step runs from panel-white to
    background-black -- an ideal edge-spread target.
    """
    import torch

    from algan import BLACK, WHITE, Off, Scene, SceneManager, Square
    from algan.mobs.shapes_2d import QuadTriangulated
    from algan.rendering.lights import AmbientLight
    from algan.rendering.shaders.materials import MeshStandardMaterial

    Scene.instance().set_background_color(BLACK, True)
    SceneManager.instance().light_sources = [
        AmbientLight(color=WHITE, intensity=1.0).spawn(animate=False)
    ]
    with Off():
        mirror = QuadTriangulated(
            torch.tensor(
                (
                    (-40.0, -40.0, 3.0),
                    (40.0, -40.0, 3.0),
                    (40.0, 40.0, 3.0),
                    (-40.0, 40.0, 3.0),
                )
            ).float(),
            color=WHITE,
        )
        mirror.set_material(
            MeshStandardMaterial(metalness=1.0, roughness=roughness, color=WHITE)
        )
        mirror.spawn()
        panel = QuadTriangulated(
            torch.tensor(
                (
                    (-60.0, 3.0, -12.0),
                    (60.0, 3.0, -12.0),
                    (60.0, 60.0, -12.0),
                    (-60.0, 60.0, -12.0),
                )
            ).float(),
            color=WHITE,
        )
        panel.spawn()
        _ = Square


def _render_edge(roughness, tag):
    import cv2

    from algan import RenderSettings, SceneManager, render_to_file
    from algan.rendering.raytracing import set_fragment_shading
    from algan.rendering.raytracing import settings as rt_settings

    name = f"glossyEdge_{tag}_r{roughness:g}"
    path = os.path.join(OUT_DIR, name + ".mp4")
    SceneManager.reset()
    set_fragment_shading(True)
    rt_settings.set_analytic_aa(True, bezier=True, triangles=True)
    _build_edge_scene(roughness)
    render_to_file(
        file_path=path,
        video_settings=RenderSettings((MIRROR_W, MIRROR_H), 1, anti_alias_level=1),
    )
    cap = cv2.VideoCapture(path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"no frame for {name}")
    cv2.imwrite(os.path.join(OUT_DIR, name + ".png"), frame)
    return frame.astype(np.float64).mean(2)


def _rise_width(profile):
    """10-90% rise width of a monotone-ish edge profile, in pixels.

    Robust to the profile's absolute levels: normalises against its own 2nd and
    98th percentiles, then measures where the smoothed profile crosses 0.1 and
    0.9 with sub-pixel linear interpolation.
    """
    lo = np.percentile(profile, 2.0)
    hi = np.percentile(profile, 98.0)
    if hi - lo < 1e-6:
        return float("nan")
    y = (profile - lo) / (hi - lo)
    # A 3-tap box removes the per-pixel interleave dither without moving the
    # edge (symmetric kernel), which is what the width measurement needs.
    y = np.convolve(y, np.ones(3) / 3.0, mode="same")

    def cross(level):
        idx = np.where((y[:-1] - level) * (y[1:] - level) <= 0)[0]
        if len(idx) == 0:
            return float("nan")
        i = idx[len(idx) // 2]
        d = y[i + 1] - y[i]
        return i + (0.0 if abs(d) < 1e-9 else (level - y[i]) / d)

    a, b = cross(0.1), cross(0.9)
    return float(abs(b - a))


def lobe_deflections(roughness, n=400000, seed=1):
    """Marginal screen-axis deflection (radians) of a TRUE GGX lobe, sorted.

    Draws microfacet normals from the analytic density -- the inverse CDF of the
    same closed form Part A scores against -- reflects a head-on view about
    each, and keeps the component along the screen's y axis, which is what a
    horizontal edge resolves. Nothing here consults the kernel's tap
    construction, so agreement is evidence rather than tautology.

    Sorted once per roughness: the pixels-per-radian scale is a pure multiplier,
    so every candidate scale reads its ESF off this one array by binary search
    instead of re-sampling (which turned the fit from hours into milliseconds).
    """
    rng = np.random.default_rng(seed)
    a = roughness * roughness
    u = rng.random(n)
    theta = np.arctan(a * np.sqrt(u / np.maximum(1.0 - u, 1e-9)))
    phi = rng.random(n) * 2.0 * np.pi
    return np.sort(2.0 * theta * np.cos(phi))


def _levels(y, lo=0.1, hi=0.9):
    """(x10, x50, x90) of a monotone 0..1 profile, sub-sample interpolated."""
    out = []
    for lv in (lo, 0.5, hi):
        idx = np.where((y[:-1] - lv) * (y[1:] - lv) <= 0)[0]
        if len(idx) == 0:
            return None
        i = idx[len(idx) // 2]
        d = y[i + 1] - y[i]
        out.append(i + (0.0 if abs(d) < 1e-9 else (lv - y[i]) / d))
    return out


def _analytic_rise(sorted_defl):
    """10-90% rise of the analytic GGX edge-spread function, in RADIANS.

    The rendered 10-90% width divided by this is the pixels-per-radian the
    render implies. That number is pure camera geometry and identical for every
    roughness, so its CONSTANCY across the sweep is the parameter-free claim:
    a lobe of a different shape would need a different scale at each roughness.

    (Two absolute goodness-of-fit metrics were tried first and neither
    discriminated -- a fitted px/radian scale has a degenerate escape at
    scale -> 0, where every predicted curve collapses to a step and the arm
    that IGNORES roughness scores best; and a width-normalised shape L1 ends up
    dominated by the 4x4 rotation's residual block wobble, scoring all three
    arms equal to three decimals. They are not reported. The lobe's SHAPE is
    established by Part A against the closed form, which is exact.)
    """
    lo, hi = np.percentile(sorted_defl, [10.0, 90.0])
    return float(hi - lo)


def _edge_profile(img):
    """Column-averaged vertical profile, normalised to [0, 1]."""
    prof = img[:, MIRROR_W // 4 : 3 * MIRROR_W // 4].mean(1)
    lo, hi = np.percentile(prof, 2.0), np.percentile(prof, 98.0)
    return np.clip((prof - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def _dither_rms(img, prof01):
    """Per-pixel departure from the local mean inside the transition band.

    The interleave rotation makes neighbouring pixels sample different parts of
    the lobe, so a K-tap fan resolves the transition into K+1 levels arranged as
    an ordered dither. This measures how strong that texture is.
    """
    band = np.where((prof01 > 0.15) & (prof01 < 0.85))[0]
    if len(band) < 4:
        return float("nan")
    sub = img[band][:, MIRROR_W // 4 : 3 * MIRROR_W // 4]
    local = np.array([np.convolve(row, np.ones(9) / 9.0, mode="same") for row in sub])
    inner = slice(8, -8)
    return float(np.sqrt(np.mean((sub[:, inner] - local[:, inner]) ** 2)))


def _band_flatness(prof01):
    """Fraction of the transition band sitting on a flat step.

    The failure mode of the un-rotated fan is not noise but BANDING: every pixel
    samples the same K radial strata, so a step edge resolves into K hard bands
    rather than a gradient. That is invisible to a high-frequency noise measure
    (the bands are smooth), so it needs its own: the share of rows inside the
    transition whose local slope is under a tenth of the mean slope.
    """
    band = np.where((prof01 > 0.1) & (prof01 < 0.9))[0]
    if len(band) < 8:
        return float("nan")
    seg = prof01[band]
    d = np.abs(np.diff(seg))
    if d.mean() < 1e-9:
        return float("nan")
    return 100.0 * float(np.mean(d < 0.1 * d.mean()))


def _arm(roughs, tag, reuse=False):
    profs, widths, dith, bands = [], [], [], []
    for r in roughs:
        png = os.path.join(OUT_DIR, f"glossyEdge_{tag}_r{r:g}.png")
        if reuse and os.path.exists(png):
            import cv2

            img = cv2.imread(png).astype(np.float64).mean(2)
        else:
            img = _render_edge(r, tag)
        p01 = _edge_profile(img)
        profs.append(p01)
        widths.append(_rise_width(img[:, MIRROR_W // 4 : 3 * MIRROR_W // 4].mean(1)))
        dith.append(_dither_rms(img, p01))
        bands.append(_band_flatness(p01[::-1]))
    return profs, np.array(widths), np.array(dith), np.array(bands)


def part_b(roughs=(0.0, 0.1, 0.15, 0.2, 0.28, 0.35)):
    import algan.rendering.raytracing.settings as rt

    print("\n\nPART B -- rendered reflected edge vs the analytic GGX prediction\n")
    reuse = "--reuse" in sys.argv
    arms = {}
    for name, (on, il, sec) in {
        "K=4 interleave": (True, True, 4),
        "K=4 plain fan": (True, False, 4),
        "K=8 interleave": (True, True, 8),
        "glossy off": (False, True, 4),
    }.items():
        rt.set_glossy_reflection(on, interleave=il)
        rt.analytic_aa_secondary_samples = sec
        arms[name] = _arm(roughs, "gl_" + name.replace(" ", "").replace("=", ""), reuse)
    rt.set_glossy_reflection(True, interleave=True)
    rt.analytic_aa_secondary_samples = 4

    rr = np.array(roughs)
    m = rr > 1e-6
    DEFL.update({r: lobe_deflections(r) for r in rr[m]})
    ANALYTIC_W.update({r: _analytic_rise(DEFL[r]) for r in rr[m]})
    print(
        f"{'arm':16s} {'width exponent':>15s} {'px/rad':>11s} "
        f"{'spread %':>9s} {'dither RMS':>11s} {'banded %':>9s}"
    )
    for name, (_profs, ww, dith, bands) in arms.items():
        base = ww[0]
        exp = float("nan")
        if np.isfinite(ww[m]).all() and (ww[m] > base).all():
            extra = np.sqrt(np.maximum(ww[m] ** 2 - base**2, 1e-9))
            exp = float(np.polyfit(np.log(rr[m]), np.log(extra), 1)[0])
        # px per radian implied at each roughness, using the analytic lobe's own
        # 10-90% deflection width. Constant across the sweep <=> the rendered
        # blur tracks the analytic lobe; the VALUE is geometry, only its
        # constancy is a claim.
        sc = np.array([ww[i] / ANALYTIC_W[rr[i]] for i in np.where(m)[0]])
        spread = 100.0 * float(np.std(sc) / max(np.mean(sc), 1e-9))
        d_m = dith[m][np.isfinite(dith[m])]
        b_m = bands[m][np.isfinite(bands[m])]
        print(
            f"{name:16s} {exp:15.2f} {np.mean(sc):11.0f} {spread:9.1f} "
            f"{(d_m.mean() if len(d_m) else float('nan')):11.2f} "
            f"{(b_m.mean() if len(b_m) else float('nan')):9.1f}"
        )
    print("""
  width exponent  GGX alpha=roughness^2 predicts 2.00; a normal-perturbation
                  lobe would give 1.00. This is the model discriminator.
  px/rad, spread  rendered 10-90% width divided by the ANALYTIC lobe's own
                  10-90% deflection. That ratio is pure camera geometry, so it
                  must be the SAME at every roughness -- the SPREAD is the
                  claim, the value is not. (The ~20% offset between K=4 and
                  K=8 is a measurement artifact: the 10-90 crossings are read
                  off a 16- vs 32-step staircase, and a coarser staircase
                  biases them outward. Part A shows both sample the identical
                  distribution.)
  dither RMS      what the per-pixel rotation costs, in 0-255 levels.
  banded %        share of the transition sitting on a flat step -- the
                  un-rotated fan's failure mode, which is banding, not noise.""")
    for name, (_p, ww, _d, _b) in arms.items():
        print(f"  {name:16s} 10-90% rise px: " + " ".join(f"{w:7.2f}" for w in ww))
    print("  roughness:                       " + " ".join(f"{r:7.2f}" for r in rr))


def main():
    part = "ab"
    if "--part" in sys.argv:
        part = sys.argv[sys.argv.index("--part") + 1]
    if "a" in part:
        part_a()
    if "b" in part:
        import torch

        with torch.inference_mode():
            part_b()


if __name__ == "__main__":
    main()
