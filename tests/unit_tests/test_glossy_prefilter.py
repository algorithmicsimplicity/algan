"""The split-sum glossy route: the DFG term, the blur radius, the prefiltered
reflection buffer's composite, and four end-to-end renders through the route.

``algan/rendering/raytracing/DESIGN_glossy_prefilter.md`` is the design; the
renderer audit's REPORT.md section 4.5 is the measurement that motivated it.

The end-to-end cases at the bottom drive ``calib_glossy`` -- a rough metal wall
reflecting one small bright emitter against black, REPORT.md §4.5's own
calibration scene -- through the audit's render script, one arm per subprocess.
They depend on ``benchmarks/renderer_audit/`` being present (the crawl case
drives its probe directly); each skips if that tree is missing.

Outside the fast suite for the same reason ``test_raytracing_unit.py`` is: the
end-to-end cases drive Taichi kernel variants and the compile is charged to
whichever test reaches them first. Nothing elsewhere in the codebase can break
them either, so they would not earn the mark by ``tests/README.md``'s rule.

EVERY ARM IS ITS OWN PROCESS. The glossy mode reaches the resolve as a
``ti.static`` gate, resolved when the kernel compiles; a second arm in this
process would silently reuse the first arm's compiled code and report its
numbers as its own. That failure has produced two wrong measurements in this
repo already (see CLAUDE.md's Taichi gotchas).
"""

import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import taichi as ti
import torch

# Importing algan is what initialises Taichi and Torch for this process.
import algan  # noqa: F401  (import for its side effects: ti.init lives there)
from algan.rendering.raytracing.wavefront_kernels_taichi import (
    _env_brdf_approx,
    _material_env_brdf,
    _material_reflectance,
    _mirror_share,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_AUDIT_DIR = _REPO_ROOT / "benchmarks" / "renderer_audit"
_CALIB_SCENE = _AUDIT_DIR / "scenes" / "calib_glossy.json"

# The audit tree is a working tree of measurement scripts rather than the
# package under test, so its absence is a skip, not a failure.
_needs_audit_tree = pytest.mark.skipif(
    not _CALIB_SCENE.exists(),
    reason="benchmarks/renderer_audit/ is not present",
)

# The crawl case is four renders of two camera positions each, and it is the
# most expensive test in the repository: measured at 852 s before the pair
# batching below landed and 704 s after, against 1008 s for the whole rest of
# ``tests/unit_tests`` put together. One assertion is not worth doubling the
# unit suite, so it runs only when asked for -- ``ALGAN_RUN_GLOSSY_CRAWL=1``,
# anywhere, CI included.
#
# It used to be skipped under ``CI`` alone, which made the local suite twice
# the length of the one CI actually gates on and hid the difference behind an
# environment variable nobody sets by hand. The claim it makes is still the
# claim the feature exists for; run it when touching the glossy route, when
# re-deriving REPORT.md 4.5's table, and before a release.
_skip_slow_crawl = pytest.mark.skipif(
    not os.environ.get("ALGAN_RUN_GLOSSY_CRAWL"),
    reason=(
        "the half-pixel crawl case is eight renders (~12 min); set "
        "ALGAN_RUN_GLOSSY_CRAWL=1 to run it"
    ),
)


@ti.kernel
def _probe_env_brdf(
    f0: ti.types.ndarray(),
    nv: ti.types.ndarray(),
    rough: ti.types.ndarray(),
    out: ti.types.ndarray(),
):
    for i in range(f0.shape[0]):
        e = _env_brdf_approx(
            ti.math.vec3(f0[i, 0], f0[i, 1], f0[i, 2]), nv[i], rough[i]
        )
        for k in ti.static(range(3)):
            out[i, k] = e[k]


@ti.kernel
def _probe_material(
    rd: ti.types.ndarray(),
    nrm: ti.types.ndarray(),
    metal: ti.types.ndarray(),
    ior: ti.types.ndarray(),
    albedo: ti.types.ndarray(),
    rough: ti.types.ndarray(),
    env_out: ti.types.ndarray(),
    schlick_out: ti.types.ndarray(),
    throttle_out: ti.types.ndarray(),
):
    for i in range(rd.shape[0]):
        d = ti.math.vec3(rd[i, 0], rd[i, 1], rd[i, 2])
        n = ti.math.vec3(nrm[i, 0], nrm[i, 1], nrm[i, 2])
        a = ti.math.vec3(albedo[i, 0], albedo[i, 1], albedo[i, 2])
        e = _material_env_brdf(d, n, metal[i], ior[i], a, rough[i])
        r, _dp = _material_reflectance(d, n, metal[i], ior[i], a, 0.0)
        share = _mirror_share(rough[i])
        for k in ti.static(range(3)):
            env_out[i, k] = e[k]
            schlick_out[i, k] = r[k]
            throttle_out[i, k] = r[k] * share


def _env_brdf(f0_rows, nv_rows, rough_rows):
    n = len(f0_rows)
    f0 = torch.tensor(f0_rows, dtype=torch.float32)
    nv = torch.tensor(nv_rows, dtype=torch.float32)
    rg = torch.tensor(rough_rows, dtype=torch.float32)
    out = torch.zeros((n, 3), dtype=torch.float32)
    _probe_env_brdf(f0, nv, rg, out)
    return out


def test_env_brdf_reduces_to_fresnel_at_zero_roughness():
    """A mirror's split-sum energy IS its Fresnel reflectance.

    This is what lets the route below ``_GLOSSY_MIN_ROUGHNESS`` keep Schlick
    while the route above it uses the DFG term: the two agree across the
    threshold rather than stepping.
    """
    f0 = [[0.04] * 3, [1.0] * 3, [0.95, 0.64, 0.54]]
    out = _env_brdf(f0, [1.0, 1.0, 1.0], [0.0, 0.0, 0.0])
    for row, expect in zip(out.tolist(), f0):
        for got, want in zip(row, expect):
            assert abs(got - want) < 0.02, (got, want)

    # Grazing: every material reflects (nearly) everything, whatever its f0.
    graze = _env_brdf(f0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    assert graze.min().item() > 0.95, graze


def test_env_brdf_is_bounded_and_falls_with_roughness():
    """Directional albedo: in [0, 1] everywhere, and a rougher metal reflects
    less of what arrives (the lobe spreads past the horizon and the geometry
    term takes the difference).
    """
    rows = []
    for rough in (0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0):
        for nv in (0.05, 0.25, 0.5, 0.75, 1.0):
            rows.append((rough, nv))
    out = _env_brdf(
        [[1.0] * 3] * len(rows), [nv for _r, nv in rows], [r for r, _nv in rows]
    )
    assert out.min().item() >= 0.0, out.min().item()
    assert out.max().item() <= 1.0, out.max().item()

    for nv in (0.25, 0.5, 1.0):
        vals = [
            _env_brdf([[1.0] * 3], [nv], [r])[0, 0].item()
            for r in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
        ]
        assert all(b <= a + 1e-4 for a, b in zip(vals, vals[1:])), (nv, vals)


def test_env_brdf_beats_the_mirror_share_throttle_on_a_rough_metal():
    """The number the renderer audit measured: a metalness-1 roughness-0.35
    metal reflects ~4.7% of what it should under the throttle. The DFG term is
    the analytic answer, and it is an order of magnitude larger.
    """
    n = 1
    rd = torch.tensor([[0.0, 0.0, -1.0]] * n)
    nrm = torch.tensor([[0.0, 0.0, 1.0]] * n)
    metal = torch.tensor([1.0] * n)
    ior = torch.tensor([1.5] * n)
    albedo = torch.tensor([[1.0, 1.0, 1.0]] * n)
    rough = torch.tensor([0.35] * n)
    env = torch.zeros((n, 3))
    schlick = torch.zeros((n, 3))
    throttle = torch.zeros((n, 3))
    _probe_material(rd, nrm, metal, ior, albedo, rough, env, schlick, throttle)

    # Schlick at normal incidence on a white metal is 1: the whole lobe.
    assert abs(schlick[0, 0].item() - 1.0) < 1e-3, schlick
    # The throttle keeps a few percent of it ...
    assert throttle[0, 0].item() < 0.06, throttle
    # ... and the split-sum keeps most of it. 0.807 is the fit's exact value
    # here; what it is short of 1 is the single-scattering GGX model's own
    # energy loss (light that would have needed a second microfacet bounce),
    # which split-sum does not compensate and which is ~19% at this roughness.
    assert 0.78 < env[0, 0].item() < 0.83, env


def test_env_brdf_is_zero_for_the_unlit_sentinel():
    """``metalness < 0`` means no PBR material at all; there is no lobe to
    integrate, and a legacy/unlit surface must not gain a reflection.
    """
    rd = torch.tensor([[0.0, 0.0, -1.0]])
    nrm = torch.tensor([[0.0, 0.0, 1.0]])
    env = torch.zeros((1, 3))
    schlick = torch.zeros((1, 3))
    throttle = torch.zeros((1, 3))
    _probe_material(
        rd,
        nrm,
        torch.tensor([-1.0]),
        torch.tensor([1.5]),
        torch.tensor([[1.0, 1.0, 1.0]]),
        torch.tensor([0.4]),
        env,
        schlick,
        throttle,
    )
    assert env.abs().max().item() == 0.0, env


def test_env_brdf_index_matched_dielectric_has_no_lobe():
    """IOR 1 is index-matched with the air around it: no interface, no
    reflection. Schlick cannot express that limit, so it is an explicit gate in
    both ``_material_reflectance`` and the split-sum term.
    """
    rd = torch.tensor([[0.0, 0.0, -1.0]])
    nrm = torch.tensor([[0.0, 0.0, 1.0]])
    env = torch.zeros((1, 3))
    schlick = torch.zeros((1, 3))
    throttle = torch.zeros((1, 3))
    _probe_material(
        rd,
        nrm,
        torch.tensor([0.0]),
        torch.tensor([1.0]),
        torch.tensor([[1.0, 1.0, 1.0]]),
        torch.tensor([0.4]),
        env,
        schlick,
        throttle,
    )
    assert env.abs().max().item() < 1e-6, env


def test_env_brdf_metal_tint_rides_in_f0():
    """A coloured metal's reflection is tinted; a dielectric's is achromatic.
    The blend is ``mix(dielectric_f0, albedo, metalness)``, the same one
    ``_material_reflectance`` performs.
    """
    rd = torch.tensor([[0.0, 0.0, -1.0]] * 2)
    nrm = torch.tensor([[0.0, 0.0, 1.0]] * 2)
    albedo = torch.tensor([[0.95, 0.64, 0.54], [0.95, 0.64, 0.54]])
    env = torch.zeros((2, 3))
    schlick = torch.zeros((2, 3))
    throttle = torch.zeros((2, 3))
    _probe_material(
        rd,
        nrm,
        torch.tensor([1.0, 0.0]),
        torch.tensor([1.5, 1.5]),
        albedo,
        torch.tensor([0.3, 0.3]),
        env,
        schlick,
        throttle,
    )
    metal, dielectric = env[0], env[1]
    assert metal[0].item() > metal[1].item() > metal[2].item(), metal
    assert abs(dielectric[0].item() - dielectric[1].item()) < 1e-5, dielectric


def test_blur_sigma_matches_the_design_formula():
    """The host and the kernel must agree about the lobe's screen footprint.

    ``sigma_px = k * (2 * roughness^2) / theta_px``, ``k = d_r / (d_p + d_r)``.
    Reproduced here in Python so a change to either side has to change this
    number too (DESIGN_glossy_prefilter.md section 3).
    """
    from algan.rendering.raytracing.settings import glossy_blur_sigma_px

    theta_px = 0.3948 / 480.0  # a PREVIEW frame's 22.62 degrees over 480 rows
    # Contact: the reflected surface touches the reflector, so nothing blurs.
    assert glossy_blur_sigma_px(0.35, 0.0, 5.0, theta_px) == 0.0
    # A reflection ten times further away than the reflector is nearly the
    # full lobe angle.
    far = glossy_blur_sigma_px(0.35, 50.0, 5.0, theta_px)
    full = 2.0 * 0.35 * 0.35 / theta_px
    assert 0.85 * full < far < full, (far, full)
    # An escaped ray (no hit recorded) is an infinitely distant reflection.
    assert math.isclose(
        glossy_blur_sigma_px(0.35, float("inf"), 5.0, theta_px), full, rel_tol=1e-6
    )


# ---------------------------------------------------------------------------
# End-to-end: frames rendered through the route, one arm per process
# ---------------------------------------------------------------------------


def _render_calib_arm(out_dir, suffix, *, glossy, prefilter):
    """Render ``calib_glossy`` once in a FRESH interpreter; return its png.

    Drives the audit's own render script rather than re-authoring the scene,
    so these frames sit on exactly the geometry and camera REPORT.md §4.5's
    numbers were measured on -- at the cost of depending on the benchmark
    tree (declared in the module docstring). Resolution stays at the scene's
    480x360: smaller frames quarter an already-cheap render but move every
    number away from the table this file cites.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    # Nothing may leak in from the pytest process: an inherited ALGAN_GLOSSY_*
    # would quietly override what the caller asked this arm to be.
    for name in (
        "ALGAN_GLOSSY_REFLECTION",
        "ALGAN_GLOSSY_PREFILTER",
        "ALGAN_GLOSSY_INTERLEAVE",
    ):
        env.pop(name, None)
    env["ALGAN_USE_DAEMON"] = "0"
    # Always explicit, for the same reason glossy_probe.py is: the prefiltered
    # route is the DEFAULT half of glossy_reflection now, so an arm that left
    # this unset would silently measure whatever the environment default is.
    env["ALGAN_GLOSSY_PREFILTER"] = "1" if prefilter else "0"
    cmd = [
        sys.executable,
        str(_AUDIT_DIR / "algan_render.py"),
        str(_CALIB_SCENE),
        "--out",
        str(out_dir),
        "--suffix",
        suffix,
        "--no-tonemap",
    ]
    if glossy:
        cmd.append("--glossy")
    proc = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        timeout=1200,
        cwd=str(_REPO_ROOT),
    )
    if proc.returncode != 0:
        raise AssertionError(f"arm {suffix!r} failed:\n{proc.stderr[-2000:]}")
    return Path(json.loads(proc.stdout.strip().splitlines()[-1])["output"])


@pytest.fixture(scope="module")
def calib_arms(tmp_path_factory):
    """Render each calib_glossy configuration at most once per module.

    Returns a lookup ``arm(suffix, glossy=..., prefilter=...) -> png path``.
    Each call is a fresh interpreter (see the module docstring); the cache only
    stops tests that want the SAME configuration from paying for its render
    twice. A fresh-process re-render of a cached configuration is exactly how
    the determinism case below works.
    """
    out_dir = tmp_path_factory.mktemp("glossy_prefilter_e2e")

    def arm(suffix, *, glossy, prefilter):
        path = out_dir / f"calib_glossy.{suffix}.png"
        if not path.exists():
            _render_calib_arm(out_dir, suffix, glossy=glossy, prefilter=prefilter)
        return path

    return arm


def _reflection_spread(png_path):
    """(rms radius px, half-peak pixel count) of the reflected glow.

    The window brackets where the emitter's mirror image lands -- centroid
    measured at (314, 121) of the 480x360 frame. The wall's own level is the
    window's 10th percentile, subtracted before anything else: the two arms
    sit on very different backgrounds because split-sum spends a metal's
    ambient fill on the reflection instead of hiding behind it (REPORT.md
    §4.5.1: the wall drops to ``1 - E`` ≈ 0.34 of what the throttle drew).
    """
    import cv2
    import numpy as np

    im = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)[..., :3]
    grey = im.astype(np.float64).mean(axis=2)
    h, w = grey.shape
    win = grey[int(0.08 * h) : int(0.60 * h), int(0.45 * w) : int(0.90 * w)]
    sig = np.clip(win - np.percentile(win, 10), 0.0, None)
    total = sig.sum()
    ys, xs = np.mgrid[0 : sig.shape[0], 0 : sig.shape[1]]
    cy = (sig * ys).sum() / total
    cx = (sig * xs).sum() / total
    var_yx = (ys - cy) ** 2 + (xs - cx) ** 2
    rms = math.sqrt(float((sig * var_yx).sum() / total))
    return rms, int((sig > 0.5 * sig.max()).sum())


@_needs_audit_tree
def test_prefilter_setting_is_inert_while_glossy_reflection_is_off(calib_arms):
    """The prefilter setting must not move a pixel until glossy reflections on.

    What cannot be tested directly is the property the design actually claims
    (DESIGN_glossy_prefilter.md, first paragraph): nothing here changes a render
    with the route off, i.e. such a frame equals one rendered by the build
    BEFORE the feature existed. A test inside the tree cannot diff against
    "before", so this does the next best thing and holds the two settings the
    feature introduced up against each other: with ``glossy_reflection`` off in
    both arms, rendering with ``ALGAN_GLOSSY_PREFILTER=1`` must be
    byte-identical to rendering with ``ALGAN_GLOSSY_PREFILTER=0``. With the
    route off, every kernel gate below it compiles out and no reflection buffer
    is allocated, so there is nothing left for the flag to touch; a difference
    here means something reads it on the route-off path.

    ``glossy_reflection`` is no longer off by default, so both arms have to say
    so and be *rendered* that way -- which is why ``algan_render.py`` sets the
    setting from its flag rather than only ever turning it on. While it did the
    latter, both arms here rendered glossy and this test failed for the one
    reason it is not about.
    """
    on = calib_arms("gl_off_pf1", glossy=False, prefilter=True)
    off = calib_arms("gl_off_pf0", glossy=False, prefilter=False)
    assert on.read_bytes() == off.read_bytes()


@_needs_audit_tree
def test_prefiltered_reflection_is_substantially_wider(calib_arms):
    """With the route engaged, the reflection BLURS instead of ghosting.

    With ``glossy_reflection`` off, the single mirror ray is throttled to the
    lobe share it can honestly stand for, so the emitter arrives as a sharp
    dim ghost. With the prefilter on, the same ray lands in a buffer that is
    then blurred by the lobe's screen footprint, so it arrives as a wide soft
    glow. This measures the width both ways rather than eyeballing it:
    brightness-weighted second moment of the reflecting region about its own
    centroid, in pixels.

    Measured values (480x360, aa 3, this machine): throttled rms **7.7 px**,
    prefiltered rms **61.6 px** -- 8.0x wider. Pixels above half the region's
    own peak: 375 vs 718 (1.9x). The rms separates the arms most cleanly --
    the half-peak count barely moves because the blurred glow's skirt sits
    under half its peak while the ghost concentrates all of its energy above
    it -- so the rms is what is asserted; the counts are recorded here for
    whoever has to re-derive the threshold. The bound keeps a ~2.7x margin
    under the measured ratio.
    """
    ghost = _reflection_spread(calib_arms("gl_off_pf1", glossy=False, prefilter=True))
    glow = _reflection_spread(calib_arms("gl_on_pf1", glossy=True, prefilter=True))
    assert glow[0] > 3.0 * ghost[0], (ghost, glow)


@_needs_audit_tree
def test_prefiltered_route_is_deterministic_across_processes(calib_arms):
    """The same prefiltered frame, twice, agrees to the byte.

    This is the claim that the route cannot hiss between frames. The glossy
    ray's direction is a smooth function of position, the prefilter is a fixed
    mip pyramid, and nothing in the route draws from a RNG
    (DESIGN_glossy_prefilter.md §2.2) -- so two renders of one configuration
    must be byte-identical even when they share no process state. Separate
    processes is the honest form of the check: sharing a warm interpreter
    would hide whatever a fresh one has to rebuild.
    """
    a = calib_arms("gl_on_pf1", glossy=True, prefilter=True)
    b = _render_calib_arm(a.parent, "gl_on_pf1_again", glossy=True, prefilter=True)
    assert a.read_bytes() == b.read_bytes()


# ---------------------------------------------------------------------------
# The claim is shared by the band that made it (DESIGN_sheet_resolve.md §4.4)
# ---------------------------------------------------------------------------

# A lit flat-shaded rough metal, alone on a flat background: the geometry the
# claim-sharing rule exists for. Every interior edge of a Polyhedron is a hard
# crease, so its pixels are ONE surface's coverage subdivided into §4.4 sibling
# sheets -- which is the split that must not change what the band commits.
#
# Ambient + one directional light only, deliberately: each facet is then EXACTLY
# uniform, so the frame is piecewise constant with analytic blends at the facet
# boundaries and an interior pixel brighter than all four of its neighbours
# cannot be geometry. It is the seam.
_SEAM_SCENE = """
import sys
from algan import (
    DARKER_GRAY, ORIGIN, OUT, RED, RIGHT, UP, WHITE,
    AmbientLight, DirectionalLight, Icosahedron, MeshStandardMaterial,
    Off, PREVIEW, SETTINGS, Scene,
)

SETTINGS.paths.set(output_root=sys.argv[1], output_directory=".")
with Scene() as scene:
    Scene.set_background(DARKER_GRAY)
    with Off():
        AmbientLight(color=WHITE, intensity=0.45).spawn(animate=False)
        DirectionalLight(
            location=RIGHT * 5 + UP * 5 + OUT * 4, target=ORIGIN,
            color=WHITE, intensity=0.85,
        ).spawn(animate=False)
        solid = Icosahedron(edge_length=0.85).set_material(
            MeshStandardMaterial(color=RED, roughness=0.35, metalness=0.4)
        )
        solid.rotate(60, RIGHT)
    solid.spawn(animate=False)
    scene.save_frame("seam", video_settings=PREVIEW, overwrite=True)
"""


def _interior_local_maxima(png_path, margin):
    """Pixels strictly brighter than all four 4-neighbours by > ``margin``."""
    import numpy as np
    from PIL import Image

    v = np.array(Image.open(png_path).convert("RGB")).astype(np.int32).max(axis=2)
    centre = v[1:-1, 1:-1]
    neighbours = np.stack([v[:-2, 1:-1], v[2:, 1:-1], v[1:-1, :-2], v[1:-1, 2:]])
    return int(((centre[None] - neighbours).min(axis=0) > margin).sum())


def test_a_creases_siblings_share_the_pixels_prefiltered_claim(tmp_path):
    """No bright seam down the interior edges of a rough metal solid.

    The prefiltered glossy event is a per-pixel resource, so the first
    qualifying sheet claims it and later ones fall back to the
    ``_mirror_share`` throttle (§2.2). A §4.4 band's siblings are not "later
    sheets": they are one surface's coverage of the pixel, subdivided so each
    shades with its own normal, and §4.4's contract is that the subdivision
    changes nothing the band commits.

    It changed this. The throttle at roughness 0.35 is ~3% of Schlick where the
    split-sum ``E`` is the lobe's whole directional albedo, and the local term
    is ``alpha * (1 - R)`` -- so a crease pixel's far sibling kept energy the
    interior of the same facet gives to its reflection, and the pixel came out
    brighter than BOTH facets it blends. ``E`` is per-channel and a metal's F0
    is its albedo, so on a red metal it came out redder too: on ``tests/fast``,
    whose Icosahedron this scene is, the crease row read ``(207, 106, 94)``
    against ``(181, 103, 91)`` and ``(187, 106, 94)`` on either side -- +21 in
    red against +1 in green, outside the interval a blend of the two can reach.

    Interior local maxima on this frame: **72 before the claim was shared, 0
    after**, so the bound below is nowhere near the measurement on either side.
    The margin keeps it off the analytic blends themselves, which are exact
    convex combinations and cannot exceed their own facets.
    """
    env = dict(os.environ)
    for name in (
        "ALGAN_GLOSSY_REFLECTION",
        "ALGAN_GLOSSY_PREFILTER",
        "ALGAN_GLOSSY_INTERLEAVE",
    ):
        env.pop(name, None)
    env["ALGAN_USE_DAEMON"] = "0"
    proc = subprocess.run(
        [sys.executable, "-c", _SEAM_SCENE, str(tmp_path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=1200,
        cwd=str(_REPO_ROOT),
    )
    assert proc.returncode == 0, proc.stderr[-2000:]

    frame = tmp_path / "seam.png"
    assert frame.exists(), proc.stdout[-2000:]
    assert _interior_local_maxima(frame, margin=2) <= 2, (
        "a bright seam is back on the interior edges: "
        f"{_interior_local_maxima(frame, margin=2)} interior local maxima"
    )


@_needs_audit_tree
@_skip_slow_crawl
def test_half_pixel_camera_nudge_does_not_crawl_the_reflection(tmp_path):
    """REPORT.md §4.5's headline claim, made enforceable: nothing crawls.

    The fans' dither pattern is fixed in SCREEN space, so half a pixel of
    camera motion slides every surface into a different cell of the pattern
    while barely moving the geometry -- the reflection moves 320x as much as
    the control. The prefilter's ray direction is a smooth function of
    position instead, so its figure should sit within a small factor of the
    glossy-off control, far from either fan.

    Drives ``glossy_probe.py --crawl`` -- the audit's own implementation of
    the measurement, which renders calib_glossy twice per arm with the camera
    nudged 0.008 world units (half a pixel) and reports the mean absolute
    difference over the reflecting region as a fraction of its mean. Eight
    480x360 renders in **four** processes -- one per setting combination, with
    each arm's two camera positions batched together, since the nudge is a
    number in the scene spec and gates no kernel. Measured at **12 minutes**
    on this machine (14 before the batching), which is why it is opt-in; see
    ``_skip_slow_crawl``. Not marked ``fast`` for the reasons in the module
    docstring.

    Measured here (and printed identically by REPORT.md §4.5.1's table):
    relative mad/mean of 0.000144 (glossy off), 0.000523 (prefiltered),
    0.031970 (interleaved fan), 0.0335 (plain fan). The first three reproduce
    to every digit shown. The plain fan is the one arm whose figure moves
    between runs -- it *measures* a screen-space dither, and CUDA's split-pixel
    accumulation order is not reproducible to the last bit -- so it is quoted
    to the precision it actually holds; 0.033133 was the value first recorded.
    Batching the pair into one process does not move it: rendering that arm
    the old way, two fresh processes, returns 0.033481925380152294 against the
    batched run's 0.033481925380152294. The assertions keep an
    order-of-magnitude margin around those numbers; the comments say what
    moved if one fires.
    """
    proc = subprocess.run(
        [
            sys.executable,
            str(_AUDIT_DIR / "glossy_probe.py"),
            "--crawl",
            "0.008",
            "--scene",
            str(_CALIB_SCENE),
        ],
        capture_output=True,
        text=True,
        timeout=3600,
        cwd=str(_REPO_ROOT),
    )
    assert proc.returncode == 0, proc.stderr[-2000:]

    rows = {}
    for line in proc.stdout.splitlines():
        found = re.match(
            r"^(\S.*?)\s{2,}(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s*$", line
        )
        if found:
            rows[found.group(1)] = float(found.group(4))

    control = rows.get("glossy off")
    prefiltered = rows.get("glossy, prefiltered")
    interleaved = rows.get("glossy, interleaved fan")
    plain = rows.get("glossy, plain fan")
    missing = {
        "glossy off": control,
        "glossy, prefiltered": prefiltered,
        "glossy, interleaved fan": interleaved,
        "glossy, plain fan": plain,
    }
    assert all(v is not None for v in missing.values()), f"unparsed crawl: {rows}"

    # Sanity that the measurement still sees the artefact at all: both fans
    # must crawl hard against the quiet control (measured 222x; bound asks
    # 20x).
    assert interleaved > 20.0 * control, rows
    assert plain > 20.0 * control, rows

    # The claim itself, stated twice. Distance form: the prefiltered figure
    # sits an order of magnitude closer to the control than to the nearer fan
    # (measured |0.00052 - 0.00014| = 0.00038 vs |0.00052 - 0.03197| =
    # 0.03145, an 83x gap; bound asks 10x).
    nearest_fan = min(abs(prefiltered - interleaved), abs(prefiltered - plain))
    assert abs(prefiltered - control) * 10.0 < nearest_fan, rows

    # Absolute form, so the test fails readably if every arm drifts together:
    # the prefiltered route may spend at most a quarter of a percent of its
    # own brightness on half a pixel of motion (measured 0.052%; both fans
    # are above 3%).
    assert prefiltered < 0.0025, rows
