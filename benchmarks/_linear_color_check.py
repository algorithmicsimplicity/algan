"""Acceptance checks for the linear working space.

Algan historically had no sRGB<->linear conversion: authored colours were
display-referred, every shading and compositing operation ran on those encoded
numbers, and the float->byte write was a bare ``clamp(c) * 255``. The linear
working space decodes authored colour at the render boundary, does all the
arithmetic in linear light, and applies the sRGB OETF at the final byte write.

This script checks that change against three **external** invariants -- things
that are true of a correct pipeline, rather than things that are merely
self-consistent. Each is reported for both arms
(``SETTINGS.raytracing.linear_color_space`` on and off), so it doubles as the
before/after table.

``roundtrip``
    Decode-then-encode with no arithmetic between is the identity, so an
    **unlit flat 2-D fill must still render its authored bytes exactly**. This
    is the acceptance gate: the linear space is only worth having if it leaves
    flat content -- most of what Algan renders -- untouched. Nine colours, the
    same table ``_tonemap_render_check.py`` uses.

``additivity``
    Two invariants, because the first one alone passes vacuously:

    1. **N lights of intensity i must render exactly as one light of intensity
       N*i.** That is what "lights add" means.
    2. **The response must keep rising with total intensity.** Invariant 1 is
       satisfied by any rule that depends only on the total -- including the
       energy-conserving normalisation (``_energy_scale``), which divides by
       ``max(budget, 1)`` and therefore pins every total at or above 1.0 to the
       same pixel. Measured on the current tree, totals of 1.2, 1.5 and 1.8 all
       rendered (102, 102, 102): split and single agreed perfectly, and the
       lights were not adding at all. Monotonicity is what catches that.

    Both are measured under an exposure that keeps every case inside the
    display range, so neither can be manufactured by clipping to white.

``encoding``
    **Reflected radiance must be proportional to light intensity in linear
    light.** Sweep one light's intensity and read the byte back. A correct
    pipeline makes ``srgb_to_linear(byte/255)`` affine in intensity; the old
    display-referred pipeline makes ``byte/255`` affine in it instead. Fitting
    both models and reporting each one's residual says which space the shading
    arithmetic actually happened in, without needing to know the ambient term
    exactly -- the fit's intercept absorbs it.

    <venv-python> benchmarks/_linear_color_check.py

Outputs land in ``algan_outputs/linear_color_check/``.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

from algan import (
    SETTINGS,
    Color,
    Cube,
    DirectionalLight,
    MeshBasicMaterial,
    MeshLambertMaterial,
    Prism,
    Scene,
    Square,
)
from algan.constants.spatial import ORIGIN, OUT

OUT_DIR = "algan_outputs/linear_color_check"
PROBE_RES = (160, 120)

WHITE_C = Color((1.0, 1.0, 1.0))

# Authored colours as the bytes a user would type.
PATCHES = [
    ("white", (255, 255, 255)),
    ("grey75", (191, 191, 191)),
    ("grey50", (128, 128, 128)),
    ("grey25", (64, 64, 64)),
    ("grey10", (26, 26, 26)),
    ("red", (255, 0, 0)),
    ("green", (0, 255, 0)),
    ("blue", (0, 0, 255)),
    ("yellow", (255, 255, 0)),
]


# --------------------------------------------------------------------------
# The transfer functions, written out here rather than imported, so this script
# checks the renderer against the sRGB standard rather than against Algan's own
# transcription of it. Two implementations agreeing proves nothing if both are
# wrong -- see the AgX transposed-matrix defect in TONEMAP_FINDINGS.md.
# --------------------------------------------------------------------------


def srgb_to_linear(c):
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1.0 / 2.4) - 0.055)


def _linear_arm_supported():
    """Is there a linear-working-space toggle on this tree at all?"""
    return hasattr(SETTINGS.raytracing, "linear_color_space")


def _set_linear(enabled):
    """Select the arm.

    **Each arm must be run in its own process.** The shading kernels gate the
    illumination budget and the peak bound with ``ti.static``, which Taichi
    evaluates when it compiles the kernel and then caches. Flipping the setting
    after a render therefore does not recompile: the second arm silently reuses
    the first arm's kernel and reports its numbers. ``main()`` runs one arm per
    invocation for that reason -- see ``ALGAN_LINEAR_COLOR`` in
    ``_IMPORT_TIME_VARIABLES``, which is the same category of setting.
    """
    SETTINGS.raytracing.set(linear_color_space=bool(enabled))


def _col(rgb_bytes):
    return Color(tuple(c / 255.0 for c in rgb_bytes))


def _read(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit(f"could not read back {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _centre(img):
    """Median of a 5x5 block at the frame centre, as a (r, g, b) byte tuple."""
    h, w = img.shape[:2]
    block = img[h // 2 - 2 : h // 2 + 3, w // 2 - 2 : w // 2 + 3, :3].reshape(-1, 3)
    return tuple(int(v) for v in np.median(block, axis=0))


def _render(build, name, *, exposure=1.0):
    """Render one still of ``build``'s scene and return its centre pixel."""
    os.makedirs(OUT_DIR, exist_ok=True)
    path = f"{OUT_DIR}/{name}.png"
    snapshot = SETTINGS.snapshot()
    try:
        if exposure != 1.0:
            SETTINGS.raytracing.set(tonemap_exposure=exposure)
        scene = Scene()
        build(scene)
        scene.save_frame(
            path,
            SETTINGS.video.set(resolution=PROBE_RES),
            overwrite=True,
            background=Color((0.0, 0.0, 0.0)),
        )
    finally:
        SETTINGS.restore(snapshot)
    return _centre(_read(path))


# --------------------------------------------------------------------------
# 1. Round trip: an unlit flat fill must survive decode -> encode unchanged.
# --------------------------------------------------------------------------


def _flat_fill(rgb_bytes):
    def build(scene):
        # Far larger than the frame and face-on, so every pixel is interior:
        # no silhouette antialiasing anywhere near the readout.
        Square(side_length=60, color=_col(rgb_bytes)).spawn()

    return build


def _flat_mesh_fill(rgb_bytes):
    """The same fill as a TRIANGLE MESH rather than a bezier circuit.

    Both must round-trip, and they are not the same test: a circuit's colour
    rides ``circuit_colors``, while a uniformly-coloured, uniformly-shaded mesh
    is promoted to a 1x1 colour map in ``scene["textures"]`` and never touches
    ``tri_colors`` at all. This arm exists because the first version of this
    harness checked only the circuit, which was the one route the decode
    reached -- so it passed while every 3-D mesh in the engine rendered
    ``encode(authored)``. A round-trip invariant is only as good as the number
    of ingest routes it covers.
    """

    def build(scene):
        Prism(dimensions=(60.0, 60.0, 1.0)).set_material(
            MeshBasicMaterial(color=_col(rgb_bytes))
        ).spawn()

    return build


def run_roundtrip(arms):
    print("\n## roundtrip -- unlit flat fills must reproduce authored bytes\n")
    header = f"| {'colour':>7} | {'route':>8} | {'authored':>15} |" + "".join(
        f" {label:>15} |" for label, _ in arms
    )
    print(header)
    print(
        "| " + " | ".join(["-" * 7, "-" * 8, "-" * 15] + ["-" * 15] * len(arms)) + " |"
    )

    failures = []
    for name, rgb in PATCHES:
        for route, factory in (("circuit", _flat_fill), ("mesh", _flat_mesh_fill)):
            cells = []
            for label, enabled in arms:
                if enabled is not None:
                    _set_linear(enabled)
                got = _render(factory(rgb), f"roundtrip_{name}_{route}_{label}")
                cells.append(got)
                if got != tuple(rgb):
                    failures.append((f"{name}/{route}", label, tuple(rgb), got))
            print(
                f"| {name:>7} | {route:>8} | {str(tuple(rgb)):>15} |"
                + "".join(f" {str(c):>15} |" for c in cells)
            )

    if failures:
        print("\n  FAIL -- authored colour did not survive the pipeline:")
        for name, label, want, got in failures:
            print(f"    {name} [{label}]: authored {want}, rendered {got}")
    else:
        print("\n  PASS -- every authored colour round-trips exactly in every arm.")
    return not failures


# --------------------------------------------------------------------------
# 2. Additivity: N lights at i == 1 light at N*i.
# --------------------------------------------------------------------------


def _lit_cube(intensities):
    """A white Lambert cube face-on, lit head-on by ``len(intensities)`` lights.

    Every light sits at the same place, straight out along the camera axis, so
    each contributes the same ``n.l`` on the front face and the arms differ in
    nothing but how the same total intensity is split up.
    """

    def build(scene):
        scene.clear_lights()
        cube = Cube(side_length=4.0)
        cube.set_material(MeshLambertMaterial(color=WHITE_C))
        cube.spawn()
        for intensity in intensities:
            DirectionalLight(
                location=OUT * 6,
                target=ORIGIN,
                color=WHITE_C,
                intensity=intensity,
            ).spawn(animate=False)

    return build


# (split, single) pairs -- both deliver the same total intensity.
ADDITIVITY_CASES = [
    ((0.3, 0.3), (0.6,)),
    ((0.4, 0.4), (0.8,)),
    ((0.6, 0.6), (1.2,)),
    ((0.5, 0.5, 0.5), (1.5,)),
    ((0.9, 0.9), (1.8,)),
]

# Keeps the largest case (1.8 + ambient) inside the display range, so a match
# cannot be manufactured by both arms clipping to white.
ADDITIVITY_EXPOSURE = 0.4


def run_additivity(arms):
    print(
        "\n## additivity -- N lights at i must render as one light at N*i"
        f"  (exposure {ADDITIVITY_EXPOSURE})\n"
    )
    print(
        f"| {'split':>18} | {'single':>8} | arm | {'split px':>15} | {'single px':>15} | ok |"
    )
    print(
        "| " + " | ".join(["-" * 18, "-" * 8, "---", "-" * 15, "-" * 15, "--"]) + " |"
    )

    failures = []
    by_arm = {label: [] for label, _ in arms}
    for split, single in ADDITIVITY_CASES:
        total = sum(split)
        for label, enabled in arms:
            if enabled is not None:
                _set_linear(enabled)
            tag = f"{'-'.join(str(s) for s in split)}_{label}"
            got_split = _render(
                _lit_cube(split), f"add_split_{tag}", exposure=ADDITIVITY_EXPOSURE
            )
            got_single = _render(
                _lit_cube(single), f"add_single_{tag}", exposure=ADDITIVITY_EXPOSURE
            )
            ok = got_split == got_single
            if not ok:
                failures.append((split, single, label, got_split, got_single))
            by_arm[label].append((total, got_single[0]))
            print(
                f"| {str(split):>18} | {total:>8.2f} | {label:>3} |"
                f" {str(got_split):>15} | {str(got_single):>15} |"
                f" {'yes' if ok else 'NO':>2} |"
            )

    if failures:
        print("\n  Split and single disagree -- lights do not add:")
        for split, single, label, a, b in failures:
            print(f"    {split} vs {single} [{label}]: {a} != {b}")
    else:
        print("\n  split == single in every arm.")

    # Invariant 1 holds for any rule that depends only on the total, including
    # a normalisation that pins every total >= 1.0 to the same pixel. Adding
    # light has to keep making the surface brighter, or the lights are being
    # spent against a budget rather than summed.
    flat = []
    print("\n  response to rising total intensity:")
    for label, samples in by_arm.items():
        ordered = sorted(samples)
        trace = ", ".join(f"{t:.2f}->{b}" for t, b in ordered)
        rising = all(b2 > b1 for (_, b1), (_, b2) in zip(ordered, ordered[1:]))
        print(f"    [{label}] {trace}   {'rising' if rising else 'PLATEAUS'}")
        if not rising:
            flat.append(label)

    if flat:
        print(
            "\n  FAIL -- the response plateaus in arm(s) "
            f"{', '.join(flat)}: past some total, extra light changes nothing."
        )
    else:
        print("\n  PASS -- lights add, and adding more keeps making it brighter.")
    return not failures and not flat


# --------------------------------------------------------------------------
# 3. Encoding: in which space is reflected radiance proportional to intensity?
# --------------------------------------------------------------------------

SWEEP = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _fit_residual(x, y):
    """Max absolute residual of the best-fit line through (x, y)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    slope, intercept = np.polyfit(x, y, 1)
    return float(np.max(np.abs(y - (slope * x + intercept)))), slope, intercept


def run_encoding(arms):
    print(
        "\n## encoding -- reflected radiance must be proportional to intensity"
        " in LINEAR light\n"
    )
    for label, enabled in arms:
        if enabled is not None:
            _set_linear(enabled)
        bytes_out = []
        for intensity in SWEEP:
            px = _render(_lit_cube((intensity,)), f"sweep_{intensity}_{label}")
            bytes_out.append(px[0])

        encoded = np.asarray(bytes_out, dtype=np.float64) / 255.0
        linear = srgb_to_linear(encoded)

        res_gamma, slope_g, int_g = _fit_residual(SWEEP, encoded)
        res_linear, slope_l, int_l = _fit_residual(SWEEP, linear)

        print(f"### arm: {label}")
        print(f"| {'intensity':>9} | {'byte':>4} | {'byte/255':>8} | {'linear':>8} |")
        print("| " + " | ".join(["-" * 9, "-" * 4, "-" * 8, "-" * 8]) + " |")
        for intensity, b, e, li in zip(SWEEP, bytes_out, encoded, linear):
            print(f"| {intensity:>9.2f} | {b:>4d} | {e:>8.4f} | {li:>8.4f} |")
        print(
            f"\n  linear fit  : slope {slope_l:.4f} intercept {int_l:.4f}"
            f"  max residual {res_linear:.5f}"
        )
        print(
            f"  gamma fit   : slope {slope_g:.4f} intercept {int_g:.4f}"
            f"  max residual {res_gamma:.5f}"
        )
        verdict = "LINEAR" if res_linear < res_gamma else "display-referred (gamma)"
        print(f"  -> shading arithmetic happened in: {verdict}\n")


def main():
    """Measure one arm, in this process.

    Deliberately not both. The shading kernels gate their gamma-era
    compensations with ``ti.static``, so the arm is baked in when Taichi
    compiles the kernel and a second arm in the same process would silently
    reuse the first one's code. Run the comparison as two processes::

        ALGAN_LINEAR_COLOR = 0 < venv - python > benchmarks / _linear_color_check.py
        ALGAN_LINEAR_COLOR = 1 < venv - python > benchmarks / _linear_color_check.py
    """
    if _linear_arm_supported():
        label = "on" if SETTINGS.raytracing.linear_color_space else "off"
        print(
            f"Measuring the '{label}' arm only (ALGAN_LINEAR_COLOR"
            f"={'1' if label == 'on' else '0'}).\n"
            "Run the other arm in a separate process -- the kernel gate is\n"
            "compile-time, so both arms in one process report the first one.\n"
        )
        arms = [(label, None)]
    else:
        print(
            "NOTE: SETTINGS.raytracing.linear_color_space does not exist on this\n"
            "      tree, so this is the single-arm 'before' measurement.\n"
        )
        arms = [("before", None)]

    ok_roundtrip = run_roundtrip(arms)
    ok_additivity = run_additivity(arms)
    run_encoding(arms)

    print("\n## summary")
    print(f"  roundtrip  : {'PASS' if ok_roundtrip else 'FAIL'}")
    print(f"  additivity : {'PASS' if ok_additivity else 'FAIL'}")
    print("  encoding   : see the per-arm verdict above")


if __name__ == "__main__":
    main()
