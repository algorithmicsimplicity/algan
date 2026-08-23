"""Acceptance check for what ``Mob.opacity`` means on a closed solid.

``Mob.opacity`` is documented as "0.0 for fully transparent to 1.0 for fully
opaque", and that is a statement about the Mob, not about its triangles. So it
carries an **external** invariant, one that does not depend on knowing anything
about the renderer's internals:

    Rendering a Mob at ``opacity = a`` in front of a backdrop must give exactly
    ``a * (the Mob rendered opaque) + (1 - a) * (the backdrop)``, in linear
    light, for every Mob -- flat or solid.

That is the ordinary alpha composite. A flat ``Circle`` satisfies it. A closed
solid does not, and cannot satisfy it while both of its shells composite: a
camera ray crosses the shell twice, so the backdrop is attenuated ``(1 - a)``
twice while the extra ``a * (1 - a)`` of coverage is painted with the solid's
own interior shading. The measured consequence is that ``opacity`` under-delivers
on every built-in solid, and the author has no value they can write to get the
documented behaviour.

The check is run against **two backdrops**, dark and light, because the two
error terms separate there: over a dark backdrop the doubled attenuation is
invisible (both shells composite against near-black) and what shows is the
interior's own shading; over a light backdrop the doubled attenuation dominates.
A pipeline that only ever gets checked against black looks correct.

``circle``
    The control. A flat shape has one sheet per pixel, so it must satisfy the
    invariant exactly whatever the renderer does with solids. If this arm
    deviates, the harness itself is wrong -- read it before reading the others.

``sphere`` / ``cube``
    Smooth and flat-faced closed solids. Both declare an outside
    (``Mob.two_sided`` False), so both are crossed twice.

``sphere_noambient``
    The same sphere with the ``AmbientLight`` removed. It separates the two
    contributors: the interior's second ambient pass disappears, leaving only
    the doubled attenuation, which a dark backdrop cannot show. Expect this arm
    to look correct on the dark backdrop and wrong on the light one.

Reported per arm and authored alpha:

``dev``
    ``|rendered - exact composite|`` in channel values over the shape's
    interior, mean and max. This is the assumption-free number -- it needs no
    model of the renderer. The render suites' tolerance is 2, so a mean above
    that is a real deviation.

``eff``
    The effective alpha the render actually delivers, solved from the same
    pixels. Reported because it is the number an author can act on: "you asked
    for 0.55 and got 0.69".

    <venv-python> benchmarks/_opacity_alpha_check.py

Outputs land in ``algan_outputs/opacity_alpha_check/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from algan import (
    BLUE,
    DARKER_GRAY,
    GRAY_B,
    LEFT,
    OUT,
    PREVIEW,
    RIGHT,
    UP,
    WHITE,
    AmbientLight,
    Circle,
    Cube,
    DirectionalLight,
    MeshStandardMaterial,
    Off,
    Scene,
    Sphere,
)

OUT_DIR = Path("algan_outputs") / "opacity_alpha_check"

#: Authored opacities to check. 1.0 is the reference the composite is built
#: from, so it is rendered but not scored.
ALPHAS = (1.0, 0.75, 0.55, 0.25)

#: (name, background byte value, algan colour). Two backdrops, because the two
#: error terms are only both visible across the pair -- see the module
#: docstring.
BACKDROPS = (("dark", 34, DARKER_GRAY), ("light", 187, GRAY_B))

#: The render suites' per-channel tolerance. A mean deviation above this is not
#: rounding.
TOLERANCE = 2.0

#: How far the delivered alpha may sit from the authored one before the arm is
#: called a failure. Generous: the point is to catch 0.55 -> 0.69, not to pin
#: the last percent.
ALPHA_TOLERANCE = 0.02

_ARMS = ("circle", "sphere", "cube", "sphere_noambient")

#: Screen x of each arm's centre, in Algan units. The no-ambient sphere is
#: rendered in its own pass (it needs different lighting) at the sphere's slot.
_ARM_X = {"circle": -4.0, "sphere": 0.0, "cube": 4.0, "sphere_noambient": 0.0}


def _srgb_to_linear(u):
    c = np.asarray(u, dtype=np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(v):
    v = np.clip(np.asarray(v, dtype=np.float64), 0.0, 1.0)
    return np.where(v <= 0.0031308, v * 12.92, 1.055 * v ** (1 / 2.4) - 0.055) * 255.0


def _render(background, ambient, alpha, path):
    """One frame: the three shapes at ``alpha`` over ``background``.

    A fresh Scene per frame, so nothing about the previous alpha can survive
    into this one through the timeline.
    """
    with Scene() as scene:
        scene.set_background_color(background)
        with Off():
            if ambient:
                AmbientLight(color=WHITE, intensity=0.45).spawn(animate=False)
            DirectionalLight(
                location=RIGHT * 4 + UP * 5 + OUT * 4,
                target=LEFT * 0,
                color=WHITE,
                intensity=1.0,
            ).spawn(animate=False)
            shapes = {
                "circle": Circle(radius=1.1, color=BLUE),
                "sphere": Sphere(radius=1.1).set_material(
                    MeshStandardMaterial(color=BLUE)
                ),
                "cube": Cube(side_length=1.9).set_material(
                    MeshStandardMaterial(color=BLUE)
                ),
            }
            for name, mob in shapes.items():
                mob.move(RIGHT * _ARM_X[name])
                mob.opacity = alpha
                mob.spawn(animate=False)
        scene.save_frame(str(path), PREVIEW, overwrite=True)


def _interior(opaque, background_byte):
    """Pixels that are unambiguously the shape's interior in the opaque frame.

    Excludes the backdrop, the antialiased rim (where a partial-coverage pixel
    is a blend the invariant does not describe pixel-wise) and any channel that
    clipped -- a clipped reference cannot predict its own composite.
    """
    off_background = np.abs(opaque - background_byte).sum(-1) > 60
    unclipped = (opaque < 250).all(-1) & (opaque > 4).all(-1)
    mask = off_background & unclipped
    # Erode by two pixels to drop the antialiased rim.
    return cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)


def _ink_ratio(opaque, faded, background_byte):
    """Total ink at this alpha as a fraction of the opaque frame's total ink.

    Ink is ``sum |rendered - backdrop|`` in linear light over the WHOLE window,
    silhouette included. The interior deviation above deliberately erodes the
    antialiased rim, so nothing else here would notice a fix that got the
    interior right by under-covering the edge -- which is the failure mode the
    one-mesh ceiling's ``max(front, back)`` shape exists to avoid (a plain
    suppression flipped a rod's signed coverage error negative and notched
    1676 of 3508 interior pixels; see ``ANALYTIC_AA_ONE_MESH``). For a correct
    alpha composite this ratio is the authored alpha, edge and all.

    The absolute value is load-bearing, not tidiness. Signed, a shape that is
    brighter than the backdrop in one channel and darker in another cancels
    against itself, and over a light backdrop the ratio's denominator collapses
    -- it read 0.579 for a flat ``Circle`` at alpha 0.55, and -0.199 for a
    sphere. Per channel ``rendered - backdrop`` is exactly ``a * (opaque -
    backdrop)``, so taking the magnitude first keeps the ratio linear in alpha
    while removing the cancellation. The ``circle`` arm is what caught this:
    when the control fails, the metric is wrong.

    CHANNELS THAT CLIPPED IN THE REFERENCE ARE EXCLUDED, per channel. Where the
    opaque frame reached 255 its true radiance is unknown and larger, so the
    denominator is truncated while the faded frame -- darker by construction --
    still reports honestly; the ratio is then inflated by an amount that has
    nothing to do with the renderer. On the lit sphere this is most of the
    highlight: 6698 of 23716 window pixels clip and carry 68% of the reference
    ink, and including them read 0.621 against an authored 0.55 on geometry
    whose interior deviation was 0.31 of a channel value. The exclusion is per
    CHANNEL rather than per pixel so a highlight that blew only the blue
    channel still contributes its red and green.

    This does not blunt what the column is for. Clipping happens where the
    shape is brightest, which is its interior; the silhouette is partial
    coverage against the backdrop and cannot clip, so every rim pixel survives
    the exclusion and an under-covered edge still shows up here.
    """
    bg = _srgb_to_linear(float(background_byte))
    # A channel is usable when the reference is inside the display range at
    # both ends -- 250/4 rather than 255/0 because the encode rounds.
    usable = (opaque < 250) & (opaque > 4)
    ink_opaque = np.where(usable, np.abs(_srgb_to_linear(opaque) - bg), 0.0).sum()
    if abs(ink_opaque) < 1e-9:
        return float("nan")
    ink_faded = np.where(usable, np.abs(_srgb_to_linear(faded) - bg), 0.0).sum()
    return float(ink_faded / ink_opaque)


def _score(opaque, faded, background_byte, alpha):
    """``(dev_mean, dev_max, eff_alpha, ink_ratio, n_pixels)`` for one arm."""
    mask = _interior(opaque, background_byte)
    ink = _ink_ratio(opaque, faded, background_byte)
    if not mask.any():
        return float("nan"), float("nan"), float("nan"), ink, 0
    bg = _srgb_to_linear(float(background_byte))
    ideal = _linear_to_srgb(alpha * _srgb_to_linear(opaque) + (1.0 - alpha) * bg)
    dev = np.abs(faded - ideal)[mask]
    span = _srgb_to_linear(opaque) - bg
    # Solve the composite for alpha where the shape is far enough from the
    # backdrop for the division to be conditioned.
    usable = mask[..., None] & (np.abs(span) > 0.02)
    eff = ((_srgb_to_linear(faded) - bg) / np.where(usable, span, 1.0))[usable]
    return dev.mean(), dev.max(), eff.mean(), ink, int(mask.sum())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keep",
        action="store_true",
        help="keep the rendered frames instead of only reporting numbers",
    )
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    frames: dict[tuple[str, bool, float], np.ndarray] = {}
    for bg_name, _bg_byte, bg_color in BACKDROPS:
        for ambient in (True, False):
            for alpha in ALPHAS:
                tag = f"{bg_name}_{'amb' if ambient else 'noamb'}_{alpha}"
                path = OUT_DIR / f"{tag}.png"
                _render(bg_color, ambient, alpha, path)
                image = cv2.imread(str(path))
                if image is None:
                    raise RuntimeError(f"render produced no frame at {path}")
                frames[(bg_name, ambient, alpha)] = image[:, :, ::-1].astype(np.float64)
                if not args.keep:
                    path.unlink(missing_ok=True)

    width = frames[(BACKDROPS[0][0], True, 1.0)].shape[1]
    scale = width / 12.34  # PREVIEW frame width in Algan units
    height = frames[(BACKDROPS[0][0], True, 1.0)].shape[0]

    def window(arm):
        cx = int(round(width / 2 + scale * _ARM_X[arm]))
        cy = height // 2
        half = int(round(scale * 1.35))
        return (slice(cy - half, cy + half), slice(cx - half, cx + half))

    failures = []
    print(
        f"{'arm':18s} {'backdrop':9s} {'alpha':>6s} {'dev mean':>9s} "
        f"{'dev max':>8s} {'eff alpha':>10s} {'ink':>7s} {'px':>6s}"
    )
    for arm in _ARMS:
        ambient = arm != "sphere_noambient"
        key_arm = "sphere" if arm == "sphere_noambient" else arm
        for bg_name, bg_byte, _ in BACKDROPS:
            opaque = frames[(bg_name, ambient, 1.0)][window(key_arm)]
            for alpha in ALPHAS[1:]:
                faded = frames[(bg_name, ambient, alpha)][window(key_arm)]
                dev_mean, dev_max, eff, ink, n = _score(opaque, faded, bg_byte, alpha)
                flag = ""
                if not np.isnan(dev_mean) and (
                    dev_mean > TOLERANCE
                    or abs(eff - alpha) > ALPHA_TOLERANCE
                    or abs(ink - alpha) > ALPHA_TOLERANCE
                ):
                    flag = "  <-- off"
                    failures.append((arm, bg_name, alpha, dev_mean, eff, ink))
                print(
                    f"{arm:18s} {bg_name:9s} {alpha:6.2f} {dev_mean:9.2f} "
                    f"{dev_max:8.2f} {eff:10.3f} {ink:7.3f} {n:6d}{flag}"
                )

    print()
    if failures:
        print(
            f"{len(failures)} arm/alpha combinations deviate from the alpha "
            f"composite (tolerance {TOLERANCE:g} channel values, "
            f"{ALPHA_TOLERANCE:g} alpha)."
        )
    else:
        print("every arm composites at its authored opacity.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
