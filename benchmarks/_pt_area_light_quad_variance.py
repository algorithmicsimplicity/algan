"""Equal-spp variance of a ``RectAreaLight``: emissive quad vs packed rows.

Roadmap section 6a-ter turns a ``RectAreaLight`` into two emissive triangles
for the path tracer instead of ``K = k*k`` packed cell rows. Both arms are
unbiased estimators of the same emitter, so the interesting question is not
"do they agree" (``tests/unit_tests/test_path_tracer.py`` answers that) but
**which one is quieter at equal sample count** -- the quad arm gains MIS
against BSDF continuations and loses nothing, so it should not be worse, and
it should be markedly better on the glossy sphere where a BSDF ray finds the
emitter and a next-event sample toward a near-delta lobe almost never does.

The scene is the one the roadmap names: one ``samples = 16`` ``RectAreaLight``
over a Lambert floor with a smooth metal sphere on it.

Both arms run in ONE process: ``pt_area_light_quads`` is read host-side at
render time with no ``ti.static`` gate behind it, so there is no compiled
variant to freeze (contrast ``_area_light_shadow_check.py``, whose flag is
frozen at import). Adaptive sampling is turned off in every arm --
``pt_error_target`` would spend a different number of samples per pixel per
arm and the comparison would stop being equal-spp.

Run::

    uv run python benchmarks/_pt_area_light_quad_variance.py
    uv run python benchmarks/_pt_area_light_quad_variance.py --spp 32 --trials 6

Measured on the CPU backend at the defaults (64x64, 16 spp per arm, 4 seeds,
1024-spp reference)::

    reference: mean abs difference between the two arms 0.807 counts
     rows arm, 16 spp: MSE 320.29   (318.12, 326.99, 327.67, 308.39)
    quads arm, 16 spp: MSE 153.18   (139.94, 180.33, 138.88, 153.56)
    quad arm is 2.09x better in MSE

The 0.8-count gap between the two 1024-spp references is the bias bound: they
are two estimators of one emitter and they agree to well under a channel
count, so the 2.09x is variance and nothing else.

Not memory-capped: these are real renders, and ``CLAUDE.md`` says a cap on a
real render segfaults inside the GPU allocator rather than raising.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algan import (  # noqa: E402
    BLACK,
    OUT,
    SETTINGS,
    SMOKE_TEST,
    UP,
    WHITE,
    MeshLambertMaterial,
    MeshStandardMaterial,
    Off,
    Prism,
    RectAreaLight,
    Scene,
    SceneManager,
    Sphere,
)

OUT_DIR = Path(__file__).resolve().parent / "_pt_area_light_quad_variance_out"


def _build(scene):
    scene.set_background(BLACK)
    Scene.clear_lights()
    floor = Prism(width=9.0, height=9.0, depth=0.2)
    floor.set_material(MeshLambertMaterial(color=WHITE))
    floor.spawn(animate=False)
    ball = Sphere(radius=1.1)
    ball.set_material(MeshStandardMaterial(color=WHITE, metalness=1.0, roughness=0.08))
    ball.move(OUT * 1.2)
    ball.spawn(animate=False)
    RectAreaLight(
        location=OUT * 4.0 + UP * 1.5,
        width=2.5,
        height=2.5,
        samples=16,
        color=WHITE,
        intensity=8.0,
    ).spawn(animate=False)


def render(name, spp, quads, seed, resolution):
    """One frame, returned as a float array of 8-bit channel counts."""
    import cv2

    settings = SMOKE_TEST.set(resolution=resolution)
    snapshot = SETTINGS.snapshot()
    SceneManager.reset()
    try:
        SETTINGS.raytracing.set(
            samples_per_pixel=int(spp),
            denoise=False,
            shadows=True,
            linear_color_space=False,
            tonemapping=False,
        )
        SETTINGS.raytracing.experimental.set(
            post_process_tonemap=False,
            pt_area_light_quads=bool(quads),
            # Equal spp in both arms: adaptive stopping would spend a
            # different budget per pixel and the MSE would measure that.
            pt_error_target=0.0,
            pt_seed=int(seed),
        )
        with Scene(video_settings=settings) as scene:
            with Off():
                _build(scene)
            result = scene.save_frame(
                OUT_DIR / name, video_settings=settings, overwrite=True
            )
    finally:
        SceneManager.reset()
        SETTINGS.restore(snapshot)
    frame = cv2.imread(str(result.output_path), cv2.IMREAD_UNCHANGED)
    return frame.astype(np.float64)[..., :3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spp", type=int, default=16, help="samples of each arm")
    parser.add_argument(
        "--reference-spp", type=int, default=1024, help="samples of the reference"
    )
    parser.add_argument("--trials", type=int, default=4, help="seeds per arm")
    parser.add_argument("--resolution", type=int, default=64)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res = (args.resolution, args.resolution)

    ref_quads = render("ref_quads.png", args.reference_spp, True, 0, res)
    ref_rows = render("ref_rows.png", args.reference_spp, False, 0, res)
    bias = float(np.abs(ref_quads - ref_rows).mean())
    reference = 0.5 * (ref_quads + ref_rows)
    print(
        f"reference {args.reference_spp} spp at {args.resolution}x{args.resolution}: "
        f"mean abs difference between the two arms {bias:.3f} counts "
        f"(they estimate the same emitter, so this bounds any bias)"
    )

    results = {}
    for label, quads in (("rows", False), ("quads", True)):
        mses = []
        for trial in range(args.trials):
            img = render(f"{label}_{trial}.png", args.spp, quads, 1 + trial, res)
            mses.append(float(((img - reference) ** 2).mean()))
        results[label] = mses
        print(
            f"{label:>5} arm, {args.spp} spp: MSE "
            f"{statistics.mean(mses):.2f} "
            f"(per trial: {', '.join(f'{m:.2f}' for m in mses)})"
        )
    ratio = statistics.mean(results["rows"]) / max(
        statistics.mean(results["quads"]), 1e-12
    )
    print(f"quad arm is {ratio:.2f}x {'better' if ratio > 1 else 'worse'} in MSE")


if __name__ == "__main__":
    main()
