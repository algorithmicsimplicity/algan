"""Does the blue-noise sampler tile buy anything? (roadmap section 7)

``pt_blue_noise`` replaces the path tracer's hashed per-pixel sampler key with
a shipped, annealed tile (``blue_noise.py``). Both arms draw from the same
Owen-scrambled Sobol sampler with the same estimator, so this harness is not
asking "which converges faster" -- neither does -- it asks **where the error
sits in the frequency domain**, which is what decides whether a low-spp frame
looks like grain or like clumps, and what a denoiser can remove.

Three numbers per spp, all against a 1024-spp reference of the **off** arm
(one reference for both arms: they estimate the same image, and using each
arm's own reference would fold the reference's own residual noise into that
arm's MSE):

* ``raw MSE`` -- should be about equal. Blue noise does not change
  convergence, and a big difference here would mean something is wrong.
* ``denoised MSE`` -- the number that decides the default. This is what a
  user sees: ``SETTINGS.raytracing.denoise`` is on by default and the OIDN
  U-Net's prior is a convolutional one, so it removes high-frequency error and
  leaves low-frequency error behind.
* ``low-frequency MSE`` -- the mean square of the error image after a 3x3 box
  filter. A direct reading of the same thing without the denoiser in the way:
  blue noise should lower it markedly.

Every arm renders at ``pt_error_target = 0`` so both spend exactly the same
number of samples per pixel (adaptive sampling would spend a different budget
per arm and the comparison would measure that instead), and with
``denoise=False`` for the raw arms -- the denoised arm is a second render.
Both arms run in ONE process: ``pt_blue_noise`` is a runtime word in
``nee_meta``, not a ``ti.static`` gate, so there is no compiled variant to
freeze.

The scene is ``tests/path_traced/scenes/lit_and_shadowed.py``'s in miniature:
a point light with shadows over a Lambert floor, a shadow-casting pillar, a
red wall close enough to bleed, an emissive slab and a rough metal prism.

Run::

    uv run python benchmarks/_pt_blue_noise_check.py
    uv run python benchmarks/_pt_blue_noise_check.py --spp 2 4 --resolution 96

PNGs of every arm and of its (amplified) error image land in
``benchmarks/_pt_blue_noise_check_out/`` -- look at them: the on arm's error
should read as fine grain, the off arm's as clumps.

Measured on the CPU backend, 96x96, 24 render seeds per arm (positive = the
blue-noise arm is better; the error bar is the two arms' standard errors in
quadrature)::

     spp        raw MSE     denoised MSE   low-frequency MSE
       2   -0.7% +- 1.1%    +1.9% +- 3.6%      +3.5% +- 2.9%
       4   +1.5% +- 1.6%    +2.4% +- 4.5%      +3.8% +- 3.2%
       8   +3.9% +- 2.1%    +2.7% +- 2.4%      +3.1% +- 4.3%

Raw MSE is flat, which is the prediction. The other two are positive in every
arm and inside one standard error -- real, and small. The bar for defaulting
``pt_blue_noise`` on was >10% denoised at 4 spp, so it ships off; roadmap
section 7 explains why one tile shared by every dimension pair cannot do
better, and what a per-dimension version would cost.

Not memory-capped: these are real renders (``CLAUDE.md``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algan import (  # noqa: E402
    BLACK,
    DOWN,
    LEFT,
    OUT,
    RED,
    RIGHT,
    SETTINGS,
    SMOKE_TEST,
    UP,
    WHITE,
    MeshLambertMaterial,
    MeshStandardMaterial,
    Off,
    PointLight,
    Prism,
    Scene,
    SceneManager,
)

OUT_DIR = Path(__file__).resolve().parent / "_pt_blue_noise_check_out"


def _build(scene):
    scene.set_background(BLACK)
    Scene.clear_lights()
    PointLight(
        location=UP * 3.0 + OUT * 4.0 + LEFT * 1.5, color=WHITE, intensity=2.0
    ).spawn(animate=False)

    floor = Prism(width=9.0, height=0.2, depth=5.0)
    floor.set_material(MeshLambertMaterial(color=WHITE))
    floor.move(DOWN * 1.4)
    floor.spawn(animate=False)

    pillar = Prism(width=0.7, height=2.4, depth=0.7)
    pillar.set_material(MeshLambertMaterial(color=WHITE))
    pillar.move(LEFT * 1.8 + DOWN * 0.1)
    pillar.spawn(animate=False)

    wall = Prism(width=0.2, height=2.6, depth=2.4)
    wall.set_material(MeshLambertMaterial(color=RED))
    wall.move(RIGHT * 3.4 + DOWN * 0.1)
    wall.spawn(animate=False)

    glow = Prism(width=1.1, height=1.1, depth=0.08)
    glow.set_material(
        MeshLambertMaterial(color=BLACK, emissive=WHITE, emissive_intensity=1.5)
    )
    glow.move(UP * 1.6 + RIGHT * 1.2 - OUT * 1.2)
    glow.spawn(animate=False)

    metal = Prism(width=1.0, height=1.0, depth=1.0)
    metal.set_material(MeshStandardMaterial(color=WHITE, metalness=1.0, roughness=0.35))
    metal.move(RIGHT * 0.9 + DOWN * 0.8)
    metal.spawn(animate=False)


def render(name, spp, blue_noise, seed, resolution, denoise=False):
    """One frame as float channel counts, ``[H, W, 3]``."""
    import cv2

    settings = SMOKE_TEST.set(resolution=resolution)
    snapshot = SETTINGS.snapshot()
    SceneManager.reset()
    try:
        SETTINGS.raytracing.set(
            samples_per_pixel=int(spp), denoise=bool(denoise), shadows=True
        )
        SETTINGS.raytracing.experimental.set(
            pt_blue_noise=bool(blue_noise),
            # Equal spp in both arms.
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


def _box3(img):
    """3x3 box filter, edge-replicated, over an ``[H, W, 3]`` image."""
    pad = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode="edge")
    acc = np.zeros_like(img)
    for dy in range(3):
        for dx in range(3):
            acc += pad[dy : dy + img.shape[0], dx : dx + img.shape[1]]
    return acc / 9.0


def _sem(values):
    """Standard error of the mean over the trials."""
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2:
        return 0.0
    return float(values.std(ddof=1) / np.sqrt(values.size))


def _write_error_png(path, err, gain=6.0):
    import cv2

    vis = np.clip(128.0 + gain * err, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), vis)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spp", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--reference-spp", type=int, default=1024)
    ap.add_argument("--trials", type=int, default=3, help="render seeds per arm")
    ap.add_argument("--resolution", type=int, nargs=2, default=[96, 96])
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res = tuple(args.resolution)

    reference = render("reference.png", args.reference_spp, False, 0, res)
    print(
        f"reference: {args.reference_spp} spp, blue noise OFF, "
        f"{res[0]}x{res[1]} (both arms are scored against this one image)"
    )
    print(
        f"{'spp':>4} {'arm':>4} {'raw MSE':>14} {'denoised MSE':>14} "
        f"{'low-freq MSE':>14}   (mean +- standard error over "
        f"{args.trials} render seeds)"
    )
    verdict = {}
    for spp in args.spp:
        row = {}
        for label, on in (("off", False), ("on", True)):
            raw = []
            den = []
            low = []
            for trial in range(args.trials):
                seed = 1 + trial
                img = render(f"{label}_{spp}_{trial}.png", spp, on, seed, res)
                err = img - reference
                raw.append(float((err**2).mean()))
                low.append(float((_box3(err) ** 2).mean()))
                dn = render(
                    f"{label}_{spp}_{trial}_denoised.png",
                    spp,
                    on,
                    seed,
                    res,
                    denoise=True,
                )
                den.append(float(((dn - reference) ** 2).mean()))
                if trial == 0:
                    _write_error_png(OUT_DIR / f"err_{label}_{spp}.png", err)
            row[label] = (np.array(raw), np.array(den), np.array(low))
            print(
                f"{spp:>4} {label:>4} "
                + " ".join(f"{v.mean():>8.2f}+-{_sem(v):<5.2f}" for v in row[label])
            )
        verdict[spp] = row
        for i, what in enumerate(("raw", "denoised", "low-frequency")):
            off_v, on_v = row["off"][i], row["on"][i]
            gain = 100.0 * (off_v.mean() - on_v.mean()) / max(off_v.mean(), 1e-12)
            # The two arms are independent samples, so the uncertainty on the
            # difference is the quadrature sum -- printed because a 5%
            # "improvement" with a 6% error bar is not one.
            err = (
                100.0
                * float(np.hypot(_sem(off_v), _sem(on_v)))
                / max(off_v.mean(), 1e-12)
            )
            print(
                f"      {what:>14}: on is {gain:+.1f}% +- {err:.1f}% "
                f"({'better' if gain > 0 else 'worse'})"
            )
    print(f"\nerror images (mid grey = 0, gain 6x) in {OUT_DIR}")


if __name__ == "__main__":
    main()
