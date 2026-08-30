"""Acceptance harness for animatable ``Light.intensity``.

Two questions the unit tests cannot answer, because both are about pixels:

1. **Does an animated intensity actually change the image?** A light's intensity
   can be recorded on the timeline, materialize to the right per-frame values and
   still reach the renderer as a constant -- the snapshot is several hops from
   the frame buffer. This renders a ramp and reads the brightness back off the
   frames.
2. **Is a constant intensity still worth exactly what it was worth before?** The
   arm that must not move. A control scene holds intensity fixed and asserts the
   frames are flat, which is the pixel-side statement of the byte-identity
   argument.

Run one arm per process (the renderer's ``ti.static`` gates are resolved at first
compile), and with ``ALGAN_USE_DAEMON=0`` so a warm daemon cannot serve a stale
module::

    ALGAN_USE_DAEMON = 0 < venv - python > benchmarks / _light_intensity_check.py

Exits non-zero and prints which check failed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from algan import (
    OUT,
    SMOKE_TEST,
    UP,
    WHITE,
    AmbientLight,
    Off,
    PointLight,
    Scene,
    Seq,
    Sphere,
    rate_funcs,
)

OUT_DIR = Path(__file__).resolve().parent / "_light_intensity_check_out"


def _frame_means(paths):
    """Mean pixel value of each rendered still, in author order."""
    import cv2

    means = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            raise SystemExit(f"could not read back {path}")
        means.append(float(image.mean()))
    return means


def _render_stills(name, animate_intensity, times):
    """Render ``times`` as stills from a one-sphere scene, return their means."""
    with Scene() as scene:
        with Off():
            scene.clear_lights()
            AmbientLight(color=WHITE, intensity=0.05).spawn()
            key = PointLight(location=UP * 3 + OUT * 4, color=WHITE, intensity=0.5)
            key.spawn()
            Sphere(radius=1.1, color=WHITE).spawn()

        if animate_intensity:
            with Seq(duration=2, rate_func=rate_funcs.identity):
                key.intensity = 4.0
        else:
            # Same recorded extent, so the same frame times mean the same thing
            # in both arms.
            scene.wait(2)

        paths = []
        for i, t in enumerate(times):
            result = scene.save_frame(
                OUT_DIR / f"{name}_{i}.png", SMOKE_TEST, at=t, overwrite=True
            )
            paths.append(Path(result.output_path))
    return _frame_means(paths)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    times = [0.0, 0.5, 1.0, 1.5, 1.99]
    failures = []

    ramp = _render_stills("ramp", True, times)
    print("animated intensity 0.5 -> 4.0, mean pixel value per frame:")
    for t, m in zip(times, ramp):
        print(f"  t={t:5.2f}  {m:8.3f}")

    # The ramp must be strictly increasing. Monotonicity is the claim, not a
    # particular slope: exposure, tonemapping and the sphere's own falloff all
    # sit between the light's intensity and the byte written to the frame.
    increments = np.diff(ramp)
    if not (increments > 0).all():
        failures.append(
            f"animated intensity did not brighten the frames monotonically: {ramp}"
        )
    # And it must move by a lot -- a mechanism that engages but moves two channel
    # values would pass a monotonicity test and still be broken.
    if ramp[-1] - ramp[0] < 5.0:
        failures.append(
            f"animated intensity moved the mean by only {ramp[-1] - ramp[0]:.3f} "
            "channel values over an 8x ramp; the value is probably not reaching "
            "the renderer per frame"
        )

    control = _render_stills("control", False, times)
    print("\nconstant intensity 0.5, mean pixel value per frame:")
    for t, m in zip(times, control):
        print(f"  t={t:5.2f}  {m:8.3f}")

    # A constant intensity must be flat: nothing in the scene moves, so any
    # variation means the per-frame read introduced one.
    spread = max(control) - min(control)
    if spread > 1e-6:
        failures.append(
            f"constant intensity varied across frames by {spread:.6f}; the "
            "per-frame read is not returning the authored constant"
        )
    # The control's first frame must equal the ramp's first frame: both author
    # intensity 0.5 and the ramp has not moved yet at t=0.
    if abs(control[0] - ramp[0]) > 1e-6:
        failures.append(
            f"t=0 differs between the animated and constant arms "
            f"({ramp[0]:.6f} vs {control[0]:.6f}); an animation that has not "
            "started yet must render as its start value"
        )

    print()
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        return 1
    print("PASS: intensity animates the image, and a constant one is flat.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
