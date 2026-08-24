r"""Measure what a solid actually transmits, against the answer theory forces.

Every other probe here compares Algan to three-gpu-pathtracer. This one does
not, and that is the point: when the two engines disagree about a transmissive
solid, the first question is which of them is wrong, and a reference renderer
cannot answer that about itself.

The scenes it reads (``calib_transmittance``, ``calib_transmittance_tinted``)
are built so the answer is forced rather than modelled:

* the backdrop is **unlit**, so its radiance is exactly its authored colour and
  no lighting model, ambient term or tonemap curve enters the ratio;
* there are **no lights at all**, so nothing is being shaded -- only
  transported;
* the ball sits on the optical axis, so a ray through its centre meets both
  interfaces at **normal incidence** and bends at neither. Its Fresnel
  reflectance is the closed form ``F = ((1 - n) / (1 + n))^2``, 0.04 at
  ior 1.5, at both crossings.

So the transmitted fraction at the centre must be ``(1 - F)^2``, times the
base colour once per interface (glTF's transmission BTDF multiplies by it on
each crossing, so a *solid* squares it). Nothing about the scene can move that
number, which makes any deviation a transport defect rather than a
disagreement about lighting.

What it catches that the engine-to-engine probes cannot:

* **The order of the tint.** A renderer that applies the albedo once, or three
  times, lands on a ratio that no amount of relighting explains. This is the
  measurement that settled ``REPORT.md`` section 9.3.1 -- Algan's mirror shows
  a transmissive ball's neighbours tinted twice, and the question was whether
  twice is right. It is.
* **A magnitude that only looks wrong.** Comparing absolute mirror-disc energy
  between engines is confounded: the path tracer ignores ``AmbientLight``
  entirely, so its sources are several times dimmer, and a ratio taken against
  them reads as a transport error that is not there.

Usage::

    <venv-python> benchmarks/renderer_audit/algan_render.py \
        benchmarks/renderer_audit/scenes/calib_transmittance.json \
        --out out --no-tonemap
    <venv-python> benchmarks/renderer_audit/transmittance_probe.py \
        scenes/calib_transmittance.json --image out/calib_transmittance.algan.png

Pass ``--image`` more than once to compare several renders of the same scene.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from material_probe import (  # noqa: E402  (path set above)
    load_rgb,
    srgb_to_linear,
)

#: Half-width of the sampled patch, in pixels. Small enough that every ray in
#: it crosses the ball near enough to normal incidence for the closed form to
#: hold (the Fresnel term is flat to 4 decimals well past this), and large
#: enough to average out the last bit of sampling noise.
PATCH = 4


def _linear_albedo(spec, name):
    for o in spec.get("objects", []):
        if o.get("name") == name:
            c = (o.get("material") or {}).get("color", [1.0, 1.0, 1.0])
            return srgb_to_linear(np.asarray(c, dtype=np.float64) * 255.0)
    raise SystemExit(f"no object named {name!r} in the spec")


def _ior(spec, name):
    for o in spec.get("objects", []):
        if o.get("name") == name:
            return float((o.get("material") or {}).get("ior", 1.5))
    raise SystemExit(f"no object named {name!r} in the spec")


def measure(spec_path, image_paths, labels, ball="ball"):
    spec = json.loads(Path(spec_path).read_text())
    albedo = _linear_albedo(spec, ball)
    ior = _ior(spec, ball)
    fresnel = ((1.0 - ior) / (1.0 + ior)) ** 2
    predicted = (1.0 - fresnel) ** 2 * albedo**2

    print(
        f"ball {ball!r}: ior {ior}, albedo linear "
        f"({albedo[0]:.3f}, {albedo[1]:.3f}, {albedo[2]:.3f})"
    )
    print(f"  Fresnel at normal incidence  F = {fresnel:.4f}")
    print(
        f"  so the centre must transmit  (1-F)^2 * albedo^2 = "
        f"({predicted[0]:.4f}, {predicted[1]:.4f}, {predicted[2]:.4f})"
    )
    print()

    for path, label in zip(image_paths, labels):
        lin = srgb_to_linear(load_rgb(path))
        h, w, _ = lin.shape
        cy, cx = h // 2, w // 2
        through = (
            lin[cy - PATCH : cy + PATCH + 1, cx - PATCH : cx + PATCH + 1]
            .reshape(-1, 3)
            .mean(axis=0)
        )
        # The bare backdrop, sampled at the same height well clear of the ball.
        direct = (
            lin[cy - PATCH : cy + PATCH + 1, 20 : 20 + 2 * PATCH + 1]
            .reshape(-1, 3)
            .mean(axis=0)
        )
        if (direct <= 1e-6).any():
            print(
                f"{label}: the backdrop sample is black -- is the ball "
                f"filling the frame, or the render empty?"
            )
            continue
        ratio = through / direct
        err = ratio / np.maximum(predicted, 1e-12)
        print(
            f"{label:16s} backdrop ({direct[0]:.4f},{direct[1]:.4f},{direct[2]:.4f})"
            f"  through ({through[0]:.4f},{through[1]:.4f},{through[2]:.4f})"
        )
        print(
            f"{'':16s} measured/predicted  "
            f"({err[0]:.4f}, {err[1]:.4f}, {err[2]:.4f})"
            f"   worst deviation {float(np.abs(err - 1.0).max()) * 100:.2f}%"
        )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scene", type=Path)
    ap.add_argument("--image", dest="images", type=Path, action="append", required=True)
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument(
        "--ball", default="ball", help="object name of the transmissive solid"
    )
    args = ap.parse_args(argv)
    labels = args.labels or [p.name for p in args.images]
    if len(labels) != len(args.images):
        raise SystemExit("--labels must match --image")
    measure(args.scene, args.images, labels, args.ball)


if __name__ == "__main__":
    main()
