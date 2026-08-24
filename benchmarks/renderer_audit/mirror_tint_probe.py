r"""Measure what a mirror in a scene is actually reflecting.

The question this answers is the one ``compare.py`` cannot: when two renderers
disagree about a mirror, is the difference *how much* light it returns, or
*which lobe* that light came from? Total energy answers the first; the colour
ratio answers the second, because a dielectric's Fresnel reflection is
achromatic while light that has passed *through* a tinted solid carries that
solid's albedo once per interface it crossed.

So for each image it reports, over the mirror object's disc:

* the **total linear energy**, which says whether one engine is returning more
  light than the other at all;
* the **tint ratios** ``g/r`` and ``g/b``, printed beside the reflected object's
  albedo and albedo squared -- landing on the albedo means one tinting, on
  albedo squared means two (in and out of a solid), and on 1.0 means an
  untinted specular reflection;
* the **concentration**, the share of the disc's energy held by its brightest
  2% of pixels, which separates a tight specular highlight from a broad patch
  carrying the same energy.

Discs are located by projecting the *scene spec* through the spec's own camera
(the same projection ``material_probe.py`` uses), so the region a number is
taken over is decided by the scene description rather than by finding blobs.

Run it against a black-background copy of the scene where you can: with a flat
background Algan's escaped secondary rays return the background colour, which
lands on the mirror as a constant and dilutes every ratio below (see
``REPORT.md`` section 9.4).

Usage::

    <venv-python> benchmarks/renderer_audit/mirror_tint_probe.py \
        scenes/matlight_pbr_subset.json \
        --images out/matlight_pbr_subset.algan.png \
                 out/matlight_pbr_subset.three_pathtrace.png \
        --labels algan three_pathtrace --mirror mirror --source glass
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from material_probe import (  # noqa: E402  (path set above)
    Projector,
    disc_mask,
    load_rgb,
    srgb_to_linear,
)


def _albedo_linear(spec, name):
    """The named object's authored colour, decoded to linear light."""
    for o in spec.get("objects", []):
        if o.get("name") == name:
            c = (o.get("material") or {}).get("color", [1.0, 1.0, 1.0])
            return srgb_to_linear(np.asarray(c, dtype=np.float64) * 255.0)
    raise SystemExit(f"no object named {name!r} in the spec")


def _find(spec, name):
    for o in spec.get("objects", []):
        if o.get("name") == name:
            return o
    raise SystemExit(f"no object named {name!r} in the spec")


def _ratios(total):
    r, g, b = (float(v) for v in total)
    return g / max(r, 1e-12), g / max(b, 1e-12)


def measure(spec_path, image_paths, labels, mirror_name, source_name, fill=1.0):
    spec = json.loads(Path(spec_path).read_text())
    r = spec.get("render", {})
    proj = Projector(
        spec["camera"], int(r.get("width", 640)), int(r.get("height", 480))
    )

    obj = _find(spec, mirror_name)
    centre = obj["position"]
    px, py, _ = proj.project(centre)
    geom = obj["geometry"]
    if geom["type"] != "sphere":
        raise SystemExit("mirror_tint_probe expects a sphere mirror")
    rad = proj.sphere_radius_px(centre, float(geom.get("radius", 1.0))) * fill

    print(f"mirror {mirror_name!r}: centre ({px:.1f}, {py:.1f})  r={rad:.1f}px")
    if source_name:
        alb = _albedo_linear(spec, source_name)
        gr, gb = _ratios(alb)
        gr2, gb2 = _ratios(alb * alb)
        print(
            f"reflected object {source_name!r}: albedo linear "
            f"({alb[0]:.3f}, {alb[1]:.3f}, {alb[2]:.3f})"
        )
        print(f"  one tinting  (albedo)    predicts g/r {gr:.2f}  g/b {gb:.2f}")
        print(f"  two tintings (albedo^2)  predicts g/r {gr2:.2f}  g/b {gb2:.2f}")
        print("  an untinted specular reflection predicts g/r 1.00  g/b 1.00")
    print()

    for path, label in zip(image_paths, labels):
        im = load_rgb(path)
        lin = srgb_to_linear(im)
        mask = disc_mask(lin.shape, px, py, rad)
        d = lin[mask]
        if not d.size:
            print(f"{label}: disc is off-frame")
            continue
        total = d.sum(axis=0)
        gr, gb = _ratios(total)
        lum = d.sum(axis=1)
        k = max(1, int(round(0.02 * lum.size)))
        conc = float(np.sort(lum)[-k:].sum() / max(lum.sum(), 1e-12))
        print(
            f"{label:16s} total ({total[0]:8.3f},{total[1]:8.3f},{total[2]:8.3f})"
            f"  g/r {gr:5.2f}  g/b {gb:5.2f}"
            f"  top-2% holds {conc * 100:5.1f}%"
            f"  mean_u8 {tuple(int(round(v)) for v in im[mask].mean(axis=0))}"
        )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scene", type=Path)
    ap.add_argument("--images", type=Path, nargs="+", required=True)
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--mirror", default="mirror", help="object name of the mirror")
    ap.add_argument(
        "--source",
        default=None,
        help="object whose albedo the reflection should be compared against",
    )
    ap.add_argument(
        "--fill",
        type=float,
        default=1.0,
        help="fraction of the silhouette radius to measure over",
    )
    args = ap.parse_args(argv)
    labels = args.labels or [p.name for p in args.images]
    if len(labels) != len(args.images):
        raise SystemExit("--labels must match --images")
    measure(args.scene, args.images, labels, args.mirror, args.source, args.fill)


if __name__ == "__main__":
    main()
