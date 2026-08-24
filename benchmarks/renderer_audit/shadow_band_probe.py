r"""Tell a soft shadow apart from a stack of hard ones.

An area light that is integrated over its emitting surface casts a **penumbra**:
brightness varies smoothly and monotonically from full light to full shadow. An
area light approximated by ``K`` point emitters, each carrying ``1/K`` of the
power, casts ``K`` hard shadows instead, and their union is a **staircase** --
the visible brightness can only take the ``K + 1`` values ``0/K, 1/K, ... K/K``,
one per number of emitters the receiver can still see.

Those two look similar at a glance and are trivially separable numerically, so
this reports:

* the **plateau levels** along a scanline through the shadow, normalised to the
  scanline's maximum. A staircase lands them on multiples of ``1/K``;
* **flatness**, the share of adjacent-pixel steps smaller than ``--eps``. A
  staircase is mostly flat with a few jumps; a true penumbra is mostly ramp;
* the **inferred emitter count** ``K``, the *smallest* small integer whose
  ``k/K`` grid the levels actually sit on. A smooth penumbra fits no such grid
  and is reported as fitting none, which is the point of the measurement.

The scanline is located from the *scene spec*: it runs horizontally, a given
number of sphere radii below the named object's centre, which is where a light
above the object throws its shadow.

Usage::

    <venv-python> benchmarks/renderer_audit/shadow_band_probe.py \
        scenes/calib_lights.json --object probe_rect \
        --images out/calib_lights.algan.png \
                 out/calib_lights.three_pathtrace.png \
        --labels algan three_pathtrace
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
    load_rgb,
    srgb_to_linear,
)


def _plateaus(row, eps, min_run):
    """Runs of near-constant value, as ``(level, length)`` pairs."""
    runs = []
    start = 0
    for i in range(1, len(row) + 1):
        if i == len(row) or abs(float(row[i]) - float(row[i - 1])) > eps:
            if i - start >= min_run:
                runs.append((float(np.median(row[start:i])), i - start))
            start = i
    return runs


def _group_levels(runs, total_len, tol=0.03, min_share=0.04):
    """Plateau levels that actually occupy the scanline.

    Runs are grouped by value and a group is kept only if it covers at least
    ``min_share`` of the scanline. Without that the smooth gradient of the light
    itself contributes a dozen one-off "levels" and swamps the shadow's own.
    """
    groups = []  # [level, total length]
    for lv, n in sorted(runs):
        if groups and lv - groups[-1][0] <= tol:
            g = groups[-1]
            g[0] = (g[0] * g[1] + lv * n) / (g[1] + n)
            g[1] += n
        else:
            groups.append([lv, n])
    return sorted(g[0] for g in groups if g[1] >= min_share * total_len)


def _infer_k(levels, kmax=8, tol=0.12):
    """The smallest ``K <= kmax`` whose ``k/K`` grid the levels sit on.

    K point emitters each carrying ``1/K`` of the power can only leave a
    receiver at ``k/K`` of full brightness, so the levels of a staircase
    identify the emitter count.

    The error is measured in units of ``k``, deliberately *not* divided by K: a
    per-K-normalised error is minimised by making K large, which fits any set of
    levels at all and reports a smooth penumbra as a hundred-emitter grid. The
    smallest K inside ``tol`` is taken, and ``kmax`` is small because a real
    emitter grid is ``k x k`` with k of 2 or 3 -- a ramp that only fits a large
    K is a ramp, not a staircase.
    """
    levels = [v for v in levels if v < 0.98]
    if len(levels) < 2:
        return None, None
    best, best_err = None, None
    for K in range(2, kmax + 1):
        err = max(abs(v * K - round(v * K)) for v in levels)
        if best_err is None or err < best_err:
            best, best_err = K, err
        if err < tol:
            return K, err
    return (None, best_err) if best_err is None or best_err >= tol else (best, best_err)


def measure(spec_path, image_paths, labels, obj_name, drop, span, eps, min_run):
    spec = json.loads(Path(spec_path).read_text())
    r = spec.get("render", {})
    proj = Projector(
        spec["camera"], int(r.get("width", 640)), int(r.get("height", 480))
    )
    obj = next((o for o in spec.get("objects", []) if o.get("name") == obj_name), None)
    if obj is None:
        raise SystemExit(f"no object named {obj_name!r} in the spec")
    px, py, _ = proj.project(obj["position"])
    rad = proj.sphere_radius_px(obj["position"], float(obj["geometry"]["radius"]))
    y = int(round(py + drop * rad))
    x0 = max(0, int(round(px - span * rad)))
    x1 = int(round(px + span * rad))
    print(
        f"object {obj_name!r}: centre ({px:.1f}, {py:.1f})  r={rad:.1f}px\n"
        f"scanline y={y}, x in [{x0}, {x1}]  "
        f"({drop} radii below centre, +-{span} radii wide)\n"
    )

    for path, label in zip(image_paths, labels):
        im = load_rgb(path)
        lin = srgb_to_linear(im).mean(axis=2)
        if not (0 <= y < lin.shape[0]):
            print(f"{label}: scanline off-frame")
            continue
        row = lin[y, x0 : min(x1, lin.shape[1])]
        peak = float(row.max())
        if peak <= 1e-9:
            print(f"{label}: scanline is black")
            continue
        norm = row / peak
        runs = _plateaus(norm, eps, min_run)
        merged = _group_levels(runs, len(norm))
        flat = float((np.abs(np.diff(norm)) < eps).mean())
        # Levels in the shadowed part only: above ~0.8 of the scanline maximum
        # what varies is the light's own falloff across the wall, not the
        # shadow, and it would otherwise be read as extra steps.
        shadow_levels = [v for v in merged if v < 0.8]
        K, err = _infer_k(shadow_levels)
        buckets = [round(float(c.mean()), 2) for c in np.array_split(norm, 24)]
        print(f"{label}:")
        print(f"  min {norm.min():.3f}   flatness {flat:.2f}")
        print("  profile (24 buckets, normalised to the scanline max):")
        print(f"    {buckets[:12]}")
        print(f"    {buckets[12:]}")
        print(f"  plateau levels below 0.8: {[round(v, 2) for v in shadow_levels]}")
        if K is not None and err is not None:
            print(
                f"  -> those sit on a k/{K} grid (error {err:.3f}): consistent "
                f"with {K} point emitters each carrying 1/{K} of the power"
            )
        elif len(shadow_levels) <= 1:
            print("  -> no intermediate levels: a hard edge, or no shadow here")
        else:
            print(
                "  -> no small-integer k/K grid fits these levels: a continuous "
                "penumbra, or a hard edge plus the light's own falloff"
            )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scene", type=Path)
    ap.add_argument("--images", type=Path, nargs="+", required=True)
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--object", required=True, help="object casting the shadow")
    ap.add_argument(
        "--drop", type=float, default=2.0, help="scanline offset in sphere radii"
    )
    ap.add_argument(
        "--span", type=float, default=4.0, help="scanline half-width in radii"
    )
    ap.add_argument("--eps", type=float, default=0.004)
    ap.add_argument("--min-run", type=int, default=4)
    args = ap.parse_args(argv)
    labels = args.labels or [p.name for p in args.images]
    if len(labels) != len(args.images):
        raise SystemExit("--labels must match --images")
    measure(
        args.scene,
        args.images,
        labels,
        args.object,
        args.drop,
        args.span,
        args.eps,
        args.min_run,
    )


if __name__ == "__main__":
    main()
