"""Interior-edge AA under the sheet resolve: the sheet_shade_split A/B.

The sheet resolve shades once per sheet at its dominant fragment, so a hard
crease -- two flat-shaded faces of one solid meeting inside a pixel: same
mesh id, same facing, no depth gap -- fuses into ONE sheet and the pixel
takes the dominant face's color outright. Along an interior (non-silhouette)
edge that is a per-pixel winner-take-all staircase where the deleted fragment
walk used to blend per fragment by exact area. ``sheet_shade_split`` keys the
compaction additionally by a flat-face shading class so crease faces become
sibling sheets, each shaded with its own normal.

This harness measures that, on a scene built to expose it: a TENT -- one
``TriangleTriangulated`` mob (one surface id), two planar faces meeting at a
ridge, rotated a few degrees in screen space so the crease line sweeps every
sub-pixel phase -- under a directional light that gives the two faces
distinct Lambert values, rendered LINEAR (``tonemapping=False``) so a crease
pixel's value is exactly ``a*cA + (1-a)*cB`` in its face coverage ``a``.

edge wobble (the primary metric, in px RMS)
    Per pixel column, the crease's sub-pixel row is recovered from the
    intensity profile (coverage integrates to position); a straight ridge
    must recover as a straight line. Winner-take-all shading quantizes the
    estimate to whole rows (sawtooth RMS ~0.29 px); analytic blending tracks
    the line to a few hundredths.

Also reported per arm: A/A byte-determinism, the compaction's fragment and
sheet totals (engagement -- the ON arm must compact MORE sheets, rule Y.1),
and the OFF-vs-ON moved-pixel population with its bounding box (the movement
must be confined to the crease rows, not the silhouettes).

``--scene icosa`` renders a lit Icosahedron (the Polyhedron zero-normal
class path, 20 facets of one surface) instead: no line metric, but the same
A/A, engagement and moved-population columns, plus PNGs for visual review.

Run:  <venv-python> benchmarks/_sheet_shade_split_check.py [--res ld|md]
      [--scene tent|icosa]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

os.environ.setdefault("ALGAN_USE_DAEMON", "0")

OUT = REPO / "algan_outputs" / "sheet_shade_split"


def build_tent():
    import torch

    from algan import (
        DARKER_GRAY,
        GRAY,
        ORIGIN,
        UP,
        WHITE,
        AmbientLight,
        DirectionalLight,
        MeshLambertMaterial,
        Off,
        Scene,
        TriangleTriangulated,
    )
    from algan import (
        OUT as OUTV,
    )

    Scene.set_background(DARKER_GRAY)
    with Off():
        # Both faces must land at DISTINCT, UNSATURATED values: a blend of
        # two clipped whites is invisible, and the first cut of this harness
        # measured exactly that (2 moved pixels on a fully blown-out tent).
        AmbientLight(color=WHITE, intensity=0.15).spawn(animate=False)
        DirectionalLight(
            location=UP * 6 + OUTV * 2,
            target=ORIGIN,
            color=WHITE,
            intensity=0.45,
        ).spawn(animate=False)
        # Two planar rectangles of ONE mob meeting at a ridge toward the
        # camera. Built axis-aligned (each face exactly planar, one geometric
        # normal) and rotated in-plane afterwards so the crease line crosses
        # pixel rows at a shallow slope.
        r1 = torch.tensor([-2.0, 0.0, 0.5])
        r2 = torch.tensor([2.0, 0.0, 0.5])
        t1 = torch.tensor([-2.0, 1.2, 0.0])
        t2 = torch.tensor([2.0, 1.2, 0.0])
        b1 = torch.tensor([-2.0, -1.2, 0.0])
        b2 = torch.tensor([2.0, -1.2, 0.0])
        corners = torch.stack([r1, r2, t2, r1, t2, t1, b1, b2, r2, b1, r2, r1]).view(
            4, 3, 3
        )
        tent = TriangleTriangulated(corners, color=GRAY)
        tent.set_material(MeshLambertMaterial(color=GRAY))
        tent.rotate(3, OUTV)
        tent.spawn(animate=False)


def build_icosa():
    from algan import (
        DARKER_GRAY,
        GOLD,
        ORIGIN,
        RIGHT,
        UP,
        WHITE,
        AmbientLight,
        DirectionalLight,
        Icosahedron,
        MeshPhongMaterial,
        Off,
        Scene,
    )
    from algan import (
        OUT as OUTV,
    )

    Scene.set_background(DARKER_GRAY)
    with Off():
        AmbientLight(color=WHITE, intensity=0.35).spawn(animate=False)
        DirectionalLight(
            location=RIGHT * 5 + UP * 6 + OUTV * 4,
            target=ORIGIN,
            color=WHITE,
            intensity=0.8,
        ).spawn(animate=False)
        ico = Icosahedron(edge_length=1.5).set_material(
            MeshPhongMaterial(color=GOLD, shininess=55)
        )
        ico.rotate(15, UP + RIGHT)
        ico.spawn(animate=False)


def edge_wobble(img):
    """RMS deviation (px) of the recovered crease row from its best-fit line.

    Per column: locate the strongest transition inside the central band,
    read the two plateaus, convert the window's intensities to coverage of
    the upper face, and integrate coverage into a sub-pixel edge row.
    """
    g = img[..., 1].astype(np.float64)
    h, w = g.shape
    y_lo, y_hi = int(h * 0.42), int(h * 0.58)
    half = 5
    xs, est = [], []
    for x in range(int(w * 0.38), int(w * 0.62)):
        col = g[:, x]
        grad = np.abs(col[y_lo + 1 : y_hi] - col[y_lo : y_hi - 1])
        y0 = y_lo + int(np.argmax(grad))
        above = float(np.median(col[y0 - 14 : y0 - 7]))
        below = float(np.median(col[y0 + 8 : y0 + 15]))
        if abs(above - below) < 8.0:
            continue
        win = col[y0 - half : y0 + half + 1]
        cov = np.clip((win - below) / (above - below), 0.0, 1.0)
        xs.append(float(x))
        est.append(float(y0 - half) + float(np.sum(cov)))
    xs = np.asarray(xs)
    est = np.asarray(est)
    if xs.size < 32:
        raise SystemExit(f"edge metric found only {xs.size} usable columns")
    fit = np.polyfit(xs, est, 1)
    resid = est - np.polyval(fit, xs)
    return float(np.sqrt(np.mean(resid**2))), float(np.abs(resid).max()), xs.size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--res", choices=("ld", "md"), default="ld")
    parser.add_argument("--scene", choices=("tent", "icosa"), default="tent")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    import cv2

    import algan.rendering.raytracing.sheets as sheets_mod
    from algan import LD, MD, SETTINGS, Scene

    # Engagement instrument (rule Y.1): record what the compaction produced.
    real_compact = sheets_mod.compact_sheets
    stats = []

    def recording_compact(coverage, *a, **k):
        out = real_compact(coverage, *a, **k)
        stats.append(
            (
                int(coverage["num_fragments"]),
                0 if out is None else int(out["num_sheets"]),
            )
        )
        return out

    sheets_mod.compact_sheets = recording_compact

    # Linear output: a crease pixel's value is then exactly the coverage
    # blend of its faces, so the edge metric inverts it without a LUT.
    SETTINGS.raytracing.set(tonemapping=False)

    (build_tent if args.scene == "tent" else build_icosa)()
    settings = MD if args.res == "md" else LD

    def render(tag):
        stats.clear()
        path = OUT / f"{args.scene}_{args.res}_{tag}.png"
        Scene.save_frame(str(path), settings, at=0.0, overwrite=True)
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        assert img is not None, path
        frags = sum(s[0] for s in stats)
        sheets = sum(s[1] for s in stats)
        return img, frags, sheets

    arms = {}
    for tag, flag in (("off", False), ("on", True)):
        SETTINGS.raytracing.experimental.set(sheet_shade_split=flag)
        img1, frags, sheets = render(tag)
        img2, frags2, sheets2 = render(f"{tag}_aa")
        identical = bool(np.array_equal(img1, img2))
        arms[tag] = (img1, frags, sheets)
        print(
            f"[{tag:>3}] fragments {frags:7d}  sheets {sheets:7d}  "
            f"A/A byte-identical: {identical}"
            + ("" if identical else "  <-- NONDETERMINISM")
        )
        assert frags2 == frags
        assert sheets2 == sheets

    (img_off, f_off, s_off), (img_on, f_on, s_on) = arms["off"], arms["on"]
    if f_on != f_off:
        print(f"WARNING: fragment totals differ off/on ({f_off} vs {f_on})")
    print(
        f"engagement: sheets {s_off} -> {s_on} "
        f"({'+' if s_on >= s_off else ''}{s_on - s_off}); "
        f"the ON arm must compact more sheets on a creased scene"
    )

    diff = np.abs(img_off.astype(np.int32) - img_on.astype(np.int32)).max(axis=2)
    moved = np.argwhere(diff > 0)
    if moved.size:
        (y0, x0), (y1, x1) = moved.min(axis=0), moved.max(axis=0)
        print(
            f"off vs on: {len(moved)} px moved, worst |d| {int(diff.max())}, "
            f"bbox rows {y0}..{y1} cols {x0}..{x1} "
            f"(image {img_off.shape[0]}x{img_off.shape[1]})"
        )
    else:
        print("off vs on: byte-identical -- the split changed NOTHING (not engaged?)")

    if args.scene == "tent":
        for tag, img in (("off", img_off), ("on", img_on)):
            rms, worst, ncols = edge_wobble(img)
            print(
                f"edge wobble [{tag:>3}]: rms {rms:.4f} px  worst {worst:.4f} px  "
                f"({ncols} columns)"
            )
    print(f"frames in {OUT}")


if __name__ == "__main__":
    main()
