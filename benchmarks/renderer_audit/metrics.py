"""Unit-free transport measurements on the calibration renders.

Every absolute brightness comparison between the two back ends is contaminated
by their different light-unit and colour-space conventions (see REPORT.md), so
the useful measurements are **ratios taken within one image**: what fraction of
a surface's brightness survives a reflection, or a trip through glass. Those
ratios cancel the units, so a difference between the back ends is a difference
in transport.

    <venv-python> benchmarks/renderer_audit/metrics.py
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent / "out"


def srgb_to_linear(u8):
    c = np.asarray(u8, dtype=np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def load_lin(name):
    im = cv2.imread(str(OUT / name), cv2.IMREAD_UNCHANGED)
    if im is None:
        return None
    return srgb_to_linear(im[:, :, :3][:, :, ::-1])


def disc_mask(shape, cx, cy, r0, r1):
    h, w = shape[:2]
    ys, xs = np.mgrid[0:h, 0:w]
    d = np.hypot(xs - cx, ys - cy)
    return (d >= r0) & (d < r1)


def glass_transmission(name):
    """calib_glass: mean linear RGB of the backdrop seen through the sphere,
    over the same seen directly. A clear ior-1.5 ball should pass most of the
    light that reaches it -- Fresnel takes ~4% at normal incidence, rising at
    the rim, and the rim also loses light to total internal reflection.
    """
    lin = load_lin(name)
    if lin is None:
        return None
    h, w = lin.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    # The sphere (r = 1.4 at z = +1, camera at z = 7, 40 deg vertical fov)
    # subtends an apparent radius of 0.33 * frame height. Sample well inside
    # that, and take the direct backdrop from an annulus safely outside it --
    # still on the blocks, whose corners reach 0.43 * h. The two regions must
    # not overlap: sampling "direct" pixels that are actually seen through the
    # glass makes the ratio track itself and hides whatever changed.
    inside = disc_mask(lin.shape, cx, cy, 0, 0.28 * h)
    direct = disc_mask(lin.shape, cx, cy, 0.36 * h, 0.41 * h)
    lit = lin.max(axis=2) > 0.01
    through = lin[inside & lit].mean() if (inside & lit).any() else 0.0
    straight = lin[direct & lit].mean() if (direct & lit).any() else 0.0
    return {
        "through_glass_mean_linear": round(float(through), 5),
        "direct_backdrop_mean_linear": round(float(straight), 5),
        "transmission_efficiency": round(float(through / max(straight, 1e-9)), 4),
    }


def mirror_reflection(name):
    """calib_mirror: the floor's own linear brightness, and the brightness of
    the floor's reflection in the lower half of each metal ball. The ratio is
    the reflection efficiency; for a metal it should be close to the metal's
    albedo (0.95 grey, 1.0/0.77/0.34 gold), reduced only by Fresnel's grazing
    behaviour and by roughness scattering light elsewhere.
    """
    lin = load_lin(name)
    if lin is None:
        return None
    h, w = lin.shape[:2]
    out = {}
    floor = lin[int(0.88 * h) : int(0.96 * h), int(0.40 * w) : int(0.60 * w)]
    out["floor_mean_linear"] = round(float(floor.mean()), 5)
    # Ball centres measured from the scene: r=1.2 at (-1.5,-0.2,0.6) and
    # (1.5,-0.2,0.6), camera (0,1.4,8) fov 40 -> roughly these image points.
    for label, fx in (("mirror", 0.34), ("gold_rough", 0.66)):
        cx, cy = fx * w, 0.60 * h
        patch = lin[
            int(cy + 0.03 * h) : int(cy + 0.09 * h),
            int(cx - 0.04 * w) : int(cx + 0.04 * w),
        ]
        out[f"{label}_lower_half_mean_linear"] = round(float(patch.mean()), 5)
        out[f"{label}_reflection_efficiency"] = round(
            float(patch.mean() / max(floor.mean(), 1e-9)), 4
        )
        # Spread of the reflected image: a mirror shows a sharp horizon (high
        # variance across the ball), a rough metal a blurred one (low variance).
        ball = lin[
            int(0.44 * h) : int(0.76 * h), int(cx - 0.09 * w) : int(cx + 0.09 * w)
        ]
        g = ball.mean(axis=2)
        out[f"{label}_ball_contrast"] = round(float(g.std() / max(g.mean(), 1e-9)), 4)
    return out


def main():
    report = {}
    for scene, fn in (
        ("calib_glass", glass_transmission),
        ("calib_mirror", mirror_reflection),
    ):
        report[scene] = {}
        for suffix in ("algan", "algan_glossy", "three_pathtrace", "three_raster"):
            res = fn(f"{scene}.{suffix}.png")
            if res is not None:
                report[scene][suffix] = res
    print(json.dumps(report, indent=2))
    (OUT / "metrics.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
