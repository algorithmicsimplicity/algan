r"""Per-object measurements on a renderer-audit frame.

``compare.py`` answers "how different are these two images"; this answers "which
*object* differs, and in what way". It projects each object in the scene spec
into pixel space using the spec's own camera -- so the disc a measurement is
taken over is decided by the scene description, not by finding blobs in either
image -- and reports statistics per object per back end.

Every statistic is a ratio or a shape where it can be, because the two engines
disagree on light units (REPORT.md Sec. 2.1) and an absolute brightness
comparison mostly measures that.

    <venv-python> benchmarks/renderer_audit/material_probe.py \\
        scenes/materials_and_lighting.json --images out/materials_and_lighting.algan.png \\
        out/materials_and_lighting.three_raster.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def srgb_to_linear(u8):
    c = np.asarray(u8, dtype=np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def load_rgb(path):
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise SystemExit(f"cannot read {path}")
    return im[:, :, :3][:, :, ::-1].astype(np.uint8)


class Projector:
    """The spec camera as a pinhole, in the spec's own (three.js) frame.

    Both back ends are pinned to this camera -- ``calib_orient`` renders
    identically in the two engines, which is what licenses using one projection
    for both images.
    """

    def __init__(self, cam, width, height):
        self.w, self.h = width, height
        self.eye = np.array(cam["position"], dtype=np.float64)
        target = np.array(cam.get("target", (0, 0, 0)), dtype=np.float64)
        up = np.array(cam.get("up", (0, 1, 0)), dtype=np.float64)
        self.fwd = target - self.eye
        self.fwd /= np.linalg.norm(self.fwd)
        self.right = np.cross(self.fwd, up)
        self.right /= np.linalg.norm(self.right)
        self.up = np.cross(self.right, self.fwd)
        self.tan_half = math.tan(math.radians(cam.get("fov", 40.0)) * 0.5)

    def project(self, p):
        """World point -> (px, py, depth along the view axis)."""
        v = np.asarray(p, dtype=np.float64) - self.eye
        z = float(np.dot(v, self.fwd))
        x = float(np.dot(v, self.right))
        y = float(np.dot(v, self.up))
        half_h = z * self.tan_half
        half_w = half_h * (self.w / self.h)
        px = (x / half_w * 0.5 + 0.5) * self.w
        py = (0.5 - y / half_h * 0.5) * self.h
        return px, py, z

    def sphere_radius_px(self, centre, radius):
        """Screen radius of a sphere's silhouette, in pixels.

        The silhouette of a sphere is the cone tangent to it, whose half-angle
        is ``asin(r/d)`` -- not ``atan(r/d)``, and not the naive ``r/d``. At the
        sizes here the three differ by under a pixel, but the tangent cone is
        the right one and costs nothing.
        """
        d = np.linalg.norm(np.asarray(centre, dtype=np.float64) - self.eye)
        if d <= radius:
            return float(self.h)
        half_angle = math.asin(radius / d)
        # Distance to the silhouette circle's plane, along the view axis.
        _, _, z = self.project(centre)
        half_h = z * self.tan_half
        return math.tan(half_angle) * z / half_h * 0.5 * self.h


def disc_mask(shape, cx, cy, r_out, r_in=0.0):
    h, w = shape[:2]
    ys, xs = np.mgrid[0:h, 0:w]
    d = np.hypot(xs - cx, ys - cy)
    return (d < r_out) & (d >= r_in)


def _fmt(v):
    return "(" + ", ".join(f"{float(x):6.3f}" for x in np.atleast_1d(v)) + ")"


def measure(spec_path, image_paths, labels=None, fill=0.75, verbose=True):
    spec = json.loads(Path(spec_path).read_text())
    r = spec["render"]
    W, H = int(r["width"]), int(r["height"])
    proj = Projector(spec["camera"], W, H)

    images = []
    for p in image_paths:
        im = load_rgb(p)
        if im.shape[0] != H or im.shape[1] != W:
            raise SystemExit(f"{p}: {im.shape[1]}x{im.shape[0]} != spec {W}x{H}")
        images.append(im)
    if labels is None:
        labels = [Path(p).suffixes[-2].lstrip(".") for p in image_paths]

    rows = []
    for obj in spec.get("objects", []):
        geo = obj["geometry"]
        if geo["type"] != "sphere":
            continue
        centre = obj.get("position", (0, 0, 0))
        cx, cy, _ = proj.project(centre)
        r_px = proj.sphere_radius_px(centre, float(geo.get("radius", 1.0)))
        # Stay inside the silhouette: the outer few percent is antialiased
        # against the background in both engines and would mix it in.
        mask = disc_mask((H, W), cx, cy, r_px * fill)
        entry = {
            "name": obj.get("name", "?"),
            "material": obj.get("material", {}).get("type", "physical"),
            "centre_px": [round(cx, 1), round(cy, 1)],
            "radius_px": round(r_px, 1),
            "pixels": int(mask.sum()),
            "per_image": {},
        }
        for label, im in zip(labels, images):
            lin = srgb_to_linear(im)
            sel = lin[mask]
            u8 = im[mask].astype(np.float64)
            # The pixel nearest the disc centre, i.e. the camera-facing point.
            ci, cj = int(round(cy)), int(round(cx))
            entry["per_image"][label] = {
                "mean_u8": [round(float(v), 1) for v in u8.mean(axis=0)],
                "mean_linear": [round(float(v), 5) for v in sel.mean(axis=0)],
                "max_u8": [int(v) for v in im[mask].max(axis=0)],
                "centre_u8": [int(v) for v in im[ci, cj]],
                # Spread of the disc's luminance: a flat unlit ball is ~0, a
                # banded one is bimodal, a speckled one is high.
                "std_linear": round(float(sel.mean(axis=1).std()), 5),
            }
        rows.append(entry)

    if verbose:
        for e in rows:
            print(
                f"\n{e['name']:14s} [{e['material']}]  "
                f"centre {e['centre_px']}  r={e['radius_px']}px  n={e['pixels']}"
            )
            for label, m in e["per_image"].items():
                print(
                    f"  {label:16s} centre_u8 {str(m['centre_u8']):18s} "
                    f"mean_u8 {_fmt(m['mean_u8'])}  "
                    f"mean_lin {_fmt(m['mean_linear'])}  std {m['std_linear']:.4f}"
                )
    return rows


def band_count(spec_path, image_path, name, bins=64, floor=0.02):
    """Count the distinct plateaus in one object's disc.

    A cel shader with N bands puts its pixels into N clusters of the shading
    value. Histogram the disc's luminance and count clusters separated by an
    empty bin -- crude, but it distinguishes 2 bands from 4, which is the
    question.
    """
    spec = json.loads(Path(spec_path).read_text())
    r = spec["render"]
    W, H = int(r["width"]), int(r["height"])
    proj = Projector(spec["camera"], W, H)
    obj = next(o for o in spec["objects"] if o.get("name") == name)
    cx, cy, _ = proj.project(obj["position"])
    r_px = proj.sphere_radius_px(obj["position"], obj["geometry"]["radius"])
    im = load_rgb(image_path)
    lum = srgb_to_linear(im).mean(axis=2)
    vals = lum[disc_mask((H, W), cx, cy, r_px * 0.85)]
    hist, edges = np.histogram(vals, bins=bins, range=(0.0, max(vals.max(), 1e-6)))
    occupied = hist > (floor * hist.sum() / bins)
    clusters, prev = 0, False
    for o in occupied:
        if o and not prev:
            clusters += 1
        prev = o
    centres = []
    run = []
    for i, o in enumerate(occupied):
        if o:
            run.append(i)
        elif run:
            w = hist[run]
            centres.append(float((edges[run] * w).sum() / max(w.sum(), 1)))
            run = []
    if run:
        w = hist[run]
        centres.append(float((edges[run] * w).sum() / max(w.sum(), 1)))
    return clusters, [round(c, 4) for c in centres]


def radial_profile(
    spec_path, image_path, name, fractions=(0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95)
):
    """Mean linear luminance in annuli, normalised to the centre.

    This is the terminator-shape measurement REPORT.md Sec. 2.2 uses: it cancels
    the engines' light-unit difference and leaves the falloff's shape.
    """
    spec = json.loads(Path(spec_path).read_text())
    r = spec["render"]
    W, H = int(r["width"]), int(r["height"])
    proj = Projector(spec["camera"], W, H)
    obj = next(o for o in spec["objects"] if o.get("name") == name)
    cx, cy, _ = proj.project(obj["position"])
    r_px = proj.sphere_radius_px(obj["position"], obj["geometry"]["radius"])
    lum = srgb_to_linear(load_rgb(image_path)).mean(axis=2)
    out = []
    for f in fractions:
        lo, hi = max(f - 0.03, 0.0) * r_px, (f + 0.03) * r_px
        m = disc_mask((H, W), cx, cy, hi, lo)
        out.append(float(lum[m].mean()) if m.any() else float("nan"))
    base = out[0] if out and out[0] else 1.0
    return [round(v / base, 4) for v in out]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scene", type=Path)
    ap.add_argument("--images", type=Path, nargs="+", required=True)
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--fill", type=float, default=0.75)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    rows = measure(
        args.scene, args.images, args.labels, args.fill, verbose=not args.json
    )
    if args.json:
        print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
