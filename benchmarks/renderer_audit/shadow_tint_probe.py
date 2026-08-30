"""Measure the colour of the shadow a transmissive sphere casts.

The experiment behind ``REPORT.md`` §4.10. A shadow query that carries one
scalar per light cannot tint what it passes, so light through green glass
arrives grey; with an RGB payload it arrives green. This reads that off a
rendered frame as a number instead of an impression.

For each sphere in a scene spec it projects the sphere's centre along the
directional light onto the backdrop, samples a small disc there, and reports
the mean **linear** RGB as a fraction of the open backdrop in the same image.
That ratio is the transmittance the shadow ray delivered, and it cancels the
two engines' light-unit convention (``REPORT.md`` §2.1) the same way §4.6's
measurement does -- so an Algan frame and a Three.js frame are directly
comparable.

Usage::

    <venv-python> benchmarks/renderer_audit/shadow_tint_probe.py \
        scenes/calib_absorption.json out/calib_absorption.algan.png [more.png ...]

Prints one row per (image, sphere). ``--json`` emits the same data as one
object per row for scripting.

Nothing here is a test and nothing is baselined: it exists to be looked at.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent


def _srgb_to_linear(u8: np.ndarray) -> np.ndarray:
    """The sRGB EOTF, on 0..255 bytes. Both engines write the OETF at the byte
    write (``REPORT.md`` §1), so this is what undoes it.
    """
    c = u8.astype(np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _norm(v):
    n = math.sqrt(sum(c * c for c in v))
    return tuple(c / n for c in v) if n > 0 else v


def _backdrop_front_z(spec):
    """The z of the face the shadows land on: the backdrop box's front face."""
    for obj in spec["objects"]:
        geom = obj.get("geometry", {})
        if geom.get("type") == "box":
            size = geom.get("size", [1, 1, 1])
            return float(obj["position"][2]) + float(size[2]) / 2.0
    raise SystemExit("scene has no box to act as a backdrop")


def _project(spec, width, height):
    """Pixel centre of each sphere's shadow on the backdrop, plus its pixel
    radius. Pinhole camera looking down -Z from ``camera.position``; the spec's
    ``fov`` is the vertical field of view in degrees (``SPEC.md``).
    """
    cam = spec["camera"]
    cz = float(cam["position"][2])
    fov = math.radians(float(cam["fov"]))
    light = None
    for lt in spec.get("lights", []):
        if lt.get("type") == "directional":
            light = _norm(tuple(float(c) for c in lt["direction"]))
            break
    if light is None:
        raise SystemExit("scene has no directional light to cast the shadow")
    back_z = _backdrop_front_z(spec)

    out = []
    for obj in spec["objects"]:
        geom = obj.get("geometry", {})
        if geom.get("type") != "sphere":
            continue
        pos = [float(c) for c in obj["position"]]
        radius = float(geom.get("radius", 1.0))
        # March the sphere centre along the light until it reaches the backdrop.
        if abs(light[2]) < 1e-9:
            continue
        t = (back_z - pos[2]) / light[2]
        sx = pos[0] + light[0] * t
        sy = pos[1] + light[1] * t
        # Perspective divide at the backdrop's depth.
        half_h = math.tan(fov / 2.0) * (cz - back_z)
        half_w = half_h * width / height
        px = (sx / half_w * 0.5 + 0.5) * width
        py = (0.5 - sy / half_h * 0.5) * height
        # The shadow is the sphere's silhouette, foreshortened along the light;
        # sample well inside it so the penumbra and the ellipse's minor axis
        # cannot leak in. 0.2 of the radius also keeps the sampled chords
        # within 2% of the centre chord ``2r``, which is what makes the
        # measured transmittance comparable to Beer-Lambert evaluated at 2r.
        pr = radius / half_h * 0.5 * height
        out.append(
            {
                "name": obj.get("name", "sphere"),
                "radius": radius,
                "px": px,
                "py": py,
                "sample_radius": max(2.0, pr * 0.2),
                "material": obj.get("material", {}),
            }
        )
    return out


def _predict(material, radius):
    """Beer-Lambert transmittance of the centre chord through a glass sphere.

    The theory this measurement is checked against, independent of either
    renderer: light crossing the sphere loses ``1 - F0`` at each of the two
    interfaces, is tinted by the base colour at each of them (glTF
    ``KHR_materials_transmission``, and what ``_scatter_impl``'s
    ``trans_w = trans_energy * tint`` does on the view ray), and is absorbed
    over the chord ``2r`` by ``KHR_materials_volume``'s
    ``sigma = -ln(linear(attenuation_color)) / attenuation_distance``.

    Refraction is deliberately absent: a shadow march travels straight, so the
    prediction is the *unbent* chord. That is why a real render's caustic core
    is brighter than this and its rim darker -- the number to compare against
    is the one at the shadow's centre.
    """
    base = np.array(
        [float(c) for c in material.get("color", [1.0, 1.0, 1.0])], dtype=np.float64
    )
    base_lin = np.where(base <= 0.04045, base / 12.92, ((base + 0.055) / 1.055) ** 2.4)
    transmission = float(material.get("transmission", 0.0))
    metalness = float(material.get("metalness", 0.0))
    ior = abs(float(material.get("ior", 1.0)))
    f0 = 0.0
    if ior > 1.0 + 1e-4:
        r0 = (1.0 - ior) / (1.0 + ior)
        f0 = r0 * r0
    interface = transmission * (1.0 - metalness) * (1.0 - f0) * base_lin

    atten = material.get("attenuation_color")
    absorb = np.ones(3)
    if atten is not None:
        a = np.array([float(c) for c in atten], dtype=np.float64)
        a_lin = np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)
        distance = float(material.get("attenuation_distance", 1.0))
        sigma = -np.log(np.maximum(a_lin, 1e-9)) / max(distance, 1e-9)
        absorb = np.exp(-sigma * (2.0 * radius))
    # Two interfaces (entry and exit), one chord between them.
    return interface * interface * absorb


def _disc_mean(lin: np.ndarray, cx: float, cy: float, r: float):
    h, w = lin.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w]
    mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= r * r
    if not mask.any():
        return None, 0
    return lin[mask].mean(axis=0), int(mask.sum())


def _open_backdrop(lin: np.ndarray, shadows, width, height):
    """A patch of lit backdrop with no sphere and no shadow on it: the
    brightest 2% of pixels, which on these scenes is exactly that.
    """
    lum = lin @ np.array([0.2126, 0.7152, 0.0722])
    thresh = np.quantile(lum, 0.98)
    mask = lum >= thresh
    return lin[mask].mean(axis=0), int(mask.sum())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scene", type=Path)
    ap.add_argument("images", type=Path, nargs="+")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    import cv2

    spec = json.loads(args.scene.read_text())
    rows = []
    for image_path in args.images:
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise SystemExit(f"cannot read {image_path}")
        height, width = bgr.shape[:2]
        lin = _srgb_to_linear(bgr[:, :, ::-1])
        spheres = _project(spec, width, height)
        open_rgb, open_n = _open_backdrop(lin, spheres, width, height)
        for s in spheres:
            mean, n = _disc_mean(lin, s["px"], s["py"], s["sample_radius"])
            if mean is None:
                continue
            ratio = mean / np.maximum(open_rgb, 1e-9)
            rows.append(
                {
                    "image": image_path.name,
                    "sphere": s["name"],
                    "radius": s["radius"],
                    "px": round(s["px"], 1),
                    "py": round(s["py"], 1),
                    "pixels": n,
                    "open_backdrop_linear": [round(float(c), 4) for c in open_rgb],
                    "shadow_linear": [round(float(c), 4) for c in mean],
                    "transmittance": [round(float(c), 4) for c in ratio],
                    "predicted": [
                        round(float(c), 4) for c in _predict(s["material"], s["radius"])
                    ],
                    # How far the shadow is from grey: 1.0 is perfectly
                    # neutral, and the whole content of §4.10.
                    "green_over_red": round(float(ratio[1] / max(ratio[0], 1e-9)), 3),
                }
            )

    if args.json:
        for row in rows:
            print(json.dumps(row))
        return

    print(
        f"{'image':30s} {'sphere':7s} {'r':>5s} "
        f"{'measured shadow transmittance':>31s} {'G/R':>6s} "
        f"{'Beer-Lambert prediction':>31s}"
    )
    for row in rows:
        t = row["transmittance"]
        p = row["predicted"]
        print(
            f"{row['image'][:30]:30s} {row['sphere'][:7]:7s} "
            f"{row['radius']:5.2f} "
            f"{t[0]:10.4f} {t[1]:10.4f} {t[2]:10.4f} {row['green_over_red']:6.2f} "
            f"{p[0]:10.4f} {p[1]:10.4f} {p[2]:10.4f}"
        )
    if rows:
        print(
            "\nopen backdrop (linear): "
            + str(rows[0]["open_backdrop_linear"])
            + "\nprediction is the unbent centre chord (no refraction, no "
            "ambient fill); see _predict."
        )


if __name__ == "__main__":
    main()
