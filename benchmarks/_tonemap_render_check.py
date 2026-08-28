"""Render with and without tonemapping, and compare authored colour to pixel.

``benchmarks/_tonemap_transfer_probe.py`` measures the tonemap curve in
isolation. This is the end-to-end confirmation that what the curve does reaches
the encoded frame, on ordinary authored geometry.

Four experiments:

``patches``
    For each authored colour, a flat fill covering the whole frame, rendered
    once with tonemapping on (the default) and once off. Every pixel of the
    frame is that one fill, so the readout needs no projection maths and no
    silhouette-AA guard: what comes back is what the pipeline did to the colour
    the user typed. The tonemap-off column is the control -- it is only a clean
    probe of the tonemap if that column reproduces the authored bytes exactly.

``background``
    The background is prefilled into the same buffer the geometry composites
    into, so it should take the same shift. With no geometry in the frame at
    all, this checks that it does.

``agx``
    The same fills under ``tonemap_method="agx"``, so the two shipped curves
    can be compared on one page.

``scene``
    One representative scene -- flat 2-D fills plus a lit sphere under a bright
    light, so the frame carries genuine above-1.0 HDR for the tonemap to have a
    job to do -- rendered both ways and written out for visual comparison, with
    a per-channel difference summary.

    <venv-python> benchmarks/_tonemap_render_check.py

Outputs land in ``algan_outputs/tonemap_check/``.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

from algan import SETTINGS, Circle, Color, PointLight, Scene, Sphere, Square

OUT_DIR = "algan_outputs/tonemap_check"

# Authored colours, as RGB bytes -- what the user typed.
PATCHES = [
    ("white", (255, 255, 255)),
    ("grey75", (191, 191, 191)),
    ("grey50", (128, 128, 128)),
    ("grey25", (64, 64, 64)),
    ("grey10", (26, 26, 26)),
    ("red", (255, 0, 0)),
    ("green", (0, 255, 0)),
    ("blue", (0, 0, 255)),
    ("yellow", (255, 255, 0)),
]


def _col(rgb_bytes):
    """Algan colours are 0-1 floats; the table is written in the bytes a user
    would name, so convert here rather than in the table.
    """
    return Color(tuple(c / 255.0 for c in rgb_bytes))


def _read(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit(f"could not read back {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _centre(img):
    h, w = img.shape[:2]
    block = img[h // 2 - 2 : h // 2 + 3, w // 2 - 2 : w // 2 + 3, :3].reshape(-1, 3)
    return tuple(int(v) for v in np.median(block, axis=0))


def run_patches():
    rows = []
    for name, rgb in PATCHES:
        # A fresh Scene per colour: constructing one makes it current, so
        # nothing from the previous colour is still spawned.
        scene = Scene()
        # Far larger than the frame and face-on, so every pixel is interior.
        Square(side_length=60, color=_col(rgb)).spawn()
        out = {}
        for tag, tonemapping in (("on", True), ("off", False)):
            SETTINGS.raytracing.set(tonemapping=tonemapping)
            path = f"{OUT_DIR}/patch_{name}_{tag}.png"
            scene.save_frame(
                path,
                SETTINGS.video.set(resolution=(64, 64)),
                overwrite=True,
                background=_col((0, 0, 0)),
            )
            out[tag] = _centre(_read(path))
        rows.append((name, rgb, out["on"], out["off"]))

    SETTINGS.raytracing.set(tonemapping=True)

    print()
    print("Flat authored fill -> encoded pixel (fill covers the whole frame)")
    print(
        f"{'colour':>8} {'authored':>16} {'tonemap ON':>16} "
        f"{'tonemap OFF':>16} {'ON-authored':>16}"
    )
    print("-" * 80)
    worst = 0
    for name, rgb, on, off in rows:
        delta = tuple(o - a for o, a in zip(on, rgb))
        worst = max(worst, max(abs(d) for d in delta))
        print(f"{name:>8} {str(rgb):>16} {str(on):>16} {str(off):>16} {str(delta):>16}")
    print()
    exact = [n for n, rgb, on, off in rows if off == rgb]
    print(f"control -- tonemap OFF reproduces the authored bytes for: {exact}")
    print(f"largest channel error introduced by the default tonemap: {worst}")
    return rows


def run_background():
    """The background is prefilled into the same buffer the geometry composites
    into, so it should be tonemapped too. This checks that it is.
    """
    rows = []
    for name, rgb in [
        ("bg_grey50", (128, 128, 128)),
        ("bg_navy", (16, 24, 64)),
        ("bg_white", (255, 255, 255)),
    ]:
        scene = Scene()  # no geometry at all: the frame is pure background
        out = {}
        for tag, tonemapping in (("on", True), ("off", False)):
            SETTINGS.raytracing.set(tonemapping=tonemapping)
            path = f"{OUT_DIR}/{name}_{tag}.png"
            scene.save_frame(
                path,
                SETTINGS.video.set(resolution=(64, 64)),
                overwrite=True,
                background=_col(rgb),
            )
            out[tag] = _centre(_read(path))
        rows.append((name, rgb, out["on"], out["off"]))

    SETTINGS.raytracing.set(tonemapping=True)
    print()
    print("Background colour (no geometry in frame) -> encoded pixel")
    print(
        f"{'background':>11} {'authored':>16} {'tonemap ON':>16} "
        f"{'tonemap OFF':>16} {'ON-authored':>16}"
    )
    print("-" * 80)
    for name, rgb, on, off in rows:
        delta = tuple(o - a for o, a in zip(on, rgb))
        print(
            f"{name:>11} {str(rgb):>16} {str(on):>16} {str(off):>16} {str(delta):>16}"
        )


def run_agx():
    """The same nine fills under the alternative curve, so the two shipped
    methods can be compared on one page.
    """
    rows = []
    for name, rgb in PATCHES:
        scene = Scene()
        Square(side_length=60, color=_col(rgb)).spawn()
        SETTINGS.raytracing.set(tonemapping=True, tonemap_method="agx")
        path = f"{OUT_DIR}/agx_{name}.png"
        scene.save_frame(
            path,
            SETTINGS.video.set(resolution=(64, 64)),
            overwrite=True,
            background=_col((0, 0, 0)),
        )
        rows.append((name, rgb, _centre(_read(path))))
    SETTINGS.raytracing.set(tonemapping=True, tonemap_method="neutral")

    print()
    print("The same fills under tonemap_method='agx'")
    print(f"{'colour':>8} {'authored':>16} {'agx':>16} {'agx-authored':>16}")
    print("-" * 62)
    for name, rgb, on in rows:
        delta = tuple(o - a for o, a in zip(on, rgb))
        print(f"{name:>8} {str(rgb):>16} {str(on):>16} {str(delta):>16}")


def run_scene():
    results = {}
    scene = Scene()
    # Flat 2-D fills: SDR only, nothing above 1.0 anywhere in them.
    Square(side_length=1.6, color=_col((255, 255, 255))).move_to(
        np.array([-2.6, 1.4, 0.0])
    ).spawn()
    Square(side_length=1.6, color=_col((128, 128, 128))).move_to(
        np.array([-0.8, 1.4, 0.0])
    ).spawn()
    Circle(radius=0.8, color=_col((255, 64, 32))).move_to(
        np.array([1.0, 1.4, 0.0])
    ).spawn()
    Circle(radius=0.8, color=_col((32, 96, 255))).move_to(
        np.array([2.8, 1.4, 0.0])
    ).spawn()
    # A lit sphere under a bright light: this is where genuine HDR lives.
    Sphere(radius=1.1).move_to(np.array([0.0, -1.5, 0.0])).spawn()
    PointLight(location=np.array([2.5, 1.0, -4.0]), intensity=12.0).spawn()

    for tag, tonemapping in (("on", True), ("off", False)):
        SETTINGS.raytracing.set(tonemapping=tonemapping)
        path = f"{OUT_DIR}/scene_{tag}.png"
        scene.save_frame(
            path,
            SETTINGS.video.set(resolution=(640, 360)),
            overwrite=True,
            background=_col((0, 0, 0)),
        )
        results[tag] = _read(path)

    SETTINGS.raytracing.set(tonemapping=True)

    on, off = results["on"].astype(np.int16), results["off"].astype(np.int16)
    diff = on - off
    npix = on.shape[0] * on.shape[1]
    print()
    print("Representative scene, tonemap ON vs OFF")
    print(
        f"  pixels differing at all       : "
        f"{int((diff != 0).any(axis=2).sum())} of {npix}"
    )
    print(f"  mean signed change per channel: {diff.mean():+.2f}")
    print(f"  min / max channel change      : {int(diff.min())} / {int(diff.max())}")
    print(
        f"  channels darkened / brightened: {int((diff < 0).sum())} / "
        f"{int((diff > 0).sum())}"
    )
    # Where the frame is not clipped with the tonemap off, the value was inside
    # the display range, so an ideal "HDR only" tonemap would leave it alone.
    sdr = off < 255
    print(
        f"  of the {int(sdr.sum())} channels NOT clipped with tonemap off, "
        f"{int((diff[sdr] != 0).sum())} still moved"
    )
    cv2.imwrite(
        f"{OUT_DIR}/scene_diff_x8.png",
        cv2.cvtColor(
            np.clip(np.abs(diff) * 8, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR
        ),
    )
    print(f"  wrote {OUT_DIR}/scene_on.png, scene_off.png, scene_diff_x8.png")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    run_patches()
    run_background()
    run_agx()
    run_scene()


if __name__ == "__main__":
    main()
