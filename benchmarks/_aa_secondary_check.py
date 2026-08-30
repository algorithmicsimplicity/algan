"""Continuation-ray supersampling: does it antialias what coverage cannot?

Analytic coverage resolves a mirror's own OUTLINE exactly, but the image seen
inside it is sampled by the continuation ray, and one continuation per pixel
aliases no matter how good the primary coverage is
(``DESIGN_analytic_aa.md`` ss7, ss17). ``analytic_aa_secondary_samples = N``
spawns N continuations from N sub-pixel positions instead.

The scenes put the high-frequency detail INSIDE the reflection/refraction, and
the metric that matters is therefore mean L1 over INTERIOR pixels (the lit region
eroded by three pixels, so the reflector's own antialiased silhouette cannot
carry the result). Scored against a supersampled aa=4 reference.

Arms, all at supersampling=1 except the reference:

    aliased        no analytic AA at all                  (the floor)
    sec=1          analytic AA, one continuation          (what shipped before)
    sec=2/4/8      analytic AA, N jittered continuations
    aa2            supersampling=2 supersampled        (what this replaces)

Run: .venv/Scripts/python.exe benchmarks/_aa_secondary_check.py [configs...]
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("ALGAN_PREFETCH_BATCHES", "0")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Benchmarks must never be measured inside a warm daemon: it keeps adaptive
# renderer state (the memory model's batch-size fit) across runs, so one
# benchmark would be timed against whatever ran before it.
os.environ.setdefault("ALGAN_USE_DAEMON", "0")

from algan import (  # noqa: E402
    BLUE,
    DOWN,
    GREEN,
    LEFT,
    OUT,
    RED,
    RIGHT,
    UP,
    YELLOW,
    Off,
    RenderSettings,
    SceneManager,
    Sphere,
    Square,
    Sync,
    render_to_file,
)
from algan.rendering.raytracing import (  # noqa: E402
    set_fragment_shading,
)
from algan.rendering.raytracing import settings as rt_settings  # noqa: E402
from algan.rendering.shaders.materials import (  # noqa: E402
    MeshPhysicalMaterial,
    MeshStandardMaterial,
)
from algan.rendering.taichi_runtime import _sync_devices  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "_rt2_out")
os.makedirs(OUT_DIR, exist_ok=True)

BASE_W, BASE_H = 320, 180
FPS = 4
SEC_ARMS = (1, 2, 4, 8)


def build_scene(cfg):
    if cfg == "mirror":
        # A big mirror ball compresses the whole surrounding scene into its
        # interior, so the reflected image is as high-frequency as anything the
        # renderer produces -- which is exactly where one continuation per pixel
        # shows. The satellites move, so this is not a static-only case.
        with Off():
            ball = Sphere().scale(1.5)
            ball.set_material(MeshStandardMaterial(metalness=1.0, roughness=0.0))
            ball.spawn()
            a = Sphere().scale(0.5).move(LEFT * 2.6 + UP * 0.8).set_color(RED)
            a.spawn()
            b = Sphere().scale(0.4).move(RIGHT * 2.4 + DOWN * 0.6)
            b.set_color(YELLOW)
            b.spawn()
            c = Sphere().scale(0.6).move(UP * 2.2).set_color(BLUE)
            c.spawn()
        with Sync():
            a.move(RIGHT * 0.5)
            c.move(DOWN * 0.4)
        return
    if cfg == "flat":
        # A FLAT mirror is the clean case: the reflected image is an ordinary
        # perspective view, so a reflected straight edge is a plain staircase
        # that N sub-pixel continuations resolve exactly the way primary
        # coverage resolves a real one. The mirror ball above adds extreme
        # minification on top, which no sample count fully fixes.
        with Off():
            mirror = Square().scale(2.6).rotate(-55, RIGHT).move(DOWN * 0.7)
            mirror.set_material(MeshStandardMaterial(metalness=1.0, roughness=0.0))
            mirror.spawn()
            bar = Square(color=YELLOW).scale(0.9).rotate(20, OUT)
            bar.move(UP * 1.1 + LEFT * 0.6)
            bar.spawn()
            dot = Sphere().scale(0.4).move(UP * 0.9 + RIGHT * 1.2)
            dot.set_color(RED)
            dot.spawn()
        bar.move(RIGHT * 0.5)
        return
    if cfg == "glass":
        # Refraction. The content BEHIND the glass has to have hard edges or
        # there is nothing for supersampling to resolve: a smooth backdrop seen
        # through a lens is still smooth, and an earlier version of this config
        # measured exactly nothing for that reason.
        with Off():
            for i in range(3):
                bar = Square(color=(YELLOW, GREEN, BLUE)[i]).scale(0.5)
                bar.rotate(25 * i - 25, OUT)
                bar.move(UP * (0.9 - 0.9 * i) + LEFT * (0.9 - 0.9 * i) - OUT * 2.0)
                bar.spawn()
            glass = Sphere().scale(1.2)
            glass.set_material(
                MeshPhysicalMaterial(transmission=0.95, roughness=0.02, ior=1.5)
            )
            glass.spawn()
        glass.move(RIGHT * 0.3)
        return
    raise SystemExit(f"unknown config {cfg}")


def render_once(cfg, aa_level, analytic, secondary, tag, reps=2):
    """Render an arm ``reps`` times and report the LAST wall time.

    Each (analytic, N) combination is its own compiled kernel variant, and that
    cold compile grows with N -- at N=8 it is minutes, dwarfing the render. Only
    the warm time says anything about cost.
    """
    dt = 0.0
    for _ in range(max(1, reps)):
        SceneManager.reset()
        set_fragment_shading(True)
        rt_settings.set_analytic_aa(
            analytic, bezier=True, triangles=True, secondary=secondary
        )
        build_scene(cfg)
        name = f"aaSec_{cfg}_{tag}"
        _sync_devices()
        t0 = time.perf_counter()
        render_to_file(
            file_name=name,
            output_dir=OUT_DIR,
            output_path="",
            render_settings=RenderSettings(
                (BASE_W, BASE_H), FPS, supersampling=aa_level
            ),
            file_extension="mp4",
        )
        _sync_devices()
        dt = time.perf_counter() - t0
        rt_settings.set_analytic_aa(False, secondary=4)
    return os.path.join(OUT_DIR, name + ".mp4"), dt


def read_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f.astype(np.float64))
    cap.release()
    return frames


def interior_l1(frames, ref):
    """Mean L1 over pixels lit in both, eroded by three pixels.

    Erosion is what makes this a measurement of the reflected IMAGE rather than
    of the reflector's silhouette, which analytic coverage already handles.
    """
    total, n = 0.0, 0
    for f, r in zip(frames, ref):
        lit = (f.max(axis=2) > 8) & (r.max(axis=2) > 8)
        inner = lit
        for _ in range(3):
            inner = (
                inner[1:-1, 1:-1]
                & inner[:-2, 1:-1]
                & inner[2:, 1:-1]
                & inner[1:-1, :-2]
                & inner[1:-1, 2:]
            )
            inner = np.pad(inner, 1, constant_values=False)
        if not inner.any():
            continue
        total += float(np.abs(f - r).mean(axis=2)[inner].mean())
        n += 1
    return total / max(n, 1)


def edge_levels(frames, ref):
    """Distinct luminance levels in the interior's high-gradient band.

    A reflected edge that steps between two values has few levels; a resolved
    one has a continuum. Counted on interior pixels only, for the same reason.
    """
    total = 0
    for f, r in zip(frames, ref):
        lit = (f.max(axis=2) > 8) & (r.max(axis=2) > 8)
        inner = lit
        for _ in range(3):
            inner = (
                inner[1:-1, 1:-1]
                & inner[:-2, 1:-1]
                & inner[2:, 1:-1]
                & inner[1:-1, :-2]
                & inner[1:-1, 2:]
            )
            inner = np.pad(inner, 1, constant_values=False)
        lum = f.mean(axis=2)
        gx = np.abs(np.diff(lum, axis=1, prepend=lum[:, :1]))
        gy = np.abs(np.diff(lum, axis=0, prepend=lum[:1, :]))
        band = ((gx + gy) > 8.0) & inner
        if not band.any():
            continue
        total += len(np.unique(np.round(lum[band]).astype(np.int32)))
    return total


def main():
    configs = [a for a in sys.argv[1:] if not a.startswith("--")] or [
        "flat",
        "mirror",
        "glass",
    ]
    all_ok = True
    for cfg in configs:
        print(f"--- {cfg} ---", flush=True)
        p_ref, _ = render_once(cfg, 4, False, 1, "aa4_ref")
        f_ref = read_frames(p_ref)
        rows = []
        p, dt = render_once(cfg, 1, False, 1, "aa1_aliased")
        rows.append(("aliased", read_frames(p), dt))
        for n in SEC_ARMS:
            p, dt = render_once(cfg, 1, True, n, f"aa1_sec{n}")
            rows.append((f"sec={n}", read_frames(p), dt))
        p, dt = render_once(cfg, 2, False, 1, "aa2_super")
        rows.append(("aa2", read_frames(p), dt))

        if any(len(fr) != len(f_ref) for _, fr, _ in rows) or not f_ref:
            print(f"[{cfg:7s}] FAIL: frame count mismatch")
            all_ok = False
            continue
        lv_ref = edge_levels(f_ref, f_ref)
        l1s, lvs, il1s = {}, {}, {}
        for label, fr, dt in rows:
            l1s[label] = float(
                np.mean([np.abs(a - r).mean() for a, r in zip(fr, f_ref)])
            )
            lvs[label] = edge_levels(fr, f_ref)
            il1s[label] = interior_l1(fr, f_ref)
            print(
                f"[{cfg:7s}] {label:8s} L1 {l1s[label]:6.3f}   interior L1 "
                f"{il1s[label]:6.3f}   interior edge levels "
                f"{lvs[label]:4d} (ref {lv_ref})   {dt:5.2f}s",
                flush=True,
            )
        # The gate is the claim and no more: N continuations resolve the
        # reflected image better than one does, in error against the reference
        # and in the gradation inside the reflector.
        #
        # NOT gated against the aa=2 arm. Supersampling antialiases everything --
        # shading, specular, shadow edges -- and analytic AA plus secondary
        # sampling deliberately antialiases only geometry and the reflected
        # image, so on some scenes it lands just short of aa=2 and on others
        # (the mirror ball) ahead of it. The aa=2 row is there to be read, not to
        # pass or fail.
        if not (l1s["sec=4"] < l1s["sec=1"]):
            print(
                f"[{cfg:7s}] FAIL: L1 sec=4 {l1s['sec=4']:.3f} "
                f"!< sec=1 {l1s['sec=1']:.3f}"
            )
            all_ok = False
        if cfg == "flat":
            # The interior figures are only a clean signal on the FLAT mirror,
            # where the reflected image is an ordinary perspective view. On a
            # mirror ball or through a lens the reflected scene is minified,
            # which no sample count resolves and which makes both interior
            # columns non-monotone in N -- they are reported there, not gated.
            if not (il1s["sec=4"] < il1s["sec=1"]):
                print(
                    f"[{cfg:7s}] FAIL: interior L1 sec=4 "
                    f"{il1s['sec=4']:.3f} !< sec=1 {il1s['sec=1']:.3f}"
                )
                all_ok = False
            if not (lvs["sec=4"] > lvs["sec=1"]):
                print(
                    f"[{cfg:7s}] FAIL: interior edge levels sec=4 "
                    f"{lvs['sec=4']} !> sec=1 {lvs['sec=1']}"
                )
                all_ok = False

    print("\nAA_SECONDARY_OK:", all_ok)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
