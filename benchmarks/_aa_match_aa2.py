"""THE gate for turning analytic AA on: does it match super_sampling_anti_aliasing=2?

The goal analytic AA has to clear before it can replace supersampling is not
"better than no AA" -- that was ss13-ss17 -- but "at least as good as the shipped
2x2 supersampled default, on everything". Supersampling antialiases every
quantity at once (geometry, shading, specular, shadow rays, reflected images,
textures); analytic coverage antialiases geometry exactly and then needs a
targeted mechanism for each of the rest (DESIGN_analytic_aa.md ss7).

So this scores three arms of the same scene against a supersampled aa=4
reference and requires

    L1(analytic @ aa=1)  <=  L1(supersampled @ aa=2)

per config, and reports the wall time of each, since the point of the exercise is
to get aa=2's quality for less than aa=2's cost.

The config matrix exists to cover one aliasing SOURCE each; do not remove one
without a replacement:

    mesh        triangle silhouettes                 (coverage)
    text        glyph outlines                       (circuit SDF coverage)
    shapes      2D shapes, translucent + bordered    (circuit coverage)
    thin        sub-pixel rods                       (sample-less triangles)
    trans       stacked translucent meshes           (per-sample transmittance)
    shadow      hard shadow edge                     (shadow-ray visibility)
    softshadow  penumbra                             (shadow-ray fan)
    spec        tight specular highlight             (shading crawl)
    mirror      mirror ball                          (reflected image)
    flat        flat mirror                           (reflected image)
    glass       refraction through a lens            (refracted image)

Run: .venv/Scripts/python.exe benchmarks/_aa_match_aa2.py [configs...]
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
    GRAY,
    GREEN,
    LEFT,
    OUT,
    RED,
    RIGHT,
    UP,
    WHITE,
    YELLOW,
    Off,
    Scene,
    SceneManager,
    Sphere,
    Square,
    Sync,
    Text,
    Torus,
    VideoSettings,
)
from algan.mobs.shapes_2d import QuadTriangulated  # noqa: E402
from algan.rendering.lights import AmbientLight, PointLight  # noqa: E402
from algan.rendering.raytracing import (  # noqa: E402
    set_fragment_shading,
    set_shadows,
)
from algan.rendering.raytracing import settings as rt_settings  # noqa: E402
from algan.rendering.shaders.materials import (  # noqa: E402
    MeshLambertMaterial,
    MeshPhysicalMaterial,
    MeshStandardMaterial,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "_rt2_out")
os.makedirs(OUT_DIR, exist_ok=True)

BASE_W, BASE_H = 320, 180
FPS = 4

# ``--box`` scores the pre-exact box-filter circuit coverage instead of the
# shipped exact angle-aware area, so this gate can be read either side of
# DESIGN_analytic_aa.md ss21.
EXACT = "--box" not in sys.argv

ALL_CONFIGS = (
    "mesh",
    "text",
    "shapes",
    "thin",
    "trans",
    "shadow",
    "softshadow",
    "spec",
    "mirror",
    "flat",
    "glass",
)
# Configs that need per-fragment shading and/or traced shadows.
FRAG = ("shadow", "softshadow", "spec", "mirror", "flat", "glass")
SHADOWED = ("shadow", "softshadow")


def _ground(y=-1.4, half=7.0):
    corners = torch.tensor(
        (
            (-half, y, -half),
            (half, y, -half),
            (half, y, half),
            (-half, y, half),
        )
    ).float()
    return QuadTriangulated(corners, color=GRAY)


def _lights(shadow_radius=0.0):
    SceneManager.instance().light_sources = [
        PointLight(
            location=UP * 6 + RIGHT * 3,
            color=WHITE,
            intensity=1.0,
            shadow_radius=shadow_radius,
        ).spawn(animate=False),
        AmbientLight(color=WHITE, intensity=0.3).spawn(animate=False),
    ]


def build_scene(cfg):
    if cfg == "mesh":
        with Off():
            a = Sphere().scale(1.3).move(LEFT * 1.5).set_color(BLUE)
            a.spawn()
            b = Sphere().scale(0.9).move(RIGHT * 1.3 + UP * 0.5)
            b.set_color(GREEN)
            b.spawn()
            t = Torus().scale(0.8).move(DOWN * 0.9).set_color(RED)
            t.spawn()
        with Sync():
            a.move(RIGHT * 0.7)
            t.rotate(50, RIGHT)
        return
    if cfg == "text":
        with Off():
            Text("Analytic AA").scale(0.5).move(UP * 0.4).spawn()
            body = Text("gjq 0123").scale(0.35).move(DOWN * 0.6)
            body.spawn()
        body.move(RIGHT * 0.3)
        return
    if cfg == "shapes":
        with Off():
            sq = Square(color=RED).scale(1.1).move(LEFT * 1.5)
            sq.spawn()
            from algan import Circle

            ci = Circle(color=GREEN).scale(0.8).move(RIGHT * 1.3)
            ci.opacity = 0.55
            ci.spawn()
            ring = Circle(color=BLUE, border_width=3).scale(0.6)
            ring.move(DOWN * 0.9)
            ring.spawn()
        sq.rotate(35, OUT)
        return
    if cfg == "thin":
        from algan import Line3D

        with Off():
            for i, th in enumerate((0.02, 0.01, 0.005)):
                Line3D(
                    start=LEFT * 1.4 + UP * (0.7 - 0.7 * i),
                    end=RIGHT * 1.4 + UP * (0.7 - 0.7 * i),
                    thickness=th,
                    color=YELLOW,
                ).spawn()
            sm = Sphere().scale(0.02).move(DOWN * 1.3).set_color(BLUE)
            sm.spawn()
        sm.move(RIGHT * 0.4)
        return
    if cfg == "trans":
        with Off():
            a = Sphere().scale(1.1).move(LEFT * 0.5).set_color(GREEN)
            a.opacity = 0.5
            a.spawn()
            b = Sphere().scale(0.9).move(RIGHT * 0.4 + UP * 0.3)
            b.set_color(RED)
            b.opacity = 0.5
            b.spawn()
        a.move(RIGHT * 0.5)
        return
    if cfg in ("shadow", "softshadow"):
        _lights(0.0 if cfg == "shadow" else 0.35)
        with Off():
            _ground().spawn()
            s = Sphere().scale(0.9).move(LEFT * 1.2 + DOWN * 0.2)
            s.set_material(MeshLambertMaterial(color=BLUE))
            s.spawn()
        s.move(RIGHT * 2.0)
        return
    if cfg == "spec":
        # A tight highlight on a smooth curved surface is the shading-crawl
        # case: the specular lobe varies over a pixel, and one shade per
        # fragment samples it once at the pixel centre.
        _lights(0.0)
        with Off():
            s = Sphere().scale(1.4)
            s.set_material(
                MeshStandardMaterial(metalness=0.0, roughness=0.05, color=BLUE)
            )
            s.spawn()
        s.move(RIGHT * 0.5)
        return
    if cfg == "mirror":
        with Off():
            ball = Sphere().scale(1.5)
            ball.set_material(MeshStandardMaterial(metalness=1.0, roughness=0.0))
            ball.spawn()
            a = Sphere().scale(0.5).move(LEFT * 2.6 + UP * 0.8).set_color(RED)
            a.spawn()
            b = Sphere().scale(0.4).move(RIGHT * 2.4 + DOWN * 0.6)
            b.set_color(YELLOW)
            b.spawn()
        a.move(RIGHT * 0.5)
        return
    if cfg == "flat":
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


def render_once(cfg, aa_level, analytic, tag, reps=2):
    """Render an arm ``reps`` times; report the LAST wall time (warm)."""
    dt = 0.0
    name = f"aaMatch_{cfg}_{tag}"
    for _ in range(max(1, reps)):
        SceneManager.reset()
        set_fragment_shading(cfg in FRAG)
        set_shadows(cfg in SHADOWED)
        rt_settings.set_analytic_aa(analytic, bezier=True, triangles=True, exact=EXACT)
        settings = VideoSettings(
            (BASE_W, BASE_H),
            frames_per_second=FPS,
            super_sampling_anti_aliasing=aa_level,
        )
        with Scene(video_settings=settings) as scene:
            build_scene(cfg)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            scene.save_video(
                os.path.join(OUT_DIR, name + ".mp4"),
                video_settings=settings,
                overwrite=True,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
        rt_settings.set_analytic_aa(False)
        set_shadows(False)
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


def main():
    configs = [a for a in sys.argv[1:] if not a.startswith("--")]
    configs = configs or list(ALL_CONFIGS)
    all_ok = True
    # The aliased arm is diagnostic, not a gate: it separates "analytic AA does
    # not antialias this quantity yet" (analytic lands between aliased and aa2)
    # from "analytic AA is actively breaking this" (analytic worse than doing
    # nothing at all), which is a bug and has to be read as one.
    print(
        f"{'config':11s} {'analytic@1':>10s} {'aa2':>8s} {'aliased':>8s} "
        f"{'ratio':>7s} {'t(an)':>7s} {'t(aa2)':>7s}  verdict",
        flush=True,
    )
    rows = []
    for cfg in configs:
        p_ref, _ = render_once(cfg, 4, False, "aa4_ref", reps=1)
        f_ref = read_frames(p_ref)
        p_an, t_an = render_once(cfg, 1, True, "aa1_analytic")
        p_s2, t_s2 = render_once(cfg, 2, False, "aa2_super")
        p_al, _ = render_once(cfg, 1, False, "aa1_aliased", reps=1)
        f_an, f_s2 = read_frames(p_an), read_frames(p_s2)
        f_al = read_frames(p_al)
        if not f_ref or len(f_an) != len(f_ref) or len(f_s2) != len(f_ref):
            print(f"{cfg:11s} FAIL: frame count mismatch")
            all_ok = False
            continue
        l1_an = float(np.mean([np.abs(a - r).mean() for a, r in zip(f_an, f_ref)]))
        l1_s2 = float(np.mean([np.abs(a - r).mean() for a, r in zip(f_s2, f_ref)]))
        l1_al = float(np.mean([np.abs(a - r).mean() for a, r in zip(f_al, f_ref)]))
        ok = l1_an <= l1_s2 * 1.02
        all_ok = all_ok and ok
        rows.append((cfg, l1_an, l1_s2, t_an, t_s2, ok))
        verdict = (
            "OK"
            if ok
            else ("BUG: worse than no AA" if l1_an > l1_al else "worse than aa2")
        )
        print(
            f"{cfg:11s} {l1_an:10.3f} {l1_s2:8.3f} {l1_al:8.3f} "
            f"{l1_an / max(l1_s2, 1e-9):7.2f} {t_an:7.2f} {t_s2:7.2f}  "
            f"{verdict}",
            flush=True,
        )

    if rows:
        ta = sum(r[3] for r in rows)
        ts = sum(r[4] for r in rows)
        print(
            f"\ntotal wall: analytic@1 {ta:.2f}s   aa2 {ts:.2f}s   "
            f"{ts / max(ta, 1e-9):.2f}x"
        )
        bad = [r[0] for r in rows if not r[5]]
        if bad:
            print("still worse than aa2:", ", ".join(bad))
    print("\nMATCH_AA2_OK:", all_ok)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
