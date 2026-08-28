"""A/B check for the one-mesh coverage cap (``ALGAN_ANALYTIC_AA_ONE_MESH``).

``DESIGN_mesh_identity.md`` ss6.6 measured what the rule buys on *quality*
(coverage error against an exact reference, and ink wobble) and measured nothing
about what it costs. Three things are new when it is on:

* a host **segment reduction** over the fragment CSR in
  ``prepare_sparse_raster_coverage`` -- two ``scatter_reduce_`` for the min/max
  surface id, two ``scatter_add_`` for the per-sheet areas, and a
  ``repeat_interleave`` to build the segment map;
* a **per-fragment f32 array** (``frag_cap``) -- which turns out to cost nothing,
  because it is allocated unconditionally in both raster paths already, so the
  arena footprint is the same in both arms;
* a **running clamp** in the resolve inner loop of ``raster_first_shade`` and
  ``raster_shadow_event_build`` (one ndarray read, one mask test, one compare,
  and a ``mesh_ink`` accumulator per committed fragment).

This measures all of it. Alternating in one process so thermal drift cancels
(cross-process wall clock swings ~2x on this machine), on three scene shapes
that differ in how much of the frame is a one-mesh pixel:

* ``diced``   -- Sphere/Cylinder/Torus, all diced logical PN. The population the
  rule is FOR: a closed solid's two sheets in every silhouette pixel.
* ``flat``    -- Cube/Icosahedron/Octahedron, flat triangle meshes. One ``sid``
  per solid under mesh_id, so the rule engages, but there are far fewer
  fragments per pixel.
* ``mixed``   -- both, plus bezier circuits and a ground plane, with shadows on
  so ``raster_shadow_event_build`` carries the clamp too. The closest shape to
  an ordinary scene, and the only arm that exercises the second resolve kernel.

Output moves -- that is the point of the rule -- so byte-identity is the wrong
gate and the pixel delta is reported as magnitude, not as pass/fail.

Usage:
    .venv/Scripts/python.exe benchmarks/_one_mesh_ab.py [reps] [quality]
"""

import hashlib
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from algan import *  # noqa: F403,E402
from algan.rendering.raytracing import settings as rt_settings  # noqa: E402
from algan.scene_manager import SceneManager  # noqa: E402

OUT_DIR = os.path.join("algan_outputs", "one_mesh_ab")
# Pinned: the render loop sizes its frame windows from live free VRAM, so an
# unpinned A/B compares two different batch splits (memory note
# "render-split free-VRAM nondeterminism").
PINNED_BYTES = 1_400_000_000

SHAPES = ("diced", "flat", "mixed")


def _lights(shadows):
    AmbientLight(color=WHITE, intensity=0.3).spawn(animate=False)
    PointLight(location=LEFT * 4 + UP * 3 + OUT * 4).spawn(animate=False)
    DirectionalLight(
        location=RIGHT * 3 + UP * 6 + OUT * 5, target=ORIGIN, color=WHITE
    ).spawn(animate=False)


def build_scene(shape):
    """One moving scene per shape.

    Everything animates: a static fast path is off-limits (memory note
    "optimization scope: general not static").
    """
    Scene.set_background_color(DARKER_GRAY)
    solids = []
    with Off():
        _lights(shape == "mixed")
        if shape in ("diced", "mixed"):
            solids.append(Sphere(radius=0.8, color=YELLOW).move(LEFT * 2))
            solids.append(Cylinder(radius=0.5, height=1.6, color=RED))
            solids.append(
                Torus(major_radius=0.7, minor_radius=0.22, color=BLUE).move(RIGHT * 2)
            )
        if shape in ("flat", "mixed"):
            solids.append(Cube(color=GREEN).move(LEFT * 2 + DOWN * 1.6))
            solids.append(Icosahedron(color=PURPLE).move(DOWN * 1.6))
            solids.append(Octahedron(color=TEAL).move(RIGHT * 2 + DOWN * 1.6))
        if shape == "mixed":
            ground = Square(color=GRAY).scale(9)
            ground.rotate(90, RIGHT).move(DOWN * 2.6)
            ground.spawn(animate=False)
            # Bezier circuits never enter the one-mesh path; they are here so the
            # arm holds a pixel population the rule must leave alone.
            Circle(color=WHITE).scale(0.5).move(UP * 2.2 + LEFT * 2).spawn(
                animate=False
            )
            Square(color=WHITE).scale(0.5).move(UP * 2.2 + RIGHT * 2).spawn(
                animate=False
            )
        for solid in solids:
            solid.spawn(animate=False)

    with Sync(run_time=2):
        for i, solid in enumerate(solids):
            solid.rotate(90 * (i + 1), UP)
        Scene.get_camera().move(RIGHT * 0.4 + UP * 0.2)


def render_once(shape, on, quality):
    path = os.path.join(OUT_DIR, f"one_mesh_{shape}_{'on' if on else 'off'}.mp4")
    rt_settings.set_analytic_aa(True, one_mesh=bool(on))
    SceneManager.reset()
    SETTINGS.computing.set(available_memory_override=PINNED_BYTES)
    if shape == "mixed":
        SETTINGS.raytracing.set(shadows=True)
    scene = SceneManager.instance().current_scene
    scene.set_video_settings(quality)
    build_scene(shape)
    t0 = time.perf_counter()
    Scene.save_video(path, quality, overwrite=True)
    return path, time.perf_counter() - t0


def read_frames(path):
    import cv2

    frames = []
    cap = cv2.VideoCapture(path)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame.copy())
    cap.release()
    return np.stack(frames)


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    quality = globals()[sys.argv[2]] if len(sys.argv) > 2 else MD
    shapes = sys.argv[3].split(",") if len(sys.argv) > 3 else SHAPES
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"quality={quality.__class__.__name__} reps={reps}", flush=True)
    for shape in shapes:
        t_off, t_on = [], []
        for _ in range(reps):
            _p, dt = render_once(shape, False, quality)
            t_off.append(dt)
            _p, dt = render_once(shape, True, quality)
            t_on.append(dt)
        off_path = os.path.join(OUT_DIR, f"one_mesh_{shape}_off.mp4")
        on_path = os.path.join(OUT_DIR, f"one_mesh_{shape}_on.mp4")
        with open(off_path, "rb") as fh:
            sha_off = hashlib.sha256(fh.read()).hexdigest()
        with open(on_path, "rb") as fh:
            sha_on = hashlib.sha256(fh.read()).hexdigest()
        sha_equal = sha_off == sha_on
        a, b = read_frames(off_path), read_frames(on_path)
        delta = np.abs(a.astype(np.int16) - b.astype(np.int16))
        moved = int((delta.max(axis=-1) > 2).sum())
        px = int(delta.shape[0] * delta.shape[1] * delta.shape[2])
        # Drop the first rep of each arm: run 1 pays kernel specialisation for
        # any variant this shape is first to reach.
        keep_off = t_off[1:] if len(t_off) > 1 else t_off
        keep_on = t_on[1:] if len(t_on) > 1 else t_on
        print(
            f"{shape:6s}: sha_equal={sha_equal} max|d|={int(delta.max()):3d} "
            f"px>2={moved} of {px} ({moved / max(px, 1):.3%}) "
            f"off={min(keep_off):6.2f}s on={min(keep_on):6.2f}s "
            f"ratio={min(keep_on) / min(keep_off):5.3f}x "
            f"(off={[f'{t:.2f}' for t in t_off]} on={[f'{t:.2f}' for t in t_on]})",
            flush=True,
        )


if __name__ == "__main__":
    main()
