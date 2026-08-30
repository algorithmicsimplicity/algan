"""A/B check for compile-time material-pipeline gating (ALGAN_FRAG_PID_GATE).

The per-hit material dispatch (``shading_taichi._run_frag_pipeline``) is
inlined into the shade kernels with every built-in stage reachable, so a
scene of plain diffuse triangles still pays ``_stage_physical``'s clearcoat +
sheen registers. With the gate on, the host hands each shade kernel the
bitmask of the pipeline ids its geometry actually carries and the absent
stages are never compiled in; a single-material batch drops the per-hit id
fetch and compare too.

Three fragment-shaded moving scenes, each rendered gate-off then gate-on in
one process (alternating, so thermal drift cancels), covering both shade
kernels and both gate shapes:

* ``raster_solo``   -- flat triangles, every mob on the default material:
  the hybrid raster front-end shades them in ``raster_first_shade`` and the
  gate collapses the dispatch to one unconditional stage. This is the shape
  of an ordinary scene.
* ``raster_mixed``  -- the same scene with three materials (unlit + phong +
  standard), so the gate compiles 3 of the 6 built-in stages instead of 6.
* ``wavefront_solo`` -- the solo scene with the raster front-end off, so
  primary shading runs in ``wavefront_shade`` (the kernel that dominates the
  profile) instead.

The gate only removes branches the kernel could never take, so the output
must be BYTE-identical; the run reports sha equality, the max u8 delta, and
the speed-up. The gate masks of the last batch are printed as engagement
proof (``ALL`` = ungated).

Note: PN patches no longer occur in ordinary scenes (``Surface`` mobs dice
into flat triangles), so the PN half of the gate stays ungated here.

Quality matters here: at PREVIEW these scenes sit on the fixed-overhead floor
(~1.5 s of prep/encode per render) and the shade kernel is too small a share
to read, so the default is MD.

Usage:
    .venv/Scripts/python.exe benchmarks/_frag_pid_gate_ab.py [reps] [quality]
"""

import hashlib
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from algan import *  # noqa: F403,E402
from algan.rendering.raytracing import settings as rt_settings  # noqa: E402
from algan.rendering.raytracing import tracer as rt_tracer  # noqa: E402
from algan.rendering.shaders.materials import (  # noqa: E402
    MeshBasicMaterial,
    MeshPhongMaterial,
    MeshStandardMaterial,
)
from algan.scene_manager import SceneManager  # noqa: E402

OUT_DIR = os.path.join("algan_outputs", "profiling")
PINNED_BYTES = 2_400_000_000

# (label, mixed materials, hybrid raster front-end)
SCENES = (
    ("raster_solo", False, True),
    ("raster_mixed", True, True),
    ("wavefront_solo", False, False),
)


def build_scene(mixed):
    Scene.set_background(DARKER_GRAY)
    with Off():
        AmbientLight(color=WHITE, intensity=0.3).spawn(animate=False)
        PointLight(location=LEFT * 4 + UP * 3 + OUT * 4).spawn(animate=False)
        DirectionalLight(
            location=RIGHT * 3 + UP * 6 + OUT * 5, target=ORIGIN, color=WHITE
        ).spawn(animate=False)

        ground = Square(color=GRAY).scale(9)
        ground.rotate(90, RIGHT).move(DOWN * 1.5)
        if mixed:
            ground.set_material(MeshStandardMaterial(roughness=0.8))
        ground.spawn(animate=False)

        spheres = []
        for x in (-1.6, 1.6):
            sphere = Sphere(radius=0.7, color=YELLOW).move(RIGHT * x + UP * 1.4)
            if mixed:
                sphere.set_material(MeshStandardMaterial(roughness=0.3))
            sphere.spawn(animate=False)
            spheres.append(sphere)

        cubes = []
        for i, x in enumerate((-2.0, 0.0, 2.0)):
            cube = Cube(color=(RED, GREEN, BLUE)[i]).move(RIGHT * x)
            if mixed:
                cube.set_material(
                    (
                        MeshBasicMaterial(),
                        MeshPhongMaterial(shininess=40.0),
                        MeshStandardMaterial(roughness=0.3, metalness=0.5),
                    )[i]
                )
            cube.spawn(animate=False)
            cubes.append(cube)

    # Optimizations must serve MOVING scenes, so everything animates.
    with Sync(duration=2):
        for i, cube in enumerate(cubes):
            cube.rotate(120 * (i + 1), UP)
        for sphere in spheres:
            sphere.move(DOWN * 0.7)
        Scene.get_camera().move(RIGHT * 0.3)


def render_once(tag, mixed, raster, gate, quality):
    path = os.path.join(OUT_DIR, f"pidgate_{tag}.mp4")
    # getattr: the same scene builder is used to profile a pre-gate build (a
    # ``git stash`` baseline), where the setting does not exist yet.
    setter = getattr(rt_settings, "set_frag_pid_gate", None)
    if setter is not None:
        setter(gate)
    SceneManager.reset()
    SETTINGS.computing.set(available_memory_override=PINNED_BYTES)
    SETTINGS.raytracing.experimental.set(
        fragment_shading=True, hybrid_raster=bool(raster)
    )
    scene = SceneManager.instance().current_scene
    scene.set_video_settings(quality)
    build_scene(mixed)
    t0 = time.perf_counter()
    Scene.save_video(path, quality, overwrite=True)
    return path, time.perf_counter() - t0, dict(rt_tracer._FRAG_PID_LAST)


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
    os.makedirs(OUT_DIR, exist_ok=True)
    for label, mixed, raster in SCENES:
        t_off, t_on = [], []
        masks = {}
        for _ in range(reps):
            _p, dt, _m = render_once(f"{label}_off", mixed, raster, False, quality)
            t_off.append(dt)
            _p, dt, masks = render_once(f"{label}_on", mixed, raster, True, quality)
            t_on.append(dt)
        off_path = os.path.join(OUT_DIR, f"pidgate_{label}_off.mp4")
        on_path = os.path.join(OUT_DIR, f"pidgate_{label}_on.mp4")
        sha_equal = hashlib.sha256(open(off_path, "rb").read()).hexdigest() == (
            hashlib.sha256(open(on_path, "rb").read()).hexdigest()
        )
        delta = np.abs(
            read_frames(off_path).astype(np.int16)
            - read_frames(on_path).astype(np.int16)
        )
        keep_off = t_off[1:] if len(t_off) > 1 else t_off
        keep_on = t_on[1:] if len(t_on) > 1 else t_on
        gate = {k: (bin(v) if v >= 0 else "ALL") for k, v in masks.items()}
        print(
            f"{label}: gate={gate} sha_equal={sha_equal} max|d|={delta.max()} "
            f"pixels>2={(delta > 2).sum()} "
            f"off={min(keep_off):6.2f}s on={min(keep_on):6.2f}s "
            f"speedup={min(keep_off) / min(keep_on):5.2f}x "
            f"(all off={['%.2f' % t for t in t_off]} "
            f"on={['%.2f' % t for t in t_on]})",
            flush=True,
        )


if __name__ == "__main__":
    main()
