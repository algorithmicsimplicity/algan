"""A/B check for the opaque any-hit shadow early-out (ALGAN_SHADOW_ANYHIT).

Two static shadow-heavy scenes, each rendered with the early-out off and on
in one process (alternating, so thermal drift cancels):

* ``opaque``  -- every mob fully opaque (triangles + PN spheres + bezier
  text all cast shadows). The tracer should select any-hit-only mode 3
  (no translucent geometry in the batch), which never runs the ordered
  march at all.
* ``mixed``   -- the same scene plus a half-opacity floating square, which
  forces mode 2 (any-hit pre-pass, march fallback) and exercises the
  translucent-attenuation path.

For each scene the two arms are compared pixel-wise (the early-out is
epsilon-equivalent, not provably bit-equal: the march's interpolated alpha
of an authored-1.0 surface is 1.0 +/- 1 ulp, where the any-hit returns an
exact 1.0). Expected: max abs u8 delta <= 2 (the suite tolerance), and in
practice 0 on clean scenes. Timed save_video runs report the win.

The ON arm's mode is selectable: ``1`` runs the any-hit walks (modes 2/3
chosen by the tracer), ``gather`` (the default) runs the kbuf gather-march
(mode 4). Byte-identity is expected for both.

Usage:
    .venv/Scripts/python.exe benchmarks/_shadow_anyhit_ab.py [reps] [1|gather]
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

OUT_DIR = os.path.join("algan_outputs", "profiling")
PINNED_BYTES = 2_400_000_000


def build_scene(mixed):
    Scene.set_background_color(DARKER_GRAY)
    SETTINGS.raytracing.set(shadows=True)
    with Off():
        AmbientLight(color=WHITE, intensity=0.3).spawn(animate=False)
        DirectionalLight(
            location=RIGHT * 3 + UP * 6 + OUT * 5,
            target=ORIGIN,
            color=WHITE,
            intensity=1.0,
            shadow_angle=0.35,
        ).spawn(animate=False)
        PointLight(location=LEFT * 4 + UP * 3 + OUT * 2).spawn(animate=False)

        ground = Square(color=GRAY).scale(9)
        ground.rotate(90, RIGHT).move(DOWN * 1.5)
        ground.spawn(animate=False)

        Sphere(radius=0.7, color=RED).move(LEFT * 1.6 + UP * 0.2).spawn(animate=False)
        Cube(color=BLUE).move(RIGHT * 1.4 + DOWN * 0.6).spawn(animate=False)
        Text("SHADOW", font_size=60, color=YELLOW).move(UP * 1.6).spawn(animate=False)
        if mixed:
            veil = Square(color=GREEN, opacity=0.5).scale(1.6)
            veil.rotate(90, RIGHT).move(UP * 0.6 + LEFT * 0.5)
            veil.spawn(animate=False)
    # A tiny camera drift keeps every frame distinct without moving shadows
    # off screen (optimizations must serve moving scenes).
    Scene.get_camera().move(RIGHT * 0.2)


def render_once(tag, mixed, anyhit):
    path = os.path.join(OUT_DIR, f"anyhit_{tag}.mp4")
    rt_settings.set_shadow_anyhit(anyhit)
    SceneManager.reset()
    SETTINGS.computing.set(available_memory_override=PINNED_BYTES)
    scene = SceneManager.instance().current_scene
    scene.set_video_settings(PREVIEW)
    build_scene(mixed)
    t0 = time.perf_counter()
    Scene.save_video(path, PREVIEW, overwrite=True)
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
    on_mode = sys.argv[2] if len(sys.argv) > 2 else "gather"
    on_value = "gather" if on_mode == "gather" else True
    print(f"ON arm mode: {on_value!r}", flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    for label, mixed in (("opaque", False), ("mixed", True)):
        # Alternate the arms within each round so warm-up and thermal drift
        # land on both sides; round 1 (kernel compiles) is discarded.
        t_off, t_on = [], []
        for _ in range(reps):
            _p, dt = render_once(f"{label}_off", mixed, False)
            t_off.append(dt)
            _p, dt = render_once(f"{label}_on", mixed, on_value)
            t_on.append(dt)
        off_path = os.path.join(OUT_DIR, f"anyhit_{label}_off.mp4")
        on_path = os.path.join(OUT_DIR, f"anyhit_{label}_on.mp4")
        off_px = read_frames(off_path)
        on_px = read_frames(on_path)
        sha_equal = hashlib.sha256(open(off_path, "rb").read()).hexdigest() == (
            hashlib.sha256(open(on_path, "rb").read()).hexdigest()
        )
        delta = np.abs(off_px.astype(np.int16) - on_px.astype(np.int16))
        keep_off = t_off[1:] if len(t_off) > 1 else t_off
        keep_on = t_on[1:] if len(t_on) > 1 else t_on
        print(
            f"{label}: sha_equal={sha_equal} max|d|={delta.max()} "
            f"pixels>2={(delta > 2).sum()} "
            f"off={min(keep_off):6.2f}s on={min(keep_on):6.2f}s "
            f"speedup={min(keep_off) / min(keep_on):5.2f}x "
            f"(all off={['%.2f' % t for t in t_off]} "
            f"on={['%.2f' % t for t in t_on]})",
            flush=True,
        )


if __name__ == "__main__":
    main()
