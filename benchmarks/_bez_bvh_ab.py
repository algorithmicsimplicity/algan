"""A/B check for the median-split bezier STBVH (``ALGAN_BEZ_BVH_SPLIT``).

``DESIGN_mesh_identity.md`` ss3.4 flips the bezier-circuit instance ordering from
4D-Morton to the recursive longest-axis median split the triangle tree already
uses. Both are pure REORDERINGS -- same instances, same opaque flags, same tree
shape -- so the set of intersections found is unchanged and the claim inherited
from the triangle tree is ~20-25% fewer traversal steps.

**Byte-identity is the wrong gate, and ss4.8 says so.** A circuit's seam
de-duplication is discovery-order sensitive, so the reorder moves output at the
epsilon level. Worse, ``_split_determinism_check.py`` documents that scenes whose
pixels split into several branches are *not* byte-reproducible run to run at all
(non-associative float ``atomic_add`` on ``pix_accum``), so an off-vs-on diff of
zero is not available even in principle on such a scene.

So this measures the reorder against the **run-to-run noise floor** of the same
scene, which is the only honest comparison:

    noise   = render(off) vs render(off)   -- two independent renders, same settings
    ab      = render(off) vs render(on)

If ``ab`` is within ``noise``, the reorder costs nothing visible. If ``ab`` is
materially larger, it moves output and needs baselines. Both distributions are
reported in full (max, count over tolerance, mean per frame) rather than reduced
to a pass/fail, because "within noise" is a judgement the reader should make from
the numbers.

Timing is reported alongside: total render wall (alternating, in one process, so
thermal drift cancels) and the ``merge collections + build BVHs`` stage, which is
where a costlier build would show up. A median split is more work to BUILD than a
Morton sort, so the build time moving up while the render moves down is the
expected shape, and the net is what decides it.

The scene is bezier-heavy on purpose -- ``Text``, ``Tex`` and 2-D shapes are all
cubic bezier circuits, and they are the only geometry this tree holds.

Usage:
    .venv/Scripts/python.exe benchmarks/_bez_bvh_ab.py [reps] [quality]
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from algan import *  # noqa: F403,E402
from algan.scene_manager import SceneManager  # noqa: E402
from algan.utils.profiling_utils import TIMERS, install_pipeline_hooks  # noqa: E402

# TIMERS only records stages that something has wrapped, and nothing wraps them
# on an ordinary render -- ``profile_scene`` installs the hooks. Without this the
# BVH-build column reads a confident 0.000s, which is worse than no column.
install_pipeline_hooks()

OUT_DIR = os.path.join("algan_outputs", "bez_bvh_ab")
PINNED_BYTES = 1_400_000_000
FONT = "Algan Test Sans"


def build_scene():
    """Many independent circuits, moving.

    Instance COUNT is what a BVH ordering can matter for, so this is a lot of
    small separate circuits rather than a few big ones, spread over the frame so
    a space-filling curve's discontinuities have somewhere to land.
    """
    Scene.set_background(DARKER_GRAY)
    shapes = []
    with Off():
        for i in range(7):
            for j in range(5):
                x = (i - 3) * 1.05
                y = (j - 2) * 1.0
                kind = (i + j) % 4
                if kind == 0:
                    mob = Circle(color=RED).scale(0.32)
                elif kind == 1:
                    mob = Square(color=GREEN).scale(0.3)
                elif kind == 2:
                    mob = Triangle(color=BLUE).scale(0.34)
                else:
                    mob = Star(color=YELLOW).scale(0.34)
                mob.move(RIGHT * x + UP * y)
                mob.spawn(animate=False)
                shapes.append(mob)
        # Glyph circuits reach the same tree by the same route.
        Text("bezier bvh ordering", font_size=26, color=WHITE, font=FONT).move(
            DOWN * 2.6
        ).spawn(animate=False)
        Tex(r"\sum_{i=0}^{n} x_i^2", font_size=30, color=WHITE).move(UP * 2.6).spawn(
            animate=False
        )

    with Sync(run_time=2):
        for i, mob in enumerate(shapes):
            mob.rotate(60 * (1 + i % 5), OUT)
        Scene.get_camera().move(RIGHT * 0.25)


def render_once(tag, split, quality):
    path = os.path.join(OUT_DIR, f"bez_{tag}.mp4")
    SETTINGS.raytracing.experimental.set(bez_bvh_split=bool(split))
    SceneManager.reset()
    SETTINGS.computing.set(available_memory_override=PINNED_BYTES)
    scene = SceneManager.instance().current_scene
    scene.set_video_settings(quality)
    build_scene()
    TIMERS.reset()
    t0 = time.perf_counter()
    Scene.save_video(path, quality, overwrite=True)
    wall = time.perf_counter() - t0
    # "STBVH build (in merge)" is the inner wrap; the outer stage also counts the
    # merge itself, so prefer the inner one and fall back to the outer.
    build = 0.0
    times = dict(TIMERS.times)
    for name, total in times.items():
        if "STBVH build" in name:
            build += float(total)
    if not build:
        for name, total in times.items():
            if "build BVHs" in name:
                build += float(total)
    return path, wall, build


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


def distribution(path_a, path_b, tol=2):
    a, b = read_frames(path_a), read_frames(path_b)
    if a.shape != b.shape:
        return {"note": f"shape mismatch {a.shape} vs {b.shape}"}
    delta = np.abs(a.astype(np.int16) - b.astype(np.int16))
    over = delta.max(axis=-1) > tol
    per_frame = over.reshape(delta.shape[0], -1).sum(axis=1)
    return {
        "max": int(delta.max()),
        f"px>{tol}": int(over.sum()),
        "px_per_frame_mean": float(per_frame.mean()),
        "px_per_frame_worst": int(per_frame.max()),
        "frames_affected": int((per_frame > 0).sum()),
        "frames": int(delta.shape[0]),
    }


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    quality = globals()[sys.argv[2]] if len(sys.argv) > 2 else MD
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"quality={quality.resolution} reps={reps}")

    walls = {"off": [], "on": []}
    builds = {"off": [], "on": []}
    # Two independent OFF renders give the noise floor; alternating off/on gives
    # the A/B. Order off, off2, on so thermal drift is spread over both arms.
    for rep in range(reps):
        _p, w, b = render_once("off", False, quality)
        walls["off"].append(w)
        builds["off"].append(b)
        if rep == 0:
            render_once("off2", False, quality)
        _p, w, b = render_once("on", True, quality)
        walls["on"].append(w)
        builds["on"].append(b)

    off = os.path.join(OUT_DIR, "bez_off.mp4")
    off2 = os.path.join(OUT_DIR, "bez_off2.mp4")
    on = os.path.join(OUT_DIR, "bez_on.mp4")
    print()
    print("NOISE FLOOR  off vs off (two independent renders, same settings):")
    print(f"  {distribution(off, off2)}")
    print("A/B          off vs on (the reorder):")
    print(f"  {distribution(off, on)}")
    print()
    keep = slice(1, None) if reps > 1 else slice(None)
    print(
        f"wall  off={min(walls['off'][keep]):6.2f}s on={min(walls['on'][keep]):6.2f}s "
        f"ratio={min(walls['on'][keep]) / min(walls['off'][keep]):5.3f}x"
    )
    print(
        f"build off={min(builds['off'][keep]):6.3f}s "
        f"on={min(builds['on'][keep]):6.3f}s  (BVH build stage)"
    )
    print(f"  all wall off={[f'{t:.2f}' for t in walls['off']]}")
    print(f"  all wall on ={[f'{t:.2f}' for t in walls['on']]}")


if __name__ == "__main__":
    main()
