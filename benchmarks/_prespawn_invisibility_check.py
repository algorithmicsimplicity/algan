"""Characterize what a not-yet-spawned mob's geometry does to a rendered frame.

The arena planner slices a fetched batch down to a prefix rather than
rematerializing a smaller one, which costs seconds per batch. A fetched window
carries every actor spawning anywhere in it, so the prefix renders those actors
while they are still un-spawned -- whereas re-fetching the prefix would leave
them out of the batch entirely. This measures the difference between the two.

Nothing un-spawned is *drawn*: materialization zeroes a mob's opacity outside
its lifespan (``AttributeTimeline.rematerialize_state_at_times``) and
``_pack_frame_visibility`` gives a primitive empty *per-frame* bounds wherever
its alpha falls below ``MIN_ALPHA``, so it never enters the BVH on those
frames. That holds across the paths that reach alpha differently: flat colours,
glow (a separate channel), semi-transparency, PN-triangle surfaces, and
colour-textured primitives -- whose fragment alpha comes from the texture
rather than the corner colours, but whose texture has the frame's opacity
multiplied into it (``Surface._build_render_primitive``).

The frames are not byte-identical even so: carrying the extra primitives
reorders the merged arrays and the STBVH, so shared-edge depth ties and
interpolation boundaries land differently. The residual is a couple of levels
on triangle edges and silhouettes of the *visible* geometry (roughly 1.7% of
pixels, mean 2, tens at silhouettes) and carries no trace of the un-spawned
shapes. Both renders are correct, so the planner slices; this script exists to
keep that residual honest -- run it if the merge, the BVH build or the
visibility cull changes, and check the difference stays edge-local rather than
growing into the un-spawned mobs' silhouettes.

One scene, recorded once, rendered twice over the same frame window; the only
difference is whether batch preparation may include actors that have not
spawned by the window's start. Everything else -- timeline, animation, batch
boundaries, chunking -- is held identical.

    .venv/Scripts/python.exe benchmarks/_prespawn_invisibility_check.py
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import algan.render_loop as rl  # noqa: E402
from algan import (  # noqa: E402
    IN,
    LD,
    LEFT,
    RED,
    RIGHT,
    UP,
    YELLOW,
    ImageMob,
    Off,
    Scene,
    SceneManager,
    Sphere,
    Square,
    Sync,
)
from algan.utils.memory_utils import empty_cache  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "_prespawn_out")
os.makedirs(OUT_DIR, exist_ok=True)
WORLD_MAP = os.path.join(os.path.dirname(__file__), "..", "tests", "world_map.jpg")

FPS = LD.frames_per_second
FIRST_HALF = 1.0  # seconds rendered before the late mobs spawn

_orig_get_batch = rl.RenderLoopMixin.get_batch_of_primitives
_state = {"drop_unspawned": False, "primitives": None}


def _filtered_get_batch(self, start_ind, end_ind, actors, budget):
    """Optionally hide actors that have not spawned by the window's start.

    This is exactly the difference between re-fetching a prefix (which selects
    actors against the prefix's own end) and slicing a longer fetched batch
    (which keeps the actors the longer window selected).
    """
    if _state["drop_unspawned"]:
        start_time = start_ind / self.frames_per_second
        actors = [a for a in actors if float(a.lifespan.start()) <= start_time]
    result = _orig_get_batch(self, start_ind, end_ind, actors, budget)
    if _state["primitives"] is None:
        _state["primitives"] = sum(
            int(getattr(p, "corners", np.zeros((1, 0))).shape[1]) for p in result[0]
        )
    return result


rl.RenderLoopMixin.get_batch_of_primitives = _filtered_get_batch


def build():
    """A moving foreground, plus mobs that spawn only at ``FIRST_HALF``."""
    with Off():
        early = Square(color=RED.set_opacity(0.7)).scale(1.2).move(LEFT * 1.5).spawn()
        early_ball = Sphere(grid_height=12, grid_width=12).scale(0.8).spawn()
    with Sync(run_time=FIRST_HALF):
        early.move(RIGHT * 0.6)
        early_ball.move(UP * 0.4)

    # Spawned only now -- after every frame the comparison looks at -- but
    # recorded inside the same fetched window, so batch preparation carries
    # their geometry through the earlier frames at opacity zero.
    kinds = os.environ.get("PRESPAWN_KINDS", "flat,glow,surface,image").split(",")
    late = []
    with Off():
        if "flat" in kinds:
            m = Square(color=YELLOW).scale(2.5).move(RIGHT * 1.0).spawn()
            late.append((m, LEFT * 0.5))
        if "glow" in kinds:
            m = Square(color=YELLOW).scale(2.0).move(LEFT * 1.0)
            m.glow = 0.9
            m.spawn()
            late.append((m, RIGHT * 0.5))
        if "surface" in kinds:
            m = Sphere(grid_height=12, grid_width=12).scale(1.6)
            m.move(IN * 0.5).spawn()
            late.append((m, UP * 0.3))
        if "image" in kinds:
            m = ImageMob(WORLD_MAP).scale(2.0).spawn()
            late.append((m, UP * 0.2))
    with Sync(run_time=FIRST_HALF):
        early.move(RIGHT * 0.6)
        for mob, delta in late:
            mob.move(delta)


def render(tag, drop_unspawned):
    # Both arms must chunk their frames identically: a render that splits on
    # memory pressure re-windows its state materialization, and CPU rate-func
    # rounding is window-dependent, which would show up as a difference of its
    # own. Hand the previous arm's memory back before starting.
    empty_cache(force_gc=True)
    SceneManager.reset()
    _state["drop_unspawned"] = drop_unspawned
    _state["primitives"] = None
    build()
    path = os.path.join(OUT_DIR, f"prespawn_{tag}.mp4")
    Scene.save_video(path, LD, animate_fade_out=False, overwrite=True)
    print(f"  {tag}: {_state['primitives']} primitives in the first batch", flush=True)
    return path


def read_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame.astype(np.int32))
    cap.release()
    return frames


def main():
    # Each arm renders in its own process. Sharing one leaves the second arm
    # starting against the first's committed VRAM, and a render that splits on
    # memory pressure re-windows its state materialization -- CPU rate-function
    # rounding is window-dependent, so that alone would register as a
    # difference and mask the one being measured.
    if len(sys.argv) > 1 and sys.argv[1] in ("excluded", "included"):
        render(sys.argv[1], sys.argv[1] == "excluded")
        return 0

    import subprocess

    for arm in ("excluded", "included"):
        subprocess.run([sys.executable, __file__, arm], check=True)

    excluded = os.path.join(OUT_DIR, "prespawn_excluded.mp4")
    included = os.path.join(OUT_DIR, "prespawn_included.mp4")
    a, b = read_frames(excluded), read_frames(included)
    compared = int(FIRST_HALF * FPS)
    if len(a) < compared or len(b) < compared:
        print(f"FAIL: too few frames ({len(a)} / {len(b)}, need {compared})")
        return 1

    worst = 0
    worst_share = 0.0
    worst_mean = 0.0
    for i in range(compared):
        per_pixel = np.abs(a[i] - b[i]).max(axis=2)
        diff = int(per_pixel.max())
        share = float((per_pixel > 0).mean())
        mean = float(per_pixel[per_pixel > 0].mean()) if diff else 0.0
        if diff > worst:
            cv2.imwrite(
                os.path.join(OUT_DIR, "worst_diff.png"),
                (per_pixel * 6).clip(0, 255).astype(np.uint8),
            )
        worst = max(worst, diff)
        worst_share = max(worst_share, share)
        worst_mean = max(worst_mean, mean)

    # Tie-break drift on the visible geometry's edges is expected and accepted.
    # Un-spawned geometry actually being *drawn* is not, and looks nothing like
    # it: a shape appearing covers a large area at full contrast rather than a
    # scatter of edge pixels a couple of levels apart.
    drawn = worst_share > 0.05 or worst_mean > 8.0
    print(
        f"{'FAIL' if drawn else 'PASS'}: {compared} pre-spawn frames, worst "
        f"frame differs on {worst_share * 100:.2f}% of pixels "
        f"(mean {worst_mean:.1f}, max {worst}); diff image in "
        f"{os.path.join(OUT_DIR, 'worst_diff.png')}"
    )
    if drawn:
        print("  un-spawned geometry looks like it is being rendered")
    return 1 if drawn else 0


if __name__ == "__main__":
    raise SystemExit(main())
