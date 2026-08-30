"""Frame-level A/B for the texture-bank/geometry time dedup family.

Renders one scene twice IN ONE PROCESS -- once with the four structural
toggles OFF (``texture_time_flat`` / ``texture_content_dedup`` /
``texture_window_collapse`` / ``merge_dedup_geometry``, i.e. the legacy
merge layout byte for byte) and once with all four ON (the defaults) -- and
requires every frame byte-identical. The toggles are read live at the merge /
primitive build, so in-process flipping through their setters is legal (none
of them sits behind a ``ti.static`` gate: the per-map time length travels as
DATA in the texture meta, which is what lets one compiled sampler serve both
layouts).

The scene is built to make every changed code path non-vacuous, and the
script asserts each fired rather than trusting the construction:

* two ImageMobs of the same file -> texture CONTENT dedup;
* a third ImageMob whose texture ANIMATES (a recorded reassignment) -> the
  flattened bank carries a real per-map time length (t > 1);
* everything static for the first half, then a Cube moves -> at least one
  merge collapses ``tri_pos``/bounds to one frame (waking the BVH builders'
  static branches) and at least one keeps them dense;
* shadows on -> the shadow path runs over both tree shapes, and the cached
  scene diagonal (``_shadow_identity_epsilons``) is exercised.

Both arms must see IDENTICAL batch windows or the comparison is confounded
by re-windowed state (chord counts, tessellation): the batch budget is made
generous and the window capped well below it, and the recorded windows are
asserted equal.

Videos are written lossless (libx264rgb crf 0) so a zero diff is the
renderer's own output.

    uv run python benchmarks/_texture_dedup_ab.py
"""

from __future__ import annotations

import os
import sys

os.environ["ALGAN_PREFETCH_BATCHES"] = "0"
# A warm daemon keeps adaptive renderer state across runs; this A/B must run
# both arms in one process of its own.
os.environ.setdefault("ALGAN_USE_DAEMON", "0")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path  # noqa: E402

import torch  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "algan_outputs" / "_texture_dedup_ab"
IMAGE = REPO / "benchmarks" / "performance" / "world_map.png"

from algan import *  # noqa: E402
from algan.rendering.raytracing import scene_builder  # noqa: E402
from algan.rendering.raytracing import settings as rt_settings  # noqa: E402
from algan.scene_manager import SceneManager  # noqa: E402


def build_scene():
    # Every arm must author the SAME scene: the animated texture is random
    # data, so the generator is pinned per build.
    torch.manual_seed(1234)
    with Off():
        img_a = ImageMob(str(IMAGE)).scale(0.6).move(LEFT * 3).spawn()
        ImageMob(str(IMAGE)).scale(0.6).move(RIGHT * 3).spawn()
        anim = ImageMob(torch.rand(24, 24, 5).clamp(0.2, 1.0)).move(UP * 2).spawn()
        Sphere().scale(0.7).move(UP * 0.5).spawn()
        cube = Cube().scale(0.5).move(DOWN * 1.5).spawn()
        Text("dedup").move(DOWN * 2.8).spawn()
    with Sync(duration=0.6):
        img_a.wait()
    with Sync(duration=0.6):
        cube.move(RIGHT)
        anim.color_texture = torch.rand(24, 24, 5).clamp(0.2, 1.0)


def set_arm(on):
    rt_settings.set_texture_time_flat(on)
    rt_settings.set_texture_content_dedup(on)
    rt_settings.set_texture_window_collapse(on)
    rt_settings.set_merge_dedup_geometry(on)
    # Held OFF in BOTH arms: this A/B's contract is byte-identity across the
    # time-dedup family alone, and texture_time_lerp (its own harness:
    # _texture_lerp_ab.py, a QUALIFIED flip) would otherwise turn the ON
    # arm's animated map into an endpoint stack -- comparing two different
    # features and voiding the dense-animated-map (t > 1) evidence.
    rt_settings.set_texture_time_lerp(False)


class MergeWatcher:
    """Records per-merge shapes so non-vacuity is asserted, not assumed."""

    def __init__(self):
        self.tex_shapes = []
        self.tex_meta_tmax = []
        self.tri_pos_frames = []
        self.windows = []
        self._orig = None

    def attach(self):
        watcher = self
        self._orig = scene_builder._merge_scene

        def watching(prims, **kwargs):
            m = watcher._orig(prims, **kwargs)
            watcher.tex_shapes.append(tuple(m["textures"].shape))
            # Cols 10-12 are the per-map time lengths; cols 13+ carry the
            # opacity region / u8 layout (row offsets, not lengths).
            watcher.tex_meta_tmax.append(int(m["tri_tex_meta"][:, 10:13].max()))
            watcher.tri_pos_frames.append(int(m["tri_pos"].shape[0]))
            return m

        scene_builder._merge_scene = watching
        # The render loop binds the name at import time in its own module.
        import algan.render_loop as rl

        rl._merge_scene = watching
        import algan.rendering.raytracing.tracer as tr

        tr._merge_scene = watching

    def detach(self):
        scene_builder._merge_scene = self._orig
        import algan.render_loop as rl

        rl._merge_scene = self._orig
        import algan.rendering.raytracing.tracer as tr

        tr._merge_scene = self._orig

    def wrap_scene(self, scene):
        watcher = self
        orig = scene.get_batch_of_primitives

        def recording(start_ind, end_ind, actors, mem):
            watcher.windows.append((int(start_ind), int(end_ind)))
            return orig(start_ind, end_ind, actors, mem)

        scene.get_batch_of_primitives = recording


def render_arm(on, out_name):
    set_arm(on)
    watcher = MergeWatcher()
    watcher.attach()
    SceneManager.reset()
    try:
        with Scene() as scene:
            watcher.wrap_scene(scene)
            build_scene()
            SETTINGS.raytracing.shadows = True
            scene.save_video(
                str(OUT_DIR / out_name),
                video_settings=PREVIEW,
                overwrite=True,
                codec="libx264rgb",
                ffmpeg_params=["-crf", "0", "-preset", "fast"],
            )
    finally:
        watcher.detach()
        SETTINGS.raytracing.shadows = False
        set_arm(True)
        rt_settings.set_texture_time_lerp(True)
    print(
        f"arm on={on}: windows={watcher.windows} textures={watcher.tex_shapes} "
        f"meta_tmax={watcher.tex_meta_tmax} tri_pos_frames={watcher.tri_pos_frames}"
    )
    return watcher


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Both arms must choose the same batch windows: give the sizer more than
    # the scene needs and cap the window well below the budget, so the arms'
    # different texture pricing cannot pick different durations.
    SETTINGS.computing.set(
        max_animation_batch_size=5, max_cpu_memory_used=8 * (1 << 30)
    )
    # Warm-up render first, discarded: the first render of a fresh process
    # populates glyph caches whose antialiasing differs from every later run
    # (tests/README.md), and an arm must not be the one that pays it.
    render_arm(True, "_warmup.mp4")
    off = render_arm(False, "arm_off.mp4")
    on = render_arm(True, "arm_on.mp4")

    if off.windows != on.windows:
        print(
            "BATCH WINDOWS DIFFER BETWEEN ARMS -- comparison confounded:\n"
            f"  off: {off.windows}\n  on:  {on.windows}"
        )
        sys.exit(1)

    # Non-vacuity: the ON arm must have exercised every changed path.
    problems = []
    if not any(s[0] == 1 for s in on.tex_shapes):
        problems.append("no merge produced a time-flattened texture bank")
    if not any(t > 1 for t in on.tex_meta_tmax):
        problems.append("no merge carried an animated map (per-map t > 1)")
    if not any(f == 1 for f in on.tri_pos_frames):
        problems.append("no merge collapsed tri_pos (static-batch path unexercised)")
    if not any(f > 1 for f in on.tri_pos_frames):
        problems.append("no merge kept tri_pos dense (moving-batch path unexercised)")
    if not any(s[0] > 1 for s in off.tex_shapes):
        problems.append(
            "OFF arm never expanded the bank -- the legacy layout was not the "
            "contrast this A/B claims"
        )
    for p in problems:
        print(f"VACUOUS: {p}")
    if problems:
        sys.exit(1)

    import cv2
    import numpy as np

    cap_a = cv2.VideoCapture(str(OUT_DIR / "arm_off.mp4"))
    cap_b = cv2.VideoCapture(str(OUT_DIR / "arm_on.mp4"))
    worst = 0
    worst_frame = -1
    differing_pixels = 0
    frames = 0
    while True:
        ok_a, frame_a = cap_a.read()
        ok_b, frame_b = cap_b.read()
        if not ok_a or not ok_b:
            if ok_a != ok_b:
                print(f"FRAME COUNT MISMATCH at {frames}")
                sys.exit(1)
            break
        delta = np.abs(frame_a.astype(np.int16) - frame_b.astype(np.int16))
        d = int(delta.max())
        differing_pixels += int((delta.max(axis=2) > 0).sum())
        if d > worst:
            worst = d
            worst_frame = frames
        frames += 1
    cap_a.release()
    cap_b.release()
    print(f"frames compared: {frames}")
    print(
        f"max absolute channel difference: {worst} "
        f"(frame {worst_frame}); differing pixels total: {differing_pixels}"
    )
    if worst != 0 or differing_pixels != 0:
        print("FAIL: outputs are not byte-identical")
        sys.exit(1)
    print("PASS: every frame of both renders is byte-identical")


if __name__ == "__main__":
    main()
