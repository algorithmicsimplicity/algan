"""Frame-level A/B for in-kernel texture time interpolation.

Three arms of one scene, rendered in one process with pinned batch windows:

* ``dense``   -- TEXTURE_TIME_LERP=0: every animated colour-texture window
  materializes one image per frame, byte for byte the stage-1-3 pipeline.
* ``lerp``    -- TEXTURE_TIME_LERP=1, TEXTURE_U8_STORAGE=0: endpoint stacks
  as plain f32 authored rows.
* ``lerp_u8`` -- both on (the defaults): u8-provenance endpoint stacks pack
  as bytes with no LUT.

Two comparisons, with different standards -- deliberately:

* ``lerp_u8`` vs ``lerp`` must be BYTE-IDENTICAL. A packed byte b decodes as
  ``b / 255`` (an IEEE division, the same bits torch's ``q / 255`` stored),
  so both arms hand the lerp the identical authored operands.
* ``lerp`` vs ``dense`` is a QUALIFIED flip, tolerance <= 2 channel values
  (the render suites' own bound): the per-frame weights are bit-identical to
  the dense replay's, but the lerp re-associates (``(post - pre) * w``
  in-kernel against the replay's ``change * w`` on the host) and the
  linear-light decode moves from torch on the host to the kernel twin --
  the same class of exception as ALGAN_WIDE_ATTR_RENDER_DEVICE.

The scene makes every changed path fire, and the script asserts each did: a
file-backed ImageMob whose u8-provenance map crossfades to its own flip (a
u8 endpoint stack) while ALSO fading (the opacity region composes with the
lerp region), a procedural float-texture Sphere crossfade (an f32 stack
through the wrap pad), an instant Off() swap (a step description), a static
ImageMob copy (content dedup), and a moving cube. Windows are pinned as in
_texture_opacity_ab.py, and the fades are split across batches so
sub-windows exercise mid-animation weights.

    uv run python benchmarks/_texture_lerp_ab.py
"""

from __future__ import annotations

import os
import sys

os.environ["ALGAN_PREFETCH_BATCHES"] = "0"
# A warm daemon keeps adaptive renderer state across runs; this A/B must run
# all arms in one process of its own.
os.environ.setdefault("ALGAN_USE_DAEMON", "0")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path  # noqa: E402

import torch  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "algan_outputs" / "_texture_lerp_ab"
IMAGE = REPO / "benchmarks" / "performance" / "world_map.png"

from algan import *  # noqa: E402
from algan.mobs.image_mob import ImageMob  # noqa: E402
from algan.mobs.shapes_3d import Sphere  # noqa: E402
from algan.rendering.raytracing import scene_builder  # noqa: E402
from algan.rendering.raytracing import settings as rt_settings  # noqa: E402
from algan.scene_manager import SceneManager  # noqa: E402


def build_scene():
    torch.manual_seed(1234)
    with Off():
        crossfading = ImageMob(str(IMAGE)).scale(0.55).move(LEFT * 3).spawn()
        ImageMob(str(IMAGE)).scale(0.55).move(RIGHT * 3).spawn()
        sphere = (
            Sphere(radius=0.9, color_texture=torch.rand(24, 20, 5).clamp(0.2, 1.0))
            .move(UP * 2)
            .spawn()
        )
        swapper = (
            Surface(color_texture=torch.rand(16, 16, 5).clamp(0.2, 1.0))
            .scale(0.5)
            .move(DOWN * 2 + LEFT * 2)
            .spawn()
        )
        cube = Cube().scale(0.5).move(DOWN * 1.5).spawn()
    # Crossfades spanning most of the clip, so the pinned 5-frame windows cut
    # them mid-animation; the image also fades, so the opacity region rides
    # beside its lerp region.
    with Sync(run_time=1.2):
        crossfading.color_texture = crossfading.color_texture.flip(0)
        crossfading.opacity = 0.35
        sphere.color_texture = torch.rand(24, 20, 5).clamp(0.2, 1.0)
        cube.move(RIGHT * 2)
    with Off():
        swapper.color_texture = torch.rand(16, 16, 5).clamp(0.2, 1.0)
    Scene.wait(0.3)


def set_arm(lerp, u8):
    rt_settings.set_texture_time_lerp(lerp)
    rt_settings.set_texture_u8_storage(u8)


class MergeWatcher:
    """Records per-merge texture facts so non-vacuity is asserted, not assumed."""

    def __init__(self):
        self.tex_rows = []
        self.metas = []
        self.windows = []
        self._orig = None

    def attach(self):
        watcher = self
        self._orig = scene_builder._merge_scene

        def watching(prims, **kwargs):
            m = watcher._orig(prims, **kwargs)
            watcher.tex_rows.append(int(m["textures"].shape[1]))
            watcher.metas.append(m["tri_tex_meta"].cpu().clone())
            return m

        scene_builder._merge_scene = watching
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

    def meta_facts(self):
        has_lerp = False
        has_u8_stack = False
        has_f32_stack = False
        has_lerp_with_opacity = False
        for meta in self.metas:
            for row in meta:
                if int(row[0]) < 0:
                    continue
                if int(row[16]) >= 0 and int(row[17]) > 1:
                    has_lerp = True
                    if int(row[15]) == -2:
                        has_u8_stack = True
                    if int(row[15]) == -1:
                        has_f32_stack = True
                    if int(row[13]) >= 0 and int(row[14]) > 1:
                        has_lerp_with_opacity = True
        return has_lerp, has_u8_stack, has_f32_stack, has_lerp_with_opacity


def render_arm(lerp, u8, out_name):
    set_arm(lerp, u8)
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
        set_arm(True, True)
    print(
        f"arm lerp={lerp} u8={u8}: windows={watcher.windows} "
        f"tex_rows={watcher.tex_rows}"
    )
    return watcher


def compare(name_a, name_b):
    import cv2
    import numpy as np

    cap_a = cv2.VideoCapture(str(OUT_DIR / name_a))
    cap_b = cv2.VideoCapture(str(OUT_DIR / name_b))
    worst = 0
    worst_frame = -1
    differing = 0
    frames = 0
    while True:
        ok_a, fa = cap_a.read()
        ok_b, fb = cap_b.read()
        if not ok_a or not ok_b:
            if ok_a != ok_b:
                print(f"FRAME COUNT MISMATCH at {frames} ({name_a} vs {name_b})")
                sys.exit(1)
            break
        delta = np.abs(fa.astype(np.int16) - fb.astype(np.int16))
        d = int(delta.max())
        differing += int((delta.max(axis=2) > 0).sum())
        if d > worst:
            worst = d
            worst_frame = frames
        frames += 1
    cap_a.release()
    cap_b.release()
    print(
        f"{name_a} vs {name_b}: {frames} frames, max channel diff {worst} "
        f"(frame {worst_frame}), differing pixels {differing}"
    )
    return worst, differing


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SETTINGS.computing.set(
        max_animation_batch_size=5, max_cpu_memory_used=8 * (1 << 30)
    )
    # Warm-up render, discarded: the first render of a fresh process populates
    # glyph/adaptive caches an arm must not be the one to pay.
    render_arm(True, True, "_warmup.mp4")
    u8_arm = render_arm(True, True, "arm_lerp_u8.mp4")
    lerp_arm = render_arm(True, False, "arm_lerp.mp4")
    dense_arm = render_arm(False, True, "arm_dense.mp4")

    problems = []
    if not (u8_arm.windows == lerp_arm.windows == dense_arm.windows):
        print(
            "BATCH WINDOWS DIFFER BETWEEN ARMS -- comparison confounded:\n"
            f"  lerp_u8: {u8_arm.windows}\n  lerp:    {lerp_arm.windows}\n"
            f"  dense:   {dense_arm.windows}"
        )
        sys.exit(1)

    has_lerp, has_u8_stack, has_f32_stack, lerp_and_fade = u8_arm.meta_facts()
    if not has_lerp:
        problems.append("no merge carried a lerp region (no described window)")
    if not has_u8_stack:
        problems.append("no merge stored a u8-packed endpoint stack")
    if not lerp_and_fade:
        problems.append(
            "no map carried a lerp region AND an opacity region -- the two "
            "were never composed"
        )
    l_lerp, l_u8, l_f32, _ = lerp_arm.meta_facts()
    if not (l_lerp and l_f32):
        problems.append("lerp arm carried no f32 endpoint stack")
    if l_u8:
        problems.append("u8 stack engaged with TEXTURE_U8_STORAGE off")
    d_lerp, *_ = dense_arm.meta_facts()
    if d_lerp:
        problems.append("dense arm carried a lerp region -- kill switch leaks")
    if not any(rows > min(u8_arm.tex_rows) * 2 for rows in dense_arm.tex_rows):
        problems.append(
            "dense arm never carried a window >2x the described bank -- the "
            "crossfades were not the contrast this A/B claims"
        )
    for p in problems:
        print(f"VACUOUS: {p}")
    if problems:
        sys.exit(1)

    worst_u8, differing_u8 = compare("arm_lerp_u8.mp4", "arm_lerp.mp4")
    worst_lerp, _ = compare("arm_lerp.mp4", "arm_dense.mp4")

    ok = True
    if worst_u8 != 0 or differing_u8 != 0:
        print("FAIL: u8 endpoint-stack flip is not byte-identical")
        ok = False
    if worst_lerp > 2:
        print(
            "FAIL: time-lerp flip exceeded the qualified tolerance "
            f"(max {worst_lerp} > 2)"
        )
        ok = False
    if not ok:
        sys.exit(1)
    print(
        "PASS: u8 stack flip byte-identical; time-lerp flip within tolerance "
        f"(max {worst_lerp} <= 2)"
    )


if __name__ == "__main__":
    main()
