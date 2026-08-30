"""Frame-level A/B for the in-sampler texture opacity + u8 storage pair.

Three arms of one scene, rendered in one process with pinned batch windows:

* ``legacy``  -- texture_opacity_in_kernel=0, texture_u8_storage=0: the host
  premultiply and f32 texel rows, byte for byte the pre-change pipeline.
* ``opacity`` -- texture_opacity_in_kernel=1, texture_u8_storage=0.
* ``u8``      -- both on (the defaults).

Two comparisons, with different standards -- deliberately:

* ``u8`` vs ``opacity`` must be BYTE-IDENTICAL. The packed layout's LUT is
  scattered from the map's own decode, so the sampler reads the f32 arm's
  own bits; the only theoretical residue is torch-CPU decoding one byte to
  two bit patterns inside one tensor (SIMD body vs scalar tail), <= 1 ulp in
  linear light. This assert is the claim that it never reaches an output
  byte.
* ``opacity`` vs ``legacy`` is a QUALIFIED flip, tolerance <= 2 channel
  values (the render suites' own bound): the multiply moves from before the
  bilinear filter (per texel, host) to after it (per sample, kernel), which
  legitimately reorders f32 rounding -- the same class of exception as
  ALGAN_WIDE_ATTR_RENDER_DEVICE. With no fade anywhere the multiply is by
  1.0 and exact; this scene fades, so the comparison is non-vacuous.

The scene makes every changed path fire, and the script asserts each did:
a fading file-backed ImageMob (u8-provenance map + a real opacity region),
a static copy of the same file (content dedup + u8 with a constant region),
a fading procedural float-texture Surface (the f32 fallback with a region),
and shadows on (the shadow march samples texture alpha through the same
sampler). Windows are pinned as in _texture_dedup_ab.py: the batch budget is
generous and the window capped below it, so the arms' different texture
pricing cannot pick different durations, and the recorded windows are
asserted equal.

    uv run python benchmarks/_texture_opacity_ab.py
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
OUT_DIR = REPO / "algan_outputs" / "_texture_opacity_ab"
IMAGE = REPO / "benchmarks" / "performance" / "world_map.png"

from algan import *  # noqa: E402
from algan.mobs.image_mob import ImageMob  # noqa: E402
from algan.rendering.raytracing import scene_builder  # noqa: E402
from algan.rendering.raytracing import settings as rt_settings  # noqa: E402
from algan.scene_manager import SceneManager  # noqa: E402


def build_scene():
    torch.manual_seed(1234)
    with Off():
        fading = ImageMob(str(IMAGE)).scale(0.6).move(LEFT * 3).spawn()
        ImageMob(str(IMAGE)).scale(0.6).move(RIGHT * 3).spawn()
        proc = (
            Surface(color_texture=torch.rand(24, 24, 5).clamp(0.2, 1.0))
            .scale(0.8)
            .move(UP * 2)
            .spawn()
        )
        Cube().scale(0.5).move(DOWN * 1.5).spawn()
    with Sync(duration=0.6):
        fading.opacity = 0.15
        proc.opacity = 0.3


def set_arm(opacity_in_kernel, u8):
    rt_settings.set_texture_opacity_in_kernel(opacity_in_kernel)
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
        has_fade_region = False
        has_u8 = False
        has_f32_with_region = False
        for meta in self.metas:
            for row in meta:
                if int(row[0]) < 0:
                    continue
                if int(row[13]) >= 0 and int(row[14]) > 1:
                    has_fade_region = True
                if int(row[15]) >= 0:
                    has_u8 = True
                if int(row[15]) < 0 and int(row[13]) >= 0:
                    has_f32_with_region = True
        return has_fade_region, has_u8, has_f32_with_region


def render_arm(opacity_in_kernel, u8, out_name):
    set_arm(opacity_in_kernel, u8)
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
        f"arm opacity={opacity_in_kernel} u8={u8}: windows={watcher.windows} "
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
    u8_arm = render_arm(True, True, "arm_u8.mp4")
    op_arm = render_arm(True, False, "arm_opacity.mp4")
    legacy_arm = render_arm(False, False, "arm_legacy.mp4")

    problems = []
    if not (u8_arm.windows == op_arm.windows == legacy_arm.windows):
        print(
            "BATCH WINDOWS DIFFER BETWEEN ARMS -- comparison confounded:\n"
            f"  u8:     {u8_arm.windows}\n  op:     {op_arm.windows}\n"
            f"  legacy: {legacy_arm.windows}"
        )
        sys.exit(1)

    fade_region, has_u8, f32_region = u8_arm.meta_facts()
    if not fade_region:
        problems.append("no merge carried a per-frame opacity region (no real fade)")
    if not has_u8:
        problems.append("no merge stored a u8-packed map")
    if not f32_region:
        problems.append("no merge exercised the f32 fallback with an opacity region")
    op_fade, op_u8, _ = op_arm.meta_facts()
    if not op_fade:
        problems.append("opacity arm carried no fade region")
    if op_u8:
        problems.append("u8 engaged with TEXTURE_U8_STORAGE off")
    lg_fade, lg_u8, lg_f32 = legacy_arm.meta_facts()
    if lg_fade or lg_u8 or lg_f32:
        problems.append("legacy arm carried new-layout meta -- kill switch leaks")
    if not any(rows > min(u8_arm.tex_rows) * 2 for rows in legacy_arm.tex_rows):
        problems.append(
            "legacy arm never carried a dense premultiplied window -- the fade "
            "was not the contrast this A/B claims"
        )
    if max(u8_arm.tex_rows) * 3 > max(op_arm.tex_rows):
        problems.append(
            "u8 bank is not ~5x under the f32 bank -- either the packing or "
            "the content dedup (whose i32 compare guards NaN byte patterns) "
            "regressed"
        )
    for p in problems:
        print(f"VACUOUS: {p}")
    if problems:
        sys.exit(1)

    worst_u8, differing_u8 = compare("arm_u8.mp4", "arm_opacity.mp4")
    worst_op, _ = compare("arm_opacity.mp4", "arm_legacy.mp4")

    ok = True
    if worst_u8 != 0 or differing_u8 != 0:
        print("FAIL: u8 storage flip is not byte-identical")
        ok = False
    if worst_op > 2:
        print(
            "FAIL: in-sampler opacity flip exceeded the qualified tolerance "
            f"(max {worst_op} > 2)"
        )
        ok = False
    if not ok:
        sys.exit(1)
    print(
        "PASS: u8 flip byte-identical; opacity flip within tolerance "
        f"(max {worst_op} <= 2)"
    )


if __name__ == "__main__":
    main()
