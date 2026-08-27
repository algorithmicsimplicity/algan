"""Warm in-process alternating A/B: the textured-fade cost of the premultiply.

The scene is the case TEXTURE_OPACITY_IN_KERNEL exists for -- a large image
texture whose mob opacity animates the whole time (so under the legacy
premultiply EVERY batch rebuilds, re-decodes and re-uploads the full map per
frame, and the batch sizer prices it per frame), beside a second static copy
of the same image and a moving cube so this stays a general moving scene.

Arms alternate in one process (new, legacy, new, legacy) after a discarded
warm-up, flipping TEXTURE_OPACITY_IN_KERNEL + TEXTURE_U8_STORAGE through
their setters -- host-side, data-driven reads, no ``ti.static`` gate, so the
in-process flip is legal. Reported per render: wall seconds, the batch
windows chosen, and the merged texture-bank rows (which prove the arm
engaged rather than trusting the flip).

    uv run python benchmarks/performance/texfade_ab_PREVIEW.py
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("ALGAN_USE_DAEMON", "0")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pathlib import Path  # noqa: E402

import torch  # noqa: E402

REPO = Path(__file__).resolve().parent.parent.parent
OUT_DIR = REPO / "algan_outputs" / "_texfade_ab"
IMAGE = Path(__file__).resolve().parent / "world_map.png"

from algan import *  # noqa: E402
from algan.mobs.image_mob import ImageMob  # noqa: E402
from algan.rendering.raytracing import scene_builder  # noqa: E402
from algan.rendering.raytracing import settings as rt_settings  # noqa: E402
from algan.scene_manager import SceneManager  # noqa: E402


def build_scene():
    torch.manual_seed(7)
    with Off():
        fading = ImageMob(str(IMAGE)).scale(0.7).move(LEFT * 2.5).spawn()
        ImageMob(str(IMAGE)).scale(0.7).move(RIGHT * 2.5).spawn()
        cube = Cube().scale(0.5).move(DOWN * 2).spawn()
    # The fade spans the whole clip: every batch carries an animating opacity.
    with Sync(run_time=3.0):
        fading.opacity = 0.1
        cube.move(RIGHT * 2)


class MergeWatcher:
    def __init__(self):
        self.tex_rows = []
        self.windows = []
        self._orig = None

    def attach(self):
        watcher = self
        self._orig = scene_builder._merge_scene

        def watching(prims, **kwargs):
            m = watcher._orig(prims, **kwargs)
            watcher.tex_rows.append(int(m["textures"].shape[1]))
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


def render_once(new_path, label):
    rt_settings.set_texture_opacity_in_kernel(new_path)
    rt_settings.set_texture_u8_storage(new_path)
    watcher = MergeWatcher()
    watcher.attach()
    SceneManager.reset()
    t0 = time.time()
    try:
        with Scene() as scene:
            watcher.wrap_scene(scene)
            build_scene()
            scene.save_video(
                str(OUT_DIR / f"{label}.mp4"),
                video_settings=PREVIEW,
                overwrite=True,
            )
    finally:
        watcher.detach()
        rt_settings.set_texture_opacity_in_kernel(True)
        rt_settings.set_texture_u8_storage(True)
    dt = time.time() - t0
    print(
        f"[{label}] {dt:6.2f} s  windows={watcher.windows} "
        f"max_tex_rows={max(watcher.tex_rows) if watcher.tex_rows else 0}",
        flush=True,
    )
    return dt, watcher


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    render_once(True, "warmup")
    times = {"new": [], "legacy": []}
    banks = {"new": 0, "legacy": 0}
    for i in range(2):
        dt, w = render_once(True, f"new_{i}")
        times["new"].append(dt)
        banks["new"] = max(banks["new"], max(w.tex_rows))
        dt, w = render_once(False, f"legacy_{i}")
        times["legacy"].append(dt)
        banks["legacy"] = max(banks["legacy"], max(w.tex_rows))
    new_best = min(times["new"])
    legacy_best = min(times["legacy"])
    print(
        f"RESULT texfade PREVIEW: new {new_best:.2f} s vs legacy "
        f"{legacy_best:.2f} s ({legacy_best / max(new_best, 1e-9):.2f}x); "
        f"bank rows {banks['new']} vs {banks['legacy']} "
        f"({banks['legacy'] / max(banks['new'], 1):.1f}x)"
    )
    if banks["legacy"] <= banks["new"]:
        print("VACUOUS: the legacy arm did not carry a larger texture bank")
        sys.exit(1)


if __name__ == "__main__":
    main()
