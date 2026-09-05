"""Alternate independent kernel instances to separate BVH gains from drift.

Both arms compile once in one process. Only the three kernels that traverse
BVHs are duplicated; no renderer setting, scene data or quality is changed.
The first render of each arm is excluded. Later A/B/B/A rounds reverse the
ordering bias. Run with the project venv, with no other GPU jobs active.
"""

from __future__ import annotations

import json
import os
import runpy
import time
from pathlib import Path

os.environ["ALGAN_USE_DAEMON"] = "0"

import torch

import algan.utils.profiling_utils as profiling
from algan import Scene
from algan.rendering.raytracing import raster_taichi, raytrace_kernels_taichi, wavefront_kernels_taichi
from algan.scene_manager import SceneManager
from algan.taichi_compat import ti
from benchmarks.performance.refit_link_control_taichi import legacy_refit_link


def main():
    captured = {}

    def capture(scene, quality, tag, **kwargs):
        captured.update(scene=scene, quality=quality, kwargs=kwargs["save_video_kwargs"])

    profiling.profile_scene = capture
    runpy.run_path(str(Path(__file__).with_name("nn_scene_UHD.py")), run_name="__main__")
    targets = [
        (raster_taichi, "raster_shadow_trace_arena"),
        (wavefront_kernels_taichi, "wavefront_traverse_events_arena"),
        (wavefront_kernels_taichi, "wavefront_shade_arena"),
    ]
    direct = [getattr(mod, name) for mod, name in targets]
    legacy = [ti.kernel(kernel._primal.func) for kernel in direct]
    direct_link = raytrace_kernels_taichi._refit_link
    out = Path(__file__).resolve().parent / "alternating_links"
    out.mkdir(exist_ok=True)
    rows = []
    sequence = ["direct", "legacy"] + ["legacy", "direct", "direct", "legacy"] * 2
    for index, arm in enumerate(sequence):
        kernels = direct if arm == "direct" else legacy
        raytrace_kernels_taichi._refit_link = direct_link if arm == "direct" else legacy_refit_link
        for (mod, name), kernel in zip(targets, kernels):
            setattr(mod, name, kernel)
        scene = SceneManager.reset()
        scene.set_video_settings(captured["quality"])
        captured["scene"]()
        torch.cuda.synchronize()
        start = time.perf_counter()
        Scene.save_video(str(out / f"{index}_{arm}.mp4"), reset=True, **captured["kwargs"])
        torch.cuda.synchronize()
        rows.append({"index": index, "arm": arm, "warm": index >= 2, "seconds": time.perf_counter() - start})
        print("ALTERNATING " + json.dumps(rows[-1]), flush=True)
        (out / "timings.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
