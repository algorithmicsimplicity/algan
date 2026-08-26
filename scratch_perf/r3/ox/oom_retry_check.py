"""Forced out-of-memory retry check for the prefetch-overlap work (brief
point 4): pin the render arena low enough that prepared batches are rejected
and must shrink, then show the render completes -- with the overlap gate off,
and with the overlap machinery active.

Usage:
    uv run python scratch_perf/r3/ox/oom_retry_check.py <serial|overlap> [tag]

This box has no GPU. Two deviations from a T4 run are therefore required for
the overlap arm, and both are printed when they are applied:

* ``available_memory_override`` only stands in for a *measured* device
  (cuda/mps), so the equivalent CPU pin is ``max_cpu_memory_used``, which is
  what sizes a CPU render arena.
* ``project_on_gpu_active`` / ``merge_on_gpu_active`` hard-require CUDA, so
  the overlap arm patches exactly that device gate (plus a finite headroom
  stand-in and two predictor seed observations -- the warmth the first T4
  batch would create) and leaves every other line of the overlap path real.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time

ARM = sys.argv[1] if len(sys.argv) > 1 else "serial"
TAG = sys.argv[2] if len(sys.argv) > 2 else ARM

os.environ["ALGAN_USE_DAEMON"] = "0"
if ARM == "overlap":
    os.environ["ALGAN_PREFETCH_GPU_PREP"] = "1"

OUT_DIR = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT_DIR, exist_ok=True)

from algan import *  # noqa: E402,F403  (after the env pins, like every harness)
from algan.render_loop import RenderLoopMixin  # noqa: E402
from algan.settings import SETTINGS  # noqa: E402

# --- the pins ---------------------------------------------------------------
# Arena = rendering_memory_fraction (0.4) x max_cpu_memory_used on CPU. Small
# enough that the first prepared batch cannot fit, large enough that shrunken
# windows of this 32x32 scene render comfortably.
ARENA_TOTAL_MB = 1
SETTINGS.computing.set(max_cpu_memory_used=ARENA_TOTAL_MB * 1024 * 1024)
# Force several batches...
SETTINGS.computing.set(max_animation_batch_size=2)

RECORDS = {"preflights": [], "renders": [], "worker_prep": [], "seeds": []}

_orig_preflight = RenderLoopMixin._prepared_batch_fits_render_arena


def recorded_preflight(self, primitive_batch, *args, **kwargs):
    fits = None
    try:
        fits = _orig_preflight(self, primitive_batch, *args, **kwargs)
        return fits
    finally:
        RECORDS["preflights"].append(
            {
                "frames": kwargs.get("num_frames"),
                "fits": bool(fits),
                "overlapped": bool(
                    getattr(primitive_batch[0], "_rt_prep_overlapped", False)
                ),
            }
        )


RenderLoopMixin._prepared_batch_fits_render_arena = recorded_preflight

_orig_render = RenderLoopMixin.render_primitive_batch


def recorded_render(self, primitive_batch, start_ind, end_ind, *args, **kwargs):
    RECORDS["renders"].append([int(start_ind), int(end_ind)])
    return _orig_render(self, primitive_batch, start_ind, end_ind, *args, **kwargs)


RenderLoopMixin.render_primitive_batch = recorded_render

if ARM == "overlap":
    # --- the three documented seams (see module docstring) -------------------
    # 1. The device gate: keep its setting/env semantics, drop only the
    #    CUDA requirement and keep the worker-thread check.
    from algan.environment import env_flag

    def seam_gate(self):
        if not env_flag(
            "ALGAN_PREFETCH_GPU_PREP", SETTINGS.computing.prefetch_gpu_prep
        ):
            return False
        return threading.current_thread().name.startswith("algan-batch-prep")

    RenderLoopMixin._overlap_gpu_prep_active = seam_gate

    # 2. Pool headroom: inf off CUDA -> a generous finite stand-in for the
    #    T4's pool (the seeded predictions must clear it the way real
    #    measurements do).
    RenderLoopMixin._gpu_merge_headroom_bytes = lambda self: 4 * 1024 * 1024 * 1024

    # 3. Predictor warmth: on the T4 the first batch measures both peaks on
    #    the render thread before any successor is prefetched. Seed the same
    #    state after this job's first preflight (peak = seed factor x inputs,
    #    i.e. exactly what the uncalibrated model would have predicted).
    _recorded_preflight = RenderLoopMixin._prepared_batch_fits_render_arena

    def seeding_preflight(self, primitive_batch, *args, **kwargs):
        if not RECORDS["seeds"]:
            self._project_peak_ratio.observe(1_000_000, 8_000_000)
            self._merge_peak_ratio.observe(1_000_000, 6_000_000)
            RECORDS["seeds"].append(True)
        return _recorded_preflight(self, primitive_batch, *args, **kwargs)

    RenderLoopMixin._prepared_batch_fits_render_arena = seeding_preflight

    _orig_worker_prep = RenderLoopMixin._prepare_batch_on_worker

    def recorded_worker_prep(self, primitive_batch, render_state):
        t0 = time.perf_counter()
        try:
            return _orig_worker_prep(self, primitive_batch, render_state)
        finally:
            RECORDS["worker_prep"].append(
                {
                    "dt": time.perf_counter() - t0,
                    "thread": threading.current_thread().name,
                    "stamped": bool(
                        getattr(primitive_batch[0], "_rt_prep_overlapped", False)
                    ),
                }
            )

    RenderLoopMixin._prepare_batch_on_worker = recorded_worker_prep


def build_scene():
    SETTINGS.raytracing.set(shadows=True)
    with Off():
        light = PointLight().spawn()
        blocks = [
            Cube(side_length=0.5).move(RIGHT * ((i % 8) - 4) + UP * (i // 8)).spawn()
            for i in range(24)
        ]
    with Sync(run_time=5):
        light.move(UP * 3)
        for block in blocks:
            block.rotate(90, OUT)


def main():
    preset_name = "SMOKE_TEST"
    preset = SMOKE_TEST
    out_path = os.path.join(OUT_DIR, f"oom_{TAG}_{preset_name}.mp4")
    scene = SceneManager.reset()
    scene.set_video_settings(preset)
    build_scene()

    t0 = time.perf_counter()
    result = Scene.save_video(
        out_path,
        video_settings=preset,
        ffmpeg_params=["-c:v", "libx264rgb", "-qp", "0"],
    )
    wall = time.perf_counter() - t0

    pf = RECORDS["preflights"]
    summary = {
        "arm": ARM,
        "status": str(result.status),
        "output": str(result.output_path),
        "wall_s": round(wall, 3),
        "arena_total_mb": ARENA_TOTAL_MB,
        "preflight_calls": len(pf),
        "rejections": sum(1 for p in pf if not p["fits"]),
        "overlapped_preflights": sum(1 for p in pf if p["overlapped"]),
        "render_windows": RECORDS["renders"],
        "worker_prep": RECORDS["worker_prep"],
    }
    print("[summary] " + json.dumps(summary))
    with open(os.path.join(OUT_DIR, f"oom_{TAG}_{preset_name}_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
