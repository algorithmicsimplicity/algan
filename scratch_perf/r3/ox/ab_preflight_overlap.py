"""A/B harness for the prefetch GPU-prep overlap: one render per process.

Adapted from scratch_perf/r2/ox/ab_preflight_overlap.py, with two additions:

* ``--cpu-seams`` -- for a box with no CUDA. Patches exactly the device gate
  (worker-thread half only), a finite pool-headroom stand-in, and seeds the two
  peak predictors after the job's first preflight (the warmth the first batch
  creates on a CUDA render). Everything else in the overlap path runs for real.
  Do NOT pass it on the T4.
* the summary reports the overlapped fraction directly:
  ``overlapped_preflights / preflight_calls``.

The toggle is read live (ALGAN_PREFETCH_GPU_PREP), so the driver alternates
arms by launching this script repeatedly with the env flipped or the setting
set. Prints JSON-ish lines with wall time, batch windows, preflight stats, and
how many batches were prepared overlapped on the worker.

    ALGAN_PREFETCH_GPU_PREP=0 uv run python scratch_perf/r3/ox/ab_preflight_overlap.py PREVIEW off [override_mb] [--cpu-seams]
    ALGAN_PREFETCH_GPU_PREP=1 uv run python scratch_perf/r3/ox/ab_preflight_overlap.py PREVIEW on  [override_mb] [--cpu-seams]

Output videos land in scratch_perf/r3/ox/out/<tag>_<preset>.mp4. Compare arms:

    uv run python benchmarks/_video_diff.py <out_off>.mp4 <out_on>.mp4
"""

from __future__ import annotations

import os
import sys
import threading
import time

os.environ["ALGAN_USE_DAEMON"] = "0"

import json  # noqa: E402

import torch  # noqa: E402

from algan import *  # noqa: E402,F403
from algan.mobs.neural_nets.neural_net import NeuralNetMLPV3  # noqa: E402
from algan.render_loop import RenderLoopMixin  # noqa: E402
from algan.scene_manager import SceneManager  # noqa: E402
from algan.settings import SETTINGS  # noqa: E402

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
CPU_SEAMS = "--cpu-seams" in sys.argv
PRESET_NAME = ARGS[0] if len(ARGS) > 0 else "PREVIEW"
TAG = ARGS[1] if len(ARGS) > 1 else "ab"
if len(ARGS) > 2:
    OVERRIDE_MB = int(ARGS[2])
    # Only binds measured (cuda/mps) devices; harmless on a CPU box.
    SETTINGS.computing.set(available_memory_override=OVERRIDE_MB * 1024 * 1024)
PRESET = {"PREVIEW": PREVIEW, "HD": HD, "SMOKE_TEST": SMOKE_TEST}[PRESET_NAME]

OUT_DIR = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT_DIR, exist_ok=True)

RECORDS = {
    "preflights": [],
    "worker_prep": [],
    "windows": [],
    "seeds": [],
}


def install_recorders():
    orig_preflight = RenderLoopMixin._prepared_batch_fits_render_arena

    def preflight_recorded(self, primitive_batch, *args, **kwargs):
        t0 = time.perf_counter()
        fits = None
        try:
            fits = orig_preflight(self, primitive_batch, *args, **kwargs)
            return fits
        finally:
            RECORDS["preflights"].append(
                {
                    "dt": time.perf_counter() - t0,
                    "fits": bool(fits),
                    "overlapped": bool(
                        getattr(primitive_batch[0], "_rt_prep_overlapped", False)
                    ),
                }
            )

    RenderLoopMixin._prepared_batch_fits_render_arena = preflight_recorded

    if hasattr(RenderLoopMixin, "_prepare_batch_on_worker"):
        orig_worker = RenderLoopMixin._prepare_batch_on_worker

        def worker_recorded(self, primitive_batch, render_state):
            t0 = time.perf_counter()
            try:
                return orig_worker(self, primitive_batch, render_state)
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

        RenderLoopMixin._prepare_batch_on_worker = worker_recorded

    orig_render = RenderLoopMixin.render_primitive_batch

    def render_recorded(self, primitive_batch, start_ind, end_ind, *args, **kwargs):
        RECORDS["windows"].append([int(start_ind), int(end_ind)])
        return orig_render(self, primitive_batch, start_ind, end_ind, *args, **kwargs)

    RenderLoopMixin.render_primitive_batch = render_recorded

    if CPU_SEAMS:
        # The three documented seams for a non-CUDA box; see the module
        # docstring. The gate keeps its setting/env semantics and drops only
        # the CUDA requirement; everything below it runs for real.
        from algan.environment import env_flag

        def seam_gate(self):
            if not env_flag(
                "ALGAN_PREFETCH_GPU_PREP", SETTINGS.computing.prefetch_gpu_prep
            ):
                return False
            return threading.current_thread().name.startswith("algan-batch-prep")

        RenderLoopMixin._overlap_gpu_prep_active = seam_gate
        # Generous stand-in: the seeded 8-10x predictions must clear it the
        # way real T4 measurements (3-6x of a multi-GB pool) do.
        RenderLoopMixin._gpu_merge_headroom_bytes = lambda self: 4 * 1024 * 1024 * 1024

        def seeding_preflight(self, primitive_batch, *args, **kwargs):
            if not RECORDS["seeds"]:
                self._project_peak_ratio.observe(1_000_000, 8_000_000)
                self._merge_peak_ratio.observe(1_000_000, 6_000_000)
                RECORDS["seeds"].append(True)
            return preflight_recorded(self, primitive_batch, *args, **kwargs)

        RenderLoopMixin._prepared_batch_fits_render_arena = seeding_preflight


def scene_fn():
    run_time = 5
    SETTINGS.raytracing.set(shadows=True)
    with Off():
        nn = NeuralNetMLPV3([5, 5, 5, 5]).move(LEFT).spawn()
        x = (
            ImageMob("benchmarks/performance/world_map.png")
            .move_next_to(nn, LEFT)
            .spawn()
        )
        label = (
            Text("Neural Net MLP v3 processing an image of the globe")
            .move_next_to(nn, DOWN)
            .spawn()
        )
    with Sync(run_time=run_time):
        nn.move(UP)
        x.color_texture = x.color_texture * 0.5
        label.move(RIGHT * 2)


def main():
    install_recorders()
    out_path = os.path.join(OUT_DIR, f"{TAG}_{PRESET_NAME}.mp4")
    scene = SceneManager.reset()
    scene.set_video_settings(PRESET)
    scene_fn()
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    result = Scene.save_video(
        out_path,
        video_settings=PRESET,
        ffmpeg_params=["-c:v", "libx264rgb", "-qp", "0"],
    )
    wall = time.perf_counter() - t0

    pf = RECORDS["preflights"]
    wp = RECORDS["worker_prep"]
    overlapped = sum(1 for p in pf if p["overlapped"])
    summary = {
        "tag": TAG,
        "preset": PRESET_NAME,
        "cpu_seams": CPU_SEAMS,
        "overlap_env": os.environ.get("ALGAN_PREFETCH_GPU_PREP", "<unset>"),
        "wall_s": round(wall, 3),
        "status": str(result.status),
        "output": str(result.output_path),
        "preflight_calls": len(pf),
        "preflight_sum_s": round(sum(p["dt"] for p in pf), 3),
        "overlapped_preflights": overlapped,
        "overlapped_fraction": round(overlapped / len(pf), 3) if pf else None,
        "worker_prep_calls": len(wp),
        "worker_prep_sum_s": round(sum(w["dt"] for w in wp), 3),
        "windows": RECORDS["windows"],
    }
    print("[summary] " + json.dumps(summary))
    with open(os.path.join(OUT_DIR, f"{TAG}_{PRESET_NAME}_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
