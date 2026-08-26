"""A/B arm for the prefetch GPU-prep overlap: one render per process.

The toggle is read live (ALGAN_PREFETCH_GPU_PREP), so the driver alternates
arms by launching this script repeatedly with the env flipped. Prints JSON-ish
lines with wall time, peak VRAM, batch windows, preflight stats, and how many
batches were prepared overlapped on the worker.

    uv run python scratch_perf/r2/ox/ab_preflight_overlap.py PREVIEW <tag> <override_mb>

Output videos land in scratch_perf/r2/ox/out/<tag>_<preset>.mp4 (one per
process invocation; the caller names them).
"""
from __future__ import annotations

import os
import sys
import time

os.environ["ALGAN_USE_DAEMON"] = "0"

import json
import subprocess

import torch

from algan import *
from algan.mobs.neural_nets.neural_net import NeuralNetMLPV3
from algan.render_loop import RenderLoopMixin
from algan.scene_manager import SceneManager
from algan.settings import SETTINGS

PRESET_NAME = sys.argv[1] if len(sys.argv) > 1 else "PREVIEW"
TAG = sys.argv[2] if len(sys.argv) > 2 else "ab"
OVERRIDE_MB = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
SETTINGS.computing.set(available_memory_override=OVERRIDE_MB * 1024 * 1024)
PRESET = {"PREVIEW": PREVIEW, "HD": HD}[PRESET_NAME]

OUT_DIR = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT_DIR, exist_ok=True)

RECORDS = {
    "preflights": [],
    "worker_prep": [],
    "windows": [],
}


def gpu_snapshot():
    try:
        mem = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.strip()
        apps = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.strip()
        return f"{mem} | tenants: {apps}"
    except Exception as e:  # noqa: BLE001
        return f"<nvidia-smi failed: {e}>"


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

    orig_worker = RenderLoopMixin._prepare_batch_on_worker

    def worker_recorded(self, primitive_batch, render_state):
        t0 = time.perf_counter()
        try:
            return orig_worker(self, primitive_batch, render_state)
        finally:
            RECORDS["worker_prep"].append(time.perf_counter() - t0)

    RenderLoopMixin._prepare_batch_on_worker = worker_recorded

    orig_render = RenderLoopMixin.render_primitive_batch

    def render_recorded(self, primitive_batch, start_ind, end_ind, *args, **kwargs):
        RECORDS["windows"].append([int(start_ind), int(end_ind)])
        return orig_render(self, primitive_batch, start_ind, end_ind, *args, **kwargs)

    RenderLoopMixin.render_primitive_batch = render_recorded


def scene_fn():
    run_time = 5
    SETTINGS.raytracing.set(shadows=True)
    with Off():
        nn = NeuralNetMLPV3([5, 5, 5, 5]).move(LEFT).spawn()
        x = ImageMob("benchmarks/performance/world_map.png").move_next_to(
            nn, LEFT
        ).spawn()
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
    print(f"[gpu-before] {gpu_snapshot()}", flush=True)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    result = Scene.save_video(
        out_path,
        video_settings=PRESET,
        ffmpeg_params=["-c:v", "libx264rgb", "-qp", "0"],
    )
    wall = time.perf_counter() - t0
    print(f"[gpu-after] {gpu_snapshot()}", flush=True)

    pf = RECORDS["preflights"]
    wp = RECORDS["worker_prep"]
    summary = {
        "tag": TAG,
        "preset": PRESET_NAME,
        "overlap_env": os.environ.get("ALGAN_PREFETCH_GPU_PREP", "<unset>"),
        "wall_s": round(wall, 3),
        "status": result.status,
        "output": str(result.output_path),
        "peak_alloc_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 2**20, 1),
        "preflight_calls": len(pf),
        "preflight_sum_s": round(sum(p["dt"] for p in pf), 3),
        "preflight_first_s": round(pf[0]["dt"], 3) if pf else 0.0,
        "worker_prep_calls": len(wp),
        "worker_prep_sum_s": round(sum(wp), 3),
        "windows": RECORDS["windows"],
    }
    print("[summary] " + json.dumps(summary))
    with open(
        os.path.join(OUT_DIR, f"{TAG}_{PRESET_NAME}_summary.json"), "w"
    ) as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
