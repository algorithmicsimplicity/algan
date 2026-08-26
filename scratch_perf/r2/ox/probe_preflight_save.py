"""Part 1 probe: drive a real save_video of the nn benchmark scene with the
profiler installed, so the ``arena preflight (batch)`` stage shows up, and
record per-batch preflight calls / fetches / chosen windows on top.

Usage:
    uv run python scratch_perf/r2/ox/probe_preflight_save.py PREVIEW <tag> <runs> [override_mb]

Writes one JSON per run to scratch_perf/r2/ox/logs/preflight_<tag>_run<i>.json
and prints a compact per-run summary.
"""
from __future__ import annotations

import os
import sys
import time

os.environ["ALGAN_USE_DAEMON"] = "0"
os.environ.setdefault("ALGAN_LOG_LEVEL", "DEBUG")

import json
import logging
import subprocess

from algan import *
from algan.mobs.neural_nets.neural_net import NeuralNetMLPV3
from algan.render_loop import RenderLoopMixin
from algan.settings import SETTINGS
from algan.utils.profiling_utils import install_instrumentation, run_once

if len(sys.argv) > 4:
    override_mb = int(sys.argv[4])
    SETTINGS.computing.set(available_memory_override=override_mb * 1024 * 1024)

PRESET_NAME = sys.argv[1] if len(sys.argv) > 1 else "PREVIEW"
TAG = sys.argv[2] if len(sys.argv) > 2 else "base"
RUNS = int(sys.argv[3]) if len(sys.argv) > 3 else 4
PRESET = {"PREVIEW": PREVIEW, "HD": HD, "MD": MD}[PRESET_NAME]

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Capture the engine's DEBUG stream (batch windows, planner decisions,
# chunk plans, PERF retry notices) alongside the structured records below.
_alg_root = logging.getLogger("algan")
_fh = logging.FileHandler(
    os.path.join(LOG_DIR, f"preflight_{TAG}_debug.log"), mode="a", encoding="utf-8"
)
_fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
_alg_root.addHandler(_fh)


# --- per-call records -------------------------------------------------------
RECORDS = {"preflights": [], "fetches": [], "prefix_selects": []}


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
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.strip()
        return f"{mem} | tenants: {apps}"
    except Exception as e:  # noqa: BLE001
        return f"<nvidia-smi failed: {e}>"


def install_recorders():
    orig_preflight = RenderLoopMixin._prepared_batch_fits_render_arena

    def preflight_recorded(
        self,
        primitive_batch,
        render_state,
        post_processes,
        transparent_background,
        *,
        require_estimates_fit=True,
        num_frames=None,
    ):
        t0 = time.perf_counter()
        try:
            fits = orig_preflight(
                self,
                primitive_batch,
                render_state,
                post_processes,
                transparent_background,
                require_estimates_fit=require_estimates_fit,
                num_frames=num_frames,
            )
            return fits
        finally:
            RECORDS["preflights"].append(
                {
                    "dt": time.perf_counter() - t0,
                    "num_frames": int(num_frames) if num_frames else None,
                    "fits": bool(fits),
                    "require_estimates_fit": bool(require_estimates_fit),
                }
            )

    RenderLoopMixin._prepared_batch_fits_render_arena = preflight_recorded

    orig_fetch = RenderLoopMixin.get_batch_of_primitives

    def fetch_recorded(self, start_ind, max_end_ind, actors, max_mem):
        t0 = time.perf_counter()
        prims, new_end, rs = orig_fetch(self, start_ind, max_end_ind, actors, max_mem)
        RECORDS["fetches"].append(
            {
                "dt": time.perf_counter() - t0,
                "start": int(start_ind),
                "requested_end": int(max_end_ind),
                "chosen_end": int(new_end),
                "num_primitives": len(prims),
            }
        )
        return prims, new_end, rs

    RenderLoopMixin.get_batch_of_primitives = fetch_recorded

    orig_prefix = RenderLoopMixin._select_largest_fitting_fetched_prefix

    def prefix_recorded(self, primitives, render_state, duration, *args, **kwargs):
        result = orig_prefix(self, primitives, render_state, duration, *args, **kwargs)
        RECORDS["prefix_selects"].append(
            {
                "fetched_duration": int(duration),
                "chosen_duration": (int(result[1]) if result is not None else None),
                "used": result is not None,
            }
        )
        return result

    RenderLoopMixin._select_largest_fitting_fetched_prefix = prefix_recorded


STAGES_OF_INTEREST = (
    "ray traced render total",
    "Scene.get_batch_of_primitives",
    "arena preflight (batch)",
    "  - project_to_screen (prewarm)",
    "merge collections + build BVHs",
    "  - refit-BVH build (in merge)",
    "scene upload to arena",
    "logical PN: dice + shade + pack (project_to_screen)",
    "video encode tail (ffmpeg drain)",
    "memory reclaim (gc + cuda cache)",
)


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
    install_instrumentation()
    install_recorders()
    print(f"[probe] override={SETTINGS.computing.available_memory_override}")
    for i in range(1, RUNS + 1):
        RECORDS["preflights"].clear()
        RECORDS["fetches"].clear()
        RECORDS["prefix_selects"].clear()
        print(f"\n===== run {i}/{RUNS} ({'cold' if i == 1 else 'warm'}) =====")
        print(f"[gpu-before] {gpu_snapshot()}")
        t0 = time.perf_counter()
        res = run_once(
            scene_fn,
            PRESET,
            f"{TAG}_{PRESET_NAME}",
            i,
            telemetry=True,
            save_video_kwargs={"ffmpeg_params": ["-preset", "ultrafast"]},
        )
        wall = time.perf_counter() - t0
        print(f"[gpu-after] {gpu_snapshot()}")

        pf = RECORDS["preflights"]
        pf_dt = [p["dt"] for p in pf]
        first_pf = pf_dt[0] if pf_dt else 0.0
        summary = {
            "wall_s": wall,
            "run_total_s": res["total"],
            "peak_alloc_mb": res["peak_alloc_mb"],
            "peak_reserved_mb": res["peak_reserved_mb"],
            "stages": {
                name: {
                    "calls": res["counts"].get(name, 0),
                    "incl_s": round(res["times"].get(name, 0.0), 4),
                }
                for name in STAGES_OF_INTEREST
            },
            "preflight_calls": len(pf),
            "preflight_sum_s": round(sum(pf_dt), 4),
            "preflight_first_s": round(first_pf, 4),
            "preflight_dts_ms": [round(d * 1000, 1) for d in pf_dt],
            "preflight_num_frames": [p["num_frames"] for p in pf],
            "preflight_fits": [p["fits"] for p in pf],
            "fetches": list(RECORDS["fetches"]),
            "prefix_selects": list(RECORDS["prefix_selects"]),
        }
        path = os.path.join(LOG_DIR, f"preflight_{TAG}_{PRESET_NAME}_run{i}.json")
        with open(path, "w") as f:
            json.dump(summary, f, indent=1)
        print(
            f"[summary] wall {wall:.2f}s | batches={len(RECORDS['fetches'])} "
            f"preflights={len(pf)} preflight_sum={sum(pf_dt):.3f}s "
            f"first={first_pf:.3f}s ({100 * first_pf / max(sum(pf_dt), 1e-9):.0f}%)"
        )
        print(
            "[stages] "
            + " | ".join(
                f"{name.strip()}={res['times'].get(name, 0.0):.3f}s"
                f"(x{res['counts'].get(name, 0)})"
                for name in STAGES_OF_INTEREST
            )
        )
        print(f"[windows] {[f['chosen_end'] - f['start'] for f in RECORDS['fetches']]}")
        print(f"[prefix] {RECORDS['prefix_selects']}")
        print(f"[written] {path}")


if __name__ == "__main__":
    main()
