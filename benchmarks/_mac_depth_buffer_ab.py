"""Warm sheet-optimization A/B, with optional bracketing CPU controls.

Run with the venv interpreter directly. --matched runs all blocks on ONE Mac
VM. Each backend gets its own process, with both buffer variants warmed before
the measured A/B/B/A/A/B renders. Coarse timers add no synchronization.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import json
import os
import platform
import random
import signal
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path("algan_outputs/depth_buffer_ab")
PREFIX = "DEPTH_BUFFER_AB"


def run_logged(command, log_path, timeout, shutdown_timeout=60):
    """Preserve child output on disk and show it while the child is running."""
    finished = threading.Event()
    renders_complete = threading.Event()

    def forward():
        pending = ""
        with log_path.open() as reader:
            while True:
                text = reader.read()
                if text:
                    print(text, end="", flush=True)
                    pending += text
                    while "\n" in pending:
                        line, pending = pending.split("\n", 1)
                        if '"event": "renders_complete"' in line:
                            renders_complete.set()
                elif finished.is_set():
                    break
                else:
                    finished.wait(0.1)

    with log_path.open("w") as stream:
        process = subprocess.Popen(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=os.name == "posix",
        )
        forwarding = threading.Thread(target=forward, daemon=True)
        forwarding.start()
        deadline = time.monotonic() + timeout
        teardown_deadline = None
        try:
            try:
                while True:
                    if renders_complete.is_set() and teardown_deadline is None:
                        teardown_deadline = time.monotonic() + shutdown_timeout
                    remaining = (
                        min(deadline, teardown_deadline or deadline) - time.monotonic()
                    )
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(command, timeout)
                    try:
                        return process.wait(timeout=min(0.25, remaining))
                    except subprocess.TimeoutExpired:
                        continue
            except subprocess.TimeoutExpired:
                phase = "teardown" if renders_complete.is_set() else "render block"
                print(
                    f"{PREFIX} timeout: {log_path}, pid={process.pid}, phase={phase}",
                    flush=True,
                )
                # Only sample a timed-out process: profiling must not perturb
                # the measured renders. Keep diagnostics bounded as well.
                if sys.platform == "darwin":
                    with contextlib.suppress(OSError, subprocess.TimeoutExpired):
                        subprocess.run(
                            [
                                "sample",
                                str(process.pid),
                                "3",
                                "-file",
                                str(log_path.with_suffix(".sample.txt")),
                            ],
                            stdout=stream,
                            stderr=subprocess.STDOUT,
                            timeout=8,
                        )
                if os.name == "posix":
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait(timeout=15)
                return 124
        finally:
            stream.flush()
            finished.set()
            forwarding.join(timeout=5)


def matched(args):
    ROOT.mkdir(parents=True, exist_ok=True)
    outcomes = []
    for tag, device, sequence in (
        ("cpu_before", "cpu", "BB"),
        ("mps", "mps", args.sequence),
        ("cpu_after", "cpu", "BB"),
    ):
        command = [
            sys.executable,
            "-u",
            __file__,
            "--device",
            device,
            "--tag",
            tag,
            "--sequence",
            sequence,
            "--quality",
            args.quality,
            "--arena-mib",
            str(args.arena_mib),
            "--target",
            args.target,
        ]
        print(f"{PREFIX} block_start: {tag} ({sequence})", flush=True)
        started = time.perf_counter()
        code = run_logged(
            command, ROOT / f"{tag}.log", timeout=1600 if device == "mps" else 500
        )
        print(f"{PREFIX} block_end: {tag}, exit_code={code}", flush=True)
        outcomes.append(
            {"tag": tag, "exit_code": code, "wall": time.perf_counter() - started}
        )
        (ROOT / "outcomes.json").write_text(json.dumps(outcomes, indent=2))
    return int(any(row["exit_code"] for row in outcomes))


def render(args):
    os.environ.update(
        ALGAN_RENDER_DEVICE=args.device,
        ALGAN_ANIMATION_DEVICE="cpu",
        ALGAN_TORCH_COMPILE="0",
        ALGAN_USE_DAEMON="0",
        ALGAN_VIDEO_ENCODER="software",
    )
    import numpy as np
    import psutil
    import torch

    from algan import (
        DOWN,
        LEFT,
        PREVIEW,
        RIGHT,
        SETTINGS,
        UHD,
        UP,
        ImageMob,
        Off,
        Scene,
        SceneManager,
        Sync,
        Text,
    )
    from algan.mobs.neural_nets.neural_net import NeuralNetMLPV3
    from algan.render_loop import RenderLoopMixin
    from algan.rendering import mps_zero_copy as zc
    from algan.rendering import taichi_runtime as runtime
    from algan.rendering.post_processing.bloom_kernels_taichi import (
        can_use_bloom_taichi,
    )
    from algan.rendering.raytracing import (
        sheet_depth_taichi,
        sheet_sibling_taichi,
        sheets,
        tracer,
    )
    from algan.settings._startup import render_device
    from algan.taichi_compat import ti
    from algan.utils import memory_utils as mu

    torch.set_num_threads(3)
    runtime.init_taichi()
    if torch.device(render_device()).type != args.device:
        raise RuntimeError("The requested render device did not engage")
    if args.device == "mps" and str(runtime._live_arch()) != str(ti.metal):
        raise RuntimeError("This measurement requires the Metal compiler backend")
    assert can_use_bloom_taichi(torch.device(args.device))
    SETTINGS.raytracing.experimental.set(
        sheet_sample_depth_kernel=True, sheet_depth_reduce_kernel=True
    )
    output = ROOT / args.tag
    output.mkdir(parents=True, exist_ok=True)
    state = {"run": 0, "chunks": 0, "batches": 0, "arenas": []}
    totals = defaultdict(lambda: {"calls": 0, "seconds": 0.0})

    def emit(event, **values):
        line = json.dumps(
            {
                "event": event,
                "device": args.device,
                "run": state["run"],
                "timestamp": time.time(),
                **values,
            },
            sort_keys=True,
        )
        with (output / "events.jsonl").open("a") as stream:
            stream.write(line + "\n")
        print(PREFIX + " " + line, flush=True)

    def pool():
        try:
            rss = psutil.Process().memory_info().rss
        except psutil.Error:
            rss = None
        result = {"rss": rss, "host_available": psutil.virtual_memory().available}
        if args.device == "mps":
            result.update(
                driver=torch.mps.driver_allocated_memory(),
                live=torch.mps.current_allocated_memory(),
                recommended=torch.mps.recommended_max_memory(),
                imports=zc.cache_stats(),
            )
            result["swap"] = subprocess.run(
                ["sysctl", "vm.swapusage"], capture_output=True, text=True, timeout=10
            ).stdout
        return result

    def replace_aliases(original, replacement):
        for module_name, module in list(sys.modules.items()):
            if module_name.startswith("algan") and module is not None:
                for name, value in list(vars(module).items()):
                    if value is original:
                        setattr(module, name, replacement)

    def hook(module, name):
        original = getattr(module, name)

        @functools.wraps(original)
        def wrapped(*a, **kw):
            started = time.perf_counter()
            try:
                return original(*a, **kw)
            finally:
                totals[name]["calls"] += 1
                totals[name]["seconds"] += time.perf_counter() - started
                if name == "_lane_first_owners":
                    # Shape metadata only: no tensor reads or GPU waits.
                    size = a[3] * sheets.AA_NUM_SAMPLES * 4
                    row = totals[name]
                    row["table_bytes_total"] = row.get("table_bytes_total", 0) + size
                    row["table_bytes_max"] = max(row.get("table_bytes_max", 0), size)

        replace_aliases(original, wrapped)

    for name in ("_lane_first_owners", "compact_sheets", "_sibling_weights"):
        hook(sheets, name)
    for name in ("sheet_lane_depths", "sheet_lane_depths_inplace"):
        hook(sheet_depth_taichi, name)
    for name in ("sibling_band_counts", "sibling_coverage_weights"):
        hook(sheet_sibling_taichi, name)

    original_arena = mu.ManualMemory.__init__

    @functools.wraps(original_arena)
    def arena_init(self, portion, device=None, managed=True, *, num_bytes=None):
        if managed and portion > 0:
            num_bytes = args.arena_mib * 2**20
        original_arena(
            self, portion, device=device, managed=managed, num_bytes=num_bytes
        )
        if managed and portion > 0:
            state["arenas"].append(len(self))

    mu.ManualMemory.__init__ = arena_init
    original_batch = RenderLoopMixin._render_primitive_batch

    @functools.wraps(original_batch)
    def batch(self, *a, **kw):
        state["batches"] += 1
        yield from original_batch(self, *a, **kw)

    RenderLoopMixin._render_primitive_batch = batch
    original_wavefront = tracer.raytrace_render_wavefront

    @functools.wraps(original_wavefront)
    def wavefront(*a, **kw):
        state["chunks"] += 1
        emit("chunk_start", chunk=state["chunks"])
        started = time.perf_counter()
        result = original_wavefront(*a, **kw)
        emit(
            "chunk_end",
            chunk=state["chunks"],
            wall=time.perf_counter() - started,
            imports=zc.cache_stats(),
        )
        return result

    replace_aliases(original_wavefront, wavefront)
    hook(tracer, "raytrace_render_wavefront")
    preset = {"UHD": UHD, "PREVIEW": PREVIEW}[args.quality]
    world_map = Path(__file__).resolve().parent / "performance/world_map.png"
    emit(
        "metadata",
        python=sys.version,
        torch=torch.__version__,
        platform=platform.platform(),
        compiler=str(ti.__version__),
        arch=str(runtime._live_arch()),
        threads=torch.get_num_threads(),
        quality=args.quality,
        target=args.target,
        arena_mib=args.arena_mib,
        sequence=args.sequence,
        seed=20260908,
        pool=pool(),
    )
    seen = set()
    for run, arm in enumerate(args.sequence, 1):
        state.update(run=run, chunks=0, batches=0, arenas=[])
        totals.clear()
        reuse = arm == "B"
        cold = arm not in seen
        seen.add(arm)
        random.seed(20260908)
        np.random.seed(20260908)
        torch.manual_seed(20260908)
        SceneManager.reset()
        Scene.set_video_settings(preset)
        SETTINGS.raytracing.set(shadows=False)
        SETTINGS.raytracing.experimental.set(
            sheet_depth_buffer_reuse=reuse and args.target == "depth",
            sheet_sibling_weights_kernel=reuse and args.target == "sibling",
        )
        with Off():
            nn = NeuralNetMLPV3([5, 5, 5, 5]).move(LEFT).spawn()
            image = ImageMob(str(world_map)).move_next_to(nn, LEFT).spawn()
            label = (
                Text("Neural Net MLP v3 processing an image of the globe")
                .move_next_to(nn, DOWN)
                .spawn()
            )
        with Sync(runtime=0.3):
            nn.move(UP)
            image.color_texture = image.color_texture * 0.5
            label.move(RIGHT * 2)
        emit("render_start", arm=arm, cold=cold, pool=pool())
        started = time.perf_counter()
        result = Scene.save_video(
            str(output / f"run{run}_{arm}.mp4"),
            video_settings=preset,
            reset=True,
            ffmpeg_params=["-crf", "17", "-preset", "ultrafast"],
        )
        wall = time.perf_counter() - started
        if args.target == "depth":
            expected_kernel = (
                "sheet_lane_depths_inplace" if reuse else "sheet_lane_depths"
            )
            other_kernel = "sheet_lane_depths" if reuse else "sheet_lane_depths_inplace"
            assert totals[expected_kernel]["calls"] > 0
            assert totals[other_kernel]["calls"] == 0
        else:
            assert totals["sheet_lane_depths"]["calls"] > 0
            assert totals["sheet_lane_depths_inplace"]["calls"] == 0
            for name in ("sibling_band_counts", "sibling_coverage_weights"):
                assert (totals[name]["calls"] > 0) == reuse
        emit(
            "render_end",
            arm=arm,
            cold=cold,
            wall=wall,
            totals=dict(totals),
            chunks=state["chunks"],
            batches=state["batches"],
            arenas=state["arenas"],
            pool=pool(),
            converted_launches=dict(zc.STATS),
            output_path=str(result.output_path),
        )
    emit("renders_complete")
    # If interpreter/compiler teardown hangs after the final render, retain a
    # Python stack as well as the parent's native timeout sample.
    import faulthandler

    faulthandler.dump_traceback_later(30, repeat=True)
    return 0


def main():
    global ROOT, PREFIX
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matched", action="store_true")
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--quality", choices=["UHD", "PREVIEW"], default="UHD")
    parser.add_argument("--tag", default="local")
    parser.add_argument("--sequence", default="ABABBAAB")
    parser.add_argument("--arena-mib", type=int, default=1720)
    parser.add_argument("--target", choices=["depth", "sibling"], default="depth")
    args = parser.parse_args()
    if not args.sequence or set(args.sequence) - {"A", "B"}:
        parser.error("--sequence must contain only A (original) and B (reuse)")
    if args.target == "sibling":
        ROOT = Path("algan_outputs/sibling_weights_ab")
        PREFIX = "SIBLING_WEIGHTS_AB"
    return matched(args) if args.matched else render(args)


if __name__ == "__main__":
    raise SystemExit(main())
