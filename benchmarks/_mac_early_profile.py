"""Attribute the first two warm chunks without inserting device barriers.

Run CPU then MPS in fresh processes on one Mac. Each does cold, warm control,
warm detailed, warm control renders. Detailed hooks time only existing calls;
ATen timings are host wall time, NOT device kernel time or graph-compile time.
The detailed window ends at the start of chunk 3, matching run 70's window.
Use --quality PREVIEW --devices cpu for a local harness smoke check.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import gc
import json
import os
import platform
import random
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", choices=("cpu", "mps"))
    parser.add_argument("--devices", nargs="+", default=["cpu", "mps"])
    parser.add_argument("--quality", default="UHD")
    parser.add_argument("--arena-mib", type=int, default=1720)
    parser.add_argument("--timeout", type=int, default=850)
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--native-sample", action="store_true")
    parser.add_argument("--native-graphs", action="store_true")
    parser.add_argument("--bloom-ab", action="store_true")
    return parser.parse_args()


class Timers:
    def __init__(self):
        self.enabled = False
        self.phase = "off"
        self.local = threading.local()
        self.rows = defaultdict(lambda: [0, 0.0, 0.0, 0.0])
        self.slow = []
        self.origin = 0.0

    @contextlib.contextmanager
    def span(self, name):
        if not self.enabled:
            yield
            return
        stack = getattr(self.local, "stack", None)
        if stack is None:
            self.local.stack = stack = []
        parent = stack[-1][0] if stack else "root"
        frame = [name, time.perf_counter(), 0.0]
        phase = self.phase
        stack.append(frame)
        try:
            yield
        finally:
            ended = time.perf_counter()
            elapsed = ended - frame[1]
            stack.pop()
            if stack:
                stack[-1][2] += elapsed
            key = (phase, threading.current_thread().name, parent, name)
            row = self.rows[key]
            row[0] += 1
            row[1] += elapsed
            row[2] += elapsed - frame[2]
            row[3] = max(row[3], elapsed)
            if elapsed >= 0.005:
                self.slow.append({"phase": phase, "thread": key[1], "parent": parent,
                                  "name": name, "start": frame[1] - self.origin,
                                  "end": ended - self.origin,
                                  "self_seconds": elapsed - frame[2]})

    def wrap(self, function, name):
        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            if not self.enabled:
                return function(*args, **kwargs)
            with self.span(name):
                return function(*args, **kwargs)
        return wrapped

    def data(self):
        return {"rows": [dict(phase=k[0], thread=k[1], parent=k[2], name=k[3],
                              count=v[0], seconds=v[1], self_seconds=v[2], maximum=v[3])
                         for k, v in self.rows.items()], "slow_calls": self.slow}


def child(args):
    # No backend changes inside this process: both compilers retain warm caches.
    os.environ.update(ALGAN_RENDER_DEVICE=args.child, ALGAN_ANIMATION_DEVICE="cpu",
                      ALGAN_TORCH_COMPILE="0", ALGAN_USE_DAEMON="0",
                      ALGAN_VIDEO_ENCODER="software")
    import numpy as np
    import psutil
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode

    from algan import (HD, MD, PREVIEW, UHD, DOWN, LEFT, RIGHT, UP, ImageMob,
                       Off, Scene, SceneManager, SETTINGS, Sync, Text)
    from algan.mobs.neural_nets.neural_net import NeuralNetMLPV3
    from algan.render_loop import RenderLoopMixin
    from algan.rendering import mps_zero_copy as zc
    from algan.rendering import taichi_runtime as runtime
    from algan.rendering.raytracing import scene_builder as sb
    from algan.rendering.raytracing import tracer
    from algan.taichi_compat import submodule, ti
    from algan.utils import memory_utils as mu
    from algan.utils import profiling_utils
    import algan.render_loop as render_loop

    if args.child == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("This measurement requires MPS; CPU fallback is forbidden")
    runtime.init_taichi()
    timer = Timers()
    output = Path("algan_outputs/early_profile") / args.child
    output.mkdir(parents=True, exist_ok=True)
    events = output / "events.jsonl"
    mode = None
    sample_process = None
    state = {"run": 0, "chunks": 0, "batches": 0, "arenas": [], "started": 0.0}
    native = None
    bloom_times = defaultdict(lambda: [0, 0.0])
    candidate = False

    def emit(kind, **values):
        record = {"event": kind, "device": args.child, "run": state["run"], **values}
        line = json.dumps(record, sort_keys=True)
        with events.open("a") as stream:
            stream.write(line + "\n")
        print("EARLY_PROFILE " + line, flush=True)

    def pool():
        try:
            rss = psutil.Process().memory_info().rss
        except psutil.Error:
            # Some local PID namespaces do not expose this process in /proc.
            rss = None
        result = {"rss": rss, "host_available": psutil.virtual_memory().available}
        if torch.backends.mps.is_available():
            result.update(driver=torch.mps.driver_allocated_memory(),
                          live=torch.mps.current_allocated_memory(),
                          recommended=torch.mps.recommended_max_memory())
        return result

    if args.native_graphs and args.child == "mps":
        import ctypes
        library = output / "graph_timers.dylib"
        source = Path(__file__).with_name("_mac_graph_timers.mm")
        subprocess.run(["clang++", "-std=c++17", "-O2", "-dynamiclib", "-framework", "Foundation",
                        str(source), "-o", str(library)], check=True, timeout=60)
        native = ctypes.CDLL(str(library.resolve()))
        native.algan_graph_install.restype = ctypes.c_int
        native.algan_graph_phase.argtypes = [ctypes.c_int]
        native.algan_graph_stats.restype = ctypes.c_char_p
        installed = native.algan_graph_install()
        emit("native_graph_hooks", installed=installed)
        if installed != 7:
            raise RuntimeError(f"Native graph method ABI check failed: {installed}")

    def replace_aliases(original, replacement):
        for module_name, module in list(sys.modules.items()):
            if module_name.startswith("algan") and module is not None:
                for key, value in list(vars(module).items()):
                    if value is original:
                        setattr(module, key, replacement)

    def hook(module, name, label):
        original = getattr(module, name)
        replacement = timer.wrap(original, label)
        setattr(module, name, replacement)
        replace_aliases(original, replacement)

    if args.bloom_ab:
        from algan.rendering.post_processing import bloom as bloom_module
        from algan.rendering.post_processing import bloom_kernels_taichi as bloom_kernels
        original_bloom_gate = bloom_kernels.can_use_bloom_taichi

        def bloom_gate(device):
            device = torch.device(device)
            if candidate and device.type == "mps":
                return runtime._live_arch() == ti.metal and zc.zero_copy_available()
            return original_bloom_gate(device)

        bloom_kernels.can_use_bloom_taichi = bloom_gate

        def time_bloom(function, name):
            @functools.wraps(function)
            def call(*a, **kw):
                started = time.perf_counter()
                try:
                    return function(*a, **kw)
                finally:
                    bloom_times[name][0] += 1
                    bloom_times[name][1] += time.perf_counter() - started
            return call

        for name in ("_downsample_bloom", "_upsample_bloom"):
            setattr(bloom_module, name, time_bloom(getattr(bloom_module, name), name))

    # Hooks are dormant in controls. No extra synchronize(), tensor copies,
    # gc.collect(), empty_cache(), import-cache clearing, or prefetch changes.
    for name in ("release_torch_memory", "get_num_available_bytes"):
        hook(mu, name, "memory." + name)
    hook(gc, "collect", "memory.gc.collect")
    hook(torch.mps, "empty_cache", "memory.mps.empty_cache")
    hook(torch.mps, "synchronize", "wait.torch.mps.synchronize")
    hook(ti, "sync", "wait.quadrants.sync")
    hook(zc, "clear_import_cache", "imports.clear_cache")
    hook(zc, "import_tensor", "imports.argument")
    external = getattr(submodule("lang._ndarray"), "ExternalMetalNdarray", None)
    if external is not None:
        hook(external, "__init__", "imports.create_external_ndarray")

    original_arena = mu.ManualMemory.__init__

    @functools.wraps(original_arena)
    def arena_init(self, portion, device=None, managed=True, *, num_bytes=None):
        if managed and portion > 0:
            num_bytes = args.arena_mib * 2**20
        with timer.span("memory.arena_allocation" if managed else "memory.unmanaged_init"):
            original_arena(self, portion, device=device, managed=managed, num_bytes=num_bytes)
        if managed and portion > 0:
            state["arenas"].append(len(self))

    mu.ManualMemory.__init__ = arena_init
    original_get = mu.ManualMemory.get_tensor

    @functools.wraps(original_get)
    def get_tensor(self, *a, **kw):
        if not timer.enabled:
            return original_get(self, *a, **kw)
        with timer.span("memory.arena_view" if self.managed else "memory.pool_allocation"):
            return original_get(self, *a, **kw)

    mu.ManualMemory.get_tensor = get_tensor
    for name in ("_prepare_merged_host_scene", "_select_largest_fitting_fetched_prefix",
                 "_prepared_batch_fits_render_arena", "_get_batch_of_primitives",
                 "_prewarm_render_batch", "_prepare_batch_on_worker", "_materialize_render_state"):
        hook(RenderLoopMixin, name, "prepare." + name)
    for name in ("_merge_scene", "copy_merged_scene_to_arena", "_prefill_background",
                 "_pack_lights", "build_deferred_bvhs"):
        hook(sb, name, "scene." + name)
    hook(render_loop, "_prepare_background_for_chunk", "scene.background_for_chunk")

    Kernel = submodule("lang.kernel_impl").Kernel
    original_kernel = Kernel.__call__

    @functools.wraps(original_kernel)
    def kernel_call(self, *a, **kw):
        if not timer.enabled:
            return original_kernel(self, *a, **kw)
        name = getattr(getattr(self, "func", None), "__name__", "unknown")
        with timer.span("kernel." + name):
            return original_kernel(self, *a, **kw)

    Kernel.__call__ = kernel_call
    profiling_utils.stage = lambda name, items=None: timer.span("stage." + name)

    class OperatorTimes(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            # Metadata only. Never retain tensors in the trace, and never read
            # tensor values to the host; neither shapes nor devices need a wait.
            def metadata(value):
                if isinstance(value, torch.Tensor):
                    return f"{value.device.type}:{value.dtype}:{tuple(value.shape)}"
                if isinstance(value, (tuple, list)):
                    return "[" + ",".join(metadata(x) for x in value[:3]) + "]"
                if isinstance(value, (str, int, float, bool, type(None), torch.dtype, torch.device)):
                    return str(value)
                return type(value).__name__

            signature = ";".join(metadata(x) for x in args[:3]) + ":" + str(kwargs.get("device", ""))
            with timer.span("aten." + str(func) + " " + signature):
                return func(*args, **kwargs)

    original_batch = RenderLoopMixin._render_primitive_batch

    @functools.wraps(original_batch)
    def batch(self, *a, **kw):
        state["batches"] += 1
        # Do not put a timer around yield-from: that charges the consumer's
        # encoding and subsequent rendering to this generator.
        yield from original_batch(self, *a, **kw)

    RenderLoopMixin._render_primitive_batch = batch
    original_wavefront = tracer.raytrace_render_wavefront

    @functools.wraps(original_wavefront)
    def wavefront(*a, **kw):
        nonlocal mode
        state["chunks"] += 1
        chunk = state["chunks"]
        elapsed = time.perf_counter() - state["started"]
        if chunk == 3 and mode is not None:
            mode.__exit__(None, None, None)
            mode = None
            timer.enabled = False
            if native is not None:
                native.algan_graph_phase(-1)
            # Preserve the completed window even if a later full render hits
            # the process budget. This I/O is outside the measured early
            # window; its separately reported duration remains in full wall.
            checkpoint = time.perf_counter()
            (output / "timings.json").write_text(json.dumps(timer.data(), indent=2))
            if native is not None:
                stats = json.loads(native.algan_graph_stats())
                (output / "native_graphs.json").write_text(json.dumps(stats, indent=2))
                emit("native_graph_stats", rows=stats)
            emit("profile_checkpoint", early_window=elapsed,
                 write_seconds=time.perf_counter() - checkpoint)
        if native is not None:
            native.algan_graph_phase(2 * chunk - 1 if timer.enabled and chunk <= 2 else -1)
        timer.phase = f"chunk{chunk}"
        emit("chunk_start", chunk=chunk, elapsed=elapsed, pool=pool())
        started = time.perf_counter()
        try:
            return original_wavefront(*a, **kw)
        finally:
            emit("chunk_end", chunk=chunk, elapsed=time.perf_counter() - state["started"],
                 wall=time.perf_counter() - started,
                 converted_launches=dict(zc.STATS))
            timer.phase = f"after_chunk{chunk}"
            if native is not None:
                native.algan_graph_phase(2 * chunk if timer.enabled and chunk <= 2 else -1)

    tracer.raytrace_render_wavefront = wavefront
    replace_aliases(original_wavefront, wavefront)
    preset = {"UHD": UHD, "HD": HD, "MD": MD, "PREVIEW": PREVIEW}[args.quality.upper()]
    world_map = Path(__file__).resolve().parent / "performance/world_map.png"

    def scene():
        SETTINGS.raytracing.set(shadows=False)
        with Off():
            nn = NeuralNetMLPV3([5, 5, 5, 5]).move(LEFT).spawn()
            image = ImageMob(str(world_map)).move_next_to(nn, LEFT).spawn()
            label = Text("Neural Net MLP v3 processing an image of the globe").move_next_to(nn, DOWN).spawn()
        with Sync(runtime=0.3):
            nn.move(UP)
            image.color_texture = image.color_texture * 0.5
            label.move(RIGHT * 2)

    emit("metadata", python=sys.version, torch=torch.__version__, platform=platform.platform(),
         taichi=str(getattr(ti, "__version__", "unknown")), arch=str(runtime._live_arch()),
         threads=torch.get_num_threads(), quality=args.quality, arena_mib=args.arena_mib,
         cpu_count=os.cpu_count(), pool=pool(), seed=20260908,
         env={k: v for k, v in os.environ.items() if k.startswith(("ALGAN_", "PYTORCH_MPS_"))})
    for run in range(1, args.runs + 1):
        state.update(run=run, chunks=0, batches=0, arenas=[])
        candidate = args.bloom_ab and run % 2 == 0
        bloom_times.clear()
        random.seed(20260908)
        np.random.seed(20260908)
        torch.manual_seed(20260908)
        SceneManager.reset()
        Scene.set_video_settings(preset)
        scene()
        detailed = run == 3 and (not args.bloom_ab or args.native_graphs)
        emit("render_start", detailed=detailed, bloom_candidate=candidate, pool=pool())
        if detailed and args.native_sample and sys.platform == "darwin":
            sample_process = subprocess.Popen(["/usr/bin/sample", str(os.getpid()), "25", "2",
                                               "-file", str(output / "native_sample.txt")],
                                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        timer.origin = state["started"] = time.perf_counter()
        timer.phase = "prelude"
        if detailed:
            timer.enabled = True
            if native is not None:
                native.algan_graph_phase(0)
            mode = OperatorTimes()
            mode.__enter__()
        try:
            Scene.save_video(str(output / f"run{run}.mp4"), video_settings=preset, reset=True,
                             ffmpeg_params=["-crf", "17", "-preset", "ultrafast"])
        finally:
            if mode is not None:
                mode.__exit__(None, None, None)
                mode = None
            timer.enabled = False
            if native is not None:
                native.algan_graph_phase(-1)
        elapsed = time.perf_counter() - state["started"]
        emit("render_end", detailed=detailed, wall=elapsed, chunks=state["chunks"],
             batches=state["batches"], arenas=state["arenas"], pool=pool(),
             bloom_candidate=candidate, bloom_times=dict(bloom_times))
        if detailed:
            (output / "timings.json").write_text(json.dumps(timer.data(), indent=2))
            if native is not None:
                stats = json.loads(native.algan_graph_stats())
                (output / "native_graphs.json").write_text(json.dumps(stats, indent=2))
                emit("native_graph_stats", rows=stats)
    if args.bloom_ab:
        # This is outside all timed renders. Both paths consume the same
        # allocated input, including a nonzero arena offset. No image encoding
        # can hide a numerical discrepancy in this float/output-byte check.
        for channels in (4, 5):
            torch.manual_seed(1234 + channels)
            frames = torch.randint(1, 220, (1, 127, 193, channels), dtype=torch.uint8)
            frames[..., 3] = 100
            answers = []
            for use_candidate in (False, True):
                candidate = use_candidate
                memory = mu.ManualMemory(0, device=args.child, num_bytes=64 * 2**20)
                memory.get_tensor((137,), torch.uint8).fill_(91)
                input_tensor = memory.get_tensor(frames.shape, frames.dtype)
                input_tensor.copy_(frames)
                before = dict(zc.STATS)
                result = bloom_module.bloom_filter(input_tensor, memory=memory).cpu()
                answers.append(result)
                emit("bloom_parity_arm", channels=channels, candidate=candidate,
                     launches=zc.STATS["converted_launches"]-before["converted_launches"],
                     staged=zc.STATS["staged_arguments"]-before["staged_arguments"])
            indices = [0, 1, 2, 4] if channels == 5 else [0, 1, 2]
            first, second = [x[..., indices] for x in answers]
            difference = (first - second).abs()
            quantized = [(x * 255).clamp(0, 255).to(torch.int16) for x in (first, second)]
            byte_difference = (quantized[0] - quantized[1]).abs()
            emit("bloom_parity", channels=channels, max_float=difference.max().item(),
                 mean_float=difference.mean().item(), max_byte=byte_difference.max().item(),
                 changed_bytes=(byte_difference != 0).sum().item(), values=byte_difference.numel())
            if byte_difference.max().item() > 2 or not torch.isfinite(second).all():
                raise AssertionError("Bloom candidate exceeded the output tolerance")
    if sample_process is not None:
        sample_process.wait(timeout=35)
    return 0


def main():
    args = arguments()
    if args.child:
        return child(args)
    output = Path("algan_outputs/early_profile")
    output.mkdir(parents=True, exist_ok=True)
    for device in args.devices:
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--child", device,
                   "--quality", args.quality, "--arena-mib", str(args.arena_mib),
                   "--runs", str(args.runs)]
        if args.native_sample:
            command.append("--native-sample")
        if args.native_graphs:
            command.append("--native-graphs")
        if args.bloom_ab:
            command.append("--bloom-ab")
        with (output / f"{device}.txt").open("w") as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            try:
                code = process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=15)
                code = 124
        print((output / f"{device}.txt").read_text(), flush=True)
        print(f"CHILD {device} exit={code}", flush=True)
        if code:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
