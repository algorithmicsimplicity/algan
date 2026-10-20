"""End-to-end video production time of a workload scene, as a user meets it.

``profile_scene`` answers "where does a render's time go"; this answers "how
long from ``python scene.py`` to an mp4 on disk", which is what a user waits
for. It records, in one process and with no instrumentation installed:

* ``import`` -- interpreter start to ``import algan`` returning;
* per run, ``author`` (recording the timeline) and ``render``
  (``Scene.save_video`` to a closed file, encoder drain included);
* ``process`` -- interpreter start to the end of the last run.

The first run of a process is the one a fresh ``python scene.py`` pays (its
kernels come from the offline cache, or are compiled when ``ALGAN_CACHE_DIR``
is empty); later runs are what the warm render daemon pays for the same
script. One ``E2E {json}`` line is printed after every run, so a session that
dies half way keeps what it measured (``agent_guidance/gpu_harnesses.md``).

    python benchmarks/performance/e2e_scene.py --scene graphics --quality HD \
        --seconds 2 --runs 3 --label base

``--lossless`` encodes RGB x264 (``libx264rgb``) at CRF 0, so ``--compare-to``
(another arm's video) decodes the rendered pixels themselves and reports the
largest channel difference; the repository's render tolerance is 2. (Plain
``libx264`` at CRF 0 is lossless only after its 4:2:0 conversion, which can
turn a 2 in one pixel into a 3 in its neighbours.) ``--cprofile N`` profiles run ``N`` and prints its top
functions by own and cumulative time
(authoring included; comma-separate several runs).

Diagnostics, all off by default: ``--compile-stats`` prints what Dynamo and
Inductor compiled or loaded during each run, and the Taichi source-key
counters; ``--torch-profile N`` records run ``N`` with ``torch.profiler`` and
prints how much of it the GPU was busy, its largest idle gaps and its top
kernels, and with ``--trace-stages`` also names the pipeline stage the host was
in during each idle gap (``profiling_utils``' stage hooks, re-pointed at
``torch.profiler.record_function`` so they add no device syncs);
``--first-launches N`` times the host side of every kernel launch in run ``N``
and prints the slowest first launches, where a kernel's compiled code is loaded
from the offline cache (or compiled) and handed to the driver;
``--torch-compile off`` renders with ``SETTINGS.computing.torch_compile`` off;
``--rt NAME=VALUE`` flips a raytracing knob through ``set_NAME`` (or the module
attribute) before the first run. Both settings are part of the Taichi kernel
cache key, so such an arm's first run compiles or loads its own kernels:
compare its later runs.
"""

from __future__ import annotations

import time

PROCESS_START = time.perf_counter()

import argparse  # noqa: E402
import contextlib  # noqa: E402
import cProfile  # noqa: E402
import importlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import pstats  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
# Only the scene modules' directory: ``algan`` itself must resolve the way a
# user's import does, so ``PYTHONPATH`` can point a run at another checkout.
sys.path.insert(0, str(HERE))
# In-process on purpose: the daemon would hide the first run's cost.
os.environ.setdefault("ALGAN_USE_DAEMON", "0")

import algan  # noqa: E402

IMPORT_SECONDS = time.perf_counter() - PROCESS_START

from algan import SETTINGS, Scene  # noqa: E402, F401
from algan.scene_manager import SceneManager  # noqa: E402


def _sync():
    import torch

    from algan.rendering.taichi_runtime import _live_arch
    from algan.settings._startup import render_device

    if _live_arch() is not None:
        from algan.taichi_compat import ti

        ti.sync()
    if render_device().type == "cuda":
        torch.cuda.synchronize()


def _checkout():
    root = Path(algan.__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "--short=12", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return str(root), commit


def _compare(reference, candidate):
    import imageio.v2 as imageio
    import numpy as np

    a = imageio.get_reader(str(reference), format="ffmpeg")
    b = imageio.get_reader(str(candidate), format="ffmpeg")
    worst = changed = frames = 0
    try:
        for left, right in zip(a, b, strict=False):
            delta = np.abs(left.astype(np.int16) - right.astype(np.int16))
            worst = max(worst, int(delta.max()))
            changed += int(np.count_nonzero(delta))
            frames += 1
        a_count = a.count_frames()
        b_count = b.count_frames()
    finally:
        a.close()
        b.close()
    return {
        "frames": frames,
        "frames_reference": a_count,
        "frames_candidate": b_count,
        "max_channel_difference": worst,
        "differing_channels": changed,
    }


def _parse_value(text):
    lowered = text.strip().lower()
    if lowered in ("true", "on", "yes"):
        return True
    if lowered in ("false", "off", "no"):
        return False
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def _apply_raytracing_knobs(assignments):
    from algan.rendering.raytracing import settings as rt

    applied = {}
    for assignment in assignments:
        name, _, raw = assignment.partition("=")
        name = name.strip()
        value = _parse_value(raw)
        setter = getattr(rt, f"set_{name}", None)
        if callable(setter):
            setter(value)
        elif hasattr(rt, name):
            setattr(rt, name, value)
        else:
            raise SystemExit(f"--rt {name}: no set_{name} or attribute in settings")
        applied[name] = getattr(rt, name, value)
    return applied


def _compile_stats():
    """What Dynamo/Inductor and the Taichi source key did since the last call."""
    out = {}
    try:
        from torch._dynamo.utils import compile_times, counters

        out["counters"] = {
            group: {k: v for k, v in values.items() if v}
            for group, values in counters.items()
            if group in ("stats", "inductor", "aot_autograd", "frames", "recompiles")
        }
        out["compile_times"] = compile_times(repr="str", aggregate=True)
        counters.clear()
    except Exception as exc:  # noqa: BLE001 -- a diagnostic
        out["dynamo"] = f"unavailable: {exc!r}"
    try:
        from algan.utils import taichi_source_key

        out["taichi_source_key"] = dict(taichi_source_key.STATS)
        taichi_source_key.reset_stats()
    except Exception as exc:  # noqa: BLE001
        out["taichi_source_key"] = f"unavailable: {exc!r}"
    try:
        from algan.utils.torch_compile import compiled_functions

        out["compiled_functions"] = dict(compiled_functions())
    except Exception:  # noqa: BLE001
        pass
    return out


def _summarize_trace(path, wall_seconds, top=30):
    """GPU busy time, idle gaps and top kernels from a chrome trace."""
    with open(path, encoding="utf-8") as handle:
        events = json.load(handle).get("traceEvents", [])
    device = [
        e
        for e in events
        if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
    ]
    if not device:
        return {"device_events": 0}
    intervals = sorted(
        (float(e["ts"]), float(e["ts"]) + float(e["dur"])) for e in device
    )
    busy = 0.0
    gaps = []
    start, end = intervals[0]
    for lo, hi in intervals[1:]:
        if lo > end:
            busy += end - start
            gaps.append((lo - end, end))
            start, end = lo, hi
        else:
            end = max(end, hi)
    busy += end - start
    span = intervals[-1][1] - intervals[0][0]
    by_name = {}
    for e in device:
        name = e.get("name", "?")[:90]
        total, count = by_name.get(name, (0.0, 0))
        by_name[name] = (total + float(e["dur"]), count + 1)
    ranked = sorted(by_name.items(), key=lambda kv: -kv[1][0])[:top]
    gaps.sort(reverse=True)
    buckets = {
        "<0.1ms": 0.0,
        "0.1-1ms": 0.0,
        "1-10ms": 0.0,
        "10-100ms": 0.0,
        ">=100ms": 0.0,
    }
    for gap, _ in gaps:
        if gap < 100:
            buckets["<0.1ms"] += gap
        elif gap < 1000:
            buckets["0.1-1ms"] += gap
        elif gap < 10000:
            buckets["1-10ms"] += gap
        elif gap < 100000:
            buckets["10-100ms"] += gap
        else:
            buckets[">=100ms"] += gap
    # The innermost pipeline stage the host was in at each gap's midpoint
    # says what the GPU waited on (``--trace-stages``; without it there are no
    # stage annotations and every gap reads "(no stage)").
    stage_of_gap = _innermost_stage(events, [at + gap / 2 for gap, at in gaps])
    idle_by_stage = {}
    for (gap, _), name in zip(gaps, stage_of_gap):
        if gap >= 100:
            idle_by_stage[name] = idle_by_stage.get(name, 0.0) + gap
    largest = [
        {
            "ms": round(gap / 1000, 2),
            "at_s": round((at - intervals[0][0]) / 1e6, 3),
            "stage": name,
        }
        for (gap, at), name in list(zip(gaps, stage_of_gap))[:20]
    ]
    return {
        "wall_s": round(wall_seconds, 3),
        "device_span_s": round(span / 1e6, 3),
        "device_busy_s": round(busy / 1e6, 3),
        "device_idle_s": round((span - busy) / 1e6, 3),
        "idle_by_gap_size_s": {k: round(v / 1e6, 3) for k, v in buckets.items()},
        "idle_by_stage_s (gaps >= 0.1 ms)": {
            name: round(total / 1e6, 3)
            for name, total in sorted(idle_by_stage.items(), key=lambda kv: -kv[1])[:25]
        },
        "largest_gaps": largest,
        "top_device_work": [
            {"name": n, "s": round(t / 1e6, 3), "count": c} for n, (t, c) in ranked
        ],
    }


def _innermost_stage(events, times):
    """The innermost stage annotation open at each of ``times`` (host clock).

    Stage annotations nest on the thread that opened them, so one sweep over
    their starts and ends with a stack answers every query. Only the thread
    with the most annotated time is read: the render loop's, rather than a
    prefetch worker's.
    """
    stages = [
        e
        for e in events
        if e.get("ph") == "X"
        and e.get("cat") == "user_annotation"
        and not str(e.get("name", "")).startswith("ProfilerStep")
    ]
    if not stages:
        return ["(no stage)"] * len(times)
    per_thread = {}
    for e in stages:
        per_thread[e.get("tid")] = per_thread.get(e.get("tid"), 0.0) + float(e["dur"])
    main = max(per_thread, key=per_thread.get)
    sweep = []
    for index, e in enumerate(e for e in stages if e.get("tid") == main):
        start = float(e["ts"])
        sweep.append((start, 1, index, e.get("name", "?")))
        sweep.append((start + float(e["dur"]), 0, index, None))
    for index, at in enumerate(times):
        sweep.append((at, 2, index, None))
    sweep.sort(key=lambda item: (item[0], item[1]))
    stack = []
    out = ["(no stage)"] * len(times)
    for _, kind, index, name in sweep:
        if kind == 1:
            stack.append((index, name))
        elif kind == 0:
            for position in range(len(stack) - 1, -1, -1):
                if stack[position][0] == index:
                    del stack[position]
                    break
        elif stack:
            out[index] = stack[-1][1]
    return out


@contextlib.contextmanager
def _stage_annotations():
    """Every pipeline stage as a ``torch.profiler`` range, with no syncs.

    ``install_pipeline_hooks`` wraps the pipeline's entry points in
    ``TIMERS.stage``, which syncs the devices on both boundaries to time them
    -- the opposite of what an idle-gap trace wants. The wrappers look the
    method up at call time, so shadowing it on the instance turns each of
    them into a bare ``record_function`` range; the kernel hooks, which sync
    on their own, are not installed.
    """
    import torch

    from algan.utils import profiling_utils

    timers = profiling_utils.TIMERS
    with profiling_utils._temporary_instrumentation():
        profiling_utils.install_pipeline_hooks()
        timers.stage = lambda name, items=None: torch.profiler.record_function(name)
        try:
            yield
        finally:
            del timers.stage


class _LaunchTimer:
    """Host seconds of each kernel specialization's first launch.

    The first launch is where a specialization's compiled code is fetched --
    from the offline cache, or compiled -- and loaded into the driver. Later
    launches take ``taichi_fast_launch``'s cached plan and never reach
    ``Kernel.launch_kernel``, so only first launches are counted. Nothing
    here syncs the device.
    """

    def __init__(self):
        self.first = {}

    def __enter__(self):
        from algan.taichi_compat import submodule

        kernel_type = submodule("lang.kernel_impl").Kernel
        original = kernel_type.launch_kernel
        first = self.first

        def timed(kernel, *args, **kwargs):
            started = time.perf_counter()
            try:
                return original(kernel, *args, **kwargs)
            finally:
                key = args[0] if args and isinstance(args[0], tuple) else None
                name = f"{kernel.func.__qualname__}[{key[1] if key else '?'}]"
                if name not in first:
                    first[name] = time.perf_counter() - started

        self._restore = (kernel_type, original)
        kernel_type.launch_kernel = timed
        return self

    def __exit__(self, *exc):
        kernel_type, original = self._restore
        kernel_type.launch_kernel = original
        return False

    def summary(self, top=20):
        slowest = sorted(self.first.items(), key=lambda item: -item[1])[:top]
        return {
            "specializations": len(self.first),
            "first_launch_s": round(sum(self.first.values()), 3),
            "slowest_first": [[name, round(t, 4)] for name, t in slowest],
        }


def _print_profile(profiler, limit):
    for key in ("tottime", "cumulative"):
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.sort_stats(key).print_stats(limit)
        print(f"----- cProfile by {key} -----")
        print(stream.getvalue())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="graphics")
    parser.add_argument("--quality", default="HD")
    length = parser.add_mutually_exclusive_group()
    length.add_argument("--seconds", type=float, default=None)
    length.add_argument("--frames", type=int, default=None)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--label", default="run")
    parser.add_argument("--lossless", action="store_true")
    parser.add_argument("--compare-to", default=None)
    parser.add_argument(
        "--cprofile",
        default="",
        metavar="RUNS",
        help="comma-separated run indices to profile (authoring and render)",
    )
    parser.add_argument("--cprofile-limit", type=int, default=70)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--compile-stats", action="store_true")
    parser.add_argument("--torch-profile", default="", metavar="RUNS")
    parser.add_argument("--trace-stages", action="store_true")
    parser.add_argument("--first-launches", default="", metavar="RUNS")
    parser.add_argument("--torch-compile", choices=("auto", "on", "off"), default=None)
    parser.add_argument("--rt", action="append", default=[], metavar="NAME=VALUE")
    args = parser.parse_args(argv)
    profiled_runs = {int(r) for r in args.cprofile.split(",") if r.strip()}
    traced_runs = {int(r) for r in args.torch_profile.split(",") if r.strip()}
    launch_runs = {int(r) for r in args.first_launches.split(",") if r.strip()}
    if args.torch_compile is not None:
        SETTINGS.computing.torch_compile = {"auto": "auto", "on": True, "off": False}[
            args.torch_compile
        ]
    knobs = _apply_raytracing_knobs(args.rt)

    preset = getattr(algan, args.quality)
    if args.frames is not None:
        seconds = args.frames / preset.frames_per_second
    else:
        seconds = 2.0 if args.seconds is None else args.seconds
    record = importlib.import_module(f"{args.scene}_scene").scene
    out_dir = Path(args.out_dir or (HERE / "algan_outputs" / "e2e")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch

    from algan.settings._startup import render_device

    root, commit = _checkout()
    device = render_device()
    header = {
        "label": args.label,
        "scene": args.scene,
        "quality": args.quality,
        "resolution": list(preset.resolution),
        "fps": preset.frames_per_second,
        "storyboard_seconds": seconds,
        "checkout": root,
        "commit": commit,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch": torch.__version__,
        "cache_dir": os.environ.get("ALGAN_CACHE_DIR"),
        "encoder": os.environ.get("ALGAN_VIDEO_ENCODER", "(default)"),
        "lossless": args.lossless,
        "import_seconds": round(IMPORT_SECONDS, 3),
        "torch_compile": str(SETTINGS.computing.torch_compile),
        "raytracing": knobs,
    }
    print("E2E_HEADER " + json.dumps(header), flush=True)

    kwargs = {"overwrite": True, "animate_fade_out": False}
    if args.lossless:
        kwargs["codec"] = "libx264rgb"
        kwargs["ffmpeg_params"] = ["-crf", "0", "-preset", "ultrafast"]
    runs = []
    output = None
    for index in range(args.runs):
        SceneManager.reset()
        scene = Scene()
        scene.set_video_settings(preset)
        profiler = cProfile.Profile() if index in profiled_runs else None
        tracer = None
        started = time.perf_counter()
        if profiler is not None:
            profiler.enable()
        record(seconds)
        authored = time.perf_counter()
        duration = float(scene._recorded_end_time_for_render())
        output = out_dir / f"{args.label}_{args.scene}_{args.quality}.mp4"
        if index in traced_runs:
            # The render only: authoring is host work the trace would bloat.
            from torch.profiler import ProfilerActivity, profile

            tracer = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
            tracer.__enter__()
        annotate = (
            _stage_annotations()
            if tracer is not None and args.trace_stages
            else contextlib.nullcontext()
        )
        launches = _LaunchTimer() if index in launch_runs else None
        with annotate, launches or contextlib.nullcontext():
            result = scene.save_video(str(output), preset, **kwargs)
            _sync()
        if launches is not None:
            print(f"E2E_LAUNCHES run {index} " + json.dumps(launches.summary()))
        if profiler is not None:
            profiler.disable()
        finished = time.perf_counter()
        if tracer is not None:
            tracer.__exit__(None, None, None)
            trace_path = out_dir / f"{args.label}_run{index}_trace.json"
            tracer.export_chrome_trace(str(trace_path))
            summary = _summarize_trace(trace_path, finished - authored)
            print("E2E_TRACE " + json.dumps(summary), flush=True)
        if result.status != "rendered":
            raise RuntimeError(f"expected a rendered video, got {result.status}")
        frames = round(duration * preset.frames_per_second)
        item = {
            "label": args.label,
            "run": index,
            "author": round(authored - started, 3),
            "render": round(finished - authored, 3),
            "total": round(finished - started, 3),
            "since_process_start": round(finished - PROCESS_START, 3),
            "frames": frames,
            "peak_gb": (
                round(torch.cuda.max_memory_allocated() / 2**30, 2)
                if device.type == "cuda"
                else None
            ),
        }
        runs.append(item)
        print("E2E " + json.dumps(item), flush=True)
        if profiler is not None:
            _print_profile(profiler, args.cprofile_limit)
        if args.compile_stats:
            print(
                f"E2E_COMPILE run {index} " + json.dumps(_compile_stats()), flush=True
            )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

    summary = {
        "label": args.label,
        "import": round(IMPORT_SECONDS, 3),
        "first_run": runs[0]["total"],
        "process_to_first_video": round(IMPORT_SECONDS + runs[0]["total"], 3),
        "warm_runs": [r["total"] for r in runs[1:]],
        "warm_render_min": min((r["render"] for r in runs[1:]), default=None),
        "process": round(time.perf_counter() - PROCESS_START, 3),
        "output": str(output),
    }
    if args.compare_to:
        summary["parity"] = _compare(args.compare_to, output)
    print("E2E_SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
