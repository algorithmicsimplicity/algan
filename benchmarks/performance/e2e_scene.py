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

``--lossless`` encodes x264 at CRF 0 so ``--compare-to`` (another arm's video)
can decode both and report the largest channel difference; the repository's
render tolerance is 2. ``--cprofile N`` profiles run ``N`` and prints its top
functions by own and cumulative time
(authoring included; comma-separate several runs).
"""

from __future__ import annotations

import time

PROCESS_START = time.perf_counter()

import argparse  # noqa: E402
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
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
    args = parser.parse_args(argv)
    profiled_runs = {int(r) for r in args.cprofile.split(",") if r.strip()}

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
    }
    print("E2E_HEADER " + json.dumps(header), flush=True)

    kwargs = {"overwrite": True, "animate_fade_out": False}
    if args.lossless:
        kwargs["ffmpeg_params"] = ["-crf", "0", "-preset", "ultrafast"]
    runs = []
    output = None
    for index in range(args.runs):
        SceneManager.reset()
        scene = Scene()
        scene.set_video_settings(preset)
        profiler = cProfile.Profile() if index in profiled_runs else None
        started = time.perf_counter()
        if profiler is not None:
            profiler.enable()
        record(seconds)
        authored = time.perf_counter()
        duration = float(scene._recorded_end_time_for_render())
        output = out_dir / f"{args.label}_{args.scene}_{args.quality}.mp4"
        result = scene.save_video(str(output), preset, **kwargs)
        _sync()
        if profiler is not None:
            profiler.disable()
        finished = time.perf_counter()
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
