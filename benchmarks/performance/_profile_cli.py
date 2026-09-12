"""The command line shared by the workload benchmark scenes.

One scene file serves every measurement of that scene: the quality preset,
the clip length and the number of profiled runs are arguments, so the same
authoring is measured at PREVIEW and UHD, and can be cut down to a few frames
on a box whose UHD frame costs thirty seconds (the Mac runner) without a
second copy of the scene drifting away from the first.

    python benchmarks/performance/explainer_scene.py --quality UHD --seconds 0.5
    python benchmarks/performance/graphics_scene.py --quality PREVIEW --seconds 6

The clip length is the *authored* duration, so frame count = seconds x the
preset's frame rate (PREVIEW 10 fps, UHD 60 fps). ``--frames`` sets it the
other way round.

Everything else is ``profile_scene``'s: two runs (read RUN 2), the Taichi
kernel profiler off by default (it re-inits the runtime; ``--kernel-profiler``
turns it on) and libx264 at ``ultrafast`` so the encoder is representative of
a machine with a full CPU rather than the 2-4 slow cores these boxes have.
"""

from __future__ import annotations

import argparse
import os
import sys

# Every benchmark in this directory renders in-process: the warm daemon would
# both hide the cold cost and refuse the environment these scripts set.
os.environ.setdefault("ALGAN_USE_DAEMON", "0")
# The T4 picks NVENC on its own, which is the right default for throughput and
# the wrong one for a cross-box comparison; pin the encoder unless the caller
# already chose one.
os.environ.setdefault("ALGAN_VIDEO_ENCODER", "software")

import algan  # noqa: E402
from algan.utils.profiling_utils import profile_scene  # noqa: E402

PRESETS = ("PREVIEW", "LD", "MD", "HD", "UHD", "SMOKE_TEST")


def parse_args(argv=None, *, default_seconds):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality", default="PREVIEW", choices=PRESETS)
    length = parser.add_mutually_exclusive_group()
    length.add_argument(
        "--seconds",
        type=float,
        default=None,
        help=f"authored clip length (default {default_seconds}s)",
    )
    length.add_argument(
        "--frames",
        type=int,
        default=None,
        help="clip length as a frame count at the preset's frame rate",
    )
    parser.add_argument("--runs", type=int, default=2, help="profiled renders")
    parser.add_argument(
        "--kernel-profiler",
        action="store_true",
        help="enable Taichi's per-kernel GPU profiler (re-inits the runtime)",
    )
    parser.add_argument("--tag", default=None, help="suffix for the report files")
    args = parser.parse_args(argv)
    settings = getattr(algan, args.quality)
    if args.frames is not None:
        args.seconds = args.frames / settings.frames_per_second
    elif args.seconds is None:
        args.seconds = default_seconds
    args.settings = settings
    args.num_frames = round(args.seconds * settings.frames_per_second)
    return args


def run(scene_func, name, *, default_seconds, argv=None):
    """Profile ``scene_func(seconds)`` per the command line.

    ``scene_func`` receives the authored clip length and must record a scene
    of exactly that duration on the active Scene.
    """
    args = parse_args(argv, default_seconds=default_seconds)
    tag = args.tag or f"{name}_{args.quality}"
    print(
        f"[bench] {name}: quality {args.quality} "
        f"({args.settings.resolution[0]}x{args.settings.resolution[1]} @ "
        f"{args.settings.frames_per_second} fps), {args.seconds:g}s = "
        f"{args.num_frames} frames, {args.runs} run(s), tag {tag}",
        flush=True,
    )
    return profile_scene(
        lambda: scene_func(args.seconds),
        args.settings,
        tag,
        runs=args.runs,
        kernel_profiler=args.kernel_profiler,
        # The encoder is part of the measurement (the "video encode tail"
        # stage); the fast preset keeps it representative of a machine with
        # a full CPU. The output's quality is not what is measured here.
        save_video_kwargs={"ffmpeg_params": ["-crf", "17", "-preset", "ultrafast"]},
    )


if __name__ == "__main__":
    sys.exit("import this module from a scene file; it has no scene of its own")
