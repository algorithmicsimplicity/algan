"""How often a render with a live CameraView restarts its per-pass machinery.

A render with a live :class:`~algan.CameraView` runs in bounded windows of
frames (``algan/rendering/camera_views.py``, ``_window_size``), each window one
capture pass per view plus one main pass. This counts, over one render, how
many render arenas (``ManualMemory``), chunk-peak models
(``ChunkMemoryModel``), peak-ratio models and batch-prep workers
(``algan-batch-prep`` threads) the render loop creates, and times the render
warm. It also saves the rendered frames, so two checkouts can be compared byte
for byte (``--save`` on one, ``--compare`` on the other).

Usage::

    ALGAN_AUTO_DAEMON=0 python benchmarks/_camera_view_window_restarts.py
        [--frames 48] [--capture 640x360] [--runs 3] [--save out.pt]
        [--compare out.pt] [--json out.json]

The first render of the process is reported separately as cold: it pays the
kernel specialisation every later one reuses.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import statistics
import sys
import threading
import time

# A warm daemon would carry state between runs, and this is a measurement.
os.environ.setdefault("ALGAN_AUTO_DAEMON", "0")
os.environ.setdefault("ALGAN_PROGRESS", "none")

import torch  # noqa: E402

import algan  # noqa: E402
from algan import (  # noqa: E402
    CameraView,
    Off,
    Scene,
    Sphere,
    Square,
    render_loop,  # noqa: E402
)
from algan.rendering import camera_views  # noqa: E402
from algan.scene import _note_render_requested  # noqa: E402


class _Counts:
    """Constructions of each per-pass resource, counted by wrapping its class."""

    def __init__(self):
        self.arenas = 0
        self.chunk_models = 0
        self.peak_ratio_models = 0
        self.prep_workers = 0

    def as_dict(self):
        return {
            "arenas": self.arenas,
            "chunk_models": self.chunk_models,
            "peak_ratio_models": self.peak_ratio_models,
            "prep_workers": self.prep_workers,
        }


@contextlib.contextmanager
def counting():
    counts = _Counts()
    originals = (
        render_loop.ManualMemory,
        render_loop.ChunkMemoryModel,
        render_loop.PeakRatioModel,
        render_loop.ThreadPoolExecutor,
    )
    manual_memory, chunk_model, peak_ratio, executor = originals

    def counted_memory(*args, **kwargs):
        if kwargs.get("managed", True) and (len(args) < 3 or args[2]):
            counts.arenas += 1
        return manual_memory(*args, **kwargs)

    def counted_chunk_model(*args, **kwargs):
        counts.chunk_models += 1
        return chunk_model(*args, **kwargs)

    def counted_peak_ratio(*args, **kwargs):
        counts.peak_ratio_models += 1
        return peak_ratio(*args, **kwargs)

    def counted_executor(*args, **kwargs):
        pool = executor(*args, **kwargs)
        if kwargs.get("thread_name_prefix") == "algan-batch-prep":
            counts.prep_workers += 1
        return pool

    render_loop.ManualMemory = counted_memory
    render_loop.ChunkMemoryModel = counted_chunk_model
    render_loop.PeakRatioModel = counted_peak_ratio
    render_loop.ThreadPoolExecutor = counted_executor
    # A camera-view job's worker is made in camera_views (absent before it was).
    job_executor = getattr(camera_views, "ThreadPoolExecutor", None)
    if job_executor is not None:
        camera_views.ThreadPoolExecutor = counted_executor
    try:
        yield counts
    finally:
        (
            render_loop.ManualMemory,
            render_loop.ChunkMemoryModel,
            render_loop.PeakRatioModel,
            render_loop.ThreadPoolExecutor,
        ) = originals
        if job_executor is not None:
            camera_views.ThreadPoolExecutor = job_executor


def build_scene(capture):
    scene = Scene(
        algan.PREVIEW.set(resolution=(320, 180), frames_per_second=24),
        background=algan.BLACK,
    )
    with Off():
        target = Sphere(color=algan.RED).spawn()
        Square(size=1.5, color=algan.BLUE).move(algan.LEFT * 2).spawn()
        view = CameraView(resolution=capture, height=2).focus_on(target, 0.4)
        view.move_to(algan.RIGHT * 3 + algan.OUT).spawn()
    target.move(algan.UP)
    target.color = algan.GREEN
    return scene


def render(scene, frames):
    # Frames straight from the generator, which the exit notice does not count.
    _note_render_requested()
    with contextlib.closing(scene.get_frames(0, frames, post_processes=())) as stream:
        return torch.cat([batch.cpu() for batch in stream])


def run_once(capture, frames):
    scene = build_scene(capture)
    with counting() as counts:
        start = time.perf_counter()
        out = render(scene, frames)
        wall = time.perf_counter() - start
    windows = -(
        -frames
        // camera_views._window_size(
            [v for v in scene.actors if getattr(v, "_is_camera_view", False)]
        )
    )
    return out, wall, counts.as_dict(), windows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--capture", default="640x360")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--save")
    parser.add_argument("--compare")
    parser.add_argument("--json")
    args = parser.parse_args(argv)
    capture = tuple(int(n) for n in args.capture.lower().split("x"))

    cold_frames, cold_wall, cold_counts, windows = run_once(capture, args.frames)
    walls, counts = [], cold_counts
    for _ in range(args.runs):
        frames, wall, counts, windows = run_once(capture, args.frames)
        walls.append(wall)
        if not torch.equal(frames, cold_frames):
            print("warning: a warm render differs from the cold one", file=sys.stderr)
    # Any prep worker still alive after the render would be a leak.
    leaked = [
        t.name for t in threading.enumerate() if t.name.startswith("algan-batch-prep")
    ]
    report = {
        "frames": args.frames,
        "capture": list(capture),
        "windows": windows,
        "cold_seconds": round(cold_wall, 3),
        "warm_seconds": [round(w, 3) for w in walls],
        "warm_median_seconds": round(statistics.median(walls), 3) if walls else None,
        "restarts_per_render": counts,
        "leaked_prep_threads": leaked,
        "load_average": list(os.getloadavg()),
    }
    print(json.dumps(report, indent=2))
    if args.save:
        torch.save(cold_frames, args.save)
    if args.compare:
        reference = torch.load(args.compare)
        same = reference.shape == cold_frames.shape and torch.equal(
            reference, cold_frames
        )
        diff = (
            (reference.int() - cold_frames.int()).abs().max().item()
            if reference.shape == cold_frames.shape
            else None
        )
        print(
            f"frames identical to {args.compare}: {same} (max channel difference {diff})"
        )
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(report, handle, indent=2)


if __name__ == "__main__":
    main()
