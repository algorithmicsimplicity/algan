"""RENDERER_WORK_QUEUE.md item 9's first number: mode-1 vs mode-2 resolve time.

A shadowed batch launches ``sheet_resolve_shade`` twice per covered-pixel
slice -- mode 1 walks the transport and builds the shadow events, mode 2
shades reading the traced visibility -- and mode 1 recomputes every fetch
mode 2 makes. Whether memoizing those (~15 floats per sheet) can pay depends
entirely on the RATIO of mode 1's device time to mode 2's, which had never
been measured ("this container is CPU-only and cannot rank it").

This wraps the kernel's host callable, brackets every launch with a device
sync, and attributes the wall time to the launch's ``mode`` argument. The
sync makes each launch's cost include the queue it drains, so read the two
modes' TOTALS against each other, not against an unsynced profile.

    uv run python benchmarks/_resolve_mode_ratio.py [quality]
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("ALGAN_USE_DAEMON", "0")
os.environ["ALGAN_PREFETCH_BATCHES"] = "0"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from algan import *  # noqa: E402
from algan.rendering.raytracing import sheet_resolve_taichi as srt  # noqa: E402

#: Position of ``mode`` in sheet_resolve_shade's argument list (the first
#: value after the 38 shared pre_args -- see raster_pipeline's launch sites).
_MODE_ARG = 38


def _sync():
    import taichi as ti

    ti.sync()


def main():
    quality_name = sys.argv[1] if len(sys.argv) > 1 else "PREVIEW"
    quality = globals()[quality_name]

    totals = {}
    counts = {}
    orig = srt.sheet_resolve_shade

    def timed(*args, **kwargs):
        mode = args[_MODE_ARG]
        assert mode in (0, 1, 2), (
            f"arg {_MODE_ARG} is {mode!r}, not a resolve mode -- the launch "
            "signature moved; recount pre_args in raster_pipeline"
        )
        _sync()
        t0 = time.perf_counter()
        out = orig(*args, **kwargs)
        _sync()
        dt = time.perf_counter() - t0
        totals[mode] = totals.get(mode, 0.0) + dt
        counts[mode] = counts.get(mode, 0) + 1
        return out

    srt.sheet_resolve_shade = timed

    SETTINGS.raytracing.set(shadows=True)
    with Off():
        for off in (LEFT * 3, ORIGIN, RIGHT * 3):
            Sphere().scale(0.8).move(off + UP * 1.2).spawn()
        ground = Cube(color=WHITE).scale(4)
        ground.move(DOWN * 5.2)
        ground.spawn()
        mover = Cube().scale(0.5).move(DOWN * 0.8).spawn()
    with Sync(run_time=2):
        mover.move(RIGHT * 2)

    Scene.save_video(
        "algan_outputs/_resolve_mode_ratio",
        quality,
        overwrite=True,
        ffmpeg_params=["-crf", "17", "-preset", "ultrafast"],
    )
    srt.sheet_resolve_shade = orig

    for mode in sorted(totals):
        print(f"mode {mode}: {counts[mode]:4d} launches, {totals[mode] * 1e3:9.1f} ms")
    if 1 in totals and 2 in totals:
        print(
            f"mode1 / mode2 device-time ratio: {totals[1] / totals[2]:.3f} "
            f"(item 9's memoization is capped at removing most of mode 1's "
            f"re-fetch share of that numerator)"
        )
    else:
        print("VACUOUS: the render never took the shadowed two-launch path")
        sys.exit(1)


if __name__ == "__main__":
    main()
