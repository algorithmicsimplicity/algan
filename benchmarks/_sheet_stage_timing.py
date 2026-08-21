"""Where the sheet route's time goes: emission kernels vs host-torch compaction.

Answers the question a Taichi rewrite of ``sheets.compact_sheets`` /
``raster_pipeline.prepare_sparse_raster_coverage`` turns on. It splits the
coverage stage into

* the Taichi EMISSION kernels (``raster_*_count`` / ``raster_*_write``),
  already in kernel language and untouched by such a rewrite;
* the host-torch SORTS (``argsort``/``unique``/``cumsum``/``cummax``), which
  a rewrite would have to re-implement rather than remove -- they are
  cuB-backed today;
* everything else on the host: the elementwise and segmented passes a fused
  kernel would actually claim.

Per frame, so the first frame's kernel-compile and glyph-cache costs do not
contaminate the steady-state numbers.

    <venv-python> benchmarks/_sheet_stage_timing.py [width] [height] [reps]
"""

import collections
import os
import sys
import time

os.environ.setdefault("ALGAN_USE_DAEMON", "0")

import torch  # noqa: E402

from algan import *  # noqa: E402,F403
from algan.rendering.raytracing import raster_pipeline, sheets  # noqa: E402

W = int(sys.argv[1]) if len(sys.argv) > 2 else 1920
H = int(sys.argv[2]) if len(sys.argv) > 2 else 1080
REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 3

acc = collections.Counter()
calls = collections.Counter()


in_compact = [False]


def timed(mod, name, key):
    orig = getattr(mod, name)

    def wrapper(*a, **k):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            return orig(*a, **k)
        finally:
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            k2 = key + ("_in_compact" if (key == "sorts" and in_compact[0]) else "")
            acc[k2] += dt
            calls[k2] += 1

    setattr(mod, name, wrapper)


def timed_compact():
    orig = sheets.compact_sheets

    def wrapper(*a, **k):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        in_compact[0] = True
        try:
            return orig(*a, **k)
        finally:
            in_compact[0] = False
            torch.cuda.synchronize()
            acc["compact"] += time.perf_counter() - t0

    sheets.compact_sheets = wrapper


timed_compact()
timed(raster_pipeline, "prepare_sparse_raster_coverage", "prepare")
for _fn in (
    "raster_tri_count",
    "raster_tri_write",
    "raster_bez_count",
    "raster_bez_write",
):
    if hasattr(raster_pipeline, _fn):
        timed(raster_pipeline, _fn, "emission")
for _fn in ("argsort", "sort", "unique", "unique_consecutive", "cumsum", "cummax"):
    timed(torch, _fn, "sorts")


def build():
    Sphere().scale(1.6).move(LEFT * 2.6).set_color(GREEN).spawn()
    Cube().scale(1.2).move(RIGHT * 2.6).set_color(BLUE).spawn()
    glass = Sphere().scale(1.0).move(UP * 1.1).spawn()
    glass.opacity = 0.45
    Text("sheets").scale(0.7).move(DOWN * 2.2).spawn()


build()
VS = UHD.set(resolution=(W, H))
print(f"\n=== {W}x{H} ===")
print(
    f"{'frame':>5} {'wall':>9} {'prepare':>9} {'emission':>9} {'sort/oth':>8} "
    f"{'compact':>8} {'|sorts':>8} {'|elemwise':>10}"
)
for i in range(REPS):
    acc.clear()
    calls.clear()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    Scene.save_frame(f"_sheet_stage_timing_{i}.png", VS)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    prep, emit = acc["prepare"], acc["emission"]
    srt_c, comp = acc["sorts_in_compact"], acc["compact"]
    srt_o = acc["sorts"]
    print(
        f"{i:>5} {wall:8.3f}s {prep:8.3f}s {emit:8.3f}s {srt_o:7.3f}s "
        f"{comp:7.3f}s {srt_c:8.3f}s {comp - srt_c:9.3f}s"
    )
print(
    "\n(prepare = coverage stage total; emission = its Taichi count/write "
    "kernels;\n sorts = torch argsort/unique/cumsum/cummax anywhere in the "
    "frame;\n compact = sheets.compact_sheets, a subset of 'host rest')"
)
