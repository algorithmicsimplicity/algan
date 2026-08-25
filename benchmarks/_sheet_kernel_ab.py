"""A/B the compaction/emission host-pass kernels in a real render:
RASTER_FUSED_GATHER + SHEET_MASK_KERNEL + SHEET_RANK_KERNEL
+ RASTER_OPAQUE_TRUNC_KERNEL + SHEET_ONE_MESH_KERNEL
+ SHEET_SAMPLE_DEPTH_KERNEL.

Alternates the arms inside one process (wall-clock across processes swings ~2x
with thermal throttling on this hardware) and reports the median of each arm,
plus the two stages the kernels sit in. The thermal-throttling caveat -- and
so the trustworthiness of the wall-clock numbers themselves -- is a
CUDA-machine caveat; on a CPU-only box the script still runs (the sync helper
is a no-op without CUDA) but its times are CPU numbers and nothing else.

    <venv-python> benchmarks/_sheet_kernel_ab.py [width] [height] [rounds]
"""

import collections
import os
import statistics
import sys
import time

os.environ.setdefault("ALGAN_USE_DAEMON", "0")

import torch  # noqa: E402

from algan import *  # noqa: E402,F403
from algan.rendering.raytracing import raster_pipeline, sheets  # noqa: E402
from algan.settings import SETTINGS  # noqa: E402

W = int(sys.argv[1]) if len(sys.argv) > 2 else 1920
H = int(sys.argv[2]) if len(sys.argv) > 2 else 1080
ROUNDS = int(sys.argv[3]) if len(sys.argv) > 3 else 5
EXPERIMENTAL = SETTINGS.raytracing.experimental

acc = collections.Counter()


def _sync():
    """Synchronize the device before/after a timed region; a no-op where
    there is no CUDA to synchronize (torch.cuda.synchronize raises there).
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(mod, name, key):
    orig = getattr(mod, name)

    def wrapper(*a, **k):
        _sync()
        t0 = time.perf_counter()
        try:
            return orig(*a, **k)
        finally:
            _sync()
            acc[key] += time.perf_counter() - t0

    setattr(mod, name, wrapper)


timed(sheets, "compact_sheets", "compact")
timed(raster_pipeline, "_gather_fragment_arrays", "gather")

Sphere().scale(1.6).move(LEFT * 2.6).set_color(GREEN).spawn()
Cube().scale(1.2).move(RIGHT * 2.6).set_color(BLUE).spawn()
_glass = Sphere().scale(1.0).move(UP * 1.1).spawn()
_glass.opacity = 0.45
Text("sheets").scale(0.7).move(DOWN * 2.2).spawn()
VS = UHD.set(resolution=(W, H))

ARMS = {
    "torch ": {
        "raster_fused_gather": False,
        "sheet_mask_kernel": False,
        "sheet_rank_kernel": False,
        "raster_opaque_trunc_kernel": False,
        "sheet_one_mesh_kernel": False,
        "sheet_sample_depth_kernel": False,
    },
    "kernel": {
        "raster_fused_gather": True,
        "sheet_mask_kernel": True,
        "sheet_rank_kernel": True,
        "raster_opaque_trunc_kernel": True,
        "sheet_one_mesh_kernel": True,
        "sheet_sample_depth_kernel": True,
    },
}
samples = {arm: [] for arm in ARMS}
stages = {arm: collections.Counter() for arm in ARMS}

# Both arms warmed (kernel compiles, allocator, adaptive renderer state) before
# any sample is kept.
for cfg in ARMS.values():
    EXPERIMENTAL.set(**cfg)
    Scene.save_frame("_sheet_kernel_ab_warm.png", VS)

for _round in range(ROUNDS):
    for arm, cfg in ARMS.items():
        EXPERIMENTAL.set(**cfg)
        acc.clear()
        _sync()
        t0 = time.perf_counter()
        Scene.save_frame("_sheet_kernel_ab.png", VS)
        _sync()
        samples[arm].append(time.perf_counter() - t0)
        stages[arm].update(acc)

print(f"\n=== {W}x{H}, {ROUNDS} alternating rounds ===")
for arm in ARMS:
    med = statistics.median(samples[arm])
    print(
        f"  {arm}  frame {med:6.3f}s   compact_sheets "
        f"{stages[arm]['compact'] / ROUNDS:6.3f}s   gather "
        f"{stages[arm]['gather'] / ROUNDS:6.3f}s"
    )
t = statistics.median(samples["torch "])
k = statistics.median(samples["kernel"])
print(f"  speedup {t / k:.3f}x on the frame  ({(t - k) * 1000:.0f} ms saved)")
