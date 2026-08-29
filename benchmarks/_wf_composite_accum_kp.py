"""Isolate wf_composite_accum device time: throughput vs per-launch overhead.

Runs the SAME total pixel count as one big launch vs many small launches
(the production per-tile pattern) on identical buffers, device-synced, warm.
If per-pixel device time is ~flat across launch counts the kernel is
throughput-bound (optimize memory access); if small launches are much
slower per pixel it is per-launch-overhead-bound (optimize launch count).

Run: .venv/Scripts/python.exe benchmarks/_wf_composite_accum_kp.py
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("ALGAN_PREFETCH_BATCHES", "0")

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from algan.rendering.taichi_runtime import _sync_devices, init_taichi  # noqa: E402

init_taichi()

from algan.rendering.raytracing.wavefront_kernels_taichi import (  # noqa: E402
    wf_composite_accum,
)

W, H = 1280, 720
FRAMES = 20  # one memory window
PPF = W * H
TOTAL = FRAMES * PPF  # ~18.4M pixels
DEV = "cuda"
T_VAL = 1  # neutral tonemap (default)
EXPOSURE = 1.0

out = torch.randint(0, 256, (FRAMES, PPF, 5), dtype=torch.uint8, device=DEV)
pix_accum = torch.rand((TOTAL, 7), dtype=torch.float32, device=DEV)


def run(tiles, empty):
    """Composite the whole window in ``tiles`` equal launches; return device
    seconds for the whole window.
    """
    per = (TOTAL + tiles - 1) // tiles
    _sync_devices()
    t0 = time.perf_counter()
    start = 0
    while start < TOTAL:
        n = min(per, TOTAL - start)
        pa = pix_accum[start : start + n]
        wf_composite_accum(0, W, H, 0, start, pa, T_VAL, EXPOSURE, empty, out)
        start += n
    _sync_devices()
    return time.perf_counter() - t0


# Warm compile + caches (both instantiations).
for _ in range(3):
    run(16, 0)
    run(16, 1)

print(f"total pixels/window = {TOTAL:,}  ({FRAMES} frames @ {W}x{H})")
print("full (empty=0) vs lean (empty=1, no pix_accum read):")
print(f"{'tiles':>6} {'per-tile px':>12} {'full ms':>9} {'lean ms':>9} {'speedup':>8}")
for tiles in (1, 4, 16, 32, 64, 128, 240):
    full = min(run(tiles, 0) for _ in range(6))
    lean = min(run(tiles, 1) for _ in range(6))
    print(
        f"{tiles:>6} {TOTAL // tiles:>12,} {full * 1e3:>9.2f} "
        f"{lean * 1e3:>9.2f} {full / lean:>7.2f}x"
    )
