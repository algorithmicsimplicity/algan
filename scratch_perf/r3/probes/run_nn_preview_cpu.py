"""Reproduction driver: runs the stock nn-scene PREVIEW benchmark on this
CPU-only container. The profiler's kernel wrapper calls torch.cuda.synchronize()
unconditionally (profiling_utils.py:381), which raises with no NVIDIA driver,
so shim it to a no-op before the benchmark imports anything.
"""

from __future__ import annotations

import runpy
import sys
import time

import torch

if not torch.cuda.is_available():
    torch.cuda.synchronize = lambda *a, **k: None
    print("[probe] shimmed torch.cuda.synchronize (no CUDA device)")

t0 = time.time()
sys.argv = ["nn_scene_PREVIEW.py"]

runpy.run_path(
    "/home/user/algan/benchmarks/performance/nn_scene_PREVIEW.py", run_name="__main__"
)
print(f"[probe] done in {time.time() - t0:.1f}s")
