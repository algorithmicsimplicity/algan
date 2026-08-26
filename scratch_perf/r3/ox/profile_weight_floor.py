"""Per-arm profile driver: stock nn PREVIEW benchmark with the weight-floor
exit forced. One process per arm. Writes the standard profile report (bounce
table + 'drain active count' launches) to CWD.
"""

from __future__ import annotations

import os
import runpy
import sys

os.environ["ALGAN_USE_DAEMON"] = "0"

ARM = sys.argv[1]
assert ARM in ("off", "on")

import torch

if not torch.cuda.is_available():
    torch.cuda.synchronize = lambda *a, **k: None
    print("[profile] shimmed torch.cuda.synchronize (no CUDA device)")

from algan import SETTINGS

SETTINGS.raytracing.experimental.weight_floor_exit = ARM == "on"
print(
    "[profile] weight_floor_exit =",
    SETTINGS.raytracing.experimental.weight_floor_exit,
)

sys.argv = ["nn_scene_PREVIEW.py"]
runpy.run_path(
    "/home/user/algan/benchmarks/performance/nn_scene_PREVIEW.py", run_name="__main__"
)
