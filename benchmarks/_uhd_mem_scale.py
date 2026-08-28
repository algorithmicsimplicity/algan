"""Arena vs non-arena peak as a function of frame resolution (one sphere)."""

import os
import sys

os.environ.setdefault("ALGAN_USE_DAEMON", "0")
import torch

import algan  # noqa: F401
from algan import *  # noqa: F401,F403
from algan.rendering.raytracing import raster_pipeline, sheets
from algan.utils import memory_utils as mu

GB = 1 << 30
MB = 1 << 20
W, H, frac = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
SETTINGS.computing.set(rendering_memory_fraction=frac)

arenas = []
_orig_init = mu.ManualMemory.__init__
_orig_get = mu.ManualMemory.get_tensor
hw = [0]


def init(self, portion, device=None, managed=True, *, num_bytes=None):
    _orig_init(self, portion, device=device, managed=managed, num_bytes=num_bytes)
    if managed:
        arenas.append(self)


def get_tensor(self, shape, dtype=torch.float, persist=False):
    x = _orig_get(self, shape, dtype=dtype, persist=persist)
    if self.managed:
        hw[0] = max(
            hw[0], self.current_pointer + (self.length - self.current_reverse_pointer)
        )
    return x


mu.ManualMemory.__init__ = init
mu.ManualMemory.get_tensor = get_tensor

info = {}
_oc = sheets.compact_sheets


def wrap(coverage, *a, **k):
    info["frags"] = int(coverage["num_fragments"])
    info["cov"] = int(coverage["num_covered"])
    return _oc(coverage, *a, **k)


sheets.compact_sheets = wrap
raster_pipeline.compact_sheets = wrap

VS = UHD.set(resolution=(W, H))
x = Sphere().scale(3).spawn()
status = "OK"
try:
    Scene.save_frame("_scale_probe.png", VS)
except Exception as e:
    status = type(e).__name__
arena_bytes = sum(a.length for a in arenas)
hwv = hw[0]
peak = torch.cuda.max_memory_allocated()
px = W * H
print(
    f"{W}x{H} px={px / 1e6:6.2f}M frac={frac} {status:22s} "
    f"frags={info.get('frags', 0) / 1e6:5.2f}M cov={info.get('cov', 0) / 1e6:5.2f}M | "
    f"arena_reserved={arena_bytes / MB:7.1f}MB arena_used={hwv / MB:7.1f}MB "
    f"peak_total={peak / MB:7.1f}MB nonarena={(peak - arena_bytes) / MB:7.1f}MB"
)
