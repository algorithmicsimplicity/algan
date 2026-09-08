"""Probe global temporaries shared by serial and parallel Metal tasks."""
from pathlib import Path
import os, sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "algan_outputs" / "metal_globals"
OUT.mkdir(parents=True, exist_ok=True)
os.environ["QD_SHADER_DUMP_DIR"] = str(OUT)
import torch
from algan.taichi_compat import ti
from algan.rendering.taichi_runtime import init_taichi
from algan.settings._startup import render_device
init_taichi()

@ti.kernel
def serial_int(out: ti.types.ndarray(), n: int):
    a = n + 3
    b = n + 4
    for i in range(n):
        out[i] = a + b + i

@ti.kernel
def serial_mixed(out: ti.types.ndarray(), n: int, x: float):
    a = n + 3
    b = x * 2.0
    for i in range(n):
        out[i] = ti.cast(a + i, ti.f32) + b

@ti.kernel
def inline_mixed(out: ti.types.ndarray(), n: int, x: float):
    for i in range(n):
        a = n + 3
        b = x * 2.0
        out[i] = ti.cast(a + i, ti.f32) + b

for n in (8, 1025):
    out = torch.full((n,), -100.0, device=render_device())
    for k in (serial_int, serial_mixed, inline_mixed):
        out.fill_(-100)
        k(out, n, 1.25) if k is not serial_int else k(out, n)
        ti.sync()
        got = out.cpu()
        want = torch.arange(n).float() + (2*n+7 if k is serial_int else n+5.5)
        print("GLOBAL_PROBE", k.__name__, n, torch.equal(got,want),
              got[:10].tolist(), "expected",want[:10].tolist(), flush=True)
