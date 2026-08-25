"""Does a Taichi kernel see torch's pending default-stream writes without an explicit sync?"""
import os
os.environ["ALGAN_USE_DAEMON"] = "0"
import torch, taichi as ti
import algan  # initialises Taichi the engine's way
from algan.rendering.taichi_runtime import init_taichi
init_taichi()

@ti.kernel
def checksum(x: ti.types.ndarray(), out: ti.types.ndarray()):
    for i in range(x.shape[0]):
        ti.atomic_min(out[0], x[i])

n = 1 << 26  # 64M floats = 256 MB
x = torch.zeros(n, device="cuda", dtype=torch.float32)
out = torch.zeros(1, device="cuda", dtype=torch.float32)
torch.cuda.synchronize()
mismatches = 0
for trial in range(6):
    # queue a long chain of torch work on the default stream that ends by writing x
    a = torch.randn(4096, 4096, device="cuda")
    for _ in range(30):
        a = a @ a * 1e-3  # ~ tens of ms each
    x.fill_(1.0)          # queued after the matmuls: x is not 1.0 until they finish
    out.fill_(2.0)
    checksum(x, out)      # Taichi stream: does it wait for the fill?
    ti.sync(); torch.cuda.synchronize()
    got = float(out.item())
    ok = got == 1.0
    mismatches += (not ok)
    print(f"trial {trial}: min {got} expected 1.0 -> {'OK' if ok else 'STALE READ'}")
    x.zero_(); torch.cuda.synchronize()
print("mismatches", mismatches)
