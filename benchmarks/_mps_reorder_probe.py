"""Diagnose and bypass MPS index_select(out=...) on shared arena views."""
import torch
from algan.taichi_compat import ti
import algan.rendering.raytracing.tracer as tracer

@ti.kernel
def reorder(source: ti.types.ndarray(), perm: ti.types.ndarray(),
            output: ti.types.ndarray(), count: int):
    for i in range(count):
        output[i] = source[perm[i]]

def install():
    calls = 0
    def reordered(self, active, perm):
        nonlocal calls
        n = int(active.numel())
        calls += 1
        if calls == 1:
            a, p = active.cpu(), perm.cpu()
            print("REORDER_INPUT", n, a.min().item(), a.max().item(),
                  p.min().item(), p.max().item(), p.unique().numel(), flush=True)
            assert p.min() >= 0 and p.max() < n
            expected = a[p.long()]
            torch.index_select(active, 0, perm, out=self.spare[:n])
            old = self.spare[:n].cpu()
            print("REORDER_OLD", torch.equal(old,expected), old[:20].tolist(),
                  "expected",expected[:20].tolist(),
                  "source_offset",active.storage_offset(),
                  "out_offset",self.spare.storage_offset(),
                  "same_storage",active.untyped_storage().data_ptr()==self.spare.untyped_storage().data_ptr(),
                  flush=True)
        reorder(active, perm, self.spare, n)
        if calls == 1:
            got = self.spare[:n].cpu()
            print("REORDER_FIXED", torch.equal(got,expected), got[:20].tolist(), flush=True)
            torch.testing.assert_close(got,expected,rtol=0,atol=0)
        self.current, self.spare = self.spare, self.current
        self.size = n
        return self.current[:n]
    tracer._ArenaRayCompactor.reorder = reordered
