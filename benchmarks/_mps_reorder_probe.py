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
            # Reproduce the faulty write on disposable storage: exercising
            # it on the live arena would corrupt its prefix before the fix.
            probe = torch.full((64,), -123, dtype=torch.int32, device=active.device)
            probe[8:15].copy_(torch.arange(23,30,dtype=torch.int32,device=active.device))
            indexes = torch.tensor([4,1,6,0,5,2,3],device=active.device)
            wanted = torch.tensor([27,24,29,23,28,25,26],dtype=torch.int32)
            torch.index_select(probe[8:15],0,indexes,out=probe[24:31])
            old = probe.cpu()
            print("REORDER_OLD",torch.equal(old[24:31],wanted),
                  "prefix",old[:7].tolist(),"out",old[24:31].tolist(),
                  "expected",wanted.tolist(),flush=True)
        reorder(active, perm, self.spare, n)
        if calls == 1:
            got = self.spare[:n].cpu()
            print("REORDER_FIXED", torch.equal(got,expected), got[:20].tolist(), flush=True)
            torch.testing.assert_close(got,expected,rtol=0,atol=0)
        self.current, self.spare = self.spare, self.current
        self.size = n
        return self.current[:n]
    tracer._ArenaRayCompactor.reorder = reordered
