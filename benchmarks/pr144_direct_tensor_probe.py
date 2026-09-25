"""Standalone native buffers must preserve Torch semantics without CPU staging."""
import ctypes
import gc
import os
import time
from pathlib import Path

import psutil
import torch


def main():
    assert torch.__version__.split('+')[0] == '2.13.0'
    assert torch.backends.mps.is_available()
    library = ctypes.PyDLL(str(Path(os.environ['RUNNER_TEMP']) / 'pr144_direct_tensor.dylib'))
    library.pr144_direct_allocate.argtypes = [ctypes.c_uint64]
    library.pr144_direct_allocate.restype = ctypes.py_object
    library.pr144_direct_live_bytes.argtypes = []
    library.pr144_direct_live_bytes.restype = ctypes.c_uint64

    def allocate(size):
        return torch.from_dlpack(library.pr144_direct_allocate(size))

    abandoned = library.pr144_direct_allocate(1024)
    assert library.pr144_direct_live_bytes() == 1024
    del abandoned
    gc.collect()
    assert library.pr144_direct_live_bytes() == 0, 'An unconsumed capsule must release its native buffer'

    parent = allocate(1024)
    assert parent.device.type == 'mps' and parent.dtype == torch.uint8
    view = parent[128:256].view(torch.float32).reshape(4, 8)
    view.fill_(42)
    del parent
    gc.collect()
    assert library.pr144_direct_live_bytes() == 1024, 'Typed views must retain native storage'
    assert torch.flip(view, dims=(0, 1)).cpu().tolist() == [[42.0] * 8] * 4
    assert float(view.amax().cpu()) == 42.0
    torch.mps.empty_cache()
    assert library.pr144_direct_live_bytes() == 1024, 'Allocator reclamation must not invalidate live imported buffers'
    assert view.cpu().tolist() == [[42.0] * 8] * 4
    del view
    torch.mps.synchronize()
    gc.collect()
    assert library.pr144_direct_live_bytes() == 0, 'The final tensor view must release the native buffer'
    print('DIRECT_TENSOR_LIFETIME_PASS', flush=True)

    process = psutil.Process()
    start = time.monotonic()
    for iteration in range(768):
        assert psutil.virtual_memory().available > 1536 * 1024**2, 'Memory safety stop'
        data = allocate(1202590840)
        data[:8].fill_(7)
        data[-8:].fill_(13)
        assert data[:8].cpu().tolist() == [7] * 8
        assert data[-8:].cpu().tolist() == [13] * 8
        del data
        torch.mps.empty_cache()
        assert library.pr144_direct_live_bytes() == 0
        if iteration % 32 == 31:
            print('DIRECT_TENSOR_CHURN', iteration + 1, 'rss', process.memory_info().rss, 'driver', torch.mps.driver_allocated_memory(), 'elapsed', time.monotonic() - start, flush=True)
    print('DIRECT_TENSOR_CHURN_PASS', 768, flush=True)


if __name__ == '__main__':
    main()
