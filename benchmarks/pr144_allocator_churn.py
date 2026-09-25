"""Fixed-size allocator replay: at most one arena-sized tensor, no renderer."""
from __future__ import annotations

import ctypes
import gc
import os
from pathlib import Path
import sys
import time

import psutil
import torch


def main():
    # Exact observed CI arena size; not a command-line-controlled generator.
    # A single tensor is bounded below 1.2 GiB, with small fixed sentinels.
    arena_bytes = 1_202_590_720
    assert arena_bytes < 1200 * 1024 ** 2
    assert torch.__version__.split('+')[0] == '2.13.0'
    assert torch.backends.mps.is_available()
    mode = sys.argv[1]
    assert mode in ('allocate', 'fill')
    process = psutil.Process()
    lib = ctypes.CDLL(str(Path(os.environ['RUNNER_TEMP']) / 'pr144_command_provenance.dylib'))
    lib.pr144_install.restype = ctypes.c_int
    lib.pr144_test.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.pr144_op.argtypes = [ctypes.c_char_p]
    assert lib.pr144_install() == 1
    start = time.monotonic()
    for iteration in range(256):
        if psutil.virtual_memory().available < 1536 * 1024 ** 2 or process.memory_info().rss > 2560 * 1024 ** 2:
            raise RuntimeError('Allocator replay reached its fixed memory safety boundary')
        lib.pr144_test(f'allocator/{mode}/{iteration}'.encode(), 1)
        lib.pr144_op(b'torch.empty/CI-arena')
        block = torch.empty(arena_bytes, dtype=torch.uint8, device='mps')
        lib.pr144_op(b'fill/CI-arena' if mode == 'fill' else b'fill/first-eight-bytes')
        if mode == 'fill':
            block.fill_(37)
        else:
            block[:8].fill_(37)
        lib.pr144_op(b'CPU-sentinel-readback')
        assert block[:8].cpu().tolist() == [37] * 8
        torch.mps.synchronize()
        del block
        lib.pr144_op(b'empty-cache')
        torch.mps.empty_cache()
        if iteration % 16 == 0:
            gc.collect()
            print('ALLOCATOR_PASS', mode, iteration, 'rss', process.memory_info().rss,
                  'driver', torch.mps.driver_allocated_memory(), 'elapsed', time.monotonic()-start, flush=True)
    print('ALLOCATOR_ALL_PASS', mode, 256, flush=True)


if __name__ == '__main__':
    main()
