"""Bounded allocator-only control; never more than one 1.12 GiB tensor is live."""
import ctypes
import os
import time
from pathlib import Path

import psutil
import torch


def main():
    assert torch.__version__.split('+')[0] == '2.13.0'
    assert torch.backends.mps.is_available()
    output = Path('algan_outputs')
    output.mkdir(exist_ok=True)
    lib = ctypes.CDLL(str(Path(os.environ['RUNNER_TEMP']) / 'pr144_command_trace.dylib'))
    lib.pr144_install.argtypes = [ctypes.c_char_p]
    lib.pr144_install.restype = ctypes.c_int
    lib.pr144_enable.argtypes = [ctypes.c_int]
    lib.pr144_context.argtypes = [ctypes.c_char_p]
    lib.pr144_errors.argtypes = []
    lib.pr144_errors.restype = ctypes.c_uint
    assert lib.pr144_install(str(output / 'heap-errors.jsonl').encode()) == 1
    lib.pr144_enable(1)
    process = psutil.Process()
    start = time.monotonic()
    size = 1202590840  # Matches 0.4 * 70% of the four-GiB runner's host RAM.
    assert size < 0.5 * torch.mps.recommended_max_memory()
    for iteration in range(512):
        os.environ['PR144_CURRENT_TEST'] = f'heap-churn-{iteration}'
        assert psutil.virtual_memory().available > 1536 * 1024**2, 'memory safety stop before allocation'
        lib.pr144_context(f'large-allocation-{iteration}'.encode())
        data = torch.empty(size, dtype=torch.uint8, device='mps')
        assert not lib.pr144_errors(), ('allocation failed natively', iteration)
        data[:8].fill_(7)
        data[-8:].fill_(13)
        assert data[:8].cpu().tolist() == [7] * 8
        assert data[-8:].cpu().tolist() == [13] * 8
        assert not lib.pr144_errors(), ('touch failed natively', iteration)
        del data
        torch.mps.empty_cache()
        small = torch.ones(8, device='mps')
        assert small.cpu().tolist() == [1.0] * 8
        del small
        torch.mps.empty_cache()
        assert not lib.pr144_errors(), ('reclamation failed natively', iteration)
        if iteration % 32 == 31:
            print('HEAP_CHURN', iteration + 1, 'rss', process.memory_info().rss, 'driver', torch.mps.driver_allocated_memory(), 'elapsed', time.monotonic() - start, flush=True)
    print('HEAP_CHURN_PASS', 512, flush=True)


if __name__ == '__main__':
    main()
