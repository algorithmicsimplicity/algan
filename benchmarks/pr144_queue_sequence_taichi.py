"""Fixed, tiny allocations; test submission/event count rather than pipeline count."""
import ctypes
import os
import time
from pathlib import Path

import psutil
import torch

from algan.rendering import mps_zero_copy
from algan.rendering.taichi_runtime import init_taichi, shared_torch_queue
from algan.taichi_compat import ti


@ti.kernel
def increment(out: ti.types.ndarray(dtype=ti.f32, ndim=1)):
    for i in range(out.shape[0]):
        out[i] += 1.0


def main():
    assert torch.__version__.split('+')[0] == '2.13.0'
    assert torch.backends.mps.is_available()
    folder = Path('algan_outputs')
    folder.mkdir(exist_ok=True)
    library = ctypes.CDLL(str(Path(os.environ['RUNNER_TEMP']) / 'pr144_command_trace.dylib'))
    library.pr144_install.argtypes = [ctypes.c_char_p]
    library.pr144_install.restype = ctypes.c_int
    library.pr144_enable.argtypes = [ctypes.c_int]
    library.pr144_context.argtypes = [ctypes.c_char_p]
    library.pr144_errors.argtypes = []
    library.pr144_errors.restype = ctypes.c_uint
    assert library.pr144_install(str(folder / 'queue-errors.jsonl').encode()) == 1
    # Record completion errors; omit per-dispatch text from the synthetic hot loop.
    library.pr144_enable(0)
    init_taichi()
    assert shared_torch_queue(), 'This probe must exercise the shared-queue regime'
    out = torch.empty(8, dtype=torch.float32, device='mps')
    process = psutil.Process()
    start = time.monotonic()
    initial = mps_zero_copy.STATS['shared_queue_launches']
    for index in range(131072):
        os.environ['PR144_CURRENT_TEST'] = f'queue-sequence-{index}'
        out.fill_(2.0)
        increment(out)
        out.add_(4.0)
        if index % 256 == 255:
            values = out.cpu().tolist()
            assert values == [7.0] * 8, (index, values)
            assert library.pr144_errors() == 0, ('native error', index)
            rss = process.memory_info().rss
            assert rss < 2 * 1024**3 and psutil.virtual_memory().available > 768 * 1024**2, 'memory safety stop'
            if index % 4096 == 4095:
                print('QUEUE_SEQUENCE', index + 1, 'shared', mps_zero_copy.STATS['shared_queue_launches'] - initial, 'rss', rss, 'elapsed', time.monotonic() - start, flush=True)
    assert mps_zero_copy.STATS['shared_queue_launches'] - initial == 131072
    print('QUEUE_SEQUENCE_PASS', 131072, flush=True)


if __name__ == '__main__':
    main()
