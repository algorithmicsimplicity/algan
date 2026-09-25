"""Bounded, fixed-shape dispatch probe for shared-queue command lifetime."""
import argparse
import gc
import json
import os
import time

import psutil
import torch

from algan.rendering.mps_zero_copy import cache_stats
from algan.rendering.taichi_runtime import init_taichi, shared_torch_queue
from algan.taichi_compat import ti


@ti.kernel
def probe(src: ti.types.ndarray(dtype=ti.f32, ndim=1),
          out: ti.types.ndarray(dtype=ti.f32, ndim=1)):
    for i in range(out.shape[0]):
        out[i] = src[i] + 1.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--drain', action='store_true')
    parser.add_argument('--calls', type=int, default=30000)
    args = parser.parse_args()
    assert 1 <= args.calls <= 60000
    init_taichi()
    assert shared_torch_queue()
    process = psutil.Process()
    start = time.perf_counter()
    src = torch.arange(32, dtype=torch.float32, device='mps')
    out = torch.empty_like(src)
    probe(src, out)
    torch.mps.synchronize()
    ti.sync()

    def report(label):
        print('MEM', json.dumps({
            'label': label, 'drain': args.drain,
            'rss': process.memory_info().rss,
            'torch_live': torch.mps.current_allocated_memory(),
            'torch_driver': torch.mps.driver_allocated_memory(),
            'imports': cache_stats(), 'seconds': time.perf_counter() - start,
        }), flush=True)

    report('start')
    for step in range(1, args.calls + 1):
        probe(src, out)
        if step % 500 == 0:
            # Both arms wait for GPU completion. Only the drain arm asks
            # Quadrants to release the completed command buffers it retains.
            torch.mps.synchronize()
            if args.drain:
                ti.sync()
            if step % 5000 == 0:
                report(step)
            assert torch.equal(out.cpu(), torch.arange(32, dtype=torch.float32) + 1)
            if process.memory_info().rss > 3 * 1024 ** 3:
                print('MEMORY_SAFETY_STOP', step, flush=True)
                break
    report('before-final-sync')
    ti.sync()
    gc.collect()
    torch.mps.empty_cache()
    report('after-final-sync')
    print('PASS', os.getpid(), flush=True)


if __name__ == '__main__':
    main()
