"""Isolate pipeline count from tensor-shape and shader-size growth."""
import argparse
import gc
import json
import time

import psutil
import torch

from algan.rendering.taichi_runtime import init_taichi
from algan.taichi_compat import ti


@ti.kernel
def pipeline(out: ti.types.ndarray(dtype=ti.f32, ndim=1), value: ti.template()):
    for i in range(out.shape[0]):
        out[i] = float(value) + float(i)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reset-every', type=int, default=0)
    parser.add_argument('--count', type=int, default=1200)
    args = parser.parse_args()
    assert 1 <= args.count <= 2000
    init_taichi()
    out = torch.empty(8, device='mps')
    process = psutil.Process()
    start = time.perf_counter()
    print('ENV', torch.__version__, 'count', args.count, 'reset', args.reset_every, flush=True)
    for i in range(args.count):
        if args.reset_every and i and i % args.reset_every == 0:
            ti.reset()
            init_taichi()
            gc.collect()
            torch.mps.empty_cache()
        pipeline(out, i)
        ti.sync()
        torch.mps.synchronize()
        assert out.cpu().tolist() == [float(i + j) for j in range(8)]
        if i % 100 == 0:
            print('PIPEMEM', json.dumps({'i': i, 'rss': process.memory_info().rss,
                'torch_live': torch.mps.current_allocated_memory(),
                'torch_driver': torch.mps.driver_allocated_memory(),
                'elapsed': time.perf_counter() - start}), flush=True)
        if process.memory_info().rss > 3 * 1024 ** 3:
            print('MEMORY_SAFETY_STOP', i, flush=True)
            break
    print('PASS', i + 1, flush=True)


if __name__ == '__main__':
    main()
