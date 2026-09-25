"""Separate pipeline-count limits and autorelease lifetime from rendering."""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import time

os.environ["ALGAN_AUTO_DAEMON"] = "0"

import psutil
import torch
from algan.rendering.taichi_runtime import init_taichi
from algan.taichi_compat import ti


@ti.kernel
def pipeline(out: ti.types.ndarray(dtype=ti.f32, ndim=1), value: ti.template()):
    for i in range(out.shape[0]):
        out[i] = float(value) + float(i)


def native_probe():
    import Metal
    device = Metal.MTLCreateSystemDefaultDevice()
    source = '#include <metal_stdlib>\nusing namespace metal;\nkernel void probe(device float *p [[buffer(0)]], uint i [[thread_position_in_grid]]) {p[i] = 37.0f;}'
    library, error = device.newLibraryWithSource_options_error_(source, None, None)
    print('NATIVE_LIBRARY', bool(library), repr(error), flush=True)
    if library:
        function = library.newFunctionWithName_('probe')
        state, error = device.newComputePipelineStateWithFunction_error_(function, None)
        print('NATIVE_PIPELINE', bool(state), repr(error), flush=True)
        if error:
            print('NATIVE_ERROR', error.domain(), error.code(), error.userInfo(), flush=True)
    print('METAL_BYTES', device.currentAllocatedSize(), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pool', action='store_true')
    parser.add_argument('--count', type=int, default=16384)
    args = parser.parse_args()
    assert 1 <= args.count <= 16384
    lib = ctypes.CDLL('/usr/lib/libobjc.A.dylib')
    push, pop = lib.objc_autoreleasePoolPush, lib.objc_autoreleasePoolPop
    push.argtypes, push.restype = [], ctypes.c_void_p
    pop.argtypes, pop.restype = [ctypes.c_void_p], None
    init_taichi()
    out = torch.empty(8, device='mps')
    process = psutil.Process()
    start = time.monotonic()
    for i in range(args.count):
        token = push() if args.pool else None
        try:
            pipeline(out, i)
            ti.sync()
            torch.mps.synchronize()
            assert out.cpu().tolist() == [float(i+j) for j in range(8)]
        except Exception:
            print('FIRST_FAILURE', i, flush=True)
            native_probe()
            raise
        finally:
            if args.pool:
                pop(token)
        if i % 256 == 0:
            row = {'i':i,'pool':args.pool,'rss':process.memory_info().rss,'driver':torch.mps.driver_allocated_memory(),'available':psutil.virtual_memory().available,'elapsed':time.monotonic()-start}
            print('PIPELINE_LIMIT', json.dumps(row), flush=True)
            if row['rss'] > 2 * 1024**3 or row['available'] < 768 * 1024**2:
                raise RuntimeError('Bounded probe stopped at memory safety threshold')
    print('ALL_PIPELINES_PASS', args.count, 'pool', args.pool, flush=True)


if __name__ == '__main__':
    main()
