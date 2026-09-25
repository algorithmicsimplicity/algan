"""Diagnostic only: isolate native autorelease lifetime without resetting Metal."""
from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path

import pytest

_counts = {"calls": 0}


def pytest_sessionstart(session):
    import torch
    from algan.taichi_compat import submodule

    assert torch.__version__.split("+")[0] == "2.13.0"
    assert torch.backends.mps.is_available()
    library = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
    push = library.objc_autoreleasePoolPush
    push.argtypes = []
    push.restype = ctypes.c_void_p
    pop = library.objc_autoreleasePoolPop
    pop.argtypes = [ctypes.c_void_p]
    pop.restype = None
    kernel = submodule("lang.kernel_impl").Kernel
    original = kernel.__call__

    def scoped_call(self, *args, **kwargs):
        # Every thread pushes and pops its OWN nested pool. A pool owns only
        # autoreleased temporaries; compiler-owned pipelines remain retained.
        # No runtime reset, cache clear, synchronization, or retry is added.
        token = push()
        try:
            _counts["calls"] += 1
            return original(self, *args, **kwargs)
        finally:
            pop(token)

    kernel.__call__ = scoped_call
    print("AUTORELEASE_PROBE", torch.__version__, "one process; scoped native temporaries", flush=True)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    yield
    if nextitem is not None and item.path == nextitem.path:
        return
    import psutil
    import torch
    from algan.rendering.mps_zero_copy import cache_stats
    from algan.taichi_compat import program

    proc = psutil.Process()
    row = {
        "pid": os.getpid(), "after": item.nodeid,
        "rss": proc.memory_info().rss,
        "available": psutil.virtual_memory().available,
        "fds": proc.num_fds(), "threads": proc.num_threads(),
        "torch_live": torch.mps.current_allocated_memory(),
        "torch_driver": torch.mps.driver_allocated_memory(),
        "imports": cache_stats(), "program": id(program()),
        "pool_calls": _counts["calls"],
    }
    folder = Path("algan_outputs")
    folder.mkdir(exist_ok=True)
    with (folder / f"autorelease-{os.getpid()}.jsonl").open("a") as stream:
        stream.write(json.dumps(row) + "\n")
    print("POOL_RESOURCES", json.dumps(row), flush=True)
