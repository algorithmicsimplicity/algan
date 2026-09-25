"""Read-only per-module telemetry for the exact CI workload."""
from __future__ import annotations

import json
import os
import resource
import sys
from pathlib import Path

import pytest


def pytest_sessionstart(session):
    import torch
    import torchaudio
    print('DIAG_ENV', sys.executable, sys.version, torch.__version__, torchaudio.__version__, resource.getrlimit(resource.RLIMIT_NOFILE), flush=True)
    assert torch.__version__.split('+')[0] == '2.13.0'


def pytest_collection_modifyitems(config, items):
    if os.environ.get('PR144_SLICE') != 'nr_tail':
        return
    selected = [item for item in items if Path(str(item.fspath)).name >= 'test_n' or '/fast/' in str(item.fspath) or 'test_soft_shadow_fans_compile_and_render_one_frame' in item.nodeid]
    config.hook.pytest_deselected(items=[item for item in items if item not in selected])
    items[:] = selected


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
        'pid': os.getpid(), 'after': item.nodeid,
        'rss': proc.memory_info().rss,
        'available': psutil.virtual_memory().available,
        'fds': proc.num_fds(), 'threads': proc.num_threads(),
        'torch_live': torch.mps.current_allocated_memory(),
        'torch_driver': torch.mps.driver_allocated_memory(),
        'imports': cache_stats(), 'program': id(program()),
    }
    folder = Path('algan_outputs')
    folder.mkdir(exist_ok=True)
    with (folder / f'resources-{os.getpid()}.jsonl').open('a') as stream:
        stream.write(json.dumps(row) + '\n')
    print('DIAG_RESOURCES', json.dumps(row), flush=True)
