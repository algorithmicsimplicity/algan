"""Measure exact captured UHD depth inputs: wall time, temporary bytes, GPU kernels."""

# ruff: noqa: E402 -- install the memory cap before importing torch.

from __future__ import annotations

import os
import sys
import time

from benchmarks._memory_cap import cap_process_memory

cap_process_memory(4)
os.environ["ALGAN_USE_DAEMON"] = "0"

import torch

from algan.rendering.raytracing import sheets
from algan.rendering.raytracing.sheet_depth_taichi import sheet_depth_lose
from algan.rendering.taichi_runtime import taichi_init_kwargs
from algan.taichi_compat import ti

ti.init(**taichi_init_kwargs(), kernel_profiler=True)
values = tuple(x.cuda() for x in torch.load(sys.argv[1], weights_only=True))
pix, depths, sid, enforcer, subject, low = values
n = pix.numel()


def kernel():
    out = torch.empty(n, dtype=torch.int32, device=pix.device)
    sheet_depth_lose(
        pix,
        sid,
        depths,
        low,
        subject.view(torch.uint8),
        enforcer.view(torch.uint8),
        n,
        float(sheets.depth_tie_epsilon),
        float(sheets.sheet_sample_depth_cede),
        int(sheets.AA_LOSE_SHIFT),
        out,
    )
    return out


want = sheets._sample_depth_lose_reference(*values)
assert torch.equal(kernel(), want)
ti.profiler.clear_kernel_profiler_info()
for repeat in range(6):
    for name, function in (
        ("reference", lambda: sheets._sample_depth_lose_reference(*values)),
        ("kernel", kernel),
    ):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        started = time.perf_counter()
        result = function()
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        extra = torch.cuda.max_memory_allocated() - base
        assert torch.equal(result, want)
        print(
            f"{repeat} {name}: {seconds * 1000:.3f} ms, {extra / 2**20:.3f} MiB temporary",
            flush=True,
        )
        del result
ti.profiler.print_kernel_profiler_info()
