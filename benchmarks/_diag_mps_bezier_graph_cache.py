"""Diagnostic for process-lifetime MPSGraph growth from shape-varying Bezier cache ops."""

import argparse
import gc
import os
import subprocess
import time
from types import SimpleNamespace

import torch

from algan.rendering.raytracing.bezier_geometry_cache import _build_cached_circuit_edges
from algan.rendering.raytracing.raster_pipeline import _aabb_corners


def rss_mb():
    raw = subprocess.check_output(
        ["ps", "-o", "rss=", "-p", str(os.getpid())], text=True
    ).strip()
    return int(raw) / 1024.0


def mps_mb():
    return (
        torch.mps.current_allocated_memory() / (1024 * 1024),
        torch.mps.driver_allocated_memory() / (1024 * 1024),
    )


def make_args(circuits, device):
    frames = 8
    segments_per = 4
    segments_total = circuits * segments_per

    # Build on CPU so the diagnostic is not accidentally dominated by
    # shape-specialized MPS graph ops used only to synthesize the inputs.
    corners = torch.zeros((frames, segments_total, 4, 3), dtype=torch.float32).to(device)
    samples = torch.full((segments_total,), 4, dtype=torch.int32).to(device)
    segments = torch.full((circuits,), segments_per, dtype=torch.int64).to(device)
    next0 = torch.arange(segments_total, dtype=torch.int64).reshape(1, -1)
    next_inds = next0.expand(frames, -1).contiguous().to(device)
    centers = torch.zeros((frames, circuits, 3), dtype=torch.float32).to(device)
    basis_u_cpu = torch.zeros((frames, circuits, 3), dtype=torch.float32)
    basis_v_cpu = torch.zeros((frames, circuits, 3), dtype=torch.float32)
    basis_u_cpu[..., 0] = 1.0
    basis_v_cpu[..., 1] = 1.0
    basis_u = basis_u_cpu.to(device)
    basis_v = basis_v_cpu.to(device)
    return corners, samples, segments, next_inds, centers, basis_u, basis_v


def dummy_build(corners, samples, segments, next_inds, centers, basis_u, basis_v, inward_signs):
    del samples, next_inds, centers, basis_u, basis_v, inward_signs
    # Avoid cumsum/cat on MPS: this control is intended to be allocator/copy
    # traffic, not another pile of shape-keyed graph operators.
    count = int(segments.shape[0])
    offsets_cpu = torch.arange(count + 1, dtype=torch.int32) * 8
    offsets = offsets_cpu.to(corners.device)
    edges = torch.empty((corners.shape[0], count * 8, 6), dtype=corners.dtype, device=corners.device)
    return edges, offsets


def run(mode, calls):
    device = torch.device("mps")
    assert torch.backends.mps.is_available()
    scene = SimpleNamespace()
    t0 = time.perf_counter()

    # For varying modes the FIRST call is the largest live tensor footprint.
    # Every later call is smaller, so rising current RSS cannot be explained
    # by encountering a larger working set.
    sizes = [calls] * calls if mode == "fixed" else list(range(calls, 0, -1))
    torch.mps.empty_cache()
    gc.collect()
    start_rss = rss_mb()
    print("torch", torch.__version__, "mode", mode, "calls", calls, "pid", os.getpid(), flush=True)
    print("start_rss_mb", round(start_rss, 1), "mps_mb", tuple(round(x, 1) for x in mps_mb()), flush=True)

    for step, circuits in enumerate(sizes, 1):
        args = make_args(circuits, device)
        if mode == "cache":
            out = _build_cached_circuit_edges(scene, dummy_build, args, False)
            cache = scene.__dict__.pop("_bezier_geometry_cache", None)
            if cache is not None:
                cache.clear()
        elif mode == "aabb":
            corners = _aabb_corners(args[4], args[4] + 1.0)
            out = (corners,)
        else:
            out = dummy_build(*args, False)

        torch.mps.synchronize()
        del out, args
        gc.collect()
        torch.mps.empty_cache()
        if step == 1 or step % 25 == 0 or step == calls:
            cur, drv = mps_mb()
            now_rss = rss_mb()
            print(
                "STEP", step,
                "circuits", circuits,
                "rss_mb", round(now_rss, 1),
                "delta_rss_mb", round(now_rss - start_rss, 1),
                "mps_cur_mb", round(cur, 1),
                "mps_drv_mb", round(drv, 1),
                "elapsed_s", round(time.perf_counter() - t0, 1),
                flush=True,
            )

    lo = torch.zeros((8, 16, 3), device=device)
    hi = torch.ones((8, 16, 3), device=device)
    probe = _aabb_corners(lo, hi)
    torch.mps.synchronize()
    print("PROBE", tuple(probe.shape), "final_rss_mb", round(rss_mb(), 1), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("fixed", "control", "aabb", "cache"), required=True)
    p.add_argument("--calls", type=int, default=300)
    ns = p.parse_args()
    run(ns.mode, ns.calls)
