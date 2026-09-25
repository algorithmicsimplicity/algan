"""Diagnostic for process-lifetime MPSGraph growth from shape-varying Bezier cache ops."""

import argparse
import gc
import resource
import time
from types import SimpleNamespace

import torch

from algan.rendering.raytracing.bezier_geometry_cache import _build_cached_circuit_edges
from algan.rendering.raytracing.raster_pipeline import _aabb_corners


def rss_mb():
    # macOS reports ru_maxrss in bytes.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def mps_mb():
    return (
        torch.mps.current_allocated_memory() / (1024 * 1024),
        torch.mps.driver_allocated_memory() / (1024 * 1024),
    )


def make_args(index, device):
    frames = 2 + index % 7
    circuits = index
    segments_per = 4
    segments_total = circuits * segments_per

    corners = torch.zeros((frames, segments_total, 4, 3), dtype=torch.float32).to(device)
    samples = torch.full((segments_total,), 4, dtype=torch.int32).to(device)
    segments = torch.full((circuits,), segments_per, dtype=torch.int64).to(device)
    next0 = torch.arange(segments_total, dtype=torch.int64).reshape(1, -1)
    next_inds = next0.expand(frames, -1).contiguous().to(device)
    centers = torch.zeros((frames, circuits, 3), dtype=torch.float32).to(device)
    basis_u = torch.zeros((frames, circuits, 3), dtype=torch.float32).to(device)
    basis_v = torch.zeros((frames, circuits, 3), dtype=torch.float32).to(device)
    basis_u[..., 0] = 1.0
    basis_v[..., 1] = 1.0
    return corners, samples, segments, next_inds, centers, basis_u, basis_v


def dummy_build(corners, samples, segments, next_inds, centers, basis_u, basis_v, inward_signs):
    del samples, next_inds, centers, basis_u, basis_v, inward_signs
    counts = segments.to(torch.int32) * 2
    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    edges = corners.new_zeros((corners.shape[0], int(offsets[-1]), 6))
    return edges, offsets


def run(mode, calls):
    device = torch.device("mps")
    assert torch.backends.mps.is_available()
    scene = SimpleNamespace()
    t0 = time.perf_counter()
    start_rss = rss_mb()
    print("torch", torch.__version__, "mode", mode, "calls", calls, flush=True)
    print("start_rss_mb", round(start_rss, 1), "mps_mb", tuple(round(x, 1) for x in mps_mb()), flush=True)

    for i in range(1, calls + 1):
        args = make_args(i, device)
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

        # Force completion, then remove all ordinary tensor/cache ownership.
        torch.mps.synchronize()
        del out, args
        if i % 10 == 0:
            gc.collect()
            torch.mps.empty_cache()
        if i == 1 or i % 25 == 0 or i == calls:
            cur, drv = mps_mb()
            print(
                "STEP",
                i,
                "rss_mb",
                round(rss_mb(), 1),
                "delta_rss_mb",
                round(rss_mb() - start_rss, 1),
                "mps_cur_mb",
                round(cur, 1),
                "mps_drv_mb",
                round(drv, 1),
                "elapsed_s",
                round(time.perf_counter() - t0, 1),
                flush=True,
            )

    # Exercise the exact op where the real suite eventually aborts.
    lo = torch.zeros((8, 16, 3), device=device)
    hi = torch.ones((8, 16, 3), device=device)
    probe = _aabb_corners(lo, hi)
    torch.mps.synchronize()
    print("PROBE", tuple(probe.shape), "final_rss_mb", round(rss_mb(), 1), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("control", "aabb", "cache"), required=True)
    p.add_argument("--calls", type=int, default=300)
    ns = p.parse_args()
    run(ns.mode, ns.calls)
