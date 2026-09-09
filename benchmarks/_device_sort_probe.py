"""Does a device radix sort beat torch's on this box, and does it agree with it?

The compaction's cost on an Apple GPU is its sorts and the gathers that compose
them (``benchmarks/performance/reports/mac_2026_09/SHARED_QUEUE.md`` §4), so
this asks the two questions that decide whether
:mod:`algan.rendering.raytracing.device_sort` should be on:

1. **Does it agree with torch?** Every arm is compared against
   ``torch.argsort(..., stable=True)`` computed on the host over the same
   values -- not merely "is it sorted", because what the renderer needs is the
   *stable* permutation and a sort that ties differently reorders fragments
   inside a pixel.
2. **Is it faster?** Timed against the torch call it would replace, at the
   fragment counts a UHD chunk actually carries, warm (the first launch pays
   the kernel's whole pipeline build).

It also times the two alternatives to a global sort that already exist in the
tree, so the round ranks them rather than measuring one in isolation:
``sheet_sort_taichi.pixel_group_order`` (sort each pixel's run in place, no
global order at all) and torch's ``unique_consecutive``, which is what the
compaction groups with.

Run it on the Mac harness with ``ALGAN_RENDER_DEVICE=mps``, or on a CUDA box to
re-run the same comparison against CUB. It needs no scene and no render: it
brings the compiler up on the render device, builds one synthetic fragment
stream and reports a line per measurement, so a job that is reclaimed half way
still leaves its readings behind.
"""

from __future__ import annotations

import argparse
import time

import torch

import algan  # noqa: F401  -- installs the MPS zero-copy launch wrapper
from algan.rendering.raytracing import device_sort
from algan.settings._startup import render_device

#: One UHD frame is 8.3M pixels and a warm chunk carried ~2.9M fragments.
DEFAULT_SIZES = (262144, 2900000)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _sync_all(device):
    """Drain both queues: a kernel's work is not torch's to wait for."""
    from algan.taichi_compat import ti

    with_program = True
    try:
        ti.sync()
    except Exception:
        with_program = False
    _sync(device)
    return with_program


def _time(label, device, fn, runs=3):
    """Median of ``runs`` timed calls after one warm-up, printed as it lands."""
    fn()
    _sync_all(device)
    samples = []
    for _ in range(runs):
        started = time.perf_counter()
        fn()
        _sync_all(device)
        samples.append(time.perf_counter() - started)
    samples.sort()
    median = samples[len(samples) // 2]
    print(
        f"  {label:<44s} {median * 1e3:9.2f} ms   "
        f"(min {min(samples) * 1e3:.2f}, max {max(samples) * 1e3:.2f})",
        flush=True,
    )
    return median


def _fragment_stream(n, device, seed=20260909):
    """A pixel-ordered fragment stream shaped like a UHD chunk's."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    # Run lengths 1..8, so the pixel column is nondecreasing with ties, which
    # is what the compaction's own key column looks like.
    runs = torch.randint(1, 9, (n,), generator=g)
    pixel = torch.repeat_interleave(torch.arange(n), runs)[:n].contiguous()
    # Two frames of a 4K grid: the pixel ordinal passes 2**24 on purpose.
    pixel = pixel * 3 + 4 * 3840 * 2160
    group = torch.randint(0, 512, (n,), generator=g, dtype=torch.int64)
    depth = (torch.rand(n, generator=g) * 40.0 + 0.5).to(torch.float32)
    layer = torch.randint(-4096, 4096, (n,), generator=g, dtype=torch.int32)
    offsets = torch.zeros(n + 1, dtype=torch.int64)
    counts = torch.bincount(pixel - pixel.min(), minlength=1)
    counts = counts[counts > 0]
    offsets[1 : counts.numel() + 1] = torch.cumsum(counts, 0)
    num_runs = int(counts.numel())
    return {
        "pixel": pixel.to(device),
        "group": group.to(device),
        "depth": depth.to(device),
        "layer": layer.to(device),
        "offsets": offsets[: num_runs + 1].to(device),
        "num_runs": num_runs,
        "host": {"pixel": pixel, "group": group, "depth": depth, "layer": layer},
    }


def _reference_lexsort(*keys):
    """``sheets._lexsort`` on the host, the order every arm is judged against."""
    order = None
    for key in reversed(keys):
        k = key if order is None else key.index_select(0, order)
        o = torch.argsort(k, stable=True)
        order = o if order is None else order.index_select(0, o)
    return order


def _agrees(label, got, expected):
    if got is None:
        print(f"  {label:<44s} DECLINED (no kernel arm)", flush=True)
        return False
    same = torch.equal(got.to("cpu").to(torch.int64), expected)
    print(f"  {label:<44s} {'MATCHES torch' if same else 'DISAGREES'}", flush=True)
    if not same:
        differ = (got.to("cpu").to(torch.int64) != expected).nonzero().flatten()
        print(
            f"      {differ.numel()} of {expected.numel()} positions differ", flush=True
        )
    return same


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="*", default=list(DEFAULT_SIZES))
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    from algan.rendering import mps_zero_copy
    from algan.rendering.taichi_runtime import (
        _live_arch,
        ensure_taichi_for_render,
        render_job_holding_the_arch,
        taichi_launch_is_local,
    )

    device = render_device()
    print(f"render device: {device}", flush=True)
    print(f"zero copy available: {mps_zero_copy.zero_copy_available()}", flush=True)
    print(f"zero copy installed: {mps_zero_copy.installed()}", flush=True)

    with render_job_holding_the_arch():
        ensure_taichi_for_render()
        print(f"live arch: {_live_arch()}", flush=True)
        print(f"launch is local: {taichi_launch_is_local(device)}", flush=True)
        print(f"radix sort enabled: {device_sort.radix_sort_enabled()}", flush=True)
        failures = _run(args, device)
        print(f"zero copy stats: {mps_zero_copy.STATS}", flush=True)
    if failures:
        raise SystemExit(f"{failures} arm(s) disagreed with torch")


def _run(args, device):
    return sum(_measure(n, device, args.runs) for n in args.sizes)


def _measure(n, device, runs):
    """Every arm at one fragment count. Returns how many disagreed with torch."""
    from algan.rendering.raytracing.sheet_sort_taichi import pixel_group_order

    print(f"\n=== {n} fragments on {device} ===", flush=True)
    data = _fragment_stream(n, device)
    pixel, group, depth = data["pixel"], data["group"], data["depth"]
    layer, offsets = data["layer"], data["offsets"]
    num_runs = data["num_runs"]
    host = data["host"]
    failures = 0

    available = device_sort.radix_sort_available(pixel)
    print(f"  available (int64 key): {available}", flush=True)

    # -- one key, three dtypes -------------------------------------------
    for label, key, host_key in (
        ("int32", layer, host["layer"]),
        ("int64", pixel, host["pixel"]),
        ("float32", depth, host["depth"]),
    ):
        expected = torch.argsort(host_key, stable=True)
        got = device_sort.stable_argsort(key)
        failures += not _agrees(f"argsort {label}: correctness", got, expected)
        if got is not None:
            _time(
                f"argsort {label}: radix kernel",
                device,
                lambda k=key: device_sort.stable_argsort(k),
                runs,
            )
        _time(
            f"argsort {label}: torch.argsort",
            device,
            lambda k=key: torch.argsort(k, stable=True),
            runs,
        )

    # -- the compaction's own three-key order ----------------------------
    expected = _reference_lexsort(host["pixel"], host["group"], host["depth"])
    got = device_sort.stable_lexsort(pixel, group, depth)
    failures += not _agrees("lexsort(pixel, group, depth)", got, expected)
    if got is not None:
        _time(
            "lexsort: radix kernels",
            device,
            lambda: device_sort.stable_lexsort(pixel, group, depth),
            runs,
        )
    _time(
        "lexsort: torch argsort + index_select",
        device,
        lambda: _reference_lexsort(pixel, group, depth),
        runs,
    )

    # -- the alternative: sort each pixel run in place --------------------
    def run_sort():
        order = torch.empty_like(pixel)
        pixel_group_order(offsets, group, depth, order, num_runs)
        return order

    try:
        order = run_sort()
        _sync_all(device)
        failures += not _agrees("pixel_group_order kernel", order, expected)
        _time("pixel_group_order kernel", device, run_sort, runs)
    except Exception as exc:  # a kernel Metal refuses is a finding, not a crash
        print(
            f"  pixel_group_order kernel: FAILED {type(exc).__name__}: {exc}",
            flush=True,
        )
        failures += 1

    # -- what the compaction groups with ---------------------------------
    _time(
        "unique_consecutive(pixel)",
        device,
        lambda: torch.unique_consecutive(pixel, return_inverse=True),
        runs,
    )
    return failures


if __name__ == "__main__":
    main()
