"""Where the counting sort stops paying: fragment order by bin density.

``raster_pipeline._binned_fragment_order`` orders a chunk's fragment stream with
a counting sort over the chunk's pixel ordinals -- ``frames * width * height``
bins -- in place of a radix sort of the fragments. Its bins cost about 8 B each
live whatever the fragments are, so a sparse window, a few fragments over many
frames of pixels, pays for bins that hold nothing.
``_BINNED_MAX_BINS_PER_FRAGMENT`` is the density past which it hands the window
to the radix sort instead. This times both arms of
``_tie_grouped_fragment_order``'s sort at one fragment count across bin
densities, with the peak torch allocation each takes, so that limit can be
measured again on any CUDA card::

    <venv-python> benchmarks/_binned_sort_density.py
    <venv-python> benchmarks/_binned_sort_density.py --fragments 12000000 --ratios 8,16,24

The stream is shaped like a real one: about two fragments to a covered pixel,
the covered pixels spread over the window. CUDA only, as the counting sort is.
"""

from __future__ import annotations

import argparse
import time

import torch

from algan.rendering.raytracing import device_sort
from algan.rendering.raytracing import raster_pipeline as rp
from algan.rendering.taichi_runtime import init_taichi


def _stream(n, bins, device, seed=0):
    g = torch.Generator().manual_seed(seed)
    covered = torch.randint(0, bins, (max(1, n // 2),), generator=g)
    pix = covered[torch.randint(0, covered.numel(), (n,), generator=g)]
    depth = torch.rand(n, generator=g) * 20.0 + 0.5
    key = (pix << 32) | (depth.view(torch.int32).to(torch.int64) & 0xFFFFFFFF)
    ref = torch.randint(0, 4096, (n,), generator=g, dtype=torch.int32)
    return key.to(device), ref.to(device)


def _binned(key, ref, bins):
    return rp._binned_fragment_order(
        key,
        ref,
        0,
        bins,
        exact_depth=True,
        min_fragments=0,
        max_bins_per_fragment=float("inf"),
    )


def _radix(key):
    order = device_sort.stable_argsort(key)
    if order is None:
        return torch.argsort(key, stable=True)
    return order.to(torch.int64)


def _measure(reps, fn, *args):
    fn(*args)  # compiles
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    start = time.perf_counter()
    for _ in range(reps):
        out = fn(*args)
    torch.cuda.synchronize()
    seconds = (time.perf_counter() - start) / reps
    return seconds, torch.cuda.max_memory_allocated() - base, out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--fragments", type=int, default=2_000_000)
    parser.add_argument("--ratios", default="1,2,4,8,16,32,64,128")
    parser.add_argument("--reps", type=int, default=10)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("the counting sort runs on CUDA only")
    init_taichi()
    device = torch.device("cuda")
    n = args.fragments
    print(f"{torch.cuda.get_device_name()}, {n:,} fragments")
    print(
        f"{'bins/frag':>9} {'bins':>13}  {'binned ms':>9} {'MB':>7}"
        f"  {'radix ms':>9} {'MB':>7}"
    )
    for ratio in (float(r) for r in args.ratios.split(",")):
        bins = int(ratio * n)
        key, ref = _stream(n, bins, device)
        b_s, b_peak, b_order = _measure(args.reps, _binned, key, ref, bins)
        r_s, r_peak, r_order = _measure(args.reps, _radix, key)
        if b_order is None:
            raise SystemExit("the counting sort declined; is the arch CUDA?")
        # Both put the keys in order; they part only on exact-depth ties,
        # which the tie grouping after either arm re-sorts by layer anyway.
        assert torch.equal(key[b_order], key[r_order]), "the two arms disagree"
        print(
            f"{ratio:9g} {bins:13,}  {b_s * 1e3:9.2f} {b_peak / 2**20:7.0f}"
            f"  {r_s * 1e3:9.2f} {r_peak / 2**20:7.0f}"
        )
        del key, ref, b_order, r_order
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
