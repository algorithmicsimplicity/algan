"""Cost of compact_sheets' individual passes, at a 4K frame's real shapes.

``sheets.compact_sheets`` is one straight-line function, so a profiler can only
report it whole. This replays its passes standalone on arrays of the sizes a
3840x2160 sphere-and-friends frame actually produces (measured: 3.66 M
fragments, 3.29 M sheets), which is what a "should this be a Taichi kernel"
decision needs: how much of the stage is the two bit-lane loops (8 passes over
[n] each, and the obvious fusion candidates), how much is the sort primitives
(cuB-backed, nothing to win), and how much is the segmented reductions.

Timings are per call, averaged, CUDA-synchronised.

    <venv-python> benchmarks/_sheet_compact_breakdown.py [n] [nb] [reps]
"""

import os
import sys
import time

os.environ.setdefault("ALGAN_USE_DAEMON", "0")

import torch  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3_661_824
NB = int(sys.argv[2]) if len(sys.argv) > 2 else 3_290_404
REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 5
AA_NUM_SAMPLES = 8
AA_MASK_ALL = (1 << AA_NUM_SAMPLES) - 1

dev = torch.device("cuda")
g = torch.Generator(device=dev).manual_seed(0)
band_id = torch.sort(torch.randint(0, NB, (N,), generator=g, device=dev))[0]
msk = torch.randint(0, 1 << 20, (N,), generator=g, device=dev, dtype=torch.int32)
cov = torch.rand(N, generator=g, device=dev)
order = torch.randperm(N, generator=g, device=dev)
band_start = torch.zeros(N, dtype=torch.bool, device=dev)
band_start[::3] = True
positions = torch.arange(N, dtype=torch.int64, device=dev)
results = []


def bench(label, fn):
    fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(REPS):
        fn()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / REPS
    results.append((label, dt))


def rank_scan():
    band_first = torch.cummax(
        torch.where(band_start, positions, torch.zeros_like(positions)), 0
    )[0]
    bits = (msk.index_select(0, order) & AA_MASK_ALL).to(torch.int32)
    rank = torch.zeros(N, dtype=torch.int32, device=dev)
    for b in range(AA_NUM_SAMPLES):
        lane = (bits >> b) & 1
        excl = torch.cumsum(lane, 0, dtype=torch.int32) - lane
        prior = excl - excl.index_select(0, band_first)
        rank = torch.maximum(
            rank, torch.where(lane > 0, prior, torch.zeros_like(prior))
        )
    return rank


def union_loop():
    bits = (msk & AA_MASK_ALL).to(torch.int64)
    union = torch.zeros(NB, dtype=torch.int64, device=dev)
    fused = torch.zeros(NB, dtype=torch.bool, device=dev)
    lane = torch.zeros(NB, dtype=torch.int64, device=dev)
    for b in range(AA_NUM_SAMPLES):
        lane.zero_()
        lane.scatter_add_(0, band_id, (bits >> b) & 1)
        union |= (lane > 0).to(torch.int64) << b
        fused |= lane > 1
    return union


def popcount():
    bits = torch.randint(0, 256, (NB,), generator=g, device=dev, dtype=torch.int64)
    pop = torch.zeros(NB, dtype=torch.int32, device=dev)
    for b in range(AA_NUM_SAMPLES):
        pop += ((bits >> b) & 1).to(torch.int32)
    return pop


def seg_reductions():
    a = torch.zeros(NB, dtype=torch.float64, device=dev)
    a.scatter_add_(0, band_id, cov.to(torch.float64))
    first = torch.full((NB,), N, dtype=torch.int64, device=dev)
    first.scatter_reduce_(0, band_id, positions, reduce="amin", include_self=True)
    cmax = torch.zeros(NB, dtype=torch.float32, device=dev)
    cmax.scatter_reduce_(0, band_id, cov, reduce="amax", include_self=True)
    nfrag = torch.zeros(NB, dtype=torch.int64, device=dev)
    nfrag.scatter_add_(0, band_id, torch.ones_like(band_id))
    return a, first, cmax, nfrag


def lexsort():
    o = None
    for key in (cov, band_id, positions):
        k = key if o is None else key.index_select(0, o)
        s = torch.argsort(k, stable=True)
        o = s if o is None else o.index_select(0, s)
    return o


def uniques():
    cid = band_id * 16 + (positions % 16)
    return torch.unique(cid, sorted=True, return_inverse=True)


def six_gathers():
    return [
        msk.index_select(0, order),
        cov.index_select(0, order),
        positions.index_select(0, order),
        band_id.index_select(0, order),
        msk.index_select(0, order),
        cov.index_select(0, order),
    ]


bench("bit-lane rank scan (8 x cumsum/gather over [n])", rank_scan)
bench("bit-lane union + fusion detector (8 x scatter_add, x2/frame)", union_loop)
bench("bit-lane popcount over [sheets] (x2 per frame)", popcount)
bench("segmented reductions (area/first/max/count)", seg_reductions)
bench("6 x index_select on one permutation", six_gathers)
bench("_lexsort (3 stable argsorts)", lexsort)
bench("torch.unique(sorted, return_inverse)", uniques)

total = sum(d for _, d in results)
print(f"\n=== compact_sheets pass costs, n={N:,} sheets={NB:,}, mean of {REPS} ===")
for label, dt in sorted(results, key=lambda r: -r[1]):
    print(f"  {dt * 1000:7.1f} ms  {dt / total * 100:5.1f}%  {label}")
print(f"  {total * 1000:7.1f} ms  100.0%  (sum of the passes benchmarked)")
