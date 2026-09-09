# The compaction's sorts, off torch: a radix sort that is Metal-only, and a run sort that is not

`SHARED_QUEUE.md` §4 left the Apple GPU's remaining cost as one sentence: the
raster and compaction stages are torch-op bound, and MPSGraph's sort and unique
are only ~2.5x faster than this box's three CPU cores. It named two candidate
fixes -- "a Quadrants radix sort or fusing the compaction's torch stages into
kernels" -- and did not start either.

This is both of them, measured. The short version:

* **The radix sort is worth 1.3-2.1x on Metal and is a 3.5-4.5x LOSS on CUDA**,
  because torch's CUDA sort is already CUB's radix sort and Quadrants' is not
  as good. So it ships gated to MPS-friendly mode.
* **Fusing the sort into a per-pixel-run kernel is worth 21-53x on every
  backend measured**, because a fragment stream that is already grouped by
  pixel does not need a global sort at all. That kernel was already in the
  tree; what kept it off Metal was a stale predicate.

## 1. Why a global sort was reachable at all

`taichi_launch_is_local` decides whether a kernel can be launched against a
torch tensor without the compiler staging a copy of every argument through the
host, and every "kernel or torch?" gate in the renderer reads it. It said
**no** for an MPS tensor on a Metal arch, and it was right when it was written:
`Device::import_memory` is implemented for `CpuDevice` and `CudaDevice` and
nothing else, so stock Taichi copies each argument out and back around a Metal
launch -- 53x the cost of the same kernel on the CPU arch
(`DESIGN_mps_support.md` §1.3).

It stopped being right when `mps_zero_copy` landed. The Metal RHI carries its
own non-virtual `MetalDevice::import_mtl_buffer`, the wrapper reaches it in
front of every launch, and on the build Algan installs an MPS argument is
*bound* rather than copied (`DESIGN_mps_zero_copy.md` §1). Nothing updated the
predicate, so every kernel gated on it stayed unreachable on an Apple GPU.

It now asks whether the conversion is installed rather than naming CUDA. A
build without the adoption still answers False -- that is the condition, not
the platform, and getting it from the platform would call a 53x staging cost
free on any Mac running an unpatched compiler.

**What that actually switched on is small, and deliberately so.** Most call
sites have a second gate of their own and it is still closed: `refit_bvh`'s
pack kernel is behind `ALGAN_REFIT_PACK_KERNEL` (off), the run sorts behind
`ALGAN_SHEET_PIXEL_SORT` (off), the group metadata behind
`ALGAN_SHEET_METADATA_KERNEL` (off), and the **PN level searches do not become
reachable at all** -- `pn_criterion_kernel_active()` asks whether projection
ran on CUDA or the arch is the CPU, and neither is true on Metal, so
README.md's Tier-1 item 4 is untouched by this. The one path the predicate
opens on its own is `_sheet_rank_groups`, whose CUDA-by-name gate is widened to
match (§5), and it is what makes the run sort *available* to be measured.

## 2. The two candidates, timed

`benchmarks/_device_sort_probe.py`, one synthetic UHD-shaped fragment stream
(pixel-ordered, runs of 1-8, pixel ordinals past 2**24 on purpose), warm,
median of 3. **Every arm's permutation was compared against
`torch.argsort(stable=True)` computed on the host, and every one matched** --
including int64 keys and float32 depths, on both backends.

**Metal** -- the Mac harness, virtualized M1, run
[34304209461](https://github.com/algorithmicsimplicity/algan/actions/runs/34304209461),
2.9M fragments:

| arm | Quadrants radix | torch (MPSGraph) | ratio |
| --- | ---: | ---: | ---: |
| `argsort` int32 | 20.1 ms | 32.5 ms | **1.62x** |
| `argsort` int64 | 45.8 ms | 80.3 ms | **1.75x** |
| `argsort` float32 | 24.4 ms | 32.2 ms | **1.32x** |
| `_lexsort(pixel, group, depth)` | 104 ms | 215 ms | **2.06x** |
| `pixel_group_order` kernel | **4.1 ms** | 215 ms | **53x** |
| `unique_consecutive(pixel)` | -- | 9.7 ms | |

**CUDA** -- Kaggle T4, notebook `algan-t4-devsort`, same probe, same 2.9M:

| arm | Quadrants radix | torch (CUB) | ratio |
| --- | ---: | ---: | ---: |
| `argsort` int32 | 10.8 ms | 2.4 ms | 0.22x |
| `argsort` int64 | 21.1 ms | 5.0 ms | 0.24x |
| `argsort` float32 | 10.7 ms | 3.0 ms | 0.28x |
| `_lexsort(pixel, group, depth)` | 55.7 ms | 15.8 ms | 0.28x |
| `pixel_group_order` kernel | **0.75 ms** | 15.8 ms | **21x** |
| `unique_consecutive(pixel)` | -- | 0.5 ms | |

Read the two tables together and the answer is not "GPU sorts are faster than
torch sorts". It is that **torch's sort is excellent on CUDA and poor on MPS**,
and that **neither of them should be running at all** here.

## 3. Why the run sort wins by two orders of magnitude

`compact_sheets`'s P1 order is `(pixel, group, depth)` over a stream the raster
stage already emitted in pixel order, with a per-pixel CSR (`run_offsets`)
beside it. A global lexicographic sort re-derives the pixel ordering it was
handed: three stable argsorts and two `index_select`s over the whole stream,
O(n log n) with a large constant and five full passes over memory.

`sheet_sort_taichi.pixel_group_order` instead gives each pixel run to one
thread, which orders it in place -- insertion sort under 17 elements, in-place
heapsort above -- and a UHD chunk's runs are a handful of fragments each. The
work is O(n) with a tiny constant, the only allocation is the output
permutation, and the comparator carries the original index as its last key so
the result is the *stable* permutation, bit for bit, rather than merely a
sorted one.

That is the "fusing the compaction's torch stages into kernels" half of
`SHARED_QUEUE.md`'s question, and it turns out to have been written already
(`ALGAN_SHEET_PIXEL_SORT`, added and left off by default because its earlier
measurement "had not translated into a repeatable whole-render speedup").

## 4. Where the radix sort is still the answer

Not every sort in the pipeline has a pixel run to exploit:

* `raster_pipeline._exact_fragment_order` -- two global sorts over the raw
  fragment stream, before any grouping exists;
* `mps_compat.band_class_groups` -- a two-pass sort by `(band, class)`;
* `_lexsort` itself, wherever a caller has no CSR.

For those, `device_sort` routes to `qd.algorithms.sort`, wrapped by one kernel
(`radix_sort_taichi.argsort_pairs`) whose seed loop *also* performs the gather
that composes an LSD multi-key order. That matters as much as the sort: a
three-key order in torch is one `argsort` and one `index_select` per key, and
in the kernel form the `index_select`s do not exist. In
`_exact_fragment_order` it also retires the `gather_packed_key` over the packed
64-bit key -- the depth key is now derived *before* the layer permutation
rather than after, which is the same value elementwise and is what lets the
second sort gather through that permutation itself.

It is on by default only in MPS-friendly mode -- the mode whose whole premise
is that this backend's torch ops are the slow ones, and the only place the
measurement supports it. `ALGAN_DEVICE_RADIX_SORT` forces either arm on any
device, which is how the CUDA column above was taken.

## 5. Two things the round found that were not about speed

**An MPS gather makes `_lexsort` inexact above 2**24.** Each pass of the torch
arm carries its key through an `index_select`, and on MPS an integer gather
rounds through a float32 significand
(`mps_compat._MPS_EXACT_INT_BITS`, measured by `_mps_torch_op_probe.py`). So
for a key past 2**24 -- a multi-frame chunk's pixel ordinal at 4K reaches it --
the device's own `_lexsort` returns a permutation that ties rows which are not
tied. This surfaced as two `test_sheet_pixel_sort` failures on the Mac that
were the *reference* being wrong, not the kernel: the test compared a kernel
against `_lexsort` on the same device, with group keys at 2**40 chosen to
exercise exactly the width that rounds. The reference is computed on the host
now. Both the run-sort kernel and the radix sort gather inside a kernel, where
there is no such ceiling, so both are exact where the torch arm is not.

**MPS orders -0.0 before +0.0; the CPU calls them equal.** A comparison sort
ties signed zeros and a bit order does not, and torch's MPS backend behaves
like the bit order. The radix seed loop canonicalizes -0.0 (and NaN, which
torch orders last whatever its sign) so its permutation is the CPU's, which is
what the renderer's baselines were taken on. A depth is a distance along a ray
and can be neither, so this buys agreement on inputs the renderer does not
produce.
