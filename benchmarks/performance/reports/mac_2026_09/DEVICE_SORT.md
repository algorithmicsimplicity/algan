# The compaction's sorts, off torch: a run sort that ships, and a radix sort that does not

`SHARED_QUEUE.md` §4 left the Apple GPU's remaining cost as one sentence: the
raster and compaction stages are torch-op bound, and MPSGraph's sort and unique
are only ~2.5x faster than this box's three CPU cores. It named two candidate
fixes -- "a Quadrants radix sort or fusing the compaction's torch stages into
kernels" -- and did not start either.

This is both of them, measured. The short version:

* **Fusing the sort into a per-pixel-run kernel is worth 21-53x on the sort and
  5-24% of a whole warm render, on both GPUs**, because a fragment stream that
  is already grouped by pixel does not need a global sort at all. That kernel
  was already in the tree; what kept it off Metal was a stale predicate, and
  what kept it off by default was an inconclusive earlier measurement. It is
  now the default everywhere.
* **The radix sort is worth 1.3-2.1x on Metal in isolation, is a 3.5-4.5x LOSS
  on CUDA, and is a whole-render loss even on Metal once the run sort has taken
  the big sort away.** It ships **off**, behind `ALGAN_DEVICE_RADIX_SORT`, with
  the numbers below.

Both halves of `SHARED_QUEUE.md`'s question therefore have an answer, and they
are not the same answer.

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

## 2b. And then the whole render, ABBA on one box

An isolated stage is not a render. Both candidates were re-measured end to end,
in the order shown, on one machine per column, warm runs only (each arm is a
fresh process, so its first render is cold and is not counted).

**Metal** -- `_mac_postfix_profile.py --child mps --coarse --runs 3`, run
[34304926999](https://github.com/algorithmicsimplicity/algan/actions/runs/34304926999),
`nn_scene_UHD` (18 frames at 3840x2160, shadows off, a 1720 MiB arena, 18
chunks in 2 batches). Times are whole-render wall; the three columns after it
are steady-state **self** seconds over the 17 steady chunks:

| order | arm | wall (warm) | `compact_sheets` | `prepare_sparse_raster_coverage` | `kernel.argsort_pairs` |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | base | 58.0, 55.0 | 13.3, 13.5 | 9.6, 8.8 | -- |
| 2 | **run sort** | **42.8, 44.7** | 9.1, 9.6 | 6.5, 7.3 | -- |
| 3 | run sort + radix | 48.4, 46.7 | 9.9, 9.4 | 7.0, 7.6 | 0.19 s / 68 calls |
| 4 | base | 57.0, 58.6 | 15.7, 14.8 | 9.2, 10.5 | -- |

* **The run sort is 57.2 s -> 43.7 s, 23.5%** of a warm UHD render. The
  bracketing base arms differ by 1.3 s, so the drift this VM contributes is
  about a tenth of the effect. `compact_sheets` gives up 4.9 s and
  `prepare_sparse_raster_coverage` 2.6 s of that directly; the rest is
  downstream, because on Metal a stage's self time includes waiting out the
  GPU work queued ahead of its readbacks.
* **The radix sort on top costs 3.8 s, +8.7%**, and every run of the run-sort
  arm was faster than every run of this one. The sorts themselves are not what
  it costs -- `kernel.argsort_pairs` is 0.19 s over 68 calls, 2.8 ms each --
  so what it costs is the torch work around them: two extra whole-stream
  allocations per call at the discovery peak, and a key built over the whole
  stream where the torch arm builds it over a gathered one.

**CUDA** -- Kaggle T4, notebook `algan-t4-runsort`,
`nn_warm_experiment.py --runs 3`, same scene, same ABBA:

| order | arm | wall (warm) |
| --- | --- | ---: |
| 1 | base | 8.4, 8.5 |
| 2 | **run sort** | **8.0, 7.8** |
| 3 | **run sort** | **7.9, 7.8** |
| 4 | base | 8.2, 8.1 |

**8.30 s -> 7.88 s, 5.1%**, and again with no overlap: every "on" run (7.8-8.0)
beat every "off" run (8.1-8.5). Smaller than Metal's share, exactly as the
isolated numbers predict -- torch's CUDA sort was only 15.8 ms of an 8.3 s
render to begin with -- but repeatable, and in the same direction. That is what
the earlier "has not yet translated into a repeatable whole-render speedup"
note was waiting for.

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
measurement "had not translated into a repeatable whole-render speedup"). §2b
is that translation, on two GPUs, so it is on by default now. Nothing about the
output moves: the comparator's last key is the original index, so the
permutation is the stable one the global sort produced, bit for bit.

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

Those sites are what the §2b radix arm was measuring, and it lost. So the
module ships **off** behind `ALGAN_DEVICE_RADIX_SORT`, which is also how both
columns above were taken. It is kept rather than deleted for three reasons, in
order of weight:

1. **The verdict is about these call sites, not the primitive.** The sorts are
   2.8 ms each; the loss is the torch work the wrapper still does around them.
   A site that hands the kernel a key it already has, and takes an int32 order
   back without widening it, has not been priced.
2. **It is the only exact wide-key sort on MPS.** §5's gather ceiling makes
   torch's own `_lexsort` inexact past 2**24 there; this one gathers inside a
   kernel.
3. It is the measurement instrument for the next candidate. Deleting it means
   rebuilding it to ask the next question.

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

That canonicalization shipped once doing **half** of its job, and how it was
caught is the point. `value != value` is the obvious way to ask "is this NaN",
and Metal's fast-math folds it to `false` -- so the NaN branch was dead while
the signed-zero branch worked. The probe's dedicated arm disagreed with torch
on *every one* of 2.9M positions (one NaN group ordering below `-inf` instead
of last shifts every group after it), which is a far louder signal than the
handful of positions a subtle tie bug moves. Both tests read the bits now;
`sheet_sort_taichi._after` inspects NaN bits for exactly this reason and says
so.

## 6. What was verified, and one failure that is not ours

At the shipped defaults on the Mac runner, every probe arm reproduces
`torch.argsort` exactly (`declined arms: 0`, so none of them silently
abstained), the MPS smoke render is correct, and the local suites are green
(fast 559 passed with its pixel-compared render; `tests/unit_tests` 3740
passed, 178 skipped).

The Mac's own `tests/unit_tests` run reports 37 failures, and **none of them
is this work**: 33 are `latex` missing (the round was dispatched with
`latex: false`), 3 are `test_taichi_runtime_config`'s pressure-reset
assertions, which master deliberately declines on the Metal arch
(`SHARED_QUEUE.md` §3), and the last is
`test_glossy_prefilter::test_prefiltered_reflection_is_substantially_wider`.
That one was bisected on the runner rather than assumed: it fails identically
with the run sort off, with the rank-groups kernel off, and with **both** off
-- which is master's behaviour on that box -- so it is pre-existing there and
wants its own round.
