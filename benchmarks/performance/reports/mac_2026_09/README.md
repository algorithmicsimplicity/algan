# Mac GPU round, 2026-09: what the Metal render actually costs

> **The single self-contained record of this round is
> [`FINDINGS.md`](FINDINGS.md)** -- what landed, what was measured, the
> hypotheses that were refuted and by what, and the failure that is still
> open. This file is the earlier working report and its section 1 predates
> the arena cap; where the two disagree, FINDINGS.md is current.

Workload: `benchmarks/performance/nn_scene_UHD.py`, unchanged — 18 frames at
3840×2160, `shadows=False`, `libx264 -preset ultrafast`. Box: the Mac harness
(`agent_guidance/gpu_harnesses.md`), GitHub's Apple-silicon runner, a
virtualized M1 with a **real** Metal GPU, 3 CPUs, 7 GB unified,
`recommendedMaxWorkingSetSize = 4.67 GB`. Compiler: patched Quadrants 1.3.1
(`quadrants_build.yaml` run `33850787142`), zero-copy Metal ndarrays live.

> **Read the harness's own caveat first.** This box's *compute* numbers are
> sound; its **per-launch and per-copy numbers are not** — a synchronized
> dispatch measures 432 µs here against 2.0 µs on its own CPU. Nothing on this
> page ranks a many-small-kernel stage from a Mac wall time, and where a
> candidate is launch-bound that is said so explicitly.

## 0. The headline: this workload did not run on Metal at all

The first attempt died before rendering a frame:

```
algan.rendering.primitives.primitive.OutOfRenderMemory:
Insufficient memory to ray trace a single frame.
```

on a machine with 4.67 GB free. The cause is one line:
`get_num_available_bytes` clamped its **MPS** branch to `min(free, 1 GiB)`
(`algan/utils/memory_utils.py`). The render arena is
`rendering_memory_fraction` (0.4) of that figure, so **every Metal render on
every Mac sized its arena at 410 MB** — a 128 GB Mac Studio and a 8 GB Air got
the same arena — and 410 MB does not hold one 4K frame. The clamp has no
recorded rationale; the CUDA branch beside it reports real free bytes, and
`_render_device_pool_bytes` was *already* sizing the out-of-arena budgets from
`recommendedMaxWorkingSetSize` on the same device. It was an unexplained
ceiling, not a safety margin.

Sizing the branch from the device (see "Landed", below) is what makes the rest
of this page possible, and it is the largest Mac-side win in the round: it is
the difference between *cannot render 4K* and *renders 4K*.

**It also uncovered two latent defects**, because a frame window wider than one
frame per render chunk had never occurred on Metal before. Both are recorded in
§3; neither reproduces on the CPU at the same pool size and the same window, so
both are Metal-specific.

## 1. What the Apple GPU actually measures

**`nn_scene_UHD.py`, 18 frames at 3840×2160, shadows off.** With the fixes
below it renders to completion on Metal for the first time:

| | |
| --- | --- |
| cold (first render in the process) | **595.7 s** — 33 s a frame |
| of which Taichi kernel compile | ~66 s |
| emissions | 18, i.e. **one frame per render chunk** |
| warm, before the arena fix | 968–1523 s — 1.19× to 2.04× *slower* than cold |
| warm, after it | **568.5 s** — 1.9× *faster* than cold |

Warm numbers are from `benchmarks/_mps_warm_regression.py 2 UHD` (both renders
in one process, no profiler hooks; a cold pass measures 747–816 s under the
instrument rather than 595.7 s because it is a second scene build in the same
interpreter). Four jobs:

| job | cold | warm | warm arena | imports/chunk warm |
| --- | ---: | ---: | ---: | ---: |
| 34 | 747.0 s | 1522.8 s | 1212 MB | — |
| 36 | 810.4 s | 967.9 s | 1226 MB | — |
| 37 | 816.3 s | 1026.4 s | 1226 MB | 48 766 |
| 38, arena pinned | 1105.3 s [^stall] | **568.5 s** | **1898 MB** | **12 388** |

[^stall]: One chunk of job 38's cold pass took 455.6 s on its own against 25–42 s
for every other chunk. A runner stall, not a code path; it is not a cold
baseline.

**The wall-time ratio is noisy; the arena collapse is not.** Three unpinned jobs
put warm at 1.19×, 1.26× and 2.04× — so the 2.04× is an outlier, not the
headline — while the warm arena landed at 1212/1226/1226 MB every time, a
reproducible 35% drop against the cold 1898 MB.

That arena is the whole effect, and §1.1 is how it was established.

**`nn_scene`'s scene at PREVIEW (704×396), four renders in one process**
(`benchmarks/_mps_warm_regression.py`, no profiler hooks):

| render | wall | arena | batches | chunks |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 81.2 s | 1898 MB | 1 | 2 |
| 2 | **12.9 s** | 1828 MB | 1 | 2 |
| 3 | 12.7 s | 1827 MB | 1 | 2 |
| 4 | 11.1 s | 1819 MB | 1 | 2 |

Pool: `recommended_max` 4.67 GB; after a render, `driver_allocated` 3.16 GB and
`current_allocated` 1.85 GB.

**Warm is 6.3× faster, and the arena is stable** (4% drift over four renders),
so there is no general "the second render is slow" defect on this backend.

### 1.1 The warm UHD pass: a sizing defect, reached by discarding three wrong answers

The finding is one line of `get_num_available_bytes`. Getting there took four
jobs, and the three discarded explanations are worth recording because each was
plausible and each was killed by a measurement rather than by argument.

**Wrong answer 1: the preflight binary search.** Both passes log

    Prepared batch does not fit the render arena;
    binary-searching the largest fitting runtime.

and every rejected probe throws away a complete projection, merge and BVH build
(`_release_preflight_candidate` nulls every `_rt_*`), which on Metal are eager
CPU torch on three cores. A compelling story, and false: a per-chunk trace put
warm batch preparation at **8.4 s against the cold pass's 28.2 s**. Warm
preparation is the *faster* of the two. The cost is inside the chunks.

**Wrong answer 2: a gc storm from the pressure predicate.** `_gpu_memory_pressure`
judges from `driver_allocated_memory`, which after a render reads 4.56 G of a
4.67 G recommended max while live bytes are 0.00 G — permanently above the 0.8
threshold, so all nineteen `force_gc=False` reclaim sites should pay a full
`gc.collect()` and drop the import cache. Counting it killed it:
`release_torch_memory` accounts for **0.1–1.9 s a chunk** against chunks costing
25–79 s, and through the cold pass the cache clears rise 1 → 7 a chunk without
moving that pass's import count off 12 388.

**Wrong answer 3: a memory knee at the 0.8 threshold.** Cold chunks 11 and 12
cost 75.6 s and 78.9 s against a 25–37 s baseline, exactly as `driver_allocated`
crosses 3.74 G. Coincidence: chunks 13–18 fall straight back to 26–44 s while
still 7/7 pressured at 4.0–4.5 G, and the same two chunks spike in a job with a
different memory profile. It is a batch boundary.

**What it actually is.** The variable is *launch count*, and the arena sets it:

| | cold | warm | ratio |
| --- | ---: | ---: | ---: |
| zero-copy imports per chunk | 12 388 | 48 766 | 3.94× |
| cost per chunk (2–10) | 31.2 s | 55.2 s | 1.77× |
| arena | 1898 MB | 1226 MB | 0.65× |

36 378 extra imports against a 24.0 s per-chunk gap is **0.66 ms an import** —
this box's dispatch cost (432 µs synchronized). The warm render does no extra
work; it does the same work in four times as many launches.

And the arena shrinks because **`driver_allocated_memory` is a high-water mark on
Metal.** It does not come back down after `empty_cache`: within one pass it
climbs monotonically 2.91 → 4.49 G while live bytes hold flat at 1.91 G, through
seven pressured drains a chunk in the last third that never move it. Sizing from
it charges each render for the previous one's peak.

**The A/B.** Holding the second render to the first's free-byte figure
(`_mps_warm_regression.py … pin-arena`) took the warm pass to **568.5 s**, put
its import count back to **12 388 exactly**, and left warm 1.9× *faster* than
cold. `driver_allocated` reached 4.76 G doing it — past
`recommendedMaxWorkingSetSize`, with no failure — so on unified memory that
ceiling is advisory and the blocks behind the driver figure are reusable.

The fix measures `current_allocated_memory` after the drain. Its known gap is
that Taichi's allocations outside torch are not in that figure; the preflight
already searches the window down when a batch does not fit, and
`ALGAN_MPS_MEMORY_CAP` still imposes a ceiling by hand.

This supersedes the clear–drain–measure fix recorded earlier in this round,
which was necessary but not sufficient: the drain does return memory at a render
boundary, but the figure read after it does not reflect that.

**Still open.** The `max_surfaces_per_ray` truncation warning fires with wildly
varying counts across otherwise identical passes (0, 2, 62, 559 rays) and
appeared only in warm passes. Unexplained, and unrelated to the above as far as
these jobs show.

### Cost of measuring here

A macOS job is reclaimed well before `timeout_minutes` (72 min against 120,
57 min against 100) and a reclaimed job **publishes nothing**, so a command
must fit in roughly 40 minutes and must print per unit of work.
`agent_guidance/gpu_harnesses.md` now carries this; it cost four jobs an hour
each to learn.

## 2. The ranked list

Everything here was derived by reading the current tree, and by four
independent deep reads of the render loop, the sheet-route host path, the
Taichi kernels and the geometry-preparation path. Sizes are estimates unless a
row says otherwise. **The `t4_2026_09` numbers do not describe this scene any
more** — that round profiled it with `shadows=True` and 30 frames; today the
benchmark sets `shadows=False`, which deletes `raster_shadow_trace` (its #1
kernel, 15.3%) outright and shrinks `wavefront_shade` substantially.

### Tier 1 — Metal-specific, and structural

| # | candidate | mechanism | est. |
| --- | --- | --- | --- |
| 1 | **Size the Metal arena from the device** (landed) | 410 MB arena on every Mac → the device's real headroom | 4K becomes renderable at all; fewer batches everywhere below that |
| 2 | **Let the source-key index hit on Metal** (landed) | `ExternalMetalNdarray` poisoned the key, so every kernel taking an arena array paid the full Python frontend in *every process* | the index exists to remove ~12 s of a 12.5 s warm `save_frame`; on Metal it was doing nothing |
| 3 | **Fix multi-frame chunks on Metal** (§3) | two defects that only appear once a chunk holds more than one frame | unblocks 1; a chunk per frame pays every per-chunk fixed cost 18 times at UHD |
| 4 | **Extend GPU projection / merge / PN-criterion kernels to MPS** | `project_on_gpu_active()`, `merge_on_gpu_active()` and `pn_criterion_kernel_active()` all gate on **CUDA** (`raytracing/settings.py`), so on Metal the whole geometry preparation — projection, the PN level search, the merge — runs as eager CPU torch with no fused kernel anywhere, on 3 slow cores, while the GPU idles | large: preparation is 15% of a warm T4 render *with* GPU projection; on Metal it is the CPU path and the machine has 3 cores |
| 5 | **Take the torch↔Taichi fences once per region, not once per launch** | `mps_zero_copy.install_zero_copy_launch` takes `torch.mps.synchronize()` before and `ti.sync()` after **every** converted launch, which serializes the GPU completely. `DESIGN_mps_zero_copy.md` §3.3 specifies once per frame batch and the module docstring calls it "where to look first for the next speedup" | unmeasurable on this box (see the caveat above); needs a physical Mac or a kernel-count model |

### Tier 2 — general wins that the Mac feels hardest

| # | candidate | mechanism | est. |
| --- | --- | --- | --- |
| 6 | **Narrow the sheet lexsort's keys, and fold three syncs into one** (landed) | `_lexsort(pix, gkey, t)` sorted two int64 keys whose values provably fit int32: 20 radix passes over `[n]` → 12, and the three separate readbacks that precede it → one | ~1.5–2% of a T4 render; more on Metal, where each readback is a command-buffer commit and wait |
| 7 | **`torch.unique` → `unique_consecutive` where the input is provably sorted** (landed, 2 of 4 sites) | each `unique` is a clone, an index array, a full sort and a scatter; on non-decreasing input `unique_consecutive` gives the identical values and identical inverse | part of a 2–6% estimate for all four sites |
| 8 | **Densely renumber `band_id*16 + rank` by arithmetic instead of `torch.unique`** | ranks within a band are provably a contiguous prefix `{0..R}`, so `offset[band] + rank` reproduces the sorted-unique ids exactly; replaces a full `[n]` int64 sort | the largest single remaining `unique` |
| 9 | **Stop the diced attribute fan** | 21 of 26 diced attributes are corner-uniform per-mob constants that are barycentrically interpolated across three corners and then reduced back to corner 0 by `_pack_material` | ~2/3 of the dice's largest transient; buys longer windows, which compounds |
| 10 | **Let the chunk model escape one-frame chunks** | `ChunkMemoryModel.plan` returns 1 until calibrated, and with one distinct frame count observed `_safety_for` stays at the 1.6 probe margin — at 4K the single-frame peak can never satisfy `peak ≤ arena/3.2`, so the job is pinned at one frame per chunk *and* pays a 60% margin forever | pays every per-chunk fixed cost 18× at UHD; needs a forced-2-frame experiment to price |
| 11 | **Overlap the arena preflight (`prefetch_gpu_prep`)** | projection + merge + BVH run on the render thread between batches with nothing in flight; the setting exists and is off | up to 8% on a T4. A readback on the worker waits out the whole queued render — one such class measured +5.3 s of a 24 s render — but a *counted* run of the prep path (12 surfaces, 6 frames, every `.item()`/`.tolist()`/`.nonzero()`/`bool()` attributed by call site) found **128 in the batch fetch and 54 in the prewarm**, of which the merge's ~17 are already the deliberately-batched collapses. So the sync half of this is smaller than it looks; the per-mob Python dispatch is the larger half |
| 12 | **Cache the refit-BVH topology across batches** (triangles only) | consecutive batches carry the same actor set and near-identical primitive counts; refit the bounds, keep the SAH topology | ~0.2 s/batch, and it removes 30–50 of the readbacks candidate 11 needs gone |
| 13 | **Pinned, stream-ordered device→host frame handover** | `_frames_to_host` is a blocking pageable `.cpu()` on the render thread | part of a 0.64 s stage, much of which is really the chunk's kernel tail |
| 14 | **Cache `_authored_draw_order()` per render** | rebuilt every batch; its own docstring establishes it is render-invariant | small, but `O(mobs × batches)` |
| 15 | **Delete the sheet compaction's diagnostic-only outputs from the render path** | `sheet_nfrag` / `sheet_fused` / the group counters are read only by benchmark harnesses, and cost an atomic per fragment, a full `[n]` cumsum and a full `[n]` gather | 0.6–1.1% |

### Tier 3 — kernel-body candidates (measure before spending)

| # | candidate | mechanism |
| --- | --- | --- |
| 16 | **Remove the dynamic slot index from `_collect_hits`** | the k-buffer's six `ti.Vector`s are written at a runtime `slot`, which forces them into thread-private memory (224 B/thread in `wavefront_traverse_events`); the consumers are already static-indexed. Predicated static writes keep them in registers |
| 17 | **Narrow `_ss_pixel`'s lattice arithmetic to int32 operands with widening multiplies** | ~63 full 64×64 integer multiplies per candidate pixel, on an ALU that is 32-bit; every operand provably fits int32 |
| 18 | **Bucket `num_lights` as a `ti.template()` and unroll the stage light loops** | one light in this scene, but `for li in range(num_lights)` is a runtime loop in the hottest block of both shading kernels |
| 19 | **Compute `_axis_cos` once per frame** | it is `normalize(screen_point[f] - cam_origin[f])`, a per-frame constant, recomputed per active ray per bounce and per candidate pixel in the bezier kernels |
| 20 | **Bucket `_GROUP_STACK` from the built tree's depth** | two 16-entry stacks = 128 B/thread of dynamically indexed thread-private memory for a tree that is 9–11 deep |
| 21 | **Sweep `ti.loop_config(block_dim=...)`** | no kernel sets it; on Metal threadgroup size and register budget interact directly. One line per kernel, byte-identical |

### One measured caution about the cold half

A counted run of `_prewarm_render_batch` on a *trivial* scene — 12 small
surfaces, 6 frames, two contended cores — spent **930 s** against a 0.43 s
batch fetch, essentially all of it cold Inductor compilation of
`evaluate_logical_pn`, `_evaluate_logical_pn_normals_fused` and
`_snap_boundary_values_fused` across the distinct `uv` extents each dice level
produces. `evaluate_logical_pn` takes `dynamic=None`, so every distinct
trailing extent is a fresh specialization, and the dice's anisotropic
`(along, across, apex)` patterns produce many.

It is a cold cost — the T4 profile has `logical PN:   - subdivision levels` at
0.831 s cold against 0.071 s warm — so it does not move the warm ranking. What
it does move is candidate 9's justification and any proposal to *compile more*
of the dice: on a Mac each new shape variant pays that compile again, so prefer
changes that **reduce the number of distinct shapes** reaching those functions
over changes that add compiled regions. And discard the first two runs of any
prep A/B on a fresh machine, exactly as `CLAUDE.md` says for `--fast`.

### Explicitly not worth doing

* `ALGAN_ADV_OPT=1` — 2661 s of compile on CUDA for a result inside the noise
  band; the Metal chain is longer.
* Replacing the lexsort's `t` key with the coarser `depth_bin` — it sits exactly
  on the epsilon the sample-depth gate compares against.
* A static-geometry cache for *this* scene — `_update_neural_net_idle` rewrites
  every neuron subtree and every synapse tube grid on every frame, so the net
  is not static.
* Slicing an N-frame merge down to d frames — a merge is batch-wide by
  construction (visibility flags, chord decisions, texture promotion, the
  STBVH), and a different window is measurably different pixels.

## 3. One Metal defect, and what it is not

Two failures appeared on the first Metal render whose chunks held more than one
frame, and neither reproduces on the CPU with the same 4 GB pool and the same
50-frame window:

**(a) A 6.45 GB allocation** in `sheets._shade_class`, and after that was
blocked, the identical one in `sheets._prim_split_after`. Both build a
`[frames, triangles, 9]` float32 table whose height is
`int(frame_rel.amax()) + 1`.

**(b) `tracer`'s glossy scatter walked past the end of `gl_bounds`**, whose
bounds partition the covered-pixel ordinals by frame.

**They are one defect.** The instrumented run says so outright:

```
sheets._shade_class: the per-(frame, triangle) table is 7673 frames by
25082 triangles for 2000094 fragments.  Frame ordinals run -4175..7672.
```

The triangle count is right. The frame ordinal is not, and it is **negative at
one end**, which is the whole diagnosis: the emission kernel computes
`lpi = (f - time_start) * W * H + py * W + px - tile_start` and writes a
fragment only `if (lpi >= 0) and (lpi < tile_pixels)`
(`raster_taichi._pair_pixel`), with `tile_pixels` the chunk's whole ordinal
span (`raster_pipeline`'s `g1`). **A negative ordinal cannot have been written
by that kernel.** And `frag_key_u` is deliberately *uninitialized* arena memory
— the count pass says how many fragments each pair will emit and the write pass
fills exactly those slots — so a slot the write pass skipped holds whatever the
arena held before. Both failures are then the same thing: ordinals outside the
chunk's span produce an absurd table height in one place and a `gl_bounds`
partition that does not cover the stream in the other.

**Three hypotheses are ruled out, not merely unconsidered:**

* *Metal 64-bit integer arithmetic.* `benchmarks/_mps_int64_probe.py` runs the
  pack, both unpacks, the frame ordinal, the reduction, both argsorts and the
  `searchsorted` on MPS and on the CPU: **every operation agrees exactly.**
* *MPS-friendly mode* (float32 accumulators, int32 reductions). Forced on for a
  CPU render of the same scene and window, the merge produces the identical
  25,090 triangles and the render completes.
* *An unbounded frame count.* It is bounded by the kernel guard above.

What remains is a slot the write pass did not fill, and
`ALGAN_RASTER_KEY_CHECK` is the discriminator: it validates every emitted key
against `[0, tile_pixels)` right after the write pass and reports whether the
bad ones are a **contiguous tail** (the write pass emitted fewer than the count
pass promised — the two passes re-evaluate the same acceptance geometry through
*different* compiled kernel variants, `store_exact=0` against `store_exact=1`)
or **scattered** (something upstream of them).

Both tables are blocked over their frame axis regardless, so a render survives
this rather than dying on the allocation — but a wrong ordinal is wrong pixels,
not just a big buffer, which is why the warning is loud.

## 4. Landed in this round

1. **`get_num_available_bytes`'s MPS branch reports the device's real
   headroom**, not `min(free, 1 GiB)`. `ALGAN_MPS_MEMORY_CAP` restores a cap
   for A/B; `available_memory_override` still pins the figure.
2. **The source-key index keys an ndarray argument by the compiler's base
   class**, so `ExternalMetalNdarray` — the zero-copy import every Metal arena
   array arrives as — no longer poisons the key. `_dtype_name` walks the MRO for
   the same reason: the tensor element type arrives wrapped in
   `DataTypeCxxWrapper`, and the old exact-name test fell back to a `repr`
   carrying an object address, i.e. a key no second process could match.
3. **The sheet lexsort narrows its keys to int32** where the values provably
   fit — 20 radix passes over the fragment stream become 12 — and the three
   readbacks that precede it are one stacked readback, taken at the widths
   `reduction_index_dtype()` already narrows the renderer's other integer
   reductions to.
4. **Two `torch.unique` calls whose input is provably sorted became
   `unique_consecutive`.**
5. **`_shade_class`'s per-(frame, triangle) table walks its frame axis in
   blocks**, so its `[frames, N, 3, 3]` intermediates have a ceiling whatever
   the chunk holds. This is defect (a) of §3, and it is a defect on every
   device — a one-frame chunk simply never reached it.

All five are bit-identical by construction; 1 and 2 change no arithmetic at
all, and 5 is guarded by a test that pins a one-frame-per-block run against a
one-block run entry for entry.

## 5. Reproducing

```
# the Mac harness, from this branch (see agent_guidance/gpu_harnesses.md)
.github/gpu-run/mac.json:
  "command": "uv run python benchmarks/performance/nn_scene_UHD.py",
  "arms": ["mac-mps"], "quadrants_wheel": "33850787142"

# the int64 probe, seconds rather than a render
uv run python benchmarks/_mps_int64_probe.py
```
