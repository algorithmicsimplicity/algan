# Mac GPU round, 2026-09: what the Metal render actually costs

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

## 1. Where the Mac time goes

*(filled in from the profile below)*

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

## 3. The two Metal-only defects a wide window exposes

Both appeared on the first Metal render whose chunks held more than one frame,
and neither reproduces on the CPU with the same 4 GB pool and the same 50-frame
window.

**(a) `sheets._shade_class` asked for a 6.45 GB buffer at PREVIEW.** Its table
is `[num_frames, num_triangles, 9]`, and `num_frames` is
`int(frame_rel.amax()) + 1` where `frame_rel = (frag_key >> 32) // (W*H)`. A
6.45 GB table is a `num_frames` that is not the chunk's frame count by three
orders of magnitude, which points at the 64-bit key arithmetic rather than at
the table.

**(b) `tracer`'s glossy scatter walked past the end of `gl_bounds`.** The bounds
come from `searchsorted(covered_idx, per-frame edges)`; the loop can only
overrun if the last bound is below the covered count, i.e. if the ordinals or
the search disagree with the frame partition.

Both are consistent with one cause: **the sheet route's packed 64-bit fragment
key and the ordinals derived from it were never exercised on Metal with a
non-zero frame ordinal**, because the 1 GiB clamp made every Metal chunk one
frame — and in a one-frame chunk the high half of every key is zero and every
frame ordinal is 0. `benchmarks/_mps_int64_probe.py` runs exactly those
operations on both devices and reports the first disagreement; it needs no
scene and no kernels, so it costs seconds on the harness.

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
