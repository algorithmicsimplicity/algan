# REPORT: audit of `_window_pairs` — the sheet route's candidate-pair emission

Brief: `scratch_perf/r3/ox/brief_window_pairs.md` (read-only audit; nothing
under `algan/`, `benchmarks/`, `tests/` was edited). Conventions inherited
from `scratch_perf/ox/REPORT_sheet_chain.md`. All bare line numbers refer to
`algan/rendering/raytracing/raster_pipeline.py`; other files are named.
Every claim carries a line number or a measured number; reasoned-but-unmeasured
claims are labeled **[reasoned]**.

Environment of this audit: CPU-only container (no NVIDIA driver), torch CPU
backend, Taichi 1.7.4 x64, 4 cores. T4 wall seconds are quoted from existing
reports under `scratch_perf/`; counts are the ranking currency throughout,
per the brief and `DESIGN_T4_optimization.md` §0.

---

## TL;DR

`_window_pairs` is called **twice per render chunk** (triangles at `:1502`,
bezier circuits at `:1510`) and each call is **119 dispatched aten ops + 2
`.nonzero()` host syncs** when both opacity classes are live (measured, §6
probe) — the doc's "~20 tensor dispatches" (`DESIGN_hybrid_raster.md:382`)
undercounts by ~2–6x depending on what you count. Its fast path
(`RASTER_PAIR_FLAGS`, `raytracing/settings.py:985`) fires only at the
granularity the caller passes, and the caller passes the **whole chunk
window** as one unit (`:1447-1448`) — the "per-tile" skip of
`DESIGN_hybrid_raster.md` §4.11 has collapsed to a per-chunk-per-class skip;
in the nn scene it never fires because every frame carries visible candidates
of both classes **[reasoned]**. The pass is dispatch-count-bound, not
byte-bound: every op touches `[Ft, P]`-sized data (at UHD, one frame's rows),
which is how 60 calls cost 1.13 s on a T4 while moving trivial bytes. About
two thirds of the dispatches (the four `_class_pairs_flat` bodies) are
integer gathers/copies that meet the sheet-chain conventions for a
byte-identical Taichi replacement; another slice is duplicated verbatim
between the two call sites.

---

## 1. What does `_window_pairs` compute? — ANSWERED

Definition `:978-1034`. It consumes the batched screen-bounds tables built
once per batch by `precompute_circuit_screen_bounds` (:676) /
`precompute_triangle_screen_bounds` (:809) — "both use the same schema"
(:981-982) — and emits the (primitive x screen-chunk) work items that the
raster COUNT/WRITE kernels consume. Only the window-dependent parts happen
here: the per-frame row-band clamp of each primitive's bbox y-extent and the
chunk expansion of each surviving bbox into `RASTER_CHUNK`=32-pixel strips
(`raster_taichi.py:88`). Everything else (projection, clamping,
classification) was precomputed into the tables.

### Inputs (`bounds = (pre_f, pre_x, pre_m, cls_any)`, :989)

| tensor | shape | dtype | content |
|---|---|---|---|
| `pre_f` | `[F, P, 4]` | f32 | unclamped bbox rows `floor(ymin-1)` / `ceil(ymax+1)` and raw `ymin`/`ymax` (:791-794 bez, :898-901 tri) |
| `pre_x` | `[F, P, 2]` | i64 | fully clamped bbox columns `x0`/`x1` (columns are never tile-clamped; tiles are row bands) (:795-796) |
| `pre_m` | `[F, P, 5]` | bool | `bounded`; all-front reach base (`bounded & x_on`); straddler reach base (`~bounded & front_any`); opaque; translucent (:800-805, :906-912) |
| `cls_any` | `[F, 2]` host list | bool | per-frame any-opaque / any-translucent flags; `None` when `RASTER_PAIR_FLAGS` off (:916-937) |

Scalars `time_start, g0, g1, ppf, width`: `g0/g1` bound the window in global
pixel indices; `ppf = W·H`. They arrive as Python ints, so `f0_rel/f1_rel`
(:990-991) are host arithmetic. `F` = frames in the table (the chunk's frame
count; sources dedup modulo their own length, :720-729); `P` = primitives of
that kind (circuits or triangles); `Ft = f1_rel - f0_rel + 1 <= F` frames the
window actually covers.

### Outputs

`(po, pt)` (:1027-1034): one `[M, 8]` int32 table per opacity class, or
`None`. Columns (consumed at `raster_taichi.py:1934-1941`, `:2012-2019`):
`(primitive index, absolute frame, bbox x0, bbox y0, bbox width, bbox
height, chunk pixel offset, unused-zero)`. Each row = one 32-pixel horizontal
strip of one candidate's row-band-clamped bbox:

- `K` = true entries of the class mask `[Ft, P]` (candidates; from
  `.nonzero()`, :948);
- `M = Σ_k ceil(area_k / RASTER_CHUNK)`, `area_k = bw_k·bh_k` pixels
  (:951-956);
- intermediates: `idx [K]` i64, `rep [M]` chunk→candidate map (:957),
  `off [M]` strip offsets (:961), `rows [M, 8]` before the int32 cast
  (:962-975).

Downstream shapes driven by these outputs: per-spec `counts [M]`, `accepts
[M]` i32 (:1622-1627); fragment buffers sized
`num_frags = Σ counts` (:1663-1679, `.item()` at :1670);
`num_pairs = Σ pairs.shape[0]` feeds the arena memory model (:1617-1620).

### Consumers — and whether output ORDER matters: it does

Chain: `(po, pt)` → appended per kind (:1505-1516) → `_cat` (:1518-1521; a
no-op here since each list holds exactly one part; only the fallback
per-frame loop appends many) → ordered `specs` = **[bez-opaque,
bez-translucent, tri-opaque, tri-translucent]** (:1522-1527) →

1. `raster_bez_count` / `raster_tri_count` read each row's columns and write
   `pair_count[p]` / `pair_accept[p]` **indexed by row p** (:1628-1660;
   `raster_taichi.py:1934-1977`). Per-row values are order-independent; row
   identity is not.
2. `counts` concatenated in spec order, prefix-summed (:1663-1669), sliced
   per spec into `pair_offset` (:1693).
3. `raster_*_write` starts each pair at `w = pair_offset[p]` and writes
   ascending accepted chunk pixels (:2022-2039). The **global fragment array
   layout is therefore spec order → pair-row order → pixel order**.
4. `_exact_fragment_order` (:1066-1096) re-orders the stream with **two
   stable sorts** (layer descending :1076; `(pixel << 32) | depth_bin`
   ascending :1090). Stable sorts preserve input order among equal keys, so
   fragments tying on (layer, pixel, depth-bin) keep emission order — i.e.
   pair-row order. Depth bins exist precisely to rank near-ties at
   `DEPTH_TIE_EPSILON` granularity (the coplanar draw-order machinery), so
   ties are designed-for, not exotic.
5. Downstream of the sort: opaque-prefix truncation (:1793),
   `unique_consecutive` grouping (:1786), sheet compaction and resolve;
   within-tie relative order feeds those stages, and f32 area sums in the
   compaction are bitwise order-sensitive within equal keys **[reasoned —
   downstream of :1786; not separately measured]**.

Conclusion: **no consumer re-sorts in a way that makes row order free.**
`_class_pairs_flat`'s own docstring pins the contract ("Row content and
ordering are identical to per-frame `_class_pairs` calls concatenated in
ascending frame order", :943-946). A replacement must reproduce the emitted
sequence exactly — same values, same order, same dtype — which a sequential
one-thread-per-candidate kernel provides naturally (§5). Whether some scene
happens to be tie-free is unmeasured and scene-dependent; exact order is the
safe contract.

## 2. Where do its dispatches go? — ANSWERED (measured)

Method: a `TorchDispatchMode` census of one call on synthetic tables with the
real schema/dtypes, both classes populated
(`scratch_perf/r3/probes/count_window_pairs_dispatches.py`). Counts are
shape-independent (identical totals at P=6,000 and P=20,000, frames=2,
854x480).

```
_window_pairs, both classes live       119 dispatched aten ops / call
  kernel-launching                      ~96  (119 minus metadata)
  metadata-only                          23  (11 select.int, 10 reshape, 2 view)
  host-syncing                            2  (.nonzero() inside _class_pairs_flat, :948)

top ops: index.Tensor 22 (advanced-index gathers), select.int 11, sub 10,
reshape 10, add 8, mul 6, floor_divide 6, to.dtype 6, index_select 5,
__and__ 4, arange 5, remainder 3, rsub 2, clamp_ 2, clamp 2, where 2,
nonzero_numpy 2, repeat_interleave 2, cumsum 2, zeros_like 2, stack 2,
ge 1, le 1, __or__ 1
```

Measured split of the same call:

| variant | dispatched ops | syncs |
|---|---|---|
| both classes live | 119 | 2 |
| one class flag False (per-class gate :1028/:1032) | 80 | 1 |
| no covered frame has either class (full skip :1004-1005) | **0** | **0** |

One `_class_pairs_flat` body ≈ 39 ops + 1 sync (119−80; 37 measured when
invoked standalone); shared prologue + per-kind clamp/reach ≈ 41 ops.

**Doc claim refuted as written**: `DESIGN_hybrid_raster.md:382` says the skip
avoids "~20 tensor dispatches -- and the synchronizing `.nonzero()`" per
(tile, class). Measured: ~39 + 1 sync per class body; 119 + 2 per call. "~20"
matches neither convention (it is ~2x low per class body, ~6x low per call).

**Call frequency — not per tile, not per frame.** The caller
`prepare_sparse_raster_coverage` sets `g0 = 0`,
`g1 = frames_in_chunk · ppf` (:1447-1448) and calls `_window_pairs` exactly
twice per render chunk (:1502 tri, :1510 bez). The per-(frame,tile) fallback
loops (:1456-1500) run only when a kind's table is missing (kill-switches
`ALGAN_RASTER_TRI_PRECOMPUTE` / `ALGAN_RASTER_BEZ_PRECOMPUTE`, default on,
`raytracing/settings.py:906,921`). Calls per job = **2 × chunks**, chunks
being arena-model frame windows:

| job | calls | ⇒ chunks | source |
|---|---|---|---|
| nn UHD, T4 (30 f @ 3840x2160) | **60** | 30 (~1 frame/chunk) | `scratch_perf/r2/t4_abl_base.txt:36,147`; also `report_UHD_r3.txt:34,143` |
| nn PREVIEW, T4 (50 f) | **10** | 5 | `scratch_perf/report_PREVIEW_r3.txt:56,167` |
| nn PREVIEW, this CPU box (50 f) | **36** warm | 18 | profile report, §6 |

**Multiplication for a 30-frame UHD job on the T4:**

```
30 chunks × 2 kinds × 119 dispatched ops ≈ 7,140 aten dispatches/job
  (~5,760 kernel-launching; ~690 metadata; up to 120 .nonzero() host syncs)
```

Locally (PREVIEW): 18 × 2 × 119 ≈ 4,284. Every op handles `[Ft, P]`-sized or
smaller tensors (at UHD, one frame's rows), so the pass is launch-bound —
consistent with 18.8 ms/call on the T4 (1.130 s / 60, `t4_abl_base.txt:147`)
having nothing to do with bytes moved.

## 3. The fast path — ANSWERED; effectively dead in this route

**Condition** (comment `:921-926`; gates `:992-1005`, `:1028/:1032`; toggle
`RASTER_PAIR_FLAGS` / `ALGAN_RASTER_PAIR_FLAGS`, default True,
`raytracing/settings.py:976-985`): the bounds precompute reduces one
conservative per-frame `(opaque, translucent)` any-candidates pair and moves
it to the host beside the tables (`_class_any_flags` :916-937, one
`.cpu().tolist()` per window :935-936). In `_window_pairs`: if no frame the
window covers has any candidate of either class → return `(None, None)` at
:1004-1005 **before any tensor work** (measured in §2's table: 0 ops / 0
syncs — removes ~100% of the call, leaving only the host `for fr` loop
:1000-1003). If one class is absent across all covered frames → that class's
`_class_pairs_flat` is skipped (:1028/:1032): measured ~39 of 119 ops (33%)
and 1 of 2 syncs removed. Exactness: the per-window reach mask is contained
in the per-frame reach base, so a False flag provably yields an all-false
mask where `_class_pairs_flat` would have returned `None` anyway (:917-925,
:994-997).

**Is it active in the nn scene? No — and the structural reason outranks the
scene.** The predicate's granularity is the window the caller passes. When
this was designed (§4.11 of `DESIGN_hybrid_raster.md`), pair emission ran per
tile and the flags bought per-(tile,class) skips. Today the caller passes the
whole chunk window as one unit (:1447-1448), so:

- the whole-call skip fires only if **not one frame of the chunk** has a
  candidate anywhere on screen — essentially never for a rendered scene; its
  realistic user is the empty-screen/scale-0 case of
  `DESIGN_hybrid_raster.md:400-413`;
- the per-class skip fires only if a class is absent from every frame of the
  chunk.

In the nn scene specifically: mobs animate through every frame
(`nn.move(UP)`, `label.move(RIGHT*2)`,
`benchmarks/performance/nn_scene_PREVIEW.py:23-26`), the MLP/image supply
opaque-class candidates and the Text/Tex glyph circuits translucent-class
ones, every frame **[reasoned from the scene + flag semantics at
:929-936; not instrumented]**. Both flags are True in every frame ⇒ neither
skip level can fire. Consistent with measurement: a flat ~36 ms/call across
all 18 warm chunks (§6) with no collapse.

**That gap may be the whole finding, as the brief suspected**: the fast
path's value proposition evaporated when the work stopped being per-tile.
Reviving it means re-fragmenting the emission per tile or refining flags
below frame granularity — both opposite to §5's batching direction and both
costlier than cutting dispatches directly.

## 4. The two call sites — ANSWERED

`:1501-1508` (`use_tri_pre`) vs `:1509-1516` (`use_bez_pre`): textually
identical except the bounds table passed and the lists appended to. Same
schema, no adaptation.

They do **not** duplicate work on the same inputs — different tables,
different primitive axes, possibly different F/P. They DO duplicate the
kind-independent prologue: everything from the `torch.arange` at :1006
through `rl_f/rh_f` (:1017-1018) — frame indices, `lo_p/hi_p`, `row_lo/
row_hi` — depends only on `(time_start, g0, g1, ppf, width)`, which are
bit-identical between the two calls. From the census decomposition that
shared prefix is ~14 of the ~41 non-body ops per call (~28 duplicated per
chunk; ~840/job at UHD) **[split derived from the measured op list + code
reading, not isolated at runtime]**. Hoisting it into
`prepare_sparse_raster_coverage` is risk-free (identical tensors computed
once); keep the per-kind `rows = f_abs % pre_f.shape[0]` (:1012) since the
two tables' F can differ.

## 5. Reduction plan — ANSWERED (nothing implemented)

Basis for counts: the 30-chunk UHD T4 job at 7,140 dispatched ops today.
Options compose (A+B, B+C).

| # | option | after | reduction | byte-identity risk |
|---|---|---|---|---|
| A | hoist the shared prologue into `prepare_sparse_raster_coverage`, pass results into both calls | ~6,720 | −420 (−6%) | none — identical tensors computed once |
| B | kernelise the four `_class_pairs_flat` bodies; keep host `.nonzero()` + `cumsum` | ~2,700 | −4,400 (−62%) | low |
| C | additionally fuse the kind prologue (row-band clamp, `y_on`, `reach`, mask ANDs, :1013-1033) into the kernel — one kernel per (kind, class) | ~500–600 | −6,600 (−92%) | medium |
| D | batch across classes/kinds into ONE nonzero+expansion | — | marginal over C | high — rejected |

**B (recommended first target).** One Taichi kernel per class body, thread
per candidate `k = idx order`, writing its `nch_k` rows contiguously at a
host-computed offset: host keeps `.nonzero()` (the only sync), computes
`nch = ceil(area/32)` + `cumsum` exactly as now (:956, :960), kernel gathers
each candidate's columns and writes rows `(prim, f, bx0, by0, bw, bh,
j*RASTER_CHUNK, 0)` for `j < nch_k`. Integer gathers/stores — exact by
construction, same argument class as `opaque_prefix_keep` /
`sheet_lane_first_owner` in `scratch_perf/ox/REPORT_sheet_chain.md` §2.
Emission order preserved trivially (ascending candidate, ascending strip),
which §1 showed is load-bearing. Of a ~39-op class body, ~38 disappear (the
`.nonzero()` stays); 4 bodies/chunk → −152 ops/chunk → −4,560/job.

**C.** Move the clamp/reach/mask predicate into the kernel too (thread per
(frame, prim) over `[Ft, P]`), leaving per call: `.nonzero()` + `cumsum` +
1–2 launches ≈ 4–6 ops. Medium risk because the predicate is float: `fy`
clamps against `rl_f/rh_f` f32 band edges, `y_on` compares `fy[...,3] >=
rl_f - 1.0` / `fy[...,2] <= rh_f + 1.0` (:1017-1023), then f32→i64
truncating casts. All IEEE-exact operations, but the replacement must keep
them bit-faithful (no fast-math relaxation on those compares/conversions)
and be proven on randomized tables including edge cases: all-front,
straddler-clipped, behind-camera rows, empty masks, single candidate, bbox
areas exactly divisible by 32, degenerate `x0 == x1`.

**D rejected**: concatenating classes/kinds into one mask/nonzero/expansion
changes emitted row order relative to the fixed spec sequence (:1522-1527)
unless carefully re-split; §1 established order feeds stable-sort ties, so
the saving over C is not worth the proof burden.

Keep, per the sheet-chain conventions and this brief's traps: the sorts
further down the chain stay cuB-backed; nothing here touches them. The two
`.nonzero()` syncs per call survive B (they gate value-dependent sizing:
`M`, `num_frags`) and can be revisited only after B/C land.

**Tensors whose values must be proven unchanged** (the acceptance contract):

- the four returned pair tables — `po`/`pt` from :1502 and :1510 —
  elementwise **including row order**, dtype int32, and shape;
- consequently `counts`/`accepts` (they are functions of the pair rows via
  the count kernels) and the scalars `num_pairs` (:1618-1620) and
  `num_frags` (:1670);
- end-to-end: the six sorted fragment arrays (`key/ref/ab/cov/msk/opaque`,
  :1743-1745) — proving pairs equal proves fragments equal because the write
  kernels replay `accepts` at `pair_offset` deterministically
  (`raster_taichi.py:2022-2039`).

Verification shape (for whoever implements): extend the
`benchmarks/_sheet_kernel_check.py` pattern — torch arm vs kernel arm
elementwise INCLUDING order, at 4K-scale shapes plus the edge cases above;
then hashed-frame A/B renders with the new toggle off/on (module global +
env default + setter registered in `SETTINGS.raytracing.experimental`,
declared in `algan/environment.py`, per the sheet-chain conventions).

## 6. Measure call counts locally — DONE

Ran the stock benchmark on this CPU-only box (50-frame PREVIEW,
`profile_scene(..., runs=2)`). Two environment notes: the profiler's kernel
hook calls `torch.cuda.synchronize()` unconditionally
(`algan/utils/profiling_utils.py:381`) and raises with no NVIDIA driver, so
the runner shimmed that one call to a no-op
(`scratch_perf/r3/probes/run_nn_preview_cpu.py`; no repo file edited);
telemetry off. Report:
`scratch_perf/r3/probes/report_nn_PREVIEW_cpu.txt` (copy of the generated
`algan_profile_report_nn_PREVIEW.txt`).

Warm RUN 2 (the run to read, per `DESIGN_T4_optimization.md` §0):

```
raster:   - window pairs      calls   36    excl  1.313 s   (2.3% of end-to-end wall)
  ⇒ 18 chunks × 2 calls;  36.5 ms/call
raster: sparse discovery      calls   18    incl 25.488 s   (window pairs = 5.2% of it)
ray traced render total       calls   18    incl 42.696 s   (window pairs = 3.1% of render)
```

Cold RUN 1: same **36 calls**, 1.939 s (53.9 ms/call) — count identical,
only per-op cost inflates, confirming the count story. For cross-checking,
the same stage on existing T4 reports: UHD 60 calls / 1.130 s warm =
18.8 ms/call (`t4_abl_base.txt:147`; the brief's "1.14 s" figure);
PREVIEW 10 calls / 0.138-0.161 s (`report_PREVIEW_r3.txt:56,167`). Per-call
wall seconds vary by machine/residence and rank nothing here; the counts and
the 119-ops-per-call census are the transferable numbers.

## Methods and probes

All under `scratch_perf/r3/probes/` (nothing else was created or modified):

- `count_window_pairs_dispatches.py` — TorchDispatchMode census of
  `_window_pairs` / `_class_pairs_flat` on schema-faithful synthetic tables;
  produces every measured op count in §2 (119/80/0 splits, 37-op standalone
  body, 2/1/0 syncs).
- `run_nn_preview_cpu.py` — CPU runner for the stock PREVIEW benchmark with
  the `torch.cuda.synchronize` shim; wrote `nn_preview_cpu.log` and, via the
  profiler, `report_nn_PREVIEW_cpu.txt`.
- `nn_preview_cpu.log`, `repro_run.log`, `report_nn_PREVIEW_cpu.txt` — run
  logs/report (`repro_run.log` is an aborted parallel attempt at the same
  run; superseded).

## What this audit did NOT verify

- Nothing was implemented; §5's byte-identity arguments are design-time
  arguments, not measurements.
- The nn-scene cls_any flags were never instrumented at runtime — §3's "both
  classes True every frame" is reasoned from scene content and code, labeled
  there.
- The existence/novelty of within-tie groups in real fragment streams
  (§1 step 5's bracketed note) — only their structural possibility, which is
  what sets the exact-order contract.
- T4 numbers are quoted from committed reports under `scratch_perf/`
  (`t4_abl_base.txt`, `report_UHD_r3.txt`, `report_PREVIEW_r3.txt`); this
  container has no GPU and re-measured nothing on one.
- Out of scope, deliberately untouched: the wavefront tail cohort visible in
  the same local report (30 rays riding bounces 5-7) belongs to
  `DESIGN_T4_optimization.md` §5 item 5, not to `_window_pairs`.

