# Is the Taichi timeline-query kernel faster than torch, once staging is out of the picture?

**Answer: no.** On a CPU render device, where the Taichi kernels stage nothing,
the shipped torch query (`ALGAN_OPT_DISABLE` empty) beats or ties the kernel arm
(`ALGAN_OPT_DISABLE=torchquery`) on 12 of 14 real captured query shapes; the
kernel's only edge is ~1.1–1.2x at the single largest shape, within run-to-run
noise. The two arms are byte-identical everywhere. And the query stage is
**0.01% of a whole `save_video` wall** on this box — so even a decisive win
would have bought nothing.

For `DESIGN_taichi_arch_coexistence.md` §12.7's step 1, this resolves
negative: the design's own motivating example is not worth recovering, the
eligible inventory really is one kernel (the grid-normals block), and §10
should be taken at its word. Nothing here changes for §8.1, which stays
unanswered and CUDA-bound — but there is no longer a second candidate waiting
on it.

Everything below was produced by `benchmarks/_timeline_query_taichi_ab.py`
(landed as commit 45ff137), run with `ALGAN_USE_DAEMON=0` on this machine:
a 4-vCPU / 16 GB container, no GPU, Taichi arch x64, torch 2.7.1+cu126 on CPU.
The full harness output of both runs is reproduced in the timing section.

---

## How the shapes were measured

The script does not invent `[T, N, D]`. It instruments two real
`Scene.save_video()` renders — **default settings throughout**, PREVIEW video
preset (704x396 @ 10 fps) — by rebinding four entry points in
`algan.animation_timeline.timeline` (the established A/B convention; nothing
under `algan/` was edited):

* `generate_array_states` — every call's `(times, N, edits, prepared,
  active_rows)` is captured by reference, plus its shape signature
  `(T, N, D, U, len(edits), R)` and dtype;
* `_query_row_states` — captures the compact working-set queries as
  **selected-rows** inputs: those `(times, prepared, rows)` triples are exactly
  what the kernel arm receives at its two selected-row call sites;
* `_prepare_array_state_queries` — index builds, for share attribution;
* `AttributeTimeline.materialize_additional_rows` — opportunistic capture of
  the lazy-discovery route. It never fired (see caveats).

Both dispatch branches get genuinely received inputs:

* **Full width** (`active_rows=None`, kernel `_query_state_from_edits`):
  scene A gives 300 spawned Squares builtin move/rotate/opacity animations plus
  one `animate_function_of_time(_user_wave)` whose callback is defined in the
  benchmark module. A user function's `__module__` is not `algan.*`, so
  `FunctionTimeline`'s conservative working-set resolver returns `None`
  (`timeline._active_mob_ids`) and *every* attribute's batch query runs through
  `generate_array_states` full width — 7 calls per render, one per attribute.
* **Selected rows** (`_query_selected_state_from_edits`): scene B drops the
  user function, so all functions trace to their callers and each attribute's
  base query runs compactly over the window's live rows only — 7 calls, with
  `R` = the traced working-set size (half the scene animates, so `R < N`).

Each render is one batch covering its whole span (the memory budget never
split these scenes), so `T` is the whole-video frame count — 128 frames
(scene A, span [0, 8.47 s]) and 105 (scene B). That is the honest single-batch
regime of this box; rl2-scale multi-batch windows are discussed under caveats.

## The shapes found

All float32, animation tensors on the CPU, Taichi arch x64. `U` = CSR entries
(`sorted_edit_ids.shape[0]`), `E` = edit dicts in the log, `R` = selected rows
(-1 = full width). `out` = the `[T, N, D]` result the call must produce.

| branch | T | N | D | U | E | R | out |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full width | 128 | 1 | 1 | 1 | 1 | -1 | 0.00 MB |
| full width | 128 | 300 | 1 | 300 | 1 | -1 | 0.15 MB |
| full width | 128 | 1203 | 1 | 1203 | 1 | -1 | 0.62 MB |
| full width | 128 | 1203 | 1 | 4003 | 701 | -1 | 0.62 MB |
| full width | 128 | 1203 | 5 | 1203 | 1 | -1 | 3.08 MB |
| full width | 128 | 1203 | 9 | 1803 | 151 | -1 | 5.54 MB |
| full width | 128 | 5703 | 3 | 19972 | 752 | -1 | 8.76 MB |
| selected rows | 105 | 1 | 1 | 1 | – | 1 | 0.00 MB |
| selected rows | 105 | 300 | 1 | 300 | – | 300 | 0.13 MB |
| selected rows | 105 | 1203 | 1 | 1203 | – | 1203 | 0.51 MB |
| selected rows | 105 | 1203 | 1 | 3803 | – | 1203 | 0.51 MB |
| selected rows | 105 | 1203 | 5 | 1203 | – | 1203 | 2.53 MB |
| selected rows | 105 | 1203 | 9 | 1503 | – | 1203 | 4.55 MB |
| selected rows | 105 | 5703 | 3 | 15678 | – | 5703 | 7.19 MB |

Reading of the attributes: location is N=5703 (each Square contributes ~19
rows — body parts own rows too), color/basis/opacity sit at N=1203/1803/1203.
Density U/N lands between 1.2 and 4.4, matching what
`_timeline_query_parity.py` recorded for real scenes (1.7–4.4).

## Correctness first

Both arms ran on identical captured inputs before anything was timed. All 14
cases agree **byte-for-byte**: `torch.equal` holds on every case in both runs,
max absolute deviation **0.0**. (This matches the prior synthetic parity work:
`_timeline_query_parity.py` and `_timeline_query_ranked_ab.py`, 89 cases.)

## Timing

Methodology: the two arms alternate inside each round (a shared 4-vCPU box
drifts more across a blocked pair of runs than the effect being measured),
median of **9 rounds** reported, one untimed call per arm before the rounds so
cold kernel compilation is outside the timed region, `sync_devices()` around
each timed call. Both arms go through the shipped dispatch
(`generate_array_states` with `prepared=` prebuilt, as renders pass it); only
`tl._OPT_DISABLED` flips. The whole experiment was run twice in independent
processes; both tables follow.

Run 1:

```
  case                                          torch ms  taichi ms  speedup   branch
  T=128 N=1 D=1 U=1 E=1 R=-1                        0.15       0.15    1.00x   full-width
  T=128 N=300 D=1 U=300 E=1 R=-1                    0.20       0.33    0.59x   full-width
  T=128 N=1203 D=1 U=1203 E=1 R=-1                  0.22       1.53    0.15x   full-width
  T=128 N=1203 D=1 U=4003 E=701 R=-1                0.34       1.95    0.17x   full-width
  T=128 N=1203 D=5 U=1203 E=1 R=-1                  0.30       1.38    0.22x   full-width
  T=128 N=1203 D=9 U=1803 E=151 R=-1                0.76       4.01    0.19x   full-width
  T=128 N=5703 D=3 U=19972 E=752 R=-1               7.53       6.31    1.19x   full-width
  selected-rows T=105 D=1 U=1 R=1                   0.17       0.18    0.93x   selected-rows
  selected-rows T=105 D=1 U=300 R=300               0.27       0.27    1.00x   selected-rows
  selected-rows T=105 D=1 U=1203 R=1203             0.37       0.51    0.72x   selected-rows
  selected-rows T=105 D=1 U=3803 R=1203             0.40       0.72    0.55x   selected-rows
  selected-rows T=105 D=5 U=1203 R=1203             0.92       1.08    0.85x   selected-rows
  selected-rows T=105 D=9 U=1503 R=1203             1.40       1.29    1.08x   selected-rows
  selected-rows T=105 D=3 U=15678 R=5703            6.82       6.97    0.98x   selected-rows
```

Run 2 (independent process, same script): identical correctness, same shape
of answer — full-width mid shapes 0.24–0.35x, largest full-width 1.12x,
largest selected-rows 1.19x, small shapes 0.81–1.00x. Per-shape ratios move
by up to ±40% between processes at sub-millisecond sizes; **no verdict flips**
except the two largest shapes, which oscillate around parity in both branches
(0.98–1.19x). Speedup = median(torch) / median(taichi).

### Direct answers

**Does the Taichi arm win, lose, or tie, on each branch and each shape?**

* **Full width**: loses on 6 of 7 shapes — 1.7x to 6.7x slower (speedup
  0.15–0.59x) — and wins ~1.1–1.2x on the one largest shape (N=5703,
  U=19972), an edge that did not survive repeat-run noise as more than parity.
* **Selected rows**: loses or ties on 6 of 7 (0.55–1.00x, i.e. up to 1.8x
  slower), parity-to-slight-win at the largest (R=5703).
* The pattern in both branches: torch wins wherever the call is dominated by
  fixed costs (kernel launch + argument marshalling, ~90 µs a launch on this
  box per DESIGN §12.5, against 0.15–0.9 ms total for torch), and the two
  converge only where the result buffer itself is several MB.

**Is this the same shape of work as the prep kernels that measured slower?**
Yes — and the measurement agrees with `_CPU_PREP_KERNELS_ON_BY_DEFAULT`'s
prediction. The comment argues a kernel wins where there are intermediates to
fuse and loses where it is a bandwidth-bound copy. The timeline query kernel is
the second kind and worse than a copy: per `(frame, row)` it runs a serial
binary search over the row's CSR segment and copies D values — **no
intermediates to fuse**, and strictly more searches than the torch path, which
dedups by distinct timestamp rank, skips rows that are constant across the
window, and answers everything with vectorized `searchsorted` + gather. Where
torch wins here, it wins for exactly the recorded reason ("a memory-bound copy
with nothing to fuse... a kernel only adds launch overhead"). The one place the
kernel draws level — the multi-MB whole-span location query — is where the
result traffic dominates and torch's chunked-search machinery stops being free;
that is parity found by growing past the launch overhead, not a fused-intermediate
win like the normals block's 8–11x. Commit e159348's rank-deduplicated kernels
reached the same conclusion from the other side: give the kernels both of
torch's savings and they merely reach level in the render-window regime,
because "the query is memory-bound rather than search-bound" there.

**Other-than-staging reasons not to run this kernel, and what the history says**

* **Speed was never the recorded reason for the replacement — staging was.**
  The repo's history is squashed at 2e3264b, which already contains both
  implementations behind `ALGAN_OPT_DISABLE=torchquery`, so the replacement
  decision itself predates version history (`git log --all -S"torchquery"`
  finds only the initial import and c0c3669's wide-attribute change). The
  in-tree record is unanimous and consistent about why: `timeline.py`'s
  docstring (staging every argument, including the whole result, "on the
  batch-prep worker thread that is deliberately kept off the GPU"),
  `taichi_runtime.taichi_arch_is_cpu` (same), and `_timeline_query_parity.py`'s
  docstring, which records the operative incident — the staged path crashed
  `rl2/animations/main.py` with `CUDA_ERROR_OUT_OF_MEMORY` inside
  `cuMemAllocAsync` after staging "hundreds of MB of driver allocation per
  batch". Nowhere is speed claimed for either side.
* Two non-speed liabilities exist regardless of arch. First, the kernel arm is
  algorithmically dominated: it cannot express the rank/row dedup that makes
  the torch path cheap, so its cost is pinned to `T x N` searches even when
  three ranks cover the window. Second, its selected-row form preserves the
  full global layout — under the disable switch,
  `materialize_additional_rows` materializes the entire `[T, N, D]` buffer to
  fill a few discovered rows and then slices them out
  (`timeline.py:1604-1610`), where the default path produces a compact
  `[T, R, D]`. That is wasted memory and time on any arch, independent of
  which backend computes it.

## The share that matters

Attribution: `time.perf_counter()` sums accumulated inside the wrapped query
entry points (de-duplicated so calls from `generate_array_states` into
`_query_row_states` are not counted twice), divided by the wall time around
`save_video()` itself. Index builds are counted as part of the query stage;
nothing else is.

```
scene A (full-width):  save_video 223.7s | query stage 23ms  = 0.010%
  generate_array_states 16.0ms (7 calls) | _query_row_states 0ms
  index build 6.5ms (7 calls)
scene B (selected rows): save_video 164.5s | query stage 18ms = 0.011%
  _query_row_states 10.9ms (7 calls) | index build 7.1ms (7 calls)
```

So the quantity the design cares about converts as follows: recovering the
kernel at its best measured advantage (~1.2x on the largest shape) would change
a whole render by **~0.002%**. Even a hypothetical tenfold win on every query
call would move the wall by ~0.01%. For contrast, the staging tax the design
exists to remove was measured at ~14.7 s *per batch fetch* on the CUDA box
(`_timeline_query_parity.py`) — the thing that made these kernels a liability
was three orders of magnitude larger than the thing this measurement could
give back.

Caveat, stated plainly: this scene is 300 simple mobs rendered in one batch,
so the denominator (ray-tracing) is huge while the numerator (one query pass
per attribute per batch) is small. Real long scenes take more batches, but the
ratio moves slowly — batch count scales query work linearly and render work
just as linearly, and at rl2 scale (N up to 507k rows) the prior synthetic
measurements already showed torch at level or better in the regime that
matters (e159348). Nothing suggests a scene shape exists where this kernel is
both decisively faster than torch *and* a meaningful slice of wall time.

## What I did NOT verify

* **Anything CUDA.** No GPU here. The staging claims (and the OOM crash) are
  taken from the source docstrings and `OX_STAGING_AUDIT.md`, not re-measured;
  on this box nothing stages by construction.
* **No end-to-end render under `ALGAN_OPT_DISABLE=torchquery`.** Both
  instrumented renders ran defaults; the kernel arm was exercised through the
  shipped dispatch on captured inputs only. A full kernel-arm render would
  additionally exercise replay paths that consume queried state, which the
  byte-identity check already covers at the value level.
* **`materialize_additional_rows` never fired** in either scene: the
  conservative working set seeds itself from the whole window's actor list, so
  a static mob can never be "discovered" — that route needs a subtree created
  mid-replay. Its inputs would have been the same kind of
  `(times, prepared, rows)` triple captured off the compact site.
* One geometry family (bezier-circuit Squares), one quality (PREVIEW), one
  machine, two processes. Other families, qualities, and the multi-batch
  behaviour of memory-constrained large scenes are untested here.
* Absolute timings on a shared 4-vCPU container are noisy (per-shape ratios
  moved up to ±40% between processes at sub-ms sizes); the branch-level
  verdicts were stable across both runs. Medians, not means, throughout.
* Byte-identity is asserted over the 14 captured real cases, not the full
  synthetic edge-case matrix (unordered/repeated times, empty selections,
  non-finite values) — that remains covered by the two earlier scripts.

## Reproducing

```
ALGAN_USE_DAEMON=0 uv run python benchmarks/_timeline_query_taichi_ab.py
# AB_ROUNDS (default 9), AB_N_MOBS (default 300) to vary the harness
uv run ruff check --no-fix benchmarks/_timeline_query_taichi_ab.py
```

Rendered output: two throwaway videos under `algan_outputs/`
(`_timeline_query_ab_userfn.mp4`, `_timeline_query_ab_workingset.mp4`),
not baselines; no engine code changed, so no baseline moves.
