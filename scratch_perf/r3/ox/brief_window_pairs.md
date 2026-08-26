# Brief: audit `_window_pairs` — the never-examined sheet-route host pass

READ-ONLY audit. Do not edit anything under `algan/`, `benchmarks/`, or
`tests/`. Throwaway probes go in `scratch_perf/r3/probes/` only. Deliverable:
`scratch_perf/r3/ox/REPORT_window_pairs.md`.

## Context

On a Kaggle T4, `nn_scene_UHD.py` (30 frames, 3840x2160) renders in ~26 s, of
which ~5.2 s (17%) is the sheet route's host-torch passes. The profiler stage
`raster:   - window pairs` — `_window_pairs` at
`algan/rendering/raytracing/raster_pipeline.py:978` — is 1.14 s of that, the
single largest host pass after `compact_sheets` (2.3 s), and it has never been
examined. A previous round kernelised three other passes of this chain
(`scratch_perf/ox/REPORT_sheet_chain.md` — read it; the conventions it
records for byte-identical Taichi replacements of host passes apply here).
`DESIGN_T4_optimization.md` §5.2 is the plan-of-record entry.

## Questions — answer each with CONFIRMED/REFUTED/ANSWERED plus line numbers

1. **What does `_window_pairs` compute?** Inputs, outputs, and every tensor's
   shape as a function of (tiles, frames, primitives, pixels). What consumes
   its outputs, and does any consumer depend on the exact order of what it
   emits, or only on values that survive a later sort? (If a consumer
   re-sorts, reordering inside `_window_pairs` is free; say which.)
2. **Where do its dispatches go?** Count the tensor ops per call by reading
   the code (`DESIGN_hybrid_raster.md:382` claims "~20 tensor dispatches").
   How many times is it called per frame at UHD — per tile, per frame-pair,
   per batch? Give the multiplication that produces the total dispatch count
   for a 30-frame UHD job.
3. **The fast path.** `raytracing/settings.py:979` and
   `raster_pipeline.py:922` describe a condition under which `_window_pairs`
   "skips its per-tile tensor work". State the exact condition, whether it is
   active in the nn scene (check what the scene does — moving mobs? — against
   the predicate), and what fraction of the function's work the skip removes
   when it fires. If the nn scene does NOT take the fast path, say why not —
   that gap may be the whole finding.
4. **Both call sites** (`raster_pipeline.py:1502` and `:1510`): what differs
   between them, and do they duplicate work on the same inputs?
5. **Reduction plan.** Given 1–4: can the per-tile work be batched across
   tiles into one dispatch set, hoisted out of a loop, or moved into a Taichi
   kernel per the sheet-chain conventions? Rank the options by (a) expected
   dispatch-count reduction — counts, not seconds — and (b) byte-identity
   risk, and name the exact tensors whose values must be proven unchanged.
   Do NOT implement anything.
6. **Measure call counts locally.** This container has NO GPU; the CPU
   backend works. Run `benchmarks/performance/nn_scene_PREVIEW.py` (from
   `benchmarks/performance/`, env `ALGAN_USE_DAEMON=0 MPLBACKEND=Agg`) and
   report the profiler's call count and per-call cost for the
   `window pairs` stage from the profile report it writes. Wall seconds here
   are meaningless for ranking; report them but lead with counts.

## Traps

- `tests/full_renders` baselines are per-machine and FAIL here —
  pre-existing; do not chase. Do not run the full unit suite (RAM).
- `ALGAN_USE_DAEMON=0` in every probe run.
- The sorts in this chain are cuB-backed and are to be kept; the target is
  the segment construction, gathers, and the `cat`/`stack`/`copy_` family
  around them.

Every claim carries a line number or a measured number; label
reasoned-but-unmeasured claims as such.
