# Brief: diagnose the immortal rays in the sheet route's bounce loop

READ-ONLY on production code. You may create throwaway probe scripts under
`scratch_perf/r3/probes/` only. Do not edit anything under `algan/`,
`benchmarks/`, or `tests/`. Deliverable: `scratch_perf/r3/ox/REPORT_immortal_rays.md`.

## The finding you are diagnosing

The sheet route's bounce loop is instrumented per iteration (see
`_format_bounce_table` in `algan/utils/profiling_utils.py`, printed in the
profile report that `profile_scene` writes). On the nn scene at HD on a T4:

```
bounce  calls   rays in  traverse s  shade s  continuations
     0     15   1584902       2.521    4.797         484736
     1     15    484736       0.881    2.176         326470
     2     15    326470       1.159    1.827          37944
     3     15     37944       0.185    0.649            938
     4     15       938       0.185    0.226             21
     5..7  13        21     ~0.44 tot ~0.34 tot           -
```

**21 rays never terminate.** They ride to `max_iters`
(`algan/rendering/raytracing/tracer.py:2309`) and force ~3 extra launch pairs
per frame-part. `DESIGN_T4_optimization.md` §5.5: diagnose before optimising —
this may be a correctness bug, not a performance one.

## Your one objective

Identify what those rays are and why no termination condition removes them.

1. **Enumerate every exit.** List every path by which a ray leaves `active`
   in the sheet route's drain loop (`tracer.py`, the `while active.numel() > 0
   and it < max_iters` loop near line 2575 and its helpers): miss, absorbed
   below a weight threshold, no continuation spawned, pool slot denied, etc.
   Cite line numbers for each.
2. **Reproduce locally.** This container has NO GPU — renders run on the
   Taichi CPU backend and work. Run
   `benchmarks/performance/nn_scene_PREVIEW.py` (from
   `benchmarks/performance/`, env `ALGAN_USE_DAEMON=0 MPLBACKEND=Agg`) and
   read the bounce table from the profile report it writes. Does a small
   cohort ride to the last iteration on CPU too? If the scene is too slow on
   4 CPU cores, copy the scene into `scratch_perf/r3/probes/` and cut frames
   or resolution — record what you changed.
3. **Identify the survivors.** With a throwaway probe (copy code or
   monkeypatch from the probe script; do not edit production files), dump for
   the rays still active at the final iterations: pixel coords, the primitive
   id / material pipeline they last hit, their weight/alpha, and which
   continuation class spawned them (glossy-prefilter row vs pooled reflection
   vs pass-through — see `DESIGN_T4_optimization.md` §3 "Continuation-ray
   spawn floor" for the classes).
4. **For each survivor, name the exit it should have taken** and the exact
   predicate that fails. Two hypotheses to check explicitly, then look
   beyond them: (a) two reflective surfaces re-spawning each other with weight
   that never decays below the threshold (what IS the weight floor, and can a
   product of reflectivities sit above it forever?); (b) a ray whose weight is
   exactly zero — or whose contribution is provably zero — that stays active
   because the exit tests something else. State for each survivor what upper
   bound its weight places on its remaining contribution, in output byte
   values.
5. **Verdict**: correctness bug (the rays contribute wrongly or loop), or
   legitimate work with a missing cheap exit, or legitimate and best left
   alone. If a fix is warranted, propose it precisely (predicate + site) but
   DO NOT implement it.

## Traps

- `tests/full_renders` baselines are per-machine and FAIL here — that is
  pre-existing; do not chase it. Do not run the full unit suite (RAM).
- The warm algan daemon can serve a probe a stale module: set
  `ALGAN_USE_DAEMON=0` in every probe run.
- Wall time on this box is meaningless; lead with counts.
- A concurrent session may be editing files under
  `algan/rendering/raytracing/raster_pipeline.py` later today; if an import
  fails mid-run, re-read the file and retry once before reporting it.

Every claim in the report carries a line number or a measured number. Label
reasoned-but-unmeasured claims as such.
