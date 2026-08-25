# Task: cut batch-preparation cost — the attribute-read path and the neural-net idle updater

Read `D:\algan_wt_prep\CLAUDE.md` first and obey it. Work **only** inside
`D:\algan_wt_prep` (this is a git worktree on branch `perf/r2-prep`, with its own
`.venv`); run Python as `uv run python` from that directory. Another agent is
working in `D:\algan` at the same time — **never read from, write to, or run
anything in `D:\algan`.** Do not commit and do not push.

Set `ALGAN_USE_DAEMON=0` for every script you run. You share one GPU
(NVIDIA GTX 1050, 4 GB) with the other agent, so **wall-clock GPU timings are
unreliable** — for this task prefer `cProfile` CPU numbers, which is what the
work here is anyway (Algan's animation device is the CPU by design).

## Why this task exists

At `PREVIEW` quality the nn benchmark scene is *preparation*-bound, not
render-bound. On a Tesla T4 (warm steady state, whole run 6.25 s for 50 frames):

```
stage                                        calls  incl (s) incl(%)  excl (s) excl(%)
ray traced render total                          5     3.245   52.0%     0.022    0.3%
Scene.get_batch_of_primitives                    3     2.322   37.2%     0.499    8.0%
arena preflight (batch)                          3     1.519   24.3%     0.011    0.2%
AnimationTimeline.set_state_to_times             3     0.945   15.1%     0.533    8.5%
AttributeTimeline.get                        22702     0.807   12.9%     0.807   12.9%
surfaces: get_render_primitives_batched          6     0.776   12.4%     0.665   10.6%
AttributeTimeline.modify                      4578     0.187    3.0%     0.187    3.0%
AttributeTimeline.add                         2770     0.164    2.6%     0.164    2.6%
```

Prep and render are almost perfectly serial (2.32 + 3.25 + preflight ≈ 6.25),
so a second saved in prep is a second off the run.

A local `cProfile` of **one** `get_batch_of_primitives` call over a 5-frame
window (`scratch_perf/r2/probe_prep_cprofile.py`, already in this tree — run it
as `uv run python scratch_perf/r2/probe_prep_cprofile.py 17`) breaks the 0.76 s
down as:

```
ncalls  tottime  cumtime  function
     1    0.001    0.761  render_loop.py:2005(get_batch_of_primitives)
     1    0.001    0.302    _build_deferred_surfaces
     1    0.000    0.293    AnimationTimeline.set_state_to_times
     1    0.001    0.241      neural_net.py:148(_update_neural_net_idle)
    80    0.008    0.182        shapes_3d.py:1019(_move_between_points)
  2841    0.056    0.102        timeline.py:1019(AttributeTimeline.get)
  2814    0.037    0.037          {method 'clone' of Tensor}
  2841    0.010    0.162        animatable.py:1051(get_animated_attribute)
  1499    0.011    0.082        tensor_utils.py:217(broadcast_all)
    80    0.001    0.080        surface.py:3557(set_location_by_function)
    80    0.016    0.037        shapes_3d.py:939(coord_function)
    80    0.011    0.034        geometry.py:501(get_orthonormal_vector)
```

**`_update_neural_net_idle` is ~32% of preparation.** It is
`algan/mobs/neural_nets/neural_net.py:148` — an updater that runs every batch and
walks 15 neurons and 80 synapses in a **Python loop**, calling `move_to`,
`set_end_point`, `move_between_points` and `set_start_point` one mob at a time.
Every one of those is several timeline reads and writes on tensors of shape
`[T, 1, 3]`, so essentially all of the cost is Python and torch dispatch
overhead, not arithmetic.

## What to do

### Part 1 — measure first

Run the probe above and report its table. Then extend it (or add a sibling probe
under `scratch_perf/r2/`) so you can attribute the updater's own 0.241 s across
its four loops (the neuron `move_to` loop, the input-synapse `set_end_point`
loop, the hidden-layer `move_between_points` loop, the output-layer
`set_start_point` loop) and across the timeline primitives underneath them.
Report per-call costs, not just totals.

### Part 2 — the two things to fix, in this order

**(A) Batch the idle updater across its mobs.** 80 synapses whose endpoints are
being set individually should be one batched operation. Algan already has the
machinery for this — a Mob can be *packed* (one Mob standing for N logical
objects, one attribute row per member, `parent_batch_sizes` mapping component
row blocks back to members). `CLAUDE.md`'s "Mobs" section and
`agent_guidance/AGENTS_DETAILED.md` describe it, including the two invariants
that fail silently; `batch_mobs` and the `from_batches` constructors are the
entry points, and `Text` packing its glyphs is the worked example to copy.

Pick whichever of these you can make work and justify the choice:
  1. pack the synapses (and the idle neurons) at construction time in
     `NeuralNetMLP.__init__` so the updater operates on one packed Mob, or
  2. leave the mobs alone but make the updater write all 80 synapses' rows in
     one timeline operation instead of 80 — i.e. batch at the *timeline* level
     rather than the *mob* level.

Option 2 is likely the smaller change and generalises to any updater that
touches many sibling mobs; option 1 is the deeper fix. Read enough of the
timeline to decide, and say why you chose what you chose. Note the constraint
that packed members share a lifespan and cannot spawn or despawn independently —
check whether anything in `neural_net.py` (`activate`, `forward`,
`reset_input_synapses`, `zap`, `train_step`) spawns or despawns an individual
synapse before you commit to option 1.

**(B) Make the attribute-read path cheaper.** `AttributeTimeline.get`
(`algan/animation_timeline/timeline.py:1019`) is called 2841 times per prep in
this scene at ~36 µs each, and 2814 of those calls `.clone()`. Its `copy=True`
default exists so callers may mutate the result in place; the docstring already
notes that read-only callers pass `copy=False`. Audit the callers reached from
`Animatable.get_animated_attribute` (`animatable.py:1051`) and
`AnimationTimeline.get_attr` (`timeline.py:2353`) and establish which of them
genuinely need the copy. Anything whose result only feeds out-of-place
arithmetic does not. Do not blanket-flip the default — find the specific hot
callers, prove each one cannot mutate, and flip those.

Also look at `_get_attr_ranges` (`animatable.py:903`, 3093 calls) and
`_compact_span` (`timeline.py:961`, 3093 calls) for per-call work that could be
cached on the mob and invalidated by the existing `structure_version` bump.

Anything you change that could alter output must sit behind a setting toggle
following the existing convention (env var declared in `algan/environment.py`,
read live at call time), defaulting ON only once you have proved byte-identity.
A pure Python-overhead reduction that provably cannot change any tensor value
does not need a toggle — but say which category each change is in.

### Part 3 — report the numbers

Re-run the probe and report before/after: total prep time per batch, the
per-loop breakdown, and the call counts for `AttributeTimeline.get` / `.modify`
/ `.add`. Report **call counts**, not just seconds — they are the thing that
transfers to the other machine.

## Verification (all required; quote the actual output)

- **Byte-identical render.** Render the nn scene at `PREVIEW` before and after
  your change and show the outputs are identical. Use
  `scratch_perf/render_once.py` as the template, pass
  `ffmpeg_params=["-c:v", "libx264rgb", "-qp", "0"]` so the comparison is
  lossless (an H.264 re-encode turns single-channel differences into thousands
  of differing pixels), and compare with `benchmarks/_video_diff.py`. Must be
  **0 differing pixels**. Pin `SETTINGS.computing.available_memory_override` to
  a fixed value in both arms so the batch windows are identical — window changes
  legitimately move pixels and would mask a real regression.
- `uv run -m pytest -q tests/unit_tests` — the full unit suite. In particular
  `tests/unit_tests/test_neural_net*.py` (if present), the timeline tests and
  `test_ux_regressions.py`.
- `uv run -m pytest -q --fast` — report its timing line. Its pixel-comparison
  test fails on this machine even on unmodified code (the committed CUDA
  baseline came from a different GPU); confirm that against a clean checkout and
  report both, but do not chase it.
- `uv run ruff check --no-fix` and `uv run ruff format --check` on every file
  you touched.

## Report

Write `scratch_perf/r2/ox/REPORT_prep_timeline.md`: the before/after tables, what
you changed and why, which changes are provably value-preserving versus which
are behind a toggle, and — explicitly — everything you did **not** verify. If
one of the two parts turns out not to be worth doing, say so with the numbers
that show it rather than doing it anyway.

## Addendum (added while you were working) — the render device may change under you

This worktree's `.venv` was created with a **CPU-only** torch (`2.7.1+cpu`), and
a CUDA build (`2.7.1+cu128`) is being installed into it in the background right
now. A process that already imported torch keeps the build it started with; a
process started after the install finishes gets CUDA, and algan will then report
`Rendering device set to cuda` and run Taichi on `arch=cuda` instead of `x64`.

**A CPU render and a CUDA render of the same scene are not expected to be
byte-identical** — that is a device difference, not a regression. So before you
compare two renders, check that both arms printed the same device line, and if
they did not, re-run the earlier arm. Every A/B pair must be same-device.

Once CUDA is available, prefer it: a PREVIEW render on this CPU is minutes, on
the GPU it is seconds.
