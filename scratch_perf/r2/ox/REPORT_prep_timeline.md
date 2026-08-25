# Report: cutting batch-preparation cost -- the attribute-read path and the neural-net idle updater

Branch `perf/r2-prep`, worktree `D:\algan_wt_prep`. Everything ran with
`ALGAN_USE_DAEMON=0` on this machine, whose venv is **CPU-only torch 2.7.1+cpu**
(Taichi arch=x64). All wall-clock numbers below are from a machine shared with
another agent: read the deltas and call counts, not the absolute seconds.

## What changed

| change | file(s) | category |
| --- | --- | --- |
| **A. Batched idle updater** (`ALGAN_BATCHED_IDLE_UPDATER`, live env flag, default **ON**) | `algan/mobs/neural_nets/neural_net.py` | could alter output -> behind a toggle; proved bit-identical, then defaulted on |
| **B. Dead-clone removal on hot read paths** (`copy=False` at provably out-of-place-only call sites) | `algan/mobs/shapes_3d.py`, `algan/mobs/surfaces/surface.py`, `algan/render_loop.py` | pure Python-overhead/copy reduction; proved value-preserving (bitwise buffers + byte-identical full-render decode); **no toggle**, per the brief's convention |
| guard test + fast-suite membership row + env declaration | `tests/unit_tests/test_neural_net_idle.py`, `tests/README.md`, `algan/environment.py` | bookkeeping the repo's own curation tests require |

## Part 1 -- measurement (before)

`uv run python scratch_perf/r2/probe_prep_cprofile.py 17` fails on this venv as
committed: it calls `torch.cuda.synchronize()` unconditionally and this torch
has no CUDA. Sibling probe `scratch_perf/r2/probe_prep_ox.py` (same scene,
guarded sync) is what produced these numbers; cProfile inflates Python-heavy
code ~1.4x, so cross-table second comparisons are indicative only.

One `get_batch_of_primitives` over a 17-frame window, pre-change:

```
ncalls  tottime  cumtime  function
     1    0.002    2.181  get_batch_of_primitives          (cProfile-inflated)
     1    0.079    1.024  AnimationTimeline.set_state_to_times
     1    0.001    0.334  neural_net.py _update_neural_net_idle
    80    0.011    0.248  shapes_3d.py _move_between_points
  2841    0.155    0.281  timeline.py AttributeTimeline.get
  2814    0.252    0.252    {method 'clone' of Tensor}
  2841    0.014    0.360  animatable.py get_animated_attribute
  1499    0.014    0.137  tensor_utils.py broadcast_all
```

Instrumented per-loop attribution (wrapper-timed, counts exact; un-profiled so
these milliseconds are real):

| loop | calls | total | per-call | gets under it | modifies under it |
| --- | --- | --- | --- | --- | --- |
| L1 neuron `move_to` | 15 | 7.7 ms | 0.512 ms | 30 | 15 |
| L2 input `set_end_point` | 5 | 10.0 ms | 1.999 ms | 75 | 15 |
| L3 hidden `move_between_points` | 50 | 113.9 ms | 2.278 ms | 800 | 150 |
| L4 output `set_start_point` | 25 | 51.9 ms | 2.077 ms | 375 | 75 |
| whole batch | | ~1550 ms | | 2841 (~26 us each inside loops) | 258 |

(The input layer has one synapse per neuron, hence 5+50+25 = 80 synapses.)
The four loops cost ~184 ms/batch here (~12% of prep) and issue 1280 of the
2841 reads and 255 of the 258 writes.

## Part 2A -- the batched idle updater

**Chosen: option 2, batch at the timeline level.** Option 1 (packing synapses
at construction) was rejected for the reasons the brief asked me to check:
packed members share a lifespan and cannot spawn/despawn independently --
nothing in `neural_net.py` (`activate`, `forward`, `reset_input_synapses`,
`zap`, `train_step`) spawns or despawns an individual net synapse today, so the
constraint would not bite -- but packing heterogeneous
Cylinder-in-Mob-in-neuron trees through `from_batches` is a structural rewrite
far past this round's size, while the timeline-level fix generalises to any
updater over many sibling mobs.

`_update_idle_loops_batched` computes exactly what the four loops write and
lands it in **three timeline writes** instead of hundreds of per-mob ops.
Bit-identity is by *replication*: the setter dance `old + (target - old)`
(never bare `target`) on every shifted row; loop 1's intermediate recursive
shift before grids are overwritten; both offset formulas (raw basis row vs
normalized-direction-times-scale) kept left-associated as written; the
interpolation pass-through at `interpolation = 1.0`; `coord_function`
re-evaluated against the new basis verbatim. Every read happens before any
write, so an unsupported structure (capped cylinders, ragged grids, differing
`radius`/`height`/`v_range`) raises `_IdleBatchUnsupported` before any state
moves and the function falls back to the original loops. Output-layer neurons
are not idle neurons and never move; their synapses take a padded zero change.
Per-mob dependency tracing is preserved (`trace_updater_mob_access` once per
idle neuron, descendants included), so later windows materialize the same
working set.

## Part 2B -- the attribute-read path

Flipped to `copy=False`, each audited to consume its result only in
out-of-place arithmetic before any write to its rows:

- `shapes_3d.py`: `set_start_point`, `set_end_point`, `move_between_points`,
  `coord_function`, `_cap_ring_offsets` (one uncloned basis read, rows derived
  locally -- same expressions), used by every cylinder stretch;
- `surface.py`: `set_location_by_function`'s `+ self.location`;
  `get_render_primitives` / `get_render_primitives_batched` grid reads;
  `compute_grid_color`'s double clone (read uncopied, keep the one defensive
  copy the in-place ops need);
- `render_loop.py`: `_build_deferred_surfaces`' group key read
  `actor.grid.location.shape`, cloning **every surface's whole grid once per
  batch to read its shape**.

Not done, deliberately:

- **Public getters keep cloning.** `mob.location` / `mob.basis` /
  `get_shader_params()` stay `copy=True`: users may mutate or retain what they
  return (and `t += x` on a view would corrupt the buffer). The engine's own
  setters already read uncopied internally; I extended that pattern, not the
  public surface.
- **`_get_attr_ranges` / `_compact_span` caching: looked at, not worth it.**
  Post-change they run ~1400 times/batch of mostly dict lookups and two numpy
  scalar reads (`_compact_span`'s endpoint check). Under this box's noise
  (±10% run to run) the single-digit ms a memo might save is unmeasurable, and
  a cache keyed on the row map adds an invalidation surface to the hottest
  correctness path in the engine. Numbers, per the brief's standard:
  `AttributeTimeline.get` tottime went 0.155 s -> 0.115 s (2841 -> 1436 calls)
  under cProfile after A removed the loops' share; the remaining per-call cost
  is dominated by three giant texture reads/writes that no range caching
  touches.

## Part 3 -- numbers after

Same probe, final code (default arm = batched):

```
warmup 1: 96.5 ms/frame   (pre-change warmup 1: 93.8 ms/frame -- noisy)
profiled batch: T=17 wall 2036 ms   (pre-change: 2184 ms)
_update_neural_net_idle  cum 0.163 (waypoint head only; loops absent)
L1-L4 instrumented entries: none (the wrapped methods are never called)

timeline ops, whole batch:   BEFORE              AFTER
  get        2841 calls      271.6 ms inst.      1436 calls     159.4 ms inst.
  modify      258 calls       82.5 ms inst.         6 calls
  {Tensor.clone}            2814 calls                          1312 calls
```

Clean A/B (`scratch_perf/r2/ab_prep_ox.py`, one process per arm, median of 6
warm batches, T=17):

| pair | arm 0 (loops) | arm 1 (batched) | delta |
| --- | --- | --- | --- |
| 1 | 2792.5 ms | 2128.1 ms | -23.8% |
| 2 | 2295.1 ms | 1869.9 ms | -18.5% |
| 3 (final code) | 2380.2 ms | 1860.2 ms | **-21.8% (0.78x)** |

## Verification (all required steps, actual output quoted)

**Byte-identical render.** `scratch_perf/r2/render_once_lossless_ox.py`
(PREVIEW, shadows on, `libx264rgb -qp 0`,
`SETTINGS.computing.available_memory_override = 4 GiB` pinned in both arms),
compared with `benchmarks/_video_diff.py`:

```
== before (pre-change code) vs after (default ON) ==
frames compared: 50
worst channel diff: 1 (frame 23)
pixels over tol 2: worst frame 0 of 278784 (0.000%, frame -1); mean 0.0/frame;
0 of 50 frames affected

== control: identical-code rerun (flag 0 vs flag 1, final code) ==
worst channel diff: 1 ... pixels over tol 2: worst frame 0 ... 0 of 50 frames affected
```

The residual worst-channel-diff of 1 is this machine's rerun noise floor (the
documented <=2 cross-run tolerance): the identical-code control shows it too.
**0 differing pixels over tolerance in every comparison.**

Stronger than pixels, the two arms were compared at the buffer level:
`scratch_perf/r2/parity_idle_updater.py` materializes a window per process and
dumps every attribute timeline's active buffer plus all non-timeline
`direction`s; `compare_parity_ox.py` requires `torch.equal` on all of them:

```
sequential vs batched, window t=0..16, dims [5,5,5,5]: IDENTICAL: all buffers bitwise equal
same, window t=20..32, dims [5,5,5,5]:                 IDENTICAL: all buffers bitwise equal
same, window t=0..8,  dims [3,4,2]:                    IDENTICAL: all buffers bitwise equal
post-B sequential vs PRE-change dump:                  IDENTICAL: all buffers bitwise equal
post-A+B batched  vs PRE-change dump:                  IDENTICAL: all buffers bitwise equal
```

(A silent fallback would also pass parity, so the harness additionally asserts
the batched plan cache exists after a materialization: `BATCHED PATH RAN:
True | synapses: 80`.)

And because Part B touches `Cylinder`/`Surface` code shared by everything, the
PN-heavy full-render scene was rendered from a clean stash and from the working
tree and the two outputs diffed directly:

```
solids_and_camera: clean checkout vs this tree
frames compared: 239
worst channel diff: 0 (frame -1)
pixels over tol 2: ... 0 of 239 frames affected
```

(byte-identical decode, 239/239 frames.)

**Unit suite.** `uv run -m pytest -q tests/unit_tests`:
`2 failed, 2060 passed, 136 skipped` -- the two failures were the repo's own
curation tests tripped by my additions (`_LIVE_VARIABLES` must be sorted;
`tests/README.md`'s fast-suite table must name every marked file). Both fixed;
both files re-run green: `24 passed`. No behavioural failure.

**Fast suite.** `uv run -m pytest -q --fast`:
`fast suite: 209s of its 75s budget (278%) -- over budget` --
`1 failed, 276 passed, 1929 deselected`. The failure is
`test_fast_render.py::test_the_fast_scene_renders_and_matches_its_baseline`:
"fast.mp4 differs from its baseline by up to 41 channel values (worst at frame
24)". Confirmed pre-existing on this machine exactly as the brief said: a
stashed clean checkout fails with the **byte-identical message** (41 channel
values, frame 24). Not chased. The budget overshoot is machine speed (this
box's single fast render alone is ~44-60 s against a budget written for the
reference machine); my added test costs ~2.5 s of the 209 s.

Also run, though not required: the PN-family full render
(`solids_and_camera`) fails against its committed baseline identically on
clean checkout and this tree ("up to 231 channel values, worst at frame 19"
both times) -- same device/baseline mismatch as the fast test -- while the two
trees' actual outputs are byte-identical to each other (above).

**Ruff.** On every touched file:
`uv run ruff check --no-fix <files>` -> `All checks passed!`
`uv run ruff format --check <files>` -> `11 files already formatted`.

## What I did NOT verify

- **CUDA.** This venv's torch has no CUDA (`torch.cuda.is_available() == False`);
  every number and every byte-identity claim here is CPU-device. The committed
  CUDA baselines were not touched and need a CUDA machine to re-check.
- **Part B's isolated timing contribution.** A/B isolates the updater toggle
  cleanly; B rides in both arms. Its mechanistic effect is the clone count
  2814 -> 1312 per batch (the survivors being mostly irreducible texture
  copies) plus the removed per-batch grid clones, but I did not measure B alone
  against pre-change code under load noise.
- **The full test suite** (`pytest -q` with `full_renders` included) beyond the
  one PN scene named above; the brief required unit_tests + --fast.
- **Daemon interplay** -- everything ran `ALGAN_USE_DAEMON=0`; the new flag is
  declared `_LIVE_VARIABLES` so a warm daemon should swap it per run, but that
  path is untested here.
- **Updaters other than this one** generalizing onto the batched path -- only
  `_update_neural_net_idle` uses it; other nets/updaters are untouched.
- Whether `zap`/`train_step` flows interact with the batched path beyond what
  the existing unit tests cover: those flows animate *color* on the same
  synapses (untouched rows) and never spawn/despawn them, which is what the
  design relies on; `test_neural_net_idle.py::test_activated_idle_synapse_
  follows_its_moving_neurons` (history clones + activate) passes under the
  default-on flag.

## Files touched

`algan/environment.py`, `algan/mobs/neural_nets/neural_net.py`,
`algan/mobs/shapes_3d.py`, `algan/mobs/surfaces/surface.py`,
`algan/render_loop.py`, `DESIGN_optimization_targets.md`,
`tests/README.md`, `tests/unit_tests/test_neural_net_idle.py`;
new probes under `scratch_perf/r2/` (`probe_prep_ox.py`,
`parity_idle_updater.py`, `compare_parity_ox.py`,
`render_once_lossless_ox.py`, `ab_prep_ox.py`). Nothing committed, nothing
pushed; nothing outside `D:\algan_wt_prep` was read or written.
