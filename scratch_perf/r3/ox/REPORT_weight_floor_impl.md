# The weight-floor exit, implemented and verified

Brief: `scratch_perf/r3/ox/brief_weight_floor_impl.md` — §7 of
`REPORT_immortal_rays.md`, gated, verified end-to-end on this container
(Taichi **CPU** backend, `ALGAN_USE_DAEMON=0` everywhere). Branch
`claude/algan-t4-optimization-b5a10b`. No commits or pushes were made by this
session (see §9 for the concurrent session's checkpoint commits).

## 1. The change

`wavefront_shade` (`algan/rendering/raytracing/wavefront_kernels_taichi.py`),
post-loop block, between the peel-complete tests and the surface-ceiling test
(the site the diagnosis cited as :3600–3602):

```python
if ti.static(weight_floor_exit):
    # The same significance floor the in-loop test applies to pass-through
    # hits, reached by rays the bounce branches' ``break`` skipped it for.
    # Completion: do NOT touch ALLOC_TRUNC_SURFACES -- what this drops is
    # sub-floor transport, not image the ceiling cuts short.
    if ti.max(weight[0], ti.max(weight[1], weight[2])) \
            < MIN_WEIGHT:
        done = True
```

A ray whose throughput has fallen under the floor now retires even if its
last act was an in-place bounce. It is a **completion**: the existing commit
block deposits accumulated colour + leftover throughput exactly as for any
other retirement (env-map sampling included), and `ALLOC_TRUNC_SURFACES` is
untouched — the surface-ceiling block below it still counts only rays the
floor did *not* retire.

No new locals are introduced, so the Taichi block-scoping trap is not
exposed (the break-check in §6 exercises exactly that failure mode anyway).

### Gating route: a `ti.template()` argument, read live at the call sites

`weight_floor_exit: ti.template()` was added to `wavefront_shade`'s signature
(after `compact`), and both call sites in `tracer.py` (the sparse drain at
:2646-area and the classic tile loop at :3281-area) pass
`int(rt_settings.WEIGHT_FLOOR_EXIT)`.

Why this route:

- The kernel is at Taichi's runtime-argument ceiling (the packed-ndarray
  comments in its signature: environment placement, far clip, gloss base and
  `max_bounces` all ride `layer_offsets` for exactly this reason), so a new
  runtime argument was not available without disturbing ABI.
- The brief forbids baking the toggle into the kernel body from settings: a
  `ti.static` gate reading a module global compiles once and silently reuses
  the first arm's code for every later flip in-process. Passing the value as
  a `ti.template()` **argument** is the safe variant of the same idea:
  Taichi specialises per argument value, so an in-process flip compiles the
  other variant instead of reusing one. Each call site reads the setting
  live per launch.
- Off-state byte-identity is by construction: with the gate 0 the compiled
  body is exactly the pre-change kernel (proven by measurement in §5).

Toggle plumbing follows the sibling conventions exactly:

- module global + env default + explanatory comment in
  `algan/rendering/raytracing/settings.py`
  (`WEIGHT_FLOOR_EXIT = env_flag("ALGAN_WEIGHT_FLOOR_EXIT", False)`);
- registered in `_FIELD_TO_LEGACY` in `algan/settings/raytracing_settings.py`
  → automatically an experimental switch:
  `SETTINGS.raytracing.experimental.weight_floor_exit`;
- declared in `_IMPORT_TIME_VARIABLES` in `algan/environment.py`
  (alphabetical position), since the env var is read once at import;
  runtime flips go through the settings view, which the call sites read live.

### Resolve-side asymmetry (noted, not implemented)

As the brief directed, the resolve-side symmetry from report §7 (sheet
resolve skips its own floor check after a bounce too) was **not**
implemented. A bounced primary hands to the drain loop, which now retires it
one iteration later than the resolve would have. Asymmetry recorded here.

## 2. Verification 1 — A/B render, gate off vs on

nn scene (`NeuralNetMLPV3([5,5,5,5])` + world-map `ImageMob` + `Text`,
identical to `benchmarks/performance/nn_scene_PREVIEW.py`) at PREVIEW,
separate processes per arm, driver
`scratch_perf/r3/ox/ab_weight_floor.py`. Both arms pinned identically:
`SETTINGS.computing.set(available_memory_override=8 * 1024**3)`,
lossless encode `ffmpeg_params=["-c:v", "libx264rgb", "-qp", "0"]`.

```
$ uv run python benchmarks/_video_diff.py nn_preview_wf_off.mp4 nn_preview_wf_on.mp4
frames compared: 50
worst channel diff: 1 (frame 0)
pixels over tol 2: worst frame 0 of 278784 (0.000%, frame -1); mean 0.0/frame;
0 of 50 frames affected
per-frame max-diff histogram (worst 8 buckets):
  diff   1: 50 frames
```

Byte-exact pixel count (any channel differing by > 0, i.e. the strict
reading of "differing pixels"):

```
frames: 50
bytes-differing pixels (>0 any channel): total 4424, worst frame 107, frames affected 50
worst channel diff: 1
```

**Not zero**: the toggle moves 4,424 pixel-instances across all 50 frames
(worst frame 107 of 278,784 = 0.038%), every single channel delta exactly
±1 u8 LSB, none over the suites' tol-2 gate. The envelope prediction
("under half an LSB") is thereby replaced by a measurement: the difference
exists, is real, and is bounded at the smallest representable increment —
consistent with sub-floor light tipping a rounding boundary at encode, not
with visible change.

## 3. Verification 2 — the tail actually dies

Stock benchmark (`benchmarks/performance/nn_scene_PREVIEW.py`, runs=2) per
arm via `scratch_perf/r3/ox/profile_weight_floor.py`, one process per arm.
Reports preserved as `profile_nn_PREVIEW_wf_OFF.txt` / `..._ON.txt`.
Warm (RUN 2) tables, side by side:

```
      OFF (toggle off)                          ON (toggle on)
bounce  calls  rays in  cont.            bounce  calls  rays in  cont.
     0     18   758566  225662                 0     18   758566  212140
     1     18   225662  155994                 1     18   212140   64583
     2     18   155994   17695                 2     18    64583    4238
     3     18    17695     437                 3     18     4238       -
     4     18      437      30
     5     14       30      30
     6     14       30      30
     7     14       30       -
```

("continuations" is the next row's rays-in; `-` = loop ended.)

- OFF reproduces the diagnosis row-for-row: the 30-ray plateau rides
  bounces 5–7 at 100% survival.
- ON: **bounces 5–7 do not exist at all** — the last iteration is bounce 3,
  whose continuation count is zero (no bounce-4 stage anywhere). Every ray
  is retired by bounce ≤ 3, *earlier* than the report's "~bounce 5"
  prediction. The reason it over-delivers: the post-loop floor catches not
  just the diagnosed 30-ray tail but every mid-chain sub-floor in-place
  bouncer (bounce-1 continuations drop 155,994 → 64,583). All three reflect
  branches bypass the spawn gates' significance test; the fix closes that
  hole for all generations, not just the last cohort.

Launch counts ("drain active count" calls — the launch saving this fix
exists for), identical in RUN 1 and RUN 2 of each arm:

```
OFF: 132   (18 parts x bounces 0-4  +  14 x bounces 5-7)
ON:   72   (18 parts x bounces 0-3)
```

−60 launch pairs per render (−45%). Warm bounce-loop kernel time fell from
4.548 s to 3.926 s (RUN 2; −13.7%), cold 6.731 → 6.511 s. End-to-end warm
50.97 s vs ~51 s — within noise on this 4-core box, as expected for a fix
whose win is mostly launches.

## 4. Verification 2b — default decision

The brief: 0 differing pixels ⇒ ON; any pixel moves ⇒ OFF. §2 measured
movement (4,424 × ±1 LSB) ⇒ **default OFF**, which is what shipped:
`WEIGHT_FLOOR_EXIT = env_flag("ALGAN_WEIGHT_FLOOR_EXIT", False)`, opt-in via
`ALGAN_WEIGHT_FLOOR_EXIT=1` or
`SETTINGS.raytracing.experimental.weight_floor_exit = True`.

NOTE for whoever lands this: a concurrent session's checkpoint commit
(6f6a08e) records a user decision made after these measurements —
"1-LSB maximum variation is acceptable, the toggle defaults ON at landing".
That decision supersedes the brief's rule; flipping the default is the
one-line change `False` → `True` in `settings.py` plus updating the comment
above it. The measurements to cite are in §2.

## 5. Verification 3 — gate-off arm is byte-identical to the pre-change tree

Worktree at the parent commit b1b3218 (`git worktree add`), driven by
`ab_weight_floor_base.py` (identical scene/pinning/encoder; no toggle line —
the tree predates it), `PYTHONPATH=<worktree>` so the worktree's package
wins over the editable install, separate process:

```
$ uv run python /tmp/opencode/count_diff_pixels.py nn_preview_base.mp4 nn_preview_wf_off.mp4
frames: 50
bytes-differing pixels (>0 any channel): total 0, worst frame 0, frames affected 0
worst channel diff: 0

$ uv run python benchmarks/_video_diff.py nn_preview_base.mp4 nn_preview_wf_off.mp4
frames compared: 50
worst channel diff: 0 (frame -1)
pixels over tol 2: worst frame 0 of 278784 (0.000%, frame -1); mean 0.0/frame;
0 of 50 frames affected
per-frame max-diff histogram (worst 8 buckets):
  diff   0: 50 frames
```

Byte-identical, as required. This also proves the ±1 LSBs of §2 are caused
by the early retirement itself, not by encoder or windowing noise.

## 6. Verification 4 — a test that compiles the gated kernel

`tests/unit_tests/test_weight_floor_exit.py`, following
`test_area_light_soft_shadow.py`'s render-arm pattern:

- host-side: the experimental setting surfaces and drives the legacy global;
- two render arms (`gate_off` / `gate_on`): a 32×32 SMOKE_TEST save_frame of
  a reflective scene (`Sphere` with `MeshPhysicalMaterial(roughness=0.12,
  ior=5)` — the nn shells' own material family — over a Lambert ground),
  asserting completion AND that every `wavefront_shade` launch carried the
  arm's gate value. The spy patches `tracer.wavefront_shade` and inspects
  arg 46/68: patching `rt_settings` does NOT work because the drain loops
  resolve it through `raytrace_render_wavefront`'s function-local rebinding
  (a closure) — found the hard way, see the test's docstring.

All three pass (and have been run repeatedly through this task).

**Break-check**: the new block was deliberately broken with the historical
failure mode — an out-of-scope local assigned inside the gated branch and
read after it (the defect class that once shipped a broken soft-shadow fan).
The ON arm fails at kernel compile time:

```
E           taichi.lang.exception.TaichiNameError:
taichi/lang/kernel_impl.py:1117: TaichiNameError
1 failed
```

The OFF arm is unaffected (gate compiled out) — which is precisely why both
arms exist. Code restored byte-exactly (verified via git diff); test green
again.

## 7. Verification 5/6 — suites

`uv run -m pytest -q --fast`, twice:

```
run 1: 1 failed, 276 passed, 1952 deselected in 28.35s
       fast suite: 28s of its 75s budget (38%)
run 2: 1 failed, 276 passed, 1952 deselected in 16.65s
       fast suite: 17s of its 75s budget (22%)
```

The one failure is `tests/fast/test_fast_render.py` — the known baseline
drift on this machine, quoted verbatim:

```
E       AssertionError: fast.mp4 differs from its baseline by up to 5 channel values (worst at frame 27); see .../output_errors/fast.mp4
E       assert 5 <= 2
```

**Proven pre-existing**, not chased: the identical failure occurs on the
untouched b1b3218 worktree (`differs from its baseline by up to 5 channel
values (worst at frame 27)` there too). Run 1 was not inflated because the
kernel variants the fast suite needs were already warm from the §2–§5
renders (the offline cache was cleared once before those).

Touched-code files plus the new test, one process
(`test_raytracing_unit.py test_environment.py test_settings_api.py
test_inert_settings.py test_weight_floor_exit.py`):

```
58 passed, 4 skipped in 22.41s
```

## 8. Verification 7 — lint

`uv run ruff check --no-fix` on all five touched production/test files: the
only finding attributable to this change was I001 in the new test file
(fixed); the other 7 findings (I001 ×3 import-block orderings, F811 ×3
duplicate names in `wavefront_kernels_taichi.py`'s header, F841 unused
`inv_rd`) reproduce identically on the b1b3218 base tree and were left
alone. `uv run ruff format --check`: clean after formatting the new test
(`*_taichi.py` excluded per CLAUDE.md). Env plumbing smoke-checked:
`ALGAN_WEIGHT_FLOOR_EXIT=0/1` select correctly at import, the field appears
in `dir(SETTINGS.raytracing.experimental)`, and the setter round-trips.

## 9. Not verified / caveats

- **T4 timing**: this container has no GPU; the CUDA-side win (launch pairs,
  occupancy) is the operator's to measure. CPU numbers above bound the
  effect structurally (−60 drain launches/render, −13.7% warm bounce-loop
  kernel time), not performanceally.
- **CUDA compilation of the new variant** is unverifiable here — the
  `weight_floor_exit=1` variant has never been through nvcc. The break-check
  proves the Taichi frontend accepts the source; register pressure on the
  GPU monolith is unknown until first compile.
- `tests/full_renders` was deliberately not run: per-machine baselines fail
  on this box regardless (brief trap), and `--fast`'s single render plus the
  three lossless PREVIEW A/Bs above cover the pixel contract.
- **Concurrent-session commits**: while verification ran, a second Claude
  session made two checkpoint commits of this working tree (4add332, 6f6a08e,
  message text theirs, including the user-decision note in §4). This session
  authored no commits; the tree content committed there is byte-identical to
  what this report describes.
- The 30-ray cohort analysis (per-ray weights, crossing points) comes from
  the diagnosis probes and was not re-derived ray-by-ray here; the aggregate
  tables in §3 are this task's confirmation.

## Artifacts

```
scratch_perf/r3/ox/
  ab_weight_floor.py            A/B driver (off/on arms)
  ab_weight_floor_base.py       same driver for the b1b3218 worktree
  profile_weight_floor.py       per-arm stock-benchmark profile driver
  profile_nn_PREVIEW_wf_OFF.txt full profile, gate off
  profile_nn_PREVIEW_wf_ON.txt  full profile, gate on
  ab/nn_preview_base.mp4        pre-change tree render
  ab/nn_preview_wf_off.mp4      gate-off render
  ab/nn_preview_wf_on.mp4       gate-on render
tests/unit_tests/test_weight_floor_exit.py
```
