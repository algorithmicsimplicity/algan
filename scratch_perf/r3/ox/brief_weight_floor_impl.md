# Brief: the bounced-ray weight-floor exit

Implement §7 of `scratch_perf/r3/ox/REPORT_immortal_rays.md` — read it first;
it is the diagnosis this fix comes from, and its §3–§5 name every predicate
and line involved. Read `CLAUDE.md` (Taichi gotchas, linting). You are in
`/home/user/algan`, branch `claude/algan-t4-optimization-b5a10b`. Do not
commit, do not push. `ALGAN_USE_DAEMON=0` everywhere. No GPU here; the CPU
backend runs everything, and this scene's tail cohort reproduces on it
exactly (30 rays, bounces 5–7).

## The change

In `wavefront_shade`'s post-loop block (between the peel-complete test and
the surface-ceiling test, `wavefront_kernels_taichi.py` — the report cites
:3600-3601 and :3602; re-locate against the current tree), retire a ray
whose throughput has fallen under the floor even if its last act was an
in-place bounce:

```python
if ti.max(weight[0], ti.max(weight[1], weight[2])) < MIN_WEIGHT:
    done = True   # completion, not truncation: do NOT touch ALLOC_TRUNC_SURFACES
```

The existing commit block then deposits its accumulated colour and leftover
throughput exactly as for any other completion (env-map sampling included).
Do NOT implement the resolve-side symmetry (report §7 calls it optional; a
bounced primary hands to the drain loop, which now catches it one iteration
later) — note the asymmetry in your report instead.

**Gate it.** New toggle: module global + env default + setter in
`SETTINGS.raytracing.experimental`, declared in `algan/environment.py`,
following the conventions of the sibling toggles there. The gate must reach
the kernel without disturbing byte-identity when off. `wavefront_shade` is
at Taichi's runtime-argument ceiling (see the comments the recovered
spawn-counts patch left about `rs_alloc`); a `ti.template()` gate is the
clean route — it compiles a separate variant, which is fine (that is how
`dump` works in the resolve) — but remember the CLAUDE.md rule this
implies: **an in-process flip of a template-fed setting still works (Taichi
specialises on argument values), but anything a `ti.static` gate reads from
a module global is baked — so read the toggle at the call site and pass it
as the argument, never bake it into the kernel body from settings.**
Whichever route you take, state it and why.

Default: decide by measurement. If every render check below shows 0
differing pixels, default ON (the fix applies the renderer's existing
significance policy consistently); if any pixel moves, default OFF and
report the count and worst channel diff — the envelope argument in report
§7 predicts under half a u8 LSB, and your job is to replace that argument
with a measurement.

## Verification (all required; quote actual output)

1. **A/B render**: nn scene at PREVIEW, toggle off vs on, separate
   processes, `SETTINGS.computing.available_memory_override` pinned
   identically, `ffmpeg_params=["-c:v", "libx264rgb", "-qp", "0"]`,
   `benchmarks/_video_diff.py`. Report differing pixels (0 expected) and
   the worst channel diff if any.
2. **The tail must actually die.** From the profile report's bounce table
   in both arms: toggle off shows the 30-ray plateau at bounces 5–7; toggle
   on must show the cohort retiring by ~bounce 5 (the report's measured
   crossing points) and fewer drain iterations. Quote both tables. Also
   quote the launch counts ("drain active count" calls) both arms — that is
   the saving this fix exists for.
3. **Prove the gate=off arm is byte-identical to the pre-change tree**: one
   extra A/B — pre-change tree (git stash your edits or use the parent
   commit in a worktree) vs gate-off arm, 0 differing pixels required. This
   is what makes the default path safe regardless of the toggle decision.
4. **A test that compiles the gated kernel.** A host-side test cannot see a
   Taichi scoping error (that class shipped a broken fan once). Add a test
   under `tests/unit_tests/` that runs a small real render (a save_frame of
   a tiny scene with a reflective material, so the drain loop and the new
   predicate actually execute) with the toggle ON, and one with it OFF,
   asserting completion — follow `test_area_light_soft_shadow.py`'s
   pattern, whose render arms exist for exactly this reason. Confirm the
   ON-arm test fails if you deliberately break the new code (say how),
   then restore.
5. `uv run -m pytest -q --fast` twice (cold compile inflates run 1; report
   both). The fast render's known baseline drift on this machine
   (5 values @ frame 27) is pre-existing — report it, do not chase it.
6. `uv run -m pytest -q tests/unit_tests/test_raytracing_unit.py` plus any
   file covering code you touched, plus your new test. Not the whole suite
   in one process.
7. `uv run ruff check --no-fix` / `uv run ruff format --check` on touched
   files (`*_taichi.py` excluded from format).

## Traps

- Taichi scopes a local to the block it is first assigned in — declare
  before branching.
- The offline cache does not invalidate on `@ti.func` edits: clear it
  (`clear_cached_kernels()`) after any kernel edit before
  re-verifying.
- Never edit `*_taichi.py` while one of your renders runs.
- One process per arm for anything a module-global gate feeds.
- `tests/full_renders` baselines are per-machine and FAIL here —
  pre-existing, do not chase.

## Report

`scratch_perf/r3/ox/REPORT_weight_floor_impl.md`: the change and gating
route; every verification output verbatim (both bounce tables side by
side); the default you chose and the measurement that chose it; everything
NOT verified (T4 timing is the operator's; CUDA compilation of the new
variant is also unverifiable here — say so explicitly).
