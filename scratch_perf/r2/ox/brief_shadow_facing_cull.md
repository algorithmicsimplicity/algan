# Task: stop tracing shadow rays for surfaces that face away from the light

Read `D:\algan_wt_sheet\CLAUDE.md` first and obey it — especially the rules on
`*_taichi.py` files (never formatted, keep the `_taichi` suffix, `SIM`/`I002`
off there for reasons, `ti.static(bool(x))` for template gates), on never
calling `ti.init` yourself, on declaring every `ALGAN_` env var in
`algan/environment.py`, on reading settings live (`rt_settings.X` at call time),
and on `ruff check --no-fix`.

**A `ti.static` gate is baked when the kernel compiles**, so any A/B over a
static gate needs **one process per arm** — otherwise arm 2 silently reuses arm
1's code and reports its numbers as its own. This is the single most expensive
mistake available in this codebase; the brief's verification section assumes you
avoid it.

Work **only** inside `D:\algan_wt_sheet` (a git worktree on branch
`perf/r2-sheet` with its own `.venv`). Other agents are working in `D:\algan`
and `D:\algan_wt_prep` **right now** — do not read from, write to, copy into, or
run anything in any directory other than your own. Do not commit and do not
push. Set `ALGAN_USE_DAEMON=0` for every script.

You share one GPU (NVIDIA GTX 1050, 4 GB) with the other agents, so a single
pair of wall-clock numbers is not a measurement. Alternate arms in separate
processes, at least 3 each, report medians — and better, report **ray counts**,
which contention cannot distort.

## Why this task exists

Shadows are about **30% of this scene's render**. Measured on this box, the nn
benchmark scene at PREVIEW, warm RUN 2, whole run:

```
shadows on   29.35 s      kernel: raster_shadow_trace   6.33 s  (21.6%)
shadows off  20.59 s
```

On a Tesla T4 at 3840x2160 the same kernel is 4.47 s of a 29.9 s render (15%),
and a `torch.profiler` capture puts it at 16.9% of all CUDA time — second only
to `wavefront_shade`, which itself carries an inline shadow-ray block with the
same structure.

`raster_shadow_trace` (`algan/rendering/raytracing/raster_taichi.py:2744`)
traces one any-hit ray per (accepted sheet event, light, soft-shadow sample).
It **already** culls fans that cannot contribute for reasons on the *light's*
side — `_light_zero_radiance` in
`algan/rendering/raytracing/wavefront_kernels_taichi.py:151` handles a light
past its range fade, a fragment outside a spot cone, and an area sample's
backface — and the justification for that cull is written out in the kernel:

> Geometric zero-radiance culling is valid for EVERY built-in stage: each one's
> vis-multiplied terms carry lc, so a culled fan's all-lit default multiplies
> zero either way.

**There is no equivalent cull on the receiver's side.** A surface point whose
shading normal faces away from the light still gets its full fan traced, even
though a stage whose lit terms all carry `max(N·L, 0)` multiplies the result by
zero. On closed geometry that is roughly half the shaded points, per light.

## What to do

### Part 1 — establish whether the cull is sound, stage by stage

This is the whole task; the code change is small and the argument is what makes
it shippable. Do **not** implement first.

For each built-in fragment stage (the ones `pid_e < _USER_PIPELINE_BASE`
selects — lambert, phong, toon, standard, physical, and whatever else the
dispatch reaches, including `ManimMaterial`'s `get_shaded_rgb` reproduction),
read the stage and answer: **does every term that is multiplied by the shadow
visibility also carry a factor that is exactly zero when the shading normal
faces away from the light?** Write the answer per stage with the line numbers
that justify it. Watch specifically for:

- a wrap/half-lambert term, or any `N·L * 0.5 + 0.5` shaping, which stays
  positive past the horizon and would make the cull wrong;
- specular lobes evaluated from the half-vector rather than from `N·L` —
  check whether they are separately clamped by `N·L`;
- a transmission / backside term that deliberately uses the *negative* side;
- `ManimMaterial`, whose offset is added in display-referred sRGB and may not
  factor the same way.

Then: **which normal does the stage actually light with?** Two-sided geometry
flips the shading normal toward the viewer before the lighting math (see
`CLAUDE.md`'s "Shading sidedness" section and the `one_sided` material slot).
The cull must test the *same* vector the stage does, or it will cull fans that
do contribute. Establish what `event_snrm` holds at the point
`raster_shadow_trace` reads it — the raw shading normal or the already-flipped
one — and say how you determined it.

If some stages qualify and others do not, that is fine and expected: the
existing cull is already gated on `pid_e`, so gate this one the same way. If
**no** stage qualifies, say so with the evidence and stop — that is a good
result.

### Part 2 — implement it

Mirror the existing cull's shape as closely as possible: same place in the
per-light loop, same `fan_geom` style gate, the culled fan leaving the event's
all-lit default exactly as the zero-radiance skip does. Do it in **both** places
that trace shadow rays:

* `raster_shadow_trace` (the sheet route's queue), and
* the inline shadow block in `wavefront_shade`
  (`algan/rendering/raytracing/wavefront_kernels_taichi.py`) — the bounce loop's
  continuations shade through the same lights, and the T4 profile puts
  `wavefront_shade` at 29% of the render, an unknown share of which is that
  block.

Put it behind a toggle in `algan/rendering/raytracing/settings.py` following the
conventions there (env var declared in `algan/environment.py`, read live), and
default it ON **only** once byte-identity is proved. Remember that the toggle
will reach a `ti.static` template, so every A/B arm is its own process.

Be careful about the **soft-shadow fan**: a light with a non-zero radius, and an
area light's cell, sample several directions. If the receiver faces away from
the light's *centre* but toward part of an extended emitter, culling the whole
fan is wrong. Decide per sample, or prove the whole-fan cull is safe for the
emitter's extent, and say which.

Also worth checking while you are in there, and reporting even if you do not act
on it: the existing comment at `_light_zero_radiance` notes that
`_stage_default`'s fade "no longer" accumulates at `lc == 0`, so admitting it to
the cull "would be correct ... but it has not been measured and is left alone".
If it is cheap, measure it.

### Part 3 — measure

Report **counts**, not just seconds:

* shadow rays traced per frame, before and after, at PREVIEW and at HD;
* the same as a fraction of events × lights × samples;
* `raster_shadow_trace` and `wavefront_shade` wall time, medians over ≥3
  alternating processes per arm, with a sentence on how you handled the other
  agents' GPU use.

Instrumenting the ray count needs a counter that does not perturb the kernel —
an `ti.atomic_add` into a one-element ndarray behind a `ti.static` debug gate is
fine, but then the counting build is a *different* kernel from the shipping one,
so do not quote its wall time as the shipping number. Say which build each
number came from.

## Verification (all required; quote the actual output)

- **Lossless render A/B, toggle off vs on: 0 differing pixels.** Render the nn
  scene at `PREVIEW` and again at `HD`, pass
  `ffmpeg_params=["-c:v", "libx264rgb", "-qp", "0"]` (an H.264 re-encode turns
  single-channel differences into thousands of differing pixels), compare with
  `benchmarks/_video_diff.py`, and pin
  `SETTINGS.computing.available_memory_override` to the same value in both arms
  so the batch windows match. **One process per arm.**
- A scene that exercises the cases most likely to break it, rendered both ways
  and compared the same way: a **two-sided** open surface lit from behind, a
  **spot** light, a light with a non-zero `shadow_radius`, a `RectAreaLight`,
  and a mob carrying a **custom fragment pipeline** (which must keep the exact
  fan). Build it under `benchmarks/` and say what each arm proves.
- `uv run -m pytest -q tests/unit_tests` — full unit suite, with attention to
  `test_shadow_flags.py`, `test_area_light_soft_shadow.py`,
  `test_normal_orientation.py` and the tonemapping tests.
- `uv run -m pytest -q --fast` — report the timing line. Its pixel-comparison
  test fails on this machine even on unmodified code (the committed CUDA
  baseline came from a different GPU); report it and move on.
- `uv run ruff check --no-fix` and `uv run ruff format --check` on every file you
  touched.

## Report

Write `scratch_perf/r2/ox/REPORT_shadow_facing_cull.md`: the per-stage soundness
argument from Part 1 with line numbers, what normal `event_snrm` holds and how
you know, what you implemented and where, the ray-count and timing tables, the
soft-fan decision, and — explicitly — everything you did **not** verify.

## Addendum — the T4 ablation, measured after you started

Five arms on a Tesla T4 at 3840x2160, warm RUN 2, one process each:

| arm | warm | vs base |
|---|---|---|
| base | 27.92 s | — |
| `shadows=False` | 16.25 s | **-42%** |
| `max_bounces=0` | 16.40 s | -41% |
| `max_bounces=1` | 22.02 s | -21% |
| `ALGAN_ANALYTIC_AA_SECONDARY=1` | 24.97 s | -11% |

`raster_shadow_trace` alone is ~4.5 s of that 11.67 s shadow cost, so roughly
**7 s of it is shadow rays fired from inside `wavefront_shade`'s inline block**.
That is why Part 2 asks for both call sites: the inline block is the larger of
the two, not the afterthought.
