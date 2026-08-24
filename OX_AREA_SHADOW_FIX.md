# OX_AREA_SHADOW_FIX — an area light's shadow rays now integrate its emitter cells

Implements the fix specified in `/tmp/ox_area_shadow_impl.md`, on top of
`OX_AREA_SHADOW_AUDIT.md`. Branch `claude/area-shadow-banding-fix-i411ly`.
**No render and no pytest suite beyond the two files named in §7 was run;
nothing is committed or pushed.**

In one sentence: each `RectAreaLight` row now packs its emitter CELL's
half-extents and the rectangle's basis into previously-unused columns, and both
deterministic shadow fans place their samples inside that cell, in the light's
own plane (R2 sequence), so a row's visibility term is the average over its
cell instead of a point test at its centre. Radiance, power fractions and
`intensity` are untouched.

## What changed, and where

### Host side

- **`algan/rendering/lights.py`**
  - `RectAreaLight._grid_side()` — new private helper returning
    `ceil(sqrt(samples))`; `num_samples()` and `get_sample_positions()` now
    call it instead of recomputing `ceil(sqrt(...))`, and `build_aux` uses it
    for the cell sizes.
  - `RectAreaLight.build_aux` — when `rt_settings.AREA_LIGHT_SOFT_SHADOWS`
    (read live at call time) is true, additionally writes aux 6 = `hu =
    width/(2k)`, aux 7 = `hv = height/(2k)`,
    aux 8 = `sqrt(4*hu*hv/pi)` (the cell's equal-area disk radius), and
    aux 9-11 = the rectangle's `right` unit axis per frame from
    `_rect_axes(location)`. `up` is not packed; the kernel recovers it as
    `cross(normal, right)`, which is exactly how `_rect_axes` builds it.
    With the flag off nothing extra is written — bit-for-bit today's row.
  - Docstrings: module docstring's "shadows are naturally soft" claim
    replaced with the real mechanism; class docstring rewritten (per-cell
    integration, continuous penumbra, the `K * SOFT_SHADOW_SAMPLES` cost,
    defaults stated per DOCSTRINGS.md); the aux-layout table in
    `Light.build_aux` updated for rows 8 and 9-11, which are no longer
    spot/hemisphere/env-only.
- **`algan/rendering/raytracing/scene_builder.py`** — `_pack_lights`' layout
  docstring gains a paragraph stating what columns 9/10/11/12-14 carry for
  `ltype == 5` rows and that every reader of them is type-guarded.

### Kernel side (the two live shadow fans, same change in both)

- **`algan/rendering/raytracing/raster_taichi.py`** (`raster_shadow_trace`)
  and **`algan/rendering/raytracing/wavefront_kernels_taichi.py`**
  (`wavefront_shade`'s inline fan):
  - *Cell read*, guarded: next to the `radius = light_col[..., 11]` read,
    `hu`/`hv` load from packed columns 9/10 only when
    `ltype == _LT_AREA_SAMPLE`. The guard is load-bearing — those columns are
    a spot light's cone cosines there. Shape guards match each site's local
    style (`shape[2] > 11` nesting in raster; none extra in wavefront, whose
    col-11 read was already guarded only by `> 3`).
  - *Basis*: under `if radius > 0.0:`, when `(hu > 0.0) or (hv > 0.0)` the
    fan uses the light's own plane — `b1` = packed right (cols 12-14),
    `b2` = `cross(packed normal (cols 6-8), b1)` — instead of the
    `wi`-perpendicular `aref` construction, which survives verbatim in the
    `else`.
  - *Offsets*: same branch shape. Rect arm:
    `off = b1*(hu*ru) + b2*(hv*rv)` with `(ru, rv)` from R2
    (`u = 0.5 + a1*s; ru = 2*(u - floor(u)) - 1`, likewise `v`/`rv`);
    s = 0 is exactly the cell centre, so a one-sample fan degenerates to
    today's ray. The disk `else` keeps the golden-angle expressions exactly
    as they were, and the following `_LT_DIRECTIONAL` / finite-distance split
    is untouched (an area row always takes the finite-distance arm).
  - Constants live next to `_GOLDEN_ANGLE` in `wavefront_kernels_taichi.py`
    (`_R2_SEQUENCE_A1/A2`, plastic-number reciprocals) and are imported into
    `raster_taichi.py`; the comment there records why R2 rather than a
    jittered grid (deterministic, no per-cell state, uniform for any S).

### Deliberate exclusions (also recorded in the settings comment)

- **`wavefront_shadow`'s single-ray block** (deferred prepass): left alone.
  It takes `light_pos` but never `light_col`, so it has neither type nor
  radius — it treats every row as a hard point light already, and it is dead
  code (its own docstring: the tracer always compiles
  `deferred_shadows == 0`). Reviving it is a host contract change that must
  learn these columns first.
- **The Monte Carlo megakernel's NEE loop**: left alone. It reads packed
  columns 0-2 only (`raytrace_kernels_taichi.py:3686-3688`), and extended
  lights are rejected at preflight when `samples_per_pixel > 1`
  (`tracer.py`), so an area row can never reach it.

### The toggle

- **`algan/rendering/raytracing/settings.py`**: new module global
  `AREA_LIGHT_SOFT_SHADOWS = env_flag("ALGAN_AREA_LIGHT_SOFT_SHADOWS", True)`
  with a SOLID_SHELL_ALPHA-style comment block: what it does, the cost, the
  two known limits above, and why OFF needs no recompile (host-side-only
  flag, zeros packed, no `ti.static` gate — one process can render both arms).
  The adjacent `SHADOWS` comment and the `set_ray_traced_shadows` docstring
  now describe the real soft-emitter design instead of the intended one.
- **`algan/environment.py`**: `ALGAN_AREA_LIGHT_SOFT_SHADOWS` declared in
  `_IMPORT_TIME_VARIABLES` (alphabetical). Classification rationale: this is
  exactly `ALGAN_SOLID_SHELL_ALPHA`'s situation — `env_flag` runs at module
  level during `settings.py`'s import, baking the env value into the module
  global; after that the *global* is read live (`rt_settings.AREA_LIGHT_SOFT_SHADOWS`
  at pack time) and is what the SETTINGS experimental view writes. So the
  env var is import-time (a warm daemon cannot adopt a changed value) while
  the knob itself is runtime-live. `test_environment.py` enforces precisely
  this split and passes.
- **`algan/settings/raytracing_settings.py`**: `"AREA_LIGHT_SOFT_SHADOWS"`
  added to `_FIELD_TO_LEGACY`, so it surfaces as
  `SETTINGS.raytracing.experimental.area_light_soft_shadows`. Not added to
  `_PUBLIC_FIELDS`: it is an experimental switch like its neighbours.

## The four adversarial questions

**1. Is there any packed-row column you wrote that something reads for a
light type other than area? Name the guard that stops it.**

Written: packed cols 9, 10 (hu/hv), 11 (equal-area radius), 12-14 (right
axis) — and only inside `RectAreaLight.build_aux`, so only `ltype == 5` rows
ever carry non-zero values there. Every reader:

| Column | Reader | Guard |
| --- | --- | --- |
| 9, 10 | `shading_taichi.py:780-781` (spot smoothstep) | `ltype == _LT_SPOT` |
| 9, 10 | `_light_zero_radiance` (`wavefront_kernels_taichi.py:183-184`) | `ltype == _LT_SPOT` |
| 9, 10 | the two fans' new `hu`/`hv` loads | `ltype == _LT_AREA_SAMPLE` |
| 11 | both fans' `radius` gate | none (unconditional) — but that read is the intended hook; area rows now carry the equal-area radius, other types keep their own historical values there |
| 11 | `shading_taichi.py:757-758` (env-SH `by[2]`) | `ltype == _LT_ENV_SH` |
| 12-14 | `shading_taichi.py:745-747` (hemisphere ground) | `ltype == _LT_HEMISPHERE` |
| 12-14 | `shading_taichi.py:759-760` (env-SH `bz`) | `ltype == _LT_ENV_SH` |
| 12-14 | the fans' new `b1` load | reached only under `(radius > 0) and ((hu > 0) or (hv > 0))`, i.e. an area row |

Re-swept by grep after the edit: no other `light_col[..., n]` reader exists
in `algan/` (the MC kernel reads 0-2 only; the torch vertex-shading path
skips all extended rows; compact 3-column packing never reaches the reads).
Column 15 (power fraction) untouched.

**2. Does the non-rect (`hu == 0`) path in both fans compile to the same
expressions, in the same order, as before?**

Yes, expression for expression. The original disk statements were moved
verbatim into the `else:` arm of a runtime branch on `(hu > 0.0) or
(hv > 0.0)`; nothing else about their arithmetic changed — `aref`
selection, `wi.cross(aref).normalized()`, `wi.cross(b1)`, the golden-angle
`ang/rr/off` computation including operand order, and the subsequent
directional/finite-distance split are character-for-character the same
operations (only indentation/wrapping differs). The new code paths add only:
two scalar loads behind `ltype == _LT_AREA_SAMPLE` (false for every
non-area row), and one runtime boolean test per fan site. No float
expression was reordered, merged, or duplicated. The `off = vec3(0)` init in
raster stays outside the radius gate as before.

**CORRECTION (`OX_AREA_SHADOW_FIX2.md`): the closing claim below was wrong,
and the kernel did not compile.** In wavefront, assigning `off` in both arms
of the inner if/else does *not* preserve single-assignment semantics — there
is no Taichi SSA phi across arms; a local is scoped to the block it is first
assigned in, so reading `off` after the if/else raised `TaichiNameError` at
kernel-compile time (measured, not reasoned). The fix hoists
`off = ti.math.vec3(0.0, 0.0, 0.0)` above the radius gate, mirroring the
raster fan's init, with both arms still assigning `off` fully before any
read — disk-path arithmetic unchanged.

**3. With the flag OFF, is the packed row bit-for-bit today's row?**

Yes. The entire new block sits behind
`if rt_settings.AREA_LIGHT_SOFT_SHADOWS:` in `build_aux`; off, the method
executes exactly its old three writes (decay, distance, normal) over
`_blank_aux`'s output — aux 6-11 remain the zeros `_blank_aux` wrote. Kernel
side nothing depends on the flag at all: area rows then have `radius == 0.0`
so both fans take the unchanged single-ray path, and although the
`ltype`-guarded `hu`/`hv` loads still execute (deliberately, so kernels are
flag-blind), they load 0.0 and select nothing. Test
`test_flag_off_packs_todays_row` pins the zero columns plus the untouched
base fields; `test_experimental_setting_surfaces_and_drives_the_legacy_global`
pins the same through the public settings surface. What no host-side test can
pin is the rendered bytes — that is for the suites you run (see §below).

**4. What is NOT covered by the unit test, and would only show in a render?**

- The kernel changes themselves — compilation of both edited kernels
  (including the wavefront fan's `off`, which as written did NOT compile:
  TaichiNameError — corrected and now compile-tested, see
  `OX_AREA_SHADOW_FIX2.md`), and every
  line of the rect branch: the `cross(normal, right)` up-axis recovery, the
  R2 sample placement, the horizon culls applied to in-plane offset
  directions, and the interaction with `sec_aa` sub-pixel origins (Design B's
  offsets are origin-independent by construction, but that is reasoned, not
  exercised).
- That the union over K rows is actually continuous (the staircase is gone):
  the tests pin tiling algebra, not the integrated visibility field.
- Cost/perf: `SOFT_SHADOW_SAMPLES`× more shadow rays per area row shifts
  arena peaks → different chunk windows via the memory model; possible new
  truncation-counter hits (`MAX_SHADOW_LIGHTS`, surfaces-per-ray ceilings)
  on dense scenes.
- Pixel baselines: `tests/full_renders` act 3 (`materials_and_lighting`,
  shadows on) will move — both device sets, CPU regenerable here, CUDA needs
  a CUDA machine. `tests/fast` pins nothing here (no area light, no shadows)
  but certifies non-area bytes did not move.
- CUDA-specific behaviour of everything above (this container has no GPU).

## Verification run (all of it, verbatim outcomes)

- `ruff check --no-fix algan/ tests/` → **25 errors, all pre-existing.**
  Verified by running the same command on HEAD via `git stash`: identical
  finding list (line numbers only shifted: settings.py I001 2545→2590,
  wavefront F841 3505→3569). Includes `shading_taichi.py:399` D209, known
  pre-existing. **Zero findings introduced.**
- `ruff format --check` on lights.py, raytracing/settings.py,
  environment.py, raytracing_settings.py, scene_builder.py,
  test_area_light_soft_shadow.py → **"6 files already formatted"**.
  (Kernel files are formatter-excluded by config.)
- `.venv/bin/python -c "import algan"` → **ok** (Rendering device set to
  cpu; Taichi 1.7.4 arch=x64).
- `.venv/bin/python -m pytest -q tests/unit_tests/test_area_light_soft_shadow.py
  tests/unit_tests/test_environment.py` → **27 passed** (9 new + 18
  environment) in ~7s.

No render, no baseline comparison, no other suite was executed.

## What I did not verify

- Everything in question 4 above — kernels uncompiled, renders unrendered.
- Whether `pn_criterion_kernel`/fast-math tessellation interacts here: it
  does not (no tessellation, projection or criterion touched), asserted from
  the diff, not measured.
- That no third-party downstream consumer (docs examples render at docs
  build time) breaks: `docs/source/advanced_user_tutorials/lighting_and_
  shadows.rst` renders a `RectAreaLight` scene with shadows on; its pixels
  will change with the fix, which is intended, but I could not build docs.
- The audit's falloff defect (REPORT.md §6.7) is untouched by design; the
  equal-area radius in column 11 is visibility-gating metadata and feeds no
  radiance path (`_light_eval` reads columns 3-8/15 for area rows, never 11).
- Daemon interplay beyond classification: an A/B across the flag in one warm
  daemon process should work because the flag is read at pack time, but this
  was not exercised.
