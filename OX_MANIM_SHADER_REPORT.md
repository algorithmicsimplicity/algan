# OX_MANIM_SHADER_REPORT — step 1: implement `manim_shader`

Step 1 of 2 per `ox_brief_impl1.md`: adds Manim's default 3-D lighting as a
torch shader, an in-kernel stage (repurposing built-in material id 0), a
`ManimMaterial`, and the registry wiring. Nothing was committed; the work is in
the tree.

## What changed and why

| File | Change |
|---|---|
| `algan/rendering/shaders/material_shaders.py` | New `manim_shader` next to `lambert_shader`: the canonical nine fixed parameters plus one extra, `flat_shading`. Per light it computes Manim's offset `0.5 * dot(n, to_sun)**3` (halved when negative) multiplied by `light_color[..., :3] * light_intensity`. Under `_linear_color_space()` it encodes the base to display-referred sRGB (`linear_to_srgb` from `algan/utils/color_space.py` — not hand-written), adds, clamps `[0, 1]`, decodes back; otherwise adds and clamps directly. Returns through `_recombine(out, glow)`. Docstring states what is reproduced exactly (Manim's scalar offset under the single white intensity-1 `PointLight` with decay 0/distance 0 that `use_manim_defaults()` installs), the tint as a strict generalisation, and the audit §7d caveat (exposure 1 / tonemapping off). |
| `algan/rendering/raytracing/shading_taichi.py` | `_MID_DEFAULT = 0` → `_MID_MANIM = 0`; `_stage_default` replaced by `_stage_manim` (same 14-argument stage contract); `_BUILTIN_STAGE_FNS` position 0 swapped. The round trip is gated `ti.static(bool(_linear_color_space()))` exactly as `_energy_scale` gates its budget. No `_energy_scale`, no ambient, no emissive. Prose updated at the three sites the audit lists: module docstring + pid table, `MAX_SHADOW_LIGHTS` comment, `_light_eval`'s power-fraction paragraph (no built-in stage consumes `frac` any more). Also added the `color_space_taichi` import and updated `builtin_pipeline_fn`'s id list. |
| `algan/rendering/raytracing/wavefront_kernels_taichi.py` | Dropped the `_MID_DEFAULT` import; removed the fan-cull special case at the shadow-fan site (`fan_geom = 1` for every built-in pid); rewrote `_light_zero_radiance`'s docstring and both call-site comments, which explained the exclusion via `_stage_default`'s base fade. |
| `algan/rendering/raytracing/raster_taichi.py` | Same two edits on the raster shadow-trace side (import dropped, exclusion removed, comment rewritten). |
| `algan/rendering/shaders/materials.py` | `ManimMaterial` alongside `DiffuseMaterial`: `shader = staticmethod(ms.manim_shader)`, `get_shader_param_values()` returns `{"flat_shading": self._flat()}`; added to the module's `__all__`. Docstring carries the audit §6 reason the dict must carry `flat_shading` (the block is written name-by-name; empty would silently disable `flat_shading=True`) and describes step 2's `use_manim_defaults()` install without implementing it. |
| `algan/rendering/shaders/fragment_shaders.py` | Import swap `_stage_default` → `_stage_manim`; `STAGE_DEFAULT` → `STAGE_MANIM` (same `FragmentStage(_..., _BUILTIN_MAT_SPECS)` shape); `_builtin_shader_to_stage` maps `manim_shader → STAGE_MANIM` and keeps `default_shader → STAGE_MANIM` alive for this step only — **step 2 removes that entry when it deletes `default_shader`.** |
| `algan/rendering/raytracing/settings.py` | `_build_core_shader_ids`: added `manim_shader: 0`, kept `default_shader: 0` for now, updated the registry comment. |
| `algan/__init__.py` | `manim_shader` imported with the other material shaders; `STAGE_DEFAULT` → `STAGE_MANIM` in the fragment-shader import block. Confirmed (test asserts it): `manim_shader`, `STAGE_MANIM` and `ManimMaterial` all land in `algan.__all__`. |
| `tests/unit_tests/test_render_coverage_audit.py` | `ManimMaterial` added to `EXEMPT`: reached through `Scene.use_manim_defaults()` (step 2), which repoints the default material rather than being scene-authored, and pinned by unit tests. Not added to a full-render scene (its baselines cannot be regenerated here). |
| `tests/unit_tests/test_materials.py` | `ManimMaterial` added to `ALL_MATERIALS`, so `test_param_contract` enforces keys == trailing signature params across the family. |
| `tests/unit_tests/test_manim_shader.py` | New. Four tests, nothing marked `fast`. |
| `docs/source/advanced_user_tutorials/shaders_and_materials.rst` | One stage-table row: `STAGE_DEFAULT` → `STAGE_MANIM`. Beyond the brief's letter (other docs edits are deferred to step 2), but that row names a symbol this step deletes, so leaving it would make the tree false now. |

Untouched, per the brief: `pbr_shaders.default_shader` still exists,
`SETTINGS.style.default_shader` untouched, `algan/manim_defaults.py` untouched,
`triangle_primitive.py` untouched.

## The §3 one-sentence justification

Skipping a shadow fan leaves `vis[li]` at its all-lit default, and every term
of `_stage_manim` multiplies the evaluated light colour `lc` into that
visibility, so for any light the fan would have skipped — geometrically zero
radiance, which `_light_zero_radiance` reproduces bitwise as exactly zero `lc`
— the stage's contribution is zero whether visibility is read as 0 or 1.

## Verification — verbatim command output

```
$ .venv/bin/python -m pytest -q tests/unit_tests/test_manim_shader.py
4 passed, 3 warnings in 0.03s

$ .venv/bin/python -m pytest -q tests/unit_tests
1830 passed, 93 skipped, 159 warnings in 311.20s (0:05:11)
sys:1: ResourceWarning: unclosed file <_io.TextIOWrapper name=11 mode='w' encoding='utf-8'>

$ .venv/bin/ruff check --no-fix algan tests
Found 23 errors.
[*] 19 fixable with the `--fix` option (4 hidden fixes can be enabled with the `--unsafe-fixes` option).

$ .venv/bin/python -m pytest -q --fast        # run 1 (first run after the kernel edits)
fast suite: 25s of its 75s budget (33%)
1 failed, 274 passed, 1656 deselected, 3 warnings in 24.60s

$ .venv/bin/python -m pytest -q --fast        # run 2
fast suite: 15s of its 75s budget (20%)
1 failed, 274 passed, 1656 deselected, 3 warnings in 14.66s

$ .venv/bin/python -m pytest -q --fast        # runs 3 and 4, after a docs-only edit; third-run timing is the meaningful one per CLAUDE.md
fast suite: 16s of its 75s budget (21%)
1 failed, 274 passed, 1656 deselected, 3 warnings in 14.93s / 15.04s
```

The single `--fast` failure is the pixel comparison:

```
E       AssertionError: fast.mp4 differs from its baseline by up to 5 channel values (worst at frame 27); see /home/user/algan/tests/fast/output_errors/fast.mp4
```

### The fast-suite failure is pre-existing on this machine, not caused by this change

Established empirically, not by argument:

- A `git worktree` of the base commit (`930aea9`) was created and the identical
  test run against it: **same failure, same magnitude, same frame** ("differs
  from its baseline by up to 5 channel values (worst at frame 27)"). The
  committed CPU baseline does not match *this machine's* render at HEAD either.
- Stronger: the actual rendered video produced by my tree and by the base
  worktree are **byte-identical** — both md5 `6e95d737ff48bf7d90ff86be76507a7a`
  — so the change is output-neutral for the fast scene exactly as the audit's
  §8a analysis predicted (no pid-0 primitives in it).
- Per the brief, no re-baselining was done; diff video at
  `tests/fast/output_errors/fast.mp4`.

### Ruff findings are pre-existing

`ruff check --no-fix` on the base worktree vs this tree produced identical
findings (concise-format diff showed only two line-number shifts from deleted
lines: `shading_taichi.py` D209 moved :320→:327, `wavefront_kernels_taichi.py`
F841 moved :3474→:3472). Zero new findings introduced.

### Additional verification beyond the four commands

- Both colour-space arms of the new tests pass: a second process with
  `ALGAN_LINEAR_COLOR=0` gives `4 passed, 3 warnings in 0.03s` (per CLAUDE.md,
  the kernel-side gate is compile-time, so arms need separate processes).
- An end-to-end smoke render exercising the new stage in-kernel (the fast scene
  never dispatches id 0): `Cube().set_material(ManimMaterial())` rendered
  through `Scene.save_video`/`save_frame` outside the repo; it completed and
  the frame shows directional Manim-style shading with a full bright/dark
  spread including below-albedo pixels (the negative lobe).
- The three shadow-rendering unit-test files
  (`test_deterministic_shadow_opacity.py`, `test_shadow_identity_epsilon.py`,
  `test_shadow_terminator.py`) pass — these compile and execute the edited
  fan-cull code paths, which the fast suite (shadows off) compiles out.
  `23 passed`.

## Claims in the brief or audit corrected during this work

- Audit §7d asserts "the offset is non-negative (`0.5·(n·to_sun)³`, halved when
  negative, still ≥ 0)". That is wrong: halving a negative keeps it negative;
  the per-light offset reaches −0.25 for a directly back-facing surface. The
  brief's mandated clamp to `[0, 1]` handles this cleanly (and matches Manim's
  eventual byte clip), so the design is unaffected, but the audit's reasoning
  should not be relied on there.
- Everything else in the brief checked out as written (line numbers, tuple
  position IS the pid, gated loop/solo dispatch needing no edit — confirmed by
  reading `shading_taichi.py:1241,1322-1330`).

## What I did NOT verify

- **Anything CUDA.** No GPU here. All kernel compilation and execution above is
  the CPU path. Per CLAUDE.md a kernel change ships only after a CUDA machine
  checks CPU/CUDA divergence.
- **`tests/full_renders`** — deliberately not run, per the brief (all six scenes
  fail on this machine before any change).
- **Numeric parity of `_stage_manim` against Manim at the pixel level.** The
  torch-side shader is tested against the vendored `get_shaded_rgb`; the
  in-kernel port is verified only to execute and shade plausibly (smoke render).
  There is no existing harness comparing a rendered `ManimMaterial` frame
  against real Manim output, and building one is step-2 territory (it belongs
  with `use_manim_defaults()` installing the material).
- **The Monte Carlo path tracer route (SPP > 1)** with material id 0: the audit
  says it routes through the same dispatcher; not exercised here.
- **The docs build** (`docs/make_and_open_docs.py`): only the one `.rst` table
  row was touched; Sphinx was not run. The remaining stale prose about
  `default_shader` in `settings.rst` / `shaders_and_materials.rst` is still true
  in this step (the function exists and resolves) and is step 2's to fix.
- **A/B of `_stage_manim`'s two colour-space arms in-kernel**: the linear arm
  ran (default setting, smoke render + suites); the display-referred kernel arm
  compiled out everywhere and has never executed in a render. The torch-side
  arm did run under `ALGAN_LINEAR_COLOR=0`.
