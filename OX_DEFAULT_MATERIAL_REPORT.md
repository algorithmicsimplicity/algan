# OX_DEFAULT_MATERIAL_REPORT — step 2: delete `default_shader`, default 3-D Mobs to a Diffuse material

Step 2 of 2 per `ox_brief_impl2.md`, building on step 1 (`manim_shader`,
`_stage_manim` at material id 0, `ManimMaterial`). Nothing was committed; all
work is in the tree. The goal as stated in the brief is met: **a 3-D Mob that
sets no material of its own now renders as `DiffuseMaterial()`
(`lambert_shader`), and `Scene.use_manim_defaults()` makes it render as
`ManimMaterial()` instead — while 2-D content stays unlit.**

## What changed and why

| File | Change |
|---|---|
| `algan/rendering/shaders/pbr_shaders.py` | `default_shader` deleted entirely (definition + docstring mention). Module docstring now says what actually shades an unconfigured Mob (`DiffuseMaterial`). The now-unused `normalize` import went with it (it was `default_shader`'s only user). |
| `algan/rendering/shaders/material_shaders.py` | New named constant **`SHADER_FIXED_PARAM_COUNT = 9`** beside the calling-convention paragraph, which now cites it instead of `pbr_shaders.default_shader`. Its docstring names the pinning test. |
| `algan/animatable_base/mob_materials.py` | `set_shader` slices a shader's extra parameters off after `SHADER_FIXED_PARAM_COUNT` instead of `len(inspect.signature(default_shader).parameters)`; import dropped. |
| `algan/rendering/raytracing/primitives.py` | `_ordered_shader_param_values` uses the constant; for a parameter the mob does not carry it now falls back to `self.default_material_params` *before* the shader signature's default. `_pack_material` seeds the material block from `default_material_params` first, with the mob's own registered values overwriting them by name (see below for the two-pass shape this forced). Import of `default_shader` replaced by the constant. |
| `algan/rendering/primitives/triangle_primitive.py` | The fallback point: `shader = SETTINGS.style.default_shader` becomes "take `.shader` off `SETTINGS.style.default_material` and keep its `get_shader_param_values()` on the primitive". `default_material_params: dict = {}` is a class attribute so every construction path (including the `triangle_collection` merge branch) stays valid; the merge branch carries `[0]`'s mapping along. A `None` default material still degrades to unlit id 1, exactly as the old `None` default did. **`get_batch_identifier` gained a trailing flag** separating bare (seeded) from mob-authored-parameter primitives — see "the batch-identifier change" below. |
| `algan/settings/style_settings.py` | `default_shader` field renamed to `default_material`; `__post_init__` rejects a non-`None` value without a `.shader` attribute with `AlganConfigurationError`, duck-typed (no rendering import at module scope, per audit Q2). Module docstring updated. No alias left. |
| `algan/__init__.py` | Install line is now `SETTINGS.style.set(default_material=DiffuseMaterial())`; `default_shader` removed from the `pbr_shaders` import block (drops out of `__all__` automatically — verified: not in `algan.__all__`, `DiffuseMaterial` present). |
| `algan/rendering/shaders/fragment_shaders.py` | `default_shader → STAGE_MANIM` entry removed from `_builtin_shader_to_stage` (the entry step 1 kept alive). |
| `algan/rendering/raytracing/settings.py` | `default_shader: 0` removed from `_build_core_shader_ids` (its import too); registry comment no longer mentions it. |
| `algan/mobs/image_mob.py` | `ImageMob.__init__` calls `self.set_shader(null_shader)` after `super().__init__` (children exist; before spawn), so a picture plane renders unlit instead of picking up lit-Lambert shading. Class/module docstrings updated ("lights like any other 3-D Mob" is no longer true). |
| `algan/mobs/triangulated_bezier_circuit.py` | `TriangulatedBezierCircuit.__init__` calls `self.set_shader(null_shader)` after `add_children`, so triangulated circuit fills (glyph fills via `TexTriangulated`/`TextTriangulated`, function-plot curves, morph conversions) stay unlit like their untriangulated twins. Same applied at the `tile_region2` site (:669). Module docstring updated. |
| `algan/manim_defaults.py` | Installs `ManimMaterial()` instead of `basic_material_shader`. Comment rewritten per the brief: Manim applies no lighting to flat 2-D VMobjects and Algan's 2-D content never consults this setting (§4); what the setting reaches is 3-D geometry, where Manim *does* shade via `get_shaded_rgb`, which is what `ManimMaterial` reproduces — each half attributed to its engine. |
| `algan/scene.py` / `algan/rendering/shaders/materials.py` | `use_manim_defaults`' `shading` parameter docstring rewritten (the old "unlit flat colour as the default shading" sentence reads false now); `ManimMaterial`'s "will install" became "installs". |
| Tests | `test_manim_defaults.py`: fixture save/restore + assertion moved to `default_material`; assertion strengthened to `isinstance(..., ManimMaterial) and .shader is manim_shader`. `test_render_coverage_audit.py`: `DiffuseMaterial` EXEMPT reason updated from "legacy material API" to the default material. `test_materials.py`: new `test_fixed_param_count_constant_matches_the_signature` asserts `_NUM_BASE_PARAMS == ms.SHADER_FIXED_PARAM_COUNT`; stale `default_shader` comment replaced. **New `tests/unit_tests/test_default_material.py`** (7 tests): imported default, bare-Solid packs `_MID_LAMBERT`, `ImageMob` and a `TriangulatedBezierCircuit` fill pack `_MID_UNLIT`, plain `Circle` never reaches the triangle fallback (asserted on primitive kind, not a shader attribute), non-material default raises, configured default material's parameters reach the packed block and an explicit mob value wins. None marked `fast`. |
| Docs | `shaders_and_materials.rst`: new "What an unconfigured Mob gets" subsection in the materials part; Vertex Shaders section rewritten around `SHADER_FIXED_PARAM_COUNT` with no `default_shader` references; `STAGE_MANIM` table row cross-referenced to `Scene.use_manim_defaults`. `settings.rst`: ``default_material`` entry rewritten. |

## Brief claims checked and corrected

- **`triangulated_bezier_circuit.py:669` is dead code.** It sits inside
  `tile_region2` (:569), which has **no callers anywhere** in `algan/`,
  `tests/`, `benchmarks/` or `docs/` (grepped). The brief presents :669 and
  :1014 as the two production sites of the route; only :1014 (inside
  `TriangulatedBezierCircuit.__init__`) is live, and it covers every class the
  brief lists (`plots.Arrow` subclasses it at plots.py:41; `FunctionPlotMob`'s
  curve, `TexTriangulated`/`TextTriangulated` and the morph conversions all go
  through it). I still applied `null_shader` at :669 with a comment saying the
  helper is currently uncalled, so a revival cannot reintroduce lit circuit
  fills — but the live change is the one in `__init__`.
- **Native `plots.Arrow` cannot be constructed at all — pre-existing.**
  `Arrow.__init__` passes an svgelements `Path` to `TriangulatedBezierCircuit`,
  which crashes at `path[..., :2]` (`TypeError`). Verified identical on a
  `git worktree` of HEAD (`Arrow FAILS AT HEAD TOO: TypeError`), and
  `algan/mobs/plots.py` is byte-identical to HEAD in this tree. The new test
  therefore uses `TriangulatedBezierCircuit` directly with a hand-built closed
  path (the brief allows "another TriangulatedBezierCircuit"), which also keeps
  it free of font/LaTeX dependencies.
- Everything else in the brief and audit checked out as written (line numbers,
  Q1 inventory, Q2 round-trip machinery being field-name driven, the four
  `null_shader` properties, §8a's fast-scene material coverage).

## §2 parameter-agreement finding (verified, machine-checked)

`DiffuseMaterial().get_shader_param_values()` returns `emissive=(0,0,0)`,
`emissive_intensity=1.0`, `flat_shading=0.0`, `env_map_intensity=1.0`.
`raytracing/settings.py::_MAT_SLOTS` assigns them slots `(0,3)`, `(3,1)`,
`(10,1)`, `(11,1)`; `_MAT_DEFAULTS` holds `0,0,0 / 1.0 / 0.0 / 1.0` there.
**They agree exactly**, so whether or not a no-material mob's parameters are
packed explicitly, it renders identically as `DiffuseMaterial()`. Re-verified
at the end of the work with a direct script comparing each value against
`_MAT_DEFAULTS` at its assigned slot.

## §4 batching-key finding

The check the audit did not cover: `Surface.get_render_primitives_batched`
(surfaces/surface.py:703) does **not** build one primitive for several peers.
Its caller `_build_deferred_surfaces` (render_loop.py:1820) groups surfaces by
`(grid_width, grid_height, grid.location.shape)`, but the grouping only shares
the geometry tensor pass — the function returns **one primitive per surface**,
each built by that surface's own `_build_render_primitive` with
`shader=self.shader` (surface.py:3073). Two peers resolving to different
shaders therefore cannot land in one primitive through this path; downstream,
primitives still group by `get_batch_identifier`, whose shader-id component
separates different shaders. **No fix needed.**

### Related hazard found and fixed: mixed bare/authored collections

Making bare mobs carry `lambert_shader` means a bare mob and an explicit
`MeshLambertMaterial` mob now share a batch identifier — a collection the old
code never formed (bare mobs used to carry `id(default_shader)`). That matters
because the collection merge transposes members' parameter rows column-wise:
members with different parameter-list widths would be silently truncated to
the shortest (a bare mob registers zero parameters). Concretely, a scene with a
bare Sphere and an explicit red-emissive `MeshLambertMaterial` Sphere would
have silently dropped the latter's emissive. Fix: `get_batch_identifier` now
appends a flag distinguishing "mob-authored parameters" from "default-seeded",
so the two kinds batch separately; within each kind the merge arithmetic is
width-consistent. Pixel-neutral (grouping affects launch structure, not
pixels), and confirmed by the byte-identical fast-scene output.

## The `_pack_material` seed mechanics

Seeds are written as constants *after* the block is sized from the mob's own
per-frame rows, and *before* those rows overwrite them by name. An earlier
draft folded seeds into the same pairs list and broke
`test_logical_pn_tessellation.py`: a seed flattened from `emissive=[3]` inflated
`Tm` (the block's time-row count, computed as `max(v.shape[0])`) from 1 to 3,
which then rejected `one_sided`'s broadcast. Seeds contribute nothing to `Tm`;
with an empty mapping every path is byte-for-byte the old code (same pairs, same
order, same Tm).

## Verification — verbatim command output

```
$ .venv/bin/python -m pytest -q tests/unit_tests
1838 passed, 93 skipped, 159 warnings in 308.92s (0:05:08)
sys:1: ResourceWarning: unclosed file <_io.TextIOWrapper name=11 mode='w' encoding='utf-8'>

$ .venv/bin/ruff check --no-fix algan tests
Found 23 errors.
[*] 19 fixable with the `--fix` option (4 hidden fixes can be enabled with the `--unsafe-fixes` option).

$ .venv/bin/python -m pytest -q --fast        # run 1
fast suite: 15s of its 75s budget (20%)
1 failed, 274 passed, 1664 deselected, 3 warnings in 15.12s

$ .venv/bin/python -m pytest -q --fast        # run 2
fast suite: 15s of its 75s budget (20%)
1 failed, 274 passed, 1664 deselected, 3 warnings in 14.72s
```

(The ruff summary line above is the final count; the full finding list was
diffed against a base-commit worktree rather than counted — see below.)

### The single `--fast` failure is pre-existing, and the render is byte-identical

- Step 1 established the same failure on a `git worktree` of the base commit:
  same message, same magnitude, same frame. It recurred identically here:
  `AssertionError: fast.mp4 differs from its baseline by up to 5 channel values
  (worst at frame 27)` — the committed CPU baseline does not match this
  machine's render, before or after this change.
- Stronger, per §8a: the rendered video produced under this change is
  **byte-identical** to the base commit's render — md5
  `6e95d737ff48bf7d90ff86be76507a7a`, the exact hash recorded in
  `OX_MANIM_SHADER_REPORT.md` for both the step-1 tree and the base worktree —
  on both runs. So nothing in this change reached a mob that already had a
  material, exactly as the audit's §8a analysis predicted. No re-baselining was
  done; diff video at `tests/fast/output_errors/fast.mp4`.

### Ruff findings are pre-existing

`ruff check --no-fix --output-format=concise algan tests` on this tree vs a
base-commit worktree produced **identical findings** (diff empty; 23 errors in
both, none in files this change touched). During the work ruff flagged five
genuine new findings in my new/edited files (unused `normalize` import left by
the deletion; import order + three compound assertions in the new test file);
all were fixed before the final run.

### Additional verification beyond the four commands

- Full unit-suite result above includes the fixed-in-flight
  `test_logical_pn_packs_only_regular_flat_triangle_geometry` (see seed/Tm note)
  and the new `test_default_material.py` (7 tests).
- `tests/unit_tests/test_doc_examples.py` passes: the rewritten docs blocks
  resolve against the live `algan` namespace.
- Structural Sphinx build (`docs/make_and_open_docs.py --skip-examples
  --no-open`): build succeeded with 2 warnings, **identical to the base
  worktree's 2 warnings** (one autosummary warning at generated
  `reference/algan.rendering.shaders.materials.rst:50`, one missing-graphviz
  `dot`). None introduced by these edits.
- End-of-work script check: `default_shader` absent from `algan.__all__`,
  `SETTINGS.style.default_material` prints `DiffuseMaterial()`, and the four
  parameter values match `_MAT_DEFAULTS` at their `_MAT_SLOTS` slots.
- Final grep: `rg -n "default_shader" algan/ tests/ benchmarks/ docs/source/`
  returns no matches. Remaining mentions live only in dated documents
  (`OX_LIGHTING_AUDIT.md`, the briefs, this and prior reports).

## What I did NOT verify

- **Anything CUDA.** No GPU here; everything above is the CPU path. Per
  CLAUDE.md, a change like this ships only after a CUDA machine checks
  divergence — though this change touches no kernel code, so kernel-side
  divergence risk is nil by construction.
- **`tests/full_renders`** — deliberately not run, per the brief (all six
  scenes fail on this machine before any change; handled separately). Four of
  the six scenes are predicted by audit §8a to move once baselines are
  regenerated on a baseline-matching machine
  (`complex_hierarchy_become`, `solids_and_camera`, `shapes_and_timeline`,
  `text_and_media`).
- **A pixel render of a bare 3-D Mob.** The unit tests pin the packed material
  id and parameter block, not a rendered frame; no harness here compares a
  rendered `DiffuseMaterial` default frame to anything. The Manim-parity render
  harness (rendering a material-less 3-D mob under `use_manim_defaults`)
  suggested by step 1 remains unbuilt.
- **The Monte Carlo path tracer (SPP > 1)** with the new defaults: the packed
  ids flow the same way, but nothing here exercised that route.
- **Windows behaviour** (single-process render locking etc.) — Linux only.
- **`benchmarks/_default_shading_check.py` was not executed.** It already
  refers to `SETTINGS.style.default_material` (written in anticipation of this
  step) and should now run as documented; running it produces images that need
  eyes, which is a human step.
