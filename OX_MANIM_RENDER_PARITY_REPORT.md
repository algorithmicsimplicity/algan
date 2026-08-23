# OX_MANIM_RENDER_PARITY_REPORT — step 3: the in-kernel Manim stage matches
# Manim at the pixel level

Step 3 per `ox_brief_impl3.md`, closing the gap step 1's report named: the
torch-side `manim_shader` was already pinned against the vendored
`get_shaded_rgb`, but `_stage_manim` — the in-kernel port almost every rendered
frame actually goes through — had only been smoke-tested. It is now verified by
a **render**: real frames through material id 0, individual pixels compared
against the vendored Manim function. Nothing committed; no shading code touched.

**Result: exact parity — 0 bytes of deviation on every asserted pixel and their
entire 5×5 neighbourhoods, under both the default analytic-raster route and
(checked as an extra) the classic wavefront route.**

## What was built

`tests/unit_tests/test_manim_shader_render.py` (new, not marked `fast`; kept
separate from `test_manim_shader.py`'s pure-torch tests so a failure names its
layer). Two tests, each one render with two pixel assertions:

1. `test_in_kernel_stage_matches_get_shaded_rgb` — explicit rig stated by the
   test itself: one white intensity-1 `PointLight` (decay 0, distance 0) at
   `(4, 3, 2)`; camera moved to `OUT * 20`, `look_at(ORIGIN)`, vertical fov
   framing 8 world units at the origin plane (the same numbers Manim's rig
   happens to produce); black background.
2. `test_use_manim_defaults_reaches_bare_solids` — solids with **no material of
   their own**, rendered after `Scene.use_manim_defaults()`; expected values use
   the vendored function at the installed light position,
   `from_manim_coordinates(MANIM_LIGHT_SOURCE)`. This is the end-to-end check
   that step 2's default really reaches geometry: the face mobs carry
   `shader is None`, so the Manim shading the pixels verify can only have come
   from `SETTINGS.style.default_material` at primitive-build time.

Shared scene: axis-aligned `Cube` solids (`Polyhedron` → proven-outward normals,
`one_sided` shading declared — confirmed in source at `shapes_3d.py:1560` and
`shapes_2d.py:551` before relying on it; their zero vertex normals hit the
kernel's degenerate-normal fallback to the geometric normal), albedo
`(0.72, 0.38, 0.13)`, full opacity, resolution 160×160, `anti_alias_level=1`,
`samples_per_pixel=1`, tonemapping off, exposure 1, shadows off, bloom removed
(`post_processes=()`), FXAA off. Each test asserts:

- the centre pixel of the centred cube's front face,
- a second face turned away from that rig's light (halved negative lobe),
- every pixel of a 5×5 window around each, against its own prediction,

with expected bytes = `floor(255 * clamp(get_shaded_rgb(...), 0, 1) + 0.5)`
from the **imported vendored function**, tolerance 1 byte/channel (never
needed — see below), plus direction checks (lit face above authored colour,
turned-away face below it) that would catch a flipped row order or lobe sign.

### How "the world position of the sampled pixel" is stated exactly

The harness replicates the renderer's own two formulas rather than trusting
prose: forward projection per `raster_pipeline.precompute_triangle_projection`
(`sx = u·half_h + half_w`, `sy = v·half_h + half_h` off the render basis), and
per-pixel rays by inverting `raytrace_kernels_taichi._generate_ray`. The kernel
evaluates a fully covered fragment's barycentrics exactly at the pixel centre
(`raster_taichi._ss_pixel`: sample pattern sums to zero; centroid offset is zero
at full coverage), so the expected value is taken at the ray-through-pixel-centre
intersection with the face plane — computed in float64 from the Scene's own
camera tensors. Coverage guards assert every checked pixel's four corner rays
land strictly inside the face (and, for the off-centre second faces, strictly on
one side of the triangulation diagonal), so no other surface or background
contributes.

Two harness-design findings along the way, both fixed in the harness (not in
shading code):

1. **Odd render resolution breaks analytic replication by up to half a pixel.**
   The tracer passes `float(width // 2)` / `float(height // 2)` as the
   half-screen extents (`tracer.py`, render call sites). At 161 px that is 80.0
   instead of 80.5, so every kernel-side sample sits systematically off the
   analytic projection — worth only ~0.5 bytes where shading is flat, but 2–7
   bytes where it is steep. A near-field-light probe fit the offset to
   (+0.5 px, +0.5 px) at 0.52 bytes RMS before the cause was found in source.
   The test uses an even resolution, where the conventions coincide exactly.
   *Engine observation, not changed here:* on odd resolutions every render's
   sampling grid differs from the projection tables' convention by up to half a
   pixel — self-consistent, so invisible without an external reference, but it
   is a real subtlety for any future pixel-exact harness.
2. **A quad face's f0–f2 triangulation diagonal passes through the face
   centre** by construction, so the centred-face pixel is always composited
   from two partial fragments evaluated at their owned-sample centroids. On a
   view-perpendicular face that blend lands within a byte of the centre value
   (measured), so the main assertion keeps its literal "centre of the front
   face" placement; the second faces sample well inside one triangle under a
   strict single-triangle guard.

Also observed while building this (engine behaviour, unchanged): the
`location=` constructor kwarg does not move a `Polyhedron`'s already-built face
geometry (the mob's location attribute moves; the rendered faces stay at the
origin-relative vertices) — `.move_to()` inside `Off()` propagates correctly,
which is what the test uses, and is what every scene in the repository does.

## Measured numbers (rendered vs expected, printed by the tests)

```
[front face of the centred cube (turned away from the light)]        test 1
  pixel             : col=80 row=79      sampled world pos : [0.0225 0.0225 -2]
  rendered RGB      : [168, 81, 17]      expected RGB      : [168.0, 81.0, 17.0]
  |difference|      : [0.0, 0.0, 0.0]    worst over 5x5    : 0
[front face of the forward cube (in front of the test light)]
  pixel             : col=130 row=63     sampled world pos : [3.03 0.99 4.0]
  rendered RGB      : [221, 135, 71]     expected RGB      : [221.0, 135.0, 71.0]
  |difference|      : [0.0, 0.0, 0.0]    worst over 5x5    : 0

[front face of the centred cube (lit by Manim's light)]              test 2
  pixel             : col=80 row=79      sampled world pos : [0.0225 0.0225 -2]
  rendered RGB      : [208, 121, 57]     expected RGB      : [208.0, 121.0, 57.0]
  |difference|      : [0.0, 0.0, 0.0]    worst over 5x5    : 0
[top face of the low cube (turned away from Manim's light)]
  pixel             : col=60 row=128     sampled world pos : [-1.005155 -2.5 0.618557]
  rendered RGB      : [177, 90, 27]      expected RGB      : [177.0, 90.0, 27.0]
  |difference|      : [0.0, 0.0, 0.0]    worst over 5x5    : 1
```

The tolerance is 1 byte per channel; the largest deviation measured anywhere,
on either route, was **1 byte** (single channel of one neighbourhood pixel).

## Verification — verbatim command output

```
$ .venv/bin/python -m pytest -q tests/unit_tests/test_manim_shader_render.py
2 passed, 3 warnings in 6.12s

$ .venv/bin/python -m pytest -q tests/unit_tests
1840 passed, 93 skipped, 159 warnings in 346.36s (0:05:46)
sys:1: ResourceWarning: unclosed file <_io.TextIOWrapper name=11 mode='w' encoding='utf-8'>

$ .venv/bin/python -m pytest -q --fast
fast suite: 17s of its 75s budget (23%)
1 failed, 274 passed, 1666 deselected, 3 warnings in 17.02s

$ .venv/bin/ruff check --no-fix algan tests
Found 23 errors.
[*] 19 fixable with the `--fix` option (4 hidden fixes can be enabled with the `--unsafe-fixes` option).
```

- The single `--fast` failure is verbatim the pre-existing one established in
  steps 1–2 against a base-commit worktree:
  `fast.mp4 differs from its baseline by up to 5 channel values (worst at frame
  27)`. Same message, same magnitude, same frame; not re-baselined.
- Ruff: 23 errors, identical count to steps 1–2; `--output-format=concise`
  grep for the new file returns nothing — zero findings introduced. The new
  file also passes `ruff format --check`.
- Full-suite arithmetic: 1838 + 2 new = 1840 passed ✓.
- Extra, beyond the brief: both render tests also pass with
  `ALGAN_HYBRID_RASTER=0` (classic supersampled-wavefront route, separate
  process per CLAUDE.md's compile-time-gate rule) — `2 passed, 3 warnings in
  3.76s`.

## Docs row fix

`docs/source/advanced_user_tutorials/shaders_and_materials.rst`: the
`STAGE_MANIM` row now reads "per light, a ``0.5 * (n . to_light) ** 3`` offset,
halved when the surface faces away from the light and scaled by the light's
colour" — replacing the stale achromatic/no-factor/no-halving wording, all
three corrections the brief called for.

## What I did NOT verify

- **Anything CUDA.** No GPU here; every render above is the CPU path.
- **The Monte Carlo path tracer (SPP > 1)** routing through the same
  dispatcher: not exercised.
- **Both sRGB arms in-kernel.** Renders ran under the default linear working
  space. The display-referred arm compiles out behind `ti.static` and would
  need its own process (`ALGAN_LINEAR_COLOR=0`); the torch-side shader is
  tested against the vendored function in both arms by `test_manim_shader.py`,
  but no render ran in the other arm.
- **A comparison against real Manim output files.** Expected values come from
  the vendored Manim function evaluated at matched world positions, which is
  what "matching Manim" means here; no Manim installation was driven to produce
  reference images (and Manim has no single-pixel still-render contract to
  diff against for arbitrary rigs).
- **Non-default exposure/tonemap/coloured-light/multi-light rigs.** Parity is
  claimed for the white intensity-1 rig both tests state (and Manim's own
  installed rig); the coloured-light tint is the documented strict
  generalisation, unpinned by renders.
- **Grazing-incidence faces beyond the harness lesson.** Faces viewed at
  grazing angles compress many world units into a pixel; there, sub-byte
  evaluation detail is worth whole bytes and point comparisons are
  ill-conditioned. The probe that established this (flat-gradient light ⇒ exact
  0 deviation on such a face) is described above but lives only in this
  report's history, not as a committed benchmark.
