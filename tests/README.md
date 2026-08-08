# Algan test suites

Three directories, two suites. Always run them with the project venv — the
system Python has no taichi.

## The fast suite — run this one

```bash
.venv/Scripts/python.exe -m pytest -q --fast
```

**This is the suite to run after every change.** It is everything *not* marked
`slow`, and it holds itself to two minutes so it stays inside a development
loop (measured 88–106 s on CUDA over consecutive runs, of which the render is
40–47 s). It prints where it landed against that budget when it finishes:

```
fast suite: 88s of its 120s budget (73%)
```

That figure moves by a good fraction between runs, because most of the render's
cost is Taichi specialising a kernel and that is sensitive to what the process
did beforehand. It is reported rather than enforced for exactly that reason: a
timing assertion here would be a flake.

Give it no path, so it uses the `testpaths` from `pyproject.toml`.

What it covers: the whole behavioural suite — the timeline and its replay, the
transform hierarchy, layout, actor registration, settings, batch sizing and the
arena, materials, the animations, the public API surface — plus **one real
render, compared pixel-wise** (`tests/fast/`), which is the only thing in the
loop that can see a renderer regression.

What it gives up, and where that is covered instead:

| Left out | Why | Covered by |
| --- | --- | --- |
| The other four render scenes | ~2 minutes each | `tests/full_renders/` |
| Brute-force tracer references | Taichi specialises a megakernel per test's geometry; tens of seconds each | `tests/unit_tests/test_raytracing_unit.py` |
| PN surfaces *in the render* | ~20 s of kernel specialisation on its own | `test_logical_pn_tessellation.py` and `test_surface_autotune.py` behaviourally; `full_renders/solids_and_camera` for pixels |
| Shadows, refraction, glow, Monte Carlo, glTF, camera moves | Another kernel variant or tracer path each | `tests/full_renders/` |
| Point clouds, the `import algan` subprocess check | Re-confirm a known defect / a second interpreter | The full suite |

## The full suite

```bash
.venv/Scripts/python.exe -m pytest -q
```

Everything, about twelve minutes on CUDA. Run it before pushing, after touching
the renderer, and whenever the fast suite's coverage table above says the thing
you changed lives here.

| Directory | What it protects | Cost |
| --- | --- | --- |
| `tests/unit_tests/` | Behaviour that can break without raising: the timeline, the transform hierarchy, settings, batch sizing, materials, the public API surface. | ~60 s (~90 s including the `slow` ones) |
| `tests/fast/` | One dense scene, rendered and compared pixel-wise: the renderer coverage the fast loop can afford. | 40–47 s |
| `tests/full_renders/` | What the renderer actually draws across five dense scenes, compared pixel-wise against checked-in baselines. | ~10 minutes on CUDA |

## What `slow` means

`slow` marks a test as **outside the fast suite** — it is a budget decision, not
a description. Renders, pixel comparisons and anything that costs more than
about a second earn it.

Mark a new test `slow` when the fast suite reports itself over budget. Prefer
marking the *newly added* expensive test over an old one: the budget is a
first-come constraint, and silently evicting existing coverage to fit a new
test is how a fast suite stops being worth running.

One trap when choosing what to mark: with Taichi, **the cost is per kernel
variant, not per test**, and it is charged to whichever test reaches that
variant first. Excluding the slowest Monte Carlo test in
`test_raytracing_unit.py` did not save its seven seconds — it moved them to the
next test that needed the same kernel. A group that shares a kernel has to
leave together or not at all, which is why that module is marked at module
level.

## The full-render suite

`tests/full_renders/scenes/` holds **five dense scenes**, not one scene per
concept. Each one packs a whole subsystem into a single render while keeping
everything laid out in labelled, non-overlapping rows, so a regression reads as
a diff in one column rather than as a mystery.

| Scene | Covers |
| --- | --- |
| `shapes_and_timeline` | 2-D bezier circuits (fills, non-convex triangulation, inward borders, analytic AA), all four animation contexts and their nesting, rate functions, every indication animation, `become`, updaters, `wave_color`, `draw_border_then_fill`, `NumericDisplay`, the spawn/despawn lifecycle, and the raw primitives underneath it all. |
| `solids_and_camera` | Analytic PN surfaces vs. flat meshes side by side, the Platonic solids, `Surface`, `Arrow3D`/`Line3D`/`Dot3D`/`ConvexHull3D`, parent-and-child transforms in one block, the movement helpers, screen-relative layout, and every camera motion. |
| `materials_and_lighting` | All nine `Mesh*Material` classes and the presets, animated material parameters, all six light types, shadows, glow through bloom and tonemapping, opacity, and the reflection/refraction paths of the wavefront tracer. |
| `text_and_media` | `Text`/`MarkupText`/`Tex`/`MathTex`/`Paragraph`/`Code` and the triangulated variants, `write()`, per-glyph addressing, Tex-to-Tex morphing, `ImageMob` (textured and per-pixel), glTF import with PBR and normal maps, and composed fragment shaders. |
| `manim_compat_and_plots` | `Axes(...).plot(...)` and the other delegated builders, `NumberPlane`, `BarChart`, matrices and tables, `Graph`, braces, the Manim-flavoured shapes, and `ApplyMatrix`/`ApplyComplexFunction`/`Homotopy`/`MoveAlongPath`/`AnimatedBoundary`. |

### Scene conventions

A scene file **records** an animation; it never renders one. The harness owns
the `Scene`, the settings and the comparison. Scene files may import only
`from algan import *` (plus `torch`, which building raw primitives genuinely
needs), and must carry a module docstring saying what they are for.
`tests/unit_tests/test_render_coverage_audit.py` enforces all of that, and also
fails if a public renderable class, material or light appears in no scene and
is not listed in its `EXEMPT` dict with a reason.

The harness makes `tests/full_renders/` the working directory while a scene
runs, so `assets/world_map.jpg` resolves; and it snapshots `SETTINGS` around
each scene, so a scene may turn a renderer feature on for itself
(`materials_and_lighting` turns shadows on).

`assets/textured_icosphere.glb` is a compact UV-mapped model with embedded
albedo, metallic/roughness and normal textures. Keeping the fixture small is
deliberate: it keeps the render inside the arena without weakening importer or
material coverage.

## The fast suite's render

`tests/fast/scene.py` is a sixth scene under the same conventions, kept apart
from the five above so that the full-render suite and its coverage audit stay
what they are. Its docstring is worth reading before editing it: it is shaped
by the kernel-variant cost, which is why it is one scene rather than several
and why it contains no `Surface` geometry.

## Re-baselining

Both render suites are re-baselined by rendering with the baselines writable,
then **looking at the result** before committing:

```bash
ALGAN_UPDATE_FULL_RENDER_BASELINES=1 .venv/Scripts/python.exe -m pytest tests/full_renders -q
```

```bash
ALGAN_UPDATE_FAST_BASELINE=1 .venv/Scripts/python.exe -m pytest tests/fast -q
```

Both variables are read by the harnesses rather than by the package, so
`import algan` warns that it does not recognise them. That is expected.

Frames are compared channel-wise with a tolerance of 2 by the
`assert_video_matches_baseline` fixture in `tests/conftest.py`, which both
suites share so they cannot drift apart on tolerance. That tolerance is not
slack for sloppiness: torch's CPU rate-function evaluation rounds differently
depending on the materialization window, so byte-identity across re-windowed
state is unattainable. A failure writes a diff video to the suite's
`output_errors/`.

On Windows, run render work **one process at a time** — a killed or timed-out
run orphans children that keep the output mp4s locked.

## The unit suite

Organised by subsystem. The files worth knowing about:

- `test_timeline_*`, `test_active_timeline_materialization.py`, `test_lifecycle.py` —
  the recording/replay engine: overlapping edits, replay windows, the state
  query, spawn lifetimes.
- `test_mob_*`, `test_parent_child_basis.py`, `test_scene_*` — transforms,
  layout, hierarchy, and the actor registration that decides whether geometry
  reaches the renderer at all.
- `test_settings_api.py`, `test_environment.py` — the `SETTINGS` root, its
  validation, the experimental-switch gate, and the startup-only environment.
- `test_materials.py`, `test_fragment_shaders.py`, `test_indication_animations.py`,
  `test_rate_functions.py` — the authoring surface users touch most.
- `test_memory_model.py`, `test_render_batch_sizing.py`, `test_manual_memory.py` —
  batch sizing and the arena. These are cheap and guard a component that
  silently degrades rather than failing.
- `test_raytracing_unit.py` — brute-force references for the tracer. Slow.
- `test_render_coverage_audit.py` — keeps the render suite honest (above).

### Known defects pinned as `xfail`

Three real bugs are recorded as strict `xfail`s rather than deleted tests, so
they stay visible and the suite tells you when they are fixed:

- `test_manim_compat_movement.py` — a parent `Group.move()` leaves a
  compatibility Mob's backing Manim object behind, so the next delegated call
  (`rotate`, `scale`, `set`, …) teleports the Mob back to where the parent
  found it.
- `test_indication_animations.py` — `Indicate`'s scale pulse writes a
  `scale_coefficient` timeline row directly instead of going through the
  property setter that turns a scale into a basis, so only its colour flash is
  visible.
- `test_point_cloud_rendering.py` — the whole point-cloud family (`DotCloud`,
  `PointCloudDot`, `TrueDot`, `PGroup`) is exported and constructs its points,
  but defines no `get_render_primitives`, so it can never draw anything.

A strict `xfail` fails the suite if it starts passing. That is deliberate: when
one of these is fixed, the test tells you to remove the marker (and, for the
point clouds, to move them out of the audit's `EXEMPT` list and into a scene).

## Legacy

`tests/test_files/` and `tests/run_test.py` are the previous render suite — one
scene per concept, with its own baselines in `tests/expected_outputs_*`. The
five scenes above supersede it; it is already outside `testpaths` in
`pyproject.toml` and can be deleted once you are happy with the new baselines.
