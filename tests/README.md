# Algan test suites

Two suites, with different jobs:

| Suite | What it protects | Cost |
| --- | --- | --- |
| `tests/unit_tests/` | Behaviour that can break without raising: the timeline, the transform hierarchy, settings, batch sizing, materials, the public API surface. | Seconds to a few minutes |
| `tests/full_renders/` | What the renderer actually draws, compared pixel-wise against checked-in baselines. | ~10 minutes on CUDA |

Always run them with the project venv — the system Python has no taichi:

```bash
.venv/Scripts/python.exe -m pytest tests/unit_tests -q
```

```bash
.venv/Scripts/python.exe -m pytest -q --skip-slow
```

The second command is the fast feedback loop: it skips the GPU renders and the
pixel comparisons, which are the only tests marked `slow`. Give it no path so it
uses the `testpaths` from `pyproject.toml` — passing `tests` collects the legacy
`tests/test_files/` scenes too, which *render on import* and collide with the
unit-test module names.

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

### Re-baselining

A change that legitimately alters output is re-baselined by rendering with the
baselines writable, then **looking at the result** before committing:

```bash
ALGAN_UPDATE_FULL_RENDER_BASELINES=1 .venv/Scripts/python.exe -m pytest tests/full_renders -q
```

Frames are compared channel-wise with a tolerance of 2. That is not slack for
sloppiness: torch's CPU rate-function evaluation rounds differently depending on
the materialization window, so byte-identity across re-windowed state is
unattainable. A failure writes a diff video to `tests/full_renders/output_errors/`.

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
