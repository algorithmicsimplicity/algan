# Algan test suites

Three directories, two suites. Always run them with the project venv — the
system Python has no taichi. Commands below are written as `<venv-python>`;
`CLAUDE.md` defines the per-platform interpreter path, and `uv run python`
works on either.

## The fast suite — run this one

```bash
<venv-python> -m pytest -q --fast
```

**This is the suite to run after every change.** It is everything *not* marked
`slow`, and it holds itself to two and a half minutes so it stays inside a
development loop (measured 112–147 s on CUDA over consecutive warm runs, median
~135 s, of which the render is about 50 s). It prints where it landed against
that budget when it finishes:

```
fast suite: 134s of its 150s budget (89%)
```

That figure moves by a good fraction between runs, because most of the render's
cost is Taichi specialising a kernel and that is sensitive to what the process
did beforehand. It is reported rather than enforced for exactly that reason: a
timing assertion here would be a flake.

**Wait for the third consecutive run before believing the number.** Taichi's
cost is per kernel variant, charged to whichever test reaches it first, so any
change that touches a kernel makes the next run pay a cold compile that has
nothing to do with the suite's size. A measured sequence immediately after
adding two small kernels ran 194 s → 160 s → 112 s, and only the last is the
suite. Marking a test `slow` off run 1 evicts coverage to pay for a compile that
would not have happened again.

The budget itself was 120 s until the behavioural suite grew from 419 to 466
unit tests and every run started reporting itself over. Raising it is a
deliberate trade of loop time for coverage, not a formality: if the number stops
meaning anything, the suite creeps.

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
<venv-python> -m pytest -q
```

Everything, about twelve minutes on CUDA. Run it before pushing, after touching
the renderer, and whenever the fast suite's coverage table above says the thing
you changed lives here.

| Directory | What it protects | Cost |
| --- | --- | --- |
| `tests/unit_tests/` | Behaviour that can break without raising: the timeline, the transform hierarchy, settings, batch sizing, materials, the public API surface. | ~60 s (~90 s including the `slow` ones) |
| `tests/fast/` | One dense scene, rendered and compared pixel-wise: the renderer coverage the fast loop can afford. | ~50 s |
| `tests/full_renders/` | What the renderer actually draws across six dense scenes, compared pixel-wise against checked-in baselines. | ~12 minutes on CUDA |

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

`tests/full_renders/scenes/` holds **six dense scenes**, not one scene per
concept. Each one packs a whole subsystem into a single render while keeping
everything laid out in labelled, non-overlapping rows, so a regression reads as
a diff in one column rather than as a mystery.

| Scene | Covers |
| --- | --- |
| `complex_hierarchy_become` | Arbitrary hierarchy-to-hierarchy `become`: primitive-aware pairing across different tree shapes, cubic-bezier/Surface/mesh conversion, collapsed-target growth and surplus-source collapse for unequal leaf counts, Image-only dissolve, and parent transforms after target-tree reconstruction. |
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

`tests/fast/scene.py` is a seventh scene under the same conventions, kept apart
from the six above so that the full-render suite and its coverage audit stay
what they are. Its docstring is worth reading before editing it: it is shaped
by the kernel-variant cost, which is why it is one scene rather than several
and why it contains no `Surface` geometry.

## Baselines are per device

Each render suite keeps one baseline directory per device —
`expected_outputs_cuda/` and `expected_outputs_cpu/` — and the harness picks
between them with `torch.cuda.is_available()`. A machine with no baseline
directory for its device renders the scene and then skips the comparison, so
**a suite that reports itself green on a new device may not have compared
anything**; check for skips before believing it.

Both sets are checked in. They are *not* interchangeable, and the differences
between them are larger than the tolerance by design:

- **PN surfaces** (`Sphere`, `Cylinder`, `Cone`, `Torus`, `Surface`) differ
  across their interiors, because the subdivision-level criterion kernel runs
  under Taichi's `fast_math` and flips borderline tessellation levels
  differently per backend. Measured at up to 8% of a frame's pixels.
- **Silhouettes and specular highlights** differ by up to ~75 channel values on
  edge pixels, from float ordering.
Text used to differ far more than either — up to ~230 — and that one was never
a device difference at all. `Text` defaults to `font=""`, which Pango resolves
through fontconfig, so the glyph advances changed with whatever the machine had
installed. `Tex`/`MathTex` never had the problem: they go through LaTeX and
`dvisvgm` to outlines and match across devices at zero shift, which is what
identified fonts as the cause.

## Baselines are per machine too, which decides what CI runs

Per *device* understates it: the full-render baselines do not survive a change
of CPU either. Measured, not assumed — a GitHub Actions `ubuntu-latest` runner
rendered these scenes against baselines produced on another CPU:

| Scene | Max deviation | |
| --- | ---: | --- |
| `tests/fast` | 0 | matched |
| `shapes_and_timeline` | 0 | matched |
| `text_and_media` | 29 | failed |
| `complex_hierarchy_become` | 44 | failed |
| `manim_compat_and_plots` | 50 | failed |
| `solids_and_camera` | 53 | failed |
| `materials_and_lighting` | 204 | failed |

against a tolerance of 2. The split is not arbitrary: everything that matched is
built from 2-D circuits and flat triangle meshes, and everything that moved
carries PN surfaces, shadows, refraction or glTF — which is what
`pn_criterion_kernel` under `fast_math` predicts, since which tessellation
levels sit on a boundary depends on the CPU evaluating the criterion.

So **CI runs `tests/unit_tests` and `tests/fast`**, the two that are portable,
and `test_full_render_scene` skips itself when `CI` is set. Run it anyway with
`ALGAN_RUN_FULL_RENDERS=1` — on the machine whose baselines these are, or to
re-measure the spread.

Neither obvious shortcut is worth taking. Raising the tolerance would have to
reach ~204 to pass, which is far past where it stops catching regressions;
re-baselining on a runner just moves the failure onto the developer's machine.
The real fix, if this suite should ever gate CI, is making the level criterion
independent of the host CPU — probably dropping `fast_math` on those kernels —
which is a renderer change that moves every baseline including CUDA.

One thing this measurement did confirm: the fast scene contains `Text` and
`Tex` and matched exactly on a machine with a different font set, which is the
evidence that vendoring the fonts works.

## Fonts are vendored, not borrowed

`tests/assets/fonts/` holds the **Algan Test Sans** and **Algan Test Mono**
faces, and `tests/conftest.py` registers them with Pango before any scene runs.
Every `Text`, `MarkupText` and `Paragraph` call in a scene names one of them
(`font=FONT`), and `Code` names the mono family through its `paragraph_config` —
its own default is the `"Monospace"` fontconfig alias, which is host-dependent
in exactly the same way.

This is what stops a container image with a different font set from shifting
every glyph and failing the suite as if the renderer had regressed. It is
enforced, not left to review: `test_scene_text_pins_a_vendored_font` fails on
any Text-like call in a scene that does not pass `font=`, because one unpinned
call reintroduces the drift for the whole scene.

The faces are the Liberation fonts with their name tables rewritten. Renaming
is deliberate — the SIL OFL reserves the upstream name, and a distinct family
means a system installation of Liberation can never shadow the vendored files.
See `tests/assets/fonts/LICENSE.txt`.

**When adding a scene**, give its text `font=FONT` and declare
`FONT = "Algan Test Sans"` under the star import, as the existing scenes do.

## Re-baselining

Baselines are re-baselined **for the device you are on** — the environment
variables below write to whichever `expected_outputs_<device>/` matches the
current machine, so re-baselining on CPU cannot repair a CUDA baseline or vice
versa.

Both render suites are re-baselined by rendering with the baselines writable,
then **looking at the result** before committing:

```bash
ALGAN_UPDATE_FULL_RENDER_BASELINES=1 <venv-python> -m pytest tests/full_renders -q
```

```bash
ALGAN_UPDATE_FAST_BASELINE=1 <venv-python> -m pytest tests/fast -q
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

The second thing the tolerance absorbs is **split pixels**. A pixel that spawns
three or more branches — reflective or refractive geometry under analytic AA,
where each covered pixel takes several sub-pixel reflection taps — sums those
branches into `pix_accum` with `ti.atomic_add` in GPU scheduling order, and
float addition is not associative, so such a scene renders slightly differently
every run. The effect is bounded at one channel value by the `u8` truncation in
the compositor (measured: `|d| = 1` on tens of samples out of 165M, absorbed
entirely by the video encoder), which is why the render suites are not flaky
from it. It does mean a scene like that cannot be a *byte-identical* A/B parity
fixture; `AGENTS_DETAILED.md` covers how to pick one, and
`benchmarks/_split_determinism_check.py` measures a scene's own run-to-run
floor.

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
- `test_doc_examples.py` — keeps `docs/` honest against `algan/` (below).

### The documentation examples

`test_doc_examples.py` extracts every Python block in `docs/source` and checks it
in three tiers, because the blocks do not all support the same checking:

| Tier | Covers | Catches | In `--fast` |
| --- | --- | --- | --- |
| `test_doc_example_uses_public_api` | every block, statically | a name or setting the docs still use after it was renamed or removed | yes, ~1 s |
| `test_doc_example_authors_without_error` | blocks that are complete scripts, with rendering stubbed | anything that raises while *authoring*: wrong constructor arguments, a value of the wrong width, a method that is gone | yes, ~14 s |
| `test_doc_example_renders` | the same scripts, rendered at `SMOKE_TEST` | render-time failures — an updater that raises once it is evaluated over a batch of frames | no — opt in with `ALGAN_RUN_DOC_RENDERS=1` |

Most documented code is a *fragment* — a few lines operating on an undefined
`mob` — which can never be a runnable scene without inventing scaffolding around
it. That is why tier 1 exists and why it only flags **capitalized** free names:
classes and constants are what get renamed, and lowercase names are the reader's
own variables.

A block that is deliberately not runnable opts out with an reStructuredText
comment on the line above it, which never reaches the rendered page:

```rst
.. algan-doc-check: skip -- needs an asset that does not ship with the docs

.. code-block:: python
```

Use it for anti-examples showing what raises, Manim-side snippets in a migration
comparison, and examples needing assets or system packages the repository does
not carry. For a block broken by a bug that is already being worked on, add it to
`KNOWN_BROKEN` in that module with a reason instead, so it is skipped loudly
rather than quietly deleted.

The render tier is gated on an environment variable rather than on `slow` alone,
and the distinction matters: `slow` only drops a test from `--fast`, and CI names
its paths explicitly instead of passing that flag (see the comment in
`.github/workflows/test.yaml`). Rendering all ~82 documented scripts in one
process peaked at **14.7 GB** and was OOM-killed, which is how it took a runner
down before the gate was added. Run it deliberately, on a machine with headroom:

```bash
ALGAN_RUN_DOC_RENDERS=1 <venv-python> -m pytest -q tests/unit_tests/test_doc_examples.py -k renders
```

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
six scenes above supersede it; it is already outside `testpaths` in
`pyproject.toml` and can be deleted once you are happy with the new baselines.
