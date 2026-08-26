# Algan test suites

Three directories, two suites. Always run them with the project venv — the
system Python has no taichi. Commands below are written as `<venv-python>`;
`CLAUDE.md` defines the per-platform interpreter path, and `uv run python`
works on either.

## The fast suite — run this one

```bash
<venv-python> -m pytest -q --fast
```

**This is the suite to run after every change.** It is a hand-picked set of
tests marked `fast`, and **nothing else runs** — a test with no marker is
outside it, including one added five minutes ago. It prints where it landed
against its budget when it finishes:

```
fast suite: 21s of its 75s budget (28%)
```

That figure moves by a good fraction between runs, because most of the render's
cost is Taichi specialising a kernel and that is sensitive to what the process
did beforehand. It is reported rather than enforced for exactly that reason: a
timing assertion here would be a flake. The 21 s above is measured, on a warm
cache on a 4-vCPU CPU-only container; the budget leaves room for the render to
pay a kernel compile, which is where most of the time goes when it does.

Before curation this suite was *everything not marked `slow`*: 910 of the
suite's 1038 collected tests, 112–147 s on CUDA, and every new test anywhere
joined it automatically. It is now **191**, listed below.

**Wait for the third consecutive run before believing the number.** Taichi's
cost is per kernel variant, charged to whichever test reaches it first, so any
change that touches a kernel makes the next run pay a cold compile that has
nothing to do with the suite's size. A measured sequence immediately after
adding two small kernels ran 194 s → 160 s → 112 s, and only the last is the
suite. Taking a marker off a test on the strength of run 1 evicts coverage to
pay for a compile that would not have happened again.

Give it no path, so it uses the `testpaths` from `pyproject.toml`.

### What is in it

The suite answers one question: **is Algan still working?** Not *is every
feature intact* — that is the full suite's job. So it holds the mechanisms that
every animation, every Mob and every render route through, and one end-to-end
render:

| Marked `fast` | Why it is in |
| --- | --- |
| `test_timeline_overlap.py`, `test_timeline_state_query.py`, `test_active_timeline_materialization.py` | Recording, the per-row state query and materialization at frame times. Nothing reaches the screen except through these. |
| `test_lifecycle.py` | Spawn/despawn lifespans, which decide whether a Mob exists in a frame at all. |
| `test_rate_functions.py` | Every animation is evaluated through one of these curves. |
| `test_mob_movement.py`, `test_mob_orientation.py`, `test_parent_child_basis.py`, `test_mob_layout.py` | Transforms, the path a move traces, parent→child propagation, and screen-relative placement (which composes the bounding box, the basis and the camera). |
| `test_scene_containment.py` | Which Scene owns a Mob and which managers that Scene owns — where every recorded event lands. |
| `test_settings_api.py` | `SETTINGS` is read live by every subsystem. |
| `test_ux_regressions.py` (per test, not the whole module) | The front door: `save_video`/`save_frame` and what they leave behind, contexts, `Group`, the star exports, and the errors users hit. It is a catch-all file, so its tests are marked one by one — see below. |
| `test_mesh_identity.py` (per test) | Per-triangle surface identity (`tri_obj`), which the analytic-AA resolve groups fragments by and the scene merge offsets per primitive. It is a contract between the mob side (what declares a surface), the merge and two kernel walks, so a change to batching, to a composite mob's `get_render_primitives`, or to the merge's offsetting breaks it from elsewhere. Pure tensor assertions — no render, no Taichi. |
| `test_batched_surface_mobs.py` (per test) | Indexing into a packed Mob -- `Mob.__getitem__`, `_set_data_sub_inds` and `__len__` resolving one member's rows out of a shared batch. Every packed Mob (all `Text`, every point cloud) depends on it, and it breaks from the Mob base or the timeline rather than from the surface code. Only the two indexing tests are marked; the equivalence tests beside them fail only when `surface.py` changes. |
| `test_bezier_group_runs.py` (per test) | That the vectorized bezier build is still *reached*, and that splitting a clashing group into runs leaves the merged collection unchanged. The batchability gate reads timeline internals (`mob_id_to_inds`, `ranges_for`, `parent_batch_sizes`), so a timeline or packing change can silently send every circuit back to the per-actor build -- output stays correct and nothing else notices. Two tensor-only tests are marked; the frame comparison beside them renders and is not. |
| `tests/fast/test_fast_render.py` | One real scene, rendered and compared pixel-wise. The only thing in the loop that can see a renderer regression, and most of its wall clock. |

### What is not in it, and where that is covered instead

Everything else — which is most of the suite, on purpose. The general shape:
a test that only breaks when *its own* subsystem is worked on does not belong
here, because whoever works on that subsystem runs its file (or the full suite)
anyway.

| Left out | Why | Covered by |
| --- | --- | --- |
| Per-feature behaviour: the indication animations, `become`, `wave_color`, `NumericDisplay`, materials, fragment shaders, neural nets, the Manim compatibility layer, glTF/FBX import | Breaks when that feature is worked on, not when anything else is | Its own file, plus `tests/full_renders/` for pixels |
| Renderer internals: PN tessellation, surface autotune, bezier sampling, BVH refit, wavefront compaction, the frag-pid gate | Same, one subsystem each | Its own file |
| Batch sizing, the arena, texture memory, post-processing memory | Cheap, but they only move when their own module does; the fast render exercises the real path | `test_memory_model.py`, `test_render_batch_sizing.py`, `test_manual_memory.py`, … |
| Repo-consistency audits: doc examples, render coverage, the env-var registry, Manim mobject parity | They fail when you *add* public API, which the full suite and CI catch before a push | `test_doc_examples.py`, `test_render_coverage_audit.py`, `test_environment.py`, `test_manim_mobject_parity.py` |
| The other six render scenes | ~2 minutes each | `tests/full_renders/` |
| Brute-force tracer references | Taichi specialises a megakernel per test's geometry; tens of seconds each | `tests/unit_tests/test_raytracing_unit.py` |
| PN surfaces *in the render* | ~20 s of kernel specialisation on its own | `test_logical_pn_tessellation.py` and `test_surface_autotune.py` behaviourally; `full_renders/solids_and_camera` for pixels |
| Shadows, refraction, glow, Monte Carlo, glTF, camera moves | Another kernel variant or tracer path each | `tests/full_renders/` |

**CI is not the fast suite.** It names `tests/unit_tests tests/fast` as paths
and runs everything under them, `fast`-marked or not (see the comment in
`.github/workflows/test.yaml`). The fast suite is a development loop; CI can
afford twelve minutes and should keep spending them.

## The full suite

```bash
<venv-python> -m pytest -q
```

Everything, about twelve minutes on CUDA. Run it before pushing, after touching
the renderer, and whenever the fast suite's coverage table above says the thing
you changed lives here.

| Directory | What it protects | Cost |
| --- | --- | --- |
| `tests/unit_tests/` | Behaviour that can break without raising: the timeline, the transform hierarchy, settings, batch sizing, materials, the public API surface. | ~90 s |
| `tests/fast/` | One dense scene, rendered and compared pixel-wise: the renderer coverage the fast loop can afford. | ~50 s |
| `tests/full_renders/` | What the renderer actually draws across six dense scenes, compared pixel-wise against checked-in baselines. | ~12 minutes on CUDA |

## Adding a test: does it belong in the fast suite?

Default to **no**, and leave it unmarked. That is not a demotion — the full
suite runs it, CI runs it, and whoever touches the code it covers runs it.

Mark it `fast` only if **a change somewhere else in the codebase is liable to
break it**. Ask which file someone would have to edit for this test to start
failing: if the honest answer is "the one it tests", it is a feature test and
it stays out. If the answer is "the timeline, or the Mob base, or the Scene, or
anything that records or materializes state" — the machinery every animation
runs through — then it is a canary worth paying for on every change.

Two supporting rules:

- **Cheap is not a reason.** A test that costs 5 ms and never fires except when
  its own module changes still costs a reader's attention and a maintainer's
  judgement; hundreds of those are exactly what this suite was curated out of.
- **The budget is a first-come constraint.** If the suite reports itself over,
  the fix is to take the marker off whatever went in most recently, not to
  raise the number. Raising it is a deliberate trade of loop time for coverage.

One trap when weighing cost: with Taichi, **the cost is per kernel variant, not
per test**, and it is charged to whichever test reaches that variant first.
Excluding the slowest Monte Carlo test in `test_raytracing_unit.py` did not save
its seven seconds — it moved them to the next test that needed the same kernel.
A group that shares a kernel joins or stays out together, which is why that
module is discussed as a whole in its docstring.

Marking is per module (`pytestmark = pytest.mark.fast`, with a comment saying
why the module is in) or per test, whichever matches how coherent the file is.
A module-level mark means the *whole file* is core, new tests included, so use
it only where that is true. It is true of the timeline and transform files —
each is about one mechanism, so a new test in them is the same kind of test.
It is not true of `test_ux_regressions.py`, which is a catch-all: within an hour
of this suite being curated, a merge added two tests to it, one of them a
six-second subprocess its author had deliberately kept out of the loop, and a
module-level mark would have enrolled both. That file is marked per test.

There is no `slow` marker any more: it meant "outside the fast suite", which is
now what every unmarked test already is. `--strict-markers` is on, so a stale
`@pytest.mark.slow` fails collection rather than sitting there doing nothing,
and `test_fast_suite_curation.py` fails if a `fast` marker appears that the
table above does not explain.

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

**Never baseline the first render on a fresh machine — render twice and keep
the second.** The first run of a scene containing `Tex`/`MathTex` populates the
persistent Manim SVG geometry cache (`algan_cache/`), and its glyph
antialiasing is not what every subsequent run produces. Measured while
re-baselining `text_and_media` on a fresh container: the cold run differed from
the two warm runs after it by up to **18 channel values** across 100 of 182
frames — nine times the tolerance — confined to `MathTex` glyph edges, while
runs two and three were byte-identical to each other and the warm output sat
closer to the CUDA baseline than the cold one did. Baseline the cold render and
the suite fails on the very next run, on the same machine, for no reason anyone
would think to look for. The other five scenes were bit-stable cold-to-warm, so
this is specifically a Tex-geometry-cache effect.

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
fixture; `../agent_guidance/memory_perf.md` covers how to pick one, and
`benchmarks/_split_determinism_check.py` measures a scene's own run-to-run
floor.

On Windows, run render work **one process at a time** — a killed or timed-out
run orphans children that keep the output mp4s locked.

## The unit suite

Organised by subsystem. The files worth knowing about (★ = in the fast suite):

- ★ `test_timeline_*`, `test_active_timeline_materialization.py`, `test_lifecycle.py` —
  the recording/replay engine: overlapping edits, replay windows, the state
  query, spawn lifetimes.
- ★ `test_mob_movement.py`, `test_mob_orientation.py`, `test_mob_layout.py`,
  `test_parent_child_basis.py`, `test_scene_containment.py` — transforms,
  layout, hierarchy, Scene ownership.
- `test_scene_actor_registration.py` — the actor registration that decides
  whether a *particular* composite's geometry reaches the renderer at all. One
  test per composite that once got it wrong, so it stays out of the fast suite.
- ★ `test_settings_api.py` — the `SETTINGS` root, its validation and the
  experimental-switch gate. `test_environment.py` covers the environment — how
  `ALGAN_` variables parse, and the rule that the package reaches them only
  through `algan/environment.py`'s accessors, which is what keeps its registry
  of declared names honest — and is an audit, so it is not in the fast suite.
- ★ `test_ux_regressions.py`, `test_rate_functions.py` — the authoring surface
  users touch most. `test_materials.py`, `test_fragment_shaders.py` and
  `test_indication_animations.py` are per-feature and stay out.
- `test_memory_model.py`, `test_render_batch_sizing.py`, `test_manual_memory.py` —
  batch sizing and the arena. Cheap, and they guard a component that silently
  degrades rather than failing — but they only move when their own module does,
  and the fast suite's render exercises the real path.
- `test_raytracing_unit.py` — brute-force references for the tracer. Expensive.
- `test_render_truncations.py` — the instrument on the render path's four fixed
  ceilings: what each reports, that a ceiling warns once per render rather than
  once per batch, and two scenes built to exceed one. A feature test — it breaks
  when the instrument does — so it stays out of the fast suite.
- `test_render_coverage_audit.py` — keeps the render suite honest (above).
- `test_fast_suite_curation.py` — keeps the `fast` markers and the membership
  table above in step, so the suite cannot grow without someone saying why.
- `test_doc_examples.py` — keeps `docs/` honest against `algan/` (below).

### The documentation examples

`test_doc_examples.py` extracts every Python block in `docs/source` and checks it
in three tiers, because the blocks do not all support the same checking:

| Tier | Covers | Catches | Runs |
| --- | --- | --- | --- |
| `test_doc_example_uses_public_api` | every block, statically | a name or setting the docs still use after it was renamed or removed | with the suite, ~1 s |
| `test_doc_example_authors_without_error` | blocks that are complete scripts, with rendering stubbed | anything that raises while *authoring*: wrong constructor arguments, a value of the wrong width, a method that is gone | with the suite, ~35 s |
| `test_doc_example_renders` | the same scripts, rendered at `SMOKE_TEST` | render-time failures — an updater that raises once it is evaluated over a batch of frames | opt in with `ALGAN_RUN_DOC_RENDERS=1` |

None of the three is in the fast suite. They are an audit of `docs/` against
`algan/`: they fail when public API is renamed or removed, which is a thing you
find out about before pushing rather than on every save.

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

The render tier is gated on an environment variable, not merely left out of the
fast suite, and the distinction matters: being outside `--fast` does nothing for
CI, which names its paths explicitly instead of passing that flag (see the
comment in `.github/workflows/test.yaml`). That gap is how this tier took a
runner down once: before the texture-timeline fix it peaked at **14.7 GB** and
was OOM-killed part way through.

With that fixed it renders 77 examples in about **two minutes at 2.3 GB**, so the
gate is now a time budget rather than a memory cliff — two minutes is more than
the fast suite's whole allowance, and CI would pay it on every run. Worth
revisiting if the render-time coverage is wanted in CI; measure a cold Taichi
cache first, since the number above is from a warm one. Run it locally with:

```bash
ALGAN_RUN_DOC_RENDERS=1 <venv-python> -m pytest -q tests/unit_tests/test_doc_examples.py -k renders
```

### Recording a known defect

Three bugs were once pinned here as strict `xfail`s — a parent `Group.move()`
desynchronizing a compatibility Mob's backing Manim object, `Indicate`'s scale
pulse writing a `scale_coefficient` row instead of a basis, and the point-cloud
family having no `get_render_primitives` at all. All three are fixed, and the
tests that recorded them now assert the working behaviour: see
`test_a_parent_group_move_keeps_the_backing_mobject_in_step`,
`test_indicate_grows_the_mob_in_the_middle`, and
`test_point_cloud_mob_produces_render_primitives`. The point clouds have left
the coverage audit's `EXEMPT` list and appear in `shapes_and_timeline`.

There are no `xfail`s in the suite today. If you need to record a new defect
rather than fix it, a strict `xfail` is still the way — it keeps the bug visible
and fails the suite when it starts passing, which is what tells you to turn the
test around and drop the marker. It is a built-in marker, so `--strict-markers`
has nothing to say about it; only project-specific markers need the entry in
`pyproject.toml`.

## Legacy

`tests/test_files/` and `tests/run_test.py` are the previous render suite — one
scene per concept, with its own baselines in `tests/expected_outputs_*`. The
six scenes above supersede it; it is already outside `testpaths` in
`pyproject.toml` and can be deleted once you are happy with the new baselines.
