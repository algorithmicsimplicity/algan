# Algan test suites

Four directories, three suites. Always run them with the project venv — the
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
| `test_easings.py` | Every animation is evaluated through one of these curves. |
| `test_mob_movement.py`, `test_mob_orientation.py`, `test_parent_child_basis.py`, `test_mob_layout.py` | Transforms, the path a move traces, parent→child propagation, and screen-relative placement (which composes the bounding box, the basis and the camera). |
| `test_mob_reparenting.py` | That the hierarchy is read when an animation is *recorded*, not when it plays: a parent transform resolves the descendant union to rows and the event keeps them, so re-parenting afterwards redirects the next animation without rewriting the last one. That contract lives in the timeline (`modify_attribute_and_record`, `replay_inds`) and in two version-keyed descendant caches, none of which are in this file — a mutation that forgets to bump a version does not error, it silently drops a member from a transform. Tensor-only, no render. |
| `test_scene_containment.py` | Which Scene owns a Mob and which managers that Scene owns — where every recorded event lands. |
| `test_settings_api.py` | `SETTINGS` is read live by every subsystem. |
| `test_torch_compile.py` | `SETTINGS.computing.torch_compile` and the `compiled` decorator every fused pipeline function goes through. The switch is read at every call, so a change to how the section validates or restores it reaches the whole render path; and the fallback contract — a compile failure warns once and runs eagerly, the function's own errors propagate — is what keeps the switch from ever failing a render. Dynamo's `eager` backend only: no Inductor build, no render, about two seconds. |
| `test_ux_regressions.py` (per test, not the whole module) | The front door: `save_video`/`save_frame` and what they leave behind, contexts, `Group`, the star exports, and the errors users hit. It is a catch-all file, so its tests are marked one by one — see below. |
| `test_spatial_constants.py` | `IN`/`OUT` are aliases of `INWARD`/`OUTWARD` so a script may rebind the short names, which only holds while Algan's own source reads the long ones. An AST walk over the whole package, so any module that reaches for `IN` or `OUT` trips it — including one added today. Cheap: no Taichi, no render. |
| `test_mesh_identity.py` (per test) | Per-triangle surface identity (`tri_obj`), which the analytic-AA resolve groups fragments by and the scene merge offsets per primitive. It is a contract between the mob side (what declares a surface), the merge and two kernel walks, so a change to batching, to a composite mob's `get_render_primitives`, or to the merge's offsetting breaks it from elsewhere. Pure tensor assertions — no render, no Taichi. |
| `test_batched_surface_mobs.py` (per test) | Indexing into a packed Mob -- `Mob.__getitem__`, `_set_data_sub_inds` and `__len__` resolving one member's rows out of a shared batch. Every packed Mob (all `Text`, every point cloud) depends on it, and it breaks from the Mob base or the timeline rather than from the surface code. Only the two indexing tests are marked; the equivalence tests beside them fail only when `surface.py` changes. |
| `test_bezier_group_runs.py` (per test) | That the vectorized bezier build is still *reached*, and that splitting a clashing group into runs leaves the merged collection unchanged. The batchability gate reads timeline internals (`mob_id_to_inds`, `ranges_for`, `parent_batch_sizes`), so a timeline or packing change can silently send every circuit back to the per-actor build -- output stays correct and nothing else notices. Two tensor-only tests are marked; the frame comparison beside them renders and is not. |
| `test_neural_net_idle.py` | The batched idle-updater fast path (`ALGAN_BATCHED_IDLE_UPDATER`) must write exactly what the per-mob loops it replaced write. It reads rows through the same timeline machinery as any updater (`trace_updater_mob_access`, `_compact_index`, `modify`), so a change to replay or materialization can silently desync the two arms; the marked test materializes one window under each arm on a freshly built net and requires bit-equal buffers. Tensor-only, no render; the file's other two tests stay unmarked. |
| `test_public_api_surface.py` | `algan.__all__` is assembled by rules over every exported module, not written by hand, so it moves from anywhere: adding a module-level helper to an exported module publishes a new name, and moving a class between the native and `algan.manim` surfaces withdraws one. The snapshot catches both at the commit that causes them rather than at release. It also holds the native/`algan.manim` boundary — no Manim adapter may shadow a native class — which a rename in either direction can break. Pure Python, no Taichi, no render: about half a second. |
| `test_mps_friendly.py` (per test) | MPS-friendly mode substitutes a float32 accumulator, an int32 reduction or a log-step scan wherever Metal cannot run the wide one, and the *call sites* are all over the renderer while the substitutions live in one module. So the marked tests are the cheap half — the flag's resolution, the dtype selectors, the scan against `cummax`/`cummin`, and an AST walk that fails if any renderer module reaches for `torch.float64`, `ti.f64` or `cummax` directly. That walk is the one that trips on a change made elsewhere: a new float64 accumulator anywhere under `algan/rendering/` is an Apple GPU that aborts, and nothing else in the loop would say so. No Taichi, no render. The kernel-variant comparisons and the end-to-end render beside them are unmarked. |
| `test_kernel_control_flow.py` | A `continue` under a compile-time gate — `if ti.static(...)` or a statically unrolled `for ... in ti.static(range(n))` — emits a bare `ContinueStmt` and leaves every statement after it in an already-terminated block, which is **invalid SPIR-V**. Same shape of argument as the walk above: it trips on a kernel written anywhere under `algan/`, and the consequence is an Apple GPU that will not build a pipeline, which nothing else in the loop can see (LLVM executes the invalid module correctly, so the CPU and CUDA suites are both green on it). `../algan/rendering/DESIGN_mps_support.md` §1.2c is what it cost the once. An AST walk: no Taichi, no render, about half a second. |
| `test_arena_args.py` (one test) | That every launch site of an arena-converted kernel passes the argument count its wrapper packs from. The list a site must match (`call_params`) lives in the kernel module, the sites live in the tracer, the raster pipeline and the path tracer, and nothing in Python connects them — so adding a kernel parameter breaks the sites from elsewhere, and the wrapper's own arity check only fires when that particular launch renders. Only the static arity test is marked; the prologue/spec comparisons beside it move when their own kernel does, and the Metal buffer-count walk parses the whole package. Source reading, no Taichi, no render: about half a second. |
| `test_weight_floor_exit.py` (one test) | That the argument *index* the file's render arms assert on still names `weight_floor_exit`. The arms are unmarked by design — a render each — so when `vis_lights` joined the parameter list after the gate, the stale count reached master and only the 25-minute CI run saw it. This reads the count and the index straight off the kernel module for nothing. It does not replace the arms: it cannot see whether either gate variant compiles, which is what they are for. |
| `test_shape_anchors.py` | Where a shape's `location` sits, which is the point every rotation and every scale is taken about, and the origin the texture grid is laid out from. It is derived from geometry in two places (`_circuit_location_and_basis` for circuits, `Polyhedron` for the flat solids) and consumed by the transforms, the frame and the layout — so it moves from the Mob base, from the circuit frame, or from either derivation, and an anchor that drifts off a shape's centroid does not raise: it makes the shape orbit a point beside itself, which only shows up in pixels. Construction and tensor assertions, no render. |
| `test_taichi_warmstart.py` (one test, not the module) | Only `test_the_memoization_is_live_on_this_compiler`, and only because the thing it guards is invisible: the warm-start memoization is version-gated to the compiler internals it patches, so a compiler bump in `pyproject.toml`, a new backend in `taichi_compat`, or an env-var rename turns it off — and a silent no-op reads exactly like a slow machine. That is not hypothetical; it cost ~25 s per render for the length of the Quadrants evaluation (`../taichi_patches/PLAN.md` §6.1). Everything else in the file is a feature test for the memo and stays out. No Taichi init, no render: it reads one module-level string. |
| `test_taichi_fast_launch.py` (one test, not the module) | Only `test_the_dispatcher_is_live_on_this_compiler`, for the same reason as its warm-start twin: the launch-plan dispatcher is version-gated to the compiler's launch internals, and a compiler bump or backend rename turns it off with nothing to show but ~0.2-0.4 ms more per kernel launch. Everything else in the file compiles real kernels to hold the dispatcher's key and fallbacks against the compiler's own instantiation choices, and stays out. No Taichi init, no render: it reads one module-level string. |
| `test_taichi_source_key.py` (one test, not the module) | Only `test_the_index_installs_on_this_compiler`, for the same reason as the warm-start row above: the source-keyed cache index (`algan/utils/taichi_source_key.py`) rests on Quadrants internals — `Kernel._try_load_fastcache`, `src_hasher`, `Program.load_fast_cache` — so a compiler bump turns it off from elsewhere, and an index that silently stands down is a warm frontend that quietly went from ~1 s back to ~12 s. The value rules, the closure walk, the hook control flow and the subprocess renders beside it are feature tests and stay out. No Taichi init, no render: it reads the version gate. |
| `test_taichi_early_return.py` (one test, not the module) | Only `test_the_rewrite_is_live_on_this_compiler`, for the same reason as the two rows above: the early-`return` rewrite for inlined `@ti.func` bodies (`algan/utils/taichi_early_return.py`) wraps one compile-path function per compiler — `FuncBase.get_tree_and_ctx` on Quadrants, `kernel_impl._get_tree_and_ctx` on taichi — and is version-gated to them, so a compiler bump in `pyproject.toml`, a new backend in `taichi_compat`, or an env-var rename turns it off from elsewhere. A silent no-op looks like nothing at all until a shader stage that used to compile fails with the compiler's own "Return inside non-static if/for" message. It reads the gate and compiles one early-return func on the live compiler: about a second, no render. The forty-odd feature tests beside it hold the rewrite against real compilation and stay out. |
| `tests/fast/test_fast_render.py` | One real scene, rendered and compared pixel-wise. The only thing in the loop that can see a renderer regression, and most of its wall clock. |

### What is not in it, and where that is covered instead

Everything else — which is most of the suite, on purpose. The general shape:
a test that only breaks when *its own* subsystem is worked on does not belong
here, because whoever works on that subsystem runs its file (or the full suite)
anyway.

| Left out | Why | Covered by |
| --- | --- | --- |
| Per-feature behaviour: the indication animations, `become`, `wave_color`, `DecimalNumber`, materials, fragment shaders, neural nets, the Manim compatibility layer, glTF/FBX import | Breaks when that feature is worked on, not when anything else is | Its own file, plus `tests/full_renders/` for pixels |
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

It runs those paths twice, on `ubuntu-latest` and on `macos-latest`. Linux
takes the ordinary `auto` probe and lands on the CPU. **macOS is pinned to
`ALGAN_RENDER_DEVICE=cpu`**, because `auto` there resolves to MPS — the runner
does offer one — and Algan does not run on MPS: the raster pipeline allocates
in `float64`, which MPS refuses, and `ti.gpu` on a Mac resolves through Vulkan,
whose SPIR-V builder refuses `f64` in the same kernels. 88 tests failed that
way on the first macOS CI run. Supporting MPS means taking `float64` out of the
raster pipeline and the kernels; until that happens the Mac job tests the CPU
path, which is what makes it a portability check rather than a standing
failure. The `algan check` step ahead of the tests prints the device that came
out, along with whether LaTeX and FFmpeg are on `PATH`.

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
| `tests/path_traced/` | What the `samples_per_pixel > 1` path tracer draws across three small scenes, compared pixel-wise against checked-in baselines. | ~15 s warm |

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
| `shapes_and_timeline` | 2-D bezier circuits (fills, non-convex triangulation, inward borders, analytic AA), all four animation contexts and their nesting, rate functions, every indication animation, `become`, updaters, `wave_color`, `DrawBorderThenFill`, `DecimalNumber`, the spawn/despawn lifecycle, and the raw primitives underneath it all. |
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

## The path-traced suite

`tests/path_traced/scenes/` holds four small scenes rendered through the
`samples_per_pixel > 1` wavefront path tracer (each scene file sets
`samples_per_pixel` itself, and the harness asserts the plan chose the path
tracer). They are deliberately tiny — 128×72, five frames — because the path
tracer pays per (pixel, sample, frame) and behavioural correctness lives in
`tests/unit_tests/test_path_tracer.py`; these baselines exist to catch what
only pixels can see.

| Scene | Covers |
| --- | --- |
| `translucency_and_order` | Deterministic 2-D compositing under PT: same-depth author order, depth-separated overlap, and the closed-shell opacity ring on a rotating translucent solid. |
| `lit_and_shadowed` | NEE direct lighting with shadows, the sampled emitter table, GGX metal, and diffuse colour bleed. |
| `environment_and_refraction` | Environment-map NEE and escape, mirror reflection, and refraction through the nested-IOR stack. |
| `authored_under_many_lights` | The authored-appearance branch past the shadow cap: a toon floor and a Manim box under 24 point lights plus the two direction-less rows, sampling their light rows rather than summing them (roadmap §6a-bis). |

The path tracer promises convergence, not byte-identical frames, but its
accumulation happens to be atomic-free and its sampler is a pure function of
path identity, so at a pinned memory budget the same tolerance applies as
everywhere else. Like the full-render suite the baselines are per machine,
and the suite skips in CI (`ALGAN_RUN_PATH_TRACED=1` overrides).

The committed `expected_outputs_cpu/` set was rendered on a cloud CPU
container. The `expected_outputs_cuda/` set was rendered on a Kaggle Tesla T4
(`benchmarks/performance/reports/t4_2026_09/pt-cudabase-1.txt`: recorded,
re-rendered byte-identically in the same session, and byte-identically again
in a second session). Two of the four scenes — `environment_and_refraction`
and `translucency_and_order` — are byte-identical between the two devices;
`lit_and_shadowed` and `authored_under_many_lights` differ by a few counts
where the two backends round the sampler differently. Re-recording either
set is the same procedure on that device: run
`ALGAN_UPDATE_PATH_TRACED_BASELINES=1 <venv-python> -m pytest
tests/path_traced -q` twice, check the second run's outputs against the first
(they must be byte-identical), look at the videos, and commit the directory.
A device without a committed set renders the scenes and skips the
comparisons.

## The fast suite's render

`tests/fast/scene.py` is a seventh scene under the same conventions, kept apart
from the six above so that the full-render suite and its coverage audit stay
what they are. Its docstring is worth reading before editing it: it is shaped
by the kernel-variant cost, which is why it is one scene rather than several
and why it contains no `Surface` geometry.

## Pending: the full-render and path-traced baselines want regenerating

The grid triangulation was re-wound so a surface triangle's cross product points
outward, the same rule a polyhedron's faces follow
(`agent_guidance/mobs_geometry.md`, "One rule for winding"). That moves scattered
antialiased and texture-sampled pixels — the frames are visually identical, but
the comparison is exact — so **`tests/full_renders` and `tests/path_traced` fail
against their committed baselines on every device until those are regenerated**,
CUDA and CPU alike. `tests/unit_tests` and `tests/fast` — everything CI runs —
are unaffected and green.

Regenerate with `ALGAN_UPDATE_FULL_RENDER_BASELINES=1` and
`ALGAN_UPDATE_PATH_TRACED_BASELINES=1` (see the invocations in each suite's test
module), on a machine of each device, and look at the result before committing.

**`tests/path_traced`'s CPU set was regenerated on 2026-09-04** (on the same
cloud CPU container class it was first rendered on, where the old set still
passed 4/4 before the change), for the path tracer's new fixed-seed default
(`pt_animated_seed = False`, `raytracing/DESIGN_path_tracer_roadmap.md` §0.3):
frame 0 of every scene is byte-identical to before and later frames re-roll
nothing, which is why the three videos came out at less than half their old
size. `tests/full_renders` and the CUDA path-traced set are still pending.

## Baselines are per device

Each render suite keeps one baseline directory per device —
`expected_outputs_cuda/` and `expected_outputs_cpu/` — and the harness picks
between them with `torch.cuda.is_available()`. A machine with no baseline
directory for its device renders the scene and then skips the comparison, so
**a suite that reports itself green on a new device may not have compared
anything**; check for skips before believing it.

The device is the one the render will actually run on —
`SETTINGS.computing.render_device`, so `ALGAN_RENDER_DEVICE=cpu` on a CUDA
machine compares against the CPU baseline, and an Apple Silicon Mac (where the
automatic probe resolves to MPS) does not silently compare a Metal render
against a CPU one.

**macOS is keyed separately** (`expected_outputs_macos_cpu/`), and nothing is
committed under that name, so a Mac renders and skips the comparison. That is
measured, not assumed: the x86-64 CPU baseline was copied in and run on an
Apple Silicon CI runner, and it missed by **up to 45 channel values** (worst at
frame 36) against a tolerance of 2. This is the scene that matched *exactly*
across two x86-64 machines when five of the six full-render scenes did not — so
the change of instruction set is its own axis, separate from the per-machine
spread below, and fp32 arithmetic through a path tracer does not survive it.

A Mac therefore covers kernel compilation, tessellation, LaTeX, the fonts and
the encoder, but not the pixels. To gate pixels there, render with
`ALGAN_UPDATE_FAST_BASELINE=1` on the Mac, look at the result, and commit it.

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

Every render suite is re-baselined by rendering with the baselines writable,
then **looking at the result** before committing:

```bash
ALGAN_UPDATE_FULL_RENDER_BASELINES=1 <venv-python> -m pytest tests/full_renders -q
```

```bash
ALGAN_UPDATE_FAST_BASELINE=1 <venv-python> -m pytest tests/fast -q
```

```bash
ALGAN_UPDATE_PATH_TRACED_BASELINES=1 <venv-python> -m pytest tests/path_traced -q
```

These variables are read by the harnesses rather than by the package, which is
why they are declared in `_HARNESS_VARIABLES` in `algan/environment.py`:
exporting one must not make every `import algan` warn about it. A rebaseline
always writes to the **tree**, never to the download cache — see the next
section for what to do with the result.

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

## Where the heavy baselines live

`tests/full_renders` and `tests/path_traced` carry ~21 MB of mp4s that are
re-committed whole on every rebaseline, and they are the repository's weight
problem: most of the blobs in its history. They are also the baselines **CI
never compares against** — they gate locally, on the machine that rendered
them — so every clone pays for an artifact almost no clone uses.

They are therefore hosted as **GitHub release assets** rather than carried in
every clone. `tests/fast` is deliberately not: it is 368 KB and it is the one
render baseline CI does compare, so keeping it in git means an ordinary clone
and an ordinary CI run never fetch anything.

`tests/baseline_store.py` resolves a suite's baseline directory in this order,
and its module docstring is the contract:

1. `ALGAN_BASELINE_DIR` — a directory of `<suite>/<key>/` trees, for a machine
   that keeps its own or has no network. **Final**: if it holds nothing for
   this suite, the comparison is skipped rather than downloaded.
2. the in-repo `expected_outputs_<key>/`, when it exists and has files;
3. the verified cache under `~/.algan/cache/baselines/<tag>/`;
4. a one-time download of the release asset pinned in `tests/baselines.json`.

Any failure warns once and returns "no baselines", which every suite turns into
a skip. That is the same state as an unbaselined device — so the standing
warning applies with more force than before: **a render suite that skipped
compared nothing.** Check for skips before believing a green run.

`ALGAN_NO_BASELINE_DOWNLOAD=1` forbids step 4 (offline, or a machine that must
not fetch).

### After a rebaseline

The mp4s in the tree are the source of truth; the release asset is a copy of
them. So a rebaseline is not finished when it is committed:

```bash
uv run python scripts/package_baselines.py --tag baselines-YYYY-MM-DD
gh release create baselines-YYYY-MM-DD dist/baselines/*.tar.gz --title ...
git add tests/baselines.json  # the new tag and sha256s
```

Upload **before** pushing the pointer: a tag with no assets behind it makes
every fetch warn and every comparison skip. The tarballs are byte-reproducible
(sorted members, normalized mtimes and modes, zeroed gzip header), so the
pinned sha256 is a fact about the baselines rather than about the machine that
packaged them, and anyone can re-derive it from an uploaded asset.

`scripts/package_baselines.py --verify` re-packages any local hosted baseline
directories into a temporary directory and reports whether they match the
published pointer. A clean checkout has none, so it verifies that the pointer
names a published release instead.
`tests/unit_tests/test_baseline_store.py` runs it, so a rebaseline that is
committed but never uploaded fails a test instead of going unnoticed until the
mp4s leave the tree.

The heavy MP4s are no longer committed, so a normal checkout reaches the cache
or release download. `ALGAN_BASELINE_DIR` still supplies a local canonical copy,
and a freshly rendered `expected_outputs_<key>/` directory still takes precedence
until it is packaged and published. A null tag remains supported only as a
bootstrap/test state.

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
- ★ `test_ux_regressions.py`, `test_easings.py` — the authoring surface
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
