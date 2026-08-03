# AGENTS.md

This file gives coding agents the detailed repository-specific context needed to work safely and effectively on Algan.
Treat the source code as authoritative.

Algan is in private beta and deliberately carries **no compatibility aliases**: there is exactly one public name for each thing. If you find a second spelling of an API, that is a bug to remove, not a surface to preserve.

## Project overview

Algan is a Python 3D animation and rendering library for explanatory mathematics and computer-science videos. It aims to retain Manim-style authoring while providing a Three.js-like 3D material model and a GPU-oriented renderer built with PyTorch and Taichi.

Algan is a lazy animation system. User code records scene state, animated function applications, attribute edits, lifespans, updaters, audio, and timing. Rendering later materializes that recorded state in frame batches, builds render primitives, merges them into renderer input buffers, and renders the batches.

## Development environment

Run commands from the repository root.

On the normal Windows development checkout, use the repository virtual environment rather than system Python:

```text
.venv/Scripts/python.exe
```

## Common commands

### Tests

```text
.venv/Scripts/python.exe -m pytest tests/run_test.py -q
.venv/Scripts/python.exe -m pytest tests/run_test.py -q -k "test_basic_py"
```

The render regression suite imports scene files under `tests/test_files/`, writes outputs under `tests/algan_outputs/`, and compares rendered pixels against the expected CPU or CUDA baselines. Small platform-dependent pixel differences are tolerated. Legitimate renderer changes may require deliberate baseline updates.

Do not run multiple render tests concurrently on Windows. Killed render processes can leave child processes alive and output files locked.

### Documentation

```text
.venv/Scripts/python.exe docs/make_and_open_docs.py
.venv/Scripts/python.exe docs/make_and_open_docs.py --skip-examples --no-open
```

The normal Sphinx build renders embedded examples and is therefore slow. Use `--skip-examples` for structural/autodoc checks that do not need fresh videos. Documentation source is under `docs/source/`. Checked-in API stubs under `docs/source/reference/` must be kept consistent with public modules, classes, and methods.

Use `Scene.save_video(file_path, video_settings)` and `Scene.save_frame(file_path, video_settings)` in new examples; the settings argument is positional, so `Scene.save_video("my_video", HD)` is the form the tutorials teach.

The `.. algan::` directive executes its body and embeds the resulting video. It prefers a file named after the directive, and otherwise embeds whichever video the example just wrote, so examples may name their own output. An example must produce exactly one video.

### Build and publish

```text
uv build
uv publish
```

Bump the package version in `pyproject.toml` before publishing when that file is present in the checkout.

### Ruff

Ruff is configured to apply fixes. Use `ruff check --no-fix` for inspection unless rewriting files is intentional.

Never allow Ruff or another automatic formatter to insert `from __future__ import annotations` into Taichi kernel source files. Taichi kernel modules are named `*_taichi.py` and are excluded for this reason. New kernel modules must preserve that suffix.

## Core ownership model: Scenes are independent containers

`Scene` is the unit of authoring, animation state, audio state, and rendering. Every `Scene` owns:

- its actor registry and effects;
- its camera and light-source collection;
- one ordinary `TimelineManager` instance;
- one ordinary `AnimationManager` instance;
- one ordinary `AudioManager` instance;
- its video settings, background, environment map, output state, and render memory;
- the frame-batching render loop inherited from `RenderLoopMixin`.

`TimelineManager`, `AnimationManager`, and `AudioManager` are not singletons. Do not add singleton accessors back to them, and do not make engine code consult process-global manager state after a mob has been constructed.

Only `SceneManager` remains a singleton. It owns the process-global stack of currently active Scenes. `SceneManager.instance()` returns the manager, not a Scene. The current Scene is `SceneManager.instance().current_scene`.

The active-scene stack exists to preserve concise module-level authoring while allowing nested or independently retained Scenes:

```python
from algan import *

with Scene(video_settings=SMOKE_TEST) as outer:
    outer_square = Square(scene=outer).spawn(animate=False)

    with Scene(video_settings=SMOKE_TEST) as inner:
        inner_circle = Circle(scene=inner).spawn(animate=False)
        inner.save_frame("inner.png")

    # Exiting the inner context restores outer as the active Scene.
    outer.save_video("outer.mp4")
```

Creating a `Scene` pushes it onto the stack. Entering it as a context makes it active; exiting the outermost context level terminates it. A default Scene is created lazily only when code asks for the current Scene while the stack is empty.

Prefer passing `scene=...` explicitly in engine code, helper constructors, and tests. Omitting it binds a new mob to the currently active Scene. Explicit ownership prevents nested-scene bugs and makes dependencies visible.

A mob is permanently associated with its Scene-owned timeline and managers. Do not move an existing mob between Scenes by assigning `mob.scene`. Construct or clone appropriate state inside the destination Scene instead. Mob hierarchies and `Group` objects reject children from multiple Scenes, and one animation context cannot span multiple Scenes.

`Scene.reset()` resets only that Scene's authoring state and constructs fresh timeline, animation, and audio managers. Sibling and enclosing Scenes are not reset. Existing mob references from the old timeline should be treated as invalid after reset.

`SceneManager.reset()` discards the active-scene stack and lazily creates a fresh default Scene. It is intended for complete process-level authoring resets, such as daemon reruns and test isolation.

## Active Scene methods

Methods decorated with `active_scene_method` can be called either on a Scene instance or on the `Scene` class:

```python
scene.save_frame("frame.png")  # uses scene
Scene.save_frame("frame.png")  # uses SceneManager.current_scene
```

Class-level access reports the method's real signature with `self` removed (the descriptor sets `__signature__`), so `help()`, IDE tooltips and Sphinx autodoc show the actual parameters. Any method a tutorial calls as `Scene.foo(...)` **must** carry this decorator — an undecorated method silently binds its first argument to `self` and raises a confusing `TypeError`.

This dual binding is a convenience layer, not a reason to hide Scene ownership inside library code. Prefer instance calls when a Scene reference already exists.

`Scene.instance()` and `Scene.current()` resolve the active Scene. New low-level code should normally use an explicit Scene reference rather than repeatedly resolving the active Scene.

## Animation and timeline architecture

The animation implementation lives under `algan/animation_timeline/` and the animatable/mob base implementation lives under `algan/animatable_base/`.

Each Scene's `TimelineManager` owns all recorded animation data for mobs in that Scene:

- one `AttributeTimeline` per animatable attribute;
- shared attribute buffers keyed by mob id and row ranges;
- timestamped attribute edit records;
- recorded animated-function applications;
- updater events and their dependency traces;
- mob lifespans represented as `[spawn, despawn)` intervals;
- materialization and replay state for a requested frame-time batch.

Mobs do not carry independent per-attribute animation histories. Their getters and setters read and write rows in their Scene's timeline.

### Animation contexts

`AnimationManager` owns the active context tree for one Scene. The main contexts are:

- `Seq`: sequential child animations;
- `Sync`: simultaneous child animations;
- `Lag(ratio)`: partially overlapping child animations;
- `Off`: instantaneous/non-animated changes;
- `Audio` and `Speech`: timing contexts that also register Scene-owned audio effects.

Mob methods decorated with `@animated_function` bind their Scene's `AnimationManager` automatically.

Critical timeline rule: events must be recorded against a context that is entered and exited. Context exit finalizes retroactively rescaled timestamps. Do not manually record events against the top-level context's raw timespan. `add_updater` and `remove_updater` demonstrate the correct pattern by opening an `Off(record_funcs=False, ...)` context.

Timestamps are lazy because parent contexts can rescale child timing on exit. Treat an event's final start/end as unresolved until the relevant context tree has closed.

Overlapping edits to the same timeline rows are replayed in execution order using resolved replay windows. Do not simplify this to ordinary independent interpolation without preserving same-row overlap behavior.

## Animatable and Mob model

`Animatable` handles Scene ownership, ids, timeline-backed attributes, lifespans, spawn/despawn, cloning, animated functions, and updaters.

`Mob` adds geometry-independent 3D state and behavior, including location, basis, scale, color, opacity, glow, hierarchy propagation, movement/layout, morphing, and shader/material configuration.

Parent changes normally propagate to descendants through batched timeline row operations. The canonical hierarchy is `children`/`components`; `Group.mobs` is an alias of `children`. Keep hierarchy operations Scene-homogeneous and cycle-safe.

Renderable mobs implement `get_render_primitives()`. The primary geometry families consumed by the renderer are:

- flat triangle primitives;
- PN curved triangle primitives;
- cubic Bezier circuit primitives.

Important mob implementations include:

- 2D shapes and text, represented primarily as cubic Bezier circuits;
- `Surface` and 3D shapes, represented as triangle or PN-triangle meshes;
- `TriangleMesh` and `ThreeDModelMob` for imported 3D assets;
- `PointCloud`/point-cloud mobs;
- Manim compatibility wrappers and conversion helpers.

Shader/material setup that changes primitive layout or registers shader parameters must occur before spawning unless the implementation explicitly supports timeline-safe mutation. Use the Three.js-style material classes (`MeshBasicMaterial`, `MeshStandardMaterial`, `MeshPhysicalMaterial`, and related classes) rather than restoring removed ad-hoc reflectivity/roughness APIs.

## Rendering API

The public rendering API is Scene-owned, and these are the only spellings:

```python
scene.save_video(file_path=None, video_settings=None, *, overwrite=True, reset=False,
                 background_color=None, animate_fade_out=None, post_processes=None,
                 codec=None, audio_codec=None, ffmpeg_params=None)
scene.save_frame(file_path=None, video_settings=None, at=None, *,
                 overwrite=True, background_color=None)
```

`Scene.save_video` carries the user-facing signature and documentation; `algan.utils.algan_utils._render_scene_to_file` carries the implementation. Keep them in sync — do not push parameters back into `*args, **kwargs`, because that is what made the signature invisible to `help()`, IDEs and autodoc.

Both return a `RenderResult` (`status`, `output_path`, `duration_seconds`, `render_plan`). `save_frame` returns a list of them only when `at` is a sequence.

### Output-path resolution

Both still and video output use the same resolver, `_resolve_output_destination`:

- a bare filename is placed under `SETTINGS.paths.output_root / SETTINGS.paths.output_directory`;
- a relative path with an explicit parent and an absolute path are used as supplied;
- missing still-image extensions default to `.png`;
- missing video extensions default to `.mp4` for opaque output and `.mov` for transparent output;
- parent directories are created automatically.

`output_root` defaults to the directory of `__main__.__file__`, falling back to the working directory; `output_filename` defaults to that script's stem. Do not reintroduce multiple independent `file_name`, `output_path`, and `output_dir` parameters, and do not resurrect `base_directory`.

### `save_frame`

`save_frame` renders one timestamp or a sequence of them (`at`). Multiple timestamps produce files whose names append the timestamp to the resolved stem. Temporary video-settings and background overrides are fully restored afterwards.

`save_frame` never mutates the Scene: nothing is despawned and the timeline is untouched, so it is safe to call repeatedly while authoring. When no timestamp is supplied it renders just after the current authored context time, offset by 1.5 frames. Explicit timestamps must be finite and non-negative.

Keeping the timeline untouched takes more than not recording anything: rendering *resolves* replay windows (`AnimationTimeline._resolve_replay_windows`), freezing each edit's and event's context-rescaled end time into a plain `replay_end` float. From inside an unfinished context those ends are pre-rescale — a `run_time` rescales its block retroactively, on exit — and only recording a new edit invalidates them, so a resolution left behind by a mid-authoring render silently truncates the animations of a later render. `save_frame` and `show_frame` therefore wrap their render in `AnimationTimeline.preserving_authoring_state()`, which restores the resolution state (and drops lifespans created for a render's transient mobs). Any render that leaves the Scene re-renderable must do the same — see the `reset` contract below.

### `save_video` and the `reset` contract

`reset` defaults to **False**: the Scene is left exactly as authored. Mobs stay spawned, references stay valid, the timeline keeps its recording, and rendering again produces the accumulated timeline (the earlier animation plus whatever was added). Independent clips need independent Scenes.

Three pieces of finalization are therefore conditional:

- the zero-duration guard (one frame of `wait` for an all-`Off()` scene) always runs, because it decides how many frames are rendered;
- the end-of-scene despawn of every actor runs when a fade-out was requested (it is part of the requested output) or when `reset=True`;
- `render_to_video` closes the camera and light lifespans only when the Scene is being finalized, via `despawn_camera_and_lights`.

Both lifespans extend past the last rendered frame index either way, so output is unaffected by these gates. `RenderLoopMixin.get_frames` calls `timeline_manager.clear_buffers()` when it finishes, restoring `active_state` to `current_state`; that is what makes a non-reset Scene queryable again after a render.

`reset=False` also passes `preserve_authoring_state=True` into `render_to_video`, which rolls back the two pieces of state the render itself derives: the appended `scene_times` window, and the replay-window resolution (via `preserving_authoring_state()`, as for `save_frame`). The snapshot is taken around the `get_frames` loop rather than around the whole call, because the fade-out and the zero-duration guard record on the timeline first and edits made after a snapshot would fall outside it.

Together with the conditional finalization above, that makes a `reset=False` render legal **from inside an unfinished block**: render a preview mid-`Seq`/`Speech`, keep authoring, and the final render is identical to one where the preview never happened. The frame window for such a render comes from `_recorded_end_time_for_render()`, which takes the max over the whole open context chain — the innermost open context covers only its own block, while an enclosing `Sync` can already hold animations running past it. Every open context shares one un-rescaled timeframe, so their ends are directly comparable; with all blocks closed this is just the root context's end, exactly as before.

With `reset=True` the Scene's timeline, animation and audio managers are rebuilt in `finally` on both success and failure, and authored mobs must not be reused. `overwrite=False` returns a skipped result without finalizing anything. Harnesses that re-author a scene per run (profilers, repeated benchmark passes) should pass `reset=True` explicitly.

Transparent output cannot use MP4. Use MOV or WebM, or an opaque background.

### Scene-function discovery

Use the `@scene_function` decorator for zero-argument scene entry points consumed by `render_all_funcs`. It is deliberately not named `scene`, which would collide with the conventional variable name for a Scene instance. Legacy implicit discovery of every zero-argument function remains as a warning-producing fallback and may accidentally render helpers.

`render_all_funcs` creates an isolated Scene for each function. Scene functions should either rely on that active Scene or accept no arguments and explicitly obtain it; helper constructors should still propagate Scene ownership from their inputs.

## Settings system

Runtime-adjustable public configuration is rooted at the stable process-global `SETTINGS` object:

- `SETTINGS.computing`;
- `SETTINGS.paths`;
- `SETTINGS.style`;
- `SETTINGS.video`;
- `SETTINGS.raytracing`.

Section objects have stable identity and must not be replaced. Mutate them in place with `set`:

```python
SETTINGS.video.set(HD)
SETTINGS.video.set(frames_per_second=60)
SETTINGS.raytracing.set(samples_per_pixel=1)
```

Shared presets such as `SMOKE_TEST`, `PREVIEW`, `LD`, `MD`, `HD`, `PRODUCTION`, `UHD` and `THUMBNAIL` are immutable. Calling `set` on a preset returns a new preset and leaves the shared constant unchanged:

```python
HD_60 = HD.set(frames_per_second=60)
```

Unknown field names are rejected with a close-match suggestion by both `set(...)` and direct attribute assignment — `SETTINGS.video.fps = 60` raises rather than silently attaching a junk attribute.

`SETTINGS.raytracing` is split by stability. Directly on the section are the settings that describe what the renderer *produces*: `samples_per_pixel`, `max_bounces`, `shadows`, `ambient_light`, `light_intensity`, `indirect_bounce_strength`, `glossy_reflection`, `analytic_aa`, `tonemapping`, `tonemap_method`, `tonemap_exposure`, `unsupported_feature_policy`. Every other switch is a kernel/performance gate and lives on `SETTINGS.raytracing.experimental`; writing one through the parent raises an error naming the right location. **Reads are deliberately unrestricted** — engine modules bind `rt_settings = SETTINGS.raytracing` once and read experimental switches off it on the hot path — so only mutation is gated. `to_dict()`, `as_preset()`, `_restore()` and `SETTINGS.snapshot()` continue to cover every field.

When adding a renderer toggle, add it to `_FIELD_TO_LEGACY` and leave it out of `_PUBLIC_FIELDS` unless it changes rendered output in a way users are meant to control.

Use `SETTINGS.snapshot()`/`SETTINGS.restore()` for complete public-settings state capture, and `SETTINGS.override(...)` or section-level `override(...)` for temporary changes. Do not hand-roll partial restoration that leaves live settings leaked into later tests or daemon runs.

Engine modules must read mutable settings live through `SETTINGS`. Never import a mutable ray-tracing setting by value at module import time; doing so freezes the old value and makes public setters ineffective. Immutable constants may be imported by value.

Initialization-only settings intentionally have no public mutable Python object. Set these before importing `algan`:

- `ALGAN_ANIMATION_DEVICE`;
- `ALGAN_RENDER_DEVICE`;
- `ALGAN_HOME`;
- `ALGAN_CACHE_DIR`;
- `TI_OFFLINE_CACHE_FILE_PATH`;
- `ALGAN_SOFT_SHADOW_SAMPLES`;
- `ALGAN_HDR_BUFFER_F16`.

`RENDERER_REGISTRY` and `KERNEL_REGISTRY` are runtime service registries, not user settings, and therefore live outside `SETTINGS` and outside the star-import namespace.

`SETTINGS.computing` rejects `render_device`, `animation_device` and `render_on_cpu` with a message naming the environment variable to use instead, rather than a generic "unknown setting".

The old `COMPUTING_DEFAULTS`, `DIRECTORY_DEFAULTS`, `STYLE_DEFAULTS` and `RENDERING_DEFAULTS` facades and the `algan.settings.render_settings` / `algan.settings.style_defaults` modules have been **deleted**. Do not reintroduce them.

## Rendering pipeline

The render loop is implemented in `algan/render_loop.py` as `RenderLoopMixin`, mixed into `Scene`. It is responsible for:

- choosing frame windows according to animation and render memory budgets;
- materializing the Scene timeline at frame times;
- building and batching actor render primitives;
- snapshotting camera and light state;
- optionally prefetching the next batch on a worker thread;
- projecting, merging, and uploading scene data;
- invoking the configured render kernel;
- applying post-processing and streaming frames to the writer;
- reducing the frame window and retrying on render-memory exhaustion.

`ALGAN_PREFETCH_BATCHES=0` disables next-batch prefetch. Keep Scene render-state snapshots immutable so preparation can run safely while the previous batch renders.

### Scene merge and acceleration structures

`algan/rendering/raytracing/scene_builder.py` packs projected primitives into contiguous tensor arrays grouped by geometry type and builds the corresponding spatio-temporal acceleration structures. The merged dictionary is the contract consumed by the tracer orchestration and Taichi kernels.

Do not casually change merged-field widths, ordering, dtype, or lifetime. Those changes affect memory estimators, arena preflight, kernel signatures, projection/merge paths, and potentially cached Taichi variants.

### Renderer dispatch

`render_batch_raytraced` is the production render entry point registered in `KERNEL_REGISTRY`.

- `samples_per_pixel == 1` selects the deterministic wavefront renderer. It uses bounded primary-ray tiles, traversal, shading, compaction, compositing, and a shared continuation pool for reflective/refractive splits. Tile overflow is retried with fewer primaries rather than approximated.
- `samples_per_pixel > 1` selects the Monte Carlo path-tracing megakernel. Some deterministic-only features are rejected or handled according to the unsupported-feature policy.

The deterministic renderer can use the hybrid raster pipeline for primary visibility when supported and enabled. Flat triangles and Bezier circuits can be discovered/rasterized into covered fragments, with analytic anti-aliasing options, while secondary reflection/refraction/shadow continuations remain in the ray-based wavefront system. Do not describe the current renderer as either a pure rasterizer or a pure one-primary-ray-per-pixel tracer.

The classes under `algan/rendering/primitives/` are still used for primitive construction and batching. They are not a separate supported legacy raster backend. New renderer work belongs in the active raytracing/hybrid pipeline unless a deliberate new backend is being introduced through the registries.

### Materials and fragment pipelines

Three.js-style material objects configure shaders and register animatable shader parameters. Per-fragment pipelines and custom scatter behavior are composed into flat Taichi template tuples. Nested tuples do not work as kernel template arguments.

A scene with custom fragment shading/scatter may force deterministic fragment-shading paths and alter continuation-pool requirements. Keep capability detection, memory estimation, render planning, and actual kernel dispatch consistent.

### Manual memory

`ManualMemory` is the render-time arena. It provides deterministic forward allocations and pointer restore/reset behavior. Many render paths depend on exact arena byte estimates. When adding buffers:

- update all corresponding memory estimators and preflight calculations;
- account for dtype alignment and fixed versus per-frame/per-ray scaling;
- restore arena pointers at the same lifetime boundary at which data becomes dead;
- test one-frame and multi-frame windows;
- test retry behavior rather than relying on host OOM exceptions.

The whole package enters process-global `torch.inference_mode()` during import. Importing Algan therefore disables autograd for the importing process. Do not use the same process for Algan rendering and Torch model training.

## Audio

Audio is Scene-owned. `AudioManager` stores the Scene's speech source and transcript. `Audio`/`Speech` contexts add `AudioEffect` objects to the owning Scene and derive timing from that Scene's animation manager.

Do not add process-global transcript or speech-generator state. When constructing `Speech` or `Audio` contexts in low-level code, bind the relevant Scene animation manager explicitly.

## Taichi rules and failure modes

Taichi kernel work has several repository-specific hazards:

- The offline cache does not reliably invalidate when an imported `@ti.func` changes. Clear it before trustworthy A/B tests of kernel-source edits with `clear_cache(taichi_kernels=True)`.
- Never edit `*_taichi.py` while a render process or warm daemon is running. The JIT can compile mixed old/new source.
- Restart the render daemon after changing any Algan source; imported modules remain stale. Restart is mandatory after changing Taichi source.
- Cold compilation can take minutes and separate renderer routes may compile separate megakernels.
- `ALGAN_TI_DEBUG=1` is for debugging only and severely reduces performance.
- Prefer `ti.static(bool(template_value))` for template gates rather than Python identity tests such as `is not None` inside kernel code.
- Keep Taichi template argument structures flat.
- Preserve the `*_taichi.py` filename convention.

## Performance and renderer validation

Optimize general moving and animated scenes, not only static-scene fast paths.

For performance changes:

- compare warm in-process alternating A/B runs when possible;
- use device-side kernel-profiler timings to separate launch/synchronization from kernel execution;
- avoid drawing conclusions from a single cross-process wall-clock run;
- verify the intended optimization gate actually engaged;
- record render route and relevant live settings;
- validate output parity before accepting a speedup.

Use focused parity/benchmark scripts under `benchmarks/` when present. The default path should remain output-compatible unless the change intentionally modifies rendering. If adding an experimental optimization, provide a kill switch and keep capability checks, memory estimation, and fallback behavior coherent.

For source-only correctness checks, at minimum run import/compile checks on modified non-Taichi modules. For visual renderer changes, render a minimal `SMOKE_TEST` scene or a single diagnostic frame. Do not run a long benchmark merely to prove that code imports.

## API-change discipline

Algan has removed its transitional aliases ahead of public release. The canonical forms are the only forms:

- `Scene.save_video` / `scene.save_video` — there is no module-level `render_to_file` or `render`, and no `Scene.render_to_file`/`Scene.render`;
- `Scene.save_frame` for stills, with `at` rather than `time_stamps`;
- `video_settings` / `VideoSettings` — `render_settings`, `RenderSettings` and `set_render_settings` are gone;
- one `file_path` rather than separate filename/directory arguments;
- `SETTINGS` sections rather than the old defaults globals;
- Scene-owned managers rather than singleton managers;
- `@scene_function` rather than `@scene`;
- `draw_border_then_fill(mobs)` rather than `write(mob)`; it takes any iterable of Mobs, and `Tex`/`Text` expose `.write()` as the glyph-wise shorthand.

Do not add a second spelling for something that already has a name. If a rename is genuinely warranted, rename in place and update every call site — the project is pre-release specifically so this stays cheap.

### The star-import namespace is the API

`from algan import *` is the documented entry point, so `algan.__all__` is effectively the public surface. `algan/__init__.py` builds it from a rule plus two deny-lists (`_INTERNAL_EXPORT_MODULES`, `_INTERNAL_EXPORT_NAMES`) and one allow-list (`_EXTRA_EXPORTS`). Generic helper names must not leak: `mean`, `interpolate`, `offset`, `shuffle`, `broadcast*`, `traverse`, `squish` and friends would shadow whatever the user imported before Algan.

When you add a name, decide which side it is on. Public mobs, animations, contexts, materials, shaders, constants and settings belong in the namespace; tensor utilities, mixins, primitive builders, registries and dev tooling do not. `tests/unit_tests/test_ux_regressions.py` asserts both directions.

When changing a public class, method, setting, material field, or render argument:

- update root exports in `algan/__init__.py` as needed;
- update docs and checked-in autosummary stubs;
- search docs, tests, examples and benchmarks for stale call sites and fix all of them, since nothing keeps the old name working;
- add or update tests for the new behavior.

## Vendored code

`algan/external_libraries/` contains vendored Manim, ground, and sect code. Treat it as read-only unless the task specifically requires a vendored patch and the consequences are understood. Prefer adapters and compatibility code in Algan-owned modules.

## Repository map

- `algan/scene.py` — Scene container, active-scene method binding, still/video entry points, reset and Scene-owned manager construction.
- `algan/scene_manager.py` — singleton active-Scene stack and lazy default Scene.
- `algan/render_loop.py` — materialization-to-frame pipeline, batching, prefetch, memory preflight, and video streaming.
- `algan/animation_timeline/` — animation contexts, per-Scene timeline, attribute timelines, event replay, and updater materialization.
- `algan/animatable_base/` — `Animatable`, `Mob`, hierarchy, transforms/layout, material support, and morphing.
- `algan/animations/` — built-in composable animations.
- `algan/mobs/` — shapes, text, surfaces, meshes/models, plots, point clouds, groups, and compatibility mobs.
- `algan/rendering/raytracing/` — scene merge, STBVHs, hybrid primary rasterization, deterministic wavefront kernels, Monte Carlo kernels, shading, and renderer settings compatibility.
- `algan/rendering/shaders/` — material classes, material shaders, PBR shaders, and fragment pipelines.
- `algan/rendering/post_processing/` — bloom, anti-aliasing, tonemapping, and post-process memory estimation.
- `algan/rendering/primitives/` — primitive construction/batching bases used by the active renderer.
- `algan/rendering/camera.py` and `lights.py` — Scene-owned camera and light mobs.
- `algan/settings/` — `SETTINGS` sections, presets, and startup-only environment configuration.
- `algan/sound/` — Scene-owned audio manager and audio effects.
- `algan/utils/` — memory arena, profiling, file/audio helpers, Taichi warm-start/fast-launch patches, and development utilities.
- `algan/daemon.py` — warm-process scene-script rerender daemon with Scene/settings reset between runs.
- `algan/external_libraries/` — vendored dependencies; normally do not modify.
- `tests/` — render regression and behavior tests when included in the checkout.
- `benchmarks/` — targeted profiling, A/B, and output-parity scripts when included.
- `docs/` — Sphinx documentation and rendered examples.

## Canonical authoring examples

Module-level concise authoring remains supported through the lazy default Scene:

```python
from algan import *

square = Square().spawn()
square.move(RIGHT)

with Sync():
    square.rotate(90, OUT)
    square.color = BLUE

Scene.save_video("example.mp4")
```

Explicit Scene ownership is preferred for reusable code, tests, nested scenes, and multiprocessing:

```python
from algan import *

with Scene(video_settings=SMOKE_TEST) as scene:
    square = Square(scene=scene).spawn(animate=False)
    with Sync(animation_manager=scene.animation_manager):
        square.move(RIGHT)
        square.rotate(45, OUT)

    scene.save_frame("diagnostic.png")
    result = scene.save_video(
        "diagnostic.mp4",
        SMOKE_TEST,
        overwrite=True,
        animate_fade_out=False,
    )
```

`save_video` leaves `scene` intact by default, so mob references stay valid and you can keep authoring. Remember that Algan records onto one timeline: rendering again produces the accumulated animation, not just the new part. Use a separate Scene per independent clip, and pass `reset=True` only when you deliberately want the Scene discarded.
