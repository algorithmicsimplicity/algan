# AGENTS.md

This file gives coding agents the detailed repository-specific context needed to work safely and effectively on Algan.
Treat the source code as authoritative.

`../CLAUDE.md` is the operational quick-start — commands, hazards, and the API shape — and is always loaded. This file
holds what is cross-cutting (Scene ownership, the repository map, release and Taichi/daemon mechanics) and indexes the
topic references below. **Read only the topic file your task touches.** When the docs and the source disagree, the
source wins.

## Detail references

| Touching | Read |
| --- | --- |
| `animation_timeline/`, recording, replay, materialization, updaters, audio | [`timeline.md`](timeline.md) |
| `mobs/`, `animatable_base/`, packing, circuits, PN patches, tessellation | [`mobs_geometry.md`](mobs_geometry.md) |
| `rendering/`, any `*_taichi.py`, shading, shadows, colour, post-processing | [`rendering.md`](rendering.md) |
| `ManualMemory`, batch sizing, optimization work, A/B parity fixtures | [`memory_perf.md`](memory_perf.md) |
| public names, `SETTINGS`, output paths, `ALGAN_` variables | [`api_settings.md`](api_settings.md) |

Also in this directory: `CLAUDE_CLOUD.md` (read it if you are running on Claude Code Cloud).

Algan is in private beta and deliberately carries **no compatibility aliases for its own API**: there is exactly one Algan name for each Algan thing. If you find a second spelling of an Algan API, that is a bug to remove, not a surface to preserve.

The Manim compatibility layer is the one deliberate exception, and it is a separate surface rather than a second spelling of Algan's. `Mobject = Mob`, `GenericGraph = Graph`, `install_opengl_aliases()` and the wrapper classes in `../algan/mobs/manim_compat.py`, `manim_parity.py`, `opengl_compat.py` and `point_cloud.py` exist so a Manim script keeps working under the names its author already wrote; they are exported and supported. The rule to apply: never add an Algan-side alias for an Algan name, and never delete a Manim-side name merely because it duplicates one.

## Project overview

Algan is a Python 3D animation and rendering library for explanatory mathematics and computer-science videos. It aims to retain Manim-style authoring while providing a Three.js-like 3D material model and a GPU-oriented renderer built with PyTorch and Taichi.

Algan is a lazy animation system. User code records scene state, animated function applications, attribute edits, lifespans, updaters, audio, and timing. Rendering later materializes that recorded state in frame batches, builds render primitives, merges them into renderer input buffers, and renders the batches.

## Development environment

Run commands from the repository root.

Use the repository virtual environment rather than system Python.

`uv run python` resolves the right one on every platform.

## Documentation

```text
uv run python docs/make_and_open_docs.py
uv run python docs/make_and_open_docs.py --skip-examples --no-open
```

The normal Sphinx build renders embedded examples and is therefore slow. Use `--skip-examples` for structural/autodoc checks that do not need fresh videos. Documentation source is under `../docs/source`. Checked-in API stubs under `../docs/source/reference` must be kept consistent with public modules, classes, and methods.

Use `Scene.save_video(file_path, video_settings)` and `Scene.save_frame(file_path, video_settings)` in new examples; the settings argument is positional, so `Scene.save_video("my_video", HD)` is the form the tutorials teach.

The `.. algan::` directive executes its body and embeds the resulting video. It prefers a file named after the directive, and otherwise embeds whichever video the example just wrote, so examples may name their own output. An example must produce exactly one video.

## Build and publish

```text
uv build
uv publish
```

Bump the package version in `../pyproject.toml` before publishing when that file is present in the checkout.

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

## Taichi rules and failure modes

Taichi kernel work has several repository-specific hazards:

- The offline cache does not reliably invalidate when an imported `@ti.func` changes. Clear it before trustworthy A/B tests of kernel-source edits with `clear_cache(taichi_kernels=True)`.
- Never edit `*_taichi.py` while a render is running. The JIT can compile mixed old/new source.
- The render daemon restarts itself after any Algan source change: it fingerprints every `.py` under `../algan` at startup, re-checks at each run launch, and refuses the run and shuts down if anything differs, so the script runs in a fresh process with the edited code. No hand restart, but the cold start (and, for kernel edits, a full recompile) is still paid. See `DESIGN_daemon_lifecycle.md`.
- The daemon also refuses a run whose *import-time* environment differs from the one it imported algan with (`_IMPORT_TIME_VARIABLES` in `../algan/environment.py`), because those values are already module-level defaults by then: a script setting a renderer toggle before its own `import algan` would otherwise be served with the daemon's value. Live variables are swapped in per run and can be changed mid-script. When a run ends the daemon resets its state and hands the render's GPU memory back to the driver (`gc.collect()` + `torch.cuda.empty_cache()`), so an idle daemon holds the warm process and nothing else; `ALGAN_DAEMON_RELEASE_MEMORY=0` opts out.
- Cold compilation can take minutes and separate renderer routes may compile separate megakernels.
- `ALGAN_TI_DEBUG=1` is for debugging only and severely reduces performance.
- Prefer `ti.static(bool(template_value))` for template gates rather than Python identity tests such as `is not None` inside kernel code.
- Keep Taichi template argument structures flat.
- Preserve the `*_taichi.py` filename convention.

## Vendored code

`../algan/external_libraries` contains vendored Manim, ground, and sect code. Treat it as read-only unless the task specifically requires a vendored patch and the consequences are understood. Prefer adapters and compatibility code in Algan-owned modules.

## Repository map

- `../algan/scene.py` — Scene container, active-scene method binding, still/video entry points, reset and Scene-owned manager construction.
- `../algan/scene_manager.py` — singleton active-Scene stack and lazy default Scene.
- `../algan/render_loop.py` — materialization-to-frame pipeline, batching, prefetch, memory preflight, and video streaming.
- `../algan/animation_timeline` — animation contexts, per-Scene timeline, attribute timelines, event replay, and updater materialization.
- `../algan/animatable_base` — `Animatable`, `Mob`, hierarchy, transforms/layout, material support, and morphing.
- `../algan/animations` — built-in composable animations.
- `../algan/mobs` — shapes, text, surfaces, meshes/models, plots, point clouds, groups, and compatibility mobs.
- `../algan/rendering/raytracing` — scene merge, STBVHs, hybrid primary rasterization, deterministic wavefront kernels, Monte Carlo kernels, shading, and renderer settings compatibility.
- `../algan/rendering/shaders` — material classes, material shaders, PBR shaders, and fragment pipelines.
- `../algan/rendering/post_processing` — bloom, anti-aliasing, tonemapping, and post-process memory estimation.
- `../algan/rendering/primitives` — primitive construction/batching bases used by the active renderer.
- `../algan/rendering/camera.py` and `lights.py` — Scene-owned camera and light mobs.
- `../algan/settings` — `SETTINGS` sections, presets, and startup-only environment configuration.
- `../algan/sound` — Scene-owned audio manager and audio effects.
- `../algan/utils` — memory arena, profiling, file/audio helpers, Taichi warm-start/fast-launch patches, and development utilities.
- `../algan/daemon.py` — warm-process scene-script rerender daemon with Scene/settings reset between runs.
- `../algan/external_libraries` — vendored dependencies; normally do not modify.
- `../tests` — render regression and behavior tests when included in the checkout.
- `../benchmarks` — targeted profiling, A/B, and output-parity scripts when included.
- `../docs` — Sphinx documentation and rendered examples.

