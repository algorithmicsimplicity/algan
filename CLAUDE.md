# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`AGENTS_DETAILED.md` is the detailed architecture and contract reference. This file is the
operational quick-start: commands, hazards, and the API shape. When the two
disagree, the source code wins, then AGENTS_DETAILED.md.

## Project Overview

Algan is a 3D animation engine for explanatory math videos, designed as a successor to Manim: it keeps Manim's ease of use while providing the 3D graphics capabilities of Three.js. It uses PyTorch for animation math and custom Taichi kernels for GPU ray-traced rendering. Algan produces the animations in AlgorithmicSimplicity videos.

Algan is **lazy**: running a user script does not compute animations, it *records*
them on the Scene's timeline. `Scene.save_video()` materializes that recording in
batches of frames, builds render primitives, and renders them.

## Commands

### Running Python
Always use the local venv; the default system Python lacks taichi and the other pinned dependencies.

**This is the one place the interpreter path is written down.** Commands in this
file, `AGENTS.md`, `AGENTS_DETAILED.md`, `tests/README.md` and the design docs
are all written as `<venv-python>` — substitute your platform's:

| Platform | `<venv-python>` |
| --- | --- |
| Linux / macOS | `.venv/bin/python` |
| Windows | `.venv\Scripts\python.exe` |

`uv run python` works on every platform if you would rather not think about it.

(The published contributor guide, `docs/source/contributing/development.rst`,
repeats the paths deliberately: outside contributors read it without this file.
It is the only other copy — keep the two in step.)

### Cloud sessions (Claude Code on the web)
A cloud session is a fresh Ubuntu 24.04 VM, **4 vCPUs / 16 GB RAM / 30 GB disk**, with no GPU and nothing installed beyond the base image. `.claude/hooks/session-start.sh` provisions it before you get control: apt build/LaTeX/ffmpeg packages, then `uv sync --locked --all-extras --dev`. It is a no-op on a local checkout. If a build or a Tex test fails with missing headers or a missing `latex`, read that script first — the environment is probably mid-provision or the apt step warned and continued.

What is different here, and what it means for what you can conclude:
- **No GPU.** `_auto_render_device` falls back to CPU and everything works, just slower. Renders and the suite both pass.
- **Pixel comparisons silently do not run.** Baselines are per device and only `expected_outputs_cuda/` is committed, so `tests/fast` renders its scene and then skips the comparison (`no cpu fast-suite baseline is available`), and `tests/full_renders` skips outright. **A renderer change cannot be validated in a cloud session.** Say so rather than implying the suite covered it; that verification needs a CUDA machine.
- **Do not re-baseline here.** `ALGAN_UPDATE_*_BASELINE=1` would write CPU frames, and CPU and CUDA renders are not bit-identical.
- **`download.pytorch.org` is blocked** by the default Trusted network policy, so the CPU-only Torch wheels are unreachable and there is no CPU build of Torch on PyPI. The lockfile's CUDA build is installed instead (~5 GB of `nvidia-*` wheels) and runs fine on CPU. To avoid the download, set the environment's Network access to **Custom** at claude.ai/code, add `download.pytorch.org`, and keep "include default list of common package managers" checked.
- **Watch the disk.** A full install lands around 12 GB of the 30 GB. `df` reports the allowance, not the machine, so "Avail 0" with low "Used" means the allowance is spent.

Persistence: the container is ephemeral and nothing outside git survives it. Commit anything worth keeping. Repo-level config (`CLAUDE.md`, `.claude/settings.json`, the hook) is what carries over; the environment's own **setup script** and **environment variables**, configured in the environment dialog at claude.ai/code, persist separately and are snapshotted after their first run.

### Testing
```
<venv-python> -m pytest -q --fast    # THE development loop: 112-147s
<venv-python> -m pytest -q           # everything, ~12 min, before pushing
```
- **`--fast` is the suite to run after every change.** It is everything not marked `slow`, held to a two-and-a-half-minute budget, and it prints where it landed (`fast suite: 134s of its 150s budget (89%)`). Pass no path — it uses `testpaths` from `pyproject.toml`.
- **Its self-reported time is junk until the third consecutive run.** Taichi charges a kernel variant to whichever test hits it first, so any change that touches a kernel makes run 1 pay a cold compile: a measured sequence right after adding two small kernels was 194s → 160s → 112s. Never mark a test `slow` off run 1 or 2.
- It covers the whole behavioural suite (`tests/unit_tests/`) plus **one real render compared pixel-wise** (`tests/fast/`). That render is the only thing in the loop that can see a renderer regression, and it is half the budget.
- Run the **full** suite after touching the renderer, and before pushing. `tests/README.md` has the table of what `--fast` leaves out and where each item is covered instead.
- `slow` means **outside the fast suite** — a budget decision, not a description. When the fast suite reports itself over budget, mark the *newly added* expensive test, not an old one.
- **Taichi cost is per kernel variant, not per test**, charged to whichever test hits that variant first. Marking one test `slow` can just move its seconds to the next test that needs the same kernel (this happened with `test_raytracing_unit.py`, hence its module-level mark). A group sharing a kernel leaves together or not at all. Adding PN geometry (`Sphere`/`Cylinder`/`Cone`/`Torus`/`Surface`) to `tests/fast/scene.py` costs ~20s on its own — use a `Polyhedron` subclass there.
- Renders are compared **pixel-wise** against `expected_outputs_cuda/` (or `expected_outputs_cpu/`) in each render suite's own directory. Any channel deviation > 2 fails; diff videos land in that suite's `output_errors/`.
- Small (≤2) pixel differences across runs are expected and tolerated: torch CPU rate-function evaluation rounds differently depending on materialization window, so exact byte-identity across re-windowed state is unattainable.
- On Windows, run render work **one process at a time**: killed/timed-out background runs orphan child processes that keep output mp4s locked.
- When a legitimate rendering change alters output, re-baseline with `ALGAN_UPDATE_FAST_BASELINE=1` / `ALGAN_UPDATE_FULL_RENDER_BASELINES=1` and **look at the result** before committing (this is normal practice here).
- **Cap any script whose tensor sizes come from parameters** rather than from a real scene: `benchmarks/_memory_cap.py`'s `cap_process_memory(gb)` (call it *before* importing torch). A mis-sized synthetic generator has exhausted system RAM and blue-screened this machine. Do **not** cap a real render — WDDM charges the VRAM arena against process commit, so a capped render segfaults inside CUDA instead of raising.

### Documentation
- Build: `<venv-python> docs/make_and_open_docs.py` (Sphinx; renders every embedded example video, so it is slow). Add `--skip-examples --no-open` for structural/autodoc checks.
- Source in `docs/source/`. API stubs in `docs/source/reference/` are autosummary-generated.
- **Docstrings on user-facing API follow `DOCSTRINGS.md`** — read it before writing or editing a public docstring. It is prescriptive, not a description of current code: NumPy style with types in annotations only (never repeated in the docstring), every default stated in prose, units/shapes mandatory, an `Animation` section stating recorded-vs-immediate and spawn-order constraints, and `.. algan::` examples that call `Scene.save_video()` exactly once.

### Building / Publishing
- `uv build` from the project root; `uv publish` after bumping the version in pyproject.toml.

### Linting — read before running ruff
- Ruff is configured with `fix = true`: a plain `ruff check` **rewrites files**. Use `ruff check --no-fix` unless you intend to apply fixes.
- Never let ruff (or anything else) touch `*_taichi.py` files: the auto-inserted `from __future__ import annotations` breaks Taichi kernel compilation. Kernel files are exempted in the config and their filenames MUST end in `_taichi` to keep it that way.

## Public API

Algan is in private beta and carries **no compatibility aliases**. There is one
name for each thing; if you find a second, it is a bug.

```python
from algan import *

square = Square().spawn()  # mobs must be spawned to appear / animate
square.move(RIGHT)  # recorded as a 1-second animation

with Sync():  # play simultaneously
    square.rotate(90, OUT)
    square.color = BLUE

Scene.save_video("example")  # -> algan_outputs/example.mp4
Scene.save_video("example", HD)  # one-off quality override
```

- **Output**: `Scene.save_video(file_path=None, video_settings=None, *, overwrite, reset, background_color, animate_fade_out, post_processes, codec, audio_codec, ffmpeg_params)` and `Scene.save_frame(file_path=None, video_settings=None, at=None, *, overwrite, background_color, post_processes)`. Both return `RenderResult` (`status`, `output_path`, `duration_seconds`, `render_plan`); `save_frame` returns a list only when `at` is a sequence. There is no module-level `render_to_file`/`render`, no `render_settings` keyword, and no `RenderSettings` alias.
- **`reset` defaults to False.** `save_video` leaves the Scene exactly as authored: mobs stay spawned and valid, the timeline keeps its recording, and you can render again — including a preview rendered from inside a `with` block that has not finished yet, which covers everything recorded so far and leaves the Scene untouched. Pass `reset=True` for the old destructive behavior (despawn everything, rebuild the timeline/animation/audio managers) — harnesses that re-author a scene per run should do this. `save_frame` never mutates the Scene.
- **Settings**: one process-global `SETTINGS` with sections `video`, `style`, `paths`, `computing`, `raytracing`. Sections have stable identity — mutate with `SETTINGS.video.set(HD)`, never `SETTINGS.video = HD`. Presets (`PREVIEW`, `LD`, `MD`, `HD`, `PRODUCTION`, `UHD`, `THUMBNAIL`, `SMOKE_TEST`) are immutable; `HD.set(frames_per_second=60)` returns a copy.
- **`SETTINGS.raytracing`** holds what the renderer *produces* (`samples_per_pixel`, `max_bounces`, `shadows`, lighting, tonemapping). The ~55 kernel/perf switches live on `SETTINGS.raytracing.experimental` and setting them on the parent raises with a pointer. Engine code still *reads* everything off `SETTINGS.raytracing` directly — only writes are gated.
- **`Scene.foo(...)` and `scene.foo(...)`** are the same method: `active_scene_method` binds to an instance, or resolves the active Scene when called on the class. Class-level access reports the real signature (no `self`), so `help()` and autodoc work.
- **Paths**: `SETTINGS.paths.output_root / output_directory / name`. A bare filename goes to the output directory; anything with a directory in it is used as given. `output_root` defaults to the main script's directory (CWD when there is no script) and `output_filename` to the script's stem.
- **`from algan import *` is curated.** Internal helpers (`mean`, `interpolate`, `offset`, `shuffle`, `broadcast*`, `traverse`, mixins, registries) are excluded via `_INTERNAL_EXPORT_MODULES` / `_INTERNAL_EXPORT_NAMES` in `algan/__init__.py`. When adding a public name, check it lands in `algan.__all__`; when adding a helper, check it does not. Public names are chosen to be collision-safe: the hand-drawing animation is `draw_border_then_fill(mobs)` (any iterable of Mobs), with `Tex.write()` / `Text.write()` as the glyph-wise shorthand.
- **Asset paths**: `ImageMob`, `set_texture` and `background_color` all route through `file_utils.get_image` → `resolve_asset_path`, which tries the working directory then the main script's directory, so an image beside your script loads regardless of where you launch Python.

## Architecture

### Scenes own everything

`Scene` (`scene.py`) is the unit of authoring and rendering. Each Scene owns its
actors, camera, lights, **its own** `TimelineManager`, `AnimationManager` and
`AudioManager`, its video settings, and the render loop it inherits from
`RenderLoopMixin` (`render_loop.py`).

Only `SceneManager` (`scene_manager.py`) is a singleton: it holds the process-global
stack of active Scenes. `SceneManager.instance()` returns the *manager*, not a Scene;
the current Scene is `SceneManager.instance().current_scene`, created lazily on first
use. Do not add singleton accessors back to the other managers, and do not consult
process-global manager state after a mob is constructed.

### Animation system (`algan/animation_timeline/`, `algan/animatable_base/`)

Each Scene's `TimelineManager` (`timeline.py`) owns all recorded animation data for
mobs in that Scene — mobs hold no per-object animation storage:
- One **`AttributeTimeline`** per animatable attribute (location, basis, color, opacity, ...): a shared `[1, N, W]` buffer of every mob's current values (each mob owns rows, keyed by its `id` in `mob_id_to_inds`) plus the log of timestamped edits to those rows (`EditRecord`: rows, pre-modification values, end time). `set_state_to_times(times)` materializes all buffers at the requested frame times in one batched pass per attribute (`generate_array_states`, a flat `torch.searchsorted` over a per-row composite key on the animation device — deliberately not a Taichi kernel, which would stage the whole buffer through VRAM from the batch-prep worker), then re-executes recorded function applications with per-frame interpolated arguments, then applies updaters. Edits of the same rows may overlap in time: `_resolve_replay_windows` extends each edit's effective end over the replay windows of earlier-executed edits that overlap it (transitively, unified per function application), the base state at time t is the pre-value of a row's earliest-executed edit still unfinished at t, and functions replay through their extended window (held at final parameters past their own end) so overlapping and same-end edits rematerialize in execution order.
- The **function timeline**: `FunctionApplicationEvent`s recorded by the `@animated_function` decorator, and `UpdaterEvent`s from `Mob.add_updater`.
- Every mob's **`Lifespan`** ([spawn, despawn) interval), exposed as `Animatable.lifespan` and queried via `is_spawned()` / `is_despawned()`. Sub-mobs created by indexing (`mob[i]`) share their source's id and therefore its rows and lifespan; clones get a new id. Opacity is zeroed outside a mob's lifespan during materialization.
- `get_frames` calls `timeline_manager.clear_buffers()` when it finishes, returning `active_state` to `current_state`. This is what makes `reset=False` safe: after a render the timeline is queryable again.

**`AnimationContext`s** (`animation_contexts.py`) control *when* recorded events happen: `Seq()` (sequential), `Sync()` (simultaneous), `Lag(ratio)`, `Off()` (instant, unanimated), plus `Audio`/`Speech`. Contexts nest and inherit unset parameters; `run_time` rescales all child timestamps retroactively on `__exit__`.

**CRITICAL:** timeline events must be recorded against an **entered-and-exited context**, never the top-level context — only `__exit__` syncs a context's rescaled timestamps, so events recorded on the top-level timespan all evaluate to time 0. The `animated_function` wrapper enters a child context automatically; anything recording events manually (see `add_updater`) must wrap itself in e.g. `with Off(record_funcs=False) as context:`.

Structural batch rewrites (e.g. `become`'s batch expansion) go through `_setattr_and_rebatch_without_record`, which re-allocates a mob's rows; recorded history stays with the old rows, so this is only valid on mobs with fresh history (`detach_history` provides that).

### Mobs (`algan/mobs/`, `algan/animatable_base/`)

- `Animatable` (`animatable_base/animatable.py`): Scene ownership, ids, timeline-backed attribute get/set, spawn/despawn, clone, animated functions, updaters.
- `Mob` (`animatable_base/mob.py` plus the `mob_*.py` mixins): 3D location/basis/color, spatial transforms, screen-relative layout, `become` morphing, and the shader/material API (`set_shader`, `set_fragment_shader`, `set_material` — all must be called *before* spawning).
- Attribute changes on a parent propagate to children (the hierarchy is `children`/`components`; `Group.mobs` aliases `children`).
- Shapes: 2D shapes (`shapes_2d.py`) and `Text`/`Tex` (`text.py`) are cubic bezier circuits (`bezier_circuit.py`); 3D shapes (`shapes_3d.py`) are triangle meshes via `Surface` (`surfaces/surface.py`); `ThreeDModelMob` (`three_d_models/`) imports .glb/.fbx; `ManimMob` wraps Manim mobjects.
- To be renderable, a mob defines `get_render_primitives()` returning flat triangles, PN curved triangles, or cubic bezier circuits.
- Use the Three.js-style material classes (`MeshBasicMaterial`, `MeshStandardMaterial`, `MeshPhysicalMaterial`, ...) rather than ad-hoc reflectivity/roughness APIs.

### Rendering pipeline (`algan/rendering/`)

- Entry point: `render_batch_raytraced` (`raytracing/tracer.py`), registered in `KERNEL_REGISTRY`. Dispatch: `samples_per_pixel == 1` → the **deterministic wavefront** tracer (generate → traverse → shade → composite over bounded ray tiles, per-ray state pool-allocated from `ManualMemory`); SPP > 1 → the **Monte Carlo path tracer** megakernel (`raytrace_kernels_taichi.py`). Refraction and several other features exist only on the wavefront path and force routing to it.
- The deterministic renderer can use the **hybrid raster pipeline** for primary visibility: flat triangles and bezier circuits are rasterized into covered fragments (with analytic AA options) while secondary reflection/refraction/shadow continuations stay in the ray-based wavefront system. Do not describe the renderer as either a pure rasterizer or a pure one-primary-ray-per-pixel tracer. The classes under `rendering/primitives/` are used for primitive construction/batching by the active renderer — they are not a separate legacy backend.
- Scene assembly: `raytracing/scene_builder.py` packs all primitives of a batch into contiguous per-geometry-type tensor arrays (the stringly-keyed `merged` dict consumed by the kernel orchestrators) and builds one **STBVH** (spatio-temporal BVH, `stbvh.py`) per geometry type covering all frames of the batch. Do not casually change merged-field widths, ordering, dtype, or lifetime.
- Kernels live in `*_taichi.py` files. Material pipelines and custom scatter (ray-continuation) functions are injected as `ti.template()` parameters — compose user `@ti.func`s into one func and pass **flat** tuples (nested tuples fail).
- Shaders (`shaders/`): Three.js-style `Material`s and per-vertex shaders in Python/torch; per-fragment shading and custom fragment pipelines (`fragment_shaders.py`, `FragmentStage`) execute inside the Taichi shade kernel.
- Feature toggles live in `raytracing/settings.py` as module globals with env-var defaults plus setter functions, surfaced through `SETTINGS.raytracing`. **Read them live** (`rt_settings.X` at call time) — importing them by value at module import freezes them before user code runs (this bug has shipped before).
- Post-processing (`post_processing/`): bloom/glow, FXAA/SMAA, tonemapping. `Camera` (`camera.py`): perspective/orthographic projection, fov/near/far; render code consumes an immutable camera/light snapshot per batch so batch prep for frame batch N+1 can run on a worker thread while N renders (`ALGAN_PREFETCH_BATCHES=0` disables).

### Memory

- `ManualMemory` (`utils/memory_utils.py`): a bump-allocator arena for render-time GPU tensors; callers snapshot/restore pointers to free deterministically. Render out-of-memory retries by shrinking the frame window (`OutOfRenderMemory`).
- **Batch sizing is measured at runtime, not modelled.** `rendering/memory_model.py` fits `peak(n) = a + b*n` to the arena's own high-water mark over rendered chunks, and sizes the next chunk from it. **Nothing describes what gets allocated**, so a new primitive, a new tracer path or a user's own post-process is accounted for the moment it runs — there is nothing to annotate, register or regenerate. This replaced a set of hand-written byte formulas *and* a generated calibration table; do not add either back.
- Consequences worth knowing when changing render code:
  - The **first chunk of a job is ~30% cheaper per frame** than steady state (kernel/allocator warm-up), so the model grows chunks geometrically (`PROBE_GROWTH`) and fits from the two *largest* samples rather than extrapolating off the first.
  - Batches land on different lines when the frame buffer or geometry scale changes; `chunk_signature` keys that, with geometry bucketed logarithmically so ordinary scene drift keeps a usable fit.
  - The **OOM retry is the backstop and must stay** — the model measures the batch's first frames and cannot see a scene that densifies later.
- The merge and projection build *outside* the arena in pool headroom, so the model cannot see them; they keep the deliberately generous `MERGE_GPU_PEAK_FACTOR` / `PROJECT_GPU_PEAK_FACTOR` bounds on their packed inputs.
- `ManualMemory.scope()` / the allocation recorder are **diagnostics only** — they do not participate in batch sizing. Use them to attribute arena usage per stage when investigating; do not add scopes expecting them to affect a render's memory budget.
- The whole package runs under a process-global `torch.inference_mode()` entered at import. **Importing algan disables autograd for the process** — never share a process with torch training.

## Development Notes

### Taichi gotchas (these cost real debugging time)
- The offline kernel cache does **not** invalidate on `@ti.func` edits — clear it before A/B-benchmarking kernel changes with `clear_cache(taichi_kernels=True)`.
- Never edit `*_taichi.py` while a render process or warm daemon is running: the JIT reads files at first launch and can compile half-edited code. Restart the daemon after changing any Algan source.
- Cold kernel compilation takes minutes (the Monte Carlo path tracer is a separate kernel with its own cold compile); compiled kernels are cached.
- Keep Taichi debug mode off (`ALGAN_TI_DEBUG=1` opts in); debug mode makes the megakernels ~11x slower.
- In kernels, use `ti.static(bool(x))` rather than `is not None` for template gates.

### Initialization-only settings
These are read while Torch/Taichi initialize, so they must be set **before** `import algan` and have no runtime Python object: `ALGAN_ANIMATION_DEVICE`, `ALGAN_RENDER_DEVICE`, `ALGAN_HOME`, `ALGAN_CACHE_DIR`, `TI_OFFLINE_CACHE_FILE_PATH`, `ALGAN_SOFT_SHADOW_SAMPLES`, `ALGAN_HDR_BUFFER_F16`. `SETTINGS.computing.set(render_device=...)` raises with that instruction rather than a generic "unknown setting".

### Performance discipline
- **`DESIGN_optimization_targets.md` is the plan of record for render performance.** It opens with a status table, how to reproduce the reference profile, and what to verify. Read it before starting (or resuming) optimization work, and update it when something lands.
- Optimizations must target general moving scenes, not static-only fast paths.
- The standard for optimizations is **byte-identical output** validated by an A/B parity script (see the `benchmarks/_*_check.py` / `_*_ab.py` conventions); features are gated behind settings toggles so the default path stays byte-identical.
- One shipped exception, deliberately taken: the subdivision-level criterion kernels (`pn_criterion_kernel`, default on) run under Taichi's `fast_math`, so they flip a handful of borderline tessellation levels and **moved three full-render baselines**. Bit-identity is not recoverable per-kernel there. Note what this implies generally: a change to tessellation, projection or a level criterion is **invisible to `--fast`** — `tests/fast/scene.py` has no PN geometry — so it needs `pytest -q tests/full_renders`.
- Wall-clock kernel timing is noisy (thermal throttling swings cross-process throughput ~2x); use in-process alternating A/B runs or kernel-profiler device times. `utils/profiling_utils.py` auto-hooks all Taichi kernels and pipeline stages.

### Dependencies
Core: torch, torchvision, taichi, numpy, opencv-python, moviepy, scipy, svgelements. Vendored third-party code lives in `algan/external_libraries/` (manim, ground, sect) — treat it as read-only.

## File Structure

- `algan/scene.py` — Scene container, `active_scene_method`, `save_video`/`save_frame`
- `algan/scene_manager.py` — the one singleton: active-Scene stack
- `algan/render_loop.py` — frame batching, prefetch, memory preflight, video streaming
- `algan/animation_timeline/` — contexts, per-Scene timeline, event replay, updaters
- `algan/animatable_base/` — `Animatable`, `Mob` and its mixins
- `algan/animations/` — built-in composable animations
- `algan/mobs/` — all renderable object classes
- `algan/rendering/` — camera, lights, ray tracer + Taichi kernels, shaders, post-processing
- `algan/rendering/memory_model.py` — runtime chunk-peak model that sizes render batches
- `algan/constants/` — spatial (UP, RIGHT, ORIGIN...), colors, rate functions
- `algan/settings/` — `SETTINGS` sections, presets, startup-only env configuration
- `algan/utils/` — tensor helpers, memory arena, profiling, doc-build tooling
- `algan/external_libraries/` — vendored manim/ground/sect (do not modify)
- `tests/unit_tests/` — behavioural tests; `tests/fast/` — the fast suite's one pixel-compared render; `tests/full_renders/` — five dense pixel-compared scenes (see `tests/README.md`)
- `benchmarks/` — ad-hoc A/B, parity-check and profiling scripts
- `docs/` — Sphinx docs with rendered examples
