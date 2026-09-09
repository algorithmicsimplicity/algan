# AGENTS.md

This file provides guidance to Claude when working with code in this repository.

This file is the operational quick-start: commands, hazards, and the API shape. It is short on purpose — the detail
lives in `agent_guidance/`, split by topic so you read only what your task touches:

| Touching | Read |
| --- | --- |
| `animation_timeline/`, recording, replay, materialization, updaters, audio | `agent_guidance/timeline.md` |
| `mobs/`, `animatable_base/`, packing, circuits, PN patches, tessellation | `agent_guidance/mobs_geometry.md` |
| `rendering/`, any `*_taichi.py`, shading, shadows, colour, post-processing | `agent_guidance/rendering.md` |
| `ManualMemory`, batch sizing, optimization work, A/B parity fixtures | `agent_guidance/memory_perf.md` |
| public names, `SETTINGS`, output paths, `ALGAN_` variables | `agent_guidance/api_settings.md` |
| Manim compatibility | `agent_guidance/manim_compat.md` |
| measuring on a GPU (Mac runner, Kaggle T4) | `agent_guidance/gpu_harnesses.md` |
| `*_taichi.py` | `agent_guidance/taichi.md` |

When the docs disagree, the source code wins.

## Project Overview

Algan is a 3D animation engine for explanatory math videos, designed as a successor to Manim: it keeps Manim's ease of use while providing the 3D graphics capabilities of Three.js. It uses PyTorch for animation math and Taichi-language kernels for ray-traced and path-traced rendering on CPU or GPU.

Algan is **lazy**: running a user script does not compute animations, it *records*
them on the Scene's timeline. `Scene.save_video()` materializes that recording in
batches of frames, builds render primitives, and renders them.

Each `Scene` owns its actors, camera, lights, **its own** `TimelineManager`, `AnimationManager` and `AudioManager`, its
video settings, and the render loop it inherits from `RenderLoopMixin`. Only `SceneManager` is a singleton — it holds
the process-global stack of active Scenes.

## Commands

### Running Python

Run the venv interpreter directly — `.venv/Scripts/python.exe` on Windows,
`.venv/bin/python` elsewhere.

**Do not use bare `uv run`** with a locally built compiler wheel. A sync can
replace that build with the version in `uv.lock`. Use the environment's Python
or `uv run --no-sync`; the GPU harnesses also set `UV_NO_SYNC=1`.

The locked distribution is **`algan-quadrants==1.3.0.post2`**; the import name
remains `quadrants`. That published distribution already contains Algan's
patches. Diagnostic builds can instead have the distribution name `quadrants`.
Do not install both distributions over the same import package. Follow
`quadrants_patches/PYPI.md` for published versions and
`quadrants_patches/README.md` for local build/install commands. Verify the
installed distribution and compiler flags; a version banner alone does not
identify the patch set.

### Testing
```
.venv/Scripts/python.exe -m pytest -q --fast    # curated, opt-in development suite
.venv/Scripts/python.exe -m pytest -q           # all configured suites; requires matching render baselines
```
- **`--fast` is the suite to run after every change.** It is **opt-in**: only tests marked `fast` run, everything else is deselected. It prints where it landed against a 75s budget (`fast suite: 21s of its 75s budget (28%)`). Pass no path — it uses `testpaths` from `pyproject.toml`.
- **A test you add is outside it unless you mark it.** Mark `fast` only when a change *elsewhere* in the codebase is liable to break the test — the timeline, the Mob base, the Scene, anything that records or materializes state. A test that only fails when its own module changes is a feature test: leave it unmarked. Being cheap is not a reason. `tests/README.md` lists what is in and why.
- What is in: the timeline (recording/replay/state query/materialization), lifespans, rate functions, Mob transforms + hierarchy + layout, Scene containment, `SETTINGS`, the public authoring surface (`test_ux_regressions.py`), and **one real render compared pixel-wise** (`tests/fast/`). That render is the only thing in the loop that can see a renderer regression, and it is most of the budget.
- **Separate cold compilation from warm test time.** The first test to use a kernel variant pays its compile or cache-load cost. Repeat representative runs before changing fast-suite membership; test counts and timings vary with the checkout, backend and cache state.
- Run the **full** suite after touching the renderer, and before pushing. CI runs the portable subset, `tests/unit_tests tests/fast`, without `--fast`; it does not run the heavy full-render and path-traced baseline suites.
- Renders are compared **pixel-wise** against device/mode-specific baselines: the fast baseline is committed, while heavy suites resolve their release-hosted assets. See `tests/README.md` for baseline selection. Any channel deviation > 2 fails; diff videos land in the suite's `output_errors/`.
- Small (≤2) pixel differences across runs are expected and tolerated: torch CPU rate-function evaluation rounds differently depending on materialization window, so exact byte-identity across re-windowed state is unattainable.
- On Windows, run render work **one process at a time**: killed/timed-out background runs orphan child processes that keep output mp4s locked.
- When a legitimate rendering change alters output, re-baseline with `ALGAN_UPDATE_FAST_BASELINE=1` / `ALGAN_UPDATE_FULL_RENDER_BASELINES=1` and **look at the result** before committing (this is normal practice here).
- `tests/full_renders` and `tests/path_traced` baselines are **hosted as release assets rather than committed to git** (they are most of the repository's history and CI never compares them; `tests/fast` stays in git). A rebaseline of either is not finished until `scripts/package_baselines.py --tag ...` has been run and its tarballs uploaded — `tests/README.md`, "Where the heavy baselines live". An unresolvable baseline **fails** the comparison. The explicit macOS unbaselined opt-out and CI exclusions can still skip comparisons; inspect skips before claiming pixel coverage.
- **Cap any script whose tensor sizes come from parameters** rather than from a real scene: `benchmarks/_memory_cap.py`'s `cap_process_memory(gb)` (call it *before* importing torch). A mis-sized synthetic generator has exhausted system RAM and blue-screened this machine. Do **not** cap a real render — WDDM charges the VRAM arena against process commit, so a capped render segfaults inside CUDA instead of raising.
- A change to tessellation, projection or a level criterion is **invisible to `--fast`** (`tests/fast/scene.py` has no PN geometry) — it needs `pytest -q tests/full_renders`.

### Documentation
- Build: `.venv/Scripts/python.exe docs/make_and_open_docs.py` (Sphinx; renders every embedded example video, so it is slow). Add `--skip-examples --no-open` for structural/autodoc checks.
- Source in `docs/source/`. API stubs in `docs/source/reference/` are autosummary-generated.
- **Docstrings on user-facing API follow `DOCSTRINGS.md`** — read it before writing or editing a public docstring. It is prescriptive, not a description of current code: NumPy style with types in annotations only (never repeated in the docstring), every default stated in prose, units/shapes mandatory, an `Animation` section stating recorded-vs-immediate and spawn-order constraints, and `.. algan::` examples that call `Scene.save_video()` exactly once.

### Linting — read before running ruff
- **`*_taichi.py` files are linted but never formatted.** They must keep the `_taichi` suffix: the config keys three things off it. `I002` is off there because the `from __future__ import annotations` it would insert turns a kernel's runtime-evaluated annotations (`ti.f32`, `ti.types.ndarray()`) into strings and breaks compilation. `SIM` is off because its advice is unsound in a kernel — `SIM109`'s `x in (a, b)` is a `TaichiSyntaxError`, and `SIM102` collapsing `if ti.static(gate): if cond:` into one `and` turns a compile-time gate into a runtime one. And `[tool.ruff.format]` excludes them outright, so `ruff format` never rewraps a kernel body.
- Ruff's `F401` fix is the one to watch in kernel modules: they re-export names to each other (`wavefront_kernels_taichi` gets `max_shadow_lights` via `raytrace_kernels_taichi`), and dropping an "unused" import breaks the import at load time. Mark a deliberate hop `# noqa: F401` with a comment saying who consumes it.

### Authoring Algan

```python
from algan import *

square = Square().spawn()  # mobs must be spawned to appear / animate
square.move(RIGHT)  # recorded as a 1-second animation

with Sync():  # play simultaneously
    square.rotate(90, OUT)
    square.color = BLUE

Scene.save_video("example")  # -> algan_outputs/example.mp4
```

- **Output**: `Scene.save_video(file_path=None, video_settings=None, *, overwrite, reset, background, animate_fade_out, post_processes, codec, audio_codec, ffmpeg_params)` and `Scene.save_frame(file_path=None, video_settings=None, at=None, *, overwrite, background, post_processes)`. Both return `RenderResult`; `save_frame` returns a list only when `at` is a sequence. There is no module-level `render_to_file`/`render`, no `render_settings` keyword, and no `RenderSettings` alias.
- By default, rendering preserves authored state so you can render again. `save_video(reset=True)` explicitly resets the Scene; requested fade-out and the zero-duration video guard can record timeline events. See `agent_guidance/api_settings.md` for the preservation contract. A one-off quality override is `Scene.save_video("example", HD)`; `Scene.view()` opens the interactive viewer instead of writing a video.
- **Viewer**: `Scene.view(video_settings=None, *, port, open_browser, block)` — reached from the Scene only. There is deliberately **no module-level `view`**: the name is far too general to spend on a star-import, and `algan.__all__` is a curated namespace a user dumps into their own. `scene.view(...)` and `Scene.view(...)` are the same method. It serves a local page that plays the Scene, shows its mob hierarchy and attributes at the playhead, and reports the depth-sorted fragment list behind any pixel. Frames render lazily, nothing is written to disk, and the Scene is left as authored. It renders at `PREVIEW`'s resolution but the Scene's own frame rate, so the frame indices it reports are the video's. `block=True` (the default) serves until Ctrl-C — on the warm daemon that occupies it for the duration, since the daemon runs one script at a time.
- **Settings**: one process-global `SETTINGS` with sections `video`, `style`, `paths`, `computing`, `raytracing`. Sections have stable identity — mutate with `SETTINGS.video.set(HD)`, never `SETTINGS.video = HD`. Presets (`PREVIEW`, `LD`, `MD`, `HD`, `PRODUCTION`, `UHD`, `THUMBNAIL`, `SMOKE_TEST`) are immutable; `HD.set(frames_per_second=60)` returns a copy. `SETTINGS.video`'s fields are `resolution`, `frames_per_second` (`fps`/`FPS`), `supersampling` (`ssaa`/`SSAA`), `fxaa` and `audio_sample_rate`.
- **`SETTINGS.raytracing`** holds what the renderer *produces* (`samples_per_pixel`, `max_bounces`, `shadows`, lighting, tonemapping). The kernel/performance switches live on `SETTINGS.raytracing.experimental` and setting them on the parent raises with a pointer. Engine code still *reads* everything off `SETTINGS.raytracing` directly — only writes are gated.
- **`Scene.foo(...)` and `scene.foo(...)`** are the same method: `active_scene_method` binds to an instance, or resolves the active Scene when called on the class.
- **Paths**: `SETTINGS.paths.output_root / output_directory / name`. A bare filename goes to the output directory; anything with a directory in it is used as given.
- **`from algan import *` is curated.** Internal helpers are excluded via `_INTERNAL_EXPORT_MODULES` / `_INTERNAL_EXPORT_NAMES` in `algan/__init__.py`. When adding a public name, check it lands in `algan.__all__`; when adding a helper, check it does not.
- Use the Three.js-style material classes (`MeshBasicMaterial`, `MeshStandardMaterial`, `MeshPhysicalMaterial`, ...) rather than ad-hoc reflectivity/roughness APIs. Shader/material setup (`set_shader`, `set_fragment_shader`, `set_material`) and the geometry declarations (`two_sided`, `casts_shadows`, `receives_shadows`) must all be set **before spawning**.

Full contracts — output-path resolution, the `reset` contract, the settings system, the star-import rule and
API-change discipline — are in `agent_guidance/api_settings.md`.

## Development Notes

### Environment variables
Every `ALGAN_` variable the package honors is declared in `algan/environment.py`, and every read goes through that module's `env_flag` / `env_int` / `env_float` / `env_str` / `env_is_set` accessors, which **reject an undeclared name**. Adding a knob is two steps: put the name in the right tuple in `algan/environment.py`, then read it with an accessor at the point of use. `tests/unit_tests/test_environment.py` enforces that nothing reaches an `ALGAN_` variable through `os` directly.

Some variables are **initialization-only** (set before `import algan`, no runtime object), and variables an A/B script sets before `import algan` do not reach a warm daemon, so it refuses the run. Both lists of record, and the daemon rules, are in `agent_guidance/api_settings.md`. The bar for initialization-only is that **no runtime object could own the value** — not merely that the read happens at import.

**`ALGAN_RENDER_DEVICE` is not one of them.** It seeds `SETTINGS.computing.render_device`, which is the runtime source of truth and is settable between renders; a warm daemon adopts a client's differing value rather than refusing the run. Engine code reads the device with `algan.settings._startup.render_device()` and must **never bind it at import** — `taichi_runtime.ensure_taichi_for_render()` re-selects Taichi's arch at the start of a render job, so a bound copy renders on the wrong device. See `agent_guidance/api_settings.md` (the initialization-only section) and `agent_guidance/gpu_harnesses.md` for how this is checked on a real GPU.

## File Structure

- `algan/scene.py` — Scene container, `active_scene_method`, `save_video`/`save_frame`
- `algan/scene_manager.py` — the one singleton: active-Scene stack
- `algan/render_loop.py` — frame batching, prefetch, memory preflight, video streaming
- `algan/animation_timeline/` — contexts, per-Scene timeline, event replay, updaters
- `algan/animatable_base/` — `Animatable`, `Mob` and its mixins
- `algan/animations/` — built-in composable animations
- `algan/mobs/` — all renderable object classes; `manim_compat` and friends implement the compatibility layer, `manim_adapters` gives a curated subset a native root spelling
- `algan/manim/` — the public face of that layer, reached as `import algan.manim as mn`
- `algan/rendering/` — camera, lights, ray tracer + Taichi kernels, shaders, post-processing
- `algan/viewer/` — `Scene.view()`: the interactive GUI (a local web app), its lazy frame service and its per-pixel fragment inspector
- `algan/rendering/memory_model.py` — runtime chunk-peak model that sizes render batches
- `algan/constants/` — spatial (UP, RIGHT, ORIGIN...), colors, easing curves (`easings`)
- `algan/settings/` — `SETTINGS` sections, presets, startup-only env configuration
- `algan/utils/` — tensor helpers, memory arena, profiling, doc-build tooling
- `algan/external_libraries/` — vendored manim/ground/sect (do not modify by hand; the Manim
  subset is regenerated by `scripts/vendor_manim.py` — see its `manim/VENDORING.md`)
- `tests/unit_tests/` — behavioural tests; `tests/fast/` — the fast suite's one pixel-compared render; `tests/full_renders/` — six dense pixel-compared scenes (see `tests/README.md`)
- `benchmarks/` — ad-hoc A/B, parity-check and profiling scripts
- `docs/` — Sphinx docs with rendered examples
- `agent_guidance/` — the detail references indexed at the top of this file
