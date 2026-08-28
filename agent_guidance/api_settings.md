# Public API, settings and environment

The authoring surface and its stability rules. Read this before adding or renaming a
public name, a setting, or an `ALGAN_` variable.

## Rendering API

The public rendering API is Scene-owned, and these are the only spellings:

```python
scene.save_video(file_path=None, video_settings=None, *, overwrite=True, reset=False,
                 background_color=None, animate_fade_out=None, post_processes=None,
                 codec=None, audio_codec=None, ffmpeg_params=None)
scene.save_frame(file_path=None, video_settings=None, at=None, *,
                 overwrite=True, background_color=None, post_processes=None)
```

`Scene.save_video` carries the user-facing signature and documentation; `algan.utils.algan_utils._render_scene_to_file` carries the implementation. Keep them in sync — do not push parameters back into `*args, **kwargs`, because that is what made the signature invisible to `help()`, IDEs and autodoc.

Both return a `RenderResult` (`status`, `output_path`, `duration_seconds`, `render_plan`). `save_frame` returns a list of them only when `at` is a sequence.

`render_plan` is the last batch's `RenderPlan`, also left on `scene.last_render_plan`: which renderer ran, what it could not honor, and `truncations` — a `TruncationCounts` of how often each of the render path's four fixed ceilings bound (`../algan/rendering/raytracing/truncation.py`). Those counters are unconditional and render-job-scoped, so a zero is a reading rather than a missing instrument, and each ceiling warns **once per render** at `WARNING` — not `PERF`, which is for the budget events (batch splits, pool retries) that are the memory model working as designed. A truncation moves the image.

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

Unknown field names are rejected with a close-match suggestion by both `set(...)` and direct attribute assignment — `SETTINGS.video.frame_rate = 60` raises rather than silently attaching a junk attribute.

A field may declare **aliases**, and `SETTINGS.video` is the one section that does: `fps`/`FPS` for `frames_per_second`, and `ssaa`/`SSAA` for `super_sampling_anti_aliasing`. The mechanism is the `settings_aliases` class decorator in `algan/settings/abstract_settings.py`, applied outside `@dataclass` so it wraps the generated `__init__`; the alias is resolved to the declared name once, at each entry point (`__init__`, `set`, `__setattr__`), and everything downstream — validation, `dataclasses.replace`, the write-back loop — sees declared names only. So an alias is a *spelling*, not a field: it is absent from `to_dict()`, from `dataclasses.fields` and from `SETTINGS.snapshot()`, which is what lets state saved through one spelling restore through the other. Naming one field by two spellings in a single call raises rather than resolving to whichever came last.

This is the **one** deliberate exception to "there is one Algan name for each Algan thing", alongside `IN`/`OUT`. Do not add an alias for a field because its name is long; these two exist because the abbreviations are what the rest of the world calls them. Library code writes the declared name.

`SETTINGS.raytracing` is split by stability. Directly on the section are the settings that describe what the renderer *produces*: `samples_per_pixel`, `max_bounces`, `shadows`, `ambient_light`, `light_intensity`, `indirect_bounce_strength`, `glossy_reflection`, `analytic_aa`, `tonemapping`, `tonemap_method`, `tonemap_exposure`, `unsupported_feature_policy`. Every other switch is a kernel/performance gate and lives on `SETTINGS.raytracing.experimental`; writing one through the parent raises an error naming the right location. **Reads are deliberately unrestricted** — engine modules bind `rt_settings = SETTINGS.raytracing` once and read experimental switches off it on the hot path — so only mutation is gated. `to_dict()`, `as_preset()`, `_restore()` and `SETTINGS.snapshot()` continue to cover every field.

Adding a renderer toggle is one edit: declare it as a lowercase module-level value with an environment default, in whichever storage module owns that subsystem (`_STORAGE_MODULES` in `algan/settings/raytracing_settings.py` lists them — the toggles module plus the BVH builders, the kernels, the raster and sheet passes, the scene builder, the tracer and the memory model, each keeping its settings beside the code and the comment that explain them). `SETTINGS.raytracing` derives its field set from those modules, so the toggle is reachable with nothing else to register — leave it out of `_PUBLIC_FIELDS` unless it changes rendered output in a way users are meant to control, and it lands on `.experimental`. Two rules follow from the derivation: the value must be a scalar, and **no helper function may share a field's name** — the later `def` silently takes the name over and the field disappears (`test_settings_api.py` pins both).

There is one spelling for each setting. The hand-maintained `_FIELD_TO_LEGACY` map from lowercase fields to UPPER_CASE globals, and the `_SETTER_OVERRIDES` map beside it, are **deleted**: they were a second source of truth that drifted, and the drift is what left nine switches with a global, a setter and no way to set them. Do not reintroduce a table that mirrors the module.

Configuration that the renderer *freezes* when it is imported — an ndarray element type the kernels are annotated with, a reciprocal, a packed header layout, a `ti.static` payload width — is still a field, so it can be read, discovered and snapshotted. Writing one is refused by `_IMPORT_FROZEN_FIELDS`, naming the environment variable to set instead: a host that builds an arity-8 BVH for kernels annotated arity-4 does not fail, it renders wrong, so a refusal beats both a silent no-op and a silent corruption. `_INERT_FIELDS` and `_IMPORT_FROZEN_FIELDS` are both checked against the names a caller passed and never against a restored snapshot, so `set(source=...)` and `SETTINGS.restore()` still round-trip every field.

`tests/unit_tests/test_settings_api.py` pins the whole arrangement: every `env_*`-backed module global in `algan/` is a field (except the four init-only ones in `settings/_startup.py`), no helper shadows a field's name, and every declaration the storage modules make is reachable.

Use `SETTINGS.snapshot()`/`SETTINGS.restore()` for complete public-settings state capture, and `SETTINGS.override(...)` or section-level `override(...)` for temporary changes. Do not hand-roll partial restoration that leaves live settings leaked into later tests or daemon runs.

`SETTINGS.raytracing` validates every write: the accepted type of each of its ~106 fields is derived from the value it ships with (three polymorphic mode switches are exempted by name), numeric fields carry a lower bound taken from their documented meaning, and floats must be finite. A setter's own `ValueError` is re-raised as an `AlganConfigurationError` naming the field; `UnsupportedFeatureError` passes through unflattened, because it is a distinct type callers catch and *is* a subclass of `AlganConfigurationError`.

Writing a section is one operation with one set of rules: `SETTINGS.video.frames_per_second = 60` routes through `set()`, so assignment validates and normalizes exactly as `set(frames_per_second=60)` does, and `set()` writes back only the fields that actually changed (identity comparison) so an unrelated field keeps its object identity. Assigning a whole *section* (`SETTINGS.video = HD`) is still refused — sections have stable identity.

Engine modules must read mutable settings live through `SETTINGS`. Never import a mutable ray-tracing setting by value at module import time; doing so freezes the old value and makes public setters ineffective. Immutable constants may be imported by value.

Initialization-only settings intentionally have no public mutable Python object. Set these before importing `algan`:

- `ALGAN_ANIMATION_DEVICE`;
- `ALGAN_HOME`;
- `ALGAN_CACHE_DIR`;
- `TI_OFFLINE_CACHE_FILE_PATH`;
- `ALGAN_SOFT_SHADOW_SAMPLES`;
- `ALGAN_TI_DEBUG`, `ALGAN_TAICHI_WARMSTART`, `ALGAN_TAICHI_FAST_LAUNCH`.

`ALGAN_HDR_BUFFER_F16` is **not** one of them any more either. It seeds `SETTINGS.raytracing.experimental.hdr_buffer_f16`, and `hdr_frame_dtype()` reads that when the frame buffer is allocated — no kernel specializes on it, so there was never anything for the import to bake in.

`ALGAN_RENDER_DEVICE` is **not** one of them any more. It seeds `SETTINGS.computing.render_device`, which owns the value from then on and can be changed between renders; `taichi_runtime.ensure_taichi_for_render()` re-selects Taichi's arch at the start of each render job when the device has moved across the CPU/GPU line. Read it with `algan.settings._startup.render_device()` — never bind it at import, which is the mistake the old `_RENDER_DEVICE` constant made unavoidable. A change is refused while a render is running and once a wide attribute (a texture) has been placed on the render device.

`RENDERER_REGISTRY` and `KERNEL_REGISTRY` are runtime service registries, not user settings, and therefore live outside `SETTINGS` and outside the star-import namespace.

`SETTINGS.computing` accepts `render_device`; it rejects `animation_device` with a message naming the environment variable to use instead, and `render_on_cpu` with one naming `render_device`, rather than a generic "unknown setting".

The old `COMPUTING_DEFAULTS`, `DIRECTORY_DEFAULTS`, `STYLE_DEFAULTS` and `RENDERING_DEFAULTS` facades and the `algan.settings.render_settings` / `algan.settings.style_defaults` modules have been **deleted**. Do not reintroduce them.

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

The one Algan-side pair that stays is `IN = INWARD` / `OUT = OUTWARD`, and it earns its keep by taking a name *out* of the library rather than adding one to it: `in` and `out` are words a script will want, so the short spellings are the script's to shadow and Algan's source reads only the long ones. `../tests/unit_tests/test_spatial_constants.py` enforces that. Write `OUTWARD` in `algan/`; write `OUT` in docs and tests.

### The star-import namespace is the API

`from algan import *` is the documented entry point, so `algan.__all__` is effectively the public surface. `algan/__init__.py` builds it from a rule plus two deny-lists (`_INTERNAL_EXPORT_MODULES`, `_INTERNAL_EXPORT_NAMES`) and one allow-list (`_EXTRA_EXPORTS`). Generic helper names must not leak: `mean`, `interpolate`, `offset`, `shuffle`, `broadcast*`, `traverse`, `squish` and friends would shadow whatever the user imported before Algan.

When you add a name, decide which side it is on. Public mobs, animations, contexts, materials, shaders, constants and settings belong in the namespace; tensor utilities, mixins, primitive builders, registries and dev tooling do not. `../tests/unit_tests/test_ux_regressions.py` asserts both directions.

When changing a public class, method, setting, material field, or render argument:

- update root exports in `algan/__init__.py` as needed;
- update docs and checked-in autosummary stubs;
- search docs, tests, examples and benchmarks for stale call sites and fix all of them, since nothing keeps the old name working;
- add or update tests for the new behavior.

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

## Environment variables

Every `ALGAN_` variable the package honors is declared in `algan/environment.py`, and every read goes through that module's `env_flag` / `env_int` / `env_float` / `env_str` / `env_is_set` accessors, which **reject an undeclared name** — that is what lets `import algan` tell a real option from a misspelled one (it warns about `ALGAN_` variables it does not know).

Adding a knob is therefore two steps: put the name in the right tuple in `algan/environment.py`, then read it with an accessor at the point of use, where the default lives next to the comment explaining it. Values parse leniently: an unusable one warns and falls back to the caller's default rather than aborting the render. `tests/unit_tests/test_environment.py` enforces the rule that nothing in the package reaches an `ALGAN_` variable through `os` directly.

### Initialization-only settings

These are read while Torch/Taichi initialize, so they must be set **before** `import algan` and have no runtime Python object: `ALGAN_ANIMATION_DEVICE`, `ALGAN_HOME`, `ALGAN_CACHE_DIR`, `TI_OFFLINE_CACHE_FILE_PATH`, `ALGAN_SOFT_SHADOW_SAMPLES` and the Taichi/warm-start trio. `_STARTUP_VARIABLES` in `algan/environment.py` is the list of record, and the daemon derives its `STARTUP_ENV` from it.

The bar for adding to that tuple is that **no runtime object could own the value** — Taichi is already initialized, the device is already chosen, the constant is already folded into a compiled kernel. "It happens to be read at import" is not the bar: `ALGAN_HDR_BUFFER_F16` sat here for exactly that reason while the dtype it selects is read at buffer allocation, and it is now `SETTINGS.raytracing.experimental.hdr_buffer_f16` with the environment variable seeding the default. `ALGAN_LOG_LEVEL` and `ALGAN_PROGRESS` were import-time for the same non-reason and are now read live, re-applied per run by the daemon (`logger.apply_environment_logging`).

`ALGAN_RENDER_DEVICE` is in that tuple too — it *is* read at startup — but it is also in `_DAEMON_ADOPTED_STARTUP_VARIABLES`, because all it does there is seed `SETTINGS.computing.render_device`. A warm daemon therefore re-applies the client's value per run (`daemon._adopt_render_device`) instead of refusing it, and the run renders where a cold one would. Anything added to that tuple needs both halves — a runtime setting that owns the value, and a daemon that re-applies it — or a mismatched run silently renders wrong.

### Variables an A/B script sets before `import algan` do not reach a warm daemon

The daemon refuses such a run. Most renderer toggles become module-level defaults during the import, which in a daemon happened at its launch — `_IMPORT_TIME_VARIABLES` in `algan/environment.py` is the list of record, checked against the call sites by `tests/unit_tests/test_environment.py`. A client whose values differ is refused and runs cold, matching what it would have rendered on its own; variables read live (`_LIVE_VARIABLES`) are swapped in per run, so flipping one *between* two renders in a script works warm. Benchmarks set `ALGAN_USE_DAEMON=0` anyway, because a warm process also carries the previous run's adaptive renderer state.

## Asset paths

`ImageMob`, `set_texture` and `background_color` all route through `file_utils.get_image` → `resolve_asset_path`, which
tries the working directory and then the main script's directory, so an image beside your script loads regardless of
where you launch Python.
