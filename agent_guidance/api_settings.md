# Public API, settings and environment

The authoring surface and its stability rules. Read this before adding or renaming a
public name, a setting, or an `ALGAN_` variable.

## Rendering API

The public rendering API is Scene-owned, and these are the only spellings:

```python
scene.save_video(file_path=None, video_settings=None, *, overwrite=True, reset=False,
                 background=None, animate_fade_out=None, post_processes=None,
                 codec=None, audio_codec=None, ffmpeg_params=None)
scene.save_frame(file_path=None, video_settings=None, at=None, *,
                 overwrite=True, background=None, post_processes=None)
```

`Scene.save_video` carries the user-facing signature and documentation; `algan.utils.algan_utils._render_scene_to_file` carries the implementation. Keep them in sync — do not push parameters back into `*args, **kwargs`, because that is what made the signature invisible to `help()`, IDEs and autodoc.

Both return a `RenderResult` (`status`, `output_path`, `duration_seconds`, `render_plan`). `save_frame` returns a list of them only when `at` is a sequence.

`render_plan` is the last batch's `RenderPlan`, also left on `scene.last_render_plan`: which renderer ran, what it could not honor, and `truncations` — a `TruncationCounts` of how often each of the render path's four fixed ceilings bound (`../algan/rendering/raytracing/truncation.py`). Those counters are unconditional and render-job-scoped, so a zero is a reading rather than a missing instrument, and each ceiling warns **once per render** at `WARNING` — not `PERF`, which is for the budget events (batch splits, pool retries) that are the memory model working as designed. A truncation moves the image.

### Output-path resolution

Both still and video output use the same resolver, `_resolve_output_destination`:

- a bare filename is placed under `SETTINGS.paths.output_root / SETTINGS.paths.output_directory`;
- a relative path with an explicit parent and an absolute path are used as supplied;
- a target that already exists as a directory, or that ends with a path separator, is a **directory**: `SETTINGS.paths.output_filename` is placed inside it. This is the same rule `algan render -o` applies (`cli._output_settings`), minus its no-suffix arm — `save_video("intro")` names a file, not a directory;
- missing still-image extensions default to `.png`;
- missing video extensions default to `.mp4` for opaque output and `.mov` for transparent output;
- parent directories are created automatically;
- the returned path is always **absolute**, in every branch, because it is also what `RenderResult.output_path` reports and what the "Finished rendering …" line prints.

The container extension is validated up front by `_check_container_is_supported`, beside `check_codec_is_available` and for the same reason: an unwritable container used to cost a whole render and then surface as a `FileNotFoundError` on the temporary file's rename. `_SUPPORTED_VIDEO_CONTAINERS` and `_SUPPORTED_IMAGE_FORMATS` in `../algan/utils/algan_utils.py` are the lists of record.

`__main__.__file__` is not always a file — `<stdin>` under a pipe, `<string>` under `exec`, `<ipython-input-3-…>` in a notebook. `_main_script_path()` reports anything that is not an existing file as no script at all, so those resolve like `script is None` instead of producing `<stdin>.mp4`.

`output_root` defaults to the directory of `__main__.__file__`, falling back to the working directory; `output_filename` defaults to that script's stem. Do not reintroduce multiple independent `file_name`, `output_path`, and `output_dir` parameters, and do not resurrect `base_directory`.

### `save_frame`

`save_frame` renders one timestamp or a sequence of them (`at`). Multiple timestamps produce files whose names append the timestamp to the resolved stem. Temporary video-settings and background overrides are fully restored afterwards.

`save_frame` never mutates the Scene: nothing is despawned and the timeline is untouched, so it is safe to call repeatedly while authoring. When no timestamp is supplied it renders just after the current authored context time, offset by 1.5 frames. Explicit timestamps must be finite and non-negative.

Keeping the timeline untouched takes more than not recording anything: rendering *resolves* replay windows (`AnimationTimeline._resolve_replay_windows`), freezing each edit's and event's context-rescaled end time into a plain `replay_end` float. From inside an unfinished context those ends are pre-rescale — a `duration` rescales its block retroactively, on exit — and only recording a new edit invalidates them, so a resolution left behind by a mid-authoring render silently truncates the animations of a later render. `save_frame` and `show_frame` therefore wrap their render in `AnimationTimeline.preserving_authoring_state()`, which restores the resolution state (and drops lifespans created for a render's transient mobs). Any render that leaves the Scene re-renderable must do the same — see the `reset` contract below.

### `save_video` and the `reset` contract

`reset` defaults to **False**: the Scene is left exactly as authored. Mobs stay spawned, references stay valid, the timeline keeps its recording, and rendering again produces the accumulated timeline (the earlier animation plus whatever was added). Independent clips need independent Scenes.

Three pieces of finalization are therefore conditional:

- the zero-duration guard (one frame of `wait` for an all-`Off()` scene) always runs, because it decides how many frames are rendered;
- the end-of-scene despawn of every actor runs when a fade-out was requested (it is part of the requested output) or when `reset=True`;
- `_render_to_video` closes the camera and light lifespans only when the Scene is being finalized, via `despawn_camera_and_lights`.

Both lifespans extend past the last rendered frame index either way, so output is unaffected by these gates. `RenderLoopMixin.get_frames` calls `timeline_manager.clear_buffers()` when it finishes, restoring `active_state` to `current_state`; that is what makes a non-reset Scene queryable again after a render.

`reset=False` also passes `preserve_authoring_state=True` into `_render_to_video`, which rolls back the two pieces of state the render itself derives: the appended `scene_times` window, and the replay-window resolution (via `preserving_authoring_state()`, as for `save_frame`). The snapshot is taken around the `get_frames` loop rather than around the whole call, because the fade-out and the zero-duration guard record on the timeline first and edits made after a snapshot would fall outside it.

Together with the conditional finalization above, that makes a `reset=False` render legal **from inside an unfinished block**: render a preview mid-`Seq`/`Speech`, keep authoring, and the final render is identical to one where the preview never happened. The frame window for such a render comes from `_recorded_end_time_for_render()`, which takes the max over the whole open context chain — the innermost open context covers only its own block, while an enclosing `Sync` can already hold animations running past it. Every open context shares one un-rescaled timeframe, so their ends are directly comparable; with all blocks closed this is just the root context's end, exactly as before.

With `reset=True` the Scene's timeline, animation and audio managers are rebuilt in `finally` on both success and failure, and authored mobs must not be reused. `overwrite=False` returns a skipped result without finalizing anything. Harnesses that re-author a scene per run (profilers, repeated benchmark passes) should pass `reset=True` explicitly.

Transparent output cannot use MP4. Use MOV or WebM, or an opaque background.

### Scene's public half and its engine half

A `Scene` is both the thing a script authors against and the object the render loop drives, and the two
sets of methods are told apart by a leading underscore. Public: `save_video`, `save_frame`, `view`,
`show_frame`, `wait`, `add`, `add_actor`, `add_effect`, `reset`, `current`, `set_background`,
`get_background`, `set_environment_map`, `set_video_settings`, `background_is_transparent`,
`get_camera`, `add_light`/`remove_light`/`clear_lights`, `length_to_pixels`/`pixels_to_length`,
`despawn_mobs`, `save_audio`, `use_manim_defaults`, `render_all_funcs`, and `get_frames` (the render
loop's entry point, which the viewer and the benchmarks both drive).

Engine-only, and therefore private: `_get_batch_of_primitives`, `_render_primitive_batch`,
`_render_background_batch`, `_batch_prep_context`, `_render_to_video`, `_initialize_frames`,
`_set_current_time`, `_increment_current_time`, `_update_max_time`, `_set_time_to_latest`,
`_get_new_id`, `_get_pixel_format`, `_terminate` and `_instance`. The last two have public
counterparts a script should reach for instead — a `with Scene() as scene:` block for the first,
`Scene.current()` for the second, which is now the one spelling for the active Scene.

### Scene-function discovery

Use the `@algan_scene` decorator for zero-argument scene entry points consumed by `render_all_funcs`. It is deliberately not named `scene`, which would collide with the conventional variable name for a Scene instance. Legacy implicit discovery of every zero-argument function remains as a warning-producing fallback and may accidentally render helpers.

`render_all_funcs` creates an isolated Scene for each function. Scene functions should either rely on that active Scene or accept no arguments and explicitly obtain it; helper constructors should still propagate Scene ownership from their inputs.

## Settings system

Runtime-adjustable public configuration is rooted at the stable process-global `SETTINGS` object:

- `SETTINGS.computing`;
- `SETTINGS.paths`;
- `SETTINGS.style`;
- `SETTINGS.video`;
- `SETTINGS.raytracing`.

Those five are the whole of it. `AlganSettings.__slots__` also carries `_skip_save_frame`, which the docs build sets so that an example's `save_frame` call renders nothing; it is an engine flag rather than a setting, so it is underscored and kept out of `dir()`, `repr()` and `snapshot()` — the list of sections comes from `AlganSettings._SECTIONS`, not from `__slots__`.

`SETTINGS.video`'s fields are `resolution`, `frames_per_second`, `supersampling`, `fxaa` and `audio_sample_rate`. `SETTINGS.paths`'s are `cache_directory`, `output_root`, `output_directory`, `output_filename` and `ffmpeg_binary`.

`ffmpeg_binary` **outranks every other candidate, for every codec**. The reason to pin a binary is that moviepy's bundled build lacks a codec yours has, so it has to beat the probe rather than join it; leave it `None` (the default) and encoder selection is byte-for-byte what it was before the setting existed. It replaces a monkey-patching `override_moviepy_ffmpeg_binary()` function that used to be star-imported — configuration belongs in `SETTINGS`, not in a global-mutating call.

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

A field may declare **aliases**, and `SETTINGS.video` is the one section that does: `fps`/`FPS` for `frames_per_second`, and `ssaa`/`SSAA` for `supersampling`. The mechanism is the `settings_aliases` class decorator in `algan/settings/abstract_settings.py`, applied outside `@dataclass` so it wraps the generated `__init__`; the alias is resolved to the declared name once, at each entry point (`__init__`, `set`, `__setattr__`), and everything downstream — validation, `dataclasses.replace`, the write-back loop — sees declared names only. So an alias is a *spelling*, not a field: it is absent from `to_dict()`, from `dataclasses.fields` and from `SETTINGS.snapshot()`, which is what lets state saved through one spelling restore through the other. Naming one field by two spellings in a single call raises rather than resolving to whichever came last.

This is one of the four deliberate exceptions to "there is one Algan name for each Algan thing" — the others being the `algan.manim` layer, `IN`/`OUT`, and `Mob`'s `.right`/`.up`/`.forward` direction properties. Do not add an alias for a field because its name is long; these two exist because the abbreviations are what the rest of the world calls them. Library code writes the declared name.

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
- `ALGAN_TI_DEBUG`, `ALGAN_TAICHI_WARMSTART`, `ALGAN_TAICHI_FAST_LAUNCH`;
- `ALGAN_TAICHI_BACKEND`.

`ALGAN_TAICHI_BACKEND` selects which Taichi-language compiler builds the kernels:
`taichi` (the default, 1.7.x) or `quadrants`, the maintained fork. Every engine module
reaches the compiler through `algan.taichi_compat` (`from algan.taichi_compat import ti`,
and `submodule("lang.impl")` for a submodule) rather than importing `taichi` directly, so
the choice is made once and a process with **both** live -- two runtimes, two CUDA
contexts, two kernel caches -- cannot be spelled. Do not add a bare `import taichi` to
`algan/`. Each backend gets its own offline-cache directory (`cache/<backend>`), and
`algan.taichi_compat` owns the places where the two spell the same thing differently
(`kernel_specializations()` for `compiled_kernels` vs `materialized_kernels`).
It is startup-only in the strictest sense -- the kernels in the process are already
compiled by the chosen backend -- so the daemon refuses a client whose value differs.

`ALGAN_HDR_BUFFER_F16` is **not** one of them any more either. It seeds `SETTINGS.raytracing.experimental.hdr_buffer_f16`, and `hdr_frame_dtype()` reads that when the frame buffer is allocated — no kernel specializes on it, so there was never anything for the import to bake in.

`ALGAN_RENDER_DEVICE` is **not** one of them any more. It seeds `SETTINGS.computing.render_device`, which owns the value from then on and can be changed between renders; `taichi_runtime.ensure_taichi_for_render()` re-selects Taichi's arch at the start of each render job when the device has moved across the CPU/GPU line. Read it with `algan.settings._startup.render_device()` — never bind it at import, which is the mistake the old `_RENDER_DEVICE` constant made unavoidable. A change is refused while a render is running and once a wide attribute (a texture) has been placed on the render device.

`RENDERER_REGISTRY` and `KERNEL_REGISTRY` are runtime service registries, not user settings, and therefore live outside `SETTINGS` and outside the star-import namespace.

`SETTINGS.computing` accepts `render_device`; it rejects `animation_device` with a message naming the environment variable to use instead.

`SETTINGS.computing.torch_compile` runs the pipeline's per-frame torch arithmetic through `torch.compile`. It is the same tri-state shape as `mps_friendly`: `'auto'` (the default) resolves to `algan.utils.torch_compile.torch_compile_support()` — on wherever `torch.compile` runs, off on Windows and on a Python Dynamo does not support — and `True`/`False` decide for themselves; `ALGAN_TORCH_COMPILE` overrides both. The mechanism is one decorator, `algan.utils.torch_compile.compiled`, which reads the switch **at every call** (so it flips between two renders in one process and the daemon adopts it), builds the compile lazily, and on a compile failure warns once and demotes that function to eager for the rest of the process. The rules for what may go inside a compiled region — pure torch, no arena calls, no `.item()`/`bool(tensor)` control flow, live settings passed in as plain arguments, `Color` converted to a plain tensor — are in the module docstring; `benchmarks/_torch_compile_ab.py` is the warm alternating A/B with frame parity, and `tests/unit_tests/test_torch_compile.py` pins the switch and the fallback contract.

**Before compiling anything else, price it with `benchmarks/_compile_candidates_ab.py`** — it wraps any `module:qualname` for one real render, runs both arms on every call (the render consumes the eager result, so it is unperturbed), and reports per-call time in each arm, what compiling would take off the render, and whether the two arms agree bit for bit. `--pn-controls` on the A/B does the same at whole-render scale for the three PN control-net builders. What a survey with it found (PN fixture and `tests/fast/scene.py`, PREVIEW, CPU, 4 cores), so the same ground is not re-broken:

- **The whole-render A/B cannot resolve a single function on these scenes.** A warm PREVIEW render of the PN fixture is ~4 s, of which `raster: sparse discovery` is 60% and every torch region the switch touches is ~1–2%; the shipped compiled set is worth ~20 ms there (`evaluate_logical_pn` 0.016 s eager against 0.006 s compiled in the stage profile) and measures 1.00x end to end, inside the ±0.2 s run-to-run spread. Serialising the prefetch (`ALGAN_PREFETCH_BATCHES=0`) does not change that — the prep is small, not hidden. Judge a candidate per call, and quote the whole-render number only as the share it is.
- **Timeline materialization is not a candidate.** `_query_row_states` — what a frame batch actually runs, `generate_array_states` is only reached with `ALGAN_OPT_DISABLE=torchquery` — is a `searchsorted`/gather chain with data-dependent shapes: bit-identical compiled but **0.4–0.5x**, and it is 0.2% of a warm render (12 calls, 0.14 ms each). `generate_array_states` is 0.39x held at one shape and recompiles on nearly every call in situ. There is no arithmetic chain here for Inductor to fuse.
- **Mob attribute accessors are not a candidate either**, and not because they are slow to compile: `AttributeTimeline.get` is 504 calls and **0.010 s** of a 4 s render, and its body is `isinstance` branches, a slice and a clone. `get_animated_attribute` / `_setattr_and_record_modification` around it are dict and index bookkeeping with no tensor arithmetic at all. Dynamo's per-call guard check is the same order as the work.
- **Geometry helpers pay, but not much.** `_circuit_parity_gathered` is 1.8x and bit-identical (~3 ms on the fast scene); `_evaluate_cubic_bezier_batch` 1.4x, bit-identical, and called twice a render. Both are worth having only if something makes the bezier build a larger share than it is. `mean_patch_edge_length` is 0.5–0.9x — too small to fuse — and differs from eager, so it is doubly out.

`SETTINGS.computing.mps_friendly` restricts the renderer to operations Apple's Metal backend can run — float32 for every float64 accumulator, int32 for the int64 min/max reductions, a log-step scan for `cummax`/`cummin` (`../algan/rendering/DESIGN_mps_support.md` §1.2 measured all three gaps). It is a **tri-state**: `'auto'` (the default) turns the mode on exactly when the render device is MPS, and `True`/`False` decide for themselves — which is what makes the mode testable on a machine with no Apple GPU, and is how `tests/unit_tests/test_mps_friendly.py` and the Linux control arm of `mps_probe.yaml` exercise it. `ALGAN_MPS_FRIENDLY` overrides both. The resolution and every substitution live in `algan/rendering/mps_compat.py`, and engine code asks it rather than testing the device: `accumulate_dtype()`, `reduction_index_dtype()`, `reduction_index_sentinel()`, their `taichi_*` twins for the kernels' `ti.template()` dtype arguments, and `cummax_values`/`cummin_values`. **The mode is not deterministic** — the accumulators it narrows are the §6.6.4 ones, widened precisely because a float32 sum is not order-reproducible — so it stays off wherever float64 exists. `test_mps_friendly.py` walks the AST of `algan/rendering/` and fails if any module but `mps_compat` names `torch.float64`, `ti.f64`, `.double()` or `cummax`/`cummin`.

## API-change discipline

- `Scene.save_video` / `scene.save_video`
- `Scene.save_frame` for stills, with `at` rather than `time_stamps`;
- `video_settings` / `VideoSettings` — `render_settings`, `RenderSettings` and `set_render_settings` are gone;
- one `file_path` rather than separate filename/directory arguments;
- `SETTINGS` sections rather than the old defaults globals;
- Scene-owned managers rather than singleton managers;
- `DrawBorderThenFill(mobs)` rather than `write(mob)`; it takes any iterable of Mobs, and `Tex`/`Text` expose `.write()` as the glyph-wise shorthand;
- `import algan.manim as mn` for the compatibility layer — it is not star-imported, and `mn.X` is under Manim's conventions where root `X` is under Algan's (see `CLAUDE.md`, "The `algan.manim` boundary");
- `runtime` rather than `duration` / `run_time`, and `easing` / `easings` rather than `rate_func` / `rate_funcs`;
- `mob` rather than `mobject`, and `element_to_mob` rather than `element_to_mobject`, on every root callable that takes one; `SVGMob`, `MobMatrix`, `MobTable`, `DashedMob` and `CurvesAsChildren` rather than Manim's `Mobject`-spelled class names. Passing an old spelling at the root raises `AlganConfigurationError` naming the new one — the mechanism is `algan/utils/api_renames.py`, applied by the adapters' generated `__init__` and by the `@_renamed_keywords` decorator on the animations. All of it still works under `algan.manim`, which is Manim's conventions by design;
- one vocabulary across the revolved solids: `radius`, `u_range` / `v_range`, `closed`, and `direction=UP` on both `Cone` and `Cylinder`. Manim's `base_radius`, `show_base`, `show_ends`, `u_min` and `checkerboard_colors` raise — a checkerboard is a `color_texture` here (`get_checkerboard(...)`, in `algan/mobs/surfaces/procedural_textures.py`), so the pattern's detail comes from the map rather than from the tessellation; `resolution` is the one Manim name kept, because it means something Algan has no other word for (patches, where `grid_width`/`grid_height` count vertices);
- `RegularPolygon(n=...)` rather than `num_vertices=`, and `Dot(location=...)` rather than `point=`, matching `Mob.location`;
- `stroke_width` / `stroke_color` rather than `border_width` / `border_color`, in Algan's unit — Manim's is twice it, and that conversion exists only at the `algan.manim` boundary.

Do not add a second spelling for something that already has a name. If a rename is genuinely warranted, rename in place and update every call site — the project is pre-release specifically so this stays cheap.

The one Algan-side pair that stays is `IN = INWARD` / `OUT = OUTWARD`, and it earns its keep by taking a name *out* of the library rather than adding one to it: `in` and `out` are words a script will want, so the short spellings are the script's to shadow and Algan's source reads only the long ones. `../tests/unit_tests/test_spatial_constants.py` enforces that. Write `OUTWARD` in `algan/`; write `OUT` in docs and tests.

### The star-import namespace is the API

`from algan import *` is the documented entry point, so `algan.__all__` is effectively the public surface. `algan/__init__.py` builds it from a rule plus two deny-lists (`_INTERNAL_EXPORT_MODULES`, `_INTERNAL_EXPORT_NAMES`) and one allow-list (`_EXTRA_EXPORTS`). Generic helper names must not leak: `mean`, `interpolate`, `offset`, `shuffle`, `broadcast*`, `traverse`, `squish` and friends would shadow whatever the user imported before Algan.

A name that belongs to the Manim compatibility layer goes in `algan/manim/` and stays out of `algan.__all__` entirely; if it is something an author reaches for directly and Algan has no native version, give it a root spelling through `algan/mobs/manim_adapters.py` instead of exporting the wrapper. That is where `manim_fov`, `manim_shader` and `ManimMaterial` live: each means "Manim's version of this", so `algan.manim` is their home and `use_manim_defaults()` installs all three at once.

`algan.manim.__all__` is curated the same way, by `_INTERNAL_MANIM_EXPORTS`: the `OpenGL*` aliases and the `MANIM_*_NAMES` parity registry stay reachable as attributes (`tests/unit_tests/test_manim_mobject_parity.py` reads them) but are not part of that module's documented surface — one is ~40 second spellings of classes already there, the other is inventory data.

**A root spelling is not a delegation.** An adapter carries its own `__signature__` and its own docstring, built in `manim_adapters._root_signature` / `_root_docstring`: displayed angle defaults are in degrees, the displayed `stroke_width` default is in Algan's unit, Manim's type aliases are dropped from the annotations, and Manim's prose is *replaced* rather than appended to, so no `.. manim::` block (a `class X(Scene)` calling `self.play`) reaches Algan's reference pages. Those bodies are generated from Manim's summary line plus the converted parameter list, and their `Notes` section says so; a hand-written one goes in `_WRAPPER_DOCSTRINGS` in `algan/mobs/manim_compat.py` (`MathTex` and `Title` have them) and is used in preference.

A supplied angle that is a non-integer float smaller than a full turn warns with `ApproximationWarning`, because `Arc(angle=PI/2)` is a legal 1.57 degree sliver and nothing else in the system would say so. Whole numbers never warn.

When you add a name, decide which side it is on. Public mobs, animations, contexts, materials, shaders, constants and settings belong in the namespace; tensor utilities, mixins, primitive builders, registries and dev tooling do not. `../tests/unit_tests/test_ux_regressions.py` asserts both directions.

When changing a public class, method, setting, material field, or render argument:

- update root exports in `algan/__init__.py` as needed;
- update the docs; the `docs/source/reference/` autosummary stubs are generated at build time and gitignored, so there is nothing to hand-edit there, but a renamed symbol still breaks any `:meth:`/`:class:` cross-reference that names it;
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

`ImageMob`, `set_texture` and `background` all route through `file_utils.get_image` → `resolve_asset_path`, which
tries the working directory and then the main script's directory, so an image beside your script loads regardless of
where you launch Python.
