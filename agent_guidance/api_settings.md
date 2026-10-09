# Public API, settings and environment

The authoring surface and its stability rules. Read this before adding or renaming a
public name, a setting, or an `ALGAN_` variable.

## Rendering API

The public rendering API is Scene-owned. These signatures use `*` to mark
keyword-only parameters; they are not runnable calls:

```text
scene.save_video(file_path=None, video_settings=None, *, overwrite=True, reset=False,
                 background=None, animate_fade_out=None, post_processes=None,
                 codec=None, audio_codec=None, ffmpeg_params=None, passes=None)
scene.save_frame(file_path=None, video_settings=None, at=None, *,
                 overwrite=True, background=None, post_processes=None, passes=None)
```

`Scene.save_video` carries the user-facing signature and documentation; `algan.utils.algan_utils._render_scene_to_file` carries the implementation. Keep them in sync — do not push parameters back into `*args, **kwargs`, because that is what made the signature invisible to `help()`, IDEs and autodoc.

Both return a `RenderResult` (`status`, `output_path`, `walltime_seconds`, `render_plan`, `passes`). A field added to it goes last, with a default: callers construct it positionally. `save_frame` returns a list of them only when `at` is a sequence.

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

`save_frame` never mutates the Scene: nothing is despawned and the timeline is untouched, so it is safe to call repeatedly while authoring. When no timestamp is supplied it renders just after the current authored context time, offset by 1.5 frames. Explicit timestamps must be finite. A negative `at` is an offset from the current authored context time; its resolved timestamp must be non-negative.

Keeping the timeline untouched takes more than not recording anything: rendering *resolves* replay windows (`AnimationTimeline._resolve_replay_windows`), freezing each edit's and event's context-rescaled end time into a plain `replay_end` float. From inside an unfinished context those ends are pre-rescale — a `duration` rescales its block retroactively, on exit — and only recording a new edit invalidates them, so a resolution left behind by a mid-authoring render silently truncates the animations of a later render. `save_frame` and `show_frame` therefore wrap their render in `AnimationTimeline.preserving_authoring_state()`, which restores the resolution state (and drops lifespans created for a render's transient mobs). Any render that leaves the Scene re-renderable must do the same — see the `reset` contract below.

### `save_video` and the `reset` contract

`reset` defaults to **False**: authored state is preserved, except for the explicit finalization operations described below. Mobs stay spawned, references stay valid, the timeline keeps its recording, and rendering again produces the accumulated timeline (the earlier animation plus whatever was added). Independent clips need independent Scenes.

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
`despawn_mobs`, `save_audio`, `add_subcaption`, `save_subtitles`, `use_manim_defaults`, `render_all_funcs`, and `get_frames` (the render
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

Aliases are fine wherever they make Algan easier to write. These two exist because the abbreviations are what the rest of the world calls them, and `settings_aliases` is the way to add another. Library code writes the declared name.

`SETTINGS.raytracing` is split by stability. Directly on the section are the settings that describe what the renderer *produces* — `_PUBLIC_FIELDS` in `algan/settings/raytracing_settings.py` is the list of record: `texture_antialiasing`, `samples_per_pixel`, `max_bounces`, `shadows`, `glossy_reflection`, `glossy_prefilter`, `analytic_aa`, `denoise`, `linear_color_space`, `tonemapping`, `tonemap_method`, `tonemap_exposure`, `unsupported_feature_policy`. Every other switch is a kernel/performance gate and lives on `SETTINGS.raytracing.experimental`; writing one through the parent raises an error naming the right location. **Reads are deliberately unrestricted** — engine modules bind `rt_settings = SETTINGS.raytracing` once and read experimental switches off it on the hot path — so only mutation is gated. `to_dict()`, `as_preset()`, `_restore()` and `SETTINGS.snapshot()` continue to cover every field.

Adding a renderer toggle is one edit: declare it as a lowercase module-level value with an environment default, in whichever storage module owns that subsystem (`_STORAGE_MODULES` in `algan/settings/raytracing_settings.py` lists them — the toggles module plus the BVH builders, the kernels, the raster and sheet passes, the scene builder, the tracer and the memory model, each keeping its settings beside the code and the comment that explain them). `SETTINGS.raytracing` derives its field set from those modules, so the toggle is reachable with nothing else to register — leave it out of `_PUBLIC_FIELDS` unless it changes rendered output in a way users are meant to control, and it lands on `.experimental`. Two rules follow from the derivation: the value must be a scalar, and **no helper function may share a field's name** — the later `def` silently takes the name over and the field disappears (`test_settings_api.py` pins both).

Each setting is stored once, in its storage module. The hand-maintained `_FIELD_TO_LEGACY` map from lowercase fields to UPPER_CASE globals, and the `_SETTER_OVERRIDES` map beside it, are **deleted**: they were a second source of truth that drifted, and the drift is what left nine switches with a global, a setter and no way to set them. Do not reintroduce a table that mirrors the module.

Configuration that the renderer *freezes* when it is imported — an ndarray element type the kernels are annotated with, a reciprocal, a packed header layout, a `ti.static` payload width — is still a field, so it can be read, discovered and snapshotted. Writing one is refused by `_IMPORT_FROZEN_FIELDS`, naming the environment variable to set instead: a host that builds an arity-8 BVH for kernels annotated arity-4 does not fail, it renders wrong, so a refusal beats both a silent no-op and a silent corruption. `_INERT_FIELDS` and `_IMPORT_FROZEN_FIELDS` are both checked against the names a caller passed and never against a restored snapshot, so `set(source=...)` and `SETTINGS.restore()` still round-trip every field.

`tests/unit_tests/test_settings_api.py` pins the whole arrangement: every `env_*`-backed module global in `algan/` is a field (except the four init-only ones in `settings/_startup.py`), no helper shadows a field's name, and every declaration the storage modules make is reachable.

Use `SETTINGS.snapshot()`/`SETTINGS.restore()` for complete public-settings state capture, and `SETTINGS.override(...)` or section-level `override(...)` for temporary changes. Do not hand-roll partial restoration that leaves live settings leaked into later tests or daemon runs.

`SETTINGS.raytracing` validates every write: the accepted type of each of its fields is derived from the value it ships with (three polymorphic mode switches are exempted by name), numeric fields carry a lower bound taken from their documented meaning, and floats must be finite. A setter's own `ValueError` is re-raised as an `AlganConfigurationError` naming the field; `UnsupportedFeatureError` passes through unflattened, because it is a distinct type callers catch and *is* a subclass of `AlganConfigurationError`.

Writing a section is one operation with one set of rules: `SETTINGS.video.frames_per_second = 60` routes through `set()`, so assignment validates and normalizes exactly as `set(frames_per_second=60)` does, and `set()` writes back only the fields that actually changed (identity comparison) so an unrelated field keeps its object identity. Assigning a whole *section* (`SETTINGS.video = HD`) is still refused — sections have stable identity.

Engine modules must read mutable settings live through `SETTINGS`. Never import a mutable ray-tracing setting by value at module import time; doing so freezes the old value and makes public setters ineffective. Immutable constants may be imported by value.

Reading live is necessary and not sufficient when the reader is a **kernel**. A `ti.static` gate reads live and then folds the answer into the compiled kernel, and a specialization is keyed on its `ti.template()` arguments — never on a setting — so the value the first render traced is the one every later render in that process gets. Those settings are listed in `rt_settings.KERNEL_COMPILED_IN_FIELDS` (`linear_color_space`, the two ambient coefficients, `rgb_shadow_tint`, `watertight_tri`), which is what lets `taichi_runtime.ensure_taichi_for_render` notice one has moved and rebuild the Taichi program before the render that changed it — the same remedy, and the same cost, as a render device that moved across the CPU/GPU line. **A new `ti.static` gate over a mutable setting must join that tuple**; `tests/unit_tests/test_compiled_in_settings.py` reads the gates out of the kernel sources and fails on one that has not. Prefer a `ti.template()` argument where the value can be one: Taichi then specializes on it and both arms coexist with no rebuild at all (`auth_sampled` in `path_tracer_taichi.py` is the worked example).

Initialization-only settings intentionally have no public mutable Python object. Set these before importing `algan`:

- `ALGAN_ANIMATION_DEVICE`;
- `ALGAN_HOME`;
- `ALGAN_CACHE_DIR`;
- `TI_OFFLINE_CACHE_FILE_PATH`;
- `ALGAN_SOFT_SHADOW_SAMPLES`;
- `ALGAN_TI_DEBUG`, `ALGAN_TAICHI_WARMSTART`, `ALGAN_TAICHI_FAST_LAUNCH`, `ALGAN_TAICHI_SOURCE_KEY`;
- `ALGAN_TAICHI_BACKEND`.

`ALGAN_TAICHI_BACKEND` selects which Taichi-language compiler builds the kernels:
`quadrants` (**the default** compiler module, supplied by the locked
`algan-quadrants` distribution) or
`taichi` (1.7.x, the dormant upstream, installed by the `taichi` extra and kept as the
A/B control and the patched-Metal-wheel path). `BACKENDS[0]` in `algan/taichi_compat.py`
is what picks the default, so that tuple's order is the decision. Every engine module
reaches the compiler through `algan.taichi_compat` (`from algan.taichi_compat import ti`,
and `submodule("lang.impl")` for a submodule) rather than importing `taichi` directly, so
the choice is made once and a process with **both** live -- two runtimes, two CUDA
contexts, two kernel caches -- cannot be spelled. Do not add a bare `import taichi` to
`algan/`, **and not to `tests/` or `benchmarks/` either** — a test that declares a
`@ti.func` with a directly imported compiler is a mixed process. Each backend gets its own
offline-cache directory (`cache/<backend>`), and `algan.taichi_compat` owns the places
where the two spell the same thing differently: `kernel_specializations()` for
`compiled_kernels` vs `materialized_kernels`, and `program()` for `get_runtime().prog`,
which is `None` before `init` on taichi and *raises* on Quadrants.
It is startup-only in the strictest sense -- the kernels in the process are already
compiled by the chosen backend -- so the daemon refuses a client whose value differs.

`ALGAN_HDR_BUFFER_F16` is **not** one of them any more either. It seeds `SETTINGS.raytracing.experimental.hdr_buffer_f16`, and `hdr_frame_dtype()` reads that when the frame buffer is allocated — no kernel specializes on it, so there was never anything for the import to bake in.

`ALGAN_RENDER_DEVICE` is **not** one of them any more. It seeds `SETTINGS.computing.render_device`, which owns the value from then on and can be changed between renders; `taichi_runtime.ensure_taichi_for_render()` re-selects Taichi's arch at the start of each render job when the device has moved across the CPU/GPU line. Read it with `algan.settings._startup.render_device()` — never bind it at import, which is the mistake the old `_RENDER_DEVICE` constant made unavoidable. A change is refused while a render is running and once a wide attribute (a texture) has been placed on the render device.

`RENDERER_REGISTRY` and `KERNEL_REGISTRY` are runtime service registries, not user settings, and therefore live outside `SETTINGS` and outside the star-import namespace.

`SETTINGS.computing` accepts `render_device`; it rejects `animation_device` with a message naming the environment variable to use instead.

`SETTINGS.computing.torch_compile` runs the pipeline's per-frame torch arithmetic through `torch.compile`. It is the same tri-state shape as `mps_friendly`: `'auto'` (the default) resolves to `algan.utils.torch_compile.torch_compile_support()` — on wherever `torch.compile` runs, off on Windows and on a Python Dynamo does not support — and `True`/`False` decide for themselves; `ALGAN_TORCH_COMPILE` overrides both. The mechanism is one decorator, `algan.utils.torch_compile.compiled`, which reads the switch **at every call** (so it flips between two renders in one process and the daemon adopts it), builds the compile lazily, and on a compile failure warns once and demotes that function to eager for the rest of the process. The rules for what may go inside a compiled region — pure torch, no arena calls, no `.item()`/`bool(tensor)` control flow, live settings passed in as plain arguments, `Color` converted to a plain tensor — are in the module docstring; `benchmarks/_torch_compile_ab.py` is the warm alternating A/B with frame parity, and `tests/unit_tests/test_torch_compile.py` pins the switch and the fallback contract.

**A fresh process pays `torch.compile` again, cache or not.** Inductor's FX-graph and AOTAutograd caches (`TORCHINDUCTOR_CACHE_DIR`, default the system temp directory) carry built graphs between processes, but Dynamo re-traces every function per process, and Inductor's CPU instruction-set probe (`torch._inductor.cpu_vec_isa.valid_vec_isa_list`) starts one torch-importing subprocess per SIMD set: 7.2 s per process on a 4-core AVX-512/AMX box with every graph a cache hit. `torch_compile._remember_vec_isa_probe` stores a probe that passed every set the CPU reports in `SETTINGS.paths.cache_directory/torch_compile/vec_isa.json`, keyed by compiler, torch, CPU flags and interpreter, and hands later processes the same list (`tests/unit_tests/test_torch_compile_vec_isa_memo.py`, which also fails if torch moves the internals it replaces). Construction-time work stays eager: `Surface._compute_pn_geometry_error` calls `evaluate_logical_pn.eager`, bit-identical to the compiled arm, because the sizing search's small nets traced two graphs nothing reused. Measured on 4 cores, a fresh process to its first 3-D video: 32.0 s -> 23.8 s; with an empty Inductor cache it is 59 s, and 20 s with the switch off.

**Compiled graphs are keyed by the resolved CPU vector width.** PyTorch 2.7's FX-graph cache includes compiler options but omits the automatically selected CPU ISA. Moving a populated cache from AVX-512 to AVX2 reused 16-lane generated loops with 8-lane C++ vector types, leaving projection coordinates unwritten and dropping whole mesh faces in the fast render. `torch_compile._inductor_compile_options` resolves `cpp.simdlen` before compilation so the width participates in the graph key. It preserves mode presets and explicit compiler overrides without changing global configuration. Keep this protection even when the CPU ISA probe is memoized: that probe and the generated-graph cache are separate caches.

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

Aliases are fine: add one wherever authors will reach for a second spelling, such as a Manim name or a common abbreviation. When a rename is warranted, rename in place and update every call site; the project is pre-release, so this stays cheap.

`IN = INWARD` / `OUT = OUTWARD` carry one extra rule: `in` and `out` are words a script will want, so the short spellings are the script's to shadow and Algan's source reads only the long ones. `../tests/unit_tests/test_spatial_constants.py` enforces that. Write `OUTWARD` in `algan/`; write `OUT` in docs and tests.

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

Only a script that can render is handed off at all: `daemon_client.script_may_render` parses the script and the project modules it imports for a render entry point (`_RENDER_NAMES`), and counts anything it cannot follow (`exec`, `runpy`, `import_module`, a `sys.path` edit) as rendering. A script that names none runs in its own process with no daemon started, because the handoff would re-run its pre-import code for nothing. A new public render entry point belongs in `_RENDER_NAMES`. An auto-started daemon is launched with `--exit-if-first-run-renders-nothing` (`daemon._started_for_nothing`).

## Asset paths

`ImageMob`, `set_texture` and `background` all route through `file_utils.get_image` → `resolve_asset_path`, which
tries the working directory and then the main script's directory, so an image beside your script loads regardless of
where you launch Python.


### Batched screenshots

`Scene.save_frame("shot", at=[0.5, 3.0, 8.5])` uses one sparse render job,
`get_frames(..., frame_indices=..., _independent_frames=True)`. It does not
render the intervening frames. Each still must render exactly as it would
alone, from only its own Mobs, which constrains how stills share batches:

- A batch carries every Mob alive anywhere in its window, so consecutive
  stills share one only while the live set is identical for all of them --
  nothing spawns or despawns between the first and the last
  (`RenderLoopMixin._still_group_ends`, read off the lifespan index without
  materializing), and at most `_STILL_GROUP_MAX_FRAMES` (16) of them. Then a
  batch costs per frame what each still costs alone, and the ordinary memory
  budget, arena preflight and OOM halving size it as they do a video window.
  Before this, two stills spanning a scene carried all of its Mobs (backprop
  scene 6 at 427x240: 3.3 GB for 2 stills, over 6 GB for 9).
- Materialization replays every recorded function and updater one frame at
  a time (`TimelineManager.replay_frame_by_frame`, see `timeline.md`), so
  each frame's state is bit for bit its alone state. Batched replay is not
  shape-blind: a basis change spreads over a large subtree through an einsum
  whose GEMM rounds differently for a different number of frames, which moved
  backprop scene 15's camera screen by two ulps and its shadows by up to 63
  levels.
- Everything a batch decides across its frames is decided per frame for
  still jobs (`scene._frame_local_batches`): circuit chord counts, closing
  vertices and outline bounds (`RayTracedBezierCircuitPrimitive.project_to_screen`;
  frames whose edge counts differ are packed with inert padding edges by
  `bezier_geometry_cache._build_frame_local_circuit_edges`, which also never
  reuses contours within a tolerance), and post-processing, bloom included
  (`_framewise_post_processes(..., batch_native=False)`). A still batched with
  a zoomed-in one used to render its curves up to 72 levels finer.
- Every choice the merge makes for the batch as a whole -- opacity and
  material gates, the materials present, triangle promotion -- notes its
  per-frame inputs (`_merge_scene(frame_signature=True)`), and a batch whose
  frames do not all note what the first does is split after the frames that
  do (`_frames_deciding_alike`). A failed attempt costs its stills'
  preparation, so speculation is rationed (`alike_hint` in
  `_get_frames_impl`): a job's first batch is a probe of two stills, and
  whole groups follow once two agree. After a split, fetches take as many
  as last agreed and double only on evidence -- a batch's first still
  noting exactly what the batch before's last one did (the merge's
  `_frame_digests`, read by `_still_batch_digests`). Stills that never
  agree -- PN solids turning, whose per-frame dicing pads rows that note
  differently -- pay for one failed probe per job: 2.0 s against 1.8-1.9 s
  for 12 stills of two turning solids at 320x180, where blindly retried
  doublings cost three or four failed probes. A batch-wide choice added
  to the merge must note its inputs, or still batches stop matching their
  stills.
- The path tracer, the wavefront memory trim and in-composite tonemapping
  keep one still per batch (`_stills_may_share_batches`).

`test_batched_stills.py` checks the grouping, the split, the Mob sets, the
per-frame replay and that grouped stills equal their alone renders. Grouped
stills are byte-identical to their alone renders on every scene checked,
synthetic and backprop scene 15's 59 checkpoint and motion stills alike. The
trap behind most of that work: PyTorch's CPU kernels are exact per element but
not shape-blind -- a transcendental op (the sRGB decode's `pow`) rounds one ulp
apart in vectorized lanes and in a loop's scalar tail, a GEMM by its blocking
-- so an element's bits can depend on how many frames share its array. Hence
the per-frame replay, and the per-frame sRGB decode of a still batch's merged
colors (`_decode_merged_colors(per_frame=True)`) and light colors
(`render_loop._decode_light_rgb`); decoded whole, an animated glow came out one
level off under bloom. Arithmetic added to still preparation over a
frame-major tensor must be elementwise IEEE (or run per frame) to keep this.
Times are quantized at the selected frame rate; repeated frame indices share
rendered pixels, while output names, return order, and overwrite policy retain
the input order. Shared render wall time is divided among the rendered results.

`Project.render_screenshots()` collects selected `save_frame` calls while
authoring each scene, then renders them together. Per-call output paths and
settings are captured; incompatible adjacent options start another job. A
relative checkpoint inside an unfinished timed context follows that context's
final rescaling, with its offset still measured in seconds. Explicit positive
`at` values stay absolute. No extra authoring flag is necessary.

Inside the scene function, a selected call returns `RenderResult` objects with
`status="deferred"`; these describe destinations, not existing images. The
project returns completed `"rendered"`/`"skipped"` results after rendering.
Authoring failures discard that scene's queue. `stop_early=True` still stops at
the last requested checkpoint, then renders the collected requests.

### Project viewer

`Project.view(scenes=None, *, video_settings=None, port=0, open_browser=True,
block=True)` authors selected scenes in project order, suppressing embedded
saves and viewers. Unlike validation, it does not synchronize transcript files.
It returns the same `ViewerHandle` lifecycle as `Scene.view`.

The project session retains authored scenes and per-scene raytracing settings,
but only one `ViewerSession` renders at a time. Switching closes/drains the old
worker and materialized reads before starting the next one, dropping its frame
cache. Scene-scoped requests carry a monotonically increasing selection version;
old requests are rejected even after returning to the same scene. The browser
also invalidates pending frames, tree loads, transcript loads and inspections.
