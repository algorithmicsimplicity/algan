# Staging audit: Taichi argument staging during batch prep

Source-reading audit only (no renders, no runtime device checks — per brief).
Question: is the PCIe round-trip staging tax actually being paid today by any
Algan Taichi launch? All paths are `algan/...` unless marked.

> **Naming note, added later.** This audit was written when the render device
> was the import-time constant `algan.settings._startup._RENDER_DEVICE`. That
> name is gone: the device is now `SETTINGS.computing.render_device`, read
> through `algan.settings._startup.render_device()`, and Taichi's arch is
> re-selected per render job by `taichi_runtime.ensure_taichi_for_render()`.
> Read `_RENDER_DEVICE` below as "the render device". Every conclusion holds —
> the gates it traces still compare against that device, and none of them cached
> it.

## Verdict summary

| # | Claim | Verdict |
| --- | --- | --- |
| 1 | `pn_criterion_kernel_active()` is the ONLY dispatch gate, and cuda-only | **REFUTED** on "ONLY"; the cuda-only half is CONFIRMED |
| 2 | When the gate is true, every tensor argument of the three kernels is already CUDA | **CONFIRMED** (one argument — `slack` — is CUDA by derivation, not by check; named below) |
| 3 | No reachable path puts a CPU tensor into one of those kernels while the gate is true | **CONFIRMED** |
| 4 | On a CPU render device the three kernels are never dispatched | **CONFIRMED** |
| 5 | Apart from the three, the only non-render kernels are the `cpu_prep_kernel_enabled` ones | **REFUTED as literally worded**; CONFIRMED under default env vars if post-processing counts as render-stage. Two extra groups found (timeline query kernels; post-processing kernels) — see inventory |

**Bottom line for the build/no-build decision: under default configuration,
nothing stages today.** The three criterion kernels run only against tensors
verified to be on the CUDA render device; the CPU batch-prep kernels refuse to
launch on a GPU arch; the one group that *would* stage (timeline query
kernels) is disabled by default precisely because it staged.

---

## Claim 1 — REFUTED (the "ONLY gate" half); cuda-only half CONFIRMED

`pn_criterion_kernel_active()` (`rendering/raytracing/settings.py:2363-2365`)
is `PN_CRITERION_KERNEL and project_on_gpu_active()`.

- `project_on_gpu_active()` (`settings.py:2318-2327`) returns False unless
  `_RENDER_DEVICE.type == "cuda"` (`settings.py:2327`).
- `PN_CRITERION_KERNEL = env_flag("ALGAN_PN_CRITERION_KERNEL", True)`
  (`settings.py:2352`); `PROJECT_ON_GPU = env_flag("ALGAN_PROJECT_ON_GPU",
  True)` (`settings.py:2296`). So under **env defaults both True**, the gate
  can only be true when the render device is CUDA. `_RENDER_DEVICE` itself is
  fixed at import from `ALGAN_RENDER_DEVICE` (default `"auto"` → cuda only if
  a CUDA runtime is usable) — `settings/_startup.py:37-70`. Runtime toggles
  `set_pn_criterion_kernel` / `set_project_on_gpu`
  (`settings.py:2355-2360`, `2312-2315`) also exist and default on.

But it is **not the only gate deciding dispatch**. The call sites never invoke
the kernels directly off this flag; they go through two builders that can veto
independently:

- `_bezier_criterion_inputs` (`rendering/raytracing/primitives.py:181-195`):
  returns `None` (torch fallback) unless the gate is true **and** all four
  tensors are `.device.type == "cuda"` and float32 (`primitives.py:190-193`).
- `_pn_criterion_inputs` (`primitives.py:198-232`): same, over six tensors
  (`primitives.py:210-214`).

Additionally each launch is guarded by a non-empty work list:
`if active.numel():` (`primitives.py:1837`), `if selected.shape[0]:`
(`primitives.py:2022`), and the bezier search exits when nothing is active
(`primitives.py:2924`). So the settings gate is *necessary but not
sufficient*; the claim as stated ("the ONLY gate") is false. The three launch
sites are indeed exactly `primitives.py:1838` (`pn_edge_chord_error`),
`:2023` (`pn_patch_flatness_error`), `:2933` (`bezier_chord_hull_error`);
no other callers exist in the tree.

## Claim 2 — CONFIRMED (with one by-derivation argument named)

When a kernel launches, `kernel is not None`, which means the builder's check
passed. Argument-by-argument:

### `pn_edge_chord_error` (launch `primitives.py:1838-1853`; signature `logical_pn_taichi.py:409-424`)

| Argument | Origin | Device |
| --- | --- | --- |
| `kernel.edge_controls` | `edge_base` from `_frame_broadcast_base(edge_controls)` (`primitives.py:216`, contiguous only, no device change); `edge_controls` built in `_dice_logical_pn` from `source_corners`/`source_normals` (`primitives.py:2176-2180`), which after `upload_primitive_source` live on the render device | checked cuda at `primitives.py:210-214` |
| `kernel.edge_stride` | int | — |
| `active.to(torch.int32)` | `torch.arange(levels.numel(), device=device)` where `device = edge_controls.device` (`primitives.py:1765-1769`) | cuda by construction |
| `samples` | `_sample_tensor(self._edge_sample_parameters, device, dtype)` (`primitives.py:1768`); cache keyed `(values, device.type, device.index, dtype)` and created **on** that device (`primitives.py:65-80`) | cuda |
| `kernel.cam_origin/screen_point/screen_basis` | `cam_o/sp/sb.contiguous()`; derived in `_dice_logical_pn` as `.to(device)` with `device = source_corners.device` (`primitives.py:2150-2160`) | checked cuda (`primitives.py:210-214`) |
| `kernel.front_sign` | computed from sp/cam_o/sb (`primitives.py:1580`) | checked cuda |
| `kernel.slack` | `mean_patch_edge_length(source_corners) * ratio`, expanded (`primitives.py:2186-2190`) — see caveat below | cuda **by derivation only** |
| `error` | `torch.zeros(active.numel(), device=device, dtype=dtype)` with `device = edge_controls.device` (`primitives.py:1836`) | cuda |

### `pn_patch_flatness_error` (launch `primitives.py:2023-2039`; signature `logical_pn_taichi.py:190-204`)

| Argument | Origin | Device |
| --- | --- | --- |
| `kernel.control_points`, `control_stride` | same builder path as above (`primitives.py:215`) | checked cuda |
| `selected.to(int32).contiguous()` | `nonzero()` output of masks over `levels`/`unresolved`, all on `control_points.device` (`primitives.py:1931-1952`) | cuda |
| `corner_uv` | `pattern.vertex_uv[triangle_indices]`; pattern from `dice_pattern(..., device=levels.device, ...)`, whose cache is keyed by device (`rendering/logical_pn.py:637-668`, key at `:651`) | cuda |
| `weights` | `_sample_tensor(...)` (`primitives.py:2013`) — device-keyed cache | cuda |
| cam trio, `front_sign`, `slack` | as above | cuda (slack by derivation) |
| `error` | `torch.zeros(selected.shape[0], device=device, ...)` (`primitives.py:2021`) | cuda |

### `bezier_chord_hull_error` (launch `primitives.py:2933-2944`; signature `logical_pn_taichi.py:297-308`)

| Argument | Origin | Device |
| --- | --- | --- |
| `kernel[0]` (corners base), `kernel[1]` (stride) | `_frame_broadcast_base(corners)`; `corners = self.corners.float().contiguous()` on its own device (`primitives.py:2793`) | checked cuda (`primitives.py:189-193`) |
| `active.to(int32)` | `torch.arange(S, device=device)`, `device = corners.device` (`primitives.py:2893`, `2920`) | cuda |
| `kernel[2..4]` (cam trio) | `.to(device)` with `device = corners.device` (`primitives.py:2797-2806`) | checked cuda |
| `error_squared` | `torch.zeros((num_active,), dtype=corners.dtype, device=device)` (`primitives.py:2926-2928`) | cuda |
| `T`, `num_subdivisions`, `screen_h/2` | scalars | — |

Why the sources are CUDA whenever the gate is true: projection runs on the
render thread via the arena preflight → `_prewarm_render_batch`
(`render_loop.py:943`, `2340-2447`), which calls
`upload_primitive_source(primitive, project_device)`
(`render_loop.py:2425-2427`; definition `rendering/raytracing/scene_builder.py:195-225`
— moves *every* pre-projection source tensor) and moves the camera/light
snapshot shim tensors with `_to_device` (`render_loop.py:2388-2417`). Inside
`project_to_screen` everything else is harmonized to `corners.device`
(`primitives.py:2797-2806`, `2150-2160`) before the builders run.

**Caveat (the one argument not covered by a check):** `slack` is absent from
the builder's checked tuple (`primitives.py:210` lists six tensors; slack is
handled at `:220-231` with only a dtype cast). It is CUDA only because it is
derived from `source_corners` on that same tensor's device
(`primitives.py:2186-2190`; `mean_patch_edge_length` at
`rendering/logical_pn.py:153`). No current path puts it elsewhere, so claim 2
holds, but this is the fragile link — adding `slack` to the tuple at
`primitives.py:210` would close it.

## Claim 3 — CONFIRMED

The builders' device checks make a CPU-tensor launch structurally impossible:
any CPU operand among the six (four, bezier) checked tensors forces
`kernel=None` and the torch criterion runs instead — no launch, no staging.
I traced every candidate escape path requested:

- **OOM retry.** Preflight catches `InsufficientMemoryException` /
  CUDA-OOM around prewarm (`render_loop.py:952-964`) and merge
  (`:1004-1017`): drops partial state, returns not-fitting; the outer loop
  refetches fresh primitives and re-runs the same guarded path
  (`:2915-2945`, `:3010-3059`). `_rt_projected` is set only after a
  successful projection (`render_loop.py:2438`), and a rejected batch is
  deleted, not reused (`:2884-2891`, `:2933-2939`). No half-projected
  primitive carrying CUDA sources reaches a CPU-device projection.
- **`require_estimates_fit` window shrinking.** Only rejects/shrinks the
  frame window (`render_loop.py:932-940`, `993-1001`, `2900-2945`); devices
  untouched.
- **Prefetch worker vs render thread.** The worker prewarms **only when
  `not project_on_gpu_active()`** (`render_loop.py:2719-2723`) — i.e. exactly
  when the criterion gate is false anyway. When the gate can be true,
  projection is deferred to the render thread inside the preflight
  (`:909-951`) with uploads as in claim 2. The worker-side `except` that
  falls back to the render thread (`:2724-2730`) therefore cannot fire while
  the gate is true.
- **`_slice_fetched_batch` / `ALGAN_REUSE_FETCHED_BATCH`.**
  `_can_slice_fetched_batch` requires `project_on_gpu_active()` AND sources
  *not* already on the render device (`render_loop.py:565-593`, esp. `:591`);
  probes slice the pristine CPU batch (`:595-602`) and each accepted probe is
  projected through the same upload-then-prewarm preflight (`:664-681`).
  Rejected candidates release their `_rt_` state (`:604-616`).
- **Unmanaged-memory early return.** If memory is unmanaged, the preflight
  returns True *without* prewarming (`render_loop.py:892-893`); projection
  then happens in `render_primitive_batch` on the source devices
  (`:1280-1293`) — all CPU → builders veto → torch path. Gate true, kernels
  still never launched.
- **SETTINGS constants / cached tensors as arguments.** `_sample_tensor`
  cache is keyed by device (`primitives.py:75-79`); `_DICE_PATTERN_CACHE`
  likewise (`logical_pn.py:651`); the module-level CPU constant
  `_APEX_OF_EDGE` (`primitives.py:237`) is `.to(edge_levels.device)`'d
  (`:1649`) and is never passed to any kernel. No kernel argument comes
  straight from a SETTINGS constant.

What I could *not* construct: any execution in which the gate is true and a
checked tensor is not CUDA while a launch proceeds. The single theoretical
gap is the unchecked `slack` (claim 2 caveat), and no code path produces a
cross-device slack today.

## Claim 4 — CONFIRMED

On a CPU render device, `project_on_gpu_active()` returns False
(`settings.py:2324-2327`), so `pn_criterion_kernel_active()` is False, both
builders return None, and all three level searches run the chunked torch
paths (`primitives.py:1855-1892`, `2041-2087`, `2964-3010+`). This holds for
any non-CUDA render device, including MPS. (Env dependence: requires only the
defaults of `ALGAN_PROJECT_ON_GPU=True` and `ALGAN_PN_CRITERION_KERNEL=True`;
explicitly setting either to 0 likewise disables the kernels everywhere.)

## Claim 5 — Inventory (claim REFUTED as literally worded)

All 15 `algan/**/*_taichi.py` modules searched; 47 `@ti.kernel` definitions
total. Partition:

**Render/ray-trace/raster pipeline kernels (launched only from
tracer/raster_pipeline/sheets/scene_builder/post stage), 35:** wavefront ×13
(`wavefront_kernels_taichi`, launched `raytracing/tracer.py:808-3977`),
sheet_compact ×8 + sheet_resolve ×2 + raster ×5 + background ×1 +
glossy_prefilter ×3 + raytrace ×3 (launched `raster_pipeline.py`,
`sheets.py`, `scene_builder.py:2194`, `tracer.py`), bloom ×3
(`post_processing/bloom.py:181,264,272,317`), tonemap ×1
(`post_processing/post_process.py:224-232`). These are the render kernels the
claim excludes.

### Group A — the `cpu_prep_kernel_enabled` kernels (the claim's list), 3

| Kernel | Launch site | Gate | Host tensors while arch CUDA? |
| --- | --- | --- | --- |
| `grid_normals_sides_crosses` (`cpunormals`) | `mobs/surfaces/surface.py:547` | `cpu_prep_kernel_enabled("cpunormals")` **and** `grid.device.type == "cpu"` (`surface.py:530-540`); name in default-on set `taichi_runtime.py:415` | Would be host tensors by construction, but cannot launch: `cpu_prep_kernel_enabled` ends in `taichi_arch_is_cpu()` (`taichi_runtime.py:428-445`) → never launches on a CUDA arch → no staging |
| `gather_grid_to_triangles` (`cpugather`) | `surface.py:517-519` | `cpu_prep_kernel_enabled("cpugather")` (off by default; needs `ALGAN_OPT_ENABLE=cpugather`) **and** `flat_grid.device.type == "cpu"` (`surface.py:488-504`) | Same — arch-gated off on CUDA |
| `apply_glow_and_opacity` (`cpucolors`) | `rendering/primitives/triangle_primitive.py:83-91` | `cpu_prep_kernel_enabled("cpucolors")` (off by default) **and** `colors.device.type == "cpu"` (`triangle_primitive.py:63-73`) | Same — arch-gated off on CUDA |

### Group B — timeline query kernels NOT gated by `cpu_prep_kernel_enabled`, 2

`_query_state_from_edits` / `_query_selected_state_from_edits`
(`animation_timeline/utils_taichi.py:5,40`), launched at
`animation_timeline/timeline.py:705` and `:716` from
`_generate_array_states_taichi`. Reachable **only** when `ALGAN_OPT_DISABLE`
contains `torchquery` (`timeline.py:670-673`, `:119-131`) — non-default.
Their arguments are the animation tensors, which are host tensors by default
(`_ANIMATION_DEVICE`, `settings/_startup.py:69`), so if someone sets that env
var on a CUDA-arch machine, **every argument including the whole
`[T, N, D]` result would stage through VRAM**. That is documented as the
reason the path is opt-in (`timeline.py:688-697`; `taichi_runtime.py:378-397`).

### Group C — post-processing kernels gated by device-matching, not `cpu_prep_kernel_enabled`, 4

| Kernel | Gate | Staging risk |
| --- | --- | --- |
| `bloom_conv1d_f32` | `can_use_bloom_taichi(input_tensor.device)` (`bloom.py:175-181`) | None: gate requires Taichi arch == tensor device (`bloom_kernels_taichi.py:111-122`) |
| `bloom_downsample_bilinear_aa_f32` | same (`bloom.py:260-273`) | None |
| `bloom_upsample_bilinear_f32` | same (`bloom.py:314-317`) | None |
| `tonemap_to_u8` | `POST_TONEMAP_KERNEL` (default True, `ALGAN_POST_TONEMAP_KERNEL`, `settings.py:2717`) + tonemapping enabled; **no explicit device check** (`post_process.py:213-232`) | None on CUDA/CPU renders: the frame lives in the render pool on `_RENDER_DEVICE` and the arch is chosen from that same device (`taichi_runtime.py:363-375`). See caveats for the MPS corner |

So: apart from the three criterion kernels, the complete set of non-render-
pipeline launches is Groups A+B+C. Group A matches the claim; B and C do not.
Under **default env vars**, B never launches; C launches but always
device-matched. Whether the claim survives therefore depends on whether
post-processing counts as "render kernels" (it is outside the ray-trace/
raster kernels strictly speaking) and whether "launches" means "can launch"
(B exists) or "launches by default" (it does not).

*(Out of scope, for completeness: `benchmarks/_*_taichi.py` define kernels
launched only by benchmark scripts; `algan/utils/taichi_utils.py::elementwise`
is an unused kernel factory; `tracer.py:3384` and `:3721-3730` import
`wavefront_textured_kernels_taichi` / `wavefront_sorted_kernels_taichi`,
which do not exist in the tree — both sit inside docstring-flagged UNSUPPORTED
legacy orchestrators and would ImportError if reached.)*

---

## Direct question 1: is each criterion kernel's input entitled to its device?

**No — entitled only transitively.** Nothing about the criteria requires
their inputs to be on the render device; the inputs are there because
*projection* runs there, and the criteria consume projection's operands:

- `control_points` / `edge_controls` / bezier `corners`: built from
  `self.corners/.normals` on whatever device projection uses
  (`primitives.py:2131-2180`, `2793`). They are on the render device because
  `PROJECT_ON_GPU` (default on) places vertex shading + packing there for
  bandwidth reasons of its own (`settings.py:2284-2296`).
- Camera-derived inputs (`cam_o/sp/sb/front_sign/slack`): on the render
  device because the prewarm shim moves the snapshot there for projection
  (`render_loop.py:2387-2417`), again incidental to the criteria.
- Derived work-list tensors (`active/selected/corner_uv/weights/error`):
  created on whatever device the controls ended up on.

If projection moved back to the CPU (`ALGAN_PROJECT_ON_GPU=0`, or a non-CUDA
device), the criteria follow it to torch-on-CPU automatically — the builders
veto (`primitives.py:190-193,210-214`) rather than staging. So the placement
could "stop" at any time by design, and nothing breaks; conversely, a future
subsystem that keeps *only* the criteria on the GPU would have to move these
inputs (and their results) deliberately, since nothing upstream guarantees
them residency for the criteria's sake.

## Direct question 2: any prep-stage tensor on the render device merely because a budget allowed it?

**No tensor's *placement* flips with budget size — budgets size windows, not
devices. But the closest case, and the one worth naming, is the wide-attribute
(texture) materialization window:**

- **Tensor:** the materialized `[T, rows, channels]` frame window of any
  attribute ≥ `WIDE_ATTR_MIN_CHANNELS = 1 << 16` channels — i.e. color
  textures — placed on `_RENDER_DEVICE` by
  `_wide_attr_materialize_device` (`animation_timeline/timeline.py:742-776`,
  applied at `:811`).
- **Budget:** `_render_device_prep_budget()` =
  `_RENDER_PREP_FRACTION = 0.4` share of VRAM outside the arena
  (`render_loop.py:79-85`, `1976-2001`, using total memory or
  `SETTINGS.computing.available_memory_override`), enforced in
  `get_batch_of_primitives`' duration binary search alongside the
  animation-device budget (`render_loop.py:2027-2068`).

On a smaller GPU this window shrinks **in frames**, not in device: the binary
search floors at duration 1 (`render_loop.py:176-189`, `best` initialized to
1), so even a tiny GPU still materializes one frame of texture on the render
device; placement is decided by channel width + device type
(`timeline.py:770-776`), never by the budget. The same logic covers the
projection uploads (`upload_primitive_source`): placed by the
`PROJECT_ON_GPU` flag, bounded in *size* by the
`PROJECT_GPU_PEAK_FACTOR = 8.0` headroom estimate
(`settings.py:2296-2309`) and the OOM retry — a small GPU renders fewer
frames per batch, it does not keep the sources on the host.

## Env-var defaults the verdicts rest on

- `ALGAN_PN_CRITERION_KERNEL` default **True** (`settings.py:2352`)
- `ALGAN_PROJECT_ON_GPU` default **True** (`settings.py:2296`)
- `ALGAN_RENDER_DEVICE` default **auto** → cuda iff usable
  (`settings/_startup.py:37-70`)
- `ALGAN_OPT_DISABLE` default empty → `torchquery` path never taken
  (`timeline.py:119-131`, `:670-673`)
- `ALGAN_OPT_ENABLE` default empty → `cpugather`/`cpucolors` off
  (`taichi_runtime.py:400-445`)
- `ALGAN_OPT_DISABLE=cpukernels` (or the kernel name) would also disable
  Group A (`taichi_runtime.py:441-442`)
- `ALGAN_POST_TONEMAP_KERNEL` default **True** (`settings.py:2717`)
- `ALGAN_REUSE_FETCHED_BATCH` default **True** (`render_loop.py:572`)

## Could not determine from source / caveats

1. **No runtime confirmation was attempted** (no GPU on this box, per brief).
   Every conclusion above is structural: gates read before launches, device
   checks in the launch path, and device-keyed caches.
2. **`slack` is unchecked** in `_pn_criterion_inputs` (`primitives.py:210`);
   safe today purely by derivation. One-line hardening suggested.
3. **MPS corner (outside the CUDA scope):** `_taichi_arch` returns `ti.gpu`
   for an MPS render device (`taichi_runtime.py:372-375`), and
   `tonemap_to_u8` has no explicit device-match guard (unlike bloom's
   `can_use_bloom_taichi`). Whether an MPS render stages or errors in post
   depends on which backend `ti.gpu` resolves to; not determinable from
   source alone, and irrelevant to the CUDA-staging question asked.
4. The dead imports noted in the claim-5 inventory
   (`wavefront_textured_kernels_taichi`, `wavefront_sorted_kernels_taichi`)
   could not be resolved because the modules are absent from the tree.
