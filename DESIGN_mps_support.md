# Algan on Apple GPUs (MPS / Metal): measured verdict

Status: **NO-GO on the port as such.** Measured on GitHub's `macos-latest` runner
(Apple Silicon, macOS 26.5.2 arm64, torch 2.7.1, taichi 1.7.4) by
`benchmarks/_mps_capability_probe.py`, run from `.github/workflows/mps_probe.yaml`.

The macOS CI job is pinned to `ALGAN_RENDER_DEVICE=cpu` because MPS renders fail.
The workflow comment attributes that to `float64` and says supporting MPS "means
taking float64 out of the raster pipeline and the kernels". **That is true and it
is the least of it.** Three independent Metal limits block the port, f64 is the
smallest, and the one that decides the question is the kernel argument limit,
which nobody had looked at.

Read §1 for the verdict, §2 for the numbers behind it, §3 for what to do instead.

**Scope, added later.** Everything below measures **Taichi on the Metal
backend**. Two of the three blockers (§1.1, §1.3) turn out to be properties of
Taichi's interop and codegen rather than of Metal, and they do not survive
dropping Taichi from the path in favour of hand-written Metal shaders dispatched
through `torch.mps.compile_shader` — see `DESIGN_metal_native_port.md`, which
sizes that alternative and does not contradict a number here.

**Timing caveat, added later.** `macos-latest` is a virtualized-GPU instance.
It is sound for the capability results — which are the ones the verdict rests on,
and they stand — but every millisecond figure in §1.3 and §2 should be read as
directional only, not as a number to plan against. The staging comparison is the
robust one (a host round-trip is a host round-trip); the compute-bound
"52x better than CPU" is the fragile one. Sizing any of this needs physical
Apple hardware.

---

## 1. The three blockers, in order of how much they cost to clear

### 1.1 Metal binds at most ~24 ndarray arguments — this is the blocker

A kernel with 24 ndarray arguments runs. **31 aborts the process:**

```
-[MTLComputePipelineDescriptorInternal setComputeFunction:withType:]:866:
    failed assertion `computeFunction must not be nil.'
```

The MSL function comes back nil — the shader did not compile — and Taichi does
not check before building the pipeline, so it is a `SIGABRT`, not an exception.
Consistent with Metal's 31-buffer-per-stage binding limit once Taichi's own
context and root buffers are counted.

Algan's renderer is built out of megakernels, and the important ones are all over
the line:

| kernel | ndarray args | fits in 24? |
| --- | --- | --- |
| `sheet_resolve_shade` | 49 | no |
| `wavefront_shade` | 38 | no |
| `raster_shadow_trace` | 30 | no |
| `wavefront_traverse_events` | 30 | no |
| `raster_tri_write` | 20 | yes |
| `raster_bez_write` | 18 | yes |
| `raster_tri_count` | 15 | yes |
| `raster_bez_count` | 14 | yes |

Every shading and ray-tracing kernel is blocked. Only the raster count/write
kernels fit. Clearing this means either splitting each megakernel into pieces of
at most 24 buffers — which cuts across exactly the fusion those kernels exist for
— or packing many arrays into single buffers and indexing inside the kernel,
which rewrites every signature in the renderer. Neither is a port; both are a
redesign of the kernel layer.

For contrast, Taichi's *own* ceiling is 64 arguments and it reports it politely
on both backends: `The number of elements in kernel arguments is too big! Do not
exceed 64 on metal backend.` The Metal limit is a third of that and announces
itself with an abort.

### 1.2 Metal has no f64, and no i64 atomics

f64, asked of Taichi directly on a Taichi-owned buffer, so torch is not in the
way:

```
RuntimeError: [spirv_ir_builder.cpp:get_primitive_type@299] Type f64 not supported.
```

This is the SPIR-V codegen, which serves **both** the Metal and Vulkan backends.
Metal Shading Language has no `double` at all, so there is no capability to
enable. (Torch refuses first with its own `Cannot convert a MPS Tensor to
float64`, which is why the earlier reading was ambiguous about whose limit it
was.)

i64 is the more interesting result, because it splits:

| | on Metal |
| --- | --- |
| i64 arithmetic, shifts, packing (`(depth << 32) \| layer`) | **works** |
| `ti.atomic_add` / `ti.atomic_min` on i64 | **aborts** |

```
Assertion failed: (p != nullptr), function bind_pipeline,
    file metal_device.mm, line 382.
```

Same failure on a torch tensor and on a Taichi-owned ndarray, so it is the
backend rather than the interop. The raster pipeline's 64-bit fragment keys are
therefore fine; `sheet_compact_taichi.py:547-554`'s i64 atomics are not.

**This is what rules out the obvious deterministic fix.** A Q32 fixed-point
accumulator is exactly order-independent and needs no f64, and it is measured
bit-identical run to run on MPS — but only through torch. As a Taichi kernel on
Metal it aborts with the above. f32 `ti.atomic_add` does work, so a
non-deterministic mode has a floor; a deterministic one does not, in a kernel.

### 1.3 Taichi stages every torch tensor through host memory

`kernel_impl.py` copies any ndarray argument that is neither a host tensor nor a
CUDA tensor on a CUDA arch to the host before the launch and back after. The only
generic `Device::import_memory` implementations in Taichi's core are `CpuDevice`
and `CudaDevice` — there is none on the gfx device that serves Metal and Vulkan,
which is what makes `taichi_launch_is_local`'s rule (§3.2) right as written.

> **Amended.** The conclusion drawn here — that no externally allocated buffer
> can be adopted by either backend — is too strong for Metal. The Metal RHI
> carries its own non-virtual `MetalDevice::import_mtl_buffer`, and it is
> compiled into the wheel Algan already installs, though only reachable from
> C++. What blocks adoption is the Python frontend, not the backend.
> `DESIGN_mps_zero_copy.md` has the evidence and what a patch would cost. It
> changes nothing about this section's measurements or about §1.1 being the
> blocker.

Measured, 32 MB per launch:

| | bandwidth-bound | compute-bound |
| --- | --- | --- |
| CPU arch, host tensors | **1.09 ms** | 166.05 ms |
| Metal, torch MPS tensors | 58.01 ms (**53x worse**) | **7.45 ms** (22x better) |
| Metal, torch host tensors | 19.82 ms | **3.18 ms** (52x better) |
| Metal, **Taichi-owned ndarray** | **1.25 ms** | — |

Unlike the other two, this one has a fix (§3.3): a Taichi-owned `ti.ndarray` is
bound by device allocation with no copy-back callback, and is **46x faster than
the torch-MPS path on the same backend**.

---

## 2. Everything else the probe established

### 2.1 Which backend

`ti.gpu` resolves to **metal**, not vulkan — `ti.gpu` is
`[cuda, metal, vulkan, ...]` and metal comes first. The `test.yaml` comment
saying it "resolves through Vulkan" is wrong; the f64 error it saw comes from the
SPIR-V codegen that Metal also uses.

### 2.2 Vulkan is not an escape, on either side

Forcing `TI_ARCH=vulkan` (verified to have taken effect — the arm reports
`vulkan`) stages torch tensors exactly like Metal: 54.93 ms with MPS tensors
against Metal's 58.01, and 26.62 ms with host tensors against Metal's 19.82.
Same order, no escape, as §1.3's missing `import_memory` predicts.

And torch has no Vulkan compute device to pair with it: `is_vulkan_available()`
is False and `torch.zeros(4, device="vulkan")` raises `NotImplementedError`.
Torch's Vulkan backend is an Android mobile-inference path.

### 2.3 Torch op coverage on MPS

4 of the 25 ops the raster path calls fail, with `PYTORCH_ENABLE_MPS_FALLBACK`
cleared so a gap is loud:

| op | failure |
| --- | --- |
| `torch.zeros(dtype=float64)` | `TypeError` |
| `scatter_reduce_(reduce="amin")` on int64 | `RuntimeError` |
| `scatter_reduce_(reduce="amax")` on int64 | `RuntimeError` |
| `cummax` f32 | `NotImplementedError` |

The amin/amax gap is not incidental: `_one_mesh_pixel_caps`
(`raster_pipeline.py:1330-1332`) uses exactly those two for the per-pixel surface
id spread. Everything else passed, including int64 `scatter_add_`, 64-bit shifts,
`view(torch.uint8)`, `argsort`/`sort` on int64, `unique_consecutive`, `bincount`.

### 2.4 How non-deterministic MPS actually is

400k fragments reduced into 5k segments, six runs:

| | f32 `scatter_add_` | fixed-point i64 |
| --- | --- | --- |
| CPU | bit-identical | bit-identical |
| MPS | **not identical, spread 1.14e-05 = 192 f32 ulps at 1.0** | bit-identical |

(A second run measured 256 ulps — it varies, which is the point.) For scale,
`raster_pipeline.py:1345-1364` records that CUDA's unordered float atomics moved
two consecutive renders by up to 28 channel values over 9.6% of a frame, and that
is why the f64 accumulator exists. 192 ulps is a larger disturbance into the same
threshold.

---

## 3. What to do

### 3.1 Do now, regardless: stop offering a device that cannot render

Independent of everything above, and worth doing whether or not MPS is ever
supported. `auto` currently selects a device that fails 88 tests.

* `_startup._auto_render_device()` should not return MPS.
* An explicit `mps` should raise `AlganConfigurationError` from `coerce_device`
  naming the reason, instead of failing deep inside the renderer.
* `cli.py:45-46` prints "Apple Silicon MPS acceleration available", which is not
  true.
* Then the `test.yaml` macOS pin is redundant — the runner resolves to CPU on its
  own — and that 15-line comment becomes a pointer here.

### 3.2 Two engine bugs this turned up — **fixed**

Both are corrected in `taichi_runtime.py`, with the truth tables pinned by
`tests/unit_tests/test_taichi_launch_pairing.py`. Neither fix moves CPU or CUDA
behaviour: on those the new answers are the old ones for every reachable
pairing, and they differ only where the old rule was wrong.

* **`taichi_launch_is_local` was wrong for MPS.** It compared device *types*, so
  it answered `True` for an MPS tensor on a Metal arch — confirmed live by the
  probe — while Taichi copied through the host, and the gate it feeds turns the
  PN criterion kernels on precisely when a launch is free. The rule is not about
  device equality: Taichi implements `Device::import_memory` for `CpuDevice` and
  `CudaDevice` and for nothing else, so a torch allocation binds without copying
  only as a host tensor on a CPU arch or a CUDA tensor on a CUDA arch. Every
  other pairing stages, including one whose two halves name the same physical
  GPU. A `taichi_arch_is_cuda` companion to `taichi_arch_is_cpu` carries the
  second half of that.
* **`_arch_matches_render_device` could not switch between two GPU backends.**
  It compared `live != ti.cpu`, making every GPU backend interchangeable, so a
  render device moving between two of them kept whichever program was already up
  and launched every kernel on the wrong device with no re-init and no error —
  its docstring claimed to rule that case out while the code was what allowed
  it. It now compares against the backends that actually serve the device, from
  a written-out mapping rather than through `adaptive_arch_select`: resolving
  `ti.gpu` means probing Vulkan and OpenGL, which is the fallback chain
  `_taichi_arch` exists to avoid and which some headless configurations crash
  inside rather than declining.

The second is also what silently defeated this probe's own first attempt to
force Vulkan, which is how it was found.

### 3.3 If Apple GPU support is ever a priority, this is its shape

Not a port. In dependency order:

1. **Taichi-owned ndarrays instead of torch tensors** at the kernel boundary.
   Worth doing on its own merits — 46x on Metal, and on CUDA it removes the
   fast-launch path's staging concerns. The cost is that Taichi 1.7.4 offers
   `ScalarNdarray` only `from_numpy`/`to_numpy`, so data crossing from torch
   still goes through the host: measured 6.48 ms in / 15.63 ms out for 16 MB on
   Metal. That relocates the copy from per-launch to per-crossing, which pays for
   a sub-pipeline that launches many times between handoffs — the wavefront
   tracer, not the raster/sheet stages. (A patched Taichi would remove the
   per-crossing copy too, by importing torch's own `MTLBuffer` — see
   `DESIGN_mps_zero_copy.md`. It needs a forked wheel and it is worth doing
   *after* step 2, not before: packing is what shrinks the patch.)
2. **Kernel argument packing** so no kernel binds more than ~24 buffers. This is
   the large one and there is no way around it (§1.1).
3. **f64 and i64 atomics out of the kernels**, replaced by f32 with a documented
   non-deterministic mode, since the deterministic fixed-point form cannot run in
   a Metal kernel (§1.2).

The payoff even then is compute-bound scenes only: 52x on the path tracer,
53x *worse* on the bandwidth-bound raster stages unless step 1 lands first.

**The right trigger to revisit** is upstream: Taichi exposing torch-MPS interop
to the Python frontend (the backend primitive already exists — see
`DESIGN_mps_zero_copy.md` — so this is a software limit rather than a hardware
one), or Metal's argument limit ceasing to bind through argument buffers.
Re-running the probe answers both in about three minutes.

### 3.4 What Mac users should be told meanwhile

The CPU path is the Mac path. It is the same arch CI exercises, so effort spent
making it faster pays twice.
