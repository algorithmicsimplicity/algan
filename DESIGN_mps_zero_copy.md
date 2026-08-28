# Zero-copy torch-MPS ↔ Taichi-Metal: what the patch would take

**What this is.** A follow-up to `DESIGN_mps_support.md` §1.3 and §3.3 step 1 —
the staging copy Taichi puts between a torch MPS tensor and a Metal kernel — and
only that. The verdict of that document stands: the argument limit (§1.1) is what
decides the port, and nothing here touches it. This one answers a narrower
question that was left open there: *can the copy be removed at all, and what
would it cost?*

**Evidence basis.** Source reading of Taichi 1.7.4 (installed wheel, and upstream
`v1.7.3` where the wheel ships no source); the Mach-O symbol table of the shipped
`taichi-1.7.4-cp311-cp311-macosx_11_0_arm64.whl`, parsed for the external/local
split; torch 2.7.1's MPS headers as shipped; and one timed from-source build of
Taichi on `macos-latest` (§4.1). No Algan render was run on Metal for this
document — the numbers it reasons about are `DESIGN_mps_support.md`'s.

**Answer in one line.** Yes, and every C++ piece already exists in Taichi — but it
has to be compiled into `taichi_python`, which means shipping a forked wheel, and
it unblocks nothing on its own.

---

## 1. Correction to `DESIGN_mps_support.md` §1.3

That section says:

> The only `import_memory` implementations in Taichi's core are `CpuDevice` and
> `CudaDevice` — there is none on the gfx device that serves Metal and Vulkan, so
> no externally allocated buffer can be adopted by either.

The first half is right and the conclusion does not follow. The generic
`Device::import_memory` virtual is indeed CPU/CUDA only, which is why
`taichi_launch_is_local`'s rule (§3.2) is correct as written and should stay.
But the Metal RHI carries its own, non-virtual import:

```cpp
DeviceAllocation MetalDevice::import_mtl_buffer(MTLBuffer_id buffer);
```

It is in `taichi/rhi/metal/metal_device.{h,mm}` upstream, it marks the allocation
`dont_destroy()` so Taichi never frees a buffer it did not allocate, and it sets
the memory's `can_map` from `[buffer contents]`. That last one comes back false
for a private-storage buffer, and torch's default tensor pool is the private one
(`at::mps::getIMPSAllocator(sharedAllocator=false)`, `MPSAllocator.h:127-142`) —
which does not matter here, because a kernel argument is bound on the GPU and
nothing in this design maps one to the host. **It is compiled into the wheel Algan already installs**:
`taichi_python.cpython-311-darwin.so` carries 365,655 symbols, 257 of them in
namespace `taichi::lang::metal` — every one local, none external — including

```
__ZN6taichi4lang5metal11MetalDevice17import_mtl_bufferEPU19objcproto9MTLBuffer11objc_object
```

So the adoption primitive exists, on the exact backend `ti.gpu` resolves to
(§2.1), and it is better suited than the `newBufferWithBytesNoCopy` route §3.3
speculates about: torch's tensors are already `MTLBuffer`s, so there is nothing to
wrap host memory around.

Two further facts complete the picture, both from upstream `v1.7.3`:

* **An ndarray argument is bound with no staging at all.** `gfx/runtime.cpp`'s
  `launch_kernel` allocates a staging buffer and memcpys only for arguments that
  arrive as raw host pointers; for one that arrives as a `DeviceAllocation` it
  does `any_arrays[indices] = devalloc` and binds it directly. That is the same
  path a Taichi-owned `ti.ndarray` takes, which is why §1.3 measured it at
  1.25 ms against 58.01 ms.
* **Metal honours buffer offsets.** The command list binds with
  `[encoder setBuffer:… offset:resource.buffer.offset atIndex:…]`, so a byte
  offset into an imported buffer is representable at the bottom of the stack.

What is missing is only the path from Python: nothing in the frontend can build
an `Ndarray` over an imported buffer. `taichi/lang/kernel_impl.py:776` special
cases exactly one pairing —

```python
tmp = v
if (str(v.device) != "cpu") and not (
    str(v.device).startswith("cuda") and taichi_arch == _ti_core.Arch.cuda
):
    host_v = v.to(device="cpu", copy=True)
    tmp = host_v
    callbacks.append(get_call_back(v, host_v))   # u.copy_(v) after the launch
```

— and everything else round-trips through the host, **per ndarray argument, per
launch, in both directions, unconditionally**, even for an argument the kernel
only reads or only writes. That branch is not an oversight: it was added as a
correctness fix for [taichi#6861](https://github.com/taichi-dev/taichi/issues/6861),
where MPS tensors previously produced silent garbage. It is also the only thing it
*could* do, because `mps_tensor.data_ptr()` is not an address: torch bit-casts the
storage pointer to `id<MTLBuffer>` (`ATen/native/mps/OperationUtils.h:110`,
`__builtin_bit_cast(id<MTLBuffer>, tensor.storage().data())`) and keeps the byte
offset separately as `storage_offset() * element_size()`. Nor is there an exchange
protocol to fall back on — `Tensor.__dlpack_device__` (`torch/_tensor.py:1746`)
raises for `mps`.

So the per-launch cost of a read/write argument today is **four** copies — MPS to
host in torch, host to device in Taichi, device to host, host to MPS — plus a
fresh `MTLBuffer` allocation per argument per launch, plus an MPS stream sync on
each `.to('cpu')`. The staging half is access-aware (`host_write` is set from
`ExternalPtrAccess::READ`); the torch half is not.

## 2. Why it cannot be a separate extension

The obvious way to avoid maintaining a fork is a small satellite module that
links against the stock wheel and calls the import itself. The symbol table says
no. Of `taichi_python`'s 365,655 symbols, 24,665 are externally defined, and the
split falls exactly wrong:

| symbol | present | linkable from outside |
| --- | --- | --- |
| `Ndarray::Ndarray(DeviceAllocation&, DataType, shape, …)` | yes | **yes** (external) |
| `MetalDevice::import_mtl_buffer` | yes | **no** (local) |
| everything else in `taichi::lang::metal` (257 syms) | yes | **no** (local) |
| `LaunchContextBuilder::set_arg_ndarray` | yes | **no** (local) |

You can construct the `Ndarray` from outside and cannot obtain the
`DeviceAllocation` to construct it over, and `MetalDevice`'s allocation table is
private, so the id cannot be forged either. `DeviceAllocation(device, alloc_id)`
being constructible from Python (it is) does not help for the same reason.

The C-API is a separate dead end in the other direction: `libtaichi_c_api.dylib`
*does* export `ti_import_metal_memory`, but it is the AOT runtime. It cannot
launch a JIT `@ti.kernel`, and exporting Algan's 52 kernels as AOT modules means
freezing every `ti.static` specialisation at build time.

**The patch has to be compiled into `taichi_python`.**

## 3. The patch

### 3.1 Taichi side

Five files, additive, all of it behind `#ifdef TI_WITH_METAL`:

| file | change |
| --- | --- |
| `taichi/program/program.{h,cpp}` | `Ndarray *create_ndarray_from_metal_buffer(DataType, shape, uint64_t buffer, uint64_t offset)` — casts `get_compute_device()` to `metal::MetalDevice`, calls `import_mtl_buffer`, builds the `Ndarray`, stores it in the existing `std::unordered_map<void*, std::unique_ptr<Ndarray>> ndarrays_` so lifetime and `delete_ndarray` work unchanged. The body wants its own `.mm` translation unit to keep Objective-C out of `program.cpp`. |
| `taichi/program/ndarray.{h,cpp}` | a `buffer_offset_` field and its accessor |
| `taichi/runtime/gfx/runtime.cpp` | bind `alloc.get_ptr(offset)` instead of the bare `DeviceAllocation` (which is offset 0) |
| `taichi/python/export_lang.cpp` | one pybind line |
| `python/taichi/lang/_ndarray.py` | a thin wrapper so the existing `set_arg_ndarray` path finds `.arr` |

Order of 150 lines. **The kernels do not change**: `ti.types.ndarray()` accepts a
`ti.Ndarray` exactly as it accepts a torch tensor, so nothing in
`algan/rendering/**/*_taichi.py` is touched.

### 3.2 The offset, and why argument packing dissolves it

Row three is the only awkward one, and it exists because of how Algan allocates.
`ManualMemory.get_tensor` hands kernels slices of one arena tensor
(`algan/utils/memory_utils.py:744`, `self.data[pointer:new_pointer]`), so
`storage_offset() != 0` for essentially every kernel argument, while an imported
`MTLBuffer` is the whole arena. The offset therefore has to travel from the
launch context down to the bind — representable, since Metal takes it, but it is
plumbing through three layers.

It disappears entirely under the fix `DESIGN_mps_support.md` §3.3 step 2 already
requires. If kernel arguments are packed — bind the arena once, pass slice bases
as integers — then there is exactly one imported buffer, at offset 0, and the
patch shrinks to rows 1, 4 and 5, about 60 lines. The two problems have one
solution, and it is the one the ~24-argument limit forces anyway.

### 3.3 Algan side

No fork involvement, roughly 100 lines:

* buffer handle from `t.untyped_storage().data_ptr()`, offset from
  `t.storage_offset() * t.element_size()` — never `t.data_ptr()`, which is an
  Objective-C object pointer with an offset added to it;
* cache the import per (handle, dtype, shape, offset), hold a reference to the
  tensor so torch's caching allocator cannot recycle the buffer underneath the
  kernel, and drop the cache when the arena is rebuilt;
* `torch.mps.synchronize()` before the batch and `ti.sync()` after — the two
  libraries hold separate command queues and torch's heaps are
  `MTLHazardTrackingModeUntracked`, so ordering has to be explicit. Once per
  frame batch, not per launch;
* align arena slices to 16 bytes;
* fall back to the stock copying path whenever the patched build is absent, so a
  stock wheel keeps working.

### 3.4 What it does not fix

Nothing else. The ~24-argument limit (§1.1), f64 (§1.2) and i64 atomics (§1.2)
are untouched, and every shading and ray-tracing kernel stays blocked. What it
buys is the 46x on the arguments that do get bound — and it buys it *without*
§3.3 step 1's residual cost, which was the per-crossing host copy that
`ScalarNdarray`'s numpy-only interface forces (6.48 ms in, 15.63 ms out for
16 MB). Imported buffers make the crossing free instead of cheaper, which matters
for the raster/sheet stages that hand off often rather than only for the
wavefront loop.

## 4. Building and shipping a patched Taichi

### 4.1 What a build costs

Measured by `.github/workflows/mps_probe.yaml`'s `taichi_build` job, stock v1.7.4,
`macos-latest` (the free Apple-silicon runner), one Python version, Vulkan and
CUDA off:

<!-- MEASUREMENT PENDING -->

### 4.2 Release logistics

* A Mac is needed to compile — `metal_device.mm` is Objective-C++ against the
  Metal framework and the wheel is arm64 against the macOS SDK. Cross-compiling
  from Linux is not a path. **A Mac is not needed to own one**: this repository is
  public, so the arm64 runner is free, and §4.1 was taken on it.
* Ship the fork as its own distribution (`algan-taichi`), not as part of Algan.
  It is rebuilt when Taichi releases — 1.7.3 to 1.7.4 was about a year and a half
  — or when a new CPython appears, **not** on Algan's release cadence.
* Make it an extra (`pip install algan[mps]`) and detect it at runtime. Then the
  default install stays on the stock wheel, a wheel build never blocks an Algan
  release, and the fallback in §3.3 is what runs when it is absent.
* The matrix is macOS-arm64 × cp310–cp313; Taichi publishes no cp39 arm64 wheel,
  so Algan's `>=3.9` floor is already unreachable on a Mac.
* One packaging caveat: the fork installs the same `taichi` import package, so a
  user with both installed gets colliding files. Gate it with an environment
  marker and say so.
* Upstream it in parallel. The change is small, additive and `#ifdef`-guarded,
  and if it lands the fork goes away.

## 5. Recommendation

Do not build this yet. It is the cheapest of the three items in
`DESIGN_mps_support.md` §3.3 and the only one with a finished design, which makes
it tempting in exactly the wrong order: on its own it speeds up four raster
kernels on a device that still cannot render a frame.

The sequence that makes sense is unchanged from §3.3 — argument packing first,
because it is the blocker and because it is what makes this patch a 60-line
change instead of a 150-line one. When that lands, this becomes the natural next
step rather than a project of its own.

**The trigger to revisit sooner** is upstream: if Taichi exposes any frontend
route to `import_mtl_buffer` — or accepts the patch in §3.1 — the fork disappears
and only the Algan-side glue in §3.3 remains.
