# The Taichi patches Algan's Apple-GPU path needs

Two patches against **Taichi v1.7.4**, applied in order onto a pristine
checkout of that tag. They are the source of truth for the forked wheel
`../algan/rendering/DESIGN_mps_zero_copy.md` §4.2 calls `algan-taichi`; nothing here is applied to
a stock install, and Algan runs on a stock wheel without them (see "What Algan
does without them" below).

`.github/workflows/taichi_build.yaml` is the executable version of everything
that follows: it clones the tag, applies these, builds a wheel on the free
Apple-silicon runner and publishes it as an artifact.

    git clone --depth 1 --branch v1.7.4 https://github.com/taichi-dev/taichi.git
    cd taichi
    git apply ../taichi_patches/0001-*.patch ../taichi_patches/0002-*.patch

Both are additive, both are guarded so a non-Metal build compiles unchanged,
and neither touches a submodule — which matters, because the alternative for
patch 0002 was SPIRV-Cross and a submodule fork is a different maintenance
proposition.

## 0001 — zero-copy ndarrays over an imported `MTLBuffer`

**What it fixes.** Taichi copies every torch ndarray argument to the host
before a launch and copies it back after, unless it is a host tensor on a CPU
arch or a CUDA tensor on a CUDA arch (`kernel_impl.py:776`). On Metal that is
four copies per read/write argument per launch — and, since Algan's converted
kernels take *two dtype views of one arena*, the second copy-back reverts what
the kernel wrote through the first, which is why an Apple GPU renders a black
frame (`../algan/rendering/DESIGN_mps_support.md` §1.3b). An argument that arrives as a
`ti.Ndarray` takes `set_arg_ndarray` instead, which registers no copy-back at
all, so this removes the mechanism rather than working around it.

**What it adds**, all reachable only through the new entry point:

| file | change |
| --- | --- |
| `rhi/metal/metal_device.{h,mm}` | `import_external_mtl_buffer(Device*, uint64_t)` — a C++-callable wrapper over the existing `MetalDevice::import_mtl_buffer`. It exists for a linkage reason: `MTLBuffer_id` is `id<MTLBuffer>` in an Objective-C++ TU and `struct MTLBuffer_t *` in a plain C++ one, so the two spellings mangle differently and a `.cpp` caller cannot link against the `.mm` definition. Taking an integer handle gives both sides one signature, and saves adding a `.mm` translation unit (and a CMake edit) to the patch. |
| `program/ndarray.h` | `buffer_offset` — where this array starts inside its allocation. Zero for every Taichi-allocated ndarray, which is why nothing else reads it. |
| `program/program.{h,cpp}` | `create_ndarray_from_metal_buffer(dtype, shape, handle, offset)`. Adopts the buffer (`import_mtl_buffer` marks it `dont_destroy`), builds the `Ndarray` over it, and stores it in the existing `ndarrays_` map so lifetime and `delete_ndarray` work unchanged. `#ifdef TI_WITH_METAL`; raises otherwise. |
| `program/launch_context_builder.{h,cpp}` | carries the offset and byte size of any argument whose offset is non-zero. Both maps stay empty on every existing path. |
| `runtime/gfx/runtime.cpp` | binds `alloc.get_ptr(offset)` with a size when there is an offset, and keeps the `DeviceAllocation` overload when there is not — so nothing changes for an ndarray Taichi allocated itself. Metal already honours the offset (`MetalShaderResourceSet::rw_buffer` sets `rsc.buffer.offset = ptr.offset`). |
| `python/export_lang.cpp` | one pybind. |
| `python/taichi/lang/_ndarray.py` | `ExternalMetalNdarray`, a thin `Ndarray` subclass so the existing launch path finds `.arr`. It takes an `element_shape`, so an array a kernel annotates `ndarray(dtype=vector(4, f16))` imports as the vector-element ndarray Taichi type-checks for rather than being rejected as `f16`. The C++ side needed nothing for that: `Ndarray`'s `DeviceAllocation` constructor already reads the element shape off a tensor `DataType`, which is exactly how `VectorNdarray` builds its own. |

**The offset is not optional.** `../algan/rendering/DESIGN_mps_zero_copy.md` §3.2 predicted that
argument packing would leave one imported buffer at offset 0 and let this row
be dropped. Packing landed deliberately partial — the seven ray-state arrays of
`sheet_resolve_shade` stayed parameters because binding them cost +18%, and the
raster and compaction kernels were never converted — and every one of those is
a `ManualMemory` slice with a non-zero `storage_offset()`.

**The caller owns lifetime.** Taichi will not free an imported buffer, and it
does not hold a reference to whatever does. Torch's caching allocator will
happily recycle a buffer whose last tensor died, so the Algan side has to keep
the owning tensor alive for as long as the ndarray is used.

## 0002 — the MSL narrowing cast, and two diagnostics

**The codegen fix** is in `codegen/spirv/spirv_ir_builder.cpp`, and it is
Taichi's bug rather than SPIRV-Cross's. `IRBuilder::cast` lowers an integer
width conversion as *two* instructions — `OpSConvert` to the signed type of the
target width, then `OpBitcast` to the target — and for the commonest narrowing
of all, i64 to i32, that second step is a bitcast between two identical type
ids. SPIRV-Cross renders the pair as a nested functional cast, and MSL cannot
parse the result where it is bound to a temporary and then used:

    int tmp16_i32 = (int(long(_76))) * 8;
    error: indirection requires pointer operand ('int' invalid)
    error: cannot initialize a variable of type 'int' with an rvalue of
           type 'int (long)'

C++'s most vexing parse: `int(long(_76))` is the function type `int(long)` with
a parameter named `_76`, so `(...) * 8` reads as a cast applied to a
dereference. Skipping the no-op bitcast leaves `int(_76)`, which cannot be a
type-id because `_76` is not a type name. The guard is the one the function
already opens with, applied to the second step as well.

Measured on the hardware by `benchmarks/_mps_metal_codegen_probe.py`: the
single-cast forms compile, the nested one does not, and the kernel it took down
(`sheet_lane_first_owner`) compiles when its argument is narrowed so no cast is
emitted at all.

**The diagnostics**, in `rhi/metal/metal_device.mm` and
`runtime/gfx/runtime.cpp`, because this bug cost more to *find* than to fix.
Every way of failing to produce a shader — an over-wide kernel
(`../algan/rendering/DESIGN_mps_support.md` §1.1), an unsupported atomic (§1.2), a codegen bug
(§1.2b) — ends at `bind_pipeline`'s `assert(p != nullptr)`, which is a
`SIGABRT` naming a line of `metal_device.mm` and neither the kernel nor the
reason. So:

* **Name every failure and let none of them be silent.** The first version of
  this patch guarded the nil library and the nil function, and
  `../algan/rendering/DESIGN_mps_support.md` §1.2c is what that missed: `sheet_resolve_shade_arena`
  reached `bind_pipeline` null having printed *nothing*, because three of the
  paths to a null pipeline carry no message at all — `newComputePipelineState`
  returning nil with `err` nil, an exception from the `CompilerMSL`
  **constructor** (which parses the module, and sits outside
  `create_compute_pipeline`'s own `try`), and `MetalDevice::create_pipeline`
  catching that exception, discarding its text, and returning `success` with
  the out-pointer left null. All three now report, all of them name the task,
  and `create_pipeline` returns `error` when it produced nothing.
* **Raise instead of aborting.** `CompiledTaichiKernel`'s constructor dropped
  the `RhiResult` on the floor and pushed the null pipeline. It now raises,
  naming the task and the kernel, so a shader the backend cannot build is a
  Python traceback and one failed test rather than a dead process and a lost
  suite.
* **Print the MSL Metal rejected.** Metal reports `program_source:<line>:<col>`
  against a shader nothing has ever shown you, so a codegen bug arrives as a
  complaint about source you cannot read. A numbered ±6-line window around
  every line the message names, not the whole shader — a Taichi megakernel runs
  to thousands of lines.
* **`TI_SHADER_DUMP_DIR`**, when set, writes each task's SPIR-V and generated
  MSL there and every failure says where they went. The window above is no help
  for the failures that happen *before* MSL exists, and the `.spv` is then the
  only artifact there is.

That last group is worth having whatever happens to the rest: it is the
difference between "the process died" and "this kernel would not compile, and
here is how far it got".

## What Algan does without them

Nothing here is required to install or run Algan. `mps_compat` detects the
patched build at runtime and falls back when it is absent — and the fallback is
**not** the stock copying path, which is wrong rather than slow for the arena
convention (§1.3b). See `algan/rendering/mps_compat.py` for what a stock wheel
on a Mac actually does.

## Upstreaming

Both are small, additive and guarded, and both are worth sending upstream —
0002 especially, since the nested cast is a plain bug that affects any Metal or
Vulkan user narrowing a 64-bit integer, and the nil-function check turns a
process abort into an error on every Metal backend user's machine. If either
lands, the fork's patch set shrinks; if both do, it disappears.
