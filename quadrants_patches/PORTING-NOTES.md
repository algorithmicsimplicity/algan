# Porting `taichi_patches/000{1,2}` to Quadrants v1.3.0

Target: `Genesis-Embodied-AI/quadrants` tag **v1.3.0** (`ab9a58ab5`). Source: Algan's
three patches against **Taichi v1.7.4** (`b4b956f`). `taichi_patches/0003`
(`operator""_f` whitespace) is not ported — Quadrants already spells it
`operator""_f` (`quadrants/common/core.h:170`), which
`taichi_patches/README.md` §0003 predicted. (`quadrants_patches/0003` is a
different patch, unrelated to Taichi's third; see §7.)

Apply in order onto a pristine checkout:

    git clone --filter=blob:none https://github.com/Genesis-Embodied-AI/quadrants.git
    cd quadrants && git checkout v1.3.0
    git apply --verbose ../quadrants_patches/0001-metal-zero-copy-ndarray.patch
    git apply --verbose ../quadrants_patches/0002-metal-codegen-and-diagnostics.patch

Verified from a fresh checkout: both apply with strict `git apply` (no fuzz, no
`-3`), and 0002 is authored against a tree that already has 0001, exactly as the
Taichi pair is — they share `quadrants/rhi/metal/metal_device.mm`.

**Not compiled.** No macOS and no Metal on the box this was ported on. Every
hunk was written against the surrounding Quadrants source rather than
pattern-substituted, but nothing here has seen a compiler. §5 lists what to
watch on the first real build.

## 0. Global renames applied throughout

| Taichi | Quadrants |
| --- | --- |
| `taichi::lang` / `taichi::lang::metal` | `quadrants::lang` / `quadrants::lang::metal` |
| `TI_ERROR`, `TI_ERROR_IF`, `TI_ASSERT`, `TI_WITH_METAL`, `TI_DLL_EXPORT` | `QD_ERROR`, `QD_ERROR_IF`, `QD_ASSERT`, `QD_WITH_METAL`, `QD_DLL_EXPORT` |
| `taichi/...` include paths | `quadrants/...` |
| `python/taichi/` | `python/quadrants/` |
| `ti.` in prose/docstrings | `qd.` |
| `TI_SHADER_DUMP_DIR` | `QD_SHADER_DUMP_DIR` |
| pybind11 (`py::`) | nanobind (`nb::`) |
| 80-column C++ formatting | 120 columns (`.clang-format` `ColumnLimit: 120`); Python at `black -l 120` |

The entry-point **names** Algan looks up are unchanged, which is the constraint
`algan/rendering/mps_zero_copy.py` imposes:

* `<compiler>.lang._ndarray.ExternalMetalNdarray` — the class
  `zero_copy_available()` probes and `import_tensor()` constructs, with the same
  five-parameter signature `(dtype, arr_shape, buffer_handle, byte_offset=0,
  element_shape=())`.
* `prog.create_ndarray_from_metal_buffer(dt, shape, buffer_handle, byte_offset)`
  — same name, same argument names, same order.
* `Ndarray::buffer_offset` and `Program::create_ndarray_from_metal_buffer` keep
  their C++ spellings too, so `taichi_patches/README.md`'s table still reads
  correctly against this port.

---

## 1. Patch 0001 — zero-copy ndarrays over an imported `MTLBuffer`

All 11 target files exist at renamed paths. Line numbers below are
`taichi 1.7.4 → quadrants v1.3.0` (the pristine-file line the hunk anchors at).

### Ported essentially unchanged (rename only)

| File | Taichi → Quadrants | Note |
| --- | --- | --- |
| `program/ndarray.h` | `:50` → `:50` | `buffer_offset` + `owns_imported_allocation` inserted between `ndarray_alloc_` and `dtype`; same anchor, same members. |
| `program/program.h` | `:275` → `:325` | Declaration inserted after `get_struct_type_with_data_layout`, before `delete_ndarray` — the same two neighbours as in Taichi. |
| `rhi/metal/metal_device.h` | `:582` → `:586` | Free `import_external_mtl_buffer(Device*, uint64_t)` declaration after the `MetalDevice` class, before `}  // namespace metal`. |
| `rhi/metal/metal_device.mm` | `:1221` → `:1280` | Definition immediately after `MetalDevice::import_mtl_buffer` (PLAN §7.3 item 1 says `:1280`; confirmed). |
| `program/program.cpp` (include) | `:40` → `:34` | `#include "quadrants/rhi/metal/metal_device.h"` added to the existing `#ifdef QD_WITH_METAL` block. |

### Rewritten

**`python/quadrants/lang/_ndarray.py`** (`:229` → `:445`, inserted after
`ScalarNdarray`, before `NdarrayHostAccessor`). Four changes, all forced:

1. **`self.shape = ...` is gone.** `Ndarray.shape` is now a read-only
   `@property` computed from `self._physical_shape` and the optional
   `_qd_layout` permutation (`python/quadrants/lang/_ndarray.py:137-156`).
   Assigning `self.shape` would raise `AttributeError`. The class sets
   `self._physical_shape = tuple(self.arr.shape)`, exactly as `ScalarNdarray`
   (`:468`) and `VectorNdarray` (`matrix.py:2030`) do.
2. **`__del__` dropped.** Taichi's `Ndarray` base had no destructor, so the
   subclass carried one. Quadrants' base `Ndarray.__del__`
   (`_ndarray.py:83-89`) already calls `prog.delete_ndarray(self.arr)` through
   `impl.get_runtime()._prog` with the null guards the property now needs.
   Keeping the override would have duplicated the free.
3. **The tensor element type is built the Quadrants way.**
   `_ti_core.get_type_factory_instance().get_tensor_type(shape, dtype)` became
   `DataTypeCxxWrapper(_qd_core.get_type_factory_instance().get_tensor_type(shape, dtype).get_ptr())`
   — the exact form `VectorNdarray` (`matrix.py:2031`) and `MatrixNdarray`
   (`matrix.py:1876`) use. `DataTypeCxxWrapper` is added to the existing
   `from quadrants.lang.util import (...)` block (ruff's isort `order-by-type`
   puts the class first, matching `matrix.py:30-40`).
4. `impl.get_runtime().prog` is kept (not `._prog`): that is what
   `ScalarNdarray.__init__` and `VectorNdarray.__init__` use at construction
   time, where a Program is guaranteed. The `._prog` spelling that PLAN §7.3
   item 2 calls for is a *teardown*-path concern, and the base `__del__` already
   handles it.

**`quadrants/program/ndarray.cpp`** (`:120` → `:99`). Taichi's one-line change
(`if (prog_)` → `if (prog_ || owns_imported_allocation)`) could not be reused:
Quadrants' destructor now does **two** things inside that guard, and the first
one dereferences `prog_`:

```cpp
Ndarray::~Ndarray() {
  if (prog_) {
    prog_->adstack_cache().erase_ndarray_data_gen(...);   // needs prog_
    ndarray_alloc_.device->dealloc_memory(ndarray_alloc_);
  }
}
```

Widening the single guard would have null-dereferenced on every imported
ndarray. The port splits it: the adstack bookkeeping stays under `if (prog_)`,
and the deallocation moves to a second `if (prog_ || owns_imported_allocation)`.

**`quadrants/program/launch_context_builder.{h,cpp}`** — declaration
`.h:134` → `.h:134`, the two new maps `.h:134` → `.h:232` (beside
`ndarray_shapes` rather than beside `array_ptrs`, which moved), definition
`.cpp:244` → `.cpp:340`. Three differences:

1. **The maps are keyed by `int`, not `std::vector<int>`.** Quadrants replaced
   the vector-valued argument id with a plain `int` everywhere
   (`set_arg_ndarray(int arg_id, ...)`, `BufferInfo::root_id` is now `int` at
   `codegen/spirv/kernel_utils.h:76`), so
   `std::unordered_map<std::vector<int>, uint64, hashing::Hasher<...>>` becomes
   `std::unordered_map<int, uint64>` — the shape of the neighbouring
   `array_runtime_sizes` (`launch_context_builder.h:221`).
2. **The recording moved into a helper**, `set_arg_ndarray_buffer_offset(int,
   const Ndarray &)`, because Quadrants has **three** entry points that take an
   `Ndarray` where Taichi had one: `set_arg_ndarray` (`:340`),
   `set_args_ndarray` (`:347`, the **bulk** form) and
   `set_arg_ndarray_with_grad` (`:376`). All three call the helper.
   `set_arg_ndarray_impl` cannot host it — it takes an `intptr_t`, not the
   `Ndarray`.
3. **The bulk form is the one that actually runs.** `python/quadrants/lang/kernel.py:702`
   batches every ndarray argument through `launch_ctx.set_args_ndarray(...)`;
   `set_arg_ndarray` has no Python caller. Recording only in the single-arg form
   (the literal Taichi port) would have left `array_byte_offsets` empty on every
   real launch. This is the hunk most worth a second reader.

**`quadrants/python/export_lang.cpp`** (`:439` → `:409`). Rewritten for
**nanobind**. The neighbouring `create_ndarray` binding (`:409-416`) is the
model, and the shape is close to identical — `py::arg` → `nb::arg`,
`py::return_value_policy::reference` → `nb::rv_policy::reference`, and the
whole thing stays a lambda returning `Ndarray *` inside the
`nb::class_<Program>` chain. Two deliberate small departures from the
neighbour: the lambda is non-capturing (`[]`, like `get_ndarray_data_ptr_as_int`
and `fill_float` two lines below) rather than `[&]`, and the default is written
`nb::arg("byte_offset") = uint64_t(0)` — **v1.3.0 has no integer-typed
`nb::arg` default anywhere to copy**, so the explicit `uint64_t` is there to
make the stored default's type unambiguous. See §5.

**`quadrants/runtime/gfx/runtime.cpp`** (`:513` → `:694`, as PLAN §7.3 item 1
predicted). The bind site is no longer a single `rw_buffer` call: Quadrants
picks between `ext_array_grads` and `any_arrays` on `bind.buffer.is_grad`, and
tolerates a missing entry with `kDeviceNullAllocation`:

```cpp
const auto &src = bind.buffer.is_grad ? ext_array_grads : any_arrays;
auto it = src.find(bind.buffer.root_id);
bindings->rw_buffer(bind.binding, it != src.end() ? it->second : kDeviceNullAllocation);
```

The offset branch is folded in ahead of that, and is gated on **three**
conditions rather than Taichi's one: an offset exists, the entry exists, **and
the binding is not a gradient**. The last is a judgement call and is new: an
offset is recorded against an *argument id*, and it describes the ndarray that
was passed for that argument, not whatever allocation its `.grad` lives in.
Binding the grad at the primal's offset would read the wrong bytes. Algan never
passes a gradient ndarray, so this arm is unreachable from Algan; it is guarded
because getting it wrong is a wrong picture rather than an error.

`MetalShaderResourceSet::rw_buffer(binding, DevicePtr, size)` still honours
`ptr.offset` — `quadrants/rhi/metal/metal_device.mm:292-305`, `rsc.buffer.offset
= ptr.offset` at `:300`. Confirmed at v1.3.0, as PLAN §7.3 item 1 says.

**`quadrants/program/program.cpp`** (`:376` → `:448`). Body ported unchanged
apart from the renames and the 120-column reflow. `program_impl_->get_compute_device()`,
`ndarrays_.insert({arr_ptr, std::move(arr)})` and the three-argument
`Ndarray(devalloc, type, shape)` constructor all still exist with the same
meaning; Quadrants added a *fourth* `Ndarray` constructor taking an explicit
`element_shape` (`program/ndarray.h:43-48`) but the port deliberately does not
use it — the element shape still arrives inside the tensor `DataType`, which is
what the Python side builds, and that keeps this function's signature identical
to Taichi's.

### Nothing dropped from 0001

Quadrants has no zero-copy MPS import of its own (PLAN §4 row 22: DLPack is
export-only, `import_mtl_buffer` has no caller). Every hunk was needed.

---

## 2. Patch 0002 — codegen and diagnostics

### DROPPED (Quadrants already fixes it)

**The MSL narrowing cast** — Taichi hunk `codegen/spirv/spirv_ir_builder.cpp`
`@@ -1224` (the redundant bitcast at `:1227`). Fixed independently in Quadrants by **`9542c0004`** (PR #543,
2026-04-25), present at v1.3.0 in
`quadrants/codegen/spirv/spirv_ir_builder.cpp:954-961`:

```cpp
// OpBitcast(T, T) is invalid per SPIR-V spec ("Result Type must not equal Operand Type"). ...
if (intermediate_dt != to) {
  ret = make_value(spv::OpBitcast, dst_type, ret);
}
```

Same defect, same place, one step earlier in the expression: Quadrants guards on
the `DataType` it just produced, Algan's patch guards on the SPIR-V type id
(`ret.stype.id != dst_type.id`). Equivalent — the intermediate type *is*
`dst_type` exactly when the ids match. Verified byte-for-byte at v1.3.0. This
confirms PLAN §4 row 23.

**The `CompiledTaichiKernel` raise** — Taichi hunk `runtime/gfx/runtime.cpp`
`@@ -250` (`CompiledTaichiKernel` ctor at `:223`). Quadrants' `CompiledQuadrantsKernel` constructor
(`quadrants/runtime/gfx/runtime.cpp:296-308`) already raises instead of pushing
a null pipeline, added by **`1da7d2ca6`** (PR #490, 2026-04-22):

```cpp
QD_ERROR_IF(res != RhiResult::success,
            "Failed to create pipeline for kernel task '{}' (RhiResult={}). ...", task_attribs[i].name, int(res));
```

It names the task and points at the RHI log, which is what Algan's version does.
Two cosmetic things it does *not* do — name the enclosing kernel as well, and
also test `vp == nullptr` — were **not** added, on the "do not duplicate a guard
Quadrants already has" rule. The `vp == nullptr` half is belt-and-braces anyway:
`create_pipeline_unique` (`rhi/public_device.h:682-688`) zero-initialises the
out-pointer, and `MetalDevice::create_pipeline` returns `error` whenever it is
still null.

Consequence: **patch 0002 no longer touches `runtime/gfx/runtime.cpp` at all.**
Only `metal_device.mm` is shared between the two patches.

**Also already present, so the corresponding Algan hunks shrank rather than
landing whole** (all in `quadrants/rhi/metal/metal_device.mm`):

| Guard | Quadrants | Commit |
| --- | --- | --- |
| `if (mtl_library == nil) return nullptr;` | `:148-150` (silent) | `b44f0e279` (#788) |
| `if (mtl_function == nil) return nullptr;` + the `computeFunction must not be nil` rationale | `:152-157` (silent) | `b44f0e279` (#788) |
| nil pipeline state reported **whether or not `err` was filled in**, with the XPC-compiler-service diagnosis | `:164-201` | `da5e8e039` (#591) |
| nil library reported both ways, same XPC text | `:1534-1563` | `da5e8e039` (#591) |
| `if (*out_pipeline == nullptr) return RhiResult::error;` | `:1387-1389` | `1da7d2ca6` (#490) |
| `QD_DUMP_MSL=1` → MSL of a *successful* translation to stderr | `:141-145` | `eff73c380` (#392) |

### Ported unchanged (rename only)

**The `ContinueStmt` / `gen_label_` fix** — `codegen/spirv/spirv_codegen.cpp`
`:1877` → **`:2016`**. `TaskCodegen::visit(ContinueStmt *)` is at
`quadrants/codegen/spirv/spirv_codegen.cpp:2016-2034`, with the
`gen_label_ = true;` at **`:2032`**, and its body is byte-identical to Taichi
1.7.4's `:1877-1895` apart from `TI_ASSERT` → `QD_ASSERT` and being an
out-of-line definition rather than an in-class one. Verified line by line.
`ir_->start_label(ir_->new_label())` replaces the flag exactly as in the Taichi
patch; `gen_label_` is thereafter never set true (its two readers in
`visit(IfStmt)`, `:1898` and `:1909`, become the `else` arm always), which is
the same end state the Taichi fork has. PLAN §4 row 24 said this was unfixed
upstream — confirmed at v1.3.0.

**`log_msl_source_context`** — the ±6-line numbered window around every
`program_source:<line>` a Metal diagnostic names. Quadrants has nothing like it
(PLAN §4 row 25: "no source window"). Ported verbatim, reflowed to 120 columns,
`RHI_LOG_ERROR` → `QD_WARN` (see below), placed in its own anonymous namespace
immediately before `MetalDevice::get_mtl_library`, as in Taichi.

### Rewritten

**`RHI_LOG_ERROR` → `QD_WARN` throughout the new code.** Adopting the rationale
Quadrants wrote at `metal_device.mm:190-198` and repeats at `:1558-1560`:
`QD_ERROR` throws a bare `std::string`, `MetalDevice::create_pipeline` is
`noexcept` and catches only `std::exception`, so a throw from anything it calls
trips `std::terminate` instead of the clean `RhiResult::error → Python
RuntimeError` path. `RHI_LOG_ERROR` is itself `QD_WARN("RHI Error : {}", msg)`
in this tree (`rhi/impl_support.h:22`), so the new helpers call `QD_WARN`
directly and format their own `[metal_device.mm] ` prefix, matching the two
sites Quadrants converted. `log_pipeline_failure` and `log_msl_source_context`
carry that reasoning in a comment so it does not get "tidied" back to
`QD_ERROR`.

**The nil-library and nil-function guards keep Quadrants' control flow and gain
Algan's message.** Quadrants returns `nullptr` from both **silently** — the
comment at `:154-155` explains why the *guard* exists but nothing is logged, and
`README.md`'s "Name every failure and let none of them be silent" is exactly
about that. So the `return nullptr;` lines are untouched and a
`log_pipeline_failure(...)` naming the task, the MSL size and the dump path is
inserted above each. Algan's own nil-function comment was dropped in favour of
Quadrants' (they say the same thing).

**The nil pipeline-state branch is left alone except for the dump path.**
Algan's patch replaced the whole block; Quadrants' message is better (it names
the kernel, the MSL byte size, and diagnoses the XPC-service drop). The only
edit is two lines appending `[dumped to <path>]` before the existing `QD_WARN`.

**`MetalDevice::create_pipeline`** — `:1309` → `:1372` (`:1380` with 0001 applied). Quadrants
already returns `error` on a null out-pointer, so only the *silent catch*
remains to fix, and it is a real gap: `catch (const std::exception &e)` binds
`e` and never reads it, which is the one place a `CompilerMSL`-**constructor**
parse failure is visible at all. The port adds `*out_pipeline = nullptr;` before
the `try`, logs `e.what()`, and adds the `catch (...)` arm. The existing
`if (*out_pipeline == nullptr)` block and its comment are untouched.

**`QD_SHADER_DUMP_DIR`** (was `TI_SHADER_DUMP_DIR`). Kept alongside Quadrants'
`QD_DUMP_MSL` rather than folded into it — they answer different questions:
`QD_DUMP_MSL` prints the MSL of a shader that *translated*, this writes the
`.spv` and the `.metal` of one that did *not*, and the `.spv` is the only
artifact that exists for a failure before MSL is produced. Quadrants has no
registry of environment variables (`QD_DUMP_MSL` is a bare `std::getenv` with no
declaration anywhere), so nothing else needed touching.

**`get_mtl_library`'s error branch** — `:1466` → `:1515` (`:1524` with 0001 applied). Quadrants
restructured it into "build `msgbuf` in one of two branches, then one
`QD_WARN`", where Taichi had the message inline under `if (err != nil)`. The
port hoists a `std::string description` out of the branch (so the source window
has something to scan after the `QD_WARN`) and calls `log_msl_source_context`
only when `description` is non-empty — i.e. never on the XPC branch, where Apple
returned no diagnostic at all and the fallback "first 40 lines" would print the
head of a translated megakernel for no reason. That gate is new; Taichi's
version had only the one branch.

---

## 3. Where things moved (quick index)

All line numbers are of the **pristine** file on each side (Taichi `b4b956f`,
Quadrants `ab9a58ab5`).

| What | Taichi 1.7.4 | Quadrants v1.3.0 |
| --- | --- | --- |
| `MetalDevice::import_mtl_buffer` | `rhi/metal/metal_device.mm:1221` | `:1280` |
| `MetalShaderResourceSet::rw_buffer(binding, DevicePtr, size)` | `rhi/metal/metal_device.mm:276` | `:292-305` (`ptr.offset` at `:300`) |
| `MetalPipeline::create_compute_pipeline` | `rhi/metal/metal_device.mm:108` | `:98` |
| `MetalDevice::create_pipeline` | `rhi/metal/metal_device.mm:1309` | `:1372` |
| `MetalDevice::get_mtl_function` | `rhi/metal/metal_device.mm:1451` | `:1503` |
| `MetalDevice::get_mtl_library` | `rhi/metal/metal_device.mm:1466` | `:1515` |
| gfx `ExtArr` bind site | `runtime/gfx/runtime.cpp:513` | `:694-697` |
| `CompiledTaichiKernel` ctor pipeline loop | `runtime/gfx/runtime.cpp:223` | `CompiledQuadrantsKernel`, `:273`; the raise at `:301-306` |
| `TaskCodegen::visit(ContinueStmt)` | `codegen/spirv/spirv_codegen.cpp:1877` (in-class) | `:2016` (out-of-line), flag at `:2032` |
| `TaskCodegen::visit(IfStmt)` (the flag's readers) | `codegen/spirv/spirv_codegen.cpp:1736` | `:1878`, reads at `:1898` and `:1909` |
| `IRBuilder::cast`, second (bitcast) step | `codegen/spirv/spirv_ir_builder.cpp:1227` | `:960`, now guarded at `:959` |
| `Program::create_ndarray` | `program/program.cpp:376` | `:448` |
| `Program::delete_ndarray` | `program/program.cpp:410` | `:472` |
| `LaunchContextBuilder::set_arg_ndarray` | `program/launch_context_builder.cpp:244` | `:340` (plus bulk `set_args_ndarray` at `:347`, `set_arg_ndarray_with_grad` at `:376`) |
| `Ndarray::~Ndarray` | `program/ndarray.cpp:120` | `:99` |
| `Ndarray::ndarray_alloc_` (insertion point) | `program/ndarray.h:50` | `:50` |
| `create_ndarray` pybind/nanobind | `python/export_lang.cpp:439-447` | `python/export_lang.cpp:409-416` |
| `Ndarray` class binding | `python/export_lang.cpp:584` (`"Ndarray"`) | `:529-543`, bound as **`"NdarrayCxx"`** |
| `ScalarNdarray` | `python/taichi/lang/_ndarray.py:229` | `python/quadrants/lang/_ndarray.py:445` |
| `VectorNdarray` (element-type model) | `python/taichi/lang/matrix.py:1766` | `python/quadrants/lang/matrix.py:2011-2038` |

Build wiring confirmed unchanged in kind: `-DQD_WITH_METAL` is set globally
(`cmake/QuadrantsCore.cmake:96-97`), `metal_device.mm` is compiled into the
static `metal_rhi` target (`quadrants/rhi/metal/CMakeLists.txt`) which
`qd_device_api` links, and a plain `.cpp` already calls across into the `.mm`
(`runtime/program_impls/metal/metal_program.cpp:21-24` calls
`metal::MetalDevice::create()`), so the linkage `import_external_mtl_buffer`
relies on has a working precedent. `metal_device.h` is already included from
plain C++ (`quadrants/python/dlpack_funcs.cpp`, `metal_program.h`).

---

## 4. Types whose spelling changed — check these first on a real build

| Type / name | Taichi | Quadrants | Where it matters |
| --- | --- | --- | --- |
| argument id | `std::vector<int>` | `int` | both new `LaunchContextBuilder` maps; `BufferInfo::root_id` |
| `Ndarray` Python binding name | `Ndarray` | `NdarrayCxx` | not referenced by the patch (we go through `Program`), noted for readers |
| element `DataType` wrapper | `_ti_core...get_tensor_type(...)` used directly | `DataTypeCxxWrapper(... .get_ptr())` | `ExternalMetalNdarray.__init__` |
| `Ndarray.shape` (Python) | plain attribute | read-only `@property` over `_physical_shape` | `ExternalMetalNdarray.__init__` |
| `DeviceAllocation` | unchanged | unchanged (`rhi/public_device.h:88`), `get_ptr(uint64_t offset = 0) const` at `:93` | gfx bind site |
| `DataType` (C++) | unchanged | unchanged | `Program::create_ndarray_from_metal_buffer` |
| pybind11 `py::arg` / `return_value_policy` | — | nanobind `nb::arg` / `nb::rv_policy` | `export_lang.cpp` |

---

## 5. Things I am less than sure of

Ranked by how likely they are to bite, and all of them are compile-or-first-run
questions rather than design ones.

1. **The nanobind default argument `nb::arg("byte_offset") = uint64_t(0)`.**
   v1.3.0 has no integer-typed `nb::arg` default anywhere in `quadrants/python/`
   to copy — every existing default is a `bool`, an enum, a `std::vector<int>{}`,
   a `DataType` or a `DebugInfo`. nanobind builds the default by `nb::cast`-ing
   the value at module-init time, which is fine for an integral type, but if the
   build objects the fix is to drop the default and always pass `byte_offset`
   (the Python caller in `ExternalMetalNdarray.__init__` already passes it
   positionally, and so does Algan's `import_tensor`).
2. **`set_args_ndarray` is the only path that matters.** I traced the launch
   path to `python/quadrants/lang/kernel.py:702` and there is no other Python
   caller of either setter, but Quadrants also has a `LaunchContextBufferCache`
   that can *skip* the whole `set_args_*` block on a cache hit and reuse a
   prepared `LaunchContextBuilder`. That is correct here — the cache key
   includes the ndarray objects, so a reused context carries the offsets its own
   arrays recorded, exactly as it already carries `array_runtime_sizes` — but it
   is reasoning, not a measurement. If a zero-copy render draws correctly on the
   first launch and wrongly on the second, this is the place to look.
3. **The `is_grad` gate at the gfx bind site.** New, not in Algan's patch,
   unreachable from Algan. If Quadrants' autodiff ever *does* want an offset
   gradient the gate is where to change it.
4. **`Ndarray::~Ndarray` split.** The reordering is mechanical, but it changes a
   destructor that Quadrants recently rewrote for the adstack cache. Worth one
   read by someone who knows what `erase_ndarray_data_gen` is for; the port
   assumes it is meaningless for an ndarray with no `Program`.
5. **`ExternalMetalNdarray` and `impl.get_runtime().ndarrays`.** The base
   `Ndarray.__init__` registers every instance with the runtime so `qd.reset()`
   can call `_reset()` on it. An imported ndarray now participates in that, which
   Taichi's version did not (Taichi's base had no such registry at 1.7.4). It
   should be right — `_reset()` just drops the Python-side handles — but it means
   a `qd.reset()` mid-render invalidates Algan's import cache without telling it.
   `mps_zero_copy.clear_import_cache()` is the existing lever if that ever shows
   up.
6. **`isalnum` / `isdigit` unqualified** (from `<cctype>`), carried over from the
   Taichi patch, which did build on macOS CI. If a stricter libc++ objects, add
   `std::`.
7. **`QD_DLL_EXPORT` on `import_external_mtl_buffer`.** Not applied, matching
   both Taichi's version and the `MetalDevice` class next to it. The project sets
   `CMAKE_CXX_VISIBILITY_PRESET hidden`, but `metal_rhi` is a *static* library
   linked into the same final object as its caller, so hidden visibility does not
   break the link. Metal is macOS-only, so the Windows dllexport question does
   not arise.
8. **Nothing was compiled, run, or rendered.** In particular no Metal shader has
   been produced by the patched `visit(ContinueStmt)`, and the new diagnostics
   have never printed. The `--fast` suite and `tests/full_renders` on a Mac are
   the first real evidence.

## 6. Follow-ups this port does not do

* **The `ContinueStmt` regression test** PLAN §7.3 item 1 asks for
  (`if qd.static(cond): continue` inside a `range_for`, run on a SPIR-V backend)
  is not written — it belongs in `tests/`, which this task may not touch.
* **PLAN §4 rows 26 and 29** (Metal f32 atomics as patch 0003, shared torch MPS
  command queue as patch 0004) are separate items and not started. Note that row
  26's source commit `b44f0e279` is *already in v1.3.0* — it is what put the nil
  guards in `create_compute_pipeline` — so on this base row 26 is a no-op and
  should be re-checked rather than ported.
* **Algan-side glue** (PLAN §7.3 item 2's thirteen breaking differences,
  `taichi_fast_launch`, `mps_zero_copy`'s sync pair) is untouched: this task was
  the patches only.

---

## 7. Patch 0003 — pre-Volta (sm_61) CUDA

Base: Quadrants `v1.3.0` (`ab9a58ab5`), LLVM 22.1.0. Touches three files, all CUDA/LLVM-runtime:
`quadrants/runtime/llvm/kernel_atomic_syncscope.h`, `quadrants/runtime/llvm/llvm_context.cpp`,
`quadrants/codegen/cuda/codegen_cuda.cpp`. No overlap with 0001/0002 (Metal).

Quadrants 1.3.0 cannot `qd.init(qd.gpu)` on a GPU older than sm_70. Two independent defects, both
introduced by LLVM 22's NVPTX backend (LLVM 15, which Taichi 1.7.4 used, has neither), both invisible
to Quadrants' own GPU CI because that CI is a T4 — sm_75, which happens to be the first capability
where both defects disappear.

---

### Defect (a) — `.sys`-scope atomics reject the runtime module at load

**Mechanism.** `kernel_atomic_syncscope()` (`kernel_atomic_syncscope.h:29-34` at v1.3.0) returns
`llvm::SyncScope::System` for everything except AMDGPU. LLVM 22's NVPTX emits an *explicit* scope
qualifier on every atomic as soon as the target is sm_60 or newer —
`NVPTXSubtarget::hasAtomScope()` is `SmVersion >= 60`, and
`NVPTXDAGToDAGISel::getAtomicScope()` returns `Scope::DefaultDevice` (no qualifier) only below that.
So System scope becomes `atom.sys.*` on Pascal and up. On the measured GTX 1050 (sm_61, driver
576.52) the driver rejects a module containing `atom.sys.cas.b64` with `CUDA_ERROR_NOT_SUPPORTED`
from `cuModuleLoadDataEx`; the same module rewritten to `atom.gpu`, `atom.cta` or unscoped `atom`
loads. Module loading is all-or-nothing, so one such instruction kills the whole runtime module.

The instruction in question is the `__atomic_compare_exchange_n` CAS loop at the tail of
`runtime_eval_adstack_max_reduce` (`adstack_runtime.cpp:652-663`). It is the *only* System-scope
atomic that survives in the CUDA runtime module because `QuadrantsLLVMContext::init_runtime_module`
runs `eliminate_unused_functions(runtime_module, ...)` keeping only `runtime_*` / `LLVMRuntime_*`
(`llvm_context.cpp:1158-1160`): this function matches the filter, whereas `stack_push` — which holds
the runtime's other two internal `__atomic_*` calls (`adstack_runtime.cpp:849,858`) — does not, and
is dropped. That is why a function Algan never calls takes the whole runtime down, and it matches the
measurement of exactly one `.sys` instruction in the runtime module.

Kernel-side atomics hit the same wall through the five emit sites that route through the helper
(`codegen_llvm.cpp:1349,1376,1417,1421,1425` and `llvm_context.cpp:343`). `atomicrmw max i64` and
`atomicrmw fmin float` have no native NVPTX instruction and are expanded to `cmpxchg` loops that
carry the scope with them, so they become `atom.sys.cas.{b64,b32}` too.

**Change 1 — `kernel_atomic_syncscope.h`.** CUDA below sm_70 now returns
`ctx->getOrInsertSyncScopeID("device")`. `"device"` is one of the five names the NVPTX backend
accepts (`NVPTXScopes::NVPTXScopes` in `NVPTXISelDAGToDAG.cpp` registers `""`, `block`, `cluster`,
`device`, `singlethread`; anything else is a fatal usage error), and it selects the `.gpu` qualifier.
This is the same argument the file already makes for AMDGPU's `"agent"`: one kernel is one device,
the host only reads results after kernel completion, and the ops this helper covers never need
host-visible mid-kernel atomicity. The capability is read with
`CUDAContext::get_instance().get_compute_capability()` under `#if defined(QD_WITH_CUDA)` — the same
accessor `llvm_context.cpp:368` and `codegen_cuda.cpp:398` already use. The new include is safe:
`-DQD_WITH_CUDA` is a global flag (`cmake/QuadrantsCore.cmake:85`), and `cuda_types.h` only pulls
`<cuda.h>` under `QD_WITH_CUDA_TOOLKIT`, whose include dirs are `PUBLIC` on the core library
(`QuadrantsCore.cmake:268`), so `codegen_llvm.cpp` can see them.

**Change 2 — `llvm_context.cpp`.** The runtime module's CAS is re-scoped in IR, in the CUDA branch of
`module_from_file`, right after the existing `cuda_compute_capability` patch and before the
`patch_intrinsic` block.

*Why it is not done in the C++ source, as PLAN §7.3 suggested.* `__atomic_compare_exchange_n` takes
no scope argument, and clang's scoped replacements do not help here. Measured with clang 18 on
`x86_64-pc-linux-gnu`:

```
__scoped_atomic_compare_exchange_n(slot, &expected, v, false,
                                   __ATOMIC_RELAXED, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE)
  ->  cmpxchg ptr %9, i64 %12, i64 %13 monotonic monotonic, align 8      <- no syncscope
__atomic_compare_exchange_n(slot, &expected, v, false,
                            __ATOMIC_RELAXED, __ATOMIC_RELAXED)
  ->  cmpxchg ptr %9, i64 %12, i64 %13 monotonic monotonic, align 8      <- byte-identical
```

The reason is the *host triple*: `runtime_module/CMakeLists.txt` compiles `runtime.cpp` (which
`#include`s `adstack_runtime.cpp` — one TU) with `${CLANG_EXECUTABLE} -c runtime.cpp -emit-llvm`, no
`-target`. Only targets with their own `TargetCodeGenInfo::getLLVMSyncScopeID` override (AMDGPU,
NVPTX, SPIR-V) honour `__MEMORY_SCOPE_*`; the base implementation maps every scope to the default
(System), and x86_64 / aarch64 use the base one. The scope therefore has to be applied host-side
after the module is retargeted to `nvptx64-nvidia-cuda`, which is exactly what `llvm_context.cpp`
exists to do — it already replaces the runtime's C++ CAS-loop `atomic_*` bodies with single
`atomicrmw`s (`patch_atomic_rmw`) and its stubs with intrinsics (`patch_intrinsic`) for the same
reason.

The rewrite is deliberately narrow: it runs only when `kernel_atomic_syncscope()` returns something
other than System, only inside `runtime_eval_adstack_max_reduce`, and only on atomics that are still
System-scoped. The runtime atomics that genuinely need System scope — the pinned-host adstack
overflow flag the host polls mid-kernel, which the header explicitly carves out — are in
`stack_push`, a different function, and are untouched.

**Correct on post-Volta.** Both changes are keyed to `kernel_atomic_syncscope()` returning a
non-System scope, which at sm_70 and above it never does. On sm_70 and newer the emitted PTX is unchanged,
byte for byte, on CUDA; AMDGPU and CPU are not on any path this patch modifies.

---

### Defect (b) — `llvm.nvvm.activemask` cannot be selected

**Mechanism — corrected relative to PLAN §7.3.** `optimized_reduction` (`codegen_cuda.cpp:362-388`)
maps a reduction onto the runtime's warp-aggregated `reduce_*` helpers (`runtime.cpp:1479-1509`,
`DEFINE_REDUCTION`), whose first statement is `cuda_active_mask()`; `llvm_context.cpp:441` patches
that stub into `llvm.nvvm.activemask`. The NVPTX pattern for it is

```
def ACTIVEMASK : BasicNVPTXInst<(outs B32:$dest), (ins), "activemask.b32",
                                [(set i32:$dest, (int_nvvm_activemask))]>,
                 Requires<[hasPTX<62>, hasSM<30>]>;          // NVPTXIntrinsics.td
```

with `class hasPTX<int v> : Predicate<"Subtarget->getPTXVersion() >= " # v>` and
`class hasSM<int v> : Predicate<"Subtarget->getSmVersion() >= " # v>` (`NVPTXInstrInfo.td:143-144`).
PLAN reads this as an sm_70 gate. It is not: the *hardware* bound is sm_30 and is met on Pascal. What
fails is `hasPTX<62>` — `activemask.b32` was added in **PTX ISA 6.2**.

Quadrants builds the NVPTX `TargetMachine` with an empty feature string
(`jit_cuda.cpp:265`, `createTargetMachine(triple, get_mcpu(), "", ...)`) and sets no PTX-version
module flag, so LLVM defaults `PTXVersion` to the *minimum* for the target SM
(`NVPTXSubtarget::initializeSubtargetDependencies` → `getMinPTXVersionForSM`):

| target | default PTX | `hasPTX<62>` |
| --- | --- | --- |
| sm_60 / sm_61 / sm_62 | 5.0 | no |
| sm_70 | 6.0 | no |
| sm_72 | 6.1 | no |
| sm_75 | 6.3 | **yes** |
| sm_80 and later | 7.0+ | yes |

That table is confirmed from the outside by the PTX header the maintainer observed on the GTX 1050:
`.version 5.0 / .target sm_61` is exactly `getMinPTXVersionForSM(SM(61)) == 50`.

Two consequences. First, the intrinsic reaches instruction selection and aborts with
`LLVM Fatal Error: Cannot select: intrinsic %llvm.nvvm.activemask`. Second — and this is the
correction — **sm_70 and sm_72 are broken today as well**, not just pre-Volta. A `cap >= 70` gate as
PLAN specifies would have left V100 exactly as broken as it is now.

**Change 3 — `codegen_cuda.cpp`.** `optimized_reduction` returns `nullptr` below
`kMinComputeCapabilityForWarpReduction = 75`, using the same
`CUDAContext::get_instance().get_compute_capability()` accessor as the `cap >= 60` half2 gate in
`visit(AtomicOpStmt *)` 20 lines below (`codegen_cuda.cpp:398-399` — note PLAN locates that gate
inside `optimized_reduction`; it is not, it is in `visit`). `nullptr` is the documented "cannot
optimize" answer: `TaskCodeGenLLVM::visit(AtomicOpStmt *)` falls through to `quant_type_atomic` →
`real_type_atomic` → `integral_type_atomic` (`codegen_llvm.cpp:1449-1459`), i.e. a plain hardware
atomic. The base-class `TaskCodeGenLLVM::optimized_reduction` returns `nullptr` unconditionally
(`codegen_llvm.cpp:1287-1289`), so this is the path the CPU backend always takes.

The threshold is a named constant with the derivation in the comment above it, so that if Quadrants
ever passes an explicit `+ptxNN` feature it is obvious the bound can drop to the hardware's own
sm_30.

**Correct on post-Volta.** For sm_75 and up — Turing, Ampere, Ada, Hopper, Blackwell, and Quadrants'
own CI — the gate never fires and codegen is unchanged. For sm_70/72 it replaces a hard compile abort
with working, slower code: a fix, not a regression, since no sm_70 user can be relying on today's
behaviour.

---

### Deviations from PLAN §7.3 "Prerequisite 0"

1. **`cap >= 70` → `cap >= 75`** for the `optimized_reduction` gate, because the LLVM predicate is on
   PTX ISA version, not SM, and Quadrants' default PTX version only reaches 6.2 at sm_75. Reasoned
   from LLVM 22 source, corroborated by the observed `.version 5.0`; not measured on sm_70 hardware.
2. **The adstack cmpxchg is re-scoped in `llvm_context.cpp`, not in `adstack_runtime.cpp`.** PLAN
   flagged this as a possibility ("check … whether the scoped builtins are available there"); they
   are not, for the host-triple reason above, measured with clang 18.
3. **The syncscope change is not "one line".** With the capability gate it is a guarded branch plus a
   guarded include. An *unconditional* `"device"` for CUDA would be one line and needs no
   `CUDAContext` at all — see below.

### Deliberately not changed

- **An unconditional CUDA → `"device"` scope.** It would be a smaller, dependency-free diff, and the
  header's own rationale for AMDGPU applies verbatim to NVPTX; PLAN's own upstreaming argument is
  that device scope is "more correct *and* faster than system scope". It is not what this patch does
  because it would change the PTX emitted on sm_70+, which this patch is required not to do. Worth
  putting to upstream as the alternative.
- **`codegen_llvm.cpp:1327` (`AtomicOpType::cas`) and the real-typed `xchg` at `:1431`.** Both build
  their atomic without the helper and so keep System scope, inconsistently with the integer path a
  few lines away. They will still emit `atom.sys.*` on pre-Volta, so `qd.atomic_cas` and float
  `qd.atomic_xchg` stay broken there. Routing them through the helper would also change AMDGPU
  behaviour, so it is left as a separate upstream question rather than folded in here.
- **`stack_push`'s overflow-flag atomics** (`adstack_runtime.cpp:849,858`). They target pinned host
  memory the host polls during kernel execution and genuinely need System scope; the header documents
  the carve-out. They are stripped from the runtime module by `eliminate_unused_functions` but are
  linked into kernels that use adstacks, where they will still emit `.sys` on pre-Volta. Autodiff on
  Pascal is therefore still expected to fail.
- **`Pointer_activate` (`node_pointer.h:48`) and `qd.simt.warp.active_mask()`.** Both call
  `cuda_active_mask()` outside `optimized_reduction`, so sparse `pointer` SNodes and that one SIMT
  builtin still hit the `Cannot select` abort below sm_75.
- **The PTX version itself.** Passing an explicit `+ptx62`-or-higher feature string in
  `jit_cuda.cpp` would fix (b) outright *and* keep the warp-aggregated fast path on Pascal, and is
  probably the better long-term fix. It changes the emitted PTX on every target and depends on the
  installed driver accepting the requested ISA, so it is not something to do without hardware.

---

### What has been verified here, and what has not

Verified on this machine (no GPU, no LLVM 22 toolchain):

- `git apply --check --whitespace=error` is clean against pristine `v1.3.0`, and 0001 → 0002 → 0003
  apply in order with no conflicts. The three files 0003 touches are disjoint from 0001's eleven and
  0002's two.
- All three files pass `clang-format --dry-run --Werror` against the repo's `.clang-format`.
- Every identifier and accessor used exists at `v1.3.0` or in LLVM 22.1:
  `AtomicCmpXchgInst::{get,set}SyncScopeID` and `AtomicRMWInst::{get,set}SyncScopeID`
  (`llvm/IR/Instructions.h`), `SyncScope::ID` = `uint8_t` with `System = 1`
  (`llvm/IR/LLVMContext.h`), `"device"` in `NVPTXScopes`, `CUDAContext::get_compute_capability()`,
  `using namespace llvm;` at `llvm_context.cpp:75` and `codegen_cuda.cpp:25`, C++17
  if-with-initializer already used at `llvm_context.cpp:998`.

Not verified — nothing in this patch has been compiled or run:

1. **A compile.** Linux with `QD_WITH_CUDA=ON` (`python build.py` or CMake) is the minimum bar; it
   would prove the three TUs still compile, in particular that pulling `cuda_context.h` into
   `codegen_llvm.cpp` via `kernel_atomic_syncscope.h` is benign in a real CUDA build, and in a
   `QD_WITH_CUDA_TOOLKIT=ON` build too.
2. **sm_61 end to end.** `qd.init(qd.gpu)` succeeds; dump the runtime module's PTX and confirm
   `atom.gpu.cas.b64` with no `atom.sys` anywhere; then the 90-frame Algan video and a frame, checked
   against the Taichi 1.7.4 arm (0 of 419,904 pixels differed when the same two defects were
   *emulated* with the `cuModuleLoadDataEx` hook plus `make_thread_local=False`, so bit-identical is
   the expected result).
3. **No regression on sm_75+.** The strongest check is PTX byte-identity before and after on a
   Turing/Ampere card, plus Quadrants' own `tests/python/test_simt.py`, `test_cuda_internals.py` and
   the reduction tests.
4. **sm_70 / sm_72.** The claim that 1.3.0 is broken there today and that this patch fixes it is
   inferred from LLVM 22's `getMinPTXVersionForSM` table, not measured. A V100 or Xavier run would
   settle it, and would also decide whether the gate belongs at 70 or 75.
5. **The "pre-Volta" attribution for (a).** The evidence is one sm_61 Windows box. Whether the driver
   refuses `atom.sys` as a function of compute capability, of WDDM, or of
   `cudaDevAttrHostNativeAtomicSupported`, is not established. `cap < 70` is safe either way — device
   scope is sufficient for these ops on every target — but if the discriminator is not the
   capability, some sm_70+ configurations would still need the device scope, which is an argument for
   the unconditional variant above.
6. **AMDGPU.** Untouched by construction (the helper's AMDGPU branch is first and unchanged; the
   `llvm_context.cpp` rewrite is inside `if (arch_ == Arch::cuda)`), but a gfx9 run of the reduction
   tests would confirm.
