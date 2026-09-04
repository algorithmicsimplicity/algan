"""ctypes bindings for ``libtaichi_c_api.so``, for the arch-coexistence probes.

``DESIGN_taichi_arch_coexistence.md`` §4 rests on one mechanism: the C API lives
in a **different shared object** from ``taichi_python*.so`` and holds its own
globals, so ``ti_create_runtime(TI_ARCH_X64, 0)`` succeeds beside a live Python
``Program`` on any arch. This module is the Python side of that, written for the
Phase 0 experiments (§8) rather than for production -- if Phase 0 says go, it is
the thing that gets promoted to ``algan/rendering/taichi_c_api.py``.

Two contracts it enforces, because the C API provides neither (§5.5):

* **Errors are return codes, not exceptions.** Every call goes through
  :func:`_check`, which reads ``ti_get_last_error`` and raises. A missed check
  is a silently wrong answer, not a crash.
* **Struct layouts are version-locked.** :data:`LAYOUT` records the sizes and
  offsets these declarations imply. ``benchmarks/_taichi_c_api_layout_check.py``
  compiles a C oracle against the installed headers and compares, so a taichi
  upgrade that moves a field fails loudly instead of corrupting memory.

Nothing here imports algan, so a probe can set ``ALGAN_*`` before importing it.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

# --- library discovery ------------------------------------------------------


def taichi_lib_root() -> Path:
    """Directory of the installed ``taichi/_lib``.

    Deliberately the literal ``taichi`` package, not ``algan.taichi_compat``:
    this module binds ``libtaichi_c_api.so`` out of *Taichi's own*
    distribution specifically (see the module docstring), which is a
    different question from which compiler Algan itself is using. Nothing
    else in this module touches algan, so importing it does not by itself
    risk a mixed-compiler process -- but a caller that also imports algan
    modules should make sure ``ALGAN_TAICHI_BACKEND=taichi`` first (see
    ``_taichi_arch_coexistence_probe.py``'s ``_require_taichi_backend``).
    """
    import taichi

    return Path(taichi.__file__).parent / "_lib"


def c_api_library_path() -> Path:
    """Path to ``libtaichi_c_api`` for this platform."""
    root = taichi_lib_root() / "c_api" / "lib"
    names = {
        "win32": ("taichi_c_api.dll",),
        "darwin": ("libtaichi_c_api.dylib",),
    }.get(sys.platform, ("libtaichi_c_api.so",))
    for name in names:
        candidate = root / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no taichi C API library under {root}")


def ensure_ti_lib_dir() -> str:
    """Point ``TI_LIB_DIR`` at ``taichi/_lib/runtime``.

    ``ti_create_runtime`` fails with a ``runtime_lib_dir`` error without it
    (§4). Set unconditionally rather than only when unset: a stale value from
    another taichi install is worse than none.
    """
    runtime_dir = str(taichi_lib_root() / "runtime")
    os.environ["TI_LIB_DIR"] = runtime_dir
    return runtime_dir


# --- enums ------------------------------------------------------------------

TI_ARCH_RESERVED = 0
TI_ARCH_VULKAN = 1
TI_ARCH_METAL = 2
TI_ARCH_CUDA = 3
TI_ARCH_X64 = 4
TI_ARCH_ARM64 = 5
TI_ARCH_OPENGL = 6
TI_ARCH_GLES = 7

TI_ARGUMENT_TYPE_I32 = 0
TI_ARGUMENT_TYPE_F32 = 1
TI_ARGUMENT_TYPE_NDARRAY = 2
TI_ARGUMENT_TYPE_TEXTURE = 3
TI_ARGUMENT_TYPE_SCALAR = 4
TI_ARGUMENT_TYPE_TENSOR = 5

TI_DATA_TYPE_F16 = 0
TI_DATA_TYPE_F32 = 1
TI_DATA_TYPE_F64 = 2
TI_DATA_TYPE_I8 = 3
TI_DATA_TYPE_I16 = 4
TI_DATA_TYPE_I32 = 5
TI_DATA_TYPE_I64 = 6
TI_DATA_TYPE_U1 = 7
TI_DATA_TYPE_U8 = 8
TI_DATA_TYPE_U16 = 9
TI_DATA_TYPE_U32 = 10
TI_DATA_TYPE_U64 = 11

TI_ERROR_SUCCESS = 0

#: ``TiNdShape.dims`` is a fixed array; overflowing it corrupts the struct.
TI_MAX_DIM_COUNT = 16

_TORCH_DTYPE_TO_TI = {
    "torch.float16": TI_DATA_TYPE_F16,
    "torch.float32": TI_DATA_TYPE_F32,
    "torch.float64": TI_DATA_TYPE_F64,
    "torch.int8": TI_DATA_TYPE_I8,
    "torch.int16": TI_DATA_TYPE_I16,
    "torch.int32": TI_DATA_TYPE_I32,
    "torch.int64": TI_DATA_TYPE_I64,
    "torch.uint8": TI_DATA_TYPE_U8,
}


def ti_data_type_of(tensor) -> int:
    """Map a torch dtype to its ``TiDataType``, refusing anything unmapped."""
    key = str(tensor.dtype)
    try:
        return _TORCH_DTYPE_TO_TI[key]
    except KeyError:
        raise TypeError(f"no TiDataType for torch dtype {key}") from None


# --- structs ----------------------------------------------------------------
#
# Transcribed from ``taichi/_lib/c_api/include/taichi/taichi_core.h``. Field
# order and type are load-bearing: the union is written by us and read by C, so
# a mismatch is memory corruption rather than an error. Keep LAYOUT in step.


class TiNdShape(ctypes.Structure):
    _fields_ = [
        ("dim_count", ctypes.c_uint32),
        ("dims", ctypes.c_uint32 * TI_MAX_DIM_COUNT),
    ]


class TiNdArray(ctypes.Structure):
    _fields_ = [
        ("memory", ctypes.c_void_p),  # TiMemory
        ("shape", TiNdShape),
        ("elem_shape", TiNdShape),
        ("elem_type", ctypes.c_int),  # TiDataType
    ]


class TiImageExtent(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("depth", ctypes.c_uint32),
        ("array_layer_count", ctypes.c_uint32),
    ]


class TiTexture(ctypes.Structure):
    _fields_ = [
        ("image", ctypes.c_void_p),  # TiImage
        ("sampler", ctypes.c_void_p),  # TiSampler
        ("dimension", ctypes.c_int),  # TiImageDimension
        ("extent", TiImageExtent),
        ("format", ctypes.c_int),  # TiFormat
    ]


class TiScalarValue(ctypes.Union):
    _fields_ = [
        ("x8", ctypes.c_uint8),
        ("x16", ctypes.c_uint16),
        ("x32", ctypes.c_uint32),
        ("x64", ctypes.c_uint64),
    ]


class TiScalar(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),  # TiDataType
        ("value", TiScalarValue),
    ]


class TiTensorValue(ctypes.Union):
    _fields_ = [
        ("x8", ctypes.c_uint8 * 128),
        ("x16", ctypes.c_uint16 * 64),
        ("x32", ctypes.c_uint32 * 32),
        ("x64", ctypes.c_uint64 * 16),
    ]


class TiTensorValueWithLength(ctypes.Structure):
    _fields_ = [
        ("length", ctypes.c_uint32),
        ("data", TiTensorValue),
    ]


class TiTensor(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),  # TiDataType
        ("contents", TiTensorValueWithLength),
    ]


class TiArgumentValue(ctypes.Union):
    _fields_ = [
        ("i32", ctypes.c_int32),
        ("f32", ctypes.c_float),
        ("ndarray", TiNdArray),
        ("texture", TiTexture),
        ("scalar", TiScalar),
        ("tensor", TiTensor),
    ]


class TiArgument(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),  # TiArgumentType
        ("value", TiArgumentValue),
    ]


#: Sizes and offsets these declarations imply, checked against a C oracle
#: compiled from the installed headers by ``_taichi_c_api_layout_check.py``.
#: This is §5.5's version lock, derived rather than hand-copied so it cannot
#: disagree with the classes above.
def layout() -> dict:
    """Sizes and field offsets of every struct declared here."""
    out = {}
    for cls in (
        TiNdShape,
        TiNdArray,
        TiImageExtent,
        TiTexture,
        TiScalarValue,
        TiScalar,
        TiTensorValue,
        TiTensorValueWithLength,
        TiTensor,
        TiArgumentValue,
        TiArgument,
    ):
        entry = {"sizeof": ctypes.sizeof(cls), "alignof": ctypes.alignment(cls)}
        for name, _ in cls._fields_:
            entry[f"offsetof.{name}"] = getattr(cls, name).offset
        out[cls.__name__] = entry
    return out


LAYOUT = layout()


# --- errors -----------------------------------------------------------------


class TaichiCApiError(RuntimeError):
    """A non-success ``ti_get_last_error``, raised where C would return one."""


# --- the library ------------------------------------------------------------


class _Lib:
    """The loaded ``libtaichi_c_api`` with argtypes declared."""

    def __init__(self, path: Path):
        self.path = path
        self.dll = ctypes.CDLL(str(path))
        d = self.dll

        d.ti_create_runtime.argtypes = [ctypes.c_int, ctypes.c_uint32]
        d.ti_create_runtime.restype = ctypes.c_void_p

        d.ti_destroy_runtime.argtypes = [ctypes.c_void_p]
        d.ti_destroy_runtime.restype = None

        d.ti_get_last_error.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_char_p,
        ]
        d.ti_get_last_error.restype = ctypes.c_int

        d.ti_set_last_error.argtypes = [ctypes.c_int, ctypes.c_char_p]
        d.ti_set_last_error.restype = None

        d.ti_import_cpu_memory.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        d.ti_import_cpu_memory.restype = ctypes.c_void_p

        d.ti_free_memory.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        d.ti_free_memory.restype = None

        d.ti_load_aot_module.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        d.ti_load_aot_module.restype = ctypes.c_void_p

        d.ti_destroy_aot_module.argtypes = [ctypes.c_void_p]
        d.ti_destroy_aot_module.restype = None

        d.ti_get_aot_module_kernel.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        d.ti_get_aot_module_kernel.restype = ctypes.c_void_p

        d.ti_launch_kernel.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(TiArgument),
        ]
        d.ti_launch_kernel.restype = None

        d.ti_wait.argtypes = [ctypes.c_void_p]
        d.ti_wait.restype = None

        d.ti_get_available_archs.argtypes = [
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_int),
        ]
        d.ti_get_available_archs.restype = None

    def check(self, context: str) -> None:
        """Raise if the C API recorded an error, and clear it either way."""
        size = ctypes.c_uint64(0)
        code = self.dll.ti_get_last_error(ctypes.byref(size), None)
        if code == TI_ERROR_SUCCESS:
            return
        message = ""
        if size.value:
            buffer = ctypes.create_string_buffer(int(size.value) + 1)
            self.dll.ti_get_last_error(ctypes.byref(size), buffer)
            message = buffer.value.decode("utf-8", "replace")
        # Reading does not clear it in 1.7.4, so clear explicitly -- otherwise
        # one stale error fails every later call.
        self.dll.ti_set_last_error(TI_ERROR_SUCCESS, None)
        raise TaichiCApiError(f"{context}: error {code}: {message}")

    def available_archs(self) -> list:
        count = ctypes.c_uint32(0)
        self.dll.ti_get_available_archs(ctypes.byref(count), None)
        if not count.value:
            return []
        archs = (ctypes.c_int * count.value)()
        self.dll.ti_get_available_archs(ctypes.byref(count), archs)
        return list(archs[: count.value])


_LIB = None


def lib() -> _Lib:
    """Load ``libtaichi_c_api`` once."""
    global _LIB
    if _LIB is None:
        ensure_ti_lib_dir()
        _LIB = _Lib(c_api_library_path())
    return _LIB


# --- the runtime ------------------------------------------------------------


class CApiRuntime:
    """One ``TiRuntime`` on ``arch``, plus the AOT modules loaded into it.

    Not thread-safe by construction, and deliberately so: §5.7 records that
    P13's evidence for cross-thread Python-side launches does not transfer to a
    different runtime object in a different shared object. One owner per
    instance until something measures otherwise.
    """

    def __init__(self, arch: int = TI_ARCH_X64, device_index: int = 0):
        self.arch = arch
        self._lib = lib()
        handle = self._lib.dll.ti_create_runtime(arch, device_index)
        self._lib.check(f"ti_create_runtime(arch={arch})")
        if not handle:
            raise TaichiCApiError(f"ti_create_runtime(arch={arch}) returned null")
        self.handle = ctypes.c_void_p(handle)
        self._modules = {}
        self._kernels = {}
        # (data_ptr, nbytes) -> TiMemory. See _import_memory: an imported CPU
        # handle CANNOT be released on an x64 runtime, so re-importing the same
        # buffer every launch grows the process without bound.
        self._imported = {}
        self.imports = 0
        self.import_hits = 0

    # -- modules --

    def load_module(self, path) -> None:
        """Load an AOT module directory and index every kernel it is asked for."""
        key = str(path)
        if key in self._modules:
            return
        handle = self._lib.dll.ti_load_aot_module(self.handle, key.encode())
        self._lib.check(f"ti_load_aot_module({key})")
        if not handle:
            raise TaichiCApiError(f"ti_load_aot_module({key}) returned null")
        self._modules[key] = ctypes.c_void_p(handle)

    def kernel(self, name: str):
        """Resolve ``name`` in the loaded modules, cached."""
        if name in self._kernels:
            return self._kernels[name]
        for module in self._modules.values():
            handle = self._lib.dll.ti_get_aot_module_kernel(module, name.encode())
            try:
                self._lib.check(f"ti_get_aot_module_kernel({name})")
            except TaichiCApiError:
                continue
            if handle:
                self._kernels[name] = ctypes.c_void_p(handle)
                return self._kernels[name]
        raise KeyError(f"kernel {name!r} in no loaded module ({list(self._modules)})")

    # -- arguments --

    def _import_memory(self, data_ptr: int, nbytes: int):
        """``ti_import_cpu_memory``, memoized on ``(data_ptr, nbytes)``.

        **The memoization is not an optimization, it is the leak fix.** A
        ``TiMemory`` from ``ti_import_cpu_memory`` cannot be released on an x64
        runtime: ``ti_free_memory`` refuses outright with ``(not supported)
        taichi::arch_is_cpu(config.arch)``, so every import is permanent for the
        life of the runtime. Measured at 90-108 bytes per import, growing
        linearly (1.75 / 4.12 / 5.12 MB at 20k / 40k / 60k launches). §5.5's
        front door as specified -- import each tensor's pointer, launch, wait --
        therefore grows the process on every launch, in a process §5.7 requires
        to outlive a render and which the daemon keeps alive across renders.

        Keying on ``(data_ptr, nbytes)`` is safe rather than merely convenient:
        the handle wraps exactly that pointer and that length, so a later tensor
        landing on the same address with the same size is described correctly by
        the same handle, and a different size takes a different key. What it
        does not do is *bound* the growth -- it only moves it from per-launch to
        per-distinct-buffer. Torch's caching allocator reuses addresses heavily,
        so in practice that is a small fixed set, but a production version of
        this shim owes that claim a measurement on a real render.
        """
        key = (data_ptr, nbytes)
        self.imports += 1
        memory = self._imported.get(key)
        if memory is not None:
            self.import_hits += 1
            return memory
        memory = self._lib.dll.ti_import_cpu_memory(
            self.handle, ctypes.c_void_p(data_ptr), ctypes.c_size_t(nbytes)
        )
        self._lib.check("ti_import_cpu_memory")
        if not memory:
            raise TaichiCApiError("ti_import_cpu_memory returned null")
        self._imported[key] = memory
        return memory

    def ndarray_argument(self, tensor) -> TiArgument:
        """Wrap a **CPU** torch tensor as an ndarray argument, without copying.

        ``ti_import_cpu_memory`` takes the pointer as-is; the caller keeps the
        tensor alive for the launch. Refuses a non-contiguous or non-CPU tensor
        rather than silently reading the wrong bytes.
        """
        if tensor.device.type != "cpu":
            raise ValueError(
                f"ti_import_cpu_memory needs a CPU tensor, got {tensor.device}"
            )
        if not tensor.is_contiguous():
            raise ValueError("ti_import_cpu_memory needs a contiguous tensor")
        if tensor.dim() > TI_MAX_DIM_COUNT:
            raise ValueError(f"ndarray rank {tensor.dim()} exceeds TI_MAX_DIM_COUNT")
        nbytes = tensor.numel() * tensor.element_size()
        memory = self._import_memory(tensor.data_ptr(), nbytes)

        argument = TiArgument()
        argument.type = TI_ARGUMENT_TYPE_NDARRAY
        nd = argument.value.ndarray
        nd.memory = memory
        nd.shape.dim_count = tensor.dim()
        for i, size in enumerate(tensor.shape):
            nd.shape.dims[i] = int(size)
        # A scalar-element ndarray has an empty element shape. Taichi reads
        # dim_count first, so the unset dims are never looked at.
        nd.elem_shape.dim_count = 0
        nd.elem_type = ti_data_type_of(tensor)
        return argument

    @staticmethod
    def i32_argument(value: int) -> TiArgument:
        argument = TiArgument()
        argument.type = TI_ARGUMENT_TYPE_SCALAR
        argument.value.scalar.type = TI_DATA_TYPE_I32
        argument.value.scalar.value.x32 = ctypes.c_uint32(
            ctypes.c_int32(int(value)).value
        ).value
        return argument

    @staticmethod
    def f32_argument(value: float) -> TiArgument:
        argument = TiArgument()
        argument.type = TI_ARGUMENT_TYPE_SCALAR
        argument.value.scalar.type = TI_DATA_TYPE_F32
        argument.value.scalar.value.x32 = ctypes.cast(
            ctypes.pointer(ctypes.c_float(float(value))),
            ctypes.POINTER(ctypes.c_uint32),
        ).contents.value
        return argument

    def argument_for(self, value) -> TiArgument:
        """Wrap one Python value as a ``TiArgument`` by its type."""
        if hasattr(value, "data_ptr"):
            return self.ndarray_argument(value)
        if isinstance(value, bool):
            return self.i32_argument(int(value))
        if isinstance(value, int):
            return self.i32_argument(value)
        if isinstance(value, float):
            return self.f32_argument(value)
        raise TypeError(f"no TiArgument for {type(value).__name__}")

    # -- launching --

    def launch(self, name: str, *values, wait: bool = True) -> None:
        """Build the argument array, launch ``name``, and (by default) wait."""
        kernel = self.kernel(name)
        arguments = [self.argument_for(value) for value in values]
        array = (TiArgument * len(arguments))(*arguments)
        self._lib.dll.ti_launch_kernel(
            self.handle, kernel, len(arguments), array if arguments else None
        )
        self._lib.check(f"ti_launch_kernel({name})")
        if wait:
            self.wait()
        # Hold the tensors past the launch: ti_import_cpu_memory borrows their
        # pointers, so a tensor freed here would be read after free.
        del values, arguments, array

    def wait(self) -> None:
        self._lib.dll.ti_wait(self.handle)
        self._lib.check("ti_wait")

    # -- lifetime --

    def destroy(self) -> None:
        """Destroy the runtime. Idempotent; safe at interpreter shutdown."""
        if getattr(self, "handle", None) is None:
            return
        for module in self._modules.values():
            self._lib.dll.ti_destroy_aot_module(module)
        self._modules.clear()
        self._kernels.clear()
        # Nothing to release here: ti_free_memory refuses on an x64 runtime
        # (see _import_memory). Destroying the runtime is what reclaims them.
        self._imported.clear()
        self._lib.dll.ti_destroy_runtime(self.handle)
        self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.destroy()
        return False
