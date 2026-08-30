"""The arena-offset calling convention, kernel side and launch side.

Metal binds at most 31 buffers to a compute stage and Taichi manages 24 of them
(`arena_binding.py`), while this renderer's widest kernels ask for up to 49
ndarray arguments. The gap closes because ``ManualMemory`` hands out views of
**one** allocation: those arrays are one buffer at N offsets, so a kernel can
take the buffer once and reconstitute each array from an offset and a shape.

This module is the half that runs. `arena_binding` plans and measures; this
binds.

What a converted kernel looks like
----------------------------------

The kernel keeps its hot per-ray state as ordinary ndarray parameters and takes
everything else through the arena::

    @ti.kernel
    def sheet_resolve_shade_arena(
            num_covered: int, ...,
            rs_ro: ti.types.ndarray(), ...,          # per-slot state, kept
            arena_f32: ti.types.ndarray(),
            arena_i32: ti.types.ndarray(),
            aoff: ti.types.ndarray(), ashp: ti.types.ndarray()):
        sheet_offsets = ti.static(ArenaView(arena_i32, aoff[0], (ashp[0],)))
        sheet_ab = ti.static(ArenaView(arena_f32, aoff[1], (ashp[1], ashp[2])))
        ...                                          # body unchanged

Every ``ti.func`` these names reach already takes them as ``ti.template()``, so
a view passes through wherever an ndarray did and no callee is touched.

Why the hot arrays stay parameters
----------------------------------

Measured on CUDA (GTX 1050, 155k covered pixels, warm alternating A/B,
`benchmarks/_arena_view_real_kernel_ab.py`): binding **all** of
``sheet_resolve_shade``'s arrays costs +18%; keeping only the seven slot-indexed
ray-state arrays as parameters costs **+1.7..3.0%**; keeping thirteen more on
top of those bought nothing further (+2.2%). So the split is exactly "arrays
indexed by the per-thread ray slot stay parameters, scene-indexed tables go
through the arena", and there is no reason to keep more than that.

The cost is not the convention. Taichi loads an ndarray's base pointer and
shapes from a global-memory argument buffer at **every use site inside the
loop** -- LICM cannot hoist them, there is no `!invariant.load` on them -- so
the arena adds a third level to what was already a two-level dependent-load
chain. `DESIGN_taichi_argument_loads.md` has the PTX, the register counts and
the fork that would remove it.

Launch side
-----------

`arena_packed` wraps a converted kernel so **callers pass the original argument
list unchanged**. It splits the arguments, checks that every arena-bound one
really is a slice of one allocation per dtype, and appends the buffers and
tables. No launch site in the renderer changed when the kernels were converted.

Each module binds the launcher to a private name and gives the public name an
ordinary ``def`` that delegates to it::

    _raster_shadow_trace_launch = arena_packed(
        __name__, "raster_shadow_trace_arena",
        _RASTER_SHADOW_TRACE_PARAMS, _RASTER_SHADOW_TRACE_ARENA)


    def raster_shadow_trace(*args):
        return _raster_shadow_trace_launch(*args)

Not decoration: a module-level assignment binding a lowercase name to a
callable is what `raytracing_settings._shadowed_fields` looks for when it hunts
for a settings field a helper has taken the name of, and three of these kernels
live in modules that store settings.
"""

import sys

import torch
from taichi.lang import impl as _ti_impl

#: dtype -> the tag used in an arena parameter's name (``arena_f32``, ...).
#: Ordered: a kernel's arena parameters appear in this order, so both sides
#: agree without either passing the order around.
DTYPE_TAGS = (
    (torch.float32, "f32"),
    (torch.int32, "i32"),
    (torch.int64, "i64"),
    (torch.float16, "f16"),
    (torch.uint8, "u8"),
)
_TAG_OF = dict(DTYPE_TAGS)
_TAG_ORDER = [t for _dt, t in DTYPE_TAGS]

#: Element offsets and shapes travel as int32, so an arena addressed in f32
#: elements tops out at 8 GB. Well past any arena we allocate, but a silent
#: wrap here would be wrong pixels rather than a crash.
_INT32_LIMIT = 2**31


class ArenaBindingError(ValueError):
    """An argument cannot be bound through the arena."""


class ArenaView(tuple):
    """A window into a flat arena, indexed exactly like the array it replaces.

    Subclasses ``tuple`` so ``ti.static`` accepts it and passes it through --
    that is what lets a view be bound to a local NAME in kernel scope (Taichi's
    assignment builder otherwise tries to create a Taichi local of type
    ``ArenaView`` and fails). The base and the shape entries are Taichi Exprs
    read from runtime tables, so no scene's geometry sizes are baked into the
    compiled kernel and a new frame does not recompile it.

    ``__getitem__`` returns the arena's own IndexExpression, which is an lvalue,
    so stores and atomics through a view work exactly like stores through the
    array.
    """

    __slots__ = ()

    def __new__(cls, buf, base, shape):
        return super().__new__(cls, (buf, base, tuple(shape)))

    @property
    def buf(self):
        return tuple.__getitem__(self, 0)

    @property
    def base(self):
        return tuple.__getitem__(self, 1)

    @property
    def shape(self):
        return tuple.__getitem__(self, 2)

    def __getitem__(self, idx):
        if not isinstance(idx, tuple):
            idx = (idx,)
        shape = self.shape
        flat = idx[0]
        for d in range(1, len(idx)):
            flat = flat * shape[d] + idx[d]
        # Python scope here, so the subscript is built through Taichi's own
        # builder rather than through AnyArray.__getitem__ (which does not
        # exist).
        return _ti_impl.subscript(None, self.buf, self.base + flat)


def _whole_storage(sample):
    """A dense view of ``sample``'s entire allocation, in ``sample``'s dtype.

    This is the buffer a converted kernel receives. It is built per launch
    rather than cached: a cache keyed by storage pointer would hold a reference
    to the tensor and so keep a released chunk's whole arena alive
    (``render_loop`` drops it deliberately, `render_loop.py`).
    """
    storage = sample.untyped_storage()
    view = torch.empty(0, dtype=sample.dtype, device=sample.device)
    view.set_(storage, 0, (storage.nbytes() // sample.element_size(),))
    return view


def pack(spec, tensors):
    """Bind ``tensors`` as offsets into one buffer per dtype.

    ``spec`` is the kernel's ``(name, tag, ndim)`` tuple in offset-table order;
    ``tensors`` are the matching tensors in the same order. Returns the
    positional tail a converted kernel takes: one arena per dtype present in
    ``spec`` (in `DTYPE_TAGS` order), then the offset table, then the shape
    table.

    The arrays do not have to come from a ``ManualMemory`` -- they have to share
    one allocation per dtype, which is the condition the kernel actually needs
    and which the arena is merely the reason for. Anything else raises, naming
    the parameter, because the alternative is a kernel reading a base pointer
    that has nothing to do with the array it was handed.
    """
    if len(spec) != len(tensors):
        raise ArenaBindingError(
            f"this kernel binds {len(spec)} arrays through the arena but was "
            f"handed {len(tensors)}"
        )

    samples = {}
    storages = {}
    offsets = []
    shapes = []
    for (name, tag, ndim), t in zip(spec, tensors):
        if not isinstance(t, torch.Tensor):
            raise ArenaBindingError(f"{name} is {type(t).__name__}, not a tensor")
        if _TAG_OF.get(t.dtype) != tag:
            raise ArenaBindingError(
                f"{name} is {t.dtype}, but the kernel binds it in the "
                f"{tag} arena"
            )
        if t.dim() != ndim:
            raise ArenaBindingError(
                f"{name} has {t.dim()} dimensions, but the kernel's binding "
                f"reads {ndim} of them -- the argument order and the kernel's "
                "prologue have drifted apart"
            )
        if not t.is_contiguous():
            raise ArenaBindingError(
                f"{name} is not contiguous (shape {tuple(t.shape)}, strides "
                f"{tuple(t.stride())}); the offset convention addresses dense "
                "spans only"
            )
        storage = t.untyped_storage()
        ptr = storage.data_ptr()
        known = storages.setdefault(tag, ptr)
        if known != ptr:
            raise ArenaBindingError(
                f"{name} is not in the same allocation as the other {tag} "
                f"arena arguments of this kernel. Every array bound through "
                "the arena has to be a view of one buffer; this one was "
                "allocated somewhere else and has to stay an ordinary ndarray "
                "parameter."
            )
        samples.setdefault(tag, t)
        byte_offset = t.data_ptr() - ptr
        itemsize = t.element_size()
        if byte_offset % itemsize:
            raise ArenaBindingError(
                f"{name} starts at byte {byte_offset}, not a multiple of its "
                f"{itemsize}-byte element"
            )
        elem_offset = byte_offset // itemsize
        if elem_offset >= _INT32_LIMIT:
            raise ArenaBindingError(
                f"{name} sits at element {elem_offset}, past what the int32 "
                "offset table can address"
            )
        offsets.append(elem_offset)
        shapes.extend(int(d) for d in t.shape)

    tags = [t for t in _TAG_ORDER if t in samples]
    arenas = [_whole_storage(samples[t]) for t in tags]
    device = arenas[0].device
    # One allocation and one host-to-device copy for both tables: they are the
    # same dtype and the kernel takes contiguous slices of it.
    table = torch.tensor(offsets + shapes, dtype=torch.int32, device=device)
    return (*arenas, table[: len(offsets)], table[len(offsets) :])


def arena_packed(module_name, kernel_attr, call_params, spec):
    """Wrap a converted kernel so callers keep passing the original arguments.

    ``call_params`` is the parameter list the kernel had **before** conversion,
    which is the argument list every launch site still passes; ``spec`` names
    the ones that now go through the arena, in offset-table order. Everything
    else keeps its original relative position, which is also the order the
    converted kernel declares them in.

    The kernel is looked up by attribute on every call rather than captured, so
    that the profiler's instrumentation -- which replaces module attributes
    (`profiling_utils.discover_taichi_kernels`) -- is seen by this wrapper
    instead of being silently bypassed.
    """
    bound_names = [name for name, _tag, _nd in spec]
    missing = [n for n in bound_names if n not in call_params]
    if missing:
        raise ArenaBindingError(
            f"{kernel_attr} binds {missing}, which its call signature does "
            "not have"
        )
    position = {n: i for i, n in enumerate(call_params)}
    bound_idx = [position[n] for n in bound_names]
    keep_idx = [i for i, n in enumerate(call_params) if n not in set(bound_names)]

    def launch(*args):
        if len(args) != len(call_params):
            raise ArenaBindingError(
                f"{kernel_attr} takes {len(call_params)} arguments, got "
                f"{len(args)}"
            )
        packed = pack(spec, [args[i] for i in bound_idx])
        kernel = getattr(sys.modules[module_name], kernel_attr)
        return kernel(*[args[i] for i in keep_idx], *packed)

    launch.__name__ = kernel_attr
    launch.arena_spec = spec
    launch.call_params = tuple(call_params)
    launch.kernel_attr = kernel_attr
    return launch
