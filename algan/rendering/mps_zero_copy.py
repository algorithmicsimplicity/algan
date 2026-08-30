"""Binding torch MPS tensors to Taichi kernels without the host round trip.

The Algan half of `taichi_patches/0001-metal-zero-copy-ndarray.patch`. Stock
Taichi copies every torch ndarray argument to the host before a Metal launch
and copies it back after (`kernel_impl.py:756-785`), which costs four copies
per read/write argument per launch **and is incorrect** for Algan's arena
calling convention: a converted kernel takes ``arena_f32`` and ``arena_i32``,
two dtype views of one allocation, and the second whole-tensor copy-back
reverts everything the kernel wrote through the first
(``DESIGN_mps_support.md`` §1.3b). The patched build can adopt torch's own
``MTLBuffer`` instead, and an argument that arrives as a ``ti.Ndarray`` takes
``set_arg_ndarray``, which registers no copy-back at all.

So this module does three things:

* **decides whether the patched build is there** (:func:`zero_copy_available`),
  which is what the render device's own availability now turns on -- an
  unpatched Mac renders on the CPU rather than drawing a black frame;
* **converts** a torch MPS tensor into an ndarray over the same buffer
  (:func:`import_tensor`), cached, because the import is per buffer and Algan
  hands the same arena to every kernel;
* **installs** that conversion in front of every launch
  (:func:`install_zero_copy_launch`), the way ``taichi_runtime`` installs its
  arch guard, so no kernel and no call site changes.

Three hazards, all of them the kind that produce a wrong picture rather than an
error, and all handled here rather than left to call sites:

**Lifetime.** Taichi marks an imported allocation ``dont_destroy`` and holds no
reference to whatever owns the buffer. Torch's caching allocator will recycle a
buffer whose last tensor died, so the cache keeps the *storage* alive for as
long as it keeps the ndarray, and :func:`clear_import_cache` is what releases
both -- the render loop drops a chunk's arena deliberately, and an import cache
that outlived it would pin the largest allocation in the process.

**Ordering.** Torch and Taichi hold separate Metal command queues and torch's
heaps are ``MTLHazardTrackingModeUntracked``, so nothing orders a torch write
against a Taichi read of the same buffer. Both syncs are taken per launch here.
That is heavier than necessary -- ``DESIGN_mps_zero_copy.md`` §3.3 wants them
once per frame batch -- and it is where to look first for the next speedup, but
a per-batch fence needs the render loop to declare its batches and a wrong
answer here is invisible.

**Element types.** Taichi type-checks an ndarray argument against the
*annotation's* element type, so an array a kernel declares as
``ndarray(dtype=vector(4, f16))`` -- every BVH-taking kernel's node array --
is refused when it arrives as plain ``f16``. The element shape therefore comes
off the annotation (:func:`_ndarray_positions`) and is passed through to the
import, because nothing about the torch tensor says which of its dimensions
Taichi considers element dimensions.

**Offsets.** ``t.data_ptr()`` on MPS is not an address: torch bit-casts the
storage pointer to ``id<MTLBuffer>`` and keeps the byte offset separately. The
buffer handle is ``t.untyped_storage().data_ptr()`` and the offset is
``t.storage_offset() * t.element_size()``; using ``data_ptr()`` would hand
Taichi an object pointer with an integer added to it.
"""

from __future__ import annotations

import threading

_LOCK = threading.Lock()
#: (buffer handle, dtype, outer shape, element shape, byte offset)
#: -> (ndarray, storage).
#: The storage is held so torch cannot recycle the buffer underneath a kernel.
_IMPORTS: dict = {}
_AVAILABLE = None
_INSTALLED = False

#: Engagement telemetry, read by ``benchmarks/_mps_render_smoke.py`` and by
#: anything else asking whether the fork is actually in the path. This module's
#: whole job is a silent substitution, so a silently DISENGAGED one -- the
#: wrapper installed but converting nothing -- looks exactly like a working
#: render right up until the frame comes out wrong. The same reason
#: ``taichi_fast_launch`` keeps its ``STATS``: a fast path nobody can see is a
#: fast path nobody can prove ran.
#:
#: ``staged`` and ``host`` are the two ways an argument still crosses the bus,
#: and they are counted rather than assumed because the cost of one is
#: invisible: a kernel argument Taichi stages is four copies and an MPS stream
#: sync per launch, and nothing in a correct-looking render says so.
#: ``staged_arguments`` is an MPS tensor this module declined to import;
#: ``host_arguments`` is one that never reached the device at all.
STATS = {
    "converted_launches": 0,
    "passthrough_launches": 0,
    "arguments": 0,
    "staged_arguments": 0,
    "host_arguments": 0,
}

#: ``(kernel, position, why)`` for every argument counted in
#: ``staged_arguments`` or ``host_arguments``, so the count can be acted on. A
#: set, because a kernel launched ten thousand times reports one row.
LEFT_ON_THE_BUS: set = set()


def zero_copy_available() -> bool:
    """Whether the running Taichi can adopt a torch MPS buffer.

    True only on the forked build (``taichi_patches/``): it tests for the two
    halves of that patch, the Python wrapper and the ``Program`` factory the
    wrapper calls, so a wheel that carries one and not the other answers False
    rather than failing at the first launch.

    Answered once. Which Taichi is installed does not change within a process,
    and this is read on the path that selects a render device.
    """
    global _AVAILABLE
    if _AVAILABLE is not None:
        return _AVAILABLE
    _AVAILABLE = False
    try:
        import taichi as ti
        from taichi.lang import impl as _impl

        if not hasattr(ti.lang._ndarray, "ExternalMetalNdarray"):
            return _AVAILABLE
        # The pybind lives on Program, so it can only be inspected through a
        # live one; the class is the reliable half to test before a program
        # exists, and the factory is checked at the first import instead.
        _AVAILABLE = True
        del _impl
    except Exception:
        _AVAILABLE = False
    return _AVAILABLE


def installed() -> bool:
    """Whether the conversion is actually wrapping kernel launches.

    Distinct from :func:`zero_copy_available`, which only says the patched
    build is importable. The two can disagree -- the install is a no-op when
    the patch is absent -- and reporting availability as though it were
    engagement is how a disengaged path stays invisible.
    """
    return _INSTALLED


def unavailable_reason() -> str:
    """Why MPS is not offered as a render device, in a sentence a user can act on.

    Kept beside :func:`zero_copy_available` so the message and the condition
    cannot drift apart.
    """
    return (
        "rendering on MPS needs the patched Taichi build (see taichi_patches/ "
        "and DESIGN_mps_zero_copy.md): stock Taichi copies every kernel "
        "argument through the host, which is not merely slow but wrong for "
        "Algan's arena convention -- two dtype views of one buffer come back "
        "with one of them reverted, and the render draws a black frame. "
        "Install the forked wheel, or render on the CPU, which is the "
        "supported Mac path."
    )


def _taichi_dtype(torch_dtype):
    """The Taichi element type for a torch dtype, or None if it has no ndarray."""
    import taichi as ti
    import torch

    return {
        torch.float32: ti.f32,
        torch.float16: ti.f16,
        torch.int32: ti.i32,
        torch.int64: ti.i64,
        torch.int16: ti.i16,
        torch.int8: ti.i8,
        torch.uint8: ti.u8,
    }.get(torch_dtype)


def import_tensor(tensor, element_shape=()):
    """An ndarray over ``tensor``'s own MTLBuffer, or None if it cannot be one.

    ``element_shape`` is what the *kernel's annotation* asks for, not anything
    read off the tensor: an argument declared
    ``ndarray(dtype=vector(4, f16), ndim=2)`` arrives as a 3-D torch tensor
    whose last dimension is the element, and Taichi type-checks an ndarray
    argument against the annotation's element type, so importing it as plain
    f16 is rejected outright (``required element type: VectorType[4, f16], but
    f16 is provided``). Nothing about the tensor says which of its dimensions
    Taichi considers element dimensions, which is why this is a parameter.

    Returns None -- meaning "leave this argument alone, let Taichi stage it" --
    for anything the import cannot represent: a tensor that is not on MPS, one
    that is not contiguous (Taichi refuses those anyway), one carrying a
    gradient, one whose dtype has no Taichi equivalent, one whose trailing
    dimensions do not match the element the kernel asked for, and one whose
    slice does not start on an element boundary. A None is always safe; it is
    the stock path.
    """
    import torch

    if not isinstance(tensor, torch.Tensor):
        return None
    if tensor.device.type != "mps" or not tensor.is_contiguous():
        return None
    if tensor.requires_grad or tensor.grad is not None:
        return None
    dtype = _taichi_dtype(tensor.dtype)
    if dtype is None:
        return None

    element_shape = tuple(int(d) for d in element_shape)
    element_ndim = len(element_shape)
    if element_ndim:
        if tensor.dim() <= element_ndim:
            return None
        if tuple(int(d) for d in tensor.shape[-element_ndim:]) != element_shape:
            return None
    outer_shape = tuple(int(d) for d in tensor.shape[: tensor.dim() - element_ndim])

    storage = tensor.untyped_storage()
    handle = storage.data_ptr()
    offset = tensor.storage_offset() * tensor.element_size()
    if element_ndim:
        # Metal loads a vector element as one vector load, so a view that
        # begins part-way through an element would be read misaligned rather
        # than refused. Every node array is its own allocation today (offset
        # 0), so this only ever fires if one becomes an arena slice.
        element_bytes = tensor.element_size()
        for extent in element_shape:
            element_bytes *= extent
        if offset % element_bytes:
            return None
    key = (handle, tensor.dtype, outer_shape, element_shape, offset)
    with _LOCK:
        hit = _IMPORTS.get(key)
        if hit is not None:
            return hit[0]

    from taichi.lang._ndarray import ExternalMetalNdarray

    array = ExternalMetalNdarray(
        dtype, list(outer_shape), handle, offset, element_shape=element_shape
    )
    with _LOCK:
        # Another thread may have imported the same buffer while this one was
        # building it. Keep whichever landed first so the cache stays one
        # ndarray per buffer, which is what makes the aliasing pair bind the
        # same allocation twice rather than two of them.
        hit = _IMPORTS.setdefault(key, (array, storage))
    return hit[0]


def clear_import_cache():
    """Drop every imported ndarray and the storages they were holding.

    Call this when a render's arena is released. Nothing here outlives a
    render, and an entry that did would pin the arena -- the largest allocation
    in the process, and one the render loop frees on purpose.
    """
    with _LOCK:
        _IMPORTS.clear()


def _ndarray_positions(kernel):
    """``{position: element shape}`` for every ndarray argument of ``kernel``.

    The element shape comes off the **annotation**, not the tensor, for the
    reason :func:`import_tensor` gives: nothing about a torch tensor says which
    of its dimensions Taichi considers element dimensions. It is ``()`` for a
    scalar-element annotation -- ``ndarray()``, ``ndarray(dtype=ti.f32)`` --
    and the vector's extent for one like ``ndarray(dtype=vector(4, f16))``,
    which is every BVH-taking kernel's node array.

    Those node arrays used to be excluded outright, and the exclusion was a
    limit of this module rather than of the backend: importing them as plain
    f16 made Taichi refuse the launch (``required element type:
    VectorType[4, f16], but f16 is provided``), which took down fourteen tests
    across the path tracer, the denoiser and the fast render, so they were
    left on the staging path. They are now imported as the vector-element
    ndarrays they are declared as -- the C++ side always accepted a tensor
    ``DataType``; it was this side that could not spell one.

    Two annotations are still skipped. A matrix element with an unspecified
    extent has no fixed element size to align a slice against, and a
    ``Layout.SOA`` argument would put the element dimensions *first*, so
    stripping them off the end would take the wrong slice of the shape. The
    second cannot happen in Taichi 1.7.4 -- ``NdarrayType.__init__`` hard-codes
    ``Layout.AOS`` -- and is guarded anyway, because getting it wrong would be
    a wrong picture rather than an error.

    The distinction between scalar and vector elements is the same one
    ``taichi_fast_launch._build_meta`` draws between its ``_EXT`` and ``_EXT_V``
    kinds, and read off the same place.

    Cached per kernel: annotations are fixed once a kernel object exists, and
    this runs on every launch.
    """
    cached = kernel.__dict__.get("_algan_zero_copy_positions")
    if cached is not None:
        return cached
    positions = {}
    try:
        from taichi.lang import kernel_impl as _ki
        from taichi.lang.enums import Layout
        from taichi.lang.matrix import MatrixType

        ndarray_annotation = _ki.ndarray_type.NdarrayType
        scalar_type_ids = _ki.primitive_types.type_ids
        for index, argument in enumerate(kernel.arguments):
            annotation = argument.annotation
            if not isinstance(annotation, ndarray_annotation):
                continue
            if getattr(annotation, "layout", None) == Layout.SOA:
                continue
            dtype = annotation.dtype
            if dtype is None or id(dtype) in scalar_type_ids:
                positions[index] = ()
            elif isinstance(dtype, MatrixType):
                shape = dtype.get_shape()
                if not shape or None in shape:
                    continue
                positions[index] = tuple(shape)
    except Exception:
        # An annotation shape this does not understand means "import nothing",
        # which is the stock path and always safe.
        positions = {}
    kernel._algan_zero_copy_positions = positions
    return positions


def _note_left_on_the_bus(kernel, index, tensor):
    """Record an ndarray argument that still crosses the bus, and why.

    Called only for arguments the import declined, which is the whole point:
    the failure mode this module has is a *silent* fallback, where a render
    stays correct and quietly pays four copies and a stream sync per launch for
    an argument nobody knows is on the staging path.
    """
    import torch

    if not isinstance(tensor, torch.Tensor):
        return
    if tensor.device.type == "mps":
        why = "not contiguous" if not tensor.is_contiguous() else str(tensor.dtype)
        STATS["staged_arguments"] += 1
    else:
        why = f"on {tensor.device.type}"
        STATS["host_arguments"] += 1
    LEFT_ON_THE_BUS.add((getattr(kernel, "__name__", repr(kernel)), index, why))


def report():
    """One paragraph on whether zero copy engaged, and what it missed."""
    lines = [
        f"available={zero_copy_available()} installed={installed()}",
        f"converted={STATS['converted_launches']} launches "
        f"({STATS['arguments']} args), "
        f"passthrough={STATS['passthrough_launches']}",
        f"still crossing the bus: {STATS['staged_arguments']} staged MPS "
        f"args, {STATS['host_arguments']} host args",
    ]
    for name, index, why in sorted(LEFT_ON_THE_BUS):
        lines.append(f"  {name} arg {index}: {why}")
    return "\n".join(lines)


def install_zero_copy_launch():
    """Convert torch MPS tensors to imported ndarrays in front of every launch.

    Installed the way ``taichi_runtime.install_render_arch_guard`` is, and for
    the same reason: the conversion has to happen for every kernel and every
    call site, and a wrapper is the only placement a future call site cannot
    forget. It must sit **outside** the fast-launch dispatcher -- that path
    already routes every non-CPU/CUDA tensor to the original launch, so by the
    time it sees these arguments they are ndarrays and it declines them, which
    is correct but only because this ran first.

    Idempotent, and a no-op unless the patched build is present.
    """
    global _INSTALLED
    if _INSTALLED or not zero_copy_available():
        return
    import taichi as ti
    import torch
    from taichi.lang.kernel_impl import Kernel

    previous_call = Kernel.__call__

    def zero_copy_call(self, *args, **kwargs):
        positions = _ndarray_positions(self)
        converted = None
        count = 0
        for index, argument in enumerate(args):
            element_shape = positions.get(index)
            if element_shape is None:
                continue
            array = import_tensor(argument, element_shape)
            if array is None:
                _note_left_on_the_bus(self, index, argument)
                continue
            if converted is None:
                converted = list(args)
            converted[index] = array
            count += 1
        if converted is None:
            STATS["passthrough_launches"] += 1
            return previous_call(self, *args, **kwargs)
        STATS["converted_launches"] += 1
        STATS["arguments"] += count
        # Both fences, per launch. Torch's queue has to have drained before a
        # kernel reads what it wrote, and the kernel has to have finished
        # before torch reads back -- separate command queues over untracked
        # heaps order nothing on their own. See the module docstring for why
        # this is not yet per batch.
        torch.mps.synchronize()
        try:
            return previous_call(self, *tuple(converted), **kwargs)
        finally:
            ti.sync()

    Kernel.__call__ = zero_copy_call
    _INSTALLED = True
