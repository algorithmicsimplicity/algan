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

**Offsets.** ``t.data_ptr()`` on MPS is not an address: torch bit-casts the
storage pointer to ``id<MTLBuffer>`` and keeps the byte offset separately. The
buffer handle is ``t.untyped_storage().data_ptr()`` and the offset is
``t.storage_offset() * t.element_size()``; using ``data_ptr()`` would hand
Taichi an object pointer with an integer added to it.
"""

from __future__ import annotations

import threading

_LOCK = threading.Lock()
#: (buffer handle, dtype, shape, byte offset) -> (ndarray, storage).
#: The storage is held so torch cannot recycle the buffer underneath a kernel.
_IMPORTS: dict = {}
_AVAILABLE = None
_INSTALLED = False


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


def import_tensor(tensor):
    """An ndarray over ``tensor``'s own MTLBuffer, or None if it cannot be one.

    Returns None -- meaning "leave this argument alone, let Taichi stage it" --
    for anything the import cannot represent: a tensor that is not on MPS, one
    that is not contiguous (Taichi refuses those anyway), one carrying a
    gradient, or one whose dtype has no Taichi equivalent. A None is always
    safe; it is the stock path.
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

    storage = tensor.untyped_storage()
    handle = storage.data_ptr()
    offset = tensor.storage_offset() * tensor.element_size()
    key = (handle, tensor.dtype, tuple(tensor.shape), offset)
    with _LOCK:
        hit = _IMPORTS.get(key)
        if hit is not None:
            return hit[0]

    from taichi.lang._ndarray import ExternalMetalNdarray

    array = ExternalMetalNdarray(dtype, list(tensor.shape), handle, offset)
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
        converted = None
        for index, argument in enumerate(args):
            array = import_tensor(argument)
            if array is None:
                continue
            if converted is None:
                converted = list(args)
            converted[index] = array
        if converted is None:
            return previous_call(self, *args, **kwargs)
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
