"""The arena-offset calling convention (``DESIGN_metal_native_port.md`` §1.2).

Metal binds at most 31 buffers to a compute stage, measured — a 32nd is a
compile error, ``'buffer' attribute parameter is out of bounds: must be between
0 and 30``. Algan's shading kernels ask for far more than that:
``sheet_resolve_shade`` takes 49 ndarray arguments and ``wavefront_shade`` 40.
That gap is what `DESIGN_mps_support.md` §1.1 called "the blocker".

It is not one, because ``ManualMemory`` already did the packing this needs.
The arena is a single ``torch.empty(n, dtype=torch.uint8)`` and every render
tensor it hands out is a slice view of it, so those 49 arguments are one
allocation at 49 offsets. Bind the arena once, pass the offsets alongside, and
a kernel reconstitutes its arrays inside the shader.

**This module is the planning half of that.** It works out which arguments are
arena-backed, at what offset, and whether what is left still fits in 31 slots --
the same problem for both backends, answerable and regression-testable on CPU
and CUDA today.

The half that *runs* is `algan/rendering/raytracing/arena_args_taichi.py`. Every
kernel that was over Taichi's 24-buffer Metal budget has since been converted to
take its cold arrays through the arena, so the table below is now history rather
than a gap: no kernel asks for more than 20 ndarray arguments
(`tests/unit_tests/test_arena_args.py`).

What a live render said before the conversion, and what those kernels still hand
their launch wrapper (``test_arena_binding_live.py`` keeps it true):

| kernel | ndarray args | arena-backed | bindings after packing |
| --- | --- | --- | --- |
| ``sheet_resolve_shade`` | 49 | 48 | 3 |
| ``wavefront_shade`` | 40 | 40 | 2 |
| ``wavefront_traverse_events`` | 34 | 34 | 2 |
| ``raster_shadow_trace`` | 32 | 24 | 10 |

``raster_shadow_trace``'s eight non-arena arguments are its ``event_*`` tables,
which are allocated outside the arena on every path measured; they are why it
keeps them as ordinary parameters rather than binding them.

Nothing here runs during a render. It is a planning library the tests drive;
production launch paths use `arena_args_taichi` instead, so this module costs a
shipped render nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

#: Buffer slots Metal gives one compute stage, indices 0..30. Measured by the
#: MPS capability probe's ``args_*`` ladder rather than taken from the spec:
#: 30 binds, 31 is a compile error naming the bound. Taichi managed only 24 on
#: the same machine because it spends slots on its own context and root buffers.
METAL_BUFFER_SLOTS = 31

#: An offset table is emitted as int32 while it can be. Beyond this the arena
#: has grown past what a 32-bit byte offset addresses and the table has to be
#: int64 -- worth knowing about rather than wrapping silently.
_INT32_LIMIT = 2**31


class ArenaBindingError(ValueError):
    """An argument cannot be addressed as an offset into the arena."""


@dataclass(frozen=True)
class ArenaSlot:
    """Where one kernel argument sits inside the arena.

    ``byte_offset`` is what a Metal kernel wants: it binds the arena as
    ``device uchar*`` and reinterprets at the offset. ``elem_offset`` is what a
    typed view wants -- bind ``arena.view(dtype)`` and index in elements -- and
    is exact only because ``byte_offset`` is a whole number of elements, which
    :func:`arena_slot` refuses to return otherwise.
    """

    index: int
    byte_offset: int
    elem_offset: int
    dtype: torch.dtype
    shape: tuple[int, ...]
    numel: int
    itemsize: int

    @property
    def nbytes(self) -> int:
        return self.numel * self.itemsize


def arena_storage_ptr(arena) -> int:
    """The storage address of an arena, given the arena or its backing tensor.

    Accepts a ``ManualMemory`` or the ``uint8`` tensor inside it, because the
    callers that have one rarely have the other.
    """
    data = getattr(arena, "data", arena)
    if data is None:
        # ``render_loop`` drops a chunk's arena the moment it is done with it
        # (``render_memory.data = None``), so an arena outlives its buffer.
        # Anything inspecting arenas across a render has to expect this and
        # say so plainly rather than report a type confusion.
        raise ArenaBindingError(
            "this arena has been released (its data is None); an arena can "
            "only be addressed while its buffer is alive"
        )
    if not isinstance(data, torch.Tensor):
        raise TypeError(f"expected a ManualMemory or a tensor, got {type(arena)!r}")
    return data.untyped_storage().data_ptr()


def arena_slot(tensor, arena, *, index: int = -1) -> ArenaSlot | None:
    """Describe ``tensor`` as an offset into ``arena``, or ``None`` if it is not.

    Returning ``None`` is the ordinary case for a texture or a persistent scene
    table, which keeps its own binding; it is not an error. An error is raised
    only for a tensor that *is* in the arena but cannot be addressed by offset,
    because that is a latent wrong-pixels bug on the Metal path rather than a
    kernel that needs one more slot.
    """
    if not isinstance(tensor, torch.Tensor):
        return None
    storage = tensor.untyped_storage()
    if storage.data_ptr() != arena_storage_ptr(arena):
        return None

    byte_offset = tensor.data_ptr() - storage.data_ptr()
    itemsize = tensor.element_size()
    name = f"argument {index}" if index >= 0 else "tensor"
    if not tensor.is_contiguous():
        # A shader reconstitutes an array from a base pointer and a length; it
        # has no stride vector, so a non-contiguous view would be read as if it
        # were dense and silently return the wrong elements.
        raise ArenaBindingError(
            f"{name} is a non-contiguous arena view "
            f"(shape {tuple(tensor.shape)}, strides {tuple(tensor.stride())}); "
            "the offset convention can only address dense spans"
        )
    if byte_offset % itemsize:
        # ``(device const float*)(arena + off)`` is undefined for an ``off``
        # that is not a multiple of 4. The arena aligns each allocation to its
        # element size, so this should be unreachable -- which is why it is
        # worth saying loudly if it ever is.
        raise ArenaBindingError(
            f"{name} starts at byte {byte_offset}, which is not a multiple of "
            f"its {itemsize}-byte element; the arena's alignment guarantee has "
            "been broken and a reinterpreting shader would read garbage"
        )
    return ArenaSlot(
        index=index,
        byte_offset=byte_offset,
        elem_offset=byte_offset // itemsize,
        dtype=tensor.dtype,
        shape=tuple(tensor.shape),
        numel=int(tensor.numel()),
        itemsize=itemsize,
    )


@dataclass(frozen=True)
class BindingPlan:
    """How one kernel's arguments would bind under the convention."""

    slots: tuple[ArenaSlot, ...]
    passthrough: tuple[int, ...]
    non_tensor: tuple[int, ...]
    limit: int

    @property
    def bindings(self) -> int:
        """Buffer slots the packed form needs.

        The arena itself and the offset table are one binding each, and are
        needed only if something is actually packed into them. Everything not
        arena-backed keeps its own slot. Scalars are not buffers -- Metal takes
        them by value through ``setBytes`` -- so they do not count.
        """
        packed = 2 if self.slots else 0
        return packed + len(self.passthrough)

    @property
    def fits(self) -> bool:
        return self.bindings <= self.limit

    def describe(self) -> str:
        return (
            f"{len(self.slots)} arena + {len(self.passthrough)} passthrough "
            f"-> {self.bindings} bindings (limit {self.limit})"
        )


def plan_bindings(args, arena, *, limit: int = METAL_BUFFER_SLOTS) -> BindingPlan:
    """Work out how ``args`` would bind, without binding anything.

    ``args`` is a kernel's positional argument list as the launch site passes
    it, scalars included, so a caller can hand this ``locals()``-style tuples
    straight through.
    """
    slots: list[ArenaSlot] = []
    passthrough: list[int] = []
    non_tensor: list[int] = []
    for i, value in enumerate(args):
        if not isinstance(value, torch.Tensor):
            non_tensor.append(i)
            continue
        slot = arena_slot(value, arena, index=i)
        if slot is None:
            passthrough.append(i)
        else:
            slots.append(slot)
    return BindingPlan(
        slots=tuple(slots),
        passthrough=tuple(passthrough),
        non_tensor=tuple(non_tensor),
        limit=limit,
    )


def offset_table(plan: BindingPlan, *, device=None, elements: bool = False):
    """The offsets a packed kernel takes beside the arena, in argument order.

    Byte offsets by default, which is the form a shader reinterpreting a
    ``device uchar*`` needs. ``elements=True`` gives the typed-view form, for a
    backend that binds ``arena.view(dtype)`` and indexes in elements.
    """
    values = [s.elem_offset if elements else s.byte_offset for s in plan.slots]
    widest = max(values, default=0)
    if widest >= _INT32_LIMIT:
        raise ArenaBindingError(
            f"offset {widest} does not fit in int32; the arena has grown past "
            "4 GB and the offset table needs to be int64 on both sides"
        )
    return torch.tensor(values, dtype=torch.int32, device=device)


def unpack(arena, slot: ArenaSlot):
    """Rebuild a slot's tensor from the arena and the offset alone.

    The host-side inverse of what a shader does with a base pointer and an
    offset, and the thing a round-trip test can compare against the original.
    """
    arena_storage_ptr(arena)  # rejects a released arena with a useful message
    data = getattr(arena, "data", arena)
    end = slot.byte_offset + slot.nbytes
    if end > data.numel():
        raise ArenaBindingError(
            f"slot ends at byte {end}, past the arena's {data.numel()} bytes"
        )
    span = data[slot.byte_offset : end]
    return span.view(slot.dtype).view(slot.shape)
