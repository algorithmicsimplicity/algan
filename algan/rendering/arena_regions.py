"""Typed byte regions and allocation-lifetime proofs for renderer dispatches.

Alias facts are region-scoped, including dtype reinterpretations. Layout metadata
is immutable; the payload and the rest of the arena are not.
"""

from __future__ import annotations

import weakref
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal

import torch

Access = Literal["read", "write", "readwrite"]
_INT32_MAX = (1 << 31) - 1
_TRACKERS: dict[tuple, weakref.ReferenceType] = {}


class ArenaRegionError(ValueError):
    """Invalid layout, alias contract, or arena lifetime."""


def _storage_key(tensor):
    return (tensor.device.type, tensor.device.index, tensor.untyped_storage()._cdata)


@dataclass(frozen=True)
class RegionLayout:
    storage: tuple
    dtype: torch.dtype
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    byte_offset: int
    byte_length: int
    storage_bytes: int
    alignment: int

    @classmethod
    def from_tensor(cls, tensor):
        if not isinstance(tensor, torch.Tensor):
            raise ArenaRegionError("A typed region requires a torch.Tensor")
        if tensor.layout != torch.strided or not tensor.is_contiguous():
            raise ArenaRegionError(
                "Typed arena regions require contiguous strided tensors"
            )
        itemsize = tensor.element_size()
        shape, strides = tuple(tensor.shape), tuple(tensor.stride())
        # Empty tensors have data_ptr()==0; storage_offset remains meaningful.
        offset = tensor.storage_offset() * itemsize
        length = tensor.numel() * itemsize
        storage = tensor.untyped_storage()
        size = storage.nbytes()
        if offset < 0 or offset % itemsize or offset + length > size:
            raise ArenaRegionError("Typed region lies outside storage or is misaligned")
        # Allocation tracking uses host byte addresses, not kernel indices.
        # A small persistent region may sit beyond 2 GiB in a large arena.
        address = storage.data_ptr() + offset
        alignment = min(256, address & -address) if address else itemsize
        return cls(
            _storage_key(tensor),
            tensor.dtype,
            shape,
            strides,
            offset,
            length,
            size,
            alignment,
        )

    def validate_kernel_indexing(self, *, byte_origin=None):
        """Check the int32 ABI relative to the buffer passed to the kernel."""
        if any(d < 0 or d > _INT32_MAX for d in self.shape):
            raise ArenaRegionError("Region shape exceeds the signed int32 kernel ABI")
        if any(s < 0 or s > _INT32_MAX for s in self.strides):
            raise ArenaRegionError("Region stride exceeds the signed int32 kernel ABI")
        # Ordinary ndarray arguments start at the view's data pointer. Packed
        # arguments use the start of the narrowed dtype hull instead.
        origin = self.byte_offset if byte_origin is None else byte_origin
        itemsize = self.dtype.itemsize
        relative = self.byte_offset - origin
        if origin < 0 or relative < 0 or relative % itemsize:
            raise ArenaRegionError("Invalid or misaligned kernel region origin")
        if (relative + self.byte_length) // itemsize > _INT32_MAX:
            raise ArenaRegionError(
                "Rebased region end exceeds the signed int32 kernel ABI"
            )

    @property
    def byte_end(self):
        return self.byte_offset + self.byte_length

    def overlaps(self, other):
        return (
            self.storage == other.storage
            and self.byte_length > 0
            and other.byte_length > 0
            and self.byte_offset < other.byte_end
            and other.byte_offset < self.byte_end
        )


class _Allocation:
    def __init__(self, tracker, layout, persistent):
        self.tracker = weakref.ref(tracker)
        self.layout = layout
        self.persistent = persistent
        self.alive = True
        self.pins = 0

    def validate(self):
        tracker = self.tracker()
        if not self.alive or tracker is None or tracker.owner() is None:
            raise ArenaRegionError(
                "Arena region was reclaimed; bind a fresh allocation"
            )


class ArenaRegionTracker:
    """Owned by ManualMemory; identities survive address reuse, not rewinds.

    Explicit typed handles retain allocation identity. An arbitrary old raw
    Torch slice cannot reveal when it was created and is not such a handle.
    """

    def __init__(self, owner):
        self.owner = weakref.ref(owner)
        self.allocations = []
        key = _storage_key(owner.data)
        _TRACKERS[key] = weakref.ref(self, lambda r: _forget_tracker(key, r))

    def register(self, tensor, persistent):
        allocation = _Allocation(self, RegionLayout.from_tensor(tensor), persistent)
        self.allocations.append(allocation)
        tensor._algan_region_allocation = allocation

    def allocation_for(self, tensor, layout):
        direct = getattr(tensor, "_algan_region_allocation", None)
        if direct is not None:
            direct.validate()
            if not _contains(direct.layout, layout):
                raise ArenaRegionError("Tensor metadata escaped its arena allocation")
            return direct
        for allocation in reversed(self.allocations):
            if allocation.alive and _contains(allocation.layout, layout):
                return allocation
        raise ArenaRegionError("No live arena allocation contains this region")

    def rewind(self, *, forward=None, reverse=None):
        invalid = [
            a
            for a in self.allocations
            if a.alive
            and (
                (
                    not a.persistent
                    and forward is not None
                    and a.layout.byte_end > forward
                )
                or (
                    a.persistent
                    and reverse is not None
                    and a.layout.byte_offset < reverse
                )
            )
        ]
        if any(a.pins for a in invalid):
            raise ArenaRegionError(
                "Cannot rewind an arena region used by an unfinished dispatch"
            )
        for allocation in invalid:
            allocation.alive = False
        self.allocations = [a for a in self.allocations if a.alive]


def _forget_tracker(key, ref):
    if _TRACKERS.get(key) is ref:
        _TRACKERS.pop(key, None)


def _contains(outer, inner):
    return (
        outer.storage == inner.storage
        and outer.byte_offset <= inner.byte_offset
        and inner.byte_end <= outer.byte_end
    )


@dataclass(frozen=True)
class TypedArenaView:
    tensor: torch.Tensor
    layout: RegionLayout
    access: Access
    name: str
    allocation: object = None

    @classmethod
    def bind(
        cls, tensor, *, name, access="read", dtype=None, ndim=None, vector_width=1
    ):
        if access not in ("read", "write", "readwrite"):
            raise ArenaRegionError(f"{name}: invalid access mode {access!r}")
        layout = RegionLayout.from_tensor(tensor)
        layout.validate_kernel_indexing()
        if dtype is not None and tensor.dtype != dtype:
            raise ArenaRegionError(f"{name}: expected {dtype}, got {tensor.dtype}")
        if ndim is not None and tensor.ndim != ndim:
            raise ArenaRegionError(f"{name}: expected rank {ndim}, got {tensor.ndim}")
        if vector_width < 1 or (
            vector_width > 1 and (not layout.shape or layout.shape[-1] != vector_width)
        ):
            raise ArenaRegionError(f"{name}: invalid vector-element layout")
        ref = _TRACKERS.get(layout.storage)
        tracker = ref() if ref is not None else None
        allocation = (
            tracker.allocation_for(tensor, layout) if tracker is not None else None
        )
        return cls(tensor, layout, access, name, allocation)

    def validate(self):
        if RegionLayout.from_tensor(self.tensor) != self.layout:
            raise ArenaRegionError(
                f"{self.name}: tensor metadata changed after binding"
            )
        self.layout.validate_kernel_indexing()
        if self.allocation is not None:
            self.allocation.validate()


@dataclass(frozen=True)
class RegionLoadMetadata:
    """Facts about exact regions, never about the whole mutable arena pointer."""

    name: str
    layout: RegionLayout
    readonly: bool
    disjoint: tuple[str, ...]
    immutable_layout: bool = True


def validate_regions(views):
    if len({v.name for v in views}) != len(views):
        raise ArenaRegionError("Dispatch region names must be unique")
    for view in views:
        view.validate()
    for i, a in enumerate(views):
        for b in views[i + 1 :]:
            if a.layout.overlaps(b.layout) and (
                a.access != "read" or b.access != "read"
            ):
                raise ArenaRegionError(f"Writable alias between {a.name} and {b.name}")
    return tuple(
        RegionLoadMetadata(
            view.name,
            view.layout,
            view.access == "read",
            tuple(
                other.name
                for other in views
                if other is not view and not view.layout.overlaps(other.layout)
            ),
        )
        for view in views
    )


@contextmanager
def lease_regions(views):
    """Prevent rewind until the caller has completed ordered device work."""
    metadata = validate_regions(views)
    allocations = {
        id(v.allocation): v.allocation for v in views if v.allocation is not None
    }
    for allocation in allocations.values():
        allocation.pins += 1
    try:
        yield metadata
    finally:
        for allocation in allocations.values():
            allocation.pins -= 1
