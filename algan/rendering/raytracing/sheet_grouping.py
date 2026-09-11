"""Destination-aware grouping and count-bounded conflict-rank keys."""

from __future__ import annotations

from operator import index
from typing import NamedTuple

import torch

from algan.rendering.mps_compat import mps_friendly
from algan.rendering.raytracing.array_ops import (
    group_ids_from_starts,
    require_tensor_outputs,
)
from algan.rendering.raytracing.sheet_order import stable_lexsort
from algan.rendering.raytracing.sheet_workspace import CompactionWorkspace


class RankGroups(NamedTuple):
    """Per-fragment IDs and the parent/rank of each resulting group."""

    ids: torch.Tensor
    parent: torch.Tensor
    rank: torch.Tensor


class RankPoolGroups(NamedTuple):
    """Compositing-group count and optional per-rank-band destination.

    ``ids is None`` preserves the no-pooling path. A supplied destination is
    unused in that case and its contents are unspecified.
    """

    count: int
    ids: torch.Tensor | None


def rank_key_base(count):
    """Radix for dense parent/rank IDs derived from at most ``count`` rows.

    Every conflict rank is below its parent's fragment count; every parent ID
    is below the stream count. Pooling uses the number of rank bands, which also
    exceeds every represented rank. Thus ``parent * base + rank`` is injective,
    and its maximum is ``count**2 - 1``. The renderer's signed-int32 row capacity
    bounds that key below 2**62. There is no fixed per-surface layer ceiling.

    This checks host capacity, not device values: dense derived IDs are the
    producer's contract. It does not make arbitrary int64 pairs safe to pack.
    """
    count = index(count)
    if not 0 <= count < 2**31:
        raise ValueError("sheet row count exceeds signed-int32 capacity")
    return max(1, count)


def validate_inverse(out, *keys):
    """Check destination metadata and input aliases before grouping writes."""
    if not keys:
        raise ValueError("grouping requires at least one key")
    shape, device = keys[0].shape, keys[0].device
    if len(shape) != 1 or any(
        k.shape != shape
        or k.device != device
        or k.dtype not in (torch.int32, torch.int64)
        for k in keys
    ):
        raise ValueError(
            "grouping keys must be equal-length integer vectors on one device"
        )
    if out is not None:
        require_tensor_outputs(
            (out,), ((shape, torch.int64),), device=device, inputs=keys
        )


def unique_ids(keys, *, consecutive=False, out=None):
    """Retain unique's policy and dynamic keys, optionally owning its inverse.

    PyTorch unique has no destination API. Its temporary inverse is copied and
    released here; this does not claim to remove the library's workspace.
    """
    validate_inverse(out, keys)
    unique = torch.unique_consecutive if consecutive else torch.unique
    values, inverse = unique(keys, return_inverse=True)
    if out is not None:
        out.copy_(inverse)
        inverse = out
    return values, inverse


def consecutive_pair_ids(first, second, *, out=None, workspace=None):
    """Count adjacent pair runs into an exact int64 inverse destination.

    No composite key, integer narrowing or sort is needed. Callers that want
    global groups must supply lexicographically ordered pairs; arbitrary pairs
    instead receive IDs for their consecutive runs. This distinction matters
    for raw conflict ranks, which can decrease within a parent.
    """
    validate_inverse(out, first, second)
    workspace = workspace or CompactionWorkspace(device=first.device)
    if workspace.device != first.device:
        raise ValueError("pair grouping workspace and inputs must share a device")
    n = first.numel()
    if out is None:
        out = torch.empty((n,), dtype=torch.int64, device=first.device)
    if not n:
        return 0, out
    with workspace.stage():
        starts = workspace.tensor((n,), torch.bool, True)
        changed = workspace.tensor((max(0, n - 1),), torch.bool)
        torch.ne(first[1:], first[:-1], out=starts[1:])
        torch.ne(second[1:], second[:-1], out=changed)
        starts[1:].logical_or_(changed)
        group_ids_from_starts(starts, out=out)
        count = int(out[-1]) + 1
    return count, out


def class_groups(band, classes, base, *, out=None, workspace=None):
    """Group the fragments by ``(band, shading class)``, with an exact inverse.

    ``compact_sheets`` subdivides each band by shading class with a composite
    key, ``band * _SHADE_CLASS_BASE + cls``, and a ``unique`` over it
    (``sheets.py`` §4.4). That is the arm below on CPU and CUDA, and it is one
    sort where the other is two, so neither has a reason to pay for the other.

    **Why the MPS arm exists.** The base is ``1 << 25``, so for a 1080p frame
    with 40956 bands the composite reaches **2**40** -- past where MPS int64
    stops being exact, and rows that differ only in their low bits merge.
    Measured, and measured as the *only* thing left: with the split off, the
    same Apple GPU compaction produced 40956 sheets, exactly the CPU's, and
    with it on, 128. So that arm groups the pairs directly, sorting them
    instead of multiplying them together, and never handles a value wider than
    the larger of the two.

    **The group order is the same either way.** ``unique(..., sorted=True)``
    orders by the composite, and because ``base`` exceeds every class the
    composite orders by ``(band, class)`` -- which is what the pair sort
    produces, so the IDs match the wide-key ones exactly and every consumer
    downstream is unaffected.

    Returns what the ``unique`` returned: the group count, the per-fragment
    group ID, and each group's band. The inverse is fixed-size and allocated
    before scratch. Group labels have dynamic size and ordinary ownership;
    callers may copy these smaller labels into their result stage after this
    function has released sorting scratch.
    """
    validate_inverse(out, band, classes)
    device, n = band.device, band.numel()
    if workspace is not None and workspace.device != device:
        raise ValueError("grouping workspace and inputs must share a device")
    workspace = workspace or CompactionWorkspace(device=device)
    if not mps_friendly():
        with workspace.stage():
            key = workspace.tensor((n,), torch.int64)
            # An int64 out buffer does not promote int32 arithmetic in torch.
            key.copy_(band)
            key.mul_(base).add_(classes)
            labels, inverse = unique_ids(key, out=out)
        # labels belongs to unique, not the scratch stage.
        labels.div_(base, rounding_mode="floor")
        return labels.numel(), inverse, labels
    if out is None:
        out = torch.empty((n,), dtype=torch.int64, device=device)
    if n == 0:
        return 0, out, torch.empty((0,), dtype=band.dtype, device=device)
    with workspace.stage():
        # int32 copies, most significant key first. The narrowing costs a pass
        # per key and saves four radix passes per key, and it is safe in the
        # strongest sense (the argument ``sheets._narrow_sort_key`` spells
        # out): an exact int32 copy of an int64 key has the same order and the
        # same indices, so the stable permutation is identical. Both bounds are
        # known without asking the device -- a band ID is an index into a
        # stream that cannot approach 2**31, and a class is documented as
        # ``[0, _SHADE_CLASS_BASE)`` with the base at 2**25.
        order = workspace.tensor((n,), torch.int64)
        stable_lexsort(
            workspace.copy(band, torch.int32),
            workspace.copy(classes, torch.int32),
            out=order,
            workspace=workspace,
        )
        bands = workspace.gather(band, order)
        cls = workspace.gather(classes, order)
        group_sorted = workspace.tensor((n,), torch.int64)
        count, _ = consecutive_pair_ids(
            bands, cls, out=group_sorted, workspace=workspace
        )
        out.scatter_(0, order, group_sorted)
    # Each group's band, written by a scatter rather than gathered through
    # ``band[starts]``. The two agree exactly -- every fragment of a group
    # carries the group's band, so the duplicates a scatter resolves in
    # whatever order all write one value -- and the difference is what the
    # boolean index costs on MPS: it is a ``nonzero`` (a count readback and a
    # compaction pass) followed by a gather, where this is one scatter beside a
    # scalar readback the caller needs anyway. Measured on the Mac runner's UHD
    # profile as 17 gathers over 2.5M fragments at 0.15 s each, the largest
    # single torch op left in the compaction after the fences moved.
    #
    # ``torch.empty``, never ``band.new_empty``: the fragment stream reaches
    # here as a :class:`~algan.constants.color.Color`-subclassed tensor (torch
    # propagates a Tensor subclass through every op that touched one, and the
    # band IDs descend from one that did), and ``Color`` overrides
    # ``new_empty`` to return an opaque black ``[R, G, B, glow, opacity]``
    # row -- five float32 values, whatever size was asked for -- so the
    # destination was neither the right dtype nor the right length. The dtype
    # is what the scatter complained about (``scatter(): Expected self.dtype
    # to be equal to src.dtype``, which is how it took down the MPS arm); the
    # length would have been next. So the allocation names its dtype and
    # device rather than inheriting them through a method a subclass may have
    # redefined.
    labels = torch.empty((count,), dtype=band.dtype, device=device)
    labels.scatter_(0, out, band)
    return count, out, labels
