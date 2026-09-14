"""CSR for one immutable, pixel-sorted fragment stream."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from algan.rendering.raytracing.array_ops import csr_offsets


@dataclass(frozen=True)
class PixelRunCSR:
    """Share one scan between consumers; replace after filtering or reordering.

    ``covered`` and ``counts`` use ordinary storage. ``offsets`` can use forward
    discovery scratch; the record must not outlive that scope. Payload-only
    changes (coverage, mask bits, material flags) do not change the runs.
    This is not a global tensor cache: the owner explicitly discards the record
    before changing fragment membership/order, even when the row count is equal.
    """

    covered: torch.Tensor
    counts: torch.Tensor
    offsets: torch.Tensor
    num_fragments: int

    @classmethod
    def from_sorted_pixels(cls, pixels, *, memory=None):
        if pixels.ndim != 1 or pixels.dtype not in (torch.int32, torch.int64):
            raise ValueError("pixel runs require a sorted integer vector")
        n = pixels.numel()
        if memory is not None:
            if memory.data.device != pixels.device:
                raise ValueError("pixel runs and their arena must share a device")
            if n >= 2**31:
                raise OverflowError("fragment run offsets must fit int32")
        covered, counts = torch.unique_consecutive(pixels, return_counts=True)
        # The discovery count pass has already checked n before narrowing. A
        # run start/count cannot exceed n. Standalone callers retain int64.
        offsets = csr_offsets(
            counts,
            out=(
                memory.get_tensor((counts.numel() + 1,), torch.int32)
                if memory is not None
                else None
            ),
        )
        return cls(covered, counts, offsets, n)

    @property
    def starts(self):
        return self.offsets[:-1]

    def require_counts(self, counts, num_fragments):
        """Reject accidental reuse with another run set, without device reads."""
        if counts is not self.counts or num_fragments != self.num_fragments:
            raise ValueError("pixel run CSR belongs to a different fragment stream")
        return self.starts
