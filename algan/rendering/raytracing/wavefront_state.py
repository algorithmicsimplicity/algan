"""Named host-side ray state with the existing tuple/kernel ABI."""

from __future__ import annotations

from typing import NamedTuple

import torch


class RayState(NamedTuple):
    """Caller-scoped arrays; the last six may be compatibility placeholders.

    The maintained event-based traversal uses transient hit batches rather
    than attaching hit storage to every continuation slot. Retaining tuple
    behavior lets diagnostic callers and the remaining legacy interfaces keep
    their existing unpacking while maintained code uses names for ownership.
    Integer column 4 is the sparse accumulator row, not disposable padding.
    """

    origin: torch.Tensor
    direction: torch.Tensor
    accumulated: torch.Tensor
    scalars: torch.Tensor
    integers: torch.Tensor
    hit_distance: torch.Tensor
    hit_layer: torch.Tensor
    hit_u: torch.Tensor
    hit_v: torch.Tensor
    hit_primitive: torch.Tensor
    hit_flags: torch.Tensor
