"""Caller-owned sheet reduction outputs, distinct from stage-local scratch."""

from __future__ import annotations

from typing import NamedTuple

import torch


class BandReduction(NamedTuple):
    area: torch.Tensor
    union: torch.Tensor
    fused: torch.Tensor | None
    sliver: torch.Tensor | None

    @classmethod
    def allocate(cls, workspace, count, *, want_fused, want_sliver):
        """Allocate in the caller's stage, before opening reduction scratch."""
        return cls(
            workspace.tensor((count,), torch.float32),
            workspace.tensor((count,), torch.int32),
            workspace.tensor((count,), torch.bool) if want_fused else None,
            workspace.tensor((count,), torch.int32) if want_sliver else None,
        )


class SheetWeights(NamedTuple):
    coverage: torch.Tensor
    mask: torch.Tensor

    @classmethod
    def allocate(cls, workspace, count):
        return cls(
            workspace.tensor((count,), torch.float32),
            workspace.tensor((count,), torch.int32),
        )


class BandComposite(NamedTuple):
    area: torch.Tensor
    union: torch.Tensor
    correction: torch.Tensor
    split: torch.Tensor

    @classmethod
    def allocate(cls, workspace, count):
        return cls(
            workspace.tensor((count,), torch.float32),
            workspace.tensor((count,), torch.int32),
            workspace.tensor((count,), torch.float32),
            workspace.tensor((count,), torch.bool),
        )
