"""Shared shadow payload copies; event generation and tracing policy stay separate."""

from __future__ import annotations

from typing import NamedTuple

import torch

from algan.rendering.mps_compat import kernel_index


class _ShadowPayload(NamedTuple):
    position: torch.Tensor
    smooth_normal: torch.Tensor
    face_normal: torch.Tensor
    frame: torch.Tensor
    mask: torch.Tensor
    footprint: torch.Tensor
    terminator: torch.Tensor


def _gather_shadow_payload(
    memory,
    indices,
    position,
    smooth_normal,
    face_normal,
    frame,
    mask,
    footprint,
    terminator,
    *,
    with_footprint=False,
    with_terminator=False,
):
    """Gather accepted event rows into caller-scoped arena storage.

    The renderer supplies unique, in-range indices on the payload's device.
    Optional fields retain their original placeholder when unused; their
    kernel branches compile out and must not read the placeholder by index.
    The caller must keep its arena scope alive through the shadow trace.
    """
    from algan.rendering.raytracing.shadow_queue_taichi import gather_shadow_payload

    n = int(indices.numel())
    payload = _ShadowPayload(
        memory.get_tensor((n, 3), position.dtype),
        memory.get_tensor((n, 3), smooth_normal.dtype),
        memory.get_tensor((n, 3), face_normal.dtype),
        memory.get_tensor((n,), frame.dtype),
        memory.get_tensor((n,), mask.dtype),
        memory.get_tensor((n, 6), footprint.dtype) if with_footprint else footprint,
        memory.get_tensor((n, 3), terminator.dtype) if with_terminator else terminator,
    )
    if n:
        gather_shadow_payload(
            kernel_index(indices),
            n,
            position,
            smooth_normal,
            face_normal,
            frame,
            mask,
            footprint,
            terminator,
            bool(with_footprint),
            bool(with_terminator),
            *payload,
        )
    return payload


def _scatter_shadow_visibility(destination, indices, source):
    """Copy RGB event visibility directly, preserving all-lit padding slots."""
    from algan.rendering.raytracing.shadow_queue_taichi import scatter_shadow_visibility

    n = int(indices.numel())
    if (
        destination.ndim != 2
        or source.ndim != 3
        or source.shape[2] != 3
        or source.shape[0] != n
        or source.shape[1] * 3 > destination.shape[1]
    ):
        raise ValueError("shadow visibility shape does not fit its padded destination")
    if n:
        scatter_shadow_visibility(
            kernel_index(indices), source, destination, n, source.shape[1]
        )
    return destination
