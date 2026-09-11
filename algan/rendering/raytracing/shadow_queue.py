"""Shared shadow transport; event generation and sampling policies stay separate."""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class ShadowTraceContext:
    """Scene/light bindings shared by primary and deferred shadow submissions.

    This is a host record, never a kernel argument. BVHs are supplied at each
    submission rather than captured here: a deferred build can replace the
    batch's placeholder trees. Source identity, event order, sample footprints,
    and the presence-specialization flags remain explicit caller policies.
    """

    scene: dict
    light_position: torch.Tensor
    light_color: torch.Tensor
    num_lights: int
    pixel_world_scale: torch.Tensor
    layer_offset_triangles: float

    def trace(
        self,
        payload,
        triangle_bvh,
        bezier_bvh,
        visibility,
        *,
        samples,
        shadow_mode,
        has_triangles,
        has_beziers,
        source_primitives,
        identity_enabled,
        self_epsilon,
        near_epsilon,
        terminator_mode,
        adaptive_taps,
    ):
        """Submit one compact event batch through the explicit packed-kernel ABI.

        ``source_primitives`` is the caller's one-word dummy when identity is
        disabled. Unused footprint/terminator fields likewise keep their
        caller-owned placeholders. The caller owns every event/output buffer
        and keeps them alive through this launch.
        """
        from algan.rendering.raytracing.raster_taichi import raster_shadow_trace
        from algan.rendering.raytracing.refit_bvh import RefitBVH

        scene = self.scene
        raster_shadow_trace(
            int(payload.frame.numel()),
            payload.position,
            payload.smooth_normal,
            payload.face_normal,
            payload.frame,
            payload.mask,
            triangle_bvh.blocks,
            triangle_bvh.node_miss,
            triangle_bvh.leaf_prim,
            triangle_bvh.leaf_tspan,
            int(triangle_bvh.first_leaf),
            scene["tri_pos"],
            scene["tri_colors"],
            scene["tri_uvs"],
            scene["tri_tex_meta"],
            scene["textures"],
            scene["tri_extra"],
            int(scene["num_colored_triangles"]),
            bezier_bvh.blocks,
            bezier_bvh.node_miss,
            bezier_bvh.leaf_prim,
            bezier_bvh.leaf_tspan,
            int(bezier_bvh.first_leaf),
            scene["circuit_meta"],
            scene["circuit_colors"],
            scene["circuit_border_colors"],
            scene["edges_2d"],
            scene["edge_accel"],
            self.light_position,
            self.light_color,
            int(self.num_lights),
            self.pixel_world_scale,
            float(self.layer_offset_triangles),
            int(isinstance(triangle_bvh, RefitBVH)),
            int(has_triangles),
            int(has_beziers),
            payload.footprint,
            payload.terminator,
            int(samples),
            visibility,
            int(shadow_mode),
            scene["tri_obj"] if identity_enabled else source_primitives,
            source_primitives,
            float(self_epsilon),
            float(near_epsilon),
            int(identity_enabled),
            int(terminator_mode),
            int(adaptive_taps),
        )
