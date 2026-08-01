from __future__ import annotations

import torch

from algan.constants.color import BLUE
from algan.rendering.primitives.primitive import RenderPrimitive
from algan.utils.tensor_utils import broadcast_all


def batch_arange(lengths, memory=None):
    if memory is None:
        offsets = lengths.cumsum(0)
        n = offsets[-1].clone()
        offsets -= lengths
        offsets = torch.repeat_interleave(offsets, lengths, output_size=n)
        return torch.arange(n, device=lengths.device) - offsets

    start_pointer = memory.current_pointer
    start_reverse_pointer = memory.current_reverse_pointer
    offsets = torch.cumsum(
        lengths, 0, out=memory.get_tensor(lengths.shape, lengths.dtype)
    )
    n = offsets[-1].long()
    offsets -= lengths
    offsets = torch.repeat_interleave(offsets, lengths, output_size=n)
    inds = torch.arange(
        n,
        device=lengths.device,
        out=memory.get_tensor((n,), dtype=torch.long, persist=True),
    )
    inds -= offsets
    memory.current_pointer = start_pointer
    inds = memory.cast(inds, torch.int)
    memory.current_reverse_pointer = start_reverse_pointer
    return inds


class BezierCircuitPrimitive(RenderPrimitive):
    def __init__(
        self,
        corners=None,
        next_segment_inds=None,
        num_segments_per_circuit=None,
        colors=BLUE,
        opacity=1,
        normals=None,
        border_width=None,
        border_color=None,
        mob_center=None,
        grid_width=None,
        grid_height=None,
        first_basis=None,
        second_basis=None,
        triangle_collection=None,
        glow=0,
        num_texture_points=0,
        filled=True,
        num_pixels_per_sample=0.5,
    ):
        # Legacy name retained for compatibility.  The ray tracer uses this as
        # the maximum screen-space curve-to-chord error in pixels.
        self.num_pixels_per_sample = num_pixels_per_sample
        self.num_bezier_parameters = 4
        self.num_texture_points = num_texture_points
        self.filled = filled
        if triangle_collection is not None:
            # Group on the already-materialized source device.  Uploading the
            # packed collection belongs to the render-memory boundary, not to
            # the CPU prefetch worker.
            device = triangle_collection[0].corners.device
            self.num_segments_per_object = torch.cat(
                [_.num_segments_per_circuit.view(-1) for _ in triangle_collection]
            ).to(device)

            self.num_texture_points = triangle_collection[0].num_texture_points
            self.filled = triangle_collection[0].filled
            self.corners = torch.cat([_.corners for _ in triangle_collection], -3).to(
                device
            )
            self.colors = torch.cat([_.colors for _ in triangle_collection], -3).to(
                device
            )
            if self.num_texture_points == 0:
                self.colors = self.colors.squeeze(-2)
            self.next_segment_inds = torch.cat(
                [_.next_segment_inds for _ in triangle_collection], -3
            ).to(device)
            self.next_segment_inds = self.next_segment_inds + torch.arange(
                self.next_segment_inds.shape[-3], device=self.next_segment_inds.device
            ).view(-1, 1, 1)

            (
                self.normals,
                self.border_width,
                self.border_color,
            ) = (
                (torch.cat(list(_), -2)).to(device)
                for _ in zip(
                    *(
                        (
                            triangle.normals,
                            triangle.border_width,
                            triangle.border_color,
                        )
                        for triangle in triangle_collection
                    )
                )
            )

            (
                self.mob_center,
                self.grid_width,
                self.grid_height,
                self.basis1,
                self.basis2,
            ) = (
                (torch.cat(list(_), 1)).to(device)
                for _ in zip(
                    *(
                        broadcast_all(
                            (
                                triangle.mob_center,
                                triangle.grid_height.int(),
                                triangle.grid_width.int(),
                                triangle.basis1,
                                triangle.basis2,
                            ),
                            [-2, -1],
                        )
                        for triangle in triangle_collection
                    )
                )
            )
            if self.num_texture_points > 0:
                self.colors = self.colors[..., (-self.num_texture_points) :, :]
            return
        self.corners = corners
        self.next_segment_inds = next_segment_inds
        self.num_segments_per_circuit = num_segments_per_circuit
        border_color, opacity, glow = broadcast_all(
            [border_color, opacity, glow], ignored_dims=[-1]
        )
        self.colors = colors.clone()
        self.colors[..., -2:-1] += glow.unsqueeze(-2)
        self.colors[..., -1:] *= opacity.unsqueeze(-2)
        self.normals = normals
        self.border_width, self.border_color, self.glow = (
            border_width,
            border_color,
            glow,
        )
        self.border_color[..., -2:-1] += glow
        self.border_color[..., -1:] *= opacity
        self.mob_center = mob_center
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.basis1 = first_basis
        self.basis2 = second_basis

    @staticmethod
    def batch_identifier_for(num_texture_points, filled):
        return f"{BezierCircuitPrimitive}_{num_texture_points}_{filled}"

    def get_batch_identifier(self):
        return BezierCircuitPrimitive.batch_identifier_for(
            self.num_texture_points, self.filled
        )
