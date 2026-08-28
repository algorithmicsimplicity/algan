"""Batched cubic bezier outlines for the renderer.

:class:`BezierCircuitPrimitive` carries 2-D shapes and glyphs to the renderer as
their control points, not as triangles. Coverage is decided analytically against
the curves, which is what keeps a circle exactly round and text crisp however far
the camera zooms in, and what makes the inside-the-outline border model exact.

Like every render primitive, instances from many Mobs are merged into one batch
before rendering, so a page of text is a single primitive rather than one per
glyph.
"""

from __future__ import annotations

import torch

from algan.constants.color import BLUE
from algan.environment import env_float
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


def _circuit_z_index(primitive):
    """A member's per-circuit ``z_index`` lane, synthesized if it has none.

    Shape ``[1, C, 1]``: one row per circuit, no time axis, because the lane
    selects between coplanar draw orders rather than describing a pose.
    """
    lane = getattr(primitive, "z_index", None)
    if lane is None:
        return torch.zeros_like(primitive.border_width[:1])
    return lane


#: Maximum screen-space curve-to-chord error, in pixels, that a circuit is
#: flattened to. Named because two builders have to agree on it: the per-actor
#: path takes it as this constructor's default, and
#: ``bezier_circuit.build_render_primitives_batched`` -- whose contract is to
#: be a byte-identical replacement for that constructor -- has to set the same
#: value on the mega-primitive it assembles by hand. It once did not, and the
#: default analytic-AA route hid the difference (see ``num_pixels_per_sample``
#: there). It moves rendered output -- a looser tolerance flattens a curve to
#: fewer, longer chords -- and trades edge memory and flatten work against
#: silhouette fidelity, so it takes an environment default rather than being a
#: bare literal. (Under analytic AA the tighter
#: ``rt_settings.analytic_aa_chord_tolerance`` overrides it.)
DEFAULT_CHORD_TOLERANCE_PIXELS = env_float("ALGAN_CHORD_TOLERANCE_PIXELS", 0.5)


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
        num_pixels_per_sample=DEFAULT_CHORD_TOLERANCE_PIXELS,
        z_index=None,
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

            self.normals = torch.cat(
                [triangle.normals for triangle in triangle_collection], -2
            ).to(device)
            self.border_width = torch.cat(
                [triangle.border_width for triangle in triangle_collection], -2
            ).to(device)
            # Per-circuit coplanar draw order. Time-invariant by construction,
            # so members concatenate on the circuit axis with no time
            # unification. ``_has_z_index`` is a plain bool so the renderer can
            # skip the bias without a device->host sync per batch.
            self._has_z_index = any(
                getattr(t, "_has_z_index", False) for t in triangle_collection
            )
            self.z_index = torch.cat(
                [_circuit_z_index(t) for t in triangle_collection], -2
            ).to(device)
            border_colors = [
                triangle.border_color.unsqueeze(-2)
                if triangle.border_color.dim() == 3
                else triangle.border_color
                for triangle in triangle_collection
            ]
            self.border_color = torch.cat(border_colors, -3).to(device)

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
                                triangle.grid_width.int(),
                                triangle.grid_height.int(),
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
                self.border_color = self.border_color[
                    ..., (-self.num_texture_points) :, :
                ]
            return
        self.corners = corners
        self.next_segment_inds = next_segment_inds
        self.num_segments_per_circuit = num_segments_per_circuit
        self.colors = colors.clone()
        self.colors[..., -2:-1] += glow.unsqueeze(-2)
        self.colors[..., -1:] *= opacity.unsqueeze(-2)
        self.normals = normals
        if border_color.dim() == 3:
            border_color = border_color.unsqueeze(-2)
        border_color, border_opacity, border_glow = broadcast_all(
            [border_color, opacity.unsqueeze(-2), glow.unsqueeze(-2)],
            ignored_dims=[-1],
        )
        self.border_width = border_width
        # ``None`` means "every circuit at 0", which is the overwhelmingly
        # common case; keeping it a Python-level distinction is what lets the
        # renderer skip the bias without reading a tensor back off the device.
        self._has_z_index = z_index is not None
        self.z_index = (
            z_index if z_index is not None else torch.zeros_like(border_width[:1])
        )
        self.border_color = border_color.clone()
        self.glow = glow
        self.border_color[..., -2:-1] += border_glow
        self.border_color[..., -1:] *= border_opacity
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
