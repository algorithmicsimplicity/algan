"""Internal cubic-PN triangle soup used as the universal morph medium."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from algan.animatable_base.mob import Mob
from algan.rendering.logical_pn import normalize_pixel_tolerance
from algan.rendering.raytracing.primitives import LogicalPNTrianglePrimitive
from algan.utils.tensor_utils import cast_to_tensor


class PNMesh(Mob):
    """A free soup of logical PN patches.

    This is deliberately internal.  Each consecutive group of three rows is a
    patch, and both positions and corner normals are timeline-backed so ordinary
    Mob interpolation can bend one soup into another.
    """

    _morph_family = "pn_soup"

    def __init__(
        self,
        corners,
        normals,
        *,
        render_tolerance=0.5,
        render_tolerance_pixels=None,
        geometry_slack_ratio=0.0,
        shader=None,
        shader_params=None,
        **kwargs,
    ):
        corners = cast_to_tensor(corners)
        normals = cast_to_tensor(normals).to(corners)
        if corners.shape[-1] != 3 or corners.shape[-2] % 3:
            raise ValueError(
                "PNMesh corners must contain complete groups of 3 vertices"
            )
        if normals.shape != corners.shape:
            raise ValueError("PNMesh normals must have the same shape as corners")

        super().__init__(location=corners, **kwargs)
        self.register_attrs_as_animatable(["normals"], PNMesh)
        self._generate_animatable_attr_set_get_methods()
        self._init_default_attr("normals", normals)
        self.num_points_per_object = 3
        self.render_tolerance = float(render_tolerance)
        if not torch.isfinite(torch.tensor(self.render_tolerance)):
            raise ValueError("render_tolerance must be finite")
        if self.render_tolerance <= 0:
            raise ValueError("render_tolerance must be greater than zero")
        # ``None`` here (the default, and what a soup converted from flat
        # geometry keeps) is the absence of an absolute bound, not a loose one:
        # a soup dices by whichever of the two tolerances is finer, so a
        # conversion that carries no pixel tolerance is judged by the
        # fraction-of-screen one alone.
        self.render_tolerance_pixels = normalize_pixel_tolerance(
            render_tolerance_pixels
        )
        # A soup is its own logical surface unless it was converted from
        # something that only approximates one, in which case the conversion
        # passes that surface's accuracy on (see ``convert_to_pn_soup``).
        self.geometry_slack_ratio = float(geometry_slack_ratio)
        self.is_primitive = True

        if shader is not None:
            self.set_shader(shader)
            params = shader_params or {}
            common = {
                name: value
                for name, value in params.items()
                if name in self.animatable_attrs
            }
            if common:
                self.set_non_recursive(**common)

    def get_render_primitives(self):
        colors = self.color.clone()
        colors[..., -1:] *= self.opacity
        colors[..., -2:-1] += self.glow
        normals = F.normalize(self.normals, p=2, dim=-1)
        primitive = LogicalPNTrianglePrimitive(
            corners=self.location,
            colors=colors,
            normals=normals,
            glow=colors[..., -2:-1].as_subclass(torch.Tensor),
            shader=self.shader,
            render_tolerance=self.render_tolerance,
            render_tolerance_pixels=self.render_tolerance_pixels,
            geometry_slack_ratio=self.geometry_slack_ratio,
            **self.get_shader_params(),
        )
        # A PN soup is what a CROSS-FAMILY ``become`` renders during the morph
        # window, so without this a non-casting mob would start casting for the
        # duration of its own morph and stop again at the far end -- both
        # endpoints honour the flag (``_MORPH_ADOPTED_ATTRS``), and the seam was
        # strictly interior to the transition.
        primitive.declare_shadow_flags(*self._resolved_shadow_flags())
        return primitive

    def _get_memory_used_per_timestep(self):
        # Timeline state (position, normal, color, opacity/glow and material)
        # plus the logical primitive's cloned position/normal/color inputs.
        variables = 24 + sum(v.shape[-1] for v in self.get_shader_params().values())
        return int(self.location.shape[-2] * variables * 4)
