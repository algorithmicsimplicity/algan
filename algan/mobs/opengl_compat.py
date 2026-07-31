"""Renderer-independent equivalents of Manim's OpenGL surface Mobjects.

The ``OpenGL`` prefix selects a renderer in Manim.  Algan has one ray-traced
surface representation, so these classes retain Manim's construction/query API
while producing native
:class:`~algan.mobs.surfaces.surface.Surface` and
:class:`~algan.mobs.group.Group` objects.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from algan.constants.color import GREY, WHITE
from algan.mobs.group import Group
from algan.mobs.shapes_2d import Line
from algan.mobs.surfaces.surface import Surface, compute_grid_vertex_normals
from algan.utils.tensor_utils import unsquish


def _resolution_pair(resolution, default=(101, 101)):
    if resolution is None:
        return default
    if isinstance(resolution, int):
        return (resolution, resolution)
    return tuple(int(value) for value in resolution)


def _call_uv_function(function, u, v):
    """Call either a torch-vectorized or ordinary scalar Manim UV function."""
    if function is None:
        return torch.stack((u, v, torch.zeros_like(u)), dim=-1)
    try:
        result = function(u, v)
        if isinstance(result, torch.Tensor):
            result = result.to(device=u.device, dtype=u.dtype)
            if result.shape[-1] == 3:
                return result
            if result.shape[0] == 3 and result.shape[1:] == u.shape:
                return result.movedim(0, -1)
        array = np.asarray(result)
        if array.shape[-1:] == (3,) and array.shape[:-1] == tuple(u.shape):
            return torch.as_tensor(array, device=u.device, dtype=u.dtype)
    except (TypeError, ValueError, RuntimeError):
        pass

    flat_u = u.detach().cpu().reshape(-1).numpy()
    flat_v = v.detach().cpu().reshape(-1).numpy()
    points = [np.asarray(function(float(uu), float(vv)), dtype=float) for uu, vv in zip(flat_u, flat_v)]
    return torch.as_tensor(
        np.asarray(points).reshape(*u.shape, 3),
        device=u.device,
        dtype=u.dtype,
    )


class OpenGLSurface(Surface):
    def __init__(
        self,
        uv_func=None,
        u_range=None,
        v_range=None,
        resolution=None,
        axes=None,
        color=GREY,
        colorscale=None,
        colorscale_axis=2,
        opacity=1.0,
        gloss=0.3,
        shadow=0.4,
        prefered_creation_axis=1,
        epsilon=1e-5,
        render_primitive=None,
        depth_test=True,
        shader_folder=None,
        **kwargs,
    ):
        self.passed_uv_func = uv_func
        self.u_range = (0, 1) if u_range is None else tuple(u_range)
        self.v_range = (0, 1) if v_range is None else tuple(v_range)
        self.resolution = _resolution_pair(resolution)
        self.axes = axes
        self.colorscale = colorscale
        self.colorscale_axis = colorscale_axis
        self.gloss = gloss
        self.shadow = shadow
        self.prefered_creation_axis = prefered_creation_axis
        self.epsilon = epsilon
        self.depth_test = depth_test
        self.shader_folder = shader_folder
        self.render_primitive = render_primitive
        kwargs.setdefault("grid_width", self.resolution[0])
        kwargs.setdefault("grid_height", self.resolution[1])
        kwargs.setdefault("color", color)
        kwargs.setdefault("opacity", opacity)
        super().__init__(coord_function=self._coord_from_unit_uv, **kwargs)

    def uv_func(self, u, v):
        if self.passed_uv_func is None:
            return (u, v, 0.0)
        return self.passed_uv_func(u, v)

    def _coord_from_unit_uv(self, uv):
        u = self.u_range[0] + uv[..., 0] * (self.u_range[1] - self.u_range[0])
        v = self.v_range[0] + uv[..., 1] * (self.v_range[1] - self.v_range[0])
        return _call_uv_function(self.passed_uv_func, u, v)

    def get_surface_points_and_nudged_points(self):
        grid = unsquish(self.grid.location, -2, self.grid_height)
        base = grid.reshape(-1, 3)
        u_values = torch.linspace(0, 1, self.grid_width, device=base.device, dtype=base.dtype)
        v_values = torch.linspace(0, 1, self.grid_height, device=base.device, dtype=base.dtype)
        u_grid, v_grid = torch.meshgrid(u_values, v_values, indexing="ij")
        unit_uv = torch.stack((u_grid, v_grid), dim=-1)
        du_uv = unit_uv.clone()
        dv_uv = unit_uv.clone()
        du_uv[..., 0] += self.epsilon / max(self.u_range[1] - self.u_range[0], 1e-12)
        dv_uv[..., 1] += self.epsilon / max(self.v_range[1] - self.v_range[0], 1e-12)
        du = self._coord_from_unit_uv(du_uv).reshape(-1, 3)
        dv = self._coord_from_unit_uv(dv_uv).reshape(-1, 3)
        return base, du, dv

    def get_unit_normals(self):
        points, du_points, dv_points = self.get_surface_points_and_nudged_points()
        return F.normalize(torch.linalg.cross(du_points - points, dv_points - points), dim=-1)

    def get_triangle_indices(self):
        nu, nv = self.resolution
        indices = []
        for u in range(nu - 1):
            for v in range(nv - 1):
                a = u * nv + v
                b = (u + 1) * nv + v
                c = u * nv + v + 1
                d = (u + 1) * nv + v + 1
                indices.extend((a, b, c, c, b, d))
        return torch.tensor(indices, dtype=torch.long, device=self.location.device)

    def sort_faces_back_to_front(self, vect=None):
        # Algan's ray tracer performs visibility ordering geometrically.
        return self


class OpenGLSurfaceGroup(Group):
    def __init__(self, *parametric_surfaces, resolution=None, **kwargs):
        self.resolution = (0, 0) if resolution is None else _resolution_pair(resolution)
        super().__init__(*parametric_surfaces, **kwargs)


class OpenGLTexturedSurface(OpenGLSurface):
    def __init__(
        self,
        uv_surface,
        image_file,
        dark_image_file=None,
        image_mode="RGBA",
        shader_folder=None,
        **kwargs,
    ):
        from algan.mobs.image_compat import _load_rgba5

        if not isinstance(uv_surface, (OpenGLSurface, Surface)):
            raise TypeError("uv_surface must be an OpenGLSurface or Surface")
        self.uv_surface = uv_surface
        self.image_file = image_file
        self.dark_image_file = dark_image_file
        if isinstance(image_mode, (str, Path)):
            image_mode = (image_mode, image_mode)
        self.image_mode = image_mode
        texture = _load_rgba5(image_file)
        self.dark_image = (
            _load_rgba5(dark_image_file)
            if dark_image_file is not None
            else None
        )

        if isinstance(uv_surface, OpenGLSurface):
            uv_func = uv_surface.passed_uv_func
            u_range = uv_surface.u_range
            v_range = uv_surface.v_range
            resolution = uv_surface.resolution
        else:
            # Native surfaces are already sampled in unit UV coordinates.
            uv_func = lambda u, v: uv_surface.coord_function(torch.stack((u, v), dim=-1))
            u_range = (0, 1)
            v_range = (0, 1)
            resolution = (uv_surface.grid_width, uv_surface.grid_height)
        kwargs.setdefault("color_texture", texture.transpose(-3, -2).flip(-2))
        super().__init__(
            uv_func=uv_func,
            u_range=u_range,
            v_range=v_range,
            resolution=resolution,
            shader_folder=shader_folder,
            **kwargs,
        )


class OpenGLSurfaceMesh(Group):
    def __init__(
        self,
        uv_surface,
        resolution=None,
        stroke_width=1,
        normal_nudge=1e-2,
        depth_test=True,
        flat_stroke=False,
        **kwargs,
    ):
        if not isinstance(uv_surface, (OpenGLSurface, Surface)):
            raise TypeError("uv_surface must be an OpenGLSurface or Surface")
        self.uv_surface = uv_surface
        self.resolution = _resolution_pair(resolution, default=(21, 21))
        self.normal_nudge = normal_nudge
        self.depth_test = depth_test
        self.flat_stroke = flat_stroke
        add_to_scene = kwargs.pop("add_to_scene", True)
        color = kwargs.pop("color", WHITE)
        scene = kwargs.get("scene")
        if scene is None:
            from algan.animation_timeline.animation_contexts import (
                active_scene_for_new_mob,
            )

            scene = active_scene_for_new_mob()
            kwargs["scene"] = scene
        lines = self._build_lines(
            stroke_width=stroke_width, color=color, scene=scene
        )
        super().__init__(*lines, add_to_scene=add_to_scene, **kwargs)

    def _sample(self, u, v):
        uv = torch.tensor([u, v], dtype=torch.get_default_dtype(), device=torch.get_default_device())
        if isinstance(self.uv_surface, OpenGLSurface):
            point = self.uv_surface._coord_from_unit_uv(uv)
        else:
            point = self.uv_surface.coord_function(uv)
        return point.reshape(3)

    def _build_lines(self, stroke_width, color, scene):
        nu, nv = self.resolution
        thickness = max(float(stroke_width), 0.1) / 2
        paths = []
        for fixed_u in torch.linspace(0, 1, nu):
            points = [self._sample(float(fixed_u), float(v)) for v in torch.linspace(0, 1, nv)]
            paths.extend(
                Line(a, b, scene=scene, border_width=thickness, border_color=color, color=color, filled=False, add_to_scene=False)
                for a, b in zip(points, points[1:])
            )
        for fixed_v in torch.linspace(0, 1, nv):
            points = [self._sample(float(u), float(fixed_v)) for u in torch.linspace(0, 1, nu)]
            paths.extend(
                Line(a, b, scene=scene, border_width=thickness, border_color=color, color=color, filled=False, add_to_scene=False)
                for a, b in zip(points, points[1:])
            )
        return paths


__all__ = [
    "OpenGLSurface",
    "OpenGLSurfaceGroup",
    "OpenGLTexturedSurface",
    "OpenGLSurfaceMesh",
]
