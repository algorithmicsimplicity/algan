"""Renderer-independent equivalents of Manim's OpenGL surface Mobjects.

The ``OpenGL`` prefix selects a renderer in Manim.  Algan has one ray-traced
surface representation, so these classes retain Manim's construction/query API
while producing native
:class:`~algan.mobs.surfaces.surface.Surface` and
:class:`~algan.mobs.group.Group` objects.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from algan.constants.color import GREY, WHITE
from algan.mobs.group import Group
from algan.mobs.shapes_2d import Line
from algan.mobs.surfaces.surface import Surface
from algan.utils.memory_utils import is_out_of_memory
from algan.utils.python_utils import integer_or_none
from algan.utils.tensor_utils import unsquish


def _resolution_pair(resolution, default=(101, 101)):
    if resolution is None:
        return default
    count = integer_or_none(resolution)
    if count is not None:
        return (count, count)
    return tuple(int(value) for value in resolution)


def _uv_points(result, u):
    """``result`` as one point per ``(u, v)``, or None if it is not that."""
    if isinstance(result, torch.Tensor):
        result = result.to(device=u.device, dtype=u.dtype)
        if result.shape[-1] == 3:
            return result
        if result.shape[0] == 3 and result.shape[1:] == u.shape:
            return result.movedim(0, -1)
    array = np.asarray(result)
    if array.shape[-1:] == (3,) and array.shape[:-1] == tuple(u.shape):
        return torch.as_tensor(array, device=u.device, dtype=u.dtype)
    return None


def _call_uv_function(function, u, v):
    """Call either a torch-vectorized or ordinary scalar Manim UV function.

    The function is called on the whole ``u`` and ``v`` tensors first. If that
    raises a TypeError, ValueError or RuntimeError, or does not return one point
    per ``(u, v)``, the function is taken to be one that takes a pair of floats
    at a time, as Manim's do, and is called that way. An error from those calls
    is chained to the first one, so the traceback shows both. Running out of
    memory is raised at once: it says nothing about how the function takes its
    input, and calling it point by point on the CPU instead would hide it.
    """
    if function is None:
        return torch.stack((u, v, torch.zeros_like(u)), dim=-1)
    # Called on one column of samples, ``(N, 1)``, whose components-first and
    # components-last results -- ``(3, N, 1)`` and ``(N, 1, 3)`` -- never share
    # a shape, as ``(3, W, H)`` and ``(W, H, 3)`` do on a 3 x 3 grid. A column
    # is still a grid to a function that broadcasts against one (a Cylinder's
    # basis rows).
    column = (u.numel(), 1)
    u_column = u.reshape(column)
    try:
        points = _uv_points(function(u_column, v.reshape(column)), u_column)
    except (TypeError, ValueError, RuntimeError) as exc:
        if is_out_of_memory(exc):
            raise
        # Inside the handler, so an error from the scalar calls is chained to
        # this one.
        return _call_uv_function_on_each_point(function, u, v)
    if points is None or points.numel() != 3 * u.numel():
        return _call_uv_function_on_each_point(function, u, v)
    return points.reshape(*u.shape, 3)


def _call_uv_function_on_each_point(function, u, v):
    flat_u = u.detach().cpu().reshape(-1).numpy()
    flat_v = v.detach().cpu().reshape(-1).numpy()
    points = [
        np.asarray(function(float(uu), float(vv)), dtype=float)
        for uu, vv in zip(flat_u, flat_v)
    ]
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
        # Passed through: Surface owns these, and leaving them out let its
        # own (0, 1) defaults overwrite the range this constructor was
        # given a few lines above.
        super().__init__(
            coord_function=self._coord_from_unit_uv,
            u_range=self.u_range,
            v_range=self.v_range,
            **kwargs,
        )
        # Surface.__init__ stores its own ``resolution`` argument -- in
        # patches, and not passed above -- over Manim's sample counts, which
        # get_triangle_indices and OpenGLTexturedSurface read.
        self.resolution = (self.grid_width, self.grid_height)
        # The frame the samples were built in; see
        # get_surface_points_and_nudged_points.
        self._built_basis = self.basis.reshape(3, 3).clone()

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
        u_values = torch.linspace(
            0, 1, self.grid_width, device=base.device, dtype=base.dtype
        )
        v_values = torch.linspace(
            0, 1, self.grid_height, device=base.device, dtype=base.dtype
        )
        u_grid, v_grid = torch.meshgrid(u_values, v_values, indexing="ij")
        unit_uv = torch.stack((u_grid, v_grid), dim=-1)
        du_uv = unit_uv.clone()
        dv_uv = unit_uv.clone()
        du_uv[..., 0] += self.epsilon / max(self.u_range[1] - self.u_range[0], 1e-12)
        dv_uv[..., 1] += self.epsilon / max(self.v_range[1] - self.v_range[0], 1e-12)
        # The grid is where the surface is now, but uv_func is where it was
        # built: carry each nudge through every transform made since, as
        # Manim's nudged points travel with its points.
        unit = self._coord_from_unit_uv(unit_uv).reshape(-1, 3)
        frame = torch.linalg.solve(self._built_basis, self.basis.reshape(3, 3))
        du = base + (self._coord_from_unit_uv(du_uv).reshape(-1, 3) - unit) @ frame
        dv = base + (self._coord_from_unit_uv(dv_uv).reshape(-1, 3) - unit) @ frame
        return base, du, dv

    def get_unit_normals(self):
        points, du_points, dv_points = self.get_surface_points_and_nudged_points()
        return F.normalize(
            torch.linalg.cross(du_points - points, dv_points - points), dim=-1
        )

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
            _load_rgba5(dark_image_file) if dark_image_file is not None else None
        )

        if isinstance(uv_surface, OpenGLSurface):
            uv_func = uv_surface.passed_uv_func
            u_range = uv_surface.u_range
            v_range = uv_surface.v_range
            resolution = uv_surface.resolution
        else:
            # Native surfaces are already sampled in unit UV coordinates. The
            # active function, not the ``coord_function`` method: a Surface
            # built from a function keeps the base class's flat plane there.
            def uv_func(u, v):
                return uv_surface.coord_function_active(torch.stack((u, v), dim=-1))

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
        lines = self._build_lines(stroke_width=stroke_width, color=color, scene=scene)
        super().__init__(*lines, add_to_scene=add_to_scene, **kwargs)

    def _sample(self, u, v):
        uv = torch.tensor(
            [u, v], dtype=torch.get_default_dtype(), device=torch.get_default_device()
        )
        if isinstance(self.uv_surface, OpenGLSurface):
            point = self.uv_surface._coord_from_unit_uv(uv)
        else:
            # As in OpenGLTexturedSurface: the function it was built from.
            point = self.uv_surface.coord_function_active(uv)
        return point.reshape(3)

    def _build_lines(self, stroke_width, color, scene):
        nu, nv = self.resolution
        thickness = max(float(stroke_width), 0.1) / 2
        paths = []
        for fixed_u in torch.linspace(0, 1, nu):
            points = [
                self._sample(float(fixed_u), float(v)) for v in torch.linspace(0, 1, nv)
            ]
            paths.extend(
                Line(
                    a,
                    b,
                    scene=scene,
                    stroke_width=thickness,
                    stroke_color=color,
                    color=color,
                    filled=False,
                    add_to_scene=False,
                )
                for a, b in zip(points, points[1:])
            )
        for fixed_v in torch.linspace(0, 1, nv):
            points = [
                self._sample(float(u), float(fixed_v)) for u in torch.linspace(0, 1, nu)
            ]
            paths.extend(
                Line(
                    a,
                    b,
                    scene=scene,
                    stroke_width=thickness,
                    stroke_color=color,
                    color=color,
                    filled=False,
                    add_to_scene=False,
                )
                for a, b in zip(points, points[1:])
            )
        return paths


__all__ = [
    "OpenGLSurface",
    "OpenGLSurfaceGroup",
    "OpenGLTexturedSurface",
    "OpenGLSurfaceMesh",
]
