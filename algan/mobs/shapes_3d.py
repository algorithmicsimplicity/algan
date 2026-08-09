from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from algan.animatable_base.animatable import animated_function
from algan.animatable_base.mob import Mob
from algan.animation_timeline.animation_contexts import Off, Sync
from algan.constants.color import WHITE
from algan.constants.math import PI
from algan.constants.spatial import LEFT, ORIGIN, OUT, RIGHT, UP
from algan.geometry.geometry import get_orthonormal_vector, project_onto_basis
from algan.mobs.group import Group
from algan.mobs.shapes_2d import Circle
from algan.mobs.surfaces.surface import Surface
from algan.utils.tensor_utils import cast_to_tensor


def _surface_resolution_kwargs(resolution, kwargs):
    """Translate Manim's surface resolution/style names to Algan's Surface."""
    if resolution is not None:
        if isinstance(resolution, int):
            u_res = v_res = resolution
        else:
            u_res, v_res = resolution
        # Manim's values count patches; Algan's values count sampled vertices.
        kwargs.setdefault("grid_width", int(u_res) + 1)
        kwargs.setdefault("grid_height", int(v_res) + 1)
    if "fill_color" in kwargs:
        kwargs.setdefault("color", kwargs.pop("fill_color"))
    if "fill_opacity" in kwargs:
        kwargs.setdefault("opacity", kwargs.pop("fill_opacity"))
    checkerboard = kwargs.pop("checkerboard_colors", None)
    if checkerboard not in (None, False):
        colors = list(checkerboard)
        if colors:
            kwargs.setdefault("color", colors[0])
        if len(colors) > 1:
            kwargs.setdefault("checkered_color", colors[1])
    # Surface faces do not have a separate raster-style stroke in Algan.
    for key in (
        "stroke_color",
        "stroke_width",
        "stroke_opacity",
        "shade_in_3d",
        "should_make_jagged",
        "surface_piece_config",
        "pre_function_handle_to_anchor_scale_factor",
    ):
        kwargs.pop(key, None)
    return kwargs


def _radial_and_axial_coordinates(points, center, axis):
    """Return distances parallel and perpendicular to a shape's main axis."""
    center = center.reshape(-1, 3)[0].to(device=points.device, dtype=points.dtype)
    axis = F.normalize(
        axis.reshape(-1, 3)[0].to(device=points.device, dtype=points.dtype),
        p=2,
        dim=-1,
    )
    relative = points - center
    axial = (relative * axis).sum(dim=-1)
    radial = (relative - axial.unsqueeze(-1) * axis).norm(dim=-1)
    return radial, axial


class Sphere(Surface):
    """A 3-D sphere with Manim-compatible constructor arguments."""

    def __init__(
        self,
        center=ORIGIN,
        radius=1,
        resolution=None,
        u_range=(0, 2 * PI),
        v_range=(0, PI),
        *args,
        **kwargs,
    ):
        self.radius = radius
        kwargs = _surface_resolution_kwargs(resolution, kwargs)
        kwargs.setdefault("location", center)
        # Surface owns u_range/v_range: assigning them before this call would be
        # overwritten by Surface.__init__'s own (0, 1) defaults.
        super().__init__(*args, u_range=u_range, v_range=v_range, **kwargs)

    def coord_function(self, coords_2d):
        # Keep the original Algan sampling orientation.  Although a sphere is
        # rotationally symmetric, rotating its latitude/longitude grid changes
        # triangle placement, normals, and therefore pixel output.
        x = coords_2d[..., 0]
        y = coords_2d[..., 1]
        longitude = -torch.pi * (1 - x) + x * torch.pi
        latitude = -torch.pi * 0.5 * (1 - y) + y * torch.pi * 0.5
        coords_3d = torch.stack(
            (
                torch.cos(latitude) * torch.cos(longitude),
                torch.sin(latitude),
                torch.cos(latitude) * torch.sin(longitude),
            ),
            dim=-1,
        )
        return coords_3d * self.radius

    def normal_function(self, uv):
        return self.coord_function(uv)

    def _pn_geometry_deviation(self, pn_points, _analytic_points, _analytic_uv):
        """Exact distance from each PN sample to this sphere's surface."""
        center = self.location.reshape(-1, 3)[0]
        radius = torch.as_tensor(
            self.radius,
            device=pn_points.device,
            dtype=pn_points.dtype,
        ).abs()
        return ((pn_points - center).norm(dim=-1) - radius).abs()


class Cone(Surface):
    """A circular cone with Manim-compatible constructor arguments."""

    def __init__(
        self,
        base_radius=1,
        height=1,
        direction=OUT,
        show_base=False,
        v_range=(0, 2 * PI),
        u_min=0,
        checkerboard_colors=False,
        radius=None,
        closed=None,
        resolution=None,
        *args,
        **kwargs,
    ):
        # Preserve Algan's older ``radius``/``closed`` spellings.
        if radius is not None:
            base_radius = radius
        if closed is not None:
            show_base = closed
        self.radius = base_radius
        self.base_radius = base_radius
        self.height = height
        self.direction = cast_to_tensor(direction)
        self.u_min = u_min
        kwargs["checkerboard_colors"] = checkerboard_colors
        kwargs = _surface_resolution_kwargs(resolution, kwargs)
        kwargs.setdefault("grid_aspect_ratio", 1 / PI)
        super().__init__(*args, v_range=v_range, **kwargs)

        direction_t = F.normalize(cast_to_tensor(direction), p=2, dim=-1)
        with Off(animation_manager=self.animation_manager):
            self.look(direction_t, axis=1)
        self.base_circle = Circle(
            scene=self.scene,
            radius=base_radius,
            color=self.color,
            add_to_scene=False,
        )
        with Off(animation_manager=self.animation_manager):
            self.base_circle.look(-direction_t, axis=2)
            self.base_circle.move_to(-direction_t * height * 0.5)
        if show_base:
            self.add_children(self.base_circle)
        self.start_point = -direction_t * height * 0.5
        self.end_point = direction_t * height * 0.5

    def coord_function(self, uv):
        u = uv[..., :1]
        phi = self.v_range[0] + uv[..., 1:] * (self.v_range[1] - self.v_range[0])
        radius = self.radius * (1 - u)
        return torch.cat(
            (
                torch.sin(phi) * radius,
                (u - 0.5) * self.height,
                torch.cos(phi) * radius,
            ),
            -1,
        )

    def normal_function(self, uv):
        xyz = self.coord_function(uv)
        radial = xyz.clone()
        radial[..., 1] = self.radius / max(float(self.height), 1e-10)
        return radial

    def _pn_geometry_deviation(self, pn_points, _analytic_points, _analytic_uv):
        """Exact distance from each PN sample to the finite conical side."""
        radial, axial = _radial_and_axial_coordinates(
            pn_points,
            self.location,
            self.get_upwards_direction(),
        )
        radius = torch.as_tensor(
            self.radius,
            device=pn_points.device,
            dtype=pn_points.dtype,
        ).abs()
        height = torch.as_tensor(
            self.height,
            device=pn_points.device,
            dtype=pn_points.dtype,
        )
        start_radial = radius
        start_axial = -height * 0.5
        radial_delta = -radius
        axial_delta = height
        length_squared = (radius * radius + height * height).clamp_min(
            torch.finfo(pn_points.dtype).eps
        )
        interpolation = (
            (radial - start_radial) * radial_delta + (axial - start_axial) * axial_delta
        ) / length_squared
        interpolation = interpolation.clamp(0, 1)
        closest_radial = start_radial + interpolation * radial_delta
        closest_axial = start_axial + interpolation * axial_delta
        return torch.sqrt(
            (radial - closest_radial).square() + (axial - closest_axial).square()
        )

    def get_start(self):
        return self.start_point

    def get_end(self):
        return self.end_point

    def get_direction(self):
        return self.direction


class Cylinder(Surface):
    """A cylinder with Manim-compatible constructor arguments."""

    def __init__(
        self,
        radius=1,
        height=1,
        direction=UP,
        v_range=(0, 2 * PI),
        show_ends=False,
        resolution=None,
        closed=None,
        *args,
        **kwargs,
    ):
        if closed is not None:
            show_ends = closed
        self.radius = radius
        self.height = height
        self._height = height
        self.direction = cast_to_tensor(direction)
        kwargs = _surface_resolution_kwargs(resolution, kwargs)
        if "grid_aspect_ratio" not in kwargs and "grid_height" not in kwargs:
            kwargs["grid_aspect_ratio"] = 1 / PI
        super().__init__(*args, v_range=v_range, **kwargs)

        direction_t = F.normalize(cast_to_tensor(direction), p=2, dim=-1)
        if not torch.allclose(direction_t, UP.to(direction_t)):
            self.look(direction_t, axis=1)
        if show_ends:
            self.add_bases(direction_t)

    def add_bases(self, direction=None):
        if direction is None:
            direction = F.normalize(cast_to_tensor(self.direction), p=2, dim=-1)
        self.bottom_cap = Circle(
            scene=self.scene, radius=self.radius, color=self.color, add_to_scene=False
        )
        self.top_cap = Circle(
            scene=self.scene, radius=self.radius, color=self.color, add_to_scene=False
        )
        self.bottom_cap.look(-direction, axis=2)
        self.top_cap.look(direction, axis=2)
        self.bottom_cap.move_to(-direction * self.height * 0.5)
        self.top_cap.move_to(direction * self.height * 0.5)
        self.base_bottom = self.bottom_cap
        self.base_top = self.top_cap
        self.add_children(self.bottom_cap, self.top_cap)
        return self

    def coord_function(self, uv):
        uv[..., 1:] /= uv[..., 1:].amax()
        u = -uv[..., :1]
        v = uv[..., 1:]
        return (
            (u * torch.pi * 2).sin() * self.radius * self.get_right_basis()
            + (v - 0.5) * self.height * self.get_upwards_basis()
            + (u * torch.pi * 2).cos() * self.radius * self.get_forward_basis()
        )

    def normal_function(self, uv):
        xyz = self.coord_function(uv)
        return project_onto_basis(
            xyz, [self.get_right_direction(), self.get_forward_direction()]
        )

    def _pn_geometry_deviation(self, pn_points, _analytic_points, _analytic_uv):
        """Exact distance from each PN sample to the cylindrical side."""
        radial, _ = _radial_and_axial_coordinates(
            pn_points,
            self.location,
            self.get_upwards_direction(),
        )
        radius = torch.as_tensor(
            self.radius,
            device=pn_points.device,
            dtype=pn_points.dtype,
        ).abs()
        return (radial - radius).abs()

    def set_direction(self, direction):
        direction = F.normalize(cast_to_tensor(direction), p=2, dim=-1)
        start = self.get_center() - direction * self.height * 0.5
        end = self.get_center() + direction * self.height * 0.5
        self.direction = direction
        return self.move_between_points(start, end)

    @animated_function(animated_args={"interpolation": 0})
    def set_start_point(self, point, interpolation=1):
        offset = self.get_upwards_basis() * 0.5
        current_end = self.location + offset
        current_start = self.location - offset
        point = current_start * (1 - interpolation) + interpolation * cast_to_tensor(
            point
        )
        self._move_between_points(point, current_end)
        return self

    @animated_function(animated_args={"interpolation": 0})
    def set_end_point(self, point, interpolation=1):
        offset = self.get_upwards_basis() * 0.5
        current_end = self.location + offset
        current_start = self.location - offset
        point = current_end * (1 - interpolation) + interpolation * cast_to_tensor(
            point
        )
        self._move_between_points(current_start, point)
        return self

    @animated_function(animated_args={"interpolation": 0})
    def move_between_points(self, start, end, interpolation=1):
        start = cast_to_tensor(start)
        end = cast_to_tensor(end)
        offset = (
            self.get_upwards_direction()
            * self.scale_coefficient[..., 1].unsqueeze(-1)
            * 0.5
        )
        current_end = self.location + offset
        current_start = self.location - offset
        start = current_start * (1 - interpolation) + interpolation * start
        end = current_end * (1 - interpolation) + interpolation * end
        self._move_between_points(start, end)
        return self

    def _move_between_points(self, start, end):
        start = cast_to_tensor(start)
        end = cast_to_tensor(end)
        with Sync(animation_manager=self.animation_manager):
            up_b = F.normalize(end - start, p=2, dim=-1)
            right_b = get_orthonormal_vector(up_b)
            forward_b = get_orthonormal_vector(up_b, right_b)
            self.move_to((start + end) * 0.5)
            self.setattr_and_record_modification(
                "basis",
                torch.cat(
                    (
                        right_b * self.scale_coefficient[..., :1],
                        end - start,
                        forward_b * self.scale_coefficient[..., 2:],
                    ),
                    -1,
                ),
            )
            self.set_location_by_function(self.coord_function)
        self.direction = up_b
        return self


class Arrow3D(Mob):
    """An arrow made from an Algan cylinder and cone."""

    def __init__(
        self,
        start=LEFT,
        end=RIGHT,
        thickness: float = 0.02,
        height: float = 0.3,
        base_radius: float = 0.08,
        color=WHITE,
        resolution=24,
        *args,
        **kwargs,
    ):
        start = cast_to_tensor(start)
        end = cast_to_tensor(end)
        vector = end - start
        length = vector.norm(p=2, dim=-1, keepdim=True)
        if float(length.reshape(-1)[0]) <= height:
            raise ValueError("Arrow3D length must be greater than its tip height")
        direction = F.normalize(vector, p=2, dim=-1)
        shaft_end = end - direction * height
        super().__init__(*args, location=(start + end) * 0.5, color=color, **kwargs)
        surface_resolution = (
            (resolution, resolution) if isinstance(resolution, int) else resolution
        )
        self.tail = Cylinder(
            scene=self.scene,
            radius=thickness,
            height=float((shaft_end - start).norm(p=2, dim=-1).reshape(-1)[0]),
            direction=direction,
            show_ends=True,
            resolution=surface_resolution,
            color=color,
            add_to_scene=False,
        )
        self.tail.move_to((start + shaft_end) * 0.5)
        self.head = Cone(
            scene=self.scene,
            base_radius=base_radius,
            height=height,
            direction=direction,
            show_base=True,
            resolution=surface_resolution,
            color=color,
            add_to_scene=False,
        )
        self.head.move_to(end - direction * height * 0.5)
        self.cone = self.head
        self.start_point = Mob(location=start, opacity=0)
        self.end_point = Mob(location=end, opacity=0)
        self.length = length
        self.add_children(self.tail, self.head)

    def get_memory_used_per_timestep(self):
        return sum(child.get_memory_used_per_timestep() for child in self.children)

    def get_render_primitives(self):
        primitives = []
        for child in self.children:
            primitive = child.get_render_primitives()
            if primitive is None:
                continue
            primitives.extend(primitive if isinstance(primitive, list) else [primitive])
        return primitives or None

    def get_start(self):
        return self.start_point.location

    def get_end(self):
        return self.end_point.location

    def get_vector(self):
        return self.get_end() - self.get_start()

    def get_unit_vector(self):
        return F.normalize(self.get_vector(), p=2, dim=-1)


class Dot3D(Sphere):
    """A spherical dot with Manim-compatible constructor arguments."""

    def __init__(
        self,
        point=torch.zeros(3),
        radius=0.08,
        color=None,
        resolution=(8, 8),
        **kwargs,
    ):
        if color is not None:
            kwargs["color"] = color
        if resolution is not None:
            if isinstance(resolution, int):
                resolution = (resolution, resolution)
            kwargs.setdefault("grid_width", int(resolution[0]))
            kwargs.setdefault("grid_height", int(resolution[1]))
        super().__init__(radius=radius, **kwargs)
        self.move_to(point)


class Line3D(Cylinder):
    """A cylindrical line between two points."""

    def __init__(
        self,
        start=torch.tensor((-1.0, 0.0, 0.0)),
        end=torch.tensor((1.0, 0.0, 0.0)),
        thickness=0.02,
        color=None,
        resolution=24,
        **kwargs,
    ):
        from algan.utils.tensor_utils import cast_to_tensor

        if color is not None:
            kwargs["color"] = color
        if isinstance(resolution, int):
            kwargs.setdefault("grid_width", max(4, int(resolution)))
            kwargs.setdefault("grid_height", 2)
        else:
            kwargs.setdefault("grid_height", int(resolution[0]))
            kwargs.setdefault("grid_width", int(resolution[1]))
        self.start = cast_to_tensor(start)
        self.end = cast_to_tensor(end)
        self.thickness = thickness
        super().__init__(radius=thickness, height=1, closed=True, **kwargs)
        self.move_between_points(self.start, self.end)

    def get_start(self):
        return self.start.clone()

    def get_end(self):
        return self.end.clone()

    def set_start_and_end_attrs(self, start, end, **kwargs):
        from algan.utils.tensor_utils import cast_to_tensor

        self.start = cast_to_tensor(
            start.get_center() if hasattr(start, "get_center") else start
        )
        self.end = cast_to_tensor(
            end.get_center() if hasattr(end, "get_center") else end
        )
        self.move_between_points(self.start, self.end)
        return self

    def move_between_points(self, start, end, interpolation=1):
        from algan.utils.tensor_utils import cast_to_tensor

        result = super().move_between_points(start, end, interpolation=interpolation)
        if interpolation == 1:
            self.start = cast_to_tensor(start)
            self.end = cast_to_tensor(end)
        return result

    @classmethod
    def parallel_to(cls, line, point, length=1, **kwargs):
        direction = line.get_end() - line.get_start()
        direction = F.normalize(direction, p=2, dim=-1)
        point = point.get_center() if hasattr(point, "get_center") else point
        return cls(
            point - direction * length / 2, point + direction * length / 2, **kwargs
        )

    @classmethod
    def perpendicular_to(cls, line, point, length=1, **kwargs):
        direction = line.get_end() - line.get_start()
        perpendicular = get_orthonormal_vector(F.normalize(direction, p=2, dim=-1))
        point = point.get_center() if hasattr(point, "get_center") else point
        return cls(
            point - perpendicular * length / 2,
            point + perpendicular * length / 2,
            **kwargs,
        )


class Torus(Surface):
    """A torus with Manim-compatible major/minor radius and resolution API."""

    def __init__(
        self,
        major_radius=3,
        minor_radius=1,
        u_range=(0, torch.pi * 2),
        v_range=(0, torch.pi * 2),
        resolution=None,
        **kwargs,
    ):
        self.major_radius = self.R = major_radius
        self.minor_radius = self.r = minor_radius
        if resolution is not None:
            if isinstance(resolution, int):
                resolution = (resolution, resolution)
            kwargs.setdefault("grid_width", int(resolution[0]))
            kwargs.setdefault("grid_height", int(resolution[1]))
        super().__init__(
            coord_function=self.coord_function,
            u_range=u_range,
            v_range=v_range,
            **kwargs,
        )

    def coord_function(self, uv):
        u = self.u_range[0] + uv[..., :1] * (self.u_range[1] - self.u_range[0])
        v = self.v_range[0] + uv[..., 1:] * (self.v_range[1] - self.v_range[0])
        ring_radius = self.major_radius - self.minor_radius * torch.cos(v)
        return torch.cat(
            (
                ring_radius * torch.cos(u),
                ring_radius * torch.sin(u),
                -self.minor_radius * torch.sin(v),
            ),
            -1,
        )

    def normal_function(self, uv):
        u = self.u_range[0] + uv[..., :1] * (self.u_range[1] - self.u_range[0])
        v = self.v_range[0] + uv[..., 1:] * (self.v_range[1] - self.v_range[0])
        return torch.cat(
            (-torch.cos(v) * torch.cos(u), -torch.cos(v) * torch.sin(u), -torch.sin(v)),
            -1,
        )

    def _pn_geometry_deviation(self, pn_points, _analytic_points, _analytic_uv):
        """Exact distance from each PN sample to this torus's surface."""
        radial, axial = _radial_and_axial_coordinates(
            pn_points,
            self.location,
            self.get_forward_direction(),
        )
        major_radius = torch.as_tensor(
            self.major_radius,
            device=pn_points.device,
            dtype=pn_points.dtype,
        ).abs()
        minor_radius = torch.as_tensor(
            self.minor_radius,
            device=pn_points.device,
            dtype=pn_points.dtype,
        ).abs()
        tube_distance = torch.sqrt((radial - major_radius).square() + axial.square())
        return (tube_distance - minor_radius).abs()

    def func(self, u, v):
        u = torch.as_tensor(u)
        v = torch.as_tensor(v)
        ring_radius = self.major_radius - self.minor_radius * torch.cos(v)
        return torch.stack(
            (
                ring_radius * torch.cos(u),
                ring_radius * torch.sin(u),
                -self.minor_radius * torch.sin(v),
            ),
            -1,
        )


def _face_style_kwargs(faces_config, kwargs):
    faces_config = dict(faces_config or {})
    out = dict(kwargs)
    if "fill_color" in faces_config:
        out["color"] = faces_config["fill_color"]
    elif "color" in faces_config:
        out["color"] = faces_config["color"]
    if "fill_opacity" in faces_config:
        out["opacity"] = faces_config["fill_opacity"]
    # Flat triangle primitives have no separate outline primitive.  Manim's
    # face stroke options are accepted by constructors but intentionally do
    # not flow into Mob.__init__ as unknown keywords.
    return out


class _PolyhedronGraph(Group):
    def __init__(self, vertices, edges, **kwargs):
        self.vertices = list(range(len(vertices)))
        self.edges = edges
        self._vertex_mobs = vertices
        super().__init__(*vertices, **kwargs)

    def __getitem__(self, item):
        if item in self.vertices:
            return self._vertex_mobs[item]
        return super().__getitem__(item)


class Polyhedron(Mob):
    """Polyhedron defined by vertex coordinates and indexed polygon faces."""

    def __init__(
        self,
        vertex_coords,
        faces_list,
        faces_config=None,
        graph_config=None,
        **kwargs,
    ):
        from algan.mobs.group import Group
        from algan.mobs.shapes_2d import TriangleTriangulated
        from algan.utils.tensor_utils import cast_to_tensor

        self.vertex_coords = cast_to_tensor(vertex_coords).reshape(-1, 3)
        self.faces_list = [list(face) for face in faces_list]
        self.vertex_indices = list(range(len(self.vertex_coords)))
        self.layout = {i: self.vertex_coords[i] for i in self.vertex_indices}
        self.face_coords = [
            [self.vertex_coords[j] for j in face] for face in self.faces_list
        ]
        self.edges = self.get_edges(self.faces_list)
        self.faces_config = dict(faces_config or {})
        self.graph_config = dict(graph_config or {})

        super().__init__(**kwargs)
        face_style = _face_style_kwargs(self.faces_config, {})
        face_groups = []
        for face in self.faces_list:
            triangles = []
            for i in range(1, len(face) - 1):
                corners = torch.stack(
                    (
                        self.vertex_coords[face[0]],
                        self.vertex_coords[face[i]],
                        self.vertex_coords[face[i + 1]],
                    )
                )
                triangles.append(
                    TriangleTriangulated(
                        corners,
                        scene=self.scene,
                        add_to_scene=False,
                        **face_style,
                    )
                )
            face_groups.append(Group(*triangles, scene=self.scene, add_to_scene=False))
        self.faces = Group(*face_groups, scene=self.scene, add_to_scene=False)

        vertex_type = self.graph_config.get("vertex_type", Dot3D)
        vertex_config = dict(self.graph_config.get("vertex_config", {}))
        vertices = [
            vertex_type(
                point=self.vertex_coords[i],
                scene=self.scene,
                add_to_scene=False,
                **vertex_config,
            )
            for i in self.vertex_indices
        ]
        self.graph = _PolyhedronGraph(
            vertices, self.edges, scene=self.scene, add_to_scene=False
        )
        self.add_children(self.faces, self.graph)

    def _face_primitive_mobs(self):
        return [
            descendant
            for descendant in self.faces.get_descendants()
            if hasattr(descendant, "get_render_primitives")
        ]

    def get_memory_used_per_timestep(self):
        return sum(
            mob.get_memory_used_per_timestep() for mob in self._face_primitive_mobs()
        )

    def get_render_primitives(self):
        primitives = []
        for mob in self._face_primitive_mobs():
            primitive = mob.get_render_primitives()
            if primitive is None:
                continue
            primitives.extend(primitive if isinstance(primitive, list) else [primitive])
        return primitives or None

    @staticmethod
    def get_edges(faces_list):
        edges = []
        seen = set()
        for face in faces_list:
            for a, b in zip(face, face[1:] + face[:1]):
                edge = tuple(sorted((a, b)))
                if edge not in seen:
                    seen.add(edge)
                    edges.append(edge)
        return edges

    def extract_face_coords(self):
        return [[self.graph[j].get_center() for j in face] for face in self.faces_list]

    def update_faces(self, _mob=None):
        # The native Algan hierarchy propagates transforms from the parent.  If
        # individual graph vertices are edited, rebuild the face triangles.
        new = Polyhedron(
            [self.graph[i].get_center() for i in self.vertex_indices],
            self.faces_list,
            scene=self.scene,
            faces_config=self.faces_config,
            graph_config=self.graph_config,
            add_to_scene=False,
        )
        with Sync(animation_manager=self.animation_manager):
            self.faces.become(new.faces)
        return self


class Prism(Polyhedron):
    """A right rectangular prism with dimensions in ``[x, y, z]`` order."""

    def __init__(self, dimensions=(3, 2, 1), **kwargs):
        from algan.utils.tensor_utils import cast_to_tensor

        self.dimensions = cast_to_tensor(dimensions).reshape(-1)
        x, y, z = self.dimensions / 2
        vertices = [
            [-x, -y, -z],
            [x, -y, -z],
            [x, y, -z],
            [-x, y, -z],
            [-x, -y, z],
            [x, -y, z],
            [x, y, z],
            [-x, y, z],
        ]
        faces = [
            [0, 3, 2, 1],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [1, 2, 6, 5],
            [2, 3, 7, 6],
            [3, 0, 4, 7],
        ]
        faces_config = dict(kwargs.pop("faces_config", {}) or {})
        for source, target in (
            ("fill_color", "fill_color"),
            ("fill_opacity", "fill_opacity"),
            ("stroke_width", "stroke_width"),
        ):
            if source in kwargs:
                faces_config[target] = kwargs.pop(source)
        super().__init__(vertices, faces, faces_config=faces_config, **kwargs)


class Cube(Prism):
    """A three-dimensional cube."""

    def __init__(
        self,
        side_length=2,
        fill_opacity=0.75,
        fill_color=None,
        stroke_width=0,
        **kwargs,
    ):
        from algan.constants.color import BLUE

        self.side_length = side_length
        if fill_color is None:
            fill_color = BLUE
        super().__init__(
            dimensions=(side_length, side_length, side_length),
            fill_opacity=fill_opacity,
            fill_color=fill_color,
            stroke_width=stroke_width,
            **kwargs,
        )


class Tetrahedron(Polyhedron):
    def __init__(self, edge_length=1, **kwargs):
        unit = edge_length * math.sqrt(2) / 4
        super().__init__(
            [
                [unit, unit, unit],
                [unit, -unit, -unit],
                [-unit, unit, -unit],
                [-unit, -unit, unit],
            ],
            [[0, 1, 2], [3, 0, 2], [0, 1, 3], [3, 1, 2]],
            **kwargs,
        )


class Octahedron(Polyhedron):
    def __init__(self, edge_length=1, **kwargs):
        unit = edge_length * math.sqrt(2) / 2
        super().__init__(
            [
                [unit, 0, 0],
                [-unit, 0, 0],
                [0, unit, 0],
                [0, -unit, 0],
                [0, 0, unit],
                [0, 0, -unit],
            ],
            [
                [2, 4, 1],
                [0, 4, 2],
                [4, 3, 0],
                [1, 3, 4],
                [3, 5, 0],
                [1, 5, 3],
                [2, 5, 1],
                [0, 5, 2],
            ],
            **kwargs,
        )


class Icosahedron(Polyhedron):
    def __init__(self, edge_length=1, **kwargs):
        a = edge_length * ((1 + math.sqrt(5)) / 4)
        b = edge_length / 2
        vertices = [
            [0, b, a],
            [0, -b, a],
            [0, b, -a],
            [0, -b, -a],
            [b, a, 0],
            [b, -a, 0],
            [-b, a, 0],
            [-b, -a, 0],
            [a, 0, b],
            [a, 0, -b],
            [-a, 0, b],
            [-a, 0, -b],
        ]
        faces = [
            [1, 8, 0],
            [1, 5, 7],
            [8, 5, 1],
            [7, 3, 5],
            [5, 9, 3],
            [8, 9, 5],
            [3, 2, 9],
            [9, 4, 2],
            [8, 4, 9],
            [0, 4, 8],
            [6, 4, 0],
            [6, 2, 4],
            [11, 2, 6],
            [3, 11, 2],
            [0, 6, 10],
            [10, 1, 0],
            [10, 7, 1],
            [11, 7, 3],
            [10, 11, 7],
            [10, 11, 6],
        ]
        super().__init__(vertices, faces, **kwargs)


class Dodecahedron(Polyhedron):
    def __init__(self, edge_length=1, **kwargs):
        a = edge_length * ((1 + math.sqrt(5)) / 4)
        b = edge_length * ((3 + math.sqrt(5)) / 4)
        c = edge_length / 2
        vertices = [
            [a, a, a],
            [a, a, -a],
            [a, -a, a],
            [a, -a, -a],
            [-a, a, a],
            [-a, a, -a],
            [-a, -a, a],
            [-a, -a, -a],
            [0, c, b],
            [0, c, -b],
            [0, -c, -b],
            [0, -c, b],
            [c, b, 0],
            [-c, b, 0],
            [c, -b, 0],
            [-c, -b, 0],
            [b, 0, c],
            [-b, 0, c],
            [b, 0, -c],
            [-b, 0, -c],
        ]
        faces = [
            [18, 16, 0, 12, 1],
            [3, 18, 16, 2, 14],
            [3, 10, 9, 1, 18],
            [1, 9, 5, 13, 12],
            [0, 8, 4, 13, 12],
            [2, 16, 0, 8, 11],
            [4, 17, 6, 11, 8],
            [17, 19, 5, 13, 4],
            [19, 7, 15, 6, 17],
            [6, 15, 14, 2, 11],
            [19, 5, 9, 10, 7],
            [7, 10, 3, 14, 15],
        ]
        super().__init__(vertices, faces, **kwargs)


class ConvexHull3D(Polyhedron):
    def __init__(self, *points, tolerance=1e-5, **kwargs):
        import numpy as np
        from scipy.spatial import ConvexHull

        array = np.asarray(points, dtype=float)
        if len(array) < 4:
            raise ValueError("ConvexHull3D requires at least four non-coplanar points")
        hull = ConvexHull(array, qhull_options=f"QJ{tolerance}")
        vertex_ids = sorted({int(i) for i in hull.simplices.reshape(-1)})
        remap = {old: new for new, old in enumerate(vertex_ids)}
        vertices = [array[i].tolist() for i in vertex_ids]
        faces = [[remap[int(i)] for i in simplex] for simplex in hull.simplices]
        super().__init__(vertices, faces, **kwargs)
