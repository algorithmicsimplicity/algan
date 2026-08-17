"""The 3-D shapes, in two families.

**Curved** shapes -- :class:`Sphere`, :class:`Cylinder`, :class:`Cone`,
:class:`Torus`, and :class:`Dot3D` / :class:`Line3D` / :class:`Arrow3D` built
from them -- are :class:`~algan.mobs.surfaces.surface.Surface` subclasses. They
carry an analytic coordinate function and a normal function, and are tessellated
per frame to whatever the camera needs, so they stay smooth as you move in.

**Faceted** shapes -- :class:`Polyhedron` and everything built on it:
:class:`Prism`, :class:`Cube`, the Platonic solids, :class:`ConvexHull3D` -- are
defined by explicit vertices and flat polygon faces. Their faces are already
planar, so they are triangulated once at construction and never refined.

Several constructors accept Manim's argument names (``resolution`` counting
patches rather than vertices, ``checkerboard_colors``, ``u_range`` / ``v_range``
in radians) so that ported scripts keep working.

Unlike 2-D shapes, these respond to light. See
:doc:`/new_user_tutorials/three_d_basics`.
"""

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


def orient_faces_outward(vertex_coords, faces_list):
    """Rewind a closed polyhedron's faces so every one of them faces outward.

    Returns a new face list, or the input unchanged when the mesh is not a
    closed orientable manifold and the question therefore has no answer.

    Two steps, both standard. First make the winding CONSISTENT: two faces that
    agree on their orientation traverse their shared edge in opposite
    directions, so a flood fill over the shared-edge graph flips whichever
    neighbour disagrees. Then fix the global sign, which consistency alone
    cannot: the signed volume of the closed shell (the divergence theorem, one
    tetrahedron per triangle of the fan) is positive exactly when the faces
    point outward, so a negative total flips all of them.

    It bails out, leaving the input alone, on anything that is not a closed
    orientable manifold -- an undirected edge used by other than two faces (an
    open mesh, a T-junction, a non-manifold fin), a flood fill that reaches a
    face two ways with contradicting orientations (a Moebius-like shell), a
    shell in more than one connected piece, or a degenerate zero volume. A
    ``Polyhedron`` is public API and takes arbitrary user geometry, so the pass
    has to be a no-op wherever "outward" is not defined rather than guess.

    Why this exists: the projected winding sign IS the renderer's backface bit
    (``raster_taichi._AA_BACKFACE_BIT``), which is what separates the near and
    far sheets of a closed mesh for the analytic-AA run rule. The face lists
    Algan ships for the Platonic solids are Manim's, and they are not
    consistently oriented -- 12 of an ``Icosahedron``'s 20 faces, 2 of 4 on a
    ``Tetrahedron``, 2 of 8 on an ``Octahedron``, 3 of 12 on a
    ``Dodecahedron``, 0 of 6 on a ``Cube``. See
    ``rendering/raytracing/DESIGN_mesh_identity.md`` ss6.5.
    """
    faces = [list(face) for face in faces_list]
    if len(faces) < 2:
        return faces_list
    # Undirected edge -> the faces using it. Each face must contribute every
    # edge exactly once; a repeated vertex inside one face makes the shell
    # non-manifold and is rejected with everything else.
    edge_faces = {}
    for fi, face in enumerate(faces):
        if len(face) < 3 or len(set(face)) != len(face):
            return faces_list
        for a, b in zip(face, face[1:] + face[:1]):
            edge_faces.setdefault((min(a, b), max(a, b)), []).append(fi)
    if any(len(v) != 2 for v in edge_faces.values()):
        return faces_list

    def _directed(face):
        return set(zip(face, face[1:] + face[:1]))

    # Flood fill. ``flip[i]`` is whether face i must be reversed to agree with
    # face 0. Neighbours agree when they traverse the shared edge in OPPOSITE
    # directions, so sharing a directed edge means exactly one of them flips.
    flip = [None] * len(faces)
    flip[0] = False
    stack = [0]
    directed = [_directed(f) for f in faces]
    seen_count = 1
    while stack:
        fi = stack.pop()
        for a, b in zip(faces[fi], faces[fi][1:] + faces[fi][:1]):
            pair = edge_faces[(min(a, b), max(a, b))]
            fj = pair[0] if pair[1] == fi else pair[1]
            # Whether fj currently traverses this edge the same way fi does,
            # corrected for fi's own pending flip.
            same = (a, b) in directed[fj]
            want = same != flip[fi]
            if flip[fj] is None:
                flip[fj] = want
                seen_count += 1
                stack.append(fj)
            elif flip[fj] != want:
                return faces_list  # not orientable
    if seen_count != len(faces):
        return faces_list  # more than one shell

    oriented = [list(reversed(f)) if flip[i] else f for i, f in enumerate(faces)]

    # Global sign, from the signed volume of the triangulated shell. The fan
    # matches Polyhedron's own triangulation, so a polygon face contributes the
    # same tetrahedra the renderer will see.
    coords = cast_to_tensor(vertex_coords).reshape(-1, 3).to(torch.float64)
    volume = 0.0
    for face in oriented:
        p0 = coords[face[0]]
        for i in range(1, len(face) - 1):
            p1 = coords[face[i]]
            p2 = coords[face[i + 1]]
            volume += float(torch.dot(p0, torch.linalg.cross(p1, p2)))
    if volume == 0.0:
        return faces_list
    if volume < 0.0:
        oriented = [list(reversed(f)) for f in oriented]
    return oriented


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
    """A 3-D sphere, tessellated from a :class:`~algan.mobs.surfaces.surface.Surface`.

    Because it is a curved :class:`~algan.mobs.surfaces.surface.Surface` rather
    than a fixed mesh, its silhouette is refined per frame to whatever the camera
    needs, so it stays round as you move in.

    Parameters
    ----------
    center
        World-space location of the sphere's center, shape ``(*, 3)`` where ``*``
        denotes zero or more batch dimensions. Python lists and floats are cast to
        tensors. Defaults to ``ORIGIN`` (the world origin).
    radius
        Radius in world units. Defaults to ``1``.
    resolution
        Manim-style grid resolution as ``(u_patches, v_patches)``, or one int for
        both. Manim counts patches and Algan counts sampled vertices, so each value
        is used as ``grid_width``/``grid_height`` plus one. Defaults to ``None``,
        meaning Algan sizes the grid itself from ``geometry_tolerance``.
    u_range, v_range
        Parametric domain, in radians, accepted for Manim compatibility.

        .. note::

            These are stored but do **not** currently change the sphere's geometry:
            :meth:`coord_function` always sweeps a full longitude/latitude grid, so
            a partial range still builds a whole sphere. Build a partial sphere with
            :class:`~algan.mobs.surfaces.surface.Surface` and your own coordinate
            function instead. Defaults to ``(0, 2 * pi)`` and ``(0, pi)``.
    *args, **kwargs
        Passed to :class:`~algan.mobs.surfaces.surface.Surface` -- notably
        ``color``, ``checkered_color``, ``grid_width``/``grid_height`` and the
        texture maps.

    Examples
    --------
    A blue sphere, sized in world units:

    .. algan:: Example1Sphere
        :save_last_frame:

        from algan import *

        Sphere(radius=0.8, color=BLUE).spawn()

        Scene.save_video()
    """

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
    """A circular cone, tessellated from a :class:`~algan.mobs.surfaces.surface.Surface`.

    The cone is open at its base by default; ``show_base`` caps it with a
    :class:`~algan.mobs.shapes_2d.Circle` added as a child. The uncapped base
    circle is always built and available as ``base_circle``.

    Parameters
    ----------
    base_radius
        Radius of the base, in world units. Defaults to ``1``.
    height
        Distance from base to tip along ``direction``, in world units. Defaults
        to ``1``.
    direction
        Direction the tip points, shape ``(*, 3)``; it need not be normalized.
        Defaults to ``OUT`` (the +z axis, out of the screen).
    show_base
        Whether to cap the base with a filled circle. Defaults to ``False``: the
        cone is open, so the camera can see inside it.
    v_range
        Angular sweep around the axis, in radians -- a Manim-parity domain, which
        is why it contradicts Algan's usual degrees. ``(0, pi)`` gives a half
        cone. Defaults to ``(0, 2 * pi)`` (the full cone).
    u_min
        Retained for Manim compatibility; stored on the instance and not used by
        :meth:`coord_function`. Defaults to ``0``.
    checkerboard_colors
        Manim's two-tone surface styling: a sequence of two colours becomes
        Algan's ``color`` and ``checkered_color``. Defaults to ``False`` (a single
        colour).
    radius, closed
        Algan's older spellings of ``base_radius`` and ``show_base``. When not
        ``None`` they win over the Manim-named argument. Both default to ``None``.
    resolution
        Manim-style grid resolution as ``(u_patches, v_patches)``, or one int for
        both; each value becomes ``grid_width``/``grid_height`` plus one, since
        Manim counts patches and Algan counts vertices. Defaults to ``None``,
        meaning Algan sizes the grid itself from ``geometry_tolerance``.
    *args, **kwargs
        Passed to :class:`~algan.mobs.surfaces.surface.Surface`.

    Examples
    --------
    A capped cone pointing up the screen:

    .. algan:: Example1Cone
        :save_last_frame:

        from algan import *

        Cone(base_radius=0.6, height=1.2, direction=UP, show_base=True).spawn()

        Scene.save_video()
    """

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
    """A cylinder, tessellated from a :class:`~algan.mobs.surfaces.surface.Surface`.

    Only the curved side is built by default; ``show_ends`` adds the two end caps
    as children (:meth:`add_bases` does the same after construction).

    Parameters
    ----------
    radius
        Radius in world units. Defaults to ``1``.
    height
        Length along ``direction``, in world units. Defaults to ``1``.
    direction
        Axis the cylinder runs along, shape ``(*, 3)``; it need not be normalized.
        Defaults to ``UP`` (the +y axis).
    v_range
        Parametric domain, in radians, accepted for Manim compatibility.

        .. note::

            Stored but not currently used by :meth:`coord_function`, which always
            sweeps the full circle, so a partial range still builds a whole
            cylinder. Defaults to ``(0, 2 * pi)``.
    show_ends
        Whether to cap both ends with filled circles. Defaults to ``False``: the
        tube is open at both ends.
    resolution
        Manim-style grid resolution as ``(u_patches, v_patches)``, or one int for
        both; each value becomes ``grid_width``/``grid_height`` plus one, since
        Manim counts patches and Algan counts vertices. Defaults to ``None``,
        meaning Algan sizes the grid itself from ``geometry_tolerance``.
    closed
        Algan's older spelling of ``show_ends``. When not ``None`` it wins.
        Defaults to ``None``.
    *args, **kwargs
        Passed to :class:`~algan.mobs.surfaces.surface.Surface`.

    Examples
    --------
    A capped cylinder lying along the screen's x axis:

    .. algan:: Example1Cylinder
        :save_last_frame:

        from algan import *

        Cylinder(radius=0.4, height=1.6, direction=RIGHT, show_ends=True).spawn()

        Scene.save_video()
    """

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
            self._setattr_and_record_modification(
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
    """An arrow: a :class:`Cylinder` shaft with a :class:`Cone` tip, grouped.

    Both parts are curved surfaces, so the arrow is tessellated per frame like
    the rest of the curved family.

    Parameters
    ----------
    start, end
        Tail and head of the arrow, shape ``(*, 3)`` in world units. Default to
        ``LEFT`` and ``RIGHT``.
    thickness
        Radius of the shaft, in world units. Defaults to ``0.02``.
    height
        Length of the conical tip, in world units, measured back from ``end``.
        Defaults to ``0.3``.
    base_radius
        Radius of the tip's base, in world units. Defaults to ``0.08``.
    color
        An Algan :class:`~algan.constants.color.Color`, a named constant such as
        ``BLUE``, or anything ``Color()`` accepts. Defaults to ``WHITE``.
    resolution
        Grid resolution for both parts, as ``(grid_width, grid_height)`` or one
        int for both. Defaults to ``24``.
    *args, **kwargs
        Passed to :class:`~algan.animatable_base.mob.Mob`.

    Raises
    ------
    ValueError
        If the distance from ``start`` to ``end`` is not greater than ``height``,
        which would leave no room for a shaft.

    Examples
    --------
    An arrow pointing up and to the right:

    .. algan:: Example1Arrow3D
        :save_last_frame:

        from algan import *

        Arrow3D(start=LEFT + DOWN, end=RIGHT + UP, thickness=0.04,
                color=BLUE).spawn()

        Scene.save_video()
    """

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

    def _get_memory_used_per_timestep(self):
        return sum(child._get_memory_used_per_timestep() for child in self.children)

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
    """A small :class:`Sphere`, for marking a point in a 3-D scene.

    Parameters
    ----------
    point
        Where to put it, shape ``(*, 3)`` in world units. Defaults to the world
        origin.
    radius
        Radius in world units. Defaults to ``0.08`` -- small enough to read as a
        marker beside shapes of unit size.
    color
        An Algan :class:`~algan.constants.color.Color`, a named constant such as
        ``BLUE``, or anything ``Color()`` accepts. Defaults to ``None``, meaning
        the :class:`Sphere` default (``GREEN``).
    resolution
        Grid resolution as ``(grid_width, grid_height)``, or one int for both.
        Defaults to ``None``, meaning Algan sizes the grid itself.
    **kwargs
        Passed to :class:`Sphere`.

    Examples
    --------
    Three markers along a line:

    .. algan:: Example1Dot3D
        :save_last_frame:

        from algan import *

        for x in (-1, 0, 1):
            Dot3D(point=RIGHT * x, radius=0.15, color=BLUE).spawn()

        Scene.save_video()
    """

    def __init__(
        self,
        point=torch.zeros(3),
        radius=0.08,
        color=None,
        resolution=None,
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
    """A thin capped :class:`Cylinder` spanning two points.

    Unlike the 2-D :class:`~algan.mobs.shapes_2d.Line`, this has real thickness
    in world units and responds to light, so it stays visible from any angle.

    Parameters
    ----------
    start, end
        The endpoints, shape ``(*, 3)`` in world units. The line is moved and
        oriented to span them. Default to ``(-1, 0, 0)`` and ``(1, 0, 0)``.
    thickness
        Radius of the tube, in world units. Defaults to ``0.02``.
    color
        An Algan :class:`~algan.constants.color.Color`, a named constant such as
        ``BLUE``, or anything ``Color()`` accepts. Defaults to ``None``, meaning
        the :class:`Cylinder` default (``GREEN``).
    resolution
        One int sets the number of samples around the tube (``grid_width``, at
        least ``4``) with two samples along its length; a pair is taken as
        ``(grid_height, grid_width)``. Defaults to ``24``.
    **kwargs
        Passed to :class:`Cylinder`.

    Examples
    --------
    An edge drawn between two marked points:

    .. algan:: Example1Line3D
        :save_last_frame:

        from algan import *

        Line3D(start=LEFT + DOWN, end=RIGHT + UP, thickness=0.05,
               color=BLUE).spawn()

        Scene.save_video()
    """

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
    """A torus, tessellated from a :class:`~algan.mobs.surfaces.surface.Surface`.

    Parameters
    ----------
    major_radius
        Distance from the torus's center to the center of its tube, in world
        units. Defaults to ``3`` -- Manim's default, and wider than the visible
        frame, so pass an explicit size when laying several shapes out together.
    minor_radius
        Radius of the tube itself, in world units. Defaults to ``1``.
    u_range
        Sweep around the ring, in radians -- a Manim-parity domain, which is why
        it contradicts Algan's usual degrees. ``(0, pi)`` gives half a ring.
        Defaults to ``(0, 2 * pi)``.
    v_range
        Sweep around the tube's cross-section, in radians. ``(0, pi)`` opens the
        tube along its length. Defaults to ``(0, 2 * pi)``.
    resolution
        Manim-style grid resolution as ``(u_vertices, v_vertices)``, or one int
        for both, used directly as ``grid_width``/``grid_height``. Defaults to
        ``None``, meaning Algan sizes the grid itself from ``geometry_tolerance``.
    **kwargs
        Passed to :class:`~algan.mobs.surfaces.surface.Surface`.

    Examples
    --------
    A ring sized to fit the frame, and half of one:

    .. algan:: Example1Torus
        :save_last_frame:

        from algan import *

        Torus(major_radius=1.2, minor_radius=0.35, color=BLUE).spawn()
        Torus(major_radius=1.2, minor_radius=0.35, u_range=(0, PI),
              color=YELLOW).move(UP * 0.1).spawn()

        Scene.save_video()
    """

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
    """A solid built from explicit vertices and indexed polygon faces.

    This is the flat-sided family of 3-D shapes. Its faces are already planar, so
    each is triangulated once at construction and nothing is refined per frame --
    the opposite of the curved
    :class:`~algan.mobs.surfaces.surface.Surface`-backed shapes.

    Parameters
    ----------
    vertex_coords
        The vertices, shape ``(N, 3)`` in world units. Any nested sequence is cast
        to a tensor and reshaped, so a list of ``[x, y, z]`` lists works.
    faces_list
        One entry per face, each a sequence of indices into ``vertex_coords``
        naming that face's corners in order. Faces may have any number of corners;
        they are triangulated for you.
    faces_config
        Style overrides applied to every face, using Manim's names --
        ``fill_color``, ``fill_opacity``, ``stroke_width``. Defaults to ``None``
        (no overrides).
    graph_config
        Retained for Manim compatibility, where it styles the vertex-and-edge
        graph drawn over the solid. Defaults to ``None``.
    **kwargs
        Passed to :class:`~algan.animatable_base.mob.Mob` -- notably ``color``
        and ``location``.

    Examples
    --------
    A square pyramid from four base corners and an apex:

    .. algan:: Example1Polyhedron
        :save_last_frame:

        from algan import *

        Polyhedron(
            [[-0.6, -0.5, -0.6], [0.6, -0.5, -0.6], [0.6, -0.5, 0.6],
             [-0.6, -0.5, 0.6], [0, 0.7, 0]],
            [[0, 1, 2, 3], [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]],
            color=BLUE,
        ).spawn()

        Scene.save_video()
    """

    _morph_family = "mesh"

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
        # Imported here rather than at module scope: the raytracing settings
        # module pulls in Taichi, and no mob module is on that import chain
        # today. Read live, per the settings-are-read-at-call-time rule.
        from algan.rendering.raytracing import settings as rt_settings

        if rt_settings.POLYHEDRON_WINDING:
            # Gated, but not because it is known to move output -- measured, the
            # fast-suite render is BYTE-IDENTICAL across this flag while
            # ALGAN_MESH_ID is off, since a per-triangle surface id makes every
            # run one fragment and the facing bit then groups nothing. With
            # MESH_ID on it does move, which is the mechanism: one id per solid
            # leaves facing as the only thing separating the two sheets. Off by
            # default until tests/full_renders has been checked on a machine
            # whose baselines those are.
            self.faces_list = orient_faces_outward(self.vertex_coords, self.faces_list)
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

    def _get_memory_used_per_timestep(self):
        return sum(
            mob._get_memory_used_per_timestep() for mob in self._face_primitive_mobs()
        )

    def get_render_primitives(self):
        primitives = []
        for mob in self._face_primitive_mobs():
            primitive = mob.get_render_primitives()
            if primitive is None:
                continue
            primitives.extend(primitive if isinstance(primitive, list) else [primitive])
        # One member per triangle -- a Cube arrives as twelve. Declare them one
        # SURFACE so the analytic-AA run rule can span a face's diagonal and a
        # silhouette corner where two faces tile the same pixel: a polyhedron is
        # a single closed solid, so summing its exact areas is what tiles mean
        # (see primitives._mesh_ids_from_collection). Deliberately not done for
        # Arrow3D, whose children are separate interpenetrating solids.
        for primitive in primitives:
            primitive.mesh_key = ("polyhedron", self.id)
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
    """A right rectangular prism -- a box -- built from six flat faces.

    Unlike the curved shapes, a :class:`Prism` is a
    :class:`~algan.mobs.shapes_3d.Polyhedron`: its faces are already flat, so
    they are their own triangles and nothing is tessellated per frame.

    Parameters
    ----------
    dimensions
        Side lengths in ``[x, y, z]`` order, in world units. The box is centered
        on the Mob's location, so each side extends half its length either way.
        Defaults to ``(3, 2, 1)``.
    **kwargs
        Passed to :class:`~algan.mobs.shapes_3d.Polyhedron`. ``fill_color``,
        ``fill_opacity`` and ``stroke_width`` are Manim's face-styling names and
        are forwarded into ``faces_config`` rather than applied to the Mob.

    Examples
    --------
    A wide, flat box, useful as a floor:

    .. algan:: Example1Prism
        :save_last_frame:

        from algan import *

        Prism(dimensions=(3, 0.2, 2), fill_color=BLUE).spawn()

        Scene.save_video()
    """

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
    """A cube -- a :class:`Prism` with three equal sides.

    Like every :class:`~algan.mobs.shapes_3d.Polyhedron` its faces are flat, so
    it is not tessellated per frame the way the curved shapes are.

    Parameters
    ----------
    side_length
        Length of each edge, in world units. Defaults to ``2``.
    fill_opacity
        Opacity of the faces, from ``0`` (invisible) to ``1`` (opaque). Defaults
        to ``0.75``, Manim's value -- a cube is slightly see-through unless you
        say otherwise.
    fill_color
        Colour of the faces: an Algan :class:`~algan.constants.color.Color`, a
        named constant such as ``BLUE``, or anything ``Color()`` accepts.
        Defaults to ``None``, meaning ``BLUE``.
    stroke_width
        Width of the outline drawn around each face, in world units. Defaults to
        ``0`` (no outline).
    **kwargs
        Passed to :class:`Prism` and on to
        :class:`~algan.mobs.shapes_3d.Polyhedron`.

    Examples
    --------
    An opaque cube, turned so three faces are visible:

    .. algan:: Example1Cube
        :save_last_frame:

        from algan import *

        cube = Cube(side_length=1.2, fill_color=BLUE, fill_opacity=1).spawn()
        cube.rotate(35, UP)

        Scene.save_video()
    """

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
    """The four-faced Platonic solid, built from flat triangular faces.

    Parameters
    ----------
    edge_length
        Length of every edge, in world units. Defaults to ``1``.
    **kwargs
        Passed to :class:`Polyhedron` -- notably ``color`` and ``location``.

    Examples
    --------
    .. algan:: Example1Tetrahedron
        :save_last_frame:

        from algan import *

        Tetrahedron(edge_length=1.5, color=BLUE).spawn()

        Scene.save_video()
    """

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
    """The eight-faced Platonic solid, built from flat triangular faces.

    Parameters
    ----------
    edge_length
        Length of every edge, in world units. Defaults to ``1``.
    **kwargs
        Passed to :class:`Polyhedron` -- notably ``color`` and ``location``.

    Examples
    --------
    .. algan:: Example1Octahedron
        :save_last_frame:

        from algan import *

        Octahedron(edge_length=1.2, color=BLUE).spawn()

        Scene.save_video()
    """

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
    """The twenty-faced Platonic solid, built from flat triangular faces.

    Parameters
    ----------
    edge_length
        Length of every edge, in world units. Defaults to ``1``.
    **kwargs
        Passed to :class:`Polyhedron` -- notably ``color`` and ``location``.

    Examples
    --------
    .. algan:: Example1Icosahedron
        :save_last_frame:

        from algan import *

        Icosahedron(edge_length=0.8, color=BLUE).spawn()

        Scene.save_video()
    """

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
    """The twelve-faced Platonic solid, built from flat pentagonal faces.

    Parameters
    ----------
    edge_length
        Length of every edge, in world units. Defaults to ``1``.
    **kwargs
        Passed to :class:`Polyhedron` -- notably ``color`` and ``location``.

    Examples
    --------
    .. algan:: Example1Dodecahedron
        :save_last_frame:

        from algan import *

        Dodecahedron(edge_length=0.6, color=BLUE).spawn()

        Scene.save_video()
    """

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
    """The convex hull of a point cloud, as a flat-faced :class:`Polyhedron`.

    Interior points are discarded: only the vertices on the hull survive into the
    solid.

    Parameters
    ----------
    *points
        The points to wrap, each a 3-D coordinate in world units, passed as
        separate arguments. At least four are required, and they must not all be
        coplanar.
    tolerance
        Qhull's joggle magnitude, used to perturb degenerate inputs into a
        solvable configuration. Raise it if a nearly-coplanar cloud fails to
        triangulate. Defaults to ``1e-5``.
    **kwargs
        Passed to :class:`Polyhedron` -- notably ``color`` and ``location``.

    Raises
    ------
    ValueError
        If fewer than four points are given.

    Examples
    --------
    The hull of five scattered points:

    .. algan:: Example1ConvexHull3D
        :save_last_frame:

        from algan import *

        ConvexHull3D(
            (-0.7, -0.5, -0.5), (0.7, -0.5, -0.5), (0, -0.5, 0.7),
            (0, 0.8, 0), (0, -0.9, 0),
            color=BLUE,
        ).spawn()

        Scene.save_video()
    """

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
