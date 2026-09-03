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

The revolved solids share one vocabulary: ``radius``, ``u_range`` / ``v_range``
and ``closed``, with ``direction`` defaulting to ``UP`` on both :class:`Cone`
and :class:`Cylinder`. Manim's spellings of those (``base_radius``,
``show_base``, ``show_ends``, ``u_min``, ``checkerboard_colors``) raise here,
naming the Algan one; they are correct under ``algan.manim``, where Manim's
classes live. ``resolution`` is the one
Manim name kept, counting patches rather than vertices as Manim does, and
``u_range`` / ``v_range`` keep Manim's names but take Algan's degrees.

Unlike 2-D shapes, these respond to light. See
:doc:`/new_user_tutorials/three_d_basics`.
"""

from __future__ import annotations

import inspect
import math

import torch
import torch.nn.functional as F

from algan.animatable_base.animatable import animated_function
from algan.animatable_base.mob import Mob
from algan.animation_timeline.animation_contexts import Off, Sync
from algan.constants.color import WHITE
from algan.constants.math import DEGREES_TO_RADIANS, PI
from algan.constants.spatial import LEFT, ORIGIN, OUTWARD, RIGHT, UP
from algan.errors import AlganConfigurationError
from algan.geometry.geometry import get_orthonormal_vector
from algan.mobs.group import Group
from algan.mobs.surfaces.surface import Surface
from algan.settings.shape_style_profiles import _manim_shape_style_for
from algan.utils.api_renames import _reject_renamed_keywords
from algan.utils.tensor_utils import cast_to_tensor, unsquish


def _rewound(face):
    """``face`` traversed the other way round, keeping its first vertex first.

    ``Polyhedron`` triangulates a polygon as a fan from ``face[0]``, so which
    vertex comes first decides which diagonals the face is cut along. A plain
    ``reversed`` moves the last vertex to the front and re-cuts the polygon;
    holding index 0 in place reverses the winding and nothing else, which is
    what makes a solid's triangulation independent of the winding its table
    happened to be written with.
    """
    return [face[0], *face[:0:-1]]


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

    oriented = [_rewound(f) if flip[i] else f for i, f in enumerate(faces)]

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
        oriented = [_rewound(f) for f in oriented]
    return oriented


def _sweep_is_full(range_, span):
    """True if a parametric ``(start, end)`` sweep covers ``span`` degrees.

    What a revolved solid's closed-shell declaration hangs on: a partial sweep
    cuts the shell open (the docstrings of ``Sphere``/``Cylinder``/``Torus``
    all promise an open surface for a partial range), and the cut edges are not
    capped, so the inside shows through. Tolerance absorbs float sweep values.
    """
    return abs(float(range_[1]) - float(range_[0]) - span) < 1e-4


def _sweep_radians(range_):
    """A ``(start, end)`` sweep, given in Algan's degrees, as radians.

    The revolved solids take their ``u_range`` / ``v_range`` in degrees like
    every other angle in Algan, and store them that way -- so a reader of
    ``sphere.v_range`` sees what they passed. The trigonometry below is the only
    thing that wants radians, so the conversion happens here rather than at the
    boundary.
    """
    return (
        float(range_[0]) * DEGREES_TO_RADIANS,
        float(range_[1]) * DEGREES_TO_RADIANS,
    )


#: Manim's spellings for the revolved solids' arguments, and Algan's. Manim
#: gives one concept two or three names across ``Cone``/``Cylinder``/``Sphere``
#: (``base_radius`` beside ``radius``, ``show_base`` beside ``show_ends``,
#: ``u_min`` beside ``u_range``); the root namespace carries one vocabulary and
#: says so when a script uses Manim's. ``algan.manim.Cone`` and friends are
#: Manim's classes under Manim's names.
#: Extra guidance for the renames whose replacement takes a different kind of
#: value, so "Algan spells it X" alone would leave the reader stuck.
_SOLID_KEYWORD_HINTS: dict[str, str] = {
    "checkerboard_colors": (
        "A checkerboard is a texture map here rather than a second vertex "
        "color, so its detail does not depend on the tessellation: "
        "`color_texture=get_checkerboard((BLUE, BLUE_E))`."
    ),
}

_SOLID_KEYWORD_RENAMES: dict[str, str] = {
    "base_radius": "radius",
    "checkerboard_colors": "color_texture",
    "show_base": "closed",
    "show_ends": "closed",
    "u_min": "u_range",
}


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


#: ``_CapDisc`` sizes its rim from ``geometry_tolerance`` and
#: ``max_grid_resolution`` *before* it can call ``Surface.__init__``, so when a
#: caller does not pass them it needs the same fallbacks that call is about to
#: apply. Lifted here from the signature itself rather than restated, so the
#: two cannot drift apart.
_SURFACE_INIT_DEFAULTS = {
    name: parameter.default
    for name, parameter in inspect.signature(Surface.__init__).parameters.items()
    if parameter.default is not inspect.Parameter.empty
}


class _CapDisc(Surface):
    """The flat disc that closes a ``Cylinder``'s end or a ``Cone``'s base.

    Private: it is what ``bottom_cap`` / ``top_cap`` / ``base_circle`` hold, not
    a shape to build directly -- ``Circle`` is the 2-D disc and ``Surface`` the
    general lit one.

    A :class:`~algan.mobs.surfaces.surface.Surface` rather than a 2-D
    :class:`~algan.mobs.shapes_2d.Circle`, so that a cap is the same kind of
    geometry as the body it closes: a triangle mesh that responds to light,
    and one that can carry the body's ``_mesh_key`` so the renderer treats the
    rim as an interior edge of a single surface rather than the boundary where
    two independently antialiased surfaces meet.

    The grid is a fan over the body's own ring: ``segments`` samples around the
    rim and two along the radius, the inner row collapsing to the centre
    (welded like a sphere's pole). The rim comes from ``rim_function``, which
    the body supplies from the same expression its own ring is sampled at, so
    every one of the body's ring vertices is a rim vertex.

    The rim count starts at ``segments`` -- the body's own ring count -- and is
    then grown in whole multiples of it until the chord polygon tracks the true
    rim curve within ``geometry_tolerance``, capped at ``max_grid_resolution``
    like any surface's search. The disc's flat *interior* is exact at any
    resolution, which is why it is not handed to the resolution search; its
    *rim* is not exact, and nothing downstream can fix it: PN cannot curve a
    flat patch's boundary, so the rim stays straight chords at every dice level
    however finely the renderer would cut. It is therefore sized here, at
    construction, against the same tolerance the rest of the surface honours.
    Whole multiples of the body's count keep the watertight joint -- every one
    of the body's ring vertices remains exactly a rim vertex, and only
    vertices strictly between them are added.

    Parameters
    ----------
    rim_function
        Maps an azimuth parameter of shape ``(*, 1)`` in ``[0, 1]`` to rim
        offsets from the disc's centre, shape ``(*, 3)``, in the body's current
        frame. Read live, so re-basing the body and rebuilding the cap's grid
        keeps the two rings together.
    direction
        The way the disc faces -- the outward normal of the solid it closes,
        shape ``(*, 3)``; it need not be normalized. Defaults to ``OUTWARD``.
    segments
        The body's ring count: the starting sample count around the rim, so
        every one of the body's ring vertices is a rim vertex from the outset.
        The rim is refined upward from here in whole multiples of it until it
        meets ``geometry_tolerance``. Defaults to ``25``.
    geometry_tolerance, max_grid_resolution
        Passed through to :class:`~algan.mobs.surfaces.surface.Surface`, and
        also what the rim refinement above is measured against. Bodies pass
        their own through; left unset they fall back to ``Surface``'s defaults.
    *args, **kwargs
        Passed to :class:`~algan.mobs.surfaces.surface.Surface`.
    """

    # A cap is part of a closed solid's skin, so its back face is that solid's
    # inside; see Mob.two_sided.
    two_sided = False

    def __init__(
        self,
        rim_function,
        direction=OUTWARD,
        segments=25,
        *args,
        **kwargs,
    ):
        self._rim_function = rim_function
        self.direction = cast_to_tensor(direction)
        # A body sweeps its ring whichever way its own coord_function does, and
        # the two built-ins disagree (Cylinder negates its azimuth, Cone does
        # not). Sweeping the fan the wrong way round winds its triangles into
        # the solid, so the direction is measured against the outward normal
        # here rather than assumed, once, from the rim function itself.
        self._reverse = not self._sweep_faces(direction)
        if "grid_width" not in kwargs:
            # The bodies pass their tolerances in on the same kwargs, and the
            # rim is sized against exactly those; left unset they fall back to
            # the same ``Surface`` defaults the call below would apply. They
            # stay in ``kwargs`` so the disc carries them like any surface.
            kwargs["grid_width"] = self._rimmed_grid_width(
                int(segments),
                float(
                    kwargs.get(
                        "geometry_tolerance",
                        _SURFACE_INIT_DEFAULTS["geometry_tolerance"],
                    )
                ),
                int(
                    kwargs.get(
                        "max_grid_resolution",
                        _SURFACE_INIT_DEFAULTS["max_grid_resolution"],
                    )
                ),
            )
        kwargs.setdefault("grid_height", 2)
        super().__init__(*args, **kwargs)

    def _rimmed_grid_width(self, segments, geometry_tolerance, max_grid_resolution):
        """The smallest whole multiple of ``segments`` whose chord polygon hugs
        the rim within ``geometry_tolerance``.

        Whole multiples because the body-to-cap joint has no welding mechanism
        beyond coincident samples (see ``Cylinder.add_bases``): refining by any
        other step would drop body ring vertices off the rim. The search walks
        upward over multipliers and takes the first count that meets the
        tolerance, or the last one ``max_grid_resolution`` affords -- a rim
        that cannot meet tolerance inside the cap is the degraded-but-
        rendering situation every surface search tolerates, not an error.
        """
        chords = max(int(segments) - 1, 1)
        width = max(3, int(segments))
        for multiplier in range(1, int(max_grid_resolution)):
            candidate = chords * multiplier + 1
            if candidate > int(max_grid_resolution):
                break
            width = candidate
            if self._rim_chord_deviation(chords * multiplier) <= float(
                geometry_tolerance
            ):
                break
        return max(3, width)

    def _rim_chord_deviation(self, chord_count):
        """The worst distance from a rim chord's midpoint to the true rim curve.

        Sampled generically off ``rim_function`` -- the exact sagitta for a
        circular rim, and the right generalisation for whatever else a body
        hands this disc. This is the only measurement the rim's accuracy gets:
        the render-time criteria compare a flat patch against its own straight
        boundary (see ``_pn_geometry_deviation``), so nothing downstream can
        refine a rim sized too coarsely here.
        """
        steps = torch.arange(chord_count + 1, dtype=torch.float32).reshape(-1, 1)
        corners = self._rim_function(steps / chord_count).reshape(-1, 3)
        midpoints = (corners[:-1] + corners[1:]) * 0.5
        arcs = self._rim_function((steps[:-1] + 0.5) / chord_count).reshape(-1, 3)
        return float((midpoints - arcs).norm(dim=-1).amax())

    def _sweep_faces(self, direction):
        """True if the rim, swept forwards, winds a fan facing ``direction``."""
        outward = F.normalize(cast_to_tensor(direction).reshape(1, 3), p=2, dim=-1)
        first = self._rim_function(torch.zeros(1, 1))
        quarter = self._rim_function(torch.full((1, 1), 0.25))
        # ``turn`` is a cross product, so it points along -normal in Algan's
        # right-handed world (OUTWARD is +z): a fan swept first -> quarter faces
        # ``direction`` when the two point OPPOSITE ways.
        turn = torch.cross(first.reshape(1, 3), quarter.reshape(1, 3), dim=-1)
        return bool((turn * outward).sum() < 0)

    def coord_function(self, uv):
        azimuth = uv[..., :1]
        if self._reverse:
            azimuth = 1 - azimuth
        # Radius on the second component: 0 at the welded centre, 1 at the rim.
        return uv[..., 1:] * self._rim_function(azimuth)

    def _pn_geometry_deviation(self, pn_points, _analytic_points, _analytic_uv):
        """Zero: the disc's planar interior is exact at any resolution.

        True of the interior only. PN cannot curve a flat patch's *boundary* --
        the rim stays straight chords however finely it is diced -- so the rim
        is sized against ``geometry_tolerance`` at construction instead
        (``_rimmed_grid_width``); there is nothing for a render-time search to
        measure or refine here.
        """
        return torch.zeros_like(pn_points[..., 0])


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
    u_range
        Azimuthal sweep around the sphere's own up axis, in degrees. The sweep
        starts at the sphere's ``LEFT`` and turns through ``OUTWARD``, ``RIGHT``
        and ``INWARD``, so ``(0, 180)`` builds the ``OUTWARD`` half -- the one
        facing the camera. Defaults to ``(0, 360)``, all the way round.
    v_range
        Pole-to-pole sweep, in degrees: ``0`` is the ``DOWN`` pole and ``180``
        the ``UP`` one. ``(0, 90)`` builds the bottom hemisphere, ``(45, 135)``
        a band around the equator. Defaults to ``(0, 180)``, pole to pole.
    *args, **kwargs
        Passed to :class:`~algan.mobs.surfaces.surface.Surface` -- notably
        ``color``, ``grid_width``/``grid_height`` and the texture maps --
        notably ``color_texture``, which
        :func:`~algan.mobs.surfaces.procedural_textures.get_checkerboard` and
        its siblings build.

    Notes
    -----
    A partial ``u_range`` or ``v_range`` builds an open shell. The cut edges are
    not capped, so the inside of the surface shows through them, and the shape
    is re-tessellated from scratch rather than carved out of the whole sphere's
    grid. The tessellation search only promises ``geometry_tolerance``, and a
    sweep that still reaches a pole needs a finer grid there than a closed
    sphere does -- so a partial sphere is not automatically the cheaper one.

    Examples
    --------
    A blue sphere, sized in world units:

    .. algan:: Example1Sphere
        :save_last_frame:

        from algan import *

        Sphere(radius=0.8, color=BLUE).spawn()

        Scene.save_video()

    A hemisphere, and a band around the equator:

    .. algan:: Example2Sphere
        :save_last_frame:

        from algan import *

        Sphere(radius=0.7, v_range=(0, 90), color=BLUE).move(LEFT * 0.9).spawn()
        Sphere(radius=0.7, v_range=(60, 120),
               color=YELLOW).move(RIGHT * 0.9).spawn()

        Scene.save_video()
    """

    # Its normals face out of the sphere, so a back-facing hit is its inside.
    two_sided = False

    def __init__(
        self,
        center=ORIGIN,
        radius=1,
        resolution=None,
        u_range=(0, 360),
        v_range=(0, 180),
        *args,
        **kwargs,
    ):
        _reject_renamed_keywords(
            "Sphere",
            kwargs,
            _SOLID_KEYWORD_RENAMES,
            manim_alternative="Sphere",
            hints=_SOLID_KEYWORD_HINTS,
        )
        self.radius = radius
        kwargs = _surface_resolution_kwargs(resolution, kwargs)
        kwargs.setdefault("location", center)
        # Surface owns u_range/v_range: assigning them before this call would be
        # overwritten by Surface.__init__'s own (0, 1) defaults.
        super().__init__(*args, u_range=u_range, v_range=v_range, **kwargs)
        # A full pole-to-pole sweep tiles the shell exactly once per crossing;
        # a partial range cuts it open (see the Notes above), and an open
        # surface must not claim the closed-solid opacity behaviour.
        self.closed_shell = _sweep_is_full(u_range, 360) and _sweep_is_full(
            v_range, 180
        )

    def coord_function(self, coords_2d):
        # Keep the original Algan sampling orientation.  Although a sphere is
        # rotationally symmetric, rotating its latitude/longitude grid changes
        # triangle placement, normals, and therefore pixel output.  That is why
        # the parametric ranges are folded into the endpoints of the existing
        # interpolation rather than applied on top of it: at the default domain
        # the two endpoints come out as -pi/+pi and -pi/2/+pi/2 exactly, so the
        # arithmetic below is bit-for-bit what it was before the ranges were
        # honoured, and only a non-default range moves a vertex.
        x = coords_2d[..., 0]
        y = coords_2d[..., 1]
        u_start, u_end = _sweep_radians(self.u_range)
        v_start, v_end = _sweep_radians(self.v_range)
        longitude_start = -torch.pi + u_start
        longitude_end = -torch.pi + u_end
        latitude_start = -torch.pi * 0.5 + v_start
        latitude_end = -torch.pi * 0.5 + v_end
        longitude = longitude_start * (1 - x) + x * longitude_end
        latitude = latitude_start * (1 - y) + y * latitude_end
        coords_3d = torch.stack(
            (
                torch.cos(latitude) * torch.cos(longitude),
                torch.sin(latitude),
                # Negated against the longitude's sine so that the grid runs the
                # same way round the sphere as it did when OUTWARD was -z. A
                # coord_function writes world coordinates directly (the grid is
                # never mapped through the Mob's basis), so this is where a
                # built-in shape carries the z convention.
                -torch.cos(latitude) * torch.sin(longitude),
            ),
            dim=-1,
        )
        return coords_3d * self.radius

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

    The cone is open at its base by default; ``closed`` caps it with a
    flat disc added as a child -- a lit triangle mesh carrying the cone's own
    mesh identity, so the rim is an interior edge of one surface. The uncapped
    disc is always built and available as ``base_circle``.

    Parameters
    ----------
    radius
        Radius of the base, in world units. Defaults to ``1``.
    height
        Distance from base to tip along ``direction``, in world units. Defaults
        to ``1``.
    direction
        Direction the tip points, shape ``(*, 3)``; it need not be normalized.
        Defaults to ``UP`` (the +y axis), the same default as
        :class:`Cylinder`'s axis.
    closed
        Whether to cap the base with a filled circle. Defaults to ``False``: the
        cone is open, so the camera can see inside it. The disc samples its rim
        more finely than the cone samples its rings -- it has to, since a flat
        disc's boundary cannot be refined at render time the way the curved
        side's is -- so a capped cone carries more triangles than the side
        alone suggests.
    v_range
        Angular sweep around the axis, in degrees. ``(0, 180)`` gives a half
        cone. Defaults to ``(0, 360)`` (the full cone).
    resolution
        Manim-style grid resolution as ``(u_patches, v_patches)``, or one int for
        both; each value becomes ``grid_width``/``grid_height`` plus one, since
        Manim counts patches and Algan counts vertices. Defaults to ``None``,
        meaning Algan sizes the grid itself from ``geometry_tolerance``.
    *args, **kwargs
        Passed to :class:`~algan.mobs.surfaces.surface.Surface` -- notably
        ``color``, and ``color_texture`` for the two-tone styling Manim spells
        ``checkerboard_colors``:
        ``color_texture=get_checkerboard((BLUE, BLUE_E))``.

    Examples
    --------
    A capped cone pointing up the screen:

    .. algan:: Example1Cone
        :save_last_frame:

        from algan import *

        Cone(radius=0.6, height=1.2, direction=UP, closed=True).spawn()

        Scene.save_video()
    """

    # The side's normals face out of the cone. An UNCAPPED cone therefore
    # shades its inside as an inside (ambient only) rather than as a second
    # lit exterior; pass ``closed=True``, or set ``two_sided = True`` on
    # the instance, if you want the old two-sided lighting.
    two_sided = False

    def __init__(
        self,
        radius=1,
        height=1,
        direction=UP,
        v_range=(0, 360),
        closed=False,
        resolution=None,
        *args,
        **kwargs,
    ):
        _reject_renamed_keywords(
            "Cone",
            kwargs,
            _SOLID_KEYWORD_RENAMES,
            manim_alternative="Cone",
            hints=_SOLID_KEYWORD_HINTS,
        )
        self.radius = radius
        self.height = height
        self.direction = cast_to_tensor(direction)
        kwargs = _surface_resolution_kwargs(resolution, kwargs)
        super().__init__(*args, v_range=v_range, **kwargs)

        direction_t = F.normalize(cast_to_tensor(direction), p=2, dim=-1)
        with Off(animation_manager=self.animation_manager):
            self.look(direction_t, with_axis="up")
        # A capped cone's base has to be a Scene actor to be drawn at all: the
        # render loop collects primitives from ``Scene.actors``, not by walking
        # the hierarchy, and ``add_children`` does not register anything. Left
        # unregistered when the cone is open, so ``base_circle`` stays available
        # (it is documented as always built) without appearing in the render --
        # and when the cone itself is detached, so a morph target stays one.
        # The cone's own azimuth runs on the SECOND uv component (see
        # coord_function), so ``grid_height`` is its count around the axis and
        # is what puts the base's rim vertices on the cone's own. The cone's
        # tolerances go with it: the disc grows that count in whole multiples
        # of it as its rim needs, measured against the accuracy the cone
        # itself was built to.
        self._mesh_key = ("solid", self.id)
        self.base_circle = _CapDisc(
            rim_function=self._cap_ring_offsets,
            scene=self.scene,
            direction=-direction_t,
            segments=self.grid_height,
            color=self.color,
            geometry_tolerance=self._geometry_tolerance,
            max_grid_resolution=self._max_grid_resolution,
            add_to_scene=bool(closed) and self._added_to_scene,
        )
        self.base_circle._mesh_key = self._mesh_key
        with Off(animation_manager=self.animation_manager):
            self.base_circle.move_to(-direction_t * height * 0.5)
        # Capped AND swept the whole way round is what closes the shell: the
        # disc seals the base, but a partial sweep still leaves the wedge's cut
        # faces open (the disc itself is whole either way -- see
        # ``_cap_ring_offsets``). Both the side and its cap must carry the
        # declaration: they share one surface id (``_mesh_key``), and the
        # renderer reads it per triangle.
        self.closed_shell = bool(closed) and _sweep_is_full(v_range, 360)
        self.base_circle.closed_shell = self.closed_shell
        if closed:
            self.add_children(self.base_circle)
        self.start_point = -direction_t * height * 0.5
        self.end_point = direction_t * height * 0.5

    def _cap_ring_offsets(self, azimuth):
        """The base ring, as offsets from its centre, for the base disc.

        The same expression ``coord_function`` samples the ring at, swept over
        the whole circle whatever ``v_range`` is -- a partial cone still gets a
        whole base, matching Manim -- and read off the live basis, so it follows
        the cone.
        """
        phi = azimuth * 2 * PI
        # Minus the forward basis, as in ``Cylinder._cap_ring_offsets`` and for
        # the same reason: ``coord_function`` puts the ring's start at -z (the
        # cone is built unrotated, so that is where the forward axis does NOT
        # point), and the rim has to land on the body's own ring vertex for
        # vertex -- ``test_normal_orientation.py`` is what says so.
        return (
            phi.sin() * self.radius * self.get_right_basis()
            - phi.cos() * self.radius * self.get_forward_basis()
        )

    def coord_function(self, uv):
        u = uv[..., :1]
        v_start, v_end = _sweep_radians(self.v_range)
        phi = v_start + uv[..., 1:] * (v_end - v_start)
        radius = self.radius * (1 - u)
        return torch.cat(
            (
                torch.sin(phi) * radius,
                (u - 0.5) * self.height,
                # See Sphere.coord_function: a built-in shape's world-space z.
                -torch.cos(phi) * radius,
            ),
            -1,
        )

    def _pn_geometry_deviation(self, pn_points, _analytic_points, _analytic_uv):
        """Exact distance from each PN sample to the finite conical side."""
        radial, axial = _radial_and_axial_coordinates(
            pn_points,
            self.location,
            self.get_up_direction(),
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

    Only the curved side is built by default; ``closed`` adds the two end
    discs as children and as Scene actors (:meth:`add_bases` does the same after
    construction).

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
        Azimuthal sweep around the cylinder's axis, in degrees. The sweep starts
        on the side the cylinder's forward direction points at (``INWARD``, for
        the default ``direction=UP``) and turns toward its left, so ``(0, 180)``
        builds the ``LEFT`` half of the tube. Defaults to ``(0, 360)``, the
        closed tube. The extent *along* the axis comes from ``height``.
    closed
        Whether to close both ends with flat discs -- lit triangle meshes
        carrying the tube's own mesh identity, so each rim is an interior edge
        of one surface. Defaults to ``False``: the tube is open at both ends.
        The discs are whole circles even when ``v_range`` is partial, matching
        Manim. Each disc samples its rim more finely than the tube samples its
        rings -- it has to, since a flat disc's boundary cannot be refined at
        render time the way the curved tube's is -- so a capped cylinder
        carries more triangles than the tube alone suggests.
    resolution
        Manim-style grid resolution as ``(u_patches, v_patches)``, or one int for
        both; each value becomes ``grid_width``/``grid_height`` plus one, since
        Manim counts patches and Algan counts vertices. Defaults to ``None``,
        meaning Algan sizes the grid itself from ``geometry_tolerance``.
    *args, **kwargs
        Passed to :class:`~algan.mobs.surfaces.surface.Surface` -- notably
        ``color``, and ``color_texture`` for the two-tone styling Manim spells
        ``checkerboard_colors``:
        ``color_texture=get_checkerboard((BLUE, BLUE_E))``.

    Notes
    -----
    A partial ``v_range`` builds an open half-pipe, not a solid: the cut runs
    the length of the tube and the inside of the surface shows through it. The
    shape is re-tessellated from scratch, and since the tessellation search only
    promises ``geometry_tolerance``, a partial sweep can end up with a denser
    grid than the closed tube.

    Examples
    --------
    A capped cylinder lying along the screen's x axis:

    .. algan:: Example1Cylinder
        :save_last_frame:

        from algan import *

        Cylinder(radius=0.4, height=1.6, direction=RIGHT, closed=True).spawn()

        Scene.save_video()

    Half a tube, cut along its length and turned so the shell faces the camera
    (unrotated, the cut runs straight through the line of sight and the half
    reads as a flat panel):

    .. algan:: Example2Cylinder
        :save_last_frame:

        from algan import *

        Cylinder(radius=0.7, height=1.4, v_range=(0, PI),
                 color=BLUE).rotate(-90, UP).spawn()

        Scene.save_video()
    """

    # The tube's normals face out; see Cone for what that means for an
    # uncapped one.
    two_sided = False

    def __init__(
        self,
        radius=1,
        height=1,
        direction=UP,
        v_range=(0, 360),
        closed=False,
        resolution=None,
        *args,
        **kwargs,
    ):
        _reject_renamed_keywords(
            "Cylinder",
            kwargs,
            _SOLID_KEYWORD_RENAMES,
            manim_alternative="Cylinder",
            hints=_SOLID_KEYWORD_HINTS,
        )
        self.radius = radius
        self.height = height
        self._height = height
        self.direction = cast_to_tensor(direction)
        kwargs = _surface_resolution_kwargs(resolution, kwargs)
        super().__init__(*args, v_range=v_range, **kwargs)

        direction_t = F.normalize(cast_to_tensor(direction), p=2, dim=-1)
        if not torch.allclose(direction_t, UP.to(direction_t)):
            self.look(direction_t, with_axis="up")
        if closed:
            self.add_bases(direction_t)

    def add_bases(self, direction=None):
        if direction is None:
            direction = F.normalize(cast_to_tensor(self.direction), p=2, dim=-1)
        offset = direction * self.height * 0.5
        if getattr(self, "bottom_cap", None) is not None:
            # Idempotent, like ``add_children``: a second call re-aims the caps
            # this cylinder already has rather than building a second pair that
            # would stay attached and registered behind the first.
            return self._place_bases(direction, -offset, offset)
        # Scene actors, not merely children: the render loop collects
        # primitives from ``Scene.actors`` and never walks the hierarchy, so a
        # cap that is only ``add_children``-ed is never drawn and the cylinder
        # renders as an open tube (see Cone.__init__ for the same reason). A
        # detached cylinder's caps stay detached with it.
        registered = self._added_to_scene
        # ``grid_width`` is the tube's own count around the axis (its azimuth
        # runs on the FIRST uv component, see coord_function), so giving the
        # discs the same puts their rim vertices on the tube's. The tube's
        # tolerances go with it: the discs grow that count in whole multiples
        # of it as their rims need, measured against the accuracy the tube
        # itself was built to.
        self._mesh_key = ("solid", self.id)
        caps = {
            "rim_function": self._cap_ring_offsets,
            "scene": self.scene,
            "segments": self.grid_width,
            "color": self.color,
            "add_to_scene": registered,
            "geometry_tolerance": self._geometry_tolerance,
            "max_grid_resolution": self._max_grid_resolution,
        }
        self.bottom_cap = _CapDisc(direction=-direction, **caps)
        self.top_cap = _CapDisc(direction=direction, **caps)
        self.bottom_cap._mesh_key = self._mesh_key
        self.top_cap._mesh_key = self._mesh_key
        # Both ends sealed AND swept the whole way round closes the shell; a
        # half-pipe with whole discs is still open along its cut. The tube and
        # its caps share one surface id (``_mesh_key``), so all three carry the
        # declaration -- the renderer reads it per triangle.
        self.closed_shell = _sweep_is_full(self.v_range, 360)
        self.bottom_cap.closed_shell = self.closed_shell
        self.top_cap.closed_shell = self.closed_shell
        self.base_bottom = self.bottom_cap
        self.base_top = self.top_cap
        self.add_children(self.bottom_cap, self.top_cap)
        self._place_bases(direction, -offset, offset)
        if self.is_spawned():
            # Called after the tube was spawned, which the class docstring
            # invites. Spawn is recursive from the parent, so a cap attached
            # afterwards never gets one of its own -- and an actor that never
            # spawned is dropped by the render loop's window index, which is
            # the open tube again. Unanimated: the solid is already on screen.
            for cap in (self.bottom_cap, self.top_cap):
                cap.spawn(animate=False)
        return self

    def _place_bases(self, direction, start, end):
        """Sit the end caps on ``start`` and ``end``, each facing outward.

        Called again by :meth:`_move_between_points`, which writes ``basis``
        directly rather than through a transform -- so the children do not
        follow it, and a cap left where it was built ends up floating beside
        the tube it is supposed to close.
        """
        if getattr(self, "bottom_cap", None) is None:
            return self
        for cap, centre, outward in (
            (self.bottom_cap, start, -direction),
            (self.top_cap, end, direction),
        ):
            cap.direction = cast_to_tensor(outward)
            cap.move_to(centre)
            # Rebuilt rather than rotated: the disc's rim is this tube's own
            # ring, so re-sampling it off the live basis is what keeps the two
            # on each other -- the same thing _move_between_points does to the
            # tube one line above.
            cap.set_location_by_function(cap.coord_function)
        return self

    def _cap_ring_offsets(self, azimuth):
        """An end ring, as offsets from its centre, for the end discs.

        The same expression ``coord_function`` samples the tube's rings at --
        negation included -- swept over the whole circle whatever ``v_range``
        is, since the discs are whole circles even on a half-pipe (matching
        Manim). Read off the live basis, so it follows the tube. The basis is
        read uncopied: every consumer feeds it into out-of-place arithmetic.
        """
        basis_rows = unsquish(self.get_animated_attribute("basis", copy=False), -1, 3)
        u = -(azimuth * 2 * PI)
        return (
            u.sin() * self.radius * basis_rows[..., 0, :]
            - u.cos() * self.radius * basis_rows[..., 2, :]
        )

    def coord_function(self, uv):
        # Same uncopied-basis read as _cap_ring_offsets: this runs per grid
        # per stretch (set_location_by_function), where the clone would be a
        # dead copy -- every result feeds out-of-place arithmetic.
        basis_rows = unsquish(self.get_animated_attribute("basis", copy=False), -1, 3)
        uv[..., 1:] /= uv[..., 1:].amax()
        # ``v_range`` is Manim's azimuthal domain, but Algan's grid carries the
        # azimuth on the *first* uv component and the axial parameter on the
        # second, so the sweep is applied to ``uv[..., :1]``.  Written as one
        # negated product so that the default (0, 2*pi) reduces to the previous
        # ``-uv * pi * 2`` bit for bit: only a partial sweep moves a vertex.
        v_start, v_end = _sweep_radians(self.v_range)
        u = -(v_start + uv[..., :1] * (v_end - v_start))
        v = uv[..., 1:]
        # The ring is swept from MINUS the forward row (``INWARD``, for the
        # default ``direction=UP``) and turns toward the tube's left, which is
        # what makes ``(0, PI)`` the LEFT half. The sign is the Mob convention
        # showing through: a Mob's forward axis points ``OUTWARD``, at the
        # viewer, so a cross-section swept from ``+forward`` would start on the
        # near side and run the other way round -- reversing the (u, v)
        # handedness, hence the grid's normals and its winding, for every
        # surface of revolution built off a basis.
        return (
            u.sin() * self.radius * basis_rows[..., 0, :]
            + (v - 0.5) * self.height * basis_rows[..., 1, :]
            - u.cos() * self.radius * basis_rows[..., 2, :]
        )

    def _pn_geometry_deviation(self, pn_points, _analytic_points, _analytic_uv):
        """Exact distance from each PN sample to the cylindrical side."""
        radial, _ = _radial_and_axial_coordinates(
            pn_points,
            self.location,
            self.get_up_direction(),
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
        # Read uncopied: every result below feeds out-of-place arithmetic
        # before any write happens (the writes land in _move_between_points),
        # so the property getters' defensive clones would be dead copies on
        # the per-synapse stretch path updaters hammer.
        basis_rows = unsquish(self.get_animated_attribute("basis", copy=False), -1, 3)
        location = self.get_animated_attribute("location", copy=False)
        offset = basis_rows[..., 1, :] * 0.5
        current_end = location + offset
        current_start = location - offset
        point = current_start * (1 - interpolation) + interpolation * cast_to_tensor(
            point
        )
        self._move_between_points(point, current_end)
        return self

    @animated_function(animated_args={"interpolation": 0})
    def set_end_point(self, point, interpolation=1):
        basis_rows = unsquish(self.get_animated_attribute("basis", copy=False), -1, 3)
        location = self.get_animated_attribute("location", copy=False)
        offset = basis_rows[..., 1, :] * 0.5
        current_end = location + offset
        current_start = location - offset
        point = current_end * (1 - interpolation) + interpolation * cast_to_tensor(
            point
        )
        self._move_between_points(current_start, point)
        return self

    @animated_function(animated_args={"interpolation": 0})
    def move_between_points(self, start, end, interpolation=1):
        start = cast_to_tensor(start)
        end = cast_to_tensor(end)
        basis_rows = unsquish(self.get_animated_attribute("basis", copy=False), -1, 3)
        location = self.get_animated_attribute("location", copy=False)
        offset = (
            F.normalize(basis_rows[..., 1, :], p=2, dim=-1)
            * basis_rows.norm(p=2, dim=-1)[..., 1].unsqueeze(-1)
            * 0.5
        )
        current_end = location + offset
        current_start = location - offset
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
            # forward = right x up is the basis convention every other Mob is
            # built with (look(), the default basis): DEFAULT_BASIS is
            # right-handed, its forward being OUTWARD and RIGHT x UP being
            # OUTWARD too. get_orthonormal_vector promises orthogonality and
            # determinism but says nothing about handedness, and the vector it
            # returned here was the other one -- a mirrored frame, which turned
            # the tessellated tube inside out: a Line3D's vertex normals and
            # winding faced INWARD, so its lit side was its inside. Nothing
            # showed while the flip in _prep_normal covered it; a transparent
            # Line3D showed it plainly.
            forward_b = torch.cross(right_b, up_b, dim=-1)
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
            self._place_bases(up_b, start, end)
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
    shaft_radius
        Radius of the shaft, in world units. Defaults to ``0.02``.
    tip_length
        Length of the conical tip, in world units, measured back from ``end``.
        Defaults to ``0.3``.
    tip_radius
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
        If the distance from ``start`` to ``end`` is not greater than
        ``tip_length``, which would leave no room for a shaft.

    Examples
    --------
    An arrow pointing up and to the right:

    .. algan:: Example1Arrow3D
        :save_last_frame:

        from algan import *

        Arrow3D(start=LEFT + DOWN, end=RIGHT + UP, shaft_radius=0.04,
                color=BLUE).spawn()

        Scene.save_video()
    """

    def __init__(
        self,
        start=LEFT,
        end=RIGHT,
        shaft_radius: float = 0.02,
        tip_length: float = 0.3,
        tip_radius: float = 0.08,
        color=WHITE,
        resolution=24,
        *args,
        **kwargs,
    ):
        start = cast_to_tensor(start)
        end = cast_to_tensor(end)
        vector = end - start
        length = vector.norm(p=2, dim=-1, keepdim=True)
        if float(length.reshape(-1)[0]) <= tip_length:
            raise AlganConfigurationError(
                "Arrow3D length must be greater than its tip_length"
            )
        direction = F.normalize(vector, p=2, dim=-1)
        shaft_end = end - direction * tip_length
        super().__init__(*args, location=(start + end) * 0.5, color=color, **kwargs)
        surface_resolution = (
            (resolution, resolution) if isinstance(resolution, int) else resolution
        )
        self.tail = Cylinder(
            scene=self.scene,
            radius=shaft_radius,
            height=float((shaft_end - start).norm(p=2, dim=-1).reshape(-1)[0]),
            direction=direction,
            closed=True,
            resolution=surface_resolution,
            color=color,
            add_to_scene=False,
        )
        self.tail.move_to((start + shaft_end) * 0.5)
        self.head = Cone(
            scene=self.scene,
            radius=tip_radius,
            height=tip_length,
            direction=direction,
            closed=True,
            resolution=surface_resolution,
            color=color,
            add_to_scene=False,
        )
        self.head.move_to(end - direction * tip_length * 0.5)
        self.cone = self.head
        # Markers, not geometry: they carry no primitives and exist so that
        # get_start/get_end report the arrow's ends. Children of this arrow, and
        # in its Scene rather than whichever one happened to be active, so that
        # they travel with it -- before, a moved arrow still reported the
        # endpoints it was built with.
        self.start_point = Mob(scene=self.scene, location=start, opacity=0)
        self.end_point = Mob(scene=self.scene, location=end, opacity=0)
        self.length = length
        self.add_children(self.tail, self.head, self.start_point, self.end_point)

    #: The arrow hands the renderer the shaft, the tip and their end discs
    #: itself -- as ``get_render_primitives`` below says, none of them is a
    #: Scene actor and it is the only thing that asks them to build. So it is
    #: one unit to ``become`` as well: one morph unit, converted through the
    #: "aggregate" adapter, with its parts kept out of the actor list. Without
    #: the family it had no converter at all and ``Arrow3D().become(Sphere())``
    #: raised; without ``draws_descendants`` its parts were published twice.
    _morph_family = "aggregate"
    draws_descendants = True

    def _morph_soup_parts(self):
        return self._renderable_descendants()

    def _renderable_descendants(self):
        """This arrow's geometry, each part immediately followed by its own caps.

        Depth-first and parent-first, which is what lets a part and its caps
        share a ``_mesh_key``: only consecutive members merge.
        """
        parts = []

        def visit(mob):
            if hasattr(mob, "get_render_primitives"):
                parts.append(mob)
            for child in mob.children:
                visit(child)

        for child in self.children:
            visit(child)
        return parts

    def _get_memory_used_per_timestep(self):
        return sum(
            part._get_memory_used_per_timestep()
            for part in self._renderable_descendants()
        )

    def _get_render_device_memory_used_per_timestep(self):
        return sum(
            part._get_render_device_memory_used_per_timestep()
            for part in self._renderable_descendants()
        )

    def get_render_primitives(self):
        # The whole subtree, not just the shaft and the head: their end discs
        # are children of theirs and are not Scene actors, so this is the only
        # thing that asks them to build. Emitting each part's discs directly
        # after it is what makes their shared _mesh_key merge.
        primitives = []
        for part in self._renderable_descendants():
            primitive = part.get_render_primitives()
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
    radius
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

        Line3D(start=LEFT + DOWN, end=RIGHT + UP, radius=0.05,
               color=BLUE).spawn()

        Scene.save_video()
    """

    def __init__(
        self,
        start=torch.tensor((-1.0, 0.0, 0.0)),
        end=torch.tensor((1.0, 0.0, 0.0)),
        radius=0.02,
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
        self.radius = radius
        super().__init__(radius=radius, height=1, closed=True, **kwargs)
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
    ring_radius
        Distance from the torus's center to the center of its tube, in world
        units. Defaults to ``1.5``.
    tube_radius
        Radius of the tube itself, in world units. Defaults to ``0.5``.
    u_range
        Sweep around the ring, in degrees. ``(0, 180)`` gives half a ring.
        Defaults to ``(0, 360)``.
    v_range
        Sweep around the tube's cross-section, in degrees. ``(0, 180)`` opens
        the tube along its length. Defaults to ``(0, 360)``.
    resolution
        Manim-style grid resolution as ``(u_patches, v_patches)``, or one int
        for both. Manim counts patches where Algan's ``grid_width`` /
        ``grid_height`` count sampled vertices, so this is one *less* than the
        grid it builds -- ``resolution=(32, 32)`` gives a 33x33 grid, matching
        :class:`Sphere`, :class:`Cone` and :class:`Cylinder`. Defaults to
        ``None``, meaning Algan sizes the grid itself from
        ``geometry_tolerance``.
    **kwargs
        Passed to :class:`~algan.mobs.surfaces.surface.Surface`.

    Examples
    --------
    A ring sized to fit the frame, and half of one:

    .. algan:: Example1Torus
        :save_last_frame:

        from algan import *

        Torus(ring_radius=1.2, tube_radius=0.35, color=BLUE).spawn()
        Torus(ring_radius=1.2, tube_radius=0.35, u_range=(0, 180),
              color=YELLOW).move(UP * 0.1).spawn()

        Scene.save_video()
    """

    # Its normals face out of the tube (given the reorientation below).
    two_sided = False

    # Manim's torus parameterization is left-handed: du x dv points INTO the
    # tube, so both the vertex normals and the triangle winding came out
    # inside-out. ``coord_function`` below stays Manim's, vertex for vertex;
    # the renderer reverses the v axis instead (Surface._grid_orientation).
    _grid_orientation = -1

    def __init__(
        self,
        ring_radius=1.5,
        tube_radius=0.5,
        u_range=(0, 360),
        v_range=(0, 360),
        resolution=None,
        **kwargs,
    ):
        _reject_renamed_keywords(
            "Torus",
            kwargs,
            _SOLID_KEYWORD_RENAMES,
            manim_alternative="Torus",
            hints=_SOLID_KEYWORD_HINTS,
        )
        self.ring_radius = ring_radius
        self.tube_radius = tube_radius
        # Through the shared translator, like Sphere/Cone/Cylinder. Torus used
        # to roll its own, which read ``resolution`` as a vertex count while its
        # three siblings read it as Manim's patch count -- so the same keyword
        # built a 32x32 grid here and a 33x33 grid there. It also never
        # translated the style names those three accept.
        kwargs = _surface_resolution_kwargs(resolution, kwargs)
        # Both sweeps whole closes the tube into a ring; a partial one cuts it
        # open along the cut (a half-ring shows its inside through the slice).
        self.closed_shell = _sweep_is_full(u_range, 360) and _sweep_is_full(
            v_range, 360
        )
        super().__init__(
            coord_function=self.coord_function,
            u_range=u_range,
            v_range=v_range,
            **kwargs,
        )

    def coord_function(self, uv):
        u_start, u_end = _sweep_radians(self.u_range)
        v_start, v_end = _sweep_radians(self.v_range)
        u = u_start + uv[..., :1] * (u_end - u_start)
        v = v_start + uv[..., 1:] * (v_end - v_start)
        sweep_radius = self.ring_radius - self.tube_radius * torch.cos(v)
        return torch.cat(
            (
                sweep_radius * torch.cos(u),
                sweep_radius * torch.sin(u),
                # See Sphere.coord_function: a built-in shape's world-space z.
                self.tube_radius * torch.sin(v),
            ),
            -1,
        )

    def _pn_geometry_deviation(self, pn_points, _analytic_points, _analytic_uv):
        """Exact distance from each PN sample to this torus's surface."""
        radial, axial = _radial_and_axial_coordinates(
            pn_points,
            self.location,
            self.get_forward_direction(),
        )
        ring_radius = torch.as_tensor(
            self.ring_radius,
            device=pn_points.device,
            dtype=pn_points.dtype,
        ).abs()
        tube_radius = torch.as_tensor(
            self.tube_radius,
            device=pn_points.device,
            dtype=pn_points.dtype,
        ).abs()
        tube_distance = torch.sqrt((radial - ring_radius).square() + axial.square())
        return (tube_distance - tube_radius).abs()

    def func(self, u, v):
        u = torch.as_tensor(u)
        v = torch.as_tensor(v)
        sweep_radius = self.ring_radius - self.tube_radius * torch.cos(v)
        return torch.stack(
            (
                sweep_radius * torch.cos(u),
                sweep_radius * torch.sin(u),
                # See Sphere.coord_function: a built-in shape's world-space z.
                self.tube_radius * torch.sin(v),
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

        # Whether the faces below are known to face outward, which decides
        # whether this solid can be shaded one-sided: ``orient_faces_outward``
        # returns its input UNCHANGED (same object) for anything that is not a
        # closed orientable manifold, and for an open or non-manifold
        # Polyhedron -- both of which the public constructor accepts --
        # "outward" has no answer, so such a mob stays two-sided.
        self._faces_are_outward = False
        if rt_settings.polyhedron_winding:
            # Gated, but not because it is known to move output -- measured, the
            # fast-suite render is BYTE-IDENTICAL across this flag while
            # ALGAN_MESH_ID is off, since a per-triangle surface id makes every
            # run one fragment and the facing bit then groups nothing. With
            # mesh_id on it does move, which is the mechanism: one id per solid
            # leaves facing as the only thing separating the two sheets. ON by
            # default since DESIGN_mesh_identity.md ss3.7, and now load-bearing
            # rather than cosmetic: one-sided shading below is declared off the
            # back of it.
            oriented = orient_faces_outward(self.vertex_coords, self.faces_list)
            self._faces_are_outward = oriented is not self.faces_list
            self.faces_list = oriented
        self.vertex_indices = list(range(len(self.vertex_coords)))
        self.layout = {i: self.vertex_coords[i] for i in self.vertex_indices}
        self.face_coords = [
            [self.vertex_coords[j] for j in face] for face in self.faces_list
        ]
        self.edges = self.get_edges(self.faces_list)
        self.faces_config = dict(faces_config or {})
        self.graph_config = dict(graph_config or {})

        # A polyhedron's vertices are given in world coordinates, so nothing
        # about them says where the solid *is* -- and left to the Mob default
        # the anchor stayed at the world origin, which is what the solid then
        # turned and scaled about, from wherever it happened to sit. Anchor it
        # at the middle of its own geometry instead. An explicit ``location``
        # still wins, and every built-in solid here is already built about its
        # own centre, so this leaves ``Cube`` and ``Prism`` exactly where they
        # were.
        if "location" not in kwargs:
            kwargs["location"] = (
                self.vertex_coords.amin(-2) + self.vertex_coords.amax(-2)
            ) * 0.5
        super().__init__(**kwargs)
        # Opt-in Manim shape profile: a mapped solid adopts Manim's fill
        # defaults on its faces unless the caller styled them (fill_color /
        # fill_opacity reach faces_config through the constructors above).
        face_config_source = self.faces_config
        style = _manim_shape_style_for(type(self))
        if style is not None:
            face_config_source = dict(face_config_source)
            if style["color"] is not None:
                face_config_source.setdefault("fill_color", style["color"])
            if style["fill_opacity"] is not None:
                face_config_source.setdefault("fill_opacity", style["fill_opacity"])
        face_style = _face_style_kwargs(face_config_source, {})
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
        # The faces are triangle mobs, which are two-sided sheets on their own;
        # what makes them the SKIN of a solid is this polyhedron, so it is this
        # polyhedron that declares it -- and only when the winding pass above
        # established which way out is. The same proof says whether the faces
        # CLOSE: ``orient_faces_outward`` returns unchanged exactly for
        # geometry where "outward" has no answer (an open mesh, a T-junction, a
        # Moebius-like shell), and those must keep compositing per crossing --
        # their far side is genuinely visible through the opening.
        self.two_sided = not self._faces_are_outward
        self.closed_shell = self._faces_are_outward
        for mob in self._face_primitive_mobs():
            mob.two_sided = self.two_sided
            mob.closed_shell = self.closed_shell

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

    #: ``get_render_primitives`` below hands the renderer every face under one
    #: ``_mesh_key``, and nothing else -- so the faces are this Mob's internals
    #: rather than Mobs of their own, and the vertex ``Dot3D``s and edge Mobs
    #: under ``self.graph`` (kept for Manim parity, where ``graph_config``
    #: styles them) are geometry it owns but never puts on screen. Both facts
    #: matter to :meth:`~algan.animatable_base.mob_morph.MobMorphMixin.become`:
    #: without this flag a morphed Polyhedron grew a wireframe and eight vertex
    #: beads a spawned one does not have, and drew each of its faces twice.
    draws_descendants = True

    def owned_subtrees(self):
        """The faces this Polyhedron draws and the graph it declines to draw.

        Both are its own construction and neither is a Mob in its own right.
        Anything else below it is a user's, and stays a Scene actor -- speaking
        for a user's child too would make their geometry vanish from a morph.
        """
        return [self.faces, self.graph]

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

    def _get_render_device_memory_used_per_timestep(self):
        return sum(
            mob._get_render_device_memory_used_per_timestep()
            for mob in self._face_primitive_mobs()
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
    width
        Side length along the world x axis, in world units. Defaults to ``3``.
    height
        Side length along the world y axis, in world units. Defaults to ``2``.
    depth
        Side length along the world z axis, in world units. Defaults to ``1``.
        The box is centered on the Mob's location, so each side extends half its
        length either way.
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

        Prism(width=3, height=0.2, depth=2, fill_color=BLUE).spawn()

        Scene.save_video()
    """

    def __init__(self, width=3, height=2, depth=1, **kwargs):
        from algan.utils.tensor_utils import cast_to_tensor

        self.width = width
        self.height = height
        self.depth = depth
        x, y, z = cast_to_tensor((width, height, depth)).reshape(-1) / 2
        # Near face first, far face second: a box is the same solid mirrored in
        # z, but which pair of corners each quad is split along is not, so the
        # table is written in Algan's z convention (OUTWARD is +z) to keep the
        # triangulation the mirror of what it was.
        vertices = [
            [-x, -y, z],
            [x, -y, z],
            [x, y, z],
            [-x, y, z],
            [-x, -y, -z],
            [x, -y, -z],
            [x, y, -z],
            [-x, y, -z],
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
    size
        Length of each edge, in world units. Defaults to ``2``.
    fill_opacity
        Opacity of the faces, from ``0`` (invisible) to ``1`` (opaque). Defaults
        to ``0.75``, Manim's value -- a cube is slightly see-through unless you
        say otherwise.
    fill_color
        Color of the faces: an Algan :class:`~algan.constants.color.Color`, a
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

        cube = Cube(size=1.2, fill_color=BLUE, fill_opacity=1).spawn()
        cube.rotate(35, UP)

        Scene.save_video()
    """

    def __init__(
        self,
        size=2,
        fill_opacity=0.75,
        fill_color=None,
        stroke_width=0,
        **kwargs,
    ):
        from algan.constants.color import BLUE

        self.size = size
        if fill_color is None:
            fill_color = BLUE
        super().__init__(
            width=size,
            height=size,
            depth=size,
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
            # The z coordinates are the mirror of the classical table. A
            # tetrahedron is the one Platonic solid that is *chiral* under
            # z -> -z (its vertex set is the even-sign-count corners of the
            # cube, and mirroring lands on the odd ones), so it is the only one
            # whose table has to carry Algan's z convention: the others mirror
            # onto themselves and are the same solid either way. Face winding
            # is not adjusted here because ``orient_faces_outward`` fixes it.
            [
                [unit, unit, -unit],
                [unit, -unit, unit],
                [-unit, unit, unit],
                [-unit, -unit, -unit],
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
            # z negated against the classical table, as in Tetrahedron: the
            # vertex set is symmetric in z, so this is a permutation that leaves
            # the solid alone and puts vertex k at the mirror of the point the
            # classical table calls k -- which keeps triangle ORDER mirrored too.
            [
                [unit, 0, 0],
                [-unit, 0, 0],
                [0, unit, 0],
                [0, -unit, 0],
                [0, 0, -unit],
                [0, 0, unit],
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
        # z negated against the classical table, as in Tetrahedron.
        vertices = [
            [0, b, -a],
            [0, -b, -a],
            [0, b, a],
            [0, -b, a],
            [b, a, 0],
            [b, -a, 0],
            [-b, a, 0],
            [-b, -a, 0],
            [a, 0, -b],
            [a, 0, b],
            [-a, 0, -b],
            [-a, 0, b],
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
        # z negated against the classical table, as in Tetrahedron: a
        # dodecahedron's vertex SET is symmetric in z, so this is a permutation
        # and the solid is unchanged, but it puts vertex k at the mirror of the
        # point the classical table calls k -- which is what makes the fan
        # triangulation of its pentagons follow Algan's z convention.
        vertices = [
            [a, a, -a],
            [a, a, a],
            [a, -a, -a],
            [a, -a, a],
            [-a, a, -a],
            [-a, a, a],
            [-a, -a, -a],
            [-a, -a, a],
            [0, c, -b],
            [0, c, b],
            [0, -c, b],
            [0, -c, -b],
            [c, b, 0],
            [-c, b, 0],
            [c, -b, 0],
            [-c, -b, 0],
            [b, 0, -c],
            [-b, 0, -c],
            [b, 0, c],
            [-b, 0, c],
        ]
        # Already outward-wound for the mirrored table above, and written
        # starting at the vertex each face's fan is cut from, so the
        # pentagons keep their diagonals. Manim's own lists are neither
        # (see orient_faces_outward, which repairs them).
        faces = [
            [1, 12, 0, 16, 18],
            [3, 18, 16, 2, 14],
            [3, 10, 9, 1, 18],
            [1, 9, 5, 13, 12],
            [12, 13, 4, 8, 0],
            [2, 16, 0, 8, 11],
            [4, 17, 6, 11, 8],
            [4, 13, 5, 19, 17],
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
            raise AlganConfigurationError(
                "ConvexHull3D requires at least four non-coplanar points"
            )
        hull = ConvexHull(array, qhull_options=f"QJ{tolerance}")
        vertex_ids = sorted({int(i) for i in hull.simplices.reshape(-1)})
        remap = {old: new for new, old in enumerate(vertex_ids)}
        vertices = [array[i].tolist() for i in vertex_ids]
        faces = [[remap[int(i)] for i in simplex] for simplex in hull.simplices]
        # Canonical order, independent of the order Qhull happened to emit its
        # simplices in. Qhull's ordering is a function of the input coordinates,
        # so the same hull built from mirrored points comes back listed
        # differently -- and the face order reaches the renderer as triangle
        # order, which moves pixels at the seams between coplanar faces. Each
        # face is rotated to start at its lowest index (winding preserved;
        # ``orient_faces_outward`` settles the direction), and the list is
        # ordered by each face's vertex SET rather than by the rotated tuple:
        # mirroring a hull reverses every face's winding, which would otherwise
        # move faces past each other in the sort and undo the canonicalisation.
        faces = [
            face[face.index(min(face)) :] + face[: face.index(min(face))]
            for face in faces
        ]
        faces.sort(key=sorted)
        super().__init__(vertices, faces, **kwargs)
