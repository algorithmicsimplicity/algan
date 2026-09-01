"""Cubic bezier circuits -- the geometry behind every 2-D shape.

:class:`BezierCircuitCubic` is a closed loop of cubic bezier curves, stored as
control points. It is what :class:`~algan.mobs.shapes_2d.Circle`,
:class:`~algan.mobs.shapes_2d.Square`, :class:`~algan.mobs.text.Text` and
:class:`~algan.mobs.text.Tex` are made of, which is why a circle is a true circle
at any zoom rather than a many-sided polygon, and why any of them can morph into
any other.

The class owns the fill/border model as well as the geometry: on a filled shape
the border is drawn *inside* the outline, so raising ``stroke_width`` eats into
the fill instead of growing the silhouette -- which keeps bordered text legible
and stops neighbouring glyphs fusing. An unfilled circuit has no interior to eat
into, so its stroke stays centred on the path.

It owns color *across* the shape too. Since there are no vertices to hang
colors off, a circuit carries a ``texture_grid_width`` x
``texture_grid_height`` grid of color samples laid over its own frame, which
the renderer samples bilinearly per fragment.
:meth:`BezierCircuitCubic.set_color_by_function` fills the grid in from ``(u,
v)`` -- the same domain :class:`~algan.mobs.surfaces.surface.Surface` uses --
:meth:`BezierCircuitCubic.set_color_by_image` from a picture, and
:meth:`~algan.mobs.shapes_2d.Line.set_color_by_function` from a single ``t``
along the path. The grid is one texel, i.e. one flat color, unless it was asked
for.

``build_render_primitives_batched`` packs many circuits into one
:class:`~algan.rendering.primitives.bezier_circuit_primitive.BezierCircuitPrimitive`
for the renderer, which evaluates the curves analytically rather than
tessellating them.
"""

from __future__ import annotations

import math

import torch.nn.functional as F

from algan.animatable_base.animatable import animated_function
from algan.animatable_base.mob import Mob
from algan.animation_timeline.animation_contexts import Off, Sync
from algan.constants.color import *
from algan.constants.spatial import OUTWARD, RIGHT, UP
from algan.geometry.geometry import rotate_vector_around_axis
from algan.mobs.nonplanar_circuit import (
    build_render_primitives as build_nonplanar_render_primitives,
)
from algan.mobs.nonplanar_circuit import classify_circuit
from algan.rendering.mps_compat import cummax_values
from algan.rendering.raytracing.utils import _unify_time
from algan.settings.renderer_settings import RENDERER_REGISTRY
from algan.settings.video_settings import PREVIEW
from algan.utils.mob_utils import pack_animatable_rows, pack_member_rows
from algan.utils.tensor_utils import *

# Three.js's fixed dielectric F0 = 0.04 corresponds to IOR 1.5; MeshStandard
# has no ``ior`` of its own, so that is the default a circuit falls back to.
DIELECTRIC_IOR = 1.5

# Ceiling on the texture-grid size ``wave_color`` will refine a circuit to. A
# wave tighter than this can afford is drawn as smoothly as the budget allows.
_MAX_WAVE_TEXTURE_RESOLUTION = 64


def _stroke_width_in_render_pixels(stroke_width, video_settings):
    """Convert an authored ``stroke_width`` to the renderer's stroke width.

    ``stroke_width`` is authored against PREVIEW's frame height so a border keeps
    its apparent weight at any resolution.  The renderer wants the FULL stroke
    width in pixels: a filled circuit lays it inside the outline, an unfilled one
    centres it on the path (``_circuit_point_region``).
    """
    return stroke_width * video_settings.resolution[1] / PREVIEW.resolution[1]


def _circuit_ior(ior, metalness):
    """Pack a material's IOR into a circuit's transport channel.

    Mirrors the triangle primitive's ``_derive_material_surface_params``: an
    unsigned magnitude feeding dielectric F0. Whether the circuit transmits is
    carried by the separate ``transmission`` channel, not by this one's sign.
    Non-PBR circuits (metalness < 0) get 0: inert, since their reflectance is 0
    anyway.
    """
    return torch.where(metalness >= 0.0, ior.abs(), torch.zeros_like(ior))


def _resample_texture_grid(value, old_size, new_size):
    """Resample per-texel circuit values between two ``(width, height)`` texture
    grids, for every circuit packed into ``value`` ``[..., N * old, C]``.

    Texels are stored with the width (first basis) axis outermost, so the packed
    rows reshape straight into a ``[width, height]`` image.
    """
    leading = value.shape[:-2]
    channels = value.shape[-1]
    image = value.reshape(-1, *old_size, channels).permute(0, 3, 1, 2)
    resized = F.interpolate(
        image, size=tuple(new_size), mode="bilinear", align_corners=True
    )
    return resized.permute(0, 2, 3, 1).reshape(*leading, -1, channels)


def _extremal_control_point_index(dists, relative_tolerance=0.0):
    """Index of the control point furthest from a circuit's centre, ties broken
    toward the lowest index.

    ``torch.argmax`` does not promise which of several equal maxima it hands
    back, and a circuit's control points tie constantly: every corner of a
    square is the same distance from its centre, and so is every point of a
    circle. Since the winner sets the whole local frame -- including the sign of
    the plane normal -- an unspecified tie-break is an unspecified basis, which
    is exactly what this avoids.

    ``relative_tolerance`` widens "equal" to a fraction of the maximum, for the
    collinear case where the two ends of a path are equidistant in exact
    arithmetic and an ulp apart in practice.
    """
    dists = dists.reshape(-1)
    threshold = dists.amax()
    if relative_tolerance:
        threshold = threshold * (1.0 - relative_tolerance)
    inds = torch.arange(dists.numel(), device=dists.device)
    return int(torch.where(dists >= threshold, inds, dists.numel()).amin())


def _circuit_location_and_basis(control_points):
    """Return the same local frame used by a standalone bezier circuit, plus
    whether its second in-plane axis had to be synthesized.

    Row 2 is the circuit's plane normal. Rows 0 and 1 span that plane and always
    share a length, so the frame is orthogonal with a square in-plane footprint
    -- which is what the texture grid, laid out along those two rows, relies on.

    Within the plane the rows are aligned to the world axes, not to the
    circuit's own geometry. They used to point at the control point furthest
    from the centre, i.e. at a *corner*: a ``Rectangle(4, 1)`` came out with row
    0 = ``(-2, 0.5, 0)``. Since a shape-``(*, 3)`` factor to
    :meth:`~.Mob.scale` scales the Mob's own right, up and forward axes, that
    made ``rect.scale([4, 1, 1])`` stretch along the diagonal and render the
    rectangle as a parallelogram. Row lengths are unchanged by the alignment, so
    ``scale_coefficient`` and anything derived from it (``Circle.radius``) keep
    their old values; only the in-plane rotation of the frame moves.

    A straight Line is the exception and keeps the geometry-derived frame: its
    control points are collinear, so there is no plane to align to, and the
    extremal displacement genuinely is the shape's own axis. Row 0 is pinned to
    the START of such a path -- see :class:`~algan.mobs.shapes_2d.Line`, whose
    documented guarantee this is.

    Returns
    -------
    tuple
        ``(location, basis, second_axis_synthesized)``: the circuit's centre,
        its flattened 3x3 frame, and whether the control points were collinear
        so that row 1 carries no extent of the shape's own.
    """
    control_points = control_points.reshape(-1, 3)
    mn = control_points.amin(-2)
    mx = control_points.amax(-2)
    location = (mn + mx) * 0.5
    if (mx - mn).norm(p=2, dim=-1) <= 1e-6:
        basis = squish(
            torch.eye(3, device=control_points.device, dtype=control_points.dtype)
        )
        return location, basis.reshape(-1), True

    disps = control_points - location
    centre_dists = disps.norm(p=2, dim=-1, keepdim=True)
    first_basis = disps[_extremal_control_point_index(centre_dists)].unsqueeze(-2)
    if first_basis.norm(p=2, dim=-1) <= 1e-4:
        first_basis = RIGHT.to(control_points) * 1e-4
    first_basis_n = F.normalize(first_basis, p=2, dim=-1)

    planar_disps = disps - dot_product(disps, first_basis_n) * first_basis_n
    dists = planar_disps.norm(p=2, dim=-1, keepdim=True)
    second_basis = planar_disps[_extremal_control_point_index(dists)].unsqueeze(-2)
    degenerate = bool(second_basis.norm(p=2, dim=-1).max() <= 1e-4)
    if degenerate:
        # Collinear control points: no plane to derive a second axis from, so
        # row 1 is synthesized perpendicular to row 0 and carries none of the
        # shape's own extent. Re-pick row 0 with a tolerance, so that the two
        # ends of a path -- equidistant from its centre in exact arithmetic,
        # an ulp apart in practice -- always resolve to the lowest-indexed
        # control point, i.e. the start.
        first_basis = disps[
            _extremal_control_point_index(centre_dists, 1e-6)
        ].unsqueeze(-2)
        first_basis_n = F.normalize(first_basis, p=2, dim=-1)
        # Clockwise about OUTWARD, so that the negated cross below lands on
        # OUTWARD here too: a straight path's face is the same face a closed
        # one's is.
        second_basis = rotate_vector_around_axis(first_basis, -90, OUTWARD, -1)
    scale = first_basis.norm(p=2, dim=-1, keepdim=True)
    second_basis = second_basis * scale / second_basis.norm(p=2, dim=-1, keepdim=True)
    # NEGATED, which is what makes a flat shape face the viewer. Row 2 is the
    # face the circuit presents, and ``cross(row 0, row 1)`` follows the order
    # the control points were authored in: every 2-D shape Algan ships is wound
    # so that it comes out INWARD, which would leave a Square stating that it
    # faces away from the camera it was drawn in front of, while
    # ``DEFAULT_BASIS`` says a Mob faces OUTWARD. The sign belongs here rather
    # than in the shapes because it is the *frame's* convention, not the paths':
    # a path's direction is drawn (``Create``) and interpolated (``become``),
    # and reversing one to fix a normal would move all of that.
    third_basis_n = -F.normalize(
        broadcast_cross_product(first_basis_n, second_basis), p=2, dim=-1
    )

    if not degenerate:
        # Swing the in-plane pair round to the world axes, keeping the plane and
        # both row lengths exactly as derived above. Row 0 takes whichever world
        # axis the plane admits, preferring x; row 1 follows from the plane's
        # orientation, so on a plane whose normal faces the camera it comes out
        # along +y -- an upright shape's own up is UP, which is what
        # ``wave_color`` means by "bottom to top". Only one of the three world
        # axes can be parallel to the normal, so the loop always settles.
        for reference in (RIGHT, UP, OUTWARD):
            candidate = reference.to(third_basis_n) - (
                dot_product(reference.to(third_basis_n), third_basis_n) * third_basis_n
            )
            if bool(candidate.norm(p=2, dim=-1).min() > 1e-4):
                first_basis_n = F.normalize(candidate, p=2, dim=-1)
                first_basis = first_basis_n * scale
                # cross(first, cross(third, first)) == third, so the frame keeps
                # the handedness the plane normal was computed with.
                second_basis = (
                    broadcast_cross_product(third_basis_n, first_basis_n) * scale
                )
                break

    basis = torch.cat((first_basis, second_basis, third_basis_n), -1)
    return location, basis.reshape(-1), degenerate


class BezierCircuitCubic(Mob):
    """A closed loop of cubic bezier curves -- the geometry every 2-D shape is
    made of.

    The curves are evaluated analytically by the renderer rather than
    tessellated, so a circuit stays exactly smooth at any zoom, and any circuit
    can :meth:`~algan.animatable_base.mob.Mob.become` any other. On a filled
    circuit the border is drawn *inside* the outline, so raising ``stroke_width``
    eats into the fill instead of growing the silhouette; an unfilled circuit has
    no interior to eat into, so its stroke stays centred on the path.

    **Color across a circuit.** A circuit carries a rectangular texture grid of
    color samples, laid across its own frame and sampled bilinearly per
    fragment by the renderer. ``texture_grid_width`` x ``texture_grid_height``
    is therefore the resolution of everything painted on the shape -- a
    gradient, an image, a color wave -- and it defaults to a single texel, i.e.
    one flat color. Raise it and fill it in with
    :meth:`~.BezierCircuitCubic.set_color_by_function` or
    :meth:`~.BezierCircuitCubic.set_color_by_image`.

    The grid's ``(u, v)`` domain is the circuit's own frame, exactly as
    :class:`~algan.mobs.surfaces.surface.Surface`'s is: ``u`` runs from 0 to 1
    along the first basis row and ``v`` along the second, which for an upright
    2-D shape means ``u`` left to right and ``v`` bottom to top. Both rows are as
    long as the distance from the centre to the furthest control point, so the
    frame spans the square that circumscribes the shape and the shape itself
    covers the middle of the domain rather than all of it.

    Parameters
    ----------
    control_points
        The cubic bezier control points, shape ``(*, 3)`` in world units, in
        groups of four: ``P0, P1, P2, P3`` per segment, with each segment
        starting where the previous one ended. A segment that starts somewhere
        else begins a new sub-circuit, which is how a shape gets holes.
    normals
        Per-control-point normals, shape ``(*, 3)``, used for lighting. Defaults
        to ``None``, meaning the circuit's own plane normal is used.
    stroke_width
        Width of the border stroke, in pixels against a 960-pixel-tall frame so
        it keeps its apparent weight at any resolution. Defaults to ``5``; pass
        ``0`` for no border.
    stroke_color
        Color of the border stroke. Defaults to ``WHITE``. The circuit's
        ``color`` is its *fill* color and does not touch the border; see
        :attr:`~.BezierCircuitCubic.stroke_color`.
    portion_of_curve_drawn
        How much of the path is drawn, from 0 (nothing) to 1 (all of it).
        Animating it is what draws a shape on. Defaults to ``1.0``.
    filled
        Whether the interior is painted. Defaults to ``True``; ``False`` leaves
        an outline whose stroke is centred on the path (what
        :class:`~algan.mobs.shapes_2d.Line` uses).
    add_texture_grid
        Whether to build the texture grid at all. Defaults to ``True``. ``False``
        leaves the circuit one color and no per-texel storage, and the
        ``set_color_by_*`` methods then have nothing to write to.
    texture_grid_width
        Number of color samples along the circuit's first basis row -- ``u``,
        left to right on an upright shape. Defaults to ``1``: one flat color,
        which is what a shape wants unless you are painting something across it.
    texture_grid_height
        Number of color samples along the second basis row (``v``). Defaults to
        ``None``, meaning match ``texture_grid_width`` -- except on a circuit
        whose control points are collinear (a straight
        :class:`~algan.mobs.shapes_2d.Line`), where the second row is synthesized
        perpendicular to the path and carries no extent of the shape, so it
        defaults to ``1`` and the grid runs along the line only.
    empty
        Whether the circuit is invisible: fill and border are forced to zero
        opacity. Defaults to ``False``. Used for shapes that exist only to
        position or morph into something else.
    z_index
        Which of two *exactly coplanar* circuits draws in front: the higher
        ``z_index`` wins. Defaults to ``0``, which leaves the shape in author
        order -- coplanar 2-D geometry draws in the order it was created, each
        composite Mob kept whole and drawn parent-first, so an arrow crossing a
        grid authored before it lands on top of that grid without being asked
        to. Raise it to override that: a label over a panel authored after it,
        a highlight over the shape it marks. Setting it propagates to the whole
        sub-hierarchy (see :attr:`~.BezierCircuitCubic.z_index`).

        It is *not* a general depth override. The renderer spends it as a bias
        of a few ten-thousandths of a world unit toward the camera -- enough to
        settle a tie between surfaces at the same depth, far too little to
        reorder anything genuinely in front of or behind. Values are small
        integers; a few hundred would start to shift the shape visibly. Matches
        Manim's attribute of the same name, both in meaning and in being a
        stable sort key over the authored order, and
        :class:`~algan.mobs.manim_mob.ManimMob` carries it across on import.
    **kwargs
        Passed to :class:`~algan.animatable_base.mob.Mob` -- notably ``color``,
        which is the fill color.

    See Also
    --------
    :meth:`~.BezierCircuitCubic.set_color_by_function` : Color it by a function of ``(u, v)``.
    :meth:`~.BezierCircuitCubic.set_color_by_image` : Paint an image across it.
    :class:`~algan.mobs.surfaces.surface.Surface` : The 3-D counterpart, with the same ``(u, v)`` conventions.

    Examples
    --------
    .. algan:: Example1BezierCircuitCubic
        :save_last_frame:

        from algan import *
        import torch

        square = Square(texture_grid_width=32, texture_grid_height=32, stroke_width=0)
        square.set_color_by_function(
            lambda uv: torch.cat(
                (uv[..., :1], uv[..., 1:], torch.zeros_like(uv[..., :1])), -1
            )
        )
        square.spawn()

        Scene.save_video()
    """

    _morph_family = "bezier"

    # Plain scalar, deliberately not timeline-backed: it selects between
    # coplanar draw orders rather than describing a pose, and animating it
    # would only ever step between discrete orderings. The class default keeps
    # the property readable on a part-built Mob (``ManimCompatMob.__getattr__``
    # would otherwise forward the miss to the backing Manim object).
    _z_index = 0.0

    # Set per batch by ``RenderLoopMixin._authored_draw_order``: the whole draw
    # order resolved to depth bins, of which the authored ``z_index`` is one
    # input. ``None`` means no render has resolved one, and a primitive built
    # directly still honours ``z_index`` on its own.
    _draw_bias = None

    def _render_draw_bias(self):
        """Depth-bin bias this circuit renders with."""
        return self.z_index if self._draw_bias is None else self._draw_bias

    @property
    def z_index(self):
        """Which of two exactly coplanar circuits draws in front (higher wins).

        ``0`` (the default) means author order, which already keeps a composite
        Mob whole and parent-first; this is the override for when that is not
        what you want. Assigning propagates to every circuit below this one in
        the hierarchy, matching Manim's ``set_z_index(..., family=True)``, so a
        composite raises as one thing rather than leaving its parts on opposite
        sides of whatever they cross.

        Animation
        ---------
        Takes effect immediately and is not animated: it selects between
        discrete orderings, so there is nothing to interpolate and no context
        (``Seq``, ``Sync``, ``Off``) changes how it applies. The write reaches
        every circuit in this Mob's sub-hierarchy; plain Mobs in between, such
        as a circuit's texture points, have no draw order and are skipped. It
        may be set before or after :meth:`~.Animatable.spawn` -- the renderer
        reads it afresh for every frame batch.
        """
        return self._z_index

    @z_index.setter
    def z_index(self, value):
        value = float(value)
        self._z_index = value
        # ``children`` is absent while the base Mob is still initializing, and
        # a circuit's texture-point children are plain Mobs with no draw order
        # of their own -- both are skipped rather than special-cased.
        pending = list(getattr(self, "children", None) or ())
        while pending:
            mob = pending.pop()
            if isinstance(mob, BezierCircuitCubic):
                mob._z_index = value
            pending.extend(getattr(mob, "children", None) or ())

    def __init__(
        self,
        control_points,
        normals=None,
        stroke_width=5,
        stroke_color=WHITE,
        portion_of_curve_drawn=1.0,
        filled=True,
        add_texture_grid=True,
        texture_grid_width=1,
        texture_grid_height=None,
        empty=False,
        z_index=0,
        **kwargs,
    ):
        self.num_bezier_parameters = 4
        self.z_index = z_index
        control_points = control_points.view(-1, control_points.shape[-1])

        kwargs2 = dict(kwargs.items())

        if "color" in kwargs2:
            # Parsed here rather than in Mob.__init__: this runs first, and it
            # indexes the value's shape, so a hex string or an RGB tuple has to
            # already be a color by now.
            color = to_color(kwargs2["color"])
            kwargs2["color"] = color.reshape(-1, color.shape[-1]).mean(-2)
        if normals is not None:
            normals = normals.reshape(-1, 3)
        (
            kwargs2["location"],
            kwargs2["basis"],
            second_axis_synthesized,
        ) = _circuit_location_and_basis(control_points)

        # Decided once, here, from the authored control points: a circuit whose
        # sub-paths do not lie in planes cannot be rendered by projecting them
        # onto one (see algan.mobs.nonplanar_circuit). The plan is topology
        # only -- the geometry it describes is rebuilt from the live control
        # points every render batch -- but the choice itself is fixed, exactly
        # as the plane in ``basis`` above is.
        self._nonplanar_plan = classify_circuit(control_points, filled)

        self.grid_width = self.grid_height = 1
        self.num_texture_points = 0
        first_basis = kwargs2["basis"][..., :3]
        second_basis = kwargs2["basis"][..., 3:6]
        self.first_basis = first_basis
        self.second_basis = second_basis

        super().__init__(**kwargs2)
        kwargs["scene"] = self.scene
        self.register_attrs_as_animatable(
            ["stroke_width", "portion_of_curve_drawn"],
            BezierCircuitCubic,
        )
        self.filled = filled
        self.empty = empty
        if self.empty:
            self.color = self.color.as_subclass(Color).set_opacity(0)

        texture_triangle_vertices = self.location.squeeze(0)
        if add_texture_grid:
            width = max(int(texture_grid_width), 1)
            if texture_grid_height is None:
                # A collinear circuit's second basis row is synthesized
                # perpendicular to the path, so every point of the shape maps to
                # the same v: sampling it more than once buys nothing.
                height = 1 if second_axis_synthesized else width
            else:
                height = max(int(texture_grid_height), 1)

            # ``linspace(-1, 1, 1)`` is -1, so a single-sample axis puts its one
            # texel at that end of the frame rather than in the middle. The
            # renderer clamps the whole axis to it either way, so it is one
            # color across the span regardless; what it does change is where
            # ``wave_color`` reads the texel's position from.
            a1 = torch.linspace(-1, 1, width).view(-1, 1, 1) * (1 + 1e-5)
            a2 = torch.linspace(-1, 1, height).view(1, -1, 1) * (1 + 1e-5)
            texture_grid_points = (a1 * first_basis + a2 * second_basis) + self.location
            texture_triangle_vertices = texture_grid_points
            self.grid_width = width
            self.grid_height = height
            texture_triangle_vertices = texture_triangle_vertices.reshape(
                -1, texture_triangle_vertices.shape[-1]
            )
            self.num_texture_points = texture_triangle_vertices.shape[-2]

            # control_points = torch.cat((control_points, texture_triangle_vertices), -2)
        self.stroke_width = cast_to_tensor(stroke_width)
        stroke_color = cast_to_tensor(stroke_color)
        if self.empty:
            stroke_color = stroke_color.as_subclass(Color).set_opacity(0)

        fill_texture_kwargs = dict(kwargs)
        fill_texture_kwargs["color"] = self.color if self.filled else stroke_color
        self.texture_points = Mob(texture_triangle_vertices, **fill_texture_kwargs)
        self.texture_points.exclude_from_boundary = True
        self.texture_points.is_primitive = True
        self.add_children(self.texture_points)

        border_texture_kwargs = dict(kwargs)
        border_texture_kwargs["color"] = stroke_color
        self.border_texture_points = Mob(
            texture_triangle_vertices, **border_texture_kwargs
        )
        self.border_texture_points.exclude_from_boundary = True
        self.border_texture_points.is_primitive = True
        # ``color`` on the circuit means fill color.  The border grid remains a
        # child so it follows transforms and participates in waves/cloning, but
        # it must not inherit ordinary fill-color writes from an ancestor.
        self.border_texture_points._excluded_from_parent_attrs = frozenset({"color"})
        self.add_children(self.border_texture_points)

        self.control_points = Mob(control_points, **fill_texture_kwargs)
        self.control_points.is_primitive = True
        self.add_children(self.control_points)
        self.control_points.num_points_per_object = 4
        self.components = [
            self.texture_points,
            self.border_texture_points,
            self.control_points,
        ]

        self.normals = normals
        self.is_primitive = True
        self.render_primitive = RENDERER_REGISTRY.bezier_circuit_primitive

    def _after_repack(self):
        """Re-decide the planar/patch/stroke split against the whole pack.

        ``batch_mobs`` clones its first member and then writes every member's
        control points in, so the plan made at construction describes one tile
        of what is now a whole sphere. Classification is per sub-path and the
        pack's sub-paths are its members', so redoing it here reaches the same
        decision the members reached individually -- which is what
        ``ManimMob(..., batch=True)`` relies on.
        """
        self._nonplanar_plan = classify_circuit(
            self.control_points.location.reshape(-1, 3), self.filled
        )

    @classmethod
    def from_batches(cls, control_point_batches, *args, **kwargs):
        """Build many independently indexable circuits without per-circuit mobs.

        ``control_point_batches`` contains one cubic-bezier tensor per logical
        object.  Geometry is concatenated once while ``parent_batch_sizes``
        retains the control-point boundaries used by rendering and indexed
        views.
        """
        batches = [
            cast_to_tensor(points).reshape(-1, 3) for points in control_point_batches
        ]
        if not batches:
            raise ValueError("from_batches requires at least one bezier circuit")
        point_counts = torch.tensor(
            [len(points) for points in batches], dtype=torch.long
        )
        if bool((point_counts % 4 != 0).any()):
            raise ValueError(
                "every cubic bezier circuit must contain a multiple of 4 points"
            )

        mob = cls(torch.cat(batches, -2), *args, **kwargs)
        locations, bases, _ = zip(
            *[_circuit_location_and_basis(points) for points in batches]
        )
        locations = torch.stack(locations, -2).unsqueeze(0)
        bases = torch.stack(bases, -2).unsqueeze(0)
        count = len(batches)

        with Off(
            record_funcs=False,
            record_attr_modifications=False,
            animation_manager=mob.animation_manager,
        ):
            pack_animatable_rows(
                mob, count, overrides={"location": locations, "basis": bases}
            )

            texture_point_count = max(mob.num_texture_points, 1)
            grid_locations = (
                torch.linspace(
                    -1,
                    1,
                    mob.grid_width,
                    device=locations.device,
                    dtype=locations.dtype,
                ).view(1, 1, -1, 1, 1)
                * (1 + 1e-5)
                * bases[..., :3].unsqueeze(-2).unsqueeze(-2)
                + torch.linspace(
                    -1,
                    1,
                    mob.grid_height,
                    device=locations.device,
                    dtype=locations.dtype,
                ).view(1, 1, 1, -1, 1)
                * (1 + 1e-5)
                * bases[..., 3:6].unsqueeze(-2).unsqueeze(-2)
                + locations.unsqueeze(-2).unsqueeze(-2)
            ).reshape(1, count * texture_point_count, 3)
            for texture_mob in (
                mob.texture_points,
                mob.border_texture_points,
            ):
                pack_member_rows(
                    texture_mob,
                    count,
                    texture_point_count,
                    overrides={"location": grid_locations},
                )

            mob.control_points.parent_batch_sizes = point_counts
        return mob

    def _refine_sampling_for_color_wave(self, direction, max_spacing, pulsed_attrs):
        """Refine the texture grids so a color wave crosses the shape smoothly.

        A circuit's fill and border are colored by bilinearly sampling the
        independent ``texture_points`` and ``border_texture_points`` grids laid
        across it. Those grids are a single sample unless
        ``texture_grid_width`` / ``texture_grid_height`` were raised by hand, so
        a shape flashes as one flat color instead of showing the wave
        travelling over it. Lay down a grid fine enough that neighbouring
        samples are no further than ``max_spacing`` apart along the wave (see
        :meth:`~.Mob._refine_sampling_for_color_wave`).

        A circuit whose grid is already rectangular was sized deliberately by
        its author, so it is left exactly as it is rather than being squared up
        for the duration of a wave.
        """
        if "color" not in pulsed_attrs:
            # Only color is stored per texture point. A circuit's opacity is a
            # single shader parameter for the whole fill, so an opacity wave --
            # the fade Text and Tex spawn with, for one -- cannot be made any
            # smoother by adding texels.
            return None
        if self.empty:
            return None
        size = self.grid_width
        if (
            self.num_texture_points < 1
            or size != self.grid_height
            or self.data_sub_inds is not None
            or self.texture_points.data_sub_inds is not None
            or self.border_texture_points.data_sub_inds is not None
        ):
            return None
        objects = self.location.shape[-2]
        expected_points = objects * self.num_texture_points
        if (
            self.texture_points.location.shape[-2] != expected_points
            or self.border_texture_points.location.shape[-2] != expected_points
        ):
            return None

        # The grid spans the full width of the circuit's own frame along each of
        # its two basis vectors, so each axis covers twice the projected length
        # of its basis vector, with ``size`` samples over it.
        def projected_span(basis):
            return 2 * dot_product(direction, basis, dim=-1).abs().amax().item()

        span = max(
            projected_span(self.basis[..., :3]), projected_span(self.basis[..., 3:6])
        )
        spacing = span if size < 2 else span / (size - 1)
        if not span > 0 or spacing <= max_spacing:
            return None
        required = int(math.ceil(span / max_spacing)) + 1
        new_size = max(size, min(required, _MAX_WAVE_TEXTURE_RESOLUTION))
        if new_size == size:
            return None
        self._set_texture_grid_resolution(new_size, new_size)

        def restore():
            self._set_texture_grid_resolution(size, size)

        return restore

    def _set_texture_grid_resolution(self, width, height):
        """Internal: rebuild the texture grid at ``width x height`` per circuit.

        The grid is laid out exactly as the constructor lays it out, so the
        renderer's texture lookup is unchanged: ``width`` samples along the first
        basis vector, each holding ``height`` samples along the second. Existing
        per-texel values are resampled onto the new grid. Row counts change, so
        callers must be prepared for the history split this performs.
        """
        old_size = (self.grid_width, self.grid_height)
        old_points = self.num_texture_points
        objects = self.location.shape[-2]
        old_values = {}
        for texture_mob in (self.texture_points, self.border_texture_points):
            values = {}
            for attr in dict.fromkeys(texture_mob.animatable_attrs):
                try:
                    values[attr] = getattr(texture_mob, attr).clone()
                except AttributeError:
                    continue
            old_values[texture_mob] = values

        # A different number of texture points cannot be interpolated from the
        # old ones, so the recorded history stays behind on a frozen clone.
        if self.is_spawned():
            self.detach_history()
        self.grid_width, self.grid_height = width, height
        self.num_texture_points = width * height

        location = self.location

        def offsets(size):
            return torch.linspace(
                -1, 1, size, device=location.device, dtype=location.dtype
            ) * (1 + 1e-5)

        first = self.basis[..., :3].unsqueeze(-2).unsqueeze(-2)
        second = self.basis[..., 3:6].unsqueeze(-2).unsqueeze(-2)
        points = (
            offsets(width).view(-1, 1, 1) * first
            + offsets(height).view(1, -1, 1) * second
            + location.unsqueeze(-2).unsqueeze(-2)
        )
        new_locations = points.reshape(*points.shape[:-4], -1, 3)
        for texture_mob in (self.texture_points, self.border_texture_points):
            texture_mob._setattr_and_rebatch_without_record("location", new_locations)

            # Every attribute stored one value per texture point has to follow
            # the new grid; otherwise later writes can no longer broadcast.
            for attr, value in old_values[texture_mob].items():
                if attr == "location" or value.shape[-2] != objects * old_points:
                    continue
                texture_mob._setattr_and_rebatch_without_record(
                    attr, _resample_texture_grid(value, old_size, (width, height))
                )

            if texture_mob.parent_batch_sizes is not None:
                texture_mob.parent_batch_sizes = torch.full(
                    (objects,),
                    self.num_texture_points,
                    dtype=texture_mob.parent_batch_sizes.dtype,
                )
            texture_mob.batch_size = objects * self.num_texture_points
        self._memory_per_timestep_cache = None
        return self

    def get_animatable_attrs(self):
        return {"stroke_width"}.union(super().get_animatable_attrs())

    #: ``filled`` and ``empty`` decide whether the circuit is a disc or a ring.
    #: ``get_render_primitives`` reads both live and neither is animatable, so a
    #: filled Square becoming an unfilled one used to stay solid -- a full-range
    #: difference over 3.6% of the frame. The colors that go with the fill
    #: arrive separately: ``texture_points`` and ``border_texture_points`` are
    #: components, and the morph recurses into them. ``z_index`` is deliberately
    #: absent -- it already reaches the endpoint on its own, and assigning it
    #: here would bypass the setter that propagates it to the sub-hierarchy.
    _MORPH_ADOPTED_ATTRS = (
        *Mob._MORPH_ADOPTED_ATTRS,
        "filled",
        "empty",
    )

    #: Both of them are also untravellable, and ``filled`` is the sharpest case
    #: of it in the package: the flag does not merely hide the interior, it
    #: decides where the stroke goes (a filled circuit lays its border INWARD
    #: from the outline, an unfilled one centres it on the path -- see
    #: ``_circuit_point_region``), so no value of anything animatable
    #: interpolates between the two. A pair that crosses it cross-fades.
    _MORPH_UNTRAVELLABLE_ATTRS = (
        *Mob._MORPH_UNTRAVELLABLE_ATTRS,
        "filled",
        "empty",
    )

    @property
    def stroke_color(self):
        """Per-vertex colors sampled across the circuit's border texture grid."""
        return self.border_texture_points.color

    @stroke_color.setter
    def stroke_color(self, value):
        self.border_texture_points.color = value

    def get_base_grid(self) -> torch.Tensor:
        """Get the circuit's texture grid, the ``(u, v)`` domain it is colored
        over.

        Values run from 0 to 1 along both axes: ``u`` along the circuit's first
        basis row, ``v`` along its second, which on an upright 2-D shape means
        ``u`` left to right and ``v`` bottom to top. Both rows are as long as the
        distance from the circuit's centre to its furthest control point, so the
        domain covers the square that circumscribes the shape and the shape sits
        in the middle of it. An axis with a single sample carries one color for the whole span
        and is evaluated at its centre, ``0.5``.

        This is the input the ``set_color_by_*`` methods evaluate their
        functions over, so it is what to write those functions in terms of.

        Returns
        -------
        torch.Tensor
            The ``(u, v)`` coordinates, shape
            ``[texture_grid_width, texture_grid_height, 2]``.

        See Also
        --------
        :meth:`~.BezierCircuitCubic.set_color_by_function` : Color the circuit over this grid.
        """
        device = self.texture_points.location.device

        def axis(size):
            if size < 2:
                return torch.full((1,), 0.5, device=device)
            return torch.linspace(0, 1, size, device=device)

        width, height = self.grid_width, self.grid_height
        return torch.stack(
            (
                axis(width).view(-1, 1).expand(-1, height),
                axis(height).view(1, -1).expand(width, -1),
            ),
            -1,
        )

    def _apply_texture_grid_colors(self, colors, what):
        """Internal: write one color per texel onto the grids that are visible.

        ``colors`` holds one entry per texel of :meth:`get_base_grid`, in that
        grid's own layout. The fill grid always takes them; an unfilled circuit
        has no interior to show them in, so its border grid takes them too --
        the same pairing the constructor makes when it hands an unfilled
        circuit's texture grids the border color.
        """
        colors = Color.add_defaults(cast_to_tensor(colors))
        colors = colors.reshape(-1, colors.shape[-1])
        if colors.shape[-2] != self.num_texture_points:
            raise ValueError(
                f"{what} must return one color per texel: expected "
                f"{self.num_texture_points} "
                f"({self.grid_width} x {self.grid_height}), got {colors.shape[-2]}"
            )
        objects = self.location.shape[-2]
        if objects > 1:
            # ``from_batches`` mobs (Text, Tex) pack every circuit's texels into
            # one row block each, and every circuit is colored over its own
            # frame, so the same grid of colors repeats per circuit.
            colors = colors.repeat(objects, 1)
        targets = [self.texture_points]
        if not self.filled:
            targets.append(self.border_texture_points)
        with Sync(animation_manager=self.animation_manager):
            for target in targets:
                target.color = colors.unsqueeze(0)
        return self

    def _require_texture_grid(self, method):
        """Internal: refuse to paint a circuit that has nowhere to paint."""
        if self.num_texture_points < 2:
            raise ValueError(
                f"{type(self).__name__}.{method} needs a texture grid with more "
                "than one texel, but this circuit has "
                f"{self.num_texture_points}. The grid is the resolution of "
                "anything painted across the shape, and it is one flat color "
                "by default -- construct the shape with e.g. "
                "texture_grid_width=64, texture_grid_height=64."
            )

    def set_color_by_function(self, function):
        """Color the circuit by a function of its ``(u, v)`` parameters.

        Gives each texel of the circuit's texture grid its own color, for
        gradients, heat maps or anything where color carries data, and the
        renderer interpolates between them across the shape. The colors travel
        with the circuit as it moves and morphs.

        The grid is the resolution of the result, and it is a single flat color
        unless you asked for more: build the shape with ``texture_grid_width`` /
        ``texture_grid_height`` (see :class:`~.BezierCircuitCubic`). On a filled
        circuit this colors the fill, leaving ``stroke_color`` alone; on an
        unfilled one, where the stroke is all there is, it colors the stroke.
        A multi-circuit mob (a :class:`~algan.mobs.text.Text`, a
        :class:`~algan.mobs.text.Tex`) colors every circuit over its own frame,
        so the pattern repeats per glyph.

        Animation
        ---------
        Recorded as an animation over the current context's duration (1 second
        by default), so the colors cross-fade smoothly. Wrap the call in
        ``Off()`` to apply it instantly.

        Parameters
        ----------
        function
            Callable taking a ``(u, v)`` tensor of shape ``[..., 2]``, with both
            coordinates in ``[0, 1]``, and returning colors of shape
            ``[..., 3]`` (RGB), ``[..., 4]`` (RGBA) or ``[..., 5]`` (RGB, glow,
            alpha -- Algan's internal channel order). Channels are in ``[0, 1]``;
            a missing alpha defaults to 1 and a missing glow to 0. Must be
            vectorized -- it is called once on the whole grid, not per texel.

        Returns
        -------
        :class:`~.BezierCircuitCubic`
            This circuit, so calls can be chained.

        Raises
        ------
        ValueError
            If the circuit has a single-texel texture grid, or if ``function``
            returns the wrong number of colors.

        See Also
        --------
        :meth:`~.BezierCircuitCubic.set_color_by_image` : Paint an image on instead.
        :meth:`~.BezierCircuitCubic.get_base_grid` : The ``(u, v)`` grid this evaluates over.

        Examples
        --------
        .. algan:: Example1BezierCircuitCubicSetColorByFunction
            :save_last_frame:

            from algan import *
            import torch

            circle = Circle(texture_grid_width=48, texture_grid_height=48)
            circle.set_color_by_function(
                lambda uv: torch.cat(
                    (uv[..., :1], torch.zeros_like(uv[..., :1]), uv[..., 1:]), -1
                )
            )
            circle.spawn()

            Scene.save_video()
        """
        self._require_texture_grid("set_color_by_function")
        return self._apply_texture_grid_colors(
            function(self.get_base_grid().clone()), "set_color_by_function's function"
        )

    def set_color_by_image(self, rgba_array_or_file_path):
        """Paint an image across the circuit.

        The image is resampled onto the circuit's texture grid and interpolated
        across the shape by the renderer, and it follows the shape as it moves
        and morphs. The image's top-left corner lands at the top left of the
        frame, which on an upright 2-D shape is ``(u, v) == (0, 1)``: ``v`` runs
        up the frame, as it does on a
        :class:`~algan.mobs.surfaces.surface.Surface`, while an image's rows run
        down the picture.

        Unlike :meth:`~algan.mobs.surfaces.surface.Surface.set_color_by_image`,
        which keeps the image at its own resolution, a circuit has no separate
        texture map: the texture grid *is* the resolution, so build the shape
        with a ``texture_grid_width`` / ``texture_grid_height`` matching the
        detail you need. Remember too that the grid spans the square
        circumscribing the shape, so the shape shows the middle of the picture.

        Animation
        ---------
        Recorded as an animation over the current context's duration (1 second
        by default): the circuit cross-fades, texel by texel, to the image. Wrap
        the call in ``Off()`` to apply it instantly.

        Parameters
        ----------
        rgba_array_or_file_path
            Path to an image file, or an RGBA array of shape ``[H, W, 4]`` or
            ``[H, W, 5]`` with channels in ``[0, 1]``. Paths resolve relative to
            the working directory and then the main script's directory, so an
            image beside your script is found either way.

        Returns
        -------
        :class:`~.BezierCircuitCubic`
            This circuit, so calls can be chained.

        Raises
        ------
        ValueError
            If the circuit has a single-texel texture grid.

        See Also
        --------
        :meth:`~.BezierCircuitCubic.set_color_by_function` : Color it by a function instead.
        :class:`~algan.mobs.image_mob.ImageMob` : An image as a Mob of its own, at full resolution.
        """
        self._require_texture_grid("set_color_by_image")
        from algan.utils.file_utils import get_image

        image = get_image(rgba_array_or_file_path)
        # ``image`` is [row, column, channel] with rows running DOWN the
        # picture; the grid is [u, v] with v running UP the circuit's frame, so
        # the resample lands on (v, u), transposes back, and flips v -- which is
        # the same flip ``mesh.image_to_texture_map`` does for a surface, for the
        # same reason. Without it the picture arrives upside down: the contract
        # is that its top-left corner lands at the top left of the frame, not
        # that its first row lands at v == 0.
        resized = F.interpolate(
            image.permute(2, 0, 1).unsqueeze(0),
            (self.grid_height, self.grid_width),
            mode="bilinear",
            antialias=True,
        ).squeeze(0)
        return self._apply_texture_grid_colors(
            resized.permute(2, 1, 0).flip(1), "set_color_by_image's image"
        )

    def get_default_color(self):
        return PURPLE

    def _get_memory_used_per_timestep(self):
        # Called for every circuit every render batch just to size batches;
        # the shape reads below go through the animated-attribute machinery,
        # so cache the result against the global structure version (row
        # re-allocation bumps it).
        from algan.animation_timeline.timeline import STRUCTURE_VERSION

        cache = getattr(self, "_memory_per_timestep_cache", None)
        if cache is not None and cache[0] == STRUCTURE_VERSION[0]:
            return cache[1]
        n_ctrl = self.control_points.location.shape[-2]
        n_tex = self.texture_points.location.shape[-2]
        n_border_tex = self.border_texture_points.location.shape[-2]
        n_loc = self.location.shape[-2]
        n_segments = max(n_ctrl // 4, 1)  # cubic beziers have 4 control points each
        # Animation state: control points (3 floats), two color textures (5
        # each), location/basis (6). Texture positions are structural sampling
        # data used by wave animation and are charged alongside their colors.
        animation_bytes = (
            n_ctrl * 3 + (n_tex + n_border_tex) * (3 + 5) + n_loc * 6
        ) * 4
        # Primitive output: control points, fill texture, border texture, and
        # per-circuit normals/border data.
        primitive_bytes = (
            n_segments * 4 * 3 * 4 + (n_tex + n_border_tex) * 5 * 4 + n_loc * 12
        )
        # Sampled edges, metadata and the content-dependent STBVH are charged
        # exactly by the final scene upload instead of guessed here (the old
        # fixed 100-sample estimate was wrong for the actual 1..512 range).
        result = int(animation_bytes + primitive_bytes)
        self._memory_per_timestep_cache = (STRUCTURE_VERSION[0], result)
        return result

    def get_render_primitives(self):
        if self.empty:
            return None
        # Derive transport directly from the material shader parameters. A
        # negative metalness sentinel marks non-PBR materials; Standard and
        # Physical materials expose metalness/roughness as animatable attrs.
        surface_template = self.opacity[..., :1]

        def material_param(name, default):
            if name in self.animatable_attrs:
                return getattr(self, name)
            return torch.full_like(surface_template, default)

        metalness = material_param("metalness", -1.0)
        roughness = material_param("roughness", 0.0)
        # Opacity is coverage and transmission is transparency: independent
        # channels, never folded together (see _derive_material_surface_params).
        transmission = material_param("transmission", 0.0).clamp(0.0, 1.0)
        ior = _circuit_ior(material_param("ior", DIELECTRIC_IOR), metalness)

        shader_vars = broadcast_all(
            [
                self.opacity,
                self.basis,
                self.glow,
                _stroke_width_in_render_pixels(
                    self.stroke_width, self.scene.video_settings
                ),
                metalness,
                roughness,
                ior,
                transmission,
            ],
            ignored_dims=[-1],
        )
        num_control_points = 4  # cubic beziers
        if self._nonplanar_plan is not None:
            # Not projectable onto one plane: this circuit renders as PN patches
            # and/or per-run circuits built from the same live control points.
            return build_nonplanar_render_primitives(
                self,
                unsquish(self.control_points.location, -2, num_control_points),
                self.texture_points.get_animated_attribute("color"),
                self.border_texture_points.get_animated_attribute("color"),
                *shader_vars[:1],
                *shader_vars[2:],
            )
        # Read the color rows as plain tensors. ``mob.color`` hands back a
        # :class:`~algan.constants.color.Color` so callers get its rgb / glow /
        # opacity views, but a Tensor subclass routes *every* subsequent
        # operation through ``__torch_function__``, and building one batch's
        # circuits performs tens of thousands of them. Nothing below this point
        # wants the views -- only the numbers.
        if self.control_points.parent_batch_sizes is None:
            return self._get_render_primitives(
                unsquish(self.control_points.location, -2, num_control_points),
                self.texture_points.get_animated_attribute("color"),
                self.border_texture_points.get_animated_attribute("color"),
                self.location,
                self.basis,
                *shader_vars,
            )
        x = self.control_points.location
        tpc = self.texture_points.get_animated_attribute("color")
        border_tpc = self.border_texture_points.get_animated_attribute("color")
        num_segments_per_circuit = (
            self.control_points.parent_batch_sizes // num_control_points
        )
        return self._get_render_primitives(
            unsquish((x), -2, num_control_points),
            (tpc),
            (border_tpc),
            self.location,
            self.basis,
            *shader_vars,
            num_segments_per_circuit,
        )

    def _get_render_primitives(
        self,
        x,
        tpc,
        border_tpc,
        loc,
        basis,
        o,
        n,
        g,
        bw,
        reflectivity,
        roughness,
        refractive_index,
        transmission,
        num_segments_per_circuit=None,
    ):
        # x = unsquish(x, -2, num_control_points)
        # assert x.shape == [*, N, num_control_points, 3], where N is number of bezier segments.
        start_points = x[..., :1, :]
        end_points = x[..., -1:, :]

        # We allow for rendering circuits with holes,
        # we treat beziers which don't start at the previous one's end as marking the start of a new circuit (i.e. a hole).
        circuit_start_mask = (start_points - end_points.roll(1, -3)).norm(
            p=2, dim=-1, keepdim=True
        ) > 1e-5
        circuit_end_mask = (end_points - start_points.roll(-1, -3)).norm(
            p=2, dim=-1, keepdim=True
        ) > 1e-5

        inds = torch.arange(x.shape[-3], device=x.device).view(-1, 1, 1)
        circuit_start_inds = torch.where(circuit_start_mask, inds, 0)
        circuit_start_inds = cummax_values(circuit_start_inds, -3)
        # circuit_start_inds now contains the index of the start of the current index's circuit.

        next_segment_inds = (inds + 1) % x.shape[-3]
        # If the current ind is the end of the circuit, then the next segment is the first ind of this circuit, otherwise it is the next ind.
        next_segment_inds = torch.where(
            circuit_end_mask, circuit_start_inds, next_segment_inds
        )
        # We subtract inds so that each ind is represented as an offset from the current ind.
        # This way, we can concatenate together offsets from different objects, and then just add a torch.arange during rendering
        # to recover the index in the new concatenated tensor.
        next_segment_inds_offset = next_segment_inds - inds

        if num_segments_per_circuit is None:
            starting_inds = circuit_start_mask[0, :, 0, 0].nonzero()[:, 0]
            num_segments_per_circuit = []
            if len(starting_inds) == 0:
                num_segments_per_circuit.append(
                    torch.tensor(
                        (circuit_start_mask.shape[-3],),
                        device=next_segment_inds.device,
                        dtype=next_segment_inds.dtype,
                    ).squeeze()
                )
            else:
                for i in range(len(starting_inds)):
                    num_segments_per_circuit.append(
                        (
                            starting_inds[(i + 1)]
                            if (i + 1) < len(starting_inds)
                            else circuit_start_mask.shape[-3]
                        )
                        - starting_inds[i]
                    )
            # num_segments_per_circuit = torch.stack(num_segments_per_circuit, 0)
            num_segments_per_circuit = torch.tensor(
                [x.shape[-3]], device=x.device, dtype=torch.long
            )
            c = tpc.unsqueeze(-3)
            border_c = border_tpc.unsqueeze(-3)
            texture_point_count = max(self.num_texture_points, 1)
            if texture_point_count > c.shape[-2]:
                c = c.expand([-1, -1, texture_point_count, -1])
            if texture_point_count > border_c.shape[-2]:
                border_c = border_c.expand([-1, -1, texture_point_count, -1])
        else:
            texture_point_count = max(self.num_texture_points, 1)
            c = unsquish(tpc, -2, texture_point_count)
            border_c = unsquish(border_tpc, -2, texture_point_count)

        prim = self.render_primitive(
            x,
            next_segment_inds_offset,
            num_segments_per_circuit,
            c,
            o,
            basis[..., -3:],
            bw,
            border_c,
            loc,
            cast_to_tensor(self.grid_width).expand(-1, loc.shape[1], -1),
            cast_to_tensor(self.grid_height).expand(-1, loc.shape[1], -1),
            basis[..., :3],
            basis[..., 3:6],
            glow=g,
            num_texture_points=self.num_texture_points,
            filled=self.filled,
            reflectivity=reflectivity,
            roughness=roughness,
            refractive_index=refractive_index,
            transmission=transmission,
            z_index=(
                None
                if (bias := self._render_draw_bias()) == 0.0
                else torch.full(
                    (1, bw.shape[-2], 1), bias, dtype=bw.dtype, device=bw.device
                )
            ),
        )
        prim.num_texture_points = self.num_texture_points
        # A circuit casts a shadow like anything else, so it honours
        # Mob.casts_shadows; receives_shadows is accepted and inert here
        # (2-D geometry renders unlit and receives no shadow to begin
        # with) -- see the primitive's declare_shadow_flags.
        prim.declare_shadow_flags(*self._resolved_shadow_flags())
        return prim

    @animated_function(animated_args={"t": 0.0})
    def draw(self, t=1.0):
        self._original_control_points = self.control_points.location.clone()
        num_frames = self.control_points.location.shape[0]
        total_control_points = self._original_control_points.shape[-2]
        points = self._original_control_points.expand(num_frames, -1, -1)

        if self.control_points.parent_batch_sizes is not None:
            num_mobs = len(self.control_points.parent_batch_sizes)
        else:
            num_mobs = 1

        num_control_points_per_mob = total_control_points // num_mobs
        N_per_mob = num_control_points_per_mob // 4

        # Reshape points to (num_frames, num_mobs, N_per_mob, 4, 3)
        points_reshaped = points.view(num_frames, num_mobs, N_per_mob, 4, 3)

        # Ensure t is a tensor and has shape (num_frames, num_mobs, 1, 1)
        t = cast_to_tensor(t).to(points.device)
        while t.dim() < 3:
            t = t.unsqueeze(0)
        if t.shape[1] != num_mobs:
            t = t.expand(-1, num_mobs, -1)
        t = t.unsqueeze(-1)  # (num_frames, num_mobs, 1, 1)

        # Calculate local b parameters
        inds_local = torch.arange(N_per_mob, device=points.device, dtype=points.dtype)
        b = (N_per_mob * t - inds_local.view(1, 1, N_per_mob, 1)).clamp(
            0.0, 1.0
        )  # (num_frames, num_mobs, N_per_mob, 1, 1)

        # Portion matrix coefficients for each segment
        mb = 1.0 - b
        b2 = b * b
        mb2 = mb * mb
        b3 = b2 * b
        mb3 = mb2 * mb

        # Construct portion_matrix of shape (num_frames, num_mobs, N_per_mob, 4, 4)
        portion_matrix = torch.zeros(
            (num_frames, num_mobs, N_per_mob, 4, 4),
            device=points.device,
            dtype=points.dtype,
        )
        portion_matrix[..., 0, 0] = 1.0

        portion_matrix[..., 1, 0] = mb.squeeze(-1)
        portion_matrix[..., 1, 1] = b.squeeze(-1)

        portion_matrix[..., 2, 0] = mb2.squeeze(-1)
        portion_matrix[..., 2, 1] = 2.0 * mb.squeeze(-1) * b.squeeze(-1)
        portion_matrix[..., 2, 2] = b2.squeeze(-1)

        portion_matrix[..., 3, 0] = mb3.squeeze(-1)
        portion_matrix[..., 3, 1] = 3.0 * mb2.squeeze(-1) * b.squeeze(-1)
        portion_matrix[..., 3, 2] = 3.0 * mb.squeeze(-1) * b2.squeeze(-1)
        portion_matrix[..., 3, 3] = b3.squeeze(-1)

        # Compute new control points
        new_points = torch.matmul(portion_matrix, points_reshaped)

        # Reshape back to (num_frames, total_control_points, 3)
        new_points = new_points.view(num_frames, total_control_points, 3)

        # Set the control points location absolute
        self.control_points.location = new_points
        return self

    def set_control_points_to_partial(self, full_control_points, start_t, end_t):
        full_control_points = cast_to_tensor(full_control_points)
        start_t = cast_to_tensor(start_t).to(full_control_points)
        end_t = cast_to_tensor(end_t).to(full_control_points)

        def frame_values(value, name):
            if value.numel() == 1:
                return value.reshape(1)
            values = value.reshape(value.shape[0], -1)
            if values.shape[1] != 1:
                raise ValueError(f"{name} must contain one value per frame")
            return values[:, 0]

        start_t = frame_values(start_t, "start_t")
        end_t = frame_values(end_t, "end_t")
        num_frames = max(full_control_points.shape[0], start_t.numel(), end_t.numel())
        if full_control_points.shape[0] == 1:
            full_control_points = full_control_points.expand(num_frames, -1, -1)
        elif full_control_points.shape[0] != num_frames:
            raise ValueError(
                "full_control_points must have one row or one row per frame"
            )
        if start_t.numel() == 1:
            start_t = start_t.expand(num_frames)
        elif start_t.numel() != num_frames:
            raise ValueError("start_t must have one value per frame")
        if end_t.numel() == 1:
            end_t = end_t.expand(num_frames)
        elif end_t.numel() != num_frames:
            raise ValueError("end_t must have one value per frame")

        total_control_points = full_control_points.shape[-2]

        if self.control_points.parent_batch_sizes is not None:
            num_mobs = len(self.control_points.parent_batch_sizes)
        else:
            num_mobs = 1

        num_control_points_per_mob = total_control_points // num_mobs
        N_per_mob = num_control_points_per_mob // 4

        points_reshaped = full_control_points.view(
            num_frames, num_mobs, N_per_mob, 4, 3
        )

        j = torch.arange(
            N_per_mob,
            device=full_control_points.device,
            dtype=full_control_points.dtype,
        ).view(1, 1, N_per_mob, 1, 1)
        s_start = j / N_per_mob
        s_end = (j + 1) / N_per_mob

        a = torch.clamp(start_t.view(-1, 1, 1, 1, 1), min=s_start, max=s_end)
        b = torch.clamp(end_t.view(-1, 1, 1, 1, 1), min=s_start, max=s_end)

        local_a = (a - s_start) * N_per_mob
        local_b = (b - s_start) * N_per_mob

        P0 = points_reshaped[..., 0, :]
        P1 = points_reshaped[..., 1, :]
        P2 = points_reshaped[..., 2, :]
        P3 = points_reshaped[..., 3, :]

        b_t = local_b.squeeze(-1)
        mb_t = 1.0 - b_t

        Q0 = P0
        Q1 = mb_t * P0 + b_t * P1
        Q2 = mb_t**2 * P0 + 2.0 * mb_t * b_t * P1 + b_t**2 * P2
        Q3 = (
            mb_t**3 * P0
            + 3.0 * mb_t**2 * b_t * P1
            + 3.0 * mb_t * b_t**2 * P2
            + b_t**3 * P3
        )

        u = torch.where(b_t > 1e-6, local_a.squeeze(-1) / b_t, torch.zeros_like(b_t))
        u = torch.clamp(u, 0.0, 1.0)
        mu = 1.0 - u

        R3 = Q3
        R2 = u * Q3 + mu * Q2
        R1 = u**2 * Q3 + 2.0 * u * mu * Q2 + mu**2 * Q1
        R0 = u**3 * Q3 + 3.0 * u**2 * mu * Q2 + 3.0 * u * mu**2 * Q1 + mu**3 * Q0

        new_points = torch.stack([R0, R1, R2, R3], -2).view(
            num_frames, total_control_points, 3
        )
        self.control_points.location = new_points
        return self


class BezierCurveCubic(BezierCircuitCubic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, filled=False, **kwargs)


def build_render_primitives_batched(actors, scene):
    """Build the merged (collection-level) bezier render primitive for
    ``actors`` in one vectorized pass.

    Byte-identical replacement for calling ``get_render_primitives()`` on
    every actor and concatenating the per-actor primitives through
    ``BezierCircuitPrimitive(triangle_collection=...)``: each attribute is
    read from its timeline once for the whole group (contiguous rows read as
    a single slice), and the per-segment circuit topology (subpath start/end
    masks, next-segment indices) is computed with per-actor index maps that
    reproduce each actor's local ``roll``/``cummax`` wrap-around semantics.

    Callers must guarantee (see ``RenderLoopMixin._is_batchable_bezier`` and
    ``_build_deferred_beziers``): stock ``BezierCircuitCubic`` build methods,
    not ``empty``, un-batched control points, singleton rows for the scalar
    attributes, and uniform ``num_texture_points`` / ``filled`` / fill and
    border texture-color row counts / primitive class across the group.
    """
    from algan.animation_timeline.timeline import RowRanges
    from algan.rendering.primitives.bezier_circuit_primitive import (
        chord_tolerance_pixels,
    )

    timeline = scene.timeline_manager
    first = actors[0]
    ntp = first.num_texture_points
    M = len(actors)

    def read(attr, mobs):
        tl = timeline.attr_to_timeline[attr]
        # Merge the per-mob cached [begin, end) runs (ranges_for) instead of
        # rebuilding them from the index tensors: this is called every frame
        # batch, and tensor->int conversion per mob dominates otherwise.
        pairs = []
        for m in mobs:
            r = tl.ranges_for(m.id)
            if r.pairs is None:  # non-contiguous rows (defensive)
                return tl.get(
                    RowRanges(
                        None,
                        tensor=torch.cat([tl.mob_id_to_inds[mm.id] for mm in mobs]),
                    )
                )
            for b, e in r.pairs:
                if pairs and pairs[-1][1] == b:
                    pairs[-1] = (pairs[-1][0], e)
                else:
                    pairs.append((b, e))
        return tl.get(RowRanges(pairs))

    # --- batched attribute reads (mirrors the per-actor property reads and
    # the ``vars`` broadcast in get_render_primitives) ---
    o = read("opacity", actors)

    def read_optional_material(attr, default):
        values = []
        for actor in actors:
            if attr in actor.animatable_attrs:
                tl = timeline.attr_to_timeline[attr]
                values.append(tl.get(tl.ranges_for(actor.id)))
            else:
                values.append(torch.full_like(o[:, :1, :1], default))
        values, _ = _unify_time(values, f"bezier {attr} merge")
        return torch.cat(values, 1)

    reflectivity = read_optional_material("metalness", -1.0)
    roughness = read_optional_material("roughness", 0.0)
    # Opacity is coverage, transmission is transparency: independent channels
    # (see _derive_material_surface_params). ``o`` is left alone.
    transmission = read_optional_material("transmission", 0.0).clamp(0.0, 1.0)
    refractive_index = _circuit_ior(
        read_optional_material("ior", DIELECTRIC_IOR), reflectivity
    )
    basis = read("basis", actors)
    g = read("glow", actors)
    bw = _stroke_width_in_render_pixels(
        read("stroke_width", actors), scene.video_settings
    )
    loc = read("location", actors)
    o, basis, g, bw = broadcast_all([o, basis, g, bw], ignored_dims=[-1])
    cp = read("location", [a.control_points for a in actors])
    tpc = read("color", [a.texture_points for a in actors])
    border_tpc = read("color", [a.border_texture_points for a in actors])

    # --- circuit topology (mirrors _get_render_primitives) ---
    loc_inds = timeline.attr_to_timeline["location"].mob_id_to_inds
    seg_counts = torch.tensor(
        [loc_inds[a.control_points.id].numel() // 4 for a in actors], dtype=torch.long
    )
    x = unsquish(cp, -2, 4)  # [T, S_total, 4, 3]
    S_tot = x.shape[-3]
    seg_offsets = seg_counts.cumsum(0) - seg_counts
    mob_of_seg = torch.repeat_interleave(torch.arange(M), seg_counts)
    off_of_seg = seg_offsets[mob_of_seg]
    gidx = torch.arange(S_tot)
    local = gidx - off_of_seg
    last_local = seg_counts[mob_of_seg] - 1

    start_points = x[..., :1, :]
    end_points = x[..., -1:, :]
    # Per-actor wrap-around neighbours: each actor's own roll(+-1, -3).
    prev_idx = torch.where(local == 0, off_of_seg + last_local, gidx - 1)
    next_idx = torch.where(local == last_local, off_of_seg, gidx + 1)
    circuit_start_mask = (start_points - end_points.index_select(-3, prev_idx)).norm(
        p=2, dim=-1, keepdim=True
    ) > 1e-5
    circuit_end_mask = (end_points - start_points.index_select(-3, next_idx)).norm(
        p=2, dim=-1, keepdim=True
    ) > 1e-5

    local_col = local.view(-1, 1, 1)
    off_col = off_of_seg.view(-1, 1, 1)
    # The per-actor where(mask, local_ind, 0) + cummax scan, run in global
    # index space: candidate values are per-actor monotone blocks (every
    # actor's candidates are >= its offset and below the next actor's), so
    # one global cummax restarts cleanly at every actor boundary.
    circuit_start_inds = torch.where(circuit_start_mask, local_col + off_col, off_col)
    circuit_start_inds = cummax_values(circuit_start_inds, -3) - off_col
    next_segment_inds = torch.where(
        local == last_local, torch.zeros_like(local), local + 1
    ).view(-1, 1, 1)
    next_segment_inds = torch.where(
        circuit_end_mask, circuit_start_inds, next_segment_inds
    )
    next_segment_inds_offset = next_segment_inds - local_col  # [T, S, 1, 1]

    # --- texture colors (mirrors the ``c`` construction) ---
    texture_point_count = max(ntp, 1)

    def texture_colors(values):
        colors = unsquish(values, -2, values.shape[-2] // M)
        if texture_point_count > colors.shape[-2]:
            colors = colors.expand([-1, -1, texture_point_count, -1])
        return colors

    c = texture_colors(tpc)  # [T, M, P, 5]
    bc = texture_colors(border_tpc)

    # --- per-primitive color/border math (mirrors
    # BezierCircuitPrimitive.__init__'s scalar path) ---
    normals = basis[..., -3:]
    colors, fill_opacity, fill_glow = broadcast_all(
        [c, o.unsqueeze(-2), g.unsqueeze(-2)], ignored_dims=[-1]
    )
    colors = colors.clone()
    colors[..., -2:-1] += fill_glow
    colors[..., -1:] *= fill_opacity
    bc, border_opacity, border_glow = broadcast_all(
        [bc, o.unsqueeze(-2), g.unsqueeze(-2)], ignored_dims=[-1]
    )
    bc = bc.clone()
    bc[..., -2:-1] += border_glow
    bc[..., -1:] *= border_opacity

    # --- collection-level assembly (mirrors the triangle_collection branch
    # of BezierCircuitPrimitive.__init__) ---
    # Keep the deferred mega-primitive on the materialized animation/source
    # device.  The prefetch worker must not upload the next batch while the
    # current one occupies the render device; upload happens at the managed
    # render-memory boundary.
    device = x.device
    cls = first.render_primitive
    mega = cls.__new__(cls)
    # The ray tracer interprets this legacy density setting as the maximum
    # screen-space curve-to-chord error in pixels. It must be the value
    # BezierCircuitPrimitive's constructor defaults to, because this builder's
    # whole contract is to be a byte-identical replacement for that
    # constructor: it stood at 1 against the per-actor path's 0.5, which the
    # default analytic-AA route hides (it clamps the tolerance to
    # analytic_aa_chord_tolerance = 0.25, so 0.5 and 1 both land on 0.25) and
    # the classic supersampled route does not -- there every batched circuit
    # was flattened to twice the per-actor path's chord error. Harmless while
    # the batched build reached a fifth of a scene's circuits; not harmless now
    # that a group clash no longer sends the rest down the other path (P9).
    mega.num_pixels_per_sample = chord_tolerance_pixels
    mega.num_bezier_parameters = 4
    mega.num_texture_points = ntp
    mega.filled = first.filled
    mega.num_segments_per_object = seg_counts.to(device)
    mega.corners = x.to(device)
    cols = colors.to(device)
    if ntp == 0:
        cols = cols.squeeze(-2)
    mega.next_segment_inds = next_segment_inds_offset.to(device) + torch.arange(
        S_tot, device=device
    ).view(-1, 1, 1)
    mega.normals = normals.to(device)
    mega.stroke_width = bw.to(device)
    mega.stroke_color = bc.to(device)

    T = loc.shape[0]

    def per_actor_int(vals):
        return (
            torch.tensor([float(v) for v in vals]).view(1, M, 1).int().expand(T, -1, -1)
        )

    # ``_is_batchable_bezier`` guarantees one attribute row -- and therefore one
    # circuit -- per actor here, so the lane is simply the actors' scalars.
    zs = [float(a._render_draw_bias()) for a in actors]
    mega._has_z_index = any(z != 0.0 for z in zs)
    mega.z_index = (
        torch.tensor(zs, dtype=bw.dtype).view(1, M, 1).to(device)
        if mega._has_z_index
        else torch.zeros((1, M, 1), dtype=bw.dtype, device=device)
    )

    mega.mob_center = loc.to(device)
    mega.grid_width = per_actor_int([a.grid_width for a in actors]).to(device)
    mega.grid_height = per_actor_int([a.grid_height for a in actors]).to(device)
    mega.basis1 = basis[..., :3].to(device)
    mega.basis2 = basis[..., 3:6].to(device)
    mega.reflectivity = reflectivity.to(device)
    mega.roughness = roughness.to(device)
    mega.refractive_index = refractive_index.to(device)
    mega.transmission = transmission.to(device)
    if ntp > 0:
        cols = cols[..., -ntp:, :]
    mega.colors = cols
    return mega
