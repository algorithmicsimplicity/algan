"""The :class:`Mob` -- anything that can appear on screen.

A Mob is an :class:`~algan.animatable_base.animatable.Animatable` with a place in
3-D space: a ``location``, a ``basis`` (its orientation and scale as three
vectors), a ``color``, an ``opacity`` and a ``glow``. Assigning to any of them
records an animation.

The class is assembled from mixins, each in its own ``mob_*.py`` module:
movement and rotation, screen-relative layout, the parent/child hierarchy,
morphing between shapes (``become``), and the shader and material API. This
module holds the core -- construction, Scene registration, batching, and the
attribute definitions the mixins operate on.

Two rules are worth stating up front. Changes to a parent propagate to its
children. And the material API (``set_shader``, ``set_fragment_shader``,
``set_material``) must be called **before** the Mob is spawned.

A Mob is not renderable by itself: a concrete subclass defines
``get_render_primitives()``, returning flat triangles, curved PN triangles, or
cubic bezier circuits.
"""

from __future__ import annotations

import difflib
from collections import defaultdict
from collections.abc import Callable

import torch
import torch.nn.functional as F

from algan.animatable_base.animatable import (
    ANIMATABLE_PROPERTY_VERSION,
    Animatable,
    animated_function,
    attr_ranges_for_mob,
)
from algan.animatable_base.mob_hierarchy import MobHierarchyMixin
from algan.animatable_base.mob_layout import MobLayoutMixin
from algan.animatable_base.mob_materials import (  # noqa: F401 -- exception re-exported
    MobMaterialsMixin,
    ModifiedProtectedAttributeError,
)
from algan.animatable_base.mob_morph import MobMorphMixin
from algan.animatable_base.mob_movement import MobMovementMixin
from algan.animatable_base.mob_orientation import MobOrientationMixin
from algan.animation_timeline.animation_contexts import (
    AnimationContext,
    NoExtra,
    Off,
    Seq,
    Sync,
    _reject_context_kwargs,
)
from algan.constants.spatial import *
from algan.geometry.geometry import (
    get_rotation_between_bases,
    map_global_to_local_coords,
    map_local_to_global_coords,
)
from algan.utils.animation_utils import animate_lagged_by_location
from algan.utils.tensor_utils import (
    cast_to_direction,
    cast_to_tensor,
    dot_product,
    squish,
    unsquish,
)

#: class -> (ANIMATABLE_PROPERTY_VERSION, settable property names contributed
#: by the class's MRO). See Mob._settable_property_names.
_SETTABLE_PROPERTY_CACHE: dict[type, tuple[int, set[str]]] = {}


class Mob(
    MobHierarchyMixin,
    MobOrientationMixin,
    MobMovementMixin,
    MobLayoutMixin,
    MobMorphMixin,
    MobMaterialsMixin,
    Animatable,
):
    """
    A Mob (Moveable Object) is an Animatable that exists at a point in 3-D
    space. Mobs posses the animatable attributes location, basis (orientation),
    scale, and color. Mobs can have child Mobs, forming a hierarchy, and when a
    parent mob is modified it will propagate that change to its descendants.

    Parameters
    ----------
    location
        Initial location in 3-D world space.
        Shape: `(*, 3)` where `*` denotes zero or more batch dimensions.
    basis
        Flattened 3x3 matrix specifying the Mob's orientation and scale.
        The rows represent the right, upwards, and forwards directions, respectively,
        and the row norms represent the scale in those directions.
        Defaults to an identity matrix (no rotation, unit scale).
        Shape: `(*, 9)` representing `(*, 3, 3)` flattened.
    color
        The color of the Mob. If None, it uses the default color defined
        by :meth:`~algan.animatable_base.animatable.Animatable.get_default_color`.
    opacity
        The opacity of the Mob (0.0 for fully transparent to 1.0 for fully opaque).
    glow
        The glow intensity of the Mob.
    *args, **kwargs
        Passed to :class:`~.Animatable` base class.

    Examples
    --------
    Create a square and move it to the left:

    .. algan:: Example1Mob

        from algan import *

        square = Square().spawn()
        square.move(LEFT)

        Scene.save_video()

    Create a mob with a specific color and scale:

    .. algan:: Example2Mob

        from algan import *

        circle = Circle(color=BLUE).scale(2).spawn()

        Scene.save_video()
    """

    # Primitive-family classifier used by ``become``.  Plain Mobs are
    # structural containers; renderable subclasses opt into a concrete family.
    _morph_family = None

    #: Whether this Mob's geometry should be lit from whichever side the ray
    #: arrives on. ``True`` (the default) is for geometry with no meaningful
    #: outside -- a 2-D shape, ``Text``, a parametric
    #: :class:`~algan.mobs.surfaces.surface.Surface`, an imported mesh whose
    #: winding nobody has checked -- where a back-facing hit is shaded with its
    #: normal flipped toward the viewer, so the surface is lit from behind
    #: instead of coming out black.
    #:
    #: The built-in solids set it ``False``: their normals face out (see
    #: ``tests/unit_tests/test_normal_orientation.py``), so a back-facing hit is
    #: genuinely the inside of the solid and is shaded as such. That is what
    #: stops a half-transparent solid's far shell from being lit like a second
    #: front shell -- the bright and dark "planes" through a fading Octahedron.
    #: Set it ``True`` on an instance to get the old two-sided lighting back
    #: (an open ``Cone`` you want lit inside, say); it must be set before the
    #: Mob is spawned, since the render primitive reads it once.
    two_sided = True

    #: Whether ``get_render_primitives`` returns geometry belonging to this
    #: Mob's DESCENDANTS as well as its own. Almost nothing does: a
    #: ``BezierCircuitCubic`` or a ``Surface`` draws its own rows and leaves its
    #: children to draw themselves. ``Polyhedron`` is the exception -- it
    #: gathers every face under one ``mesh_key`` -- and the difference decides
    #: two things for :meth:`~.Mob.become`: whether the Mob is one morph unit or
    #: several, and whether a descendant may be published to the Scene in its
    #: own right (doing so under an aggregator draws it twice, and draws
    #: geometry the aggregator deliberately omits, such as a Polyhedron's
    #: vertex-and-edge graph).
    draws_descendants = False

    def morph_soup_parts(self) -> list:
        """The Mobs an aggregate's PN conversion should convert and concatenate.

        Only meaningful for a Mob whose ``_morph_family`` is ``"aggregate"``.
        The default -- every descendant that answers ``get_render_primitives``
        -- is what ``Arrow3D`` and the point-cloud family draw, so neither has
        to override it.
        """
        return [
            descendant
            for descendant in self.get_descendants(include_self=False)
            if hasattr(descendant, "get_render_primitives")
        ]

    def owned_subtrees(self) -> list:
        """The child subtrees this Mob built for itself, when it aggregates.

        Only consulted when :attr:`draws_descendants` is set, and it narrows
        that claim: a Polyhedron speaks for the faces it draws and the
        vertex-and-edge graph it deliberately does not, but not for a child a
        user hung on it afterwards. Without the distinction, a morph into a
        Polyhedron carrying user geometry withheld that geometry from the Scene
        and it vanished. Returning an empty list means "everything below me".
        """
        return []

    #: Whether this Mob's triangles form a CLOSED shell -- every camera ray
    #: that enters the geometry crosses a second time on its way out. ``False``
    #: (the default) leaves ``opacity`` compositing once per crossing, which is
    #: right for anything open or unprovable: a 2-D shape, an uncapped
    #: :class:`~algan.mobs.shapes_3d.Cone`, a partial sphere, user polyhedron
    #: geometry whose closedness cannot be proven.
    #:
    #: On a closed shell, one attenuation of what is behind it IS the documented
    #: meaning of ``Mob.opacity`` -- rendering at opacity ``a`` must give
    #: ``a * (the Mob rendered opaque) + (1 - a) * backdrop`` -- so the
    #: renderer caps the shell's total coverage per pixel instead of letting
    #: both shells composite (the far sheet would otherwise deliver the extra
    #: ``a * (1 - a)`` of coverage painted with the interior's own shading).
    #: The built-in solids declare it; see
    #: ``tests/unit_tests/test_closed_shell_declaration.py`` for the proof that
    #: each declaration matches the geometry. Like ``two_sided``, set it before
    #: the Mob is spawned: the render primitive reads it once.
    #:
    #: Known limit: the rule reaches PRIMARY visibility only. A REFLECTION of a
    #: half-transparent solid -- its image in a mirror -- still composites both
    #: shells and so reads more opaque than the authored value, because the
    #: bounce loop that shades reflections carries no surface identity. The same
    #: is true of any render at ``samples_per_pixel > 1``, which routes to the
    #: Monte Carlo tracer instead.
    closed_shell = False

    #: Whether this Mob's geometry blocks light on its way from a light source
    #: to another surface -- whether it casts a shadow. ``True`` (the default)
    #: is what every Mob did before the flag existed. ``False`` makes the
    #: geometry invisible to shadow rays ONLY: it still renders to the camera,
    #: to reflections and to refraction exactly as it would have, and it still
    #: RECEIVES shadows unless :attr:`receives_shadows` says otherwise.
    #:
    #: This is the per-Mob half of ``SETTINGS.raytracing.shadows``, which
    #: remains the switch for the feature as a whole -- with shadows off
    #: globally, neither flag does anything. Use it where a shadow is
    #: physically implied but pedagogically in the way: a label plate lying on
    #: the scene's floor, a wireframe cage around the object being explained, an
    #: annotation arrow whose shadow reads as a second arrow.
    #:
    #: Like ``two_sided``, set it before the Mob is spawned -- the render
    #: primitive reads it once -- and note that it is a plain attribute, not an
    #: animatable one: it cannot change over the course of a render.
    casts_shadows = True

    #: Whether this Mob's surfaces are darkened by shadows cast onto them.
    #: ``True`` (the default) is what every Mob did before the flag existed.
    #: ``False`` shades the Mob as though every light reached it unobstructed,
    #: which is strictly cheaper than the default: no shadow ray is traced for
    #: its fragments at all.
    #:
    #: It does not change what the Mob does to OTHER surfaces -- a Mob that
    #: receives no shadow still casts one unless :attr:`casts_shadows` says
    #: otherwise. Use it to keep a surface legible where a correct shadow would
    #: not be: a caption laid on a shadowed floor, a colour key or legend that
    #: has to stay readable wherever it is placed.
    #:
    #: Set it before the Mob is spawned, and like ``casts_shadows`` it is a
    #: plain attribute rather than an animatable one.
    #:
    #: Two kinds of Mob ignore it, both because they were never shadowed to
    #: begin with: 2-D geometry (a shape, ``Text``) renders unlit, and so does
    #: anything with :meth:`~.Mob.set_shader` ``None``. A Mob carrying a custom
    #: fragment pipeline (:meth:`~.Mob.set_fragment_shader`) also ignores it,
    #: because the slot this rides in the material block belongs to that
    #: pipeline's own parameters -- the same reason a custom pipeline is never
    #: asked about ``two_sided``. ``casts_shadows`` has none of these
    #: exceptions: all three still cast.
    receives_shadows = True

    #: Opaque hashable identifying the SURFACE this Mob's geometry belongs to,
    #: stamped onto the primitives it builds. Parts of one solid that carry the
    #: same key -- a ``Cylinder``'s tube and its two end discs -- merge into a
    #: single surface for the renderer, so the joint between them is an interior
    #: edge rather than a boundary two independently antialiased surfaces meet
    #: at. ``None`` (the default) leaves each part its own surface. Only
    #: consecutive parts merge; see ``primitives._mesh_ids_from_collection``.
    mesh_key = None

    def __init__(
        self,
        location: torch.Tensor = ORIGIN,
        basis: torch.Tensor = squish(torch.eye(3)),
        color: Color | None = None,
        opacity: float = 1,
        glow: float = 0,
        *args,
        **kwargs,
    ):
        self.register_attrs_as_animatable(
            [
                "location",
                "basis",
                "scale_coefficient",
                "color",
                "opacity",
                "glow",
            ],
            Mob,
        )
        self.singleton_batch_indexing = False
        self.exclude_from_boundary = False
        self._prevent_recursive_sets = False
        super().__init__(*args, **kwargs)
        # Defines how attributes changes are inherited by children Mobs (e.g., additive for location, multiplicative for scale)
        self.attr_to_relations = defaultdict(lambda: (lambda x, y: y, lambda x, y: y))
        additive_relation = (lambda x, y: x + y, lambda x, y: y - x)
        self.attr_to_relations.update(
            {
                "location": additive_relation,
                "basis": (
                    lambda x, y: squish(
                        unsquish(x, -1, 3) @ unsquish(y, -1, 3), -2, -1
                    ),
                    lambda x, y: squish(
                        get_rotation_between_bases(
                            unsquish(x, -1, 3), unsquish(y, -1, 3)
                        ),
                        -2,
                        -1,
                    ),
                ),
                "scale_coefficient": (
                    lambda x, y: x * y,
                    lambda x, y: squish(
                        (
                            unsquish(y, -1, 3).norm(p=2, dim=-1, keepdim=True)
                            / unsquish(x, -1, 3).norm(p=2, dim=-1, keepdim=True)
                        ).expand(*([-1] * (x.dim())), 3),
                        -2,
                        -1,
                    ),
                ),
            }
        )

        if color is None:
            color = self.get_default_color()

        self._init_default_attr("location", cast_to_tensor(location))
        self._init_default_attr("basis", cast_to_tensor(basis))
        self._init_default_attr("color", color)
        self._init_default_attr("opacity", cast_to_tensor(opacity))
        self._init_default_attr("glow", cast_to_tensor(glow))
        self.num_points_per_object = 1
        self.shader = None

    @property
    def morph_kind(self):
        """Structural primitive kind used to dispatch :meth:`become`.

        The family separates genuinely different renderer primitives which may
        otherwise happen to use the same point packing.  The remaining fields
        retain the legacy same-kind contract for component and point layout.
        """
        return (
            self._morph_family,
            self.num_points_per_object,
            len(self.components),
        )

    def _rebatch_structural_attrs(self, repeat_indices, *, child=None):
        """Expand non-animatable geometry metadata alongside timeline rows.

        Subclasses whose render topology contains plain tensors override this
        hook. ``child`` identifies the component whose rows were expanded when
        the structural metadata lives on its parent (as for ``TriangleMesh``).
        """
        return self

    def _reorder_structural_attrs(self, permutation, *, child=None):
        """Reorder plain geometry metadata with an object-batch permutation."""
        return self

    #: Plain (non-animatable) attributes a morph endpoint must take from its
    #: target. Each one changes what the renderer draws and none of them lives
    #: on the timeline, so the same-kind path -- which copies the intersection
    #: of the two Mobs' ``animatable_attrs`` -- carried none of them: a morph
    #: ended with the target's geometry wearing the source's shading and
    #: sidedness. Subclasses extend the tuple rather than overriding the method.
    _MORPH_ADOPTED_ATTRS = (
        "shader",
        "two_sided",
        "closed_shell",
        "casts_shadows",
        "receives_shadows",
    )

    def _adopt_structural_attrs(self, target):
        """Take target-side plain geometry metadata at a morph endpoint.

        Assigned through the normal setter, which is how construction sets each
        of these: bypassing it with ``object.__setattr__`` would shadow rather
        than set anything that turns out to be a property, and the morph would
        pass an attribute check while rendering unchanged.
        """
        for attr in self._MORPH_ADOPTED_ATTRS:
            if hasattr(target, attr):
                setattr(self, attr, getattr(target, attr))
        return self

    def resolved_shadow_flags(self):
        """``(casts_shadows, receives_shadows)`` for this Mob, resolved against
        its ancestors: an opt-out anywhere above it applies to it.

        The Mob a user sets the flag on is very often not the Mob that builds
        the geometry. A :class:`~algan.mobs.shapes_3d.Cube` is a
        :class:`~algan.mobs.shapes_3d.Polyhedron` whose FACES carry the
        triangles, a ``Group`` holds whatever was put in it, and a ``Text``
        holds its glyphs -- so reading ``self.casts_shadows`` at the point the
        primitive is built would silently ignore ``cube.casts_shadows = False``,
        which is the obvious thing for a user to write. (It did: the flag
        reached the primitive as its default and the render was unchanged.)
        Walking up instead makes ``group.casts_shadows = False`` mean what it
        looks like, for a whole subtree, and lets a Mob that aggregates say it
        once rather than each aggregate propagating by hand the way
        ``two_sided`` does.

        The hierarchy is a DAG (``parents`` is a list), so this walks every
        ancestor and stops as soon as both flags are already False. Read at
        primitive-build time, which is why the flags must be set before the Mob
        is spawned.
        """
        casts = bool(getattr(self, "casts_shadows", True))
        receives = bool(getattr(self, "receives_shadows", True))
        seen = {id(self)}
        stack = list(getattr(self, "parents", None) or ())
        while stack and (casts or receives):
            node = stack.pop()
            if id(node) in seen:
                continue
            seen.add(id(node))
            casts = casts and bool(getattr(node, "casts_shadows", True))
            receives = receives and bool(getattr(node, "receives_shadows", True))
            stack.extend(getattr(node, "parents", None) or ())
        return casts, receives

    def _init_default_attr(self, attr, value):
        """Allocate ``attr``'s attribute-timeline buffer directly to ``value``
        during construction, bypassing the get/change/apply machinery of the
        normal property setter. Valid for a fresh mob (no children yet, not
        spawned, buffer not yet allocated) whose setter would only establish
        the initial value -- the state inside :meth:`__init__`. Falls back to
        the full setter if any precondition does not hold.
        """
        tm = self.scene.timeline_manager
        tl = tm.attr_to_timeline.get(attr)
        if self.children or (tl is not None and self.id in tl.mob_id_to_inds):
            setattr(self, attr, value)
            return self
        tm.add_mob_attr(self, attr, cast_to_tensor(value))
        return self

    def _expand_batch_if_necessary(self, value: torch.Tensor) -> torch.Tensor:
        """Internal helper to expand a tensor's batch dimension if it's a singleton
        and the parent has a larger batch size.
        """
        if value.shape[-2] == 1 and self.parent_batch_sizes is not None:
            return value.expand(
                *([-1 for _ in range(value.dim() - 2)]),
                len(self.parent_batch_sizes),
                -1,
            ).contiguous()
        return value

    def _distribute_over_packed_subtree(self, key, value, current_value):
        """Spread a per-member value across the rows of a packed subtree.

        A packed Mob carries one row per logical member, but its components
        carry a whole block of rows for each of them -- a surface's vertex
        grid, a circuit's control points. A recursive write covers the entire
        subtree in one tensor, so a value expressed per member has to be spread
        over those blocks first. ``parent_batch_sizes`` is the map, recording
        how many of a descendant's rows belong to each of its parent's, which
        is what it has always been documented to be for.

        Without this, ``pack.move(UP)`` raises: the change carries one row per
        member and the subtree read carries every row of every component.

        The subtree is addressed in **buffer** order, not descendant order --
        :meth:`RowRanges.from_runs` sorts and coalesces the runs -- so the
        per-member values are gathered into that order rather than simply
        concatenated. Getting this wrong is silent: the rows still line up in
        count, and every member reads a neighbour's value.

        Returns ``value`` unchanged whenever it is already broadcastable or the
        subtree cannot be covered exactly, so an unbatched Mob -- and any shape
        this cannot describe -- behaves exactly as it did before.
        """
        members = value.shape[-2]
        total = current_value.shape[-2]
        if members == 1 or members == total:
            return value
        owners = self._member_owner_index(key, members, total)
        if owners is None:
            return value
        return value.index_select(-2, owners.to(value.device))

    def _member_owner_index(self, key, members, total, partial=False):
        """Map each buffered row of ``key`` over this Mob's packed subtree to
        the member that owns it, in buffer order.

        Returns a 1-D index tensor of length ``total``, or ``None`` when the
        subtree cannot be described exactly -- an attribute with no timeline, a
        descendant missing rows, a ragged ``parent_batch_sizes`` block. With
        ``partial=True`` an individual inconsistency no longer aborts the whole
        map: that descendant's rows come back owned by nobody (``-1``), which
        lets callers leave just those rows alone instead of giving up on the
        write. See :meth:`_distribute_over_packed_subtree`.
        """
        timeline = self.scene.timeline_manager.attr_to_timeline.get(key)
        if timeline is None:
            return None
        descendants = [
            mob
            for mob in self.get_descendants(include_self=True)
            if mob is self or key not in getattr(mob, "_excluded_from_parent_attrs", ())
        ]
        rows, owners = [], []
        for mob in descendants:
            if mob.id not in timeline.mob_id_to_inds:
                if partial:
                    continue
                return None
            mob_rows = attr_ranges_for_mob(timeline, mob).tensor()
            if mob is self:
                owner = torch.arange(members, device=mob_rows.device)
            else:
                sizes = mob.parent_batch_sizes
                if sizes is None or sizes.shape[-1] != members:
                    if not partial:
                        return None
                    owners.append(
                        torch.full(
                            (mob_rows.numel(),),
                            -1,
                            dtype=torch.long,
                            device=mob_rows.device,
                        )
                    )
                    rows.append(mob_rows)
                    continue
                owner = torch.arange(members, device=mob_rows.device).repeat_interleave(
                    sizes.to(mob_rows.device)
                )
            if mob_rows.numel() != owner.numel():
                if not partial:
                    return None
                owner = torch.full(
                    (mob_rows.numel(),),
                    -1,
                    dtype=torch.long,
                    device=mob_rows.device,
                )
            rows.append(mob_rows)
            owners.append(owner)
        rows = torch.cat(rows)
        if rows.numel() != total:
            return None
        return torch.cat(owners)[torch.argsort(rows)]

    def _spread_change_over_packed_rows(self, key, change, current_value, neutral):
        """Spread a per-member change across a packed subtree's own rows,
        leaving rows that belong to no single member untouched.

        The exact-cover variant (:meth:`_distribute_over_packed_subtree`) is
        right for values that exist once per point -- a location row for every
        vertex. Rows of attributes a pack does not replicate per point cannot
        be covered that way: a packed circuit's control points share **one**
        basis row across all of its members, and no single member's change may
        claim it. Those rows receive ``neutral`` (the identity element of the
        change's composition) instead, so the write stays well-defined rather
        than aborting the whole transform.
        """
        members = change.shape[-2]
        total = current_value.shape[-2]
        if members == 1 or members == total:
            return change
        owners = self._member_owner_index(key, members, total, partial=True)
        if owners is None:
            return change
        gathered_change = change.index_select(-2, owners.clamp(min=0).to(change.device))
        # Every owner's neutral block is the same identity element, so rows
        # with no owner can take their gathered neighbour's neutral freely.
        gathered_neutral = neutral.index_select(
            -2, owners.clamp(min=0).to(neutral.device)
        )
        known = (owners >= 0).to(change.device)
        # A (..., total, 1) mask keeps this correct for any batch rank.
        known = known.reshape((1,) * (change.dim() - 2) + (total, 1)).expand(
            *change.shape[:-2], total, 1
        )
        return torch.where(known, gathered_change, gathered_neutral)

    @animated_function(
        animated_args={"interpolation": 0.0},
        unique_args=["key", "recursive", "relative"],
    )
    def apply_absolute_change_two(
        self,
        key: str,
        change1: any,
        change2: any = None,
        interpolation: float = 1.0,
        recursive: bool = True,
        relative: bool = False,
    ):
        """Animate an attribute out to one value and then on to another.

        A two-stage keyframe animation: the attribute moves from its current
        value to ``change1`` over the first half of the animation, then from
        ``change1`` to ``change2`` over the second half. This is the machinery
        behind :meth:`~.Mob.pulse_color`, and is the way to build any
        there-and-back effect on an arbitrary attribute.

        Animation
        ---------
        Recorded as an animation spanning the current context's duration
        (1 second by default), with the turning point at the halfway mark.
        Applies to descendants unless ``recursive`` is False.

        Parameters
        ----------
        key
            Name of the animatable attribute to drive, e.g. ``"location"``,
            ``"color"``, ``"opacity"``.
        change1
            Value to animate out to, reached at the halfway point.
        change2
            Value to animate on to by the end. Defaults to ``None``, meaning each
            affected part returns to its own pre-animation value -- the right
            choice for a pulse on a composite Mob, where one shared target value
            would flatten per-descendant attributes.
        interpolation
            Animation progress, filled in per frame by the animation system.
            Defaults to ``1.0``; you do not normally pass this yourself.
        recursive
            Whether to drive the attribute on descendants too. Defaults to True.
        relative
            Whether ``change1`` and ``change2`` are multipliers of each part's
            current value rather than absolute targets, e.g. a scale pulse to
            ``1.2`` times current size. Defaults to False.

        Returns
        -------
        :class:`~.Mob`
            This Mob, so calls can be chained.
        """
        # The allocation default seeds attribute rows for mobs that never had
        # this attribute set; it only sizes row allocation, the interpolation
        # below uses the un-expanded changes so they broadcast against however
        # many rows the (recursive) union covers. In the non-recursive case the
        # caller may be a record-time batched mob (animate_lagged_by_location's
        # per-element waves) whose row count is its location row count — a
        # singleton default would allocate one shared row and the batched
        # per-element write would no longer fit.
        default = (
            change1
            if (not relative and change1 is not None)
            else cast_to_tensor(getattr(self, key))
        )
        default = cast_to_tensor(default)
        if not recursive and default.shape[-2] == 1:
            default = default.expand(
                *([-1] * (default.dim() - 2)), self.location.shape[-2], -1
            )
        current_value = self.get_animated_attribute(
            key, include_descendants=recursive, default=default
        )

        # A pack's changes arrive one row per member; the union above covers
        # every row of every component. See _distribute_over_packed_subtree.
        def spread(change):
            if not torch.is_tensor(change) or change.shape[-2] == 1:
                return change
            return self._distribute_over_packed_subtree(key, change, current_value)

        if recursive:
            change1 = spread(change1)
            change2 = spread(change2)
        if relative:
            change1 = current_value * cast_to_tensor(change1)
            change2 = (
                current_value
                if change2 is None
                else current_value * cast_to_tensor(change2)
            )
        elif change2 is None:
            change2 = current_value
        interpolation = (
            cast_to_tensor(interpolation) * 2
        )  # Double interpolation for 2-stage animation

        # Calculate the interpolated value based on the two changes
        # m is a mask for when interpolation goes beyond 1.0
        mask_interp_gt_1 = (interpolation > 1).float()
        interpolated_value = (
            current_value * (1 - interpolation) + interpolation * change1
        ) * (1 - mask_interp_gt_1) + mask_interp_gt_1 * (
            change1 * (2 - interpolation) + (interpolation - 1) * change2
        )

        self._setattr_and_record_modification(
            key, interpolated_value, include_descendants=recursive
        )
        return self

    def set_opacity_via_color(self, opacity: float | torch.Tensor) -> Mob:
        """Fade the Mob by writing opacity into its color rather than its opacity.

        Each descendant's own color gets the given alpha, which fades parts that
        carry their own colors without a parent-level opacity write flattening
        them. Prefer setting
        :attr:`~algan.animatable_base.mob.Mob.opacity` for ordinary fades; this
        exists for composites where per-part color must be preserved.

        Animation
        ---------
        Recorded as an animation. All descendants fade together inside a
        :class:`~.Sync`, over the current context's duration (1 second by
        default).

        Parameters
        ----------
        opacity
            Target alpha, ``0`` for fully transparent to ``1`` for fully opaque.

        Returns
        -------
        :class:`~.Mob`
            This Mob, so calls can be chained.
        """
        with Sync(animation_manager=self.animation_manager):
            for d in self.get_descendants():
                d._original_color_set_opacity_via_color = d.color
                d.set_non_recursive(color=d.color.set_opacity(opacity))
        return self

    def pulse_color(
        self,
        color: torch.Tensor = None,
        opacity: bool = None,
        recursive=True,
        new_color=None,
    ) -> Mob:
        """Flash the Mob a different color and let it settle back.

        A two-stage animation: the color travels out to ``color`` by the halfway
        point and back to ``new_color`` by the end. Good for drawing the eye to
        one part of a diagram without leaving it recolored.

        Animation
        ---------
        Recorded as an animation over the current context's duration (1 second by
        default), with the peak of the pulse at the halfway mark. Color and
        opacity pulses run together inside a :class:`~.Sync`.

        Parameters
        ----------
        color
            Color to pulse to. Defaults to ``None``, which pulses only the
            opacity (so pass at least one of ``color`` and ``opacity``).
        opacity
            Alpha to hold for both stages of the pulse, ``0`` to ``1``. Defaults
            to ``None``, leaving opacity alone.
        recursive
            Whether descendants pulse too. Defaults to True.
        new_color
            Color to end on. Defaults to ``None``, meaning every affected part
            returns to its own current color.

        Returns
        -------
        :class:`~.Mob`
            This Mob, so calls can be chained.

        See Also
        --------
        :meth:`~.Mob.wave_color` : Run the same pulse across the Mob as a travelling wave.
        """
        with Sync(animation_manager=self.animation_manager):
            if color is not None:
                # new_color=None restores each part to its own current color
                # (resolved inside apply_absolute_change_two). Passing
                # ``self.color`` here instead would broadcast the parent's own
                # color over all descendants — for a composite whose parent Mob
                # never had its color set (Groups, NeuralNetMLP, ...) that is
                # the default BLACK, leaving the whole mob blackened after the
                # pulse. [1,1,D] tensors broadcast against however many
                # attribute rows the (recursive) write covers.
                new_color = None if new_color is None else cast_to_tensor(new_color)
                # if opacity is not None:
                #    o = cast_to_tensor(opacity)
                #    color = color.set_opacity(o)
                #    if new_color is not None:
                #        new_color = new_color.set_opacity(o)
                self.apply_absolute_change_two(
                    "color", cast_to_tensor(color), new_color, recursive=recursive
                )
            if opacity is not None:
                o = cast_to_tensor(opacity)
                self.apply_absolute_change_two("opacity", o, o, recursive=recursive)
        return self

    def wave_color(
        self,
        color: torch.Tensor = None,
        wave_length: float = 2,
        reverse: bool = False,
        direction: torch.Tensor | None = None,
        lag_duration=1,
        samples_per_wave: int | None = 12,
        refine_resolution: bool = True,
        restore_resolution: bool = True,
        **kwargs,
    ) -> Mob:
        """Send a colour pulse travelling across the Mob.

        Every renderable part of the Mob pulses, but each one starts a little
        later than the part behind it, so the colour sweeps across the shape
        instead of flashing all at once. Parts are ordered by their position
        along ``direction``.

        The colour is carried by the Mob's vertices, so a Mob sampled more
        coarsely than the wave is wide would show the pulse as a few flat facets
        -- a :class:`~algan.mobs.surfaces.surface.Surface` shaped like a flat
        sheet has vertices only at its corners, and a filled
        :class:`~algan.mobs.bezier_circuit.BezierCircuitCubic` has a single
        colour sample by default. Such Mobs are re-sampled finely enough to draw
        the wave and, by default, dropped back to their original resolution once
        the block containing the wave is over; see ``samples_per_wave`` and
        ``restore_resolution``.

        Animation
        ---------
        Recorded as an animation. The total duration is ``lag_duration`` plus one
        part's pulse, so it is set by these parameters rather than by the current
        context's duration. Re-sampling a part is a topology change, so it splits
        that Mob's history the way
        :meth:`~algan.animatable_base.mob.Mob.detach_history` does, at the start
        of the wave and, when ``restore_resolution`` is True, again when the
        enclosing block ends. The split is invisible, but it is why the
        resolution only drops back at the end of the block rather than at the
        end of the wave.

        Parameters
        ----------
        color
            Colour for the wave to pulse to. Defaults to ``None``, which pulses
            only opacity (pass one through ``**kwargs``).
        wave_length
            How spread out the wave is: each part's own pulse lasts
            ``wave_length / lag_duration`` seconds, so smaller values give a
            tighter, more sharply defined band. Defaults to ``2``.
        reverse
            Whether the wave travels the opposite way along ``direction``.
            Defaults to False.
        direction
            Direction the wave travels, shape ``(*, 3)``. Defaults to ``None``,
            meaning the Mob's own upward direction (bottom to top).
        lag_duration
            Seconds between the first part starting its pulse and the last one
            starting theirs. Defaults to ``1``.
        samples_per_wave
            How many colour samples to fit across the width of the travelling
            band. Parts already sampled at least this finely along ``direction``
            are left alone; coarser ones are temporarily refined. Defaults to
            ``12`` -- a pulse is two straight ramps, which colour interpolation
            reproduces exactly, so this only has to round off the peak between
            them and raising it buys geometry rather than smoothness. Pass
            ``None`` to leave every part's resolution exactly as it is.
        refine_resolution
            Whether parts sampled too coarsely to show the wave may be refined
            at all. Defaults to True. False leaves every part exactly as
            authored, however coarse -- the explicit form of
            ``samples_per_wave=None``. Worth setting for small on-screen mobs:
            the refinement is judged in world units relative to the mob's own
            extent, never in pixels, so a mob a few dozen pixels wide is refined
            as heavily as a full-screen one.
        restore_resolution
            Whether refined colour grids return to their original resolution
            when the enclosing animation block ends. Defaults to True. Set to
            False when a newly spawned object must retain one stable topology
            throughout and after its materialization wave.
        **kwargs
            Passed to :meth:`~.Mob.pulse_color` for each part -- notably
            ``opacity`` and ``new_color``. ``new_color`` may also be a callable
            receiving the primitive part being pulsed; this lets a composite
            settle to each part's own target color after one shared wave.

        Returns
        -------
        :class:`~.Mob`
            This Mob, so calls can be chained.
        """
        if direction is None:
            direction = self.get_upwards_direction()
        direction = direction * (-1 if reverse else 1)
        # What each part's pulse actually writes, which decides whether a finer
        # sampling can show the wave at all (see pulse_color).
        pulsed_attrs = frozenset(
            name
            for name, value in (("color", color), ("opacity", kwargs.get("opacity")))
            if value is not None
        )
        restores = (
            self._refine_parts_for_color_wave(
                direction, wave_length, lag_duration, samples_per_wave, pulsed_attrs
            )
            if refine_resolution
            else []
        )
        with AnimationContext(
            run_time_unit=wave_length / lag_duration,
            animation_manager=self.animation_manager,
        ) as wave_context:
            primitive_mobs = self._wave_pulsed_parts()
            kwargs["recursive"] = False

            def pulse_part(part):
                part_kwargs = kwargs
                new_color = kwargs.get("new_color")
                if callable(new_color):
                    part_kwargs = {**kwargs, "new_color": new_color(part)}
                part.pulse_color(color, **part_kwargs)

            animate_lagged_by_location(
                primitive_mobs,
                pulse_part,
                direction,
                lag_duration=lag_duration,
            )
        if restores and restore_resolution:
            self._schedule_wave_resolution_restore(restores, wave_context)
        return self

    def _wave_pulsed_parts(self):
        """Internal: the parts :meth:`~.Mob.wave_color` pulses one colour each.

        Only primitive parts, so the wave animates on individual rendering
        elements rather than on the containers holding them.
        """
        return [
            _
            for _ in self.get_descendants()
            if (_.is_primitive and not _.ignore_wave_animations)
        ]

    def _refine_parts_for_color_wave(
        self, direction, wave_length, lag_duration, samples_per_wave, pulsed_attrs
    ):
        """Internal: raise the sampling of any part too coarse to draw the wave.

        The pulse a part plays starts ``t * lag_duration`` seconds in, where
        ``t`` is its position along ``direction`` normalized over every pulsed
        part, and lasts ``wave_length / lag_duration`` seconds. So at any instant
        the band of parts mid-pulse spans ``wave_length / lag_duration ** 2`` of
        that normalized range: divide by ``samples_per_wave`` and we have the
        largest gap between neighbouring samples that still renders the band
        smoothly.

        Returns
        -------
        list
            Callables restoring each refined Mob's original resolution, in the
            order they were refined.
        """
        if (
            samples_per_wave is None
            or samples_per_wave <= 0
            or lag_duration <= 0
            or not pulsed_attrs
        ):
            return []
        if not self.animation_manager.context.record_funcs:
            # Nothing is being animated (an Off block, say), so there is no wave
            # to draw and no reason to pay for the geometry.
            return []
        pulsed = self._wave_pulsed_parts()
        if not pulsed:
            return []
        projections = torch.cat(
            [dot_product(direction, _.location, dim=-1).reshape(-1) for _ in pulsed], -1
        )
        band = (projections.amax() - projections.amin()).item() * (
            wave_length / lag_duration**2
        )
        if not band > 0:
            return []
        max_spacing = band / samples_per_wave

        def guarded(mob, restore):
            # Restoring re-spawns the Mob at the split it records (see
            # detach_history), so a Mob that despawns during the wave keeps the
            # refined rows it is no longer showing rather than being resurrected
            # by its own clean-up.
            def guarded_restore():
                if not mob.is_despawned():
                    restore()

            return guarded_restore

        restores = []
        for mob in list(self.get_descendants()):
            if mob.is_despawned():
                continue
            restore = mob._refine_sampling_for_color_wave(
                direction, max_spacing, pulsed_attrs
            )
            if restore is not None:
                restores.append(guarded(mob, restore))
        return restores

    def _refine_sampling_for_color_wave(self, direction, max_spacing, pulsed_attrs):
        """Internal: temporarily re-sample this Mob to render a colour wave.

        Does nothing by default. Mobs that can rebuild themselves at a different
        resolution -- :class:`~algan.mobs.surfaces.surface.Surface` from its
        ``coord_function``, :class:`~algan.mobs.bezier_circuit.BezierCircuitCubic`
        from its texture grid -- override this.

        Parameters
        ----------
        direction
            Direction the wave travels, shape ``(*, 3)``. Not normalized:
            distances are measured by projecting onto it, exactly as the wave's
            own lag is.
        max_spacing
            Largest projected gap between neighbouring colour samples that still
            draws the wave smoothly, in the units ``direction`` projects to.
        pulsed_attrs
            Names of the attributes the wave writes on each part -- ``"color"``,
            ``"opacity"``, or both. A Mob that stores one of them per sample and
            the other per object refines only for the former.

        Returns
        -------
        callable or None
            A zero-argument callable restoring the original resolution, or None
            if this Mob was left as it is.
        """
        return None

    def _schedule_wave_resolution_restore(self, restores, wave_context):
        """Internal: put a refined Mob back to its own resolution after the wave.

        Restoring is a topology change, so it goes through
        :meth:`~.Mob.detach_history`: everything recorded on the refined rows
        stays with a frozen clone that despawns at that instant. Anything
        recorded *afterwards* therefore lands on rows that only become visible
        then -- which is fine at the top level, where the cursor has already
        advanced past the wave, but not inside a ``with`` block, where a sibling
        animation recorded after this call still starts back at the block's own
        cursor. So inside a block the restore waits for the outermost open block
        to finish, by which point everything timed alongside the wave has been
        recorded against the refined Mob.
        """
        context = self.animation_manager.context
        outermost = None
        while context.prev_context is not None:
            outermost, context = context, context.prev_context
        if outermost is not None:
            outermost.add_exit_callback(lambda: [restore() for restore in restores])
            return
        cursor = context.timespan.current_time
        context.timespan.current_time = max(cursor, wave_context.timespan.end)
        try:
            for restore in restores:
                restore()
        finally:
            context.timespan.current_time = cursor

    def _prepare_buffers(self, key, value):
        tm = self.scene.timeline_manager
        tm.add_mob_attr(self, key, value, add_mob=False)
        tl = tm.attr_to_timeline[key]
        if self.id not in tl.mob_id_to_inds:
            self._try_add_to_timeline(key, value)
            return self
        current_inds = tl.mob_id_to_inds[self.id]
        value = cast_to_tensor(value)
        shared_view_has_full_buffer = (
            self.data_sub_inds is not None and current_inds.shape[0] == self.batch_size
        )
        if (
            shared_view_has_full_buffer
            or current_inds.shape[0] == value.shape[-2]
            or value.shape[-2] == 1
        ):
            return self
        current_value = self.get_animated_attribute(
            key, default=None, include_descendants=False
        )
        if current_value.shape[-2] != 1:
            raise ValueError(
                f"Attempting to set {key} which currently has value of shape {current_value.shape}"
                f"to new value with shape {value.shape}, which is not broadcastable."
            )
        # Indexed mobs share their source's timeline rows.  ``data_sub_inds``
        # is expressed in that full source index space, so expanding only to
        # the selected view's local row count would leave later indexing out
        # of bounds (for example a packed text glyph's control-point color).
        target_size = (
            self.batch_size if self.data_sub_inds is not None else value.shape[-2]
        )
        expanded = current_value.expand(
            *([-1] * (current_value.dim() - 2)), target_size, -1
        )
        tl.add(self, expanded, overwrite=True)
        return self

    @animated_function(animated_args={"interpolation": 0.0})
    def _apply_change(
        self, attr, change, recursive=True, interpolation=1.0, scope=None
    ):
        # ``scope`` (a RowRanges) addresses an explicit set of rows instead of
        # this Mob's own / its whole subtree, which is what lets one recorded
        # event carry a per-row change over many Mobs. It is an ordinary
        # recorded kwarg, so replay hands the same object back and the read and
        # the write both land on the rows the recording wrote.
        change = change * interpolation
        current_value = self.get_animated_attribute(
            attr, include_descendants=recursive, copy=False, _scope=scope
        )
        if recursive and scope is None:
            change = self._distribute_over_packed_subtree(attr, change, current_value)
        new_value = current_value + change
        return self._setattr_and_record_modification(
            attr, new_value, include_descendants=recursive, _scope=scope
        )

    @animated_function(animated_args={"interpolation": 0.0})
    def _apply_set(self, attr, value, recursive=True, interpolation=1.0):
        new_value = value * interpolation
        return self._setattr_and_record_modification(
            attr, new_value, include_descendants=recursive
        )

    def set_animated_attribute(self, attr: str, value, recursive: bool = True) -> Mob:
        """Animate one animatable attribute to a new value, by name.

        The by-name equivalent of assigning to the attribute; useful when the
        attribute is chosen at runtime. To set several at once, use
        :meth:`~.Mob.set`. Attributes whose write is more than a row write --
        derived ones (``scale_coefficient``, ``Circle.radius``,
        ``border_color``) and ``basis``, which carries the subtree's locations
        with it -- are handed to their property setter, so every name behaves
        exactly as the assignment would.

        Animation
        ---------
        Recorded as an animation: the attribute interpolates from its current
        value to ``value`` over the current context's duration (1 second by
        default).

        Parameters
        ----------
        attr
            Name of the animatable attribute, e.g. ``"location"``, ``"color"``,
            ``"opacity"``.
        value
            Target value. Must be broadcastable against the attribute's current
            shape.
        recursive
            Whether the change propagates to descendants. Defaults to True.

        Returns
        -------
        :class:`~.Mob`
            This Mob, so calls can be chained.
        """
        if self._prevent_recursive_sets:
            recursive = False
        value = cast_to_tensor(value)

        if self._writes_through_property_setter(attr):
            # The generic path below writes timeline rows, which for these
            # attributes is either nothing or half the operation: a derived
            # attribute has no rows, so it would allocate a buffer nothing
            # reads, and a hierarchical one would move each frame while leaving
            # the subtree's locations -- the actual geometry -- behind. The
            # property setter is what does the whole thing.
            prs = self._prevent_recursive_sets
            self._prevent_recursive_sets = not recursive
            try:
                setattr(self, attr, value)
            finally:
                self._prevent_recursive_sets = prs
            return self

        current_value = self.get_animated_attribute(
            attr, include_descendants=recursive, default=value, copy=False
        )
        if recursive:
            value = self._distribute_over_packed_subtree(attr, value, current_value)
        change = value - current_value
        self._apply_change(attr, change, recursive=recursive)
        return self

    def map_animated_attribute(self, attr: str, func: Callable) -> Mob:
        """Animate an attribute to a function of its own current value, across
        this Mob and every descendant at once.

        :meth:`~.Mob.set_animated_attribute` animates towards a value you supply.
        This animates towards a value *derived from what each part already has*,
        which is what you want for "half as bright as it is now" or "everything
        a quarter of its current size" -- operations where each part has a
        different starting point and a different target.

        ``func`` receives every affected value stacked into one tensor of shape
        ``(1, N, D)``, where ``N`` counts the rows this Mob and its descendants
        own for ``attr`` and ``D`` is the attribute's width (1 for ``opacity``,
        3 for ``location``, 4 for ``color``). It must return a tensor of that
        same shape -- the target values, in the same order.

        Reach for it instead of a Python loop over
        :meth:`~.Mob.get_descendants`. A loop records one animation per
        descendant, which on a large group is thousands of separate recorded
        animations; this records **one**, and renders measurably faster for it.

        Animation
        ---------
        Recorded as an animation: every affected value moves from what it is now
        to its target over the current context's duration (1 second by default),
        all together inside a :class:`~.Sync`. ``func`` is evaluated **once, at
        the moment of the call** -- it computes the destination, it is not
        re-run per frame. For a value that must be recomputed every frame, use
        :meth:`~.Mob.add_updater` instead. Only spawned Mobs record; a Mob whose
        subtree is not on screen is changed immediately and silently.

        Parameters
        ----------
        attr
            Name of the animatable attribute, e.g. ``"opacity"``, ``"color"``,
            ``"location"``, ``"glow"``. It has to be one whose whole meaning is
            its per-row value, and two kinds are not, so both are rejected
            rather than silently half-applied. A *derived* property
            (``scale_coefficient``, the row norms of ``basis``;
            ``Circle.radius``; ``border_color``) has no rows at all. A
            *hierarchical* one (``basis``) has rows, but they are only half the
            operation: a rotation or a scale has to carry the subtree's
            locations along, which is why :meth:`~.Mob.rotate`,
            :meth:`~.Mob.scale` and plain assignment are what change those.
        func
            Callable mapping the stacked current values to the stacked target
            values, both of shape ``(1, N, D)``. It is passed a copy, so it may
            modify its argument in place and return it.

        Returns
        -------
        :class:`~.Mob`
            This Mob, so calls can be chained.

        Raises
        ------
        ValueError
            If ``func`` returns a tensor whose shape is not the shape it was
            given.
        AttributeError
            If ``attr`` is not an animatable attribute of this Mob, or is a
            derived or hierarchical one that cannot be mapped row-wise.

        See Also
        --------
        :meth:`~.Mob.set_animated_attribute` : Animate an attribute to a value
            you supply, rather than to a function of the current one.
        :meth:`~.Mob.add_updater` : Recompute a value every frame instead of
            once.

        Examples
        --------
        .. algan:: Example1MobMapAnimatedAttribute

            from algan import *

            row = Group([Square().move(LEFT * 2), Circle(), Triangle().move(RIGHT * 2)])
            row.spawn()

            # Dim everything to a tenth of however visible it already is,
            # then pull every point halfway in towards the world origin.
            row.map_animated_attribute('opacity', lambda o: o * 0.1)
            row.map_animated_attribute('location', lambda p: p * 0.5)

            Scene.save_video()
        """
        self.check_properties_are_valid((attr,))
        if self._writes_through_property_setter(attr):
            self._raise_not_row_wise_error("map_animated_attribute", attr)
        current = self.get_animated_attribute(attr, include_descendants=True, copy=True)
        target = cast_to_tensor(func(current))
        if target.shape != current.shape:
            raise ValueError(
                f"map_animated_attribute({attr!r}, ...): func must return the "
                f"shape it was given, {tuple(current.shape)}, but returned "
                f"{tuple(target.shape)}."
            )
        # One recorded event carrying a per-row change, rather than one per
        # descendant: _apply_change re-reads the same rows at replay and adds
        # this change scaled by the interpolant, so each row travels from its
        # own start to its own target.
        with Sync(animation_manager=self.animation_manager):
            self._apply_change(attr, target - current, recursive=True)
        return self

    @property
    def location(self) -> torch.Tensor:
        """The Mob's position in world space, shape ``(*, 3)``.

        Assigning to this animates the Mob to the new position over the current
        context's duration (1 second by default), carrying its children along so
        their offsets from it are preserved. ``mob.location = ORIGIN`` and
        ``mob.move_to(ORIGIN)`` are the same operation.
        """
        return self.get_animated_attribute("location")

    @location.setter
    def location(self, location: torch.Tensor):
        recursive = not self._prevent_recursive_sets
        # The funnel for move, move_to, set_location and set(location=...): a
        # scalar would broadcast to the (1, 1, 1) diagonal instead of raising.
        value = cast_to_direction("location", location)
        attr = "location"

        current_value = self.get_animated_attribute(
            attr, include_descendants=False, default=value, copy=False
        )
        change = value - current_value
        self._apply_change(attr, change, recursive=recursive)
        return self

    @property
    def basis(self) -> torch.Tensor:
        """The Mob's orientation and scale, as a flattened 3x3 matrix of shape ``(*, 9)``.

        Unflattened, the three rows are the Mob's own right, upward and forward
        directions in world space, and each row's norm is the Mob's scale along
        that axis. The identity matrix therefore means unrotated at unit scale.

        Assigning to this animates the Mob to the new basis over the current
        context's duration (1 second by default), rotating and scaling its
        children with it. Concurrent writes compose rather than overwrite, so a
        rotate and a scale inside one :class:`~.Sync` both take effect.
        """
        return self.get_animated_attribute("basis")

    @property
    def normalized_basis(self) -> torch.Tensor:
        """The Mob's orientation with scale divided out, shape ``(*, 9)``.

        The same matrix as :attr:`~.Mob.basis` with every row normalized to unit
        length, so it carries the Mob's rotation and nothing else. Read-only.
        """
        return squish(
            unsquish(self.basis, -1, 3) / self.scale_coefficient.unsqueeze(-1), -2, -1
        )

    @basis.setter
    def basis(self, basis: torch.Tensor):
        value = cast_to_tensor(basis)
        inverse_relation = self.attr_to_relations["basis"][1]
        # Convert the absolute target into a relative change against the
        # current basis, so that concurrent basis writers (e.g. a rotate and a
        # scale in the same Sync) compose instead of overriding each other.
        my_basis = self.get_animated_attribute(
            "basis", include_descendants=False, default=value, copy=False
        )
        recursive = not self._prevent_recursive_sets
        change = inverse_relation(my_basis, value)
        # recursive must be passed as an explicit kwarg (not read from
        # self._prevent_recursive_sets inside _apply_basis_change) so that it
        # is recorded with the function application and replays correctly at
        # render time, when _prevent_recursive_sets has been restored.
        self._apply_basis_change(change, default_basis=value, recursive=recursive)

    @animated_function(animated_args={"interpolation": 0.0})
    def _apply_basis_change(
        self, change, default_basis=None, recursive=True, interpolation=1.0
    ):
        attr = "basis"
        relation, inverse_relation = self.attr_to_relations[attr]

        my_basis = self.get_animated_attribute(
            "basis", include_descendants=False, default=default_basis, copy=False
        )
        my_loc = self.get_animated_attribute(
            "location", include_descendants=False, copy=False
        )

        identity = inverse_relation(my_basis, my_basis)
        interpolated_change = torch.lerp(identity, change, interpolation)
        new_basis = relation(my_basis, interpolated_change)

        child_loc = self.get_animated_attribute(
            "location", include_descendants=recursive, copy=False
        )

        # A packed Mob turns about one pivot per member, so the pivots are
        # spread over each component's rows the same way an ordinary recursive
        # write is. Unbatched Mobs get these back unchanged.
        #
        # The expansion follows the subtree's **location** rows, because that
        # is what a pivot meets: ``map_global_to_local_coords`` evaluates it
        # against ``child_loc``, and those rows enumerate locations. Spreading
        # the pivot basis over *basis* rows instead used to work only where the
        # two layouts happen to mirror each other; a packed circuit's control
        # points carry one location row per point but share a single basis row,
        # so the basis-keyed spread could not cover the subtree, came back
        # unchanged, and left a 5-row pivot against 371 child rows -- which is
        # why ``rotate``/``scale`` died on such a pack while ``move`` (whose
        # change spreads over location rows) worked.
        def spread(value):
            if not recursive:
                return value
            return self._distribute_over_packed_subtree("location", value, child_loc)

        pivot_loc = spread(my_loc)
        pivot_basis = spread(my_basis)
        pivot_new_basis = spread(new_basis)
        local_coords = map_global_to_local_coords(pivot_loc, pivot_basis, child_loc)
        new_child_location = map_local_to_global_coords(
            pivot_loc, pivot_new_basis, local_coords
        )

        child_basis = self.get_animated_attribute(
            "basis", include_descendants=recursive, copy=False
        )
        # Unlike the pivots, this composition targets the subtree's own basis
        # rows, so it spreads over those -- tolerantly: a row shared by several
        # members takes no change at all rather than an arbitrary member's.
        new_child_basis = relation(
            child_basis,
            self._spread_change_over_packed_rows(
                "basis", interpolated_change, child_basis, identity
            ),
        )

        self._apply_set("location", new_child_location, recursive=recursive)
        self._apply_set("basis", new_child_basis, recursive=recursive)

    @property
    def scale_coefficient(self) -> torch.Tensor:
        """The Mob's scale along its own right, up and forward axes, shape ``(*, 3)``.

        Derived from :attr:`~.Mob.basis` as the norm of each of its rows, so
        ``(1, 1, 1)`` is unscaled. Assigning to this resizes the Mob without
        rotating it, animated over the current context's duration (1 second by
        default); :meth:`~.Mob.scale` and :meth:`~.Mob.set_scale` are the usual
        way to do that.
        """
        return unsquish(self.basis, -1, 3).norm(p=2, dim=-1, keepdim=False)

    @scale_coefficient.setter
    def scale_coefficient(self, scale_coefficient: torch.Tensor):
        """Sets the scaling factor of the Mob, re-normalizing the basis vectors.

        This ensures that setting a new scale coefficient only changes the size
        of the Mob, preserving its orientation.

        """
        scale_coefficient = cast_to_tensor(scale_coefficient)
        new_basis = squish(
            F.normalize(unsquish(self.basis, -1, 3), p=2, dim=-1)
            * scale_coefficient.unsqueeze(-1),
            -2,
            -1,
        )
        self.basis = new_basis
        return self

    def get_normal(self) -> torch.Tensor:
        """Get the Mob's surface normal, i.e. the way it faces.

        An alias for
        :meth:`~algan.animatable_base.mob_orientation.MobOrientationMixin.get_forward_direction`,
        named for the 2-D case:
        a flat shape's normal is the direction it faces out of its own plane.

        Returns
        -------
        torch.Tensor
            Unit normal, shape ``(*, 3)``.
        """
        return self.get_forward_direction()

    def set_location(self, location: torch.Tensor, recursive: bool = True) -> Mob:
        """Set the Mob's location, with control over whether children follow.

        Same as
        :meth:`~algan.animatable_base.mob_movement.MobMovementMixin.move_to`
        without the arc option; the reason to reach
        for this one is ``recursive=False``.

        Animation
        ---------
        Recorded as an animation over the current context's duration (1 second by
        default).

        Parameters
        ----------
        location
            The target location, shape ``(*, 3)``.
        recursive
            Whether children move along, keeping their offsets from this Mob.
            Defaults to True; False moves only this Mob, leaving its children
            behind.

        Returns
        -------
        :class:`~.Mob`
            This Mob, so calls can be chained.
        """
        if recursive:
            self.location = location
        else:
            self.set_non_recursive(location=location)
        return self

    def get_parts_as_mobs(self) -> list[Mob]:
        """Flatten this Mob's hierarchy into a list.

        Returns
        -------
        list[:class:`~.Mob`]
            This Mob followed by every descendant, depth-first. The Mobs are the
            live objects, not copies, so changing one changes the scene.
        """
        parts = [self]
        for child in self.children:
            parts.extend(child.get_parts_as_mobs())
        return parts

    def scale(self, scale_factor: float | torch.Tensor, recursive: bool = True) -> Mob:
        """Resize the Mob relative to its current size.

        The factor multiplies the size the Mob has now, so two calls to
        ``scale(2)`` leave it four times its original size. For an absolute
        target, use :meth:`~.Mob.set_scale`.

        Animation
        ---------
        Recorded as an animation: the Mob grows or shrinks over the current
        context's duration (1 second by default).

        Parameters
        ----------
        scale_factor
            Multiplier on the current size: ``2`` for twice as big, ``0.5`` for
            half. A tensor of shape ``(*, 3)`` scales the Mob's right, up and
            forward axes separately, which is how you stretch a shape.
        recursive
            Whether descendants scale too, keeping the Mob's proportions.
            Defaults to True; False scales this Mob alone, so a Group's children
            keep their own sizes.

        Returns
        -------
        :class:`~.Mob`
            This Mob, so calls can be chained.
        """
        # Calculate the new absolute scale coefficient
        scale_factor = cast_to_tensor(scale_factor)
        new_scale = scale_factor * self.scale_coefficient
        # Use the 'set' method to apply the new scale coefficient, which handles animation and recursion
        return (
            self.set(scale_coefficient=new_scale)
            if recursive
            else self.set_non_recursive(scale_coefficient=new_scale)
        )

    def set_scale(self, scale: float | torch.Tensor, recursive: bool = True) -> Mob:
        """Set the Mob's absolute scale, ignoring its current size.

        ``set_scale(1)`` returns the Mob to the size it was built at, whatever
        scaling has happened since. For a relative change, use
        :meth:`~.Mob.scale`.

        Animation
        ---------
        Recorded as an animation over the current context's duration (1 second by
        default).

        Parameters
        ----------
        scale
            Target scale, where ``1`` is the Mob's construction size. A tensor of
            shape ``(*, 3)`` sets the Mob's right, up and forward axes
            separately.
        recursive
            Whether descendants are scaled too. Defaults to True.

        Returns
        -------
        :class:`~.Mob`
            This Mob, so calls can be chained.
        """
        return (
            self.set(scale_coefficient=scale)
            if recursive
            else self.set_non_recursive(scale_coefficient=scale)
        )

    def refresh_history(self):
        """Clear this Mob's recorded spawn so it counts as never spawned.

        Resets the lifespan of this Mob and every descendant, which makes them
        behave as if freshly constructed. Mostly useful as part of
        :meth:`~.Mob.detach_history`; calling it on a live Mob leaves the Mob
        visible in already-recorded animation while claiming it was never
        spawned, so reach for it only if you know you want that.

        Animation
        ---------
        Not animated and not recorded. Takes effect immediately on the timeline.
        """
        for mob in self.get_descendants():
            mob.lifespan.start = lambda: -1

    def detach_history(self) -> Mob:
        """Hand this Mob's recorded animation to a hidden clone and start fresh.

        The animation recorded so far keeps playing -- it now belongs to a clone
        that despawns at this moment -- while this Mob continues with a clean
        history from here. Use it before a change that cannot be interpolated
        from the old value, because the two states have different shapes: for
        example raising a :class:`~.Surface`'s resolution, or a ``become`` between
        Mobs with different numbers of parts. Without detaching, the render-time
        replay tries to interpolate mismatched shapes and raises.

        Animation
        ---------
        Not animated: the swap happens instantly, inside ``Off()``, and the
        viewer sees no discontinuity. Everything recorded afterwards animates
        from the Mob's state at this moment.

        Returns
        -------
        :class:`~.Mob`
            This Mob, so calls can be chained.
        """
        detach_time = self.animation_manager.context.timespan.current_time
        with (
            Off(animation_manager=self.animation_manager),
            NoExtra(priority_level=1, animation_manager=self.animation_manager),
        ):
            clone_mob = self.clone(clone_data=True, spawn=False)
            descendant_map = dict(
                zip(self.get_descendants(), clone_mob.get_descendants())
            )

            # Hand this mob's recorded history over to the clone. All recorded
            # attribute edits reference this mob's current rows in the global
            # attribute timelines, so the clone takes ownership of those rows,
            # while this mob keeps the fresh rows allocated during cloning
            # (which hold the current values and no history). Past function
            # applications are re-targeted at the clone so that at render time
            # they replay onto the old rows.
            timeline = self.scene.timeline_manager
            for orig, clone in descendant_map.items():
                for attr_timeline in timeline.attr_to_timeline.values():
                    orig_inds = attr_timeline.mob_id_to_inds.get(orig.id)
                    if orig_inds is None:
                        continue
                    clone_inds = attr_timeline.mob_id_to_inds.get(clone.id)
                    if clone_inds is None:
                        # The clone allocated no rows for this attr; give it
                        # the old rows and allocate fresh rows (holding the
                        # current values) for the original.
                        attr_timeline.drop_mob(orig.id)
                        attr_timeline.reassign_inds(clone.id, orig_inds)
                        attr_timeline.add(
                            orig, attr_timeline.get(orig_inds), overwrite=True
                        )
                    else:
                        # Swap the row ownership. These reassignments must go
                        # through reassign_inds so the cached RowRanges
                        # (AttributeTimeline.ranges_for / mob_id_to_ranges) is
                        # invalidated -- otherwise later function replays resolve
                        # these ids to their pre-swap rows and write to the wrong
                        # place (e.g. a second become() morphing onto the wrong
                        # rows).
                        attr_timeline.reassign_inds(orig.id, clone_inds)
                        attr_timeline.reassign_inds(clone.id, orig_inds)
            # Only events recorded against this subtree can move, so they are
            # looked up by caller rather than by scanning every event authored
            # so far (see FunctionTimeline._by_caller).
            function_timeline = timeline.function_timeline
            for orig, clone in descendant_map.items():
                for f in list(function_timeline.events_for_caller(orig)):
                    # Functions without captured row edits replay against the
                    # caller's current topology.  If such a function begins at
                    # the detach boundary, it belongs to the replacement mob,
                    # not the historical clone whose lifespan ends there.
                    if not f.recorded_edits and f.time.start >= detach_time:
                        continue
                    function_timeline.retarget_caller(f, clone)

            for orig, clone in descendant_map.items():
                # The clone inherits the original's spawn time (this mob is
                # re-spawned at the current time below).
                clone.lifespan.start = orig.lifespan.start
                timeline.register_spawn(clone, clone.lifespan)
            clone_mob.despawn(animate=False)
            self.refresh_history()
            self.spawn(animate=False)
            timeline.register_updater_history_split(descendant_map)
            return self

    # Public properties that ``set`` would technically accept but that are
    # plumbing rather than animatable state.
    _NON_SETTABLE_PUBLIC_PROPERTIES = frozenset({"animation_manager"})

    # Timeline-backed attributes that are nonetheless a hierarchical transform:
    # a rotation or a scale has to carry the subtree's locations with it, which
    # is what the property setter does and what a flat per-row write cannot.
    # Measured: rotating every basis row of a Square by 45 degrees leaves its
    # corners exactly where they were and moves only its texture frame, because
    # a bezier circuit's geometry is its control points' locations, not its
    # frame.
    _HIERARCHICAL_ATTRIBUTES = frozenset({"basis"})

    # What to reach for instead when one of these is addressed row-wise.
    _NOT_ROW_WISE_HINTS = {
        "basis": (
            "Mob.rotate, Mob.scale and 'mob.basis = ...' all carry the subtree "
            "with them"
        ),
        "scale_coefficient": (
            "Mob.scale / Mob.set_scale resize a whole subtree, which is what "
            "changing a scale means"
        ),
    }

    def _writes_through_property_setter(self, attr: str) -> bool:
        """Whether a by-name write to ``attr`` has to go through its property
        setter instead of straight to its timeline rows.

        Two kinds qualify, and the by-name API gets both wrong if it writes
        rows directly: a *derived* property has no rows at all, and a
        *hierarchical* one has rows that are only half the operation.
        """
        return attr in self._HIERARCHICAL_ATTRIBUTES or self._is_derived_attribute(attr)

    def _is_derived_attribute(self, attr: str) -> bool:
        """Whether ``attr`` is computed from other animatable attributes rather
        than owning timeline rows of its own.

        ``scale_coefficient`` (the row norms of ``basis``), ``Circle.radius``
        and ``border_color`` are all of this kind: assigning to one works,
        because the property setter forwards the write to whatever really
        stores it, but there is no buffer for the by-name attribute API to
        address. A name that is not a settable property at all is not derived,
        just wrong, and is left to fail where it always did.
        """
        if attr in self.scene.timeline_manager.attr_to_timeline:
            return False
        member = getattr(type(self), attr, None)
        if not isinstance(member, property) or member.fset is None:
            return False
        # The property generated for a registered animatable attribute forwards
        # to set_animated_attribute itself, and its timeline is created by the
        # first write -- so it looks derived right up until it is not.
        return not getattr(member.fset, "_forwards_to_set_animated_attribute", False)

    def _raise_not_row_wise_error(self, caller: str, attr: str):
        if attr in self._HIERARCHICAL_ATTRIBUTES:
            what = (
                f"'{attr}' is a hierarchical transform -- writing its rows "
                f"directly changes each frame without carrying the subtree's "
                f"locations with it, so the geometry would not actually move"
            )
        else:
            what = (
                f"'{attr}' is a derived property -- it is computed from other "
                f"animatable attributes rather than stored per row, so there "
                f"are no values to map"
            )
        hint = self._NOT_ROW_WISE_HINTS.get(
            attr,
            f"assigning to it (mob.{attr} = ...) does work, since the property "
            f"setter forwards the write",
        )
        raise AttributeError(f"{caller}({attr!r}, ...): {what}. {hint}.")

    def _settable_property_names(self) -> set[str]:
        """Names that :meth:`~.Mob.set` can meaningfully write on this Mob.

        The registered animatable attributes, plus every public property with a
        setter anywhere in this Mob's MRO. That second half is what surfaces
        derived properties such as ``border_color``, which forwards to the
        border texture instead of owning a timeline of its own -- ``set`` has
        always accepted those, but never used to name them.

        The MRO half is a property of the *class*, so it is cached per class
        against the registration counter that
        :meth:`~algan.animatable_base.animatable.Animatable.register_attrs_as_animatable`
        bumps whenever it attaches another one -- walking every class dict in
        the MRO on every ``set`` call was one of the more expensive things a
        scene did while it was being authored.
        """
        version = ANIMATABLE_PROPERTY_VERSION[0]
        klass = type(self)
        cached = _SETTABLE_PROPERTY_CACHE.get(klass)
        if cached is None or cached[0] != version:
            names = set()
            for base in klass.__mro__:
                for name, member in list(vars(base).items()):
                    if (
                        name.startswith("_")
                        or name in self._NON_SETTABLE_PUBLIC_PROPERTIES
                    ):
                        continue
                    if isinstance(member, property) and member.fset is not None:
                        names.add(name)
            cached = (version, names)
            _SETTABLE_PROPERTY_CACHE[klass] = cached
        return cached[1] | set(self.animatable_attrs)

    def check_properties_are_valid(self, property_names):
        """Raise if any of the given names is not an animatable attribute.

        Called by :meth:`~.Mob.set` so that a typo such as
        ``mob.set(colour=BLUE)`` fails immediately with the list of available
        attributes, instead of silently animating nothing.

        Parameters
        ----------
        property_names
            Iterable of attribute names to check.

        Raises
        ------
        AttributeError
            If any name is neither an attribute of this Mob nor a registered
            animatable attribute; the message lists what is available.
        """
        settable = self._settable_property_names()
        # attr_to_timeline is Scene-wide, so it also accepts an attribute owned
        # by some other Mob in the Scene (where setting it does nothing). It
        # gates the check as it always has, but is not advertised in the message.
        accepted = {
            *settable,
            *self.scene.timeline_manager.attr_to_timeline.keys(),
        }
        for p in property_names:
            if hasattr(self, p) or p in accepted:
                continue
            suggestion = difflib.get_close_matches(p, sorted(settable), n=1)
            hint = f" Did you mean '{suggestion[0]}'?" if suggestion else ""
            raise AttributeError(
                f'"{p}" is not recognized as an animatable Mob property.'
                f"{hint} Available properties are: "
                f"{', '.join(sorted(settable))}."
            )

    def set_non_recursive(self, **kwargs) -> Mob:
        """Set attributes on this Mob only, leaving its children untouched.

        The non-propagating counterpart of :meth:`~.Mob.set`. Use it when a
        parent's own value should change while its children keep theirs -- for
        instance recolouring a Group's frame without recolouring its contents.

        Animation
        ---------
        Recorded as an animation: every attribute given moves to its new value
        together inside a :class:`~.Sync`, over the current context's duration
        (1 second by default).

        Parameters
        ----------
        **kwargs
            Animatable attribute names and their target values, e.g.
            ``mob.set_non_recursive(color=BLUE, opacity=0.5)``.

        Returns
        -------
        :class:`~.Mob`
            This Mob, so calls can be chained.
        """
        prs = self._prevent_recursive_sets
        self._prevent_recursive_sets = True
        self.set(**kwargs)
        self._prevent_recursive_sets = prs
        return self

    def set(self, **kwargs) -> Mob:
        """Set several animatable attributes at once, as one animation.

        The attributes change together rather than one after another, which is
        what you want for a single visual beat: ``mob.set(location=RIGHT,
        color=BLUE)`` slides and recolours in the same second, where two separate
        statements would take two seconds inside a :class:`~.Seq`.

        Animation
        ---------
        Recorded as an animation: all the writes go into one :class:`~.Sync`
        spanning the current context's duration (1 second by default). Changes
        propagate to descendants; use :meth:`~.Mob.set_non_recursive` if they
        should not.

        Parameters
        ----------
        **kwargs
            Animatable attribute names and their target values, e.g. ``location``,
            ``color``, ``opacity``, ``glow``, ``scale_coefficient``.

        Returns
        -------
        :class:`~.Mob`
            This Mob, so calls can be chained.

        Raises
        ------
        AttributeError
            If a name is not an animatable attribute of this Mob. The message
            lists the ones that are.

        Examples
        --------
        Move a square to the right and change its color to blue:

        .. algan:: Example1MobSet

            from algan import *

            mob = Square().spawn()
            mob.set(location=ORIGIN+RIGHT, color=BLUE)

            Scene.save_video()

        """
        _reject_context_kwargs(kwargs)
        self.check_properties_are_valid(kwargs.keys())
        with Sync(animation_manager=self.animation_manager):
            for key, value in kwargs.items():
                self.__setattr__(
                    key, value
                )  # Calls the property setters, which handle animation and recursion
        return self

    def on_create(self):
        """Play this Mob's spawn-in animation: a fade from transparent.

        Called by :meth:`~.Animatable.spawn` when ``animate=True``. Override it in
        a subclass to give a Mob its own entrance; the override should record its
        animation the same way, and is free to ignore opacity entirely.

        Animation
        ---------
        Recorded as an animation over the current context's duration (1 second by
        default). The opacity write is non-recursive, so descendants run their own
        ``on_create``.
        """
        opacity = self.opacity
        with Seq(animation_manager=self.animation_manager):
            prs = self._prevent_recursive_sets
            self._prevent_recursive_sets = True
            with Off(animation_manager=self.animation_manager):
                self.opacity = 0
            self.opacity = opacity
            self._prevent_recursive_sets = prs

    # Spawning a subtree records this entrance once for the whole set rather
    # than once per Mob (Animatable._collated_fade_in). The marker is what
    # says "this hook is a plain opacity write, so it can be collated"; an
    # override does not inherit it and keeps its own per-Mob call.
    on_create._algan_collatable_hook = True

    def on_destroy(self):
        """Play this Mob's despawn animation: a fade to transparent.

        Called by :meth:`~.Animatable.despawn` when ``animate=True``. Override it
        in a subclass to give a Mob its own exit.

        Animation
        ---------
        Recorded as an animation over the current context's duration (1 second by
        default).
        """
        self.opacity = torch.tensor((0.0,)).view(1)

    # See on_create: despawning a subtree collates these into one recorded
    # animation. Note this write is *recursive*, so before collation a Mob at
    # depth d had its rows written d+1 times over -- once by every ancestor's
    # exit as well as its own.
    on_destroy._algan_collatable_hook = True

    def _set_data_sub_inds(self, data_sub_inds: list[int] | slice):
        """Internal: restrict this Mob to a subset of a batched Mob's rows.

        This is the machinery behind indexing (``mob[2]``, ``mob[1:4]``); prefer
        :meth:`~algan.animatable_base.mob.Mob.__getitem__`, which sets this up for
        you. The sub-indices
        select which elements of the shared attribute data this Mob reads and
        writes, and are applied to its children too.

        Parameters
        ----------
        data_sub_inds
            The indices or slice to apply to the batch dimension of the
            shared data tensors.

        """
        self.batch_size = max(self.batch_size, self.location.shape[1])
        if self.parent_batch_sizes is not None:
            if self.singleton_batch_indexing and len(self.parent_batch_sizes) == 1:
                self.parent_batch_sizes = torch.ones(
                    (self.parent_batch_sizes.item(),), dtype=torch.long
                )
            sub_pbs = self.parent_batch_sizes[data_sub_inds]
            inds = torch.arange(self.batch_size).split(
                [_.item() for _ in self.parent_batch_sizes]
            )
            data_sub_inds = torch.cat(
                [inds[d] for d in data_sub_inds]
                if not isinstance(data_sub_inds, slice)
                else inds[data_sub_inds]
            )
        else:
            sub_pbs = self.parent_batch_sizes
        self.data_sub_inds = data_sub_inds
        self.parent_batch_sizes = sub_pbs
        for c in self.children:
            c._set_data_sub_inds(data_sub_inds)

    def __len__(self):
        """Number of logical objects in this Mob's batch, or 0 if it is not batched."""
        parent_batch_sizes = getattr(self, "parent_batch_sizes", None)
        if parent_batch_sizes is None:
            return 0
        # A packer that gives every member exactly one row records the count
        # compressed into a single entry, which _set_data_sub_inds expands on
        # the first index. Reading shape[-1] off that reports 1 member however
        # many there are -- the reason Tex carries its own __len__.
        if getattr(self, "singleton_batch_indexing", False) and (
            parent_batch_sizes.shape[-1] == 1
        ):
            return int(parent_batch_sizes.item())
        return parent_batch_sizes.shape[-1]

    def __getitem__(self, item: int | slice) -> Mob:
        """Get part of a batched Mob by index or slice, as a Mob.

        ``text[0]`` is the first glyph, ``text[1:4]`` the next three -- animate
        them and only those parts move. The result is a **view**: it shares the
        original's animation data and identity, so animating the slice animates
        the original's parts, and it shares the original's lifespan rather than
        needing its own spawn.

        Animation
        ---------
        Not animated: indexing only creates the view. Animate the returned Mob to
        move the parts it covers.

        Parameters
        ----------
        item
            Index of a single part, or a slice selecting several.

        Returns
        -------
        :class:`~.Mob`
            A Mob covering the selected parts, sharing data with this one.
        """
        if len(self) == 0:
            raise TypeError("Mob object is not iterable")
        # Clone the mob without cloning its data, but recursively for children structure
        cloned_mob = self.clone(
            add_to_scene=False, clone_data=False, recursive=True, animate_creation=False
        )
        # Set the data sub-indices for the cloned mob to point to the desired batch elements
        cloned_mob._set_data_sub_inds([item] if isinstance(item, int) else item)
        return cloned_mob
