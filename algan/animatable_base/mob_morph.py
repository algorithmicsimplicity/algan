"""Morphing (``become``) and structural batch-alignment machinery."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from algan.animatable_base.animatable import _rejecting_timing_kwargs
from algan.animation_timeline.animation_contexts import NoExtra, Off, Seq, Sync
from algan.animation_timeline.timeline import bump_hierarchy_version
from algan.constants import easings
from algan.errors import AlganConfigurationError
from algan.logging.logger import PERF, get_logger
from algan.utils.tensor_utils import cast_to_tensor, mid_point, squish, unsquish

if TYPE_CHECKING:
    from algan.animatable_base.mob import Mob


def linear_sum_assignment(*args, **kwargs):
    """Import SciPy's assignment solver only when movement minimization is used."""
    from scipy.optimize import linear_sum_assignment as _lsa

    return _lsa(*args, **kwargs)


def align_part_lists(mine, theirs, *, make_placeholder) -> tuple[list, list]:
    """Evenly align two part lists, filling shortfall from the counterpart.

    The first occurrence of every original short-side part is retained.  Slots
    which the legacy evenly-spread repeat formula would have filled with a
    duplicate instead receive ``make_placeholder(long_side_part, side=...)``.
    This handles an empty side as naturally as any other size.
    """
    mine = list(mine)
    theirs = list(theirs)
    target = max(len(mine), len(theirs))
    if target == 0:
        return mine, theirs

    def pad(short, long, side):
        current = len(short)
        if current == target:
            return short
        if current == 0:
            return [make_placeholder(counterpart, side=side) for counterpart in long]

        repeat_indices = [(slot * current) // target for slot in range(target)]
        seen = set()
        aligned = []
        for slot, source_index in enumerate(repeat_indices):
            if source_index not in seen:
                aligned.append(short[source_index])
                seen.add(source_index)
            else:
                aligned.append(make_placeholder(long[slot], side=side))
        return aligned

    return pad(mine, theirs, "mine"), pad(theirs, mine, "theirs")


def _identity_contains(values, item):
    return any(value is item for value in values)


def _link_stand_in(stand_in, author, *, carry_tag=True, within=None):
    """Record that ``stand_in`` draws on behalf of ``author`` (object-ID pass).

    A morph renders through Mobs the author never holds -- a triangle-soup
    stand-in, a target-class replacement spliced into a source's slot, a
    dissolve's replacement -- and
    :func:`algan.rendering.pass_identity.render_identity` follows these links
    so one object keeps one ID across a ``become``. The first link wins (a
    history clone's is set by ``detach_history``). Returns whether
    ``stand_in`` now draws for ``author``.

    A pure grouping node is not linked to anything that is, or sits inside, a
    drawn object: it has no identity of its own to hand on, and linked, every
    member still hanging from it -- a surplus child shrinking away -- would be
    reported as part of that object. Only what the ``become`` produced counts:
    ``within`` is the highest node looked at (``author`` alone by default), so
    a composite the source already sat in does not stop the link.

    With ``carry_tag`` an explicit ``pass_index`` on ``stand_in`` moves to
    ``author`` (linked or not), overriding the author's own: the Mob that
    becomes something is still the one the author tagged, as on the
    same-kind route. Callers whose author may already hold a nearer source's
    tag check that first (:meth:`MobMorphMixin._hand_on_structure_identity`).
    Private attributes only: nothing renders differently.
    """
    from algan.rendering.pass_identity import is_container

    if stand_in is author:
        return True
    if getattr(stand_in, "_pass_identity", None) is None and (
        not is_container(stand_in) or not _drawn_within(author, within)
    ):
        stand_in._pass_identity = author
    if carry_tag and getattr(stand_in, "_pass_index", None) is not None:
        author._pass_index = stand_in._pass_index
    return getattr(stand_in, "_pass_identity", None) is author


def _drawn_within(author, within):
    """Whether ``author``, or a node above it up to ``within``, is drawn."""
    from algan.rendering.pass_identity import (
        is_container,
        pass_parents,
        render_identity,
    )

    node, seen = render_identity(author), set()
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        if not is_container(node):
            return True
        if within is None or node is within:
            return False
        parents = pass_parents(node)
        node = render_identity(parents[0]) if parents else None
    return False


def _keep_pass_parents(root, parents):
    """Let a ``become()`` root that nothing stands in for resolve upward.

    The morph takes ``root`` out of ``parents`` (its place goes to the
    result), but a root no Mob draws for -- a Group that became one of its own
    members -- still holds members, the ones shrinking away, and those must
    still reach the tags and the owner above it
    (:func:`algan.rendering.pass_identity.pass_parents`).
    """
    if getattr(root, "_pass_identity", None) is None and parents:
        root._pass_parents = tuple(parents)


class MobMorphMixin:
    """``become`` plus the structural helpers used to align arbitrary Mobs."""

    @staticmethod
    def _collect_morph_primitives(root):
        """Return renderable morph units, treating structural Mobs as transparent.

        A renderer primitive owns its geometry (including structural components
        such as a Surface's grid), so traversal stops there. So does an
        aggregate that draws its own descendants (``draws_descendants``, in
        practice Polyhedron): its faces are internals rather than Mobs to pair,
        and the vertex-and-edge graph beside them is geometry it never draws at
        all. Plain Mobs and Groups do not consume a pairing slot.
        """
        primitives = []

        def visit(mob):
            # A Mob that draws its descendants is one unit however many children
            # it keeps: a Polyhedron's twelve faces arrive under a single
            # ``_mesh_key`` and are its internals, not twelve Mobs to pair.
            # Pairing them separately published each face to the Scene as well,
            # so the Polyhedron and the face both drew it -- and did the same
            # for the vertex-and-edge graph the Polyhedron never draws at all.
            if mob._morph_family is not None and (
                mob.is_primitive or mob.draws_descendants
            ):
                primitives.append(mob)
                return
            before = len(primitives)
            for child in mob.get_non_component_children():
                visit(child)
            if mob._morph_family is not None and len(primitives) == before:
                # A custom family may own renderer geometry without opting into
                # the legacy is_primitive marker. Keep it morphable rather than
                # silently dropping it from the hierarchy plan.
                primitives.append(mob)

        visit(root)
        return primitives

    @staticmethod
    def _morph_center(mob):
        center = mob.get_center()
        return center.reshape(-1, center.shape[-1]).mean(0).detach().float().cpu()

    @staticmethod
    def _primitive_compatibility_rank(source, target):
        """Lexicographic pairing preference before position/order costs.

        Each band is doubled to leave room for a **structural penalty**: a pair
        that crosses an untravellable attribute (``filled``, in practice)
        cannot show the crossing happening, so it is worth one whole band of
        compatibility to avoid making that pair at all. Three unfilled frames
        becoming three of five unfilled frames is what this buys -- the
        assignment used to hand one of them to a filled Circle purely on
        position, and the Circle then lost its fill on the morph's first frame.
        """
        rank = MobMorphMixin._primitive_family_rank(source, target)
        return rank * 2 + int(source._morph_structural_break(target))

    @staticmethod
    def _primitive_family_rank(source, target):
        """How compatible two primitives are by type and family alone."""
        if type(source) is type(target):
            return 0
        if source._morph_family == target._morph_family:
            return 1
        if "image" in {source._morph_family, target._morph_family}:
            return 4

        from algan.animatable_base.morph_conversions import get_morph_conversion

        if (
            get_morph_conversion(source._morph_family) is not None
            and get_morph_conversion(target._morph_family) is not None
        ):
            return 2
        return 3

    @staticmethod
    def _morph_extent(mob):
        size = mob.get_bounding_box_size()
        return float(size.reshape(-1, size.shape[-1]).mean(0).norm())

    #: How the default assignment weighs the three things that can distinguish
    #: one candidate pairing from another. They sum to 1 and each term is
    #: normalized to ``[0, 1]``, which is the whole point: the previous rule
    #: added a distance capped at ``1e-3`` to an order gap spanning ``[0, 1]``,
    #: so the geometry could only break exact ties and the assignment was
    #: traversal order and nothing else. Order still leads, because it is what
    #: makes ``Text("abc") -> Text("abd")`` pair a with a; position is close
    #: behind, because a Group's child order is often unrelated to its layout;
    #: size is a lighter nudge toward keeping a big part big.
    _PAIR_ORDER_WEIGHT = 0.35
    _PAIR_POSITION_WEIGHT = 0.5
    _PAIR_SIZE_WEIGHT = 0.15

    #: A size ratio of one decade saturates the size term. Beyond that the two
    #: parts are simply incomparable and a bigger number should not keep buying
    #: influence over position.
    _PAIR_SIZE_DECADE = 2.302585092994046  # log(10)

    #: Largest PN soup, in triangles, that a cross-family morph will pair by
    #: proximity before falling back to build order. See ``_record_pn_morph``
    #: for the timings this was chosen from.
    _REORDER_TRIANGLE_CAP = 2500

    def _primitive_pair_cost(
        self,
        source,
        target,
        *,
        source_index,
        target_index,
        source_count,
        target_count,
        minimize_movement,
        scene_span=None,
        geometry=None,
    ):
        # ``geometry`` memoizes each Mob's centre and extent by identity.
        # Without it every cell of an S x T cost matrix walks both subtrees --
        # twice over, since the size term needs the extent as well -- so a
        # hierarchy of a few dozen parts pays thousands of subtree walks and a
        # ``.cpu()`` sync each. The caller fills it once per assignment.
        source_center, source_extent = self._pair_geometry(source, geometry)
        target_center, target_extent = self._pair_geometry(target, geometry)
        compatibility = self._primitive_compatibility_rank(source, target)
        distance = float((source_center - target_center).norm())
        if minimize_movement:
            # The explicit opt-in stays what it was: pure proximity, so a caller
            # who asks for the least motion gets exactly that.
            secondary = distance
        else:
            source_position = source_index / max(source_count - 1, 1)
            target_position = target_index / max(target_count - 1, 1)
            order_gap = abs(source_position - target_position)
            position_gap = min(distance / (scene_span or 1.0), 1.0)
            size_gap = min(
                abs(math.log(max(target_extent, 1e-4) / max(source_extent, 1e-4)))
                / self._PAIR_SIZE_DECADE,
                1.0,
            )
            secondary = (
                self._PAIR_ORDER_WEIGHT * order_gap
                + self._PAIR_POSITION_WEIGHT * position_gap
                + self._PAIR_SIZE_WEIGHT * size_gap
            )
        return compatibility * 1e6 + secondary

    @classmethod
    def _pair_geometry(cls, mob, cache=None):
        """``(centre, extent)`` for one Mob, memoized by identity when asked."""
        if cache is not None and id(mob) in cache:
            return cache[id(mob)]
        value = (cls._morph_center(mob), cls._morph_extent(mob))
        if cache is not None:
            cache[id(mob)] = value
        return value

    @classmethod
    def _pairing_scene_span(cls, sources, targets):
        """Distance the position term is measured against.

        The spread of the parts being paired, not the frame: what "far" means
        in a morph is relative to the thing morphing, so a diagram spanning ten
        units and a glyph spanning one tenth get the same range of position
        costs and the same balance against order and size.
        """
        centers = [cls._morph_center(mob) for mob in [*sources, *targets]]
        if not centers:
            return 1.0

        stacked = torch.stack(centers)
        return max(float((stacked.amax(0) - stacked.amin(0)).norm()), 1e-3)

    def _pair_primitive_indices(self, sources, targets, minimize_movement):
        if not sources or not targets:
            return [], list(range(len(sources))), list(range(len(targets)))

        geometry = {}
        scene_span = (
            None if minimize_movement else self._pairing_scene_span(sources, targets)
        )
        costs = torch.empty((len(sources), len(targets)), dtype=torch.float64)
        for source_index, source in enumerate(sources):
            for target_index, target in enumerate(targets):
                costs[source_index, target_index] = self._primitive_pair_cost(
                    source,
                    target,
                    source_index=source_index,
                    target_index=target_index,
                    source_count=len(sources),
                    target_count=len(targets),
                    minimize_movement=minimize_movement,
                    scene_span=scene_span,
                    geometry=geometry,
                )
        source_indices, target_indices = linear_sum_assignment(costs.numpy())
        pairs = sorted(
            [
                (int(source_index), int(target_index))
                for source_index, target_index in zip(source_indices, target_indices)
            ],
            key=lambda pair: pair[1],
        )
        paired_sources = {source_index for source_index, _ in pairs}
        paired_targets = {target_index for _, target_index in pairs}
        unmatched_sources = [
            index for index in range(len(sources)) if index not in paired_sources
        ]
        unmatched_targets = [
            index for index in range(len(targets)) if index not in paired_targets
        ]
        return pairs, unmatched_sources, unmatched_targets

    @staticmethod
    def _nearest_geometry_point(sources, target, fallback):
        """Choose an existing source point from which fresh geometry can grow."""
        target_center = target.get_center()
        target_center = target_center.reshape(-1, target_center.shape[-1]).mean(0)
        best_distance = float("inf")
        best_point = None
        for source in sources:
            for descendant in source.get_descendants():
                if not hasattr(descendant, "location"):
                    continue
                location = descendant.location
                points = location.reshape(-1, location.shape[-1])
                if points.numel() == 0:
                    continue
                distances = (points - target_center.to(points)).square().sum(-1)
                point_index = distances.argmin()
                distance = float(distances[point_index].detach().cpu())
                if distance < best_distance:
                    best_distance = distance
                    best_point = points[point_index].detach().clone()
        return fallback if best_point is None else best_point

    @staticmethod
    def _is_stroke_only(mob):
        """A circuit that draws a stroke and no fill.

        Its PN conversion is honest and empty: ``_bezier_to_pn_soup`` converts
        the *interior*, and an unfilled circuit has none, so the soup it
        produces is fully transparent.
        """
        if mob._morph_family != "bezier":
            return False
        return not getattr(mob, "filled", True) or bool(getattr(mob, "empty", False))

    @classmethod
    def _pair_wants_crossfade(cls, source, target):
        """Whether a cross-family pair should dissolve rather than travel.

        The PN medium carries fills. When either end is a stroke-only circuit --
        an ``Arrow``, a ``Line``, an ``Axes``, an unfilled ``Square`` -- its
        side of the soup is transparent, so a geometric morph has nothing to
        show: the solid fades to nothing, several frames are blank, and the
        outline pops in at the end. A cross-fade is what the pair actually
        looks like, so route it there instead of travelling through an empty
        medium. Only for ``strategy="auto"``: a caller who explicitly asks for
        ``"morph"`` has asked for the geometric route and gets it.
        """
        return cls._is_stroke_only(source) or cls._is_stroke_only(target)

    @staticmethod
    def _pair_supports_geometric_morph(source, target):
        if "image" in {source._morph_family, target._morph_family}:
            return False
        if source._morph_kind == target._morph_kind:
            return True

        from algan.animatable_base.morph_conversions import get_morph_conversion

        return (
            get_morph_conversion(source._morph_family) is not None
            and get_morph_conversion(target._morph_family) is not None
        )

    @staticmethod
    def _collapse_hierarchy_at(mob, point):
        """Collapse fresh placeholder geometry without fading its material."""
        for descendant in mob.get_descendants():
            if not hasattr(descendant, "location"):
                continue
            location = descendant.location
            collapsed = point.to(location).expand_as(location).clone()
            descendant._setattr_and_rebatch_without_record("location", collapsed)
        return mob

    @staticmethod
    def _zero_hierarchy_opacity(mob):
        for descendant in mob.get_descendants():
            if hasattr(descendant, "opacity"):
                descendant._setattr_and_rebatch_without_record(
                    "opacity", torch.zeros_like(descendant.opacity)
                )
        return mob

    @staticmethod
    def _detach_from_parents(mob, *, preserve=()):
        preserve_ids = {id(parent) for parent in preserve}
        for parent in list(mob.parents):
            if id(parent) in preserve_ids:
                continue
            parent.replace_children(
                [child for child in parent.children if child is not mob]
            )
            parent.components = [
                component for component in parent.components if component is not mob
            ]
        return mob

    def _collapsed_child_placeholder(self, counterpart):
        placeholder = counterpart.clone(add_to_scene=False, spawn=False)
        own_parts = self.get_non_component_children()
        center = self.get_center() if not own_parts else counterpart.get_center()
        for mob in placeholder.get_descendants():
            location = mob.location
            collapsed = center.to(location).expand_as(location).clone()
            mob._setattr_and_rebatch_without_record("location", collapsed)
            if hasattr(mob, "opacity"):
                mob._setattr_and_rebatch_without_record(
                    "opacity", torch.zeros_like(mob.opacity)
                )
        return placeholder

    def _replace_non_component_children(self, parts):
        component_ids = {id(component) for component in self.components}
        replacement = []
        inserted = False
        for child in self.children:
            if id(child) in component_ids:
                replacement.append(child)
            elif not inserted:
                replacement.extend(parts)
                inserted = True
        if not inserted:
            replacement.extend(parts)
        self.replace_children(replacement)
        return self

    def _register_hierarchy_for_render(self, mob):
        """Publish ``mob`` to the Scene without publishing what an ancestor draws.

        Under a Mob that draws its descendants (``draws_descendants``, in
        practice ``Polyhedron``), a descendant published in its own right is
        drawn a second time, and one the aggregator deliberately omits -- the
        vertex-and-edge graph -- is drawn for the first time. That is what gave
        a morphed Polyhedron a wireframe and eight vertex beads and a doubled
        rim along every silhouette edge.

        The condition is ``draws_descendants`` and not merely "has
        ``get_render_primitives``": a ``BezierCircuitCubic`` has one and draws
        only its own rows, so its children must still be published. Withholding
        them dropped the tip off an ``Arrow`` grown as a placeholder inside a
        ``Line``.

        An aggregator speaks only for the subtrees it built for itself, which
        ``owned_subtrees`` names: a child a user hangs on a Polyhedron is not
        one of them, and withholding it made the user's geometry disappear.

        The walk seeds from what ``mob`` is already attached to rather than from
        scratch, because ``_expand_n_children`` makes a placeholder a child
        before registering it, and the aggregator that will draw it is an
        ancestor rather than anything inside the walk.
        """
        actors = self.scene.actors

        def owned_by(node):
            """Ids of the subtrees ``node`` speaks for, or None for all of them."""
            if not getattr(node, "draws_descendants", False):
                return None
            owned = node.owned_subtrees()
            if not owned:
                return "all"
            ids = set()
            for root in owned:
                ids.update(id(mob) for mob in root.get_descendants())
            return ids

        def spoken_for(node, owner_ids):
            return owner_ids == "all" or (
                owner_ids is not None and id(node) in owner_ids
            )

        def seed(node, seen=None):
            seen = set() if seen is None else seen
            for parent in getattr(node, "parents", ()):
                if id(parent) in seen:
                    continue
                seen.add(id(parent))
                owner_ids = owned_by(parent)
                if spoken_for(node, owner_ids):
                    return owner_ids
                inherited = seed(parent, seen)
                if inherited is not None:
                    return inherited
            return None

        def visit(node, owner_ids):
            draws_itself = hasattr(node, "get_render_primitives")
            if not (
                draws_itself and spoken_for(node, owner_ids)
            ) and not _identity_contains(actors, node):
                self.scene.add_actor(node)
            child_owner = owner_ids if owner_ids is not None else owned_by(node)
            for child in getattr(node, "children", ()):
                visit(child, child_owner)

        visit(mob, seed(mob))
        return mob

    def _expand_n_list(self, lst, n: int, counterparts=None) -> list:
        """Pad a list of point tensors with counterpart-shaped degenerates.

        ``counterparts`` is optional for callers of the historical helper
        signature.  ``become`` always supplies it, which lets a new path emerge
        from the existing point nearest that target path rather than from an
        arbitrary endpoint.
        """
        lst = list(lst)
        target_count = len(lst) + n
        if n <= 0:
            return lst
        if counterparts is None:
            if lst:
                counterparts = [
                    lst[(slot * len(lst)) // target_count]
                    for slot in range(target_count)
                ]
            else:
                return lst
        counterparts = list(counterparts)
        source_points = (
            torch.cat([value.reshape(-1, value.shape[-1]) for value in lst], dim=0)
            if lst
            else None
        )

        def make_placeholder(counterpart, *, side):
            if source_points is None:
                point = self.get_center().reshape(-1, self.get_center().shape[-1])[0]
                point = point.to(counterpart)
            else:
                centroid = counterpart.reshape(-1, counterpart.shape[-1]).mean(0)
                index = (source_points - centroid).square().sum(-1).argmin()
                point = source_points[index]
            return point.expand_as(counterpart).clone()

        aligned, _ = align_part_lists(
            lst, counterparts, make_placeholder=make_placeholder
        )
        return aligned

    def _expand_n_children(self, n: int, counterparts=None) -> Mob:
        """Add ``n`` counterpart-shaped, collapsed children.

        Padding a spawned source registers and instantly spawns the new parts;
        padding an unspawned target keeps every clone out of the actor registry.
        """
        parts = self.get_non_component_children()
        target_count = len(parts) + n
        if n <= 0:
            return self
        if counterparts is None:
            if parts:
                counterparts = [
                    parts[(slot * len(parts)) // target_count]
                    for slot in range(target_count)
                ]
            else:
                from algan.animatable_base.mob import Mob

                counterparts = [
                    Mob(
                        location=self.get_center(),
                        opacity=0,
                        scene=self.scene,
                        add_to_scene=False,
                    )
                    for _ in range(target_count)
                ]
        placeholders = []

        def make_placeholder(counterpart, *, side):
            placeholder = self._collapsed_child_placeholder(counterpart)
            placeholders.append(placeholder)
            return placeholder

        aligned, _ = align_part_lists(
            parts, counterparts, make_placeholder=make_placeholder
        )
        self._replace_non_component_children(aligned)
        if self.is_spawned() and not self.is_despawned():
            for placeholder in placeholders:
                self._register_hierarchy_for_render(placeholder)
                placeholder.spawn(animate=False)
        bump_hierarchy_version()
        return self

    def _expand_n_tensor(self, value: torch.Tensor, n: int) -> torch.Tensor:
        """Pad a path's segment tensor with contour-continuous degenerates.

        Repeated slots sit immediately after the source segment selected by the
        even alignment formula.  Collapse each one at that segment's end point:
        this preserves the source contour until the new target segment grows.
        Choosing an unrelated globally-nearest point can make the interpolated
        contour jump across itself even though each individual pairing is short.

        Deliberately blind to the counterpart it is padding towards, unlike its
        sibling ``_expand_n_list`` -- that is what the paragraph above is about.
        It used to *accept* a ``counterparts`` argument and ignore it, which
        read as an oversight rather than a decision; both call sites passed one.
        """
        if n <= 0:
            return value
        current = value.shape[-3]
        target = current + n
        if current == 0:
            return value
        repeat_indices = [((slot * current) // target) for slot in range(target)]
        seen = set()
        aligned = []
        for source_index in repeat_indices:
            segment = value.select(-3, source_index)
            if source_index in seen:
                segment = segment[..., -1:, :].expand_as(segment).clone()
            else:
                seen.add(source_index)
            aligned.append(segment)
        return torch.stack(aligned, dim=-3)

    def _expand_n_batch(self, n: int) -> Mob:
        """Grow an object batch by ``n``, re-batching all structural state."""
        if n <= 0:
            return self
        points_per_object = self.num_points_per_object
        current_batch_size = self.location.shape[-2] // points_per_object
        if current_batch_size <= 0:
            raise RuntimeError(
                "Cannot expand an empty object batch without a counterpart"
            )
        target_batch_size = current_batch_size + n
        repeat_indices = (
            torch.arange(target_batch_size, device=self.location.device)
            * current_batch_size
        ) // target_batch_size
        repeat_indices = repeat_indices.to(torch.long)
        repeat_list = repeat_indices.tolist()

        for attr in self.animatable_attrs:
            if not hasattr(self, attr):
                continue
            value = cast_to_tensor(getattr(self, attr))[0]
            if value.shape[-2] == 1:
                continue
            if value.shape[-2] % points_per_object:
                continue
            value_per_object = unsquish(value, -2, points_per_object)
            if value_per_object.shape[-3] != current_batch_size:
                continue
            seen = set()
            expanded = []
            for source_index in repeat_list:
                source = value_per_object[source_index]
                if source_index in seen:
                    source = source[..., -1:, :].expand(
                        *source.shape[:-2], points_per_object, source.shape[-1]
                    )
                else:
                    seen.add(source_index)
                expanded.append(source)
            self._setattr_and_rebatch_without_record(
                attr,
                squish(torch.stack(expanded, dim=-3), -3, -2).unsqueeze(0),
            )

        self._rebatch_structural_attrs(repeat_indices)
        for parent in list(self.parents):
            parent._rebatch_structural_attrs(repeat_indices, child=self)

        if self.parent_batch_sizes is not None:
            parent_batch_sizes = self.parent_batch_sizes
            if self.singleton_batch_indexing and len(parent_batch_sizes) == 1:
                self.parent_batch_sizes = torch.tensor(
                    (target_batch_size * points_per_object,),
                    dtype=parent_batch_sizes.dtype,
                    device=parent_batch_sizes.device,
                )
            else:
                objects_per_parent = parent_batch_sizes // points_per_object
                if (
                    bool((parent_batch_sizes % points_per_object != 0).any())
                    or int(objects_per_parent.sum()) != current_batch_size
                ):
                    raise RuntimeError(
                        "parent_batch_sizes does not describe the Mob's current batch"
                    )
                repeat_on_device = repeat_indices.to(parent_batch_sizes.device)
                if bool((objects_per_parent == 1).all()):
                    self.parent_batch_sizes = parent_batch_sizes.index_select(
                        0, repeat_on_device
                    )
                else:
                    parent_of_object = torch.repeat_interleave(
                        torch.arange(
                            len(parent_batch_sizes),
                            device=parent_batch_sizes.device,
                        ),
                        objects_per_parent,
                    )
                    expanded_parent = parent_of_object.index_select(0, repeat_on_device)
                    self.parent_batch_sizes = (
                        torch.bincount(
                            expanded_parent, minlength=len(parent_batch_sizes)
                        )
                        * points_per_object
                    ).to(parent_batch_sizes.dtype)
        return self

    def _reorder_batch_to_minimize_movement(self, target: Mob) -> Mob:
        """Re-pair this Mob's objects with the nearest target objects."""
        points_per_object = self.num_points_per_object
        my_points = unsquish(cast_to_tensor(self.location)[0], -2, points_per_object)
        target_points = unsquish(
            cast_to_tensor(target.location)[0], -2, points_per_object
        )
        num_objects = my_points.shape[-3]
        if num_objects <= 1 or target_points.shape[-3] != num_objects:
            return self
        distances = torch.cdist(target_points.mean(-2), my_points.mean(-2))
        target_inds, my_inds = linear_sum_assignment(distances.cpu().numpy())
        permutation = torch.empty(num_objects, dtype=torch.long)
        permutation[torch.as_tensor(target_inds, dtype=torch.long)] = torch.as_tensor(
            my_inds, dtype=torch.long
        )
        permutation = permutation.to(my_points.device)

        for attr in self.animatable_attrs:
            if not hasattr(self, attr):
                continue
            value = cast_to_tensor(getattr(self, attr))[0]
            if value.shape[-2] == 1 or value.shape[-2] % points_per_object:
                continue
            per_object = unsquish(value, -2, points_per_object)
            if per_object.shape[-3] != num_objects:
                continue
            self._setattr_and_rebatch_without_record(
                attr,
                squish(per_object.index_select(-3, permutation), -3, -2).unsqueeze(0),
            )
        self._reorder_structural_attrs(permutation)
        for parent in list(self.parents):
            parent._reorder_structural_attrs(permutation, child=self)
        return self

    def get_non_component_children(self) -> list[Mob]:
        """Return direct user children, excluding a Mob's structural components."""
        return [
            child
            for child in self.children
            if not _identity_contains(self.components, child)
        ]

    @staticmethod
    def _resample_surface_to(surface, width, height):
        old_width, old_height = surface.grid_width, surface.grid_height
        if (old_width, old_height) == (width, height):
            return surface

        # RESAMPLE THE SURFACE, NOT ITS SAMPLES, WHENEVER IT CAN SAY WHAT IT IS.
        # Reconciliation moves both sides to the finer grid, so whichever side
        # was coarser gets INTERPOLATED UP -- and interpolating a coarse grid
        # does not reproduce the surface those samples came from. A 6x6 plane
        # becoming a 4x9 wave used to end 0.161 away from the wave on a surface
        # one unit across, because the wave it ended on was a bilinear
        # re-sampling of four columns rather than the wave itself. A Surface
        # knows its own parametric function, and `_change_resolution` evaluates
        # it on the new grid (interpolating only the per-vertex attributes,
        # which is the honest thing to do with colors). Packed surfaces keep
        # the interpolating path: `_change_resolution` describes one grid, not a
        # block of them.
        if surface.grid.parent_batch_sizes is None:
            surface._change_resolution(int(width), int(height))
            return surface

        old_count = old_width * old_height
        new_count = width * height
        parent_sizes = surface.grid.parent_batch_sizes
        packed_count = None if parent_sizes is None else len(parent_sizes)
        for attr in dict.fromkeys(surface.grid.animatable_attrs):
            if not hasattr(surface.grid, attr):
                continue
            value = getattr(surface.grid, attr)
            expected = old_count if packed_count is None else packed_count * old_count
            if value.shape[-2] != expected:
                continue
            if packed_count is not None:
                value = value.reshape(
                    *value.shape[:-2], packed_count, old_count, value.shape[-1]
                )
            value = surface._resample_grid_value(
                value, old_width, old_height, width, height
            )
            if packed_count is not None:
                value = value.flatten(-3, -2)
            surface.grid._setattr_and_rebatch_without_record(attr, value)

        surface.grid_width = int(width)
        surface.grid_height = int(height)
        surface.resolution = (int(width) - 1, int(height) - 1)
        surface.__dict__.pop("_cached_base_grid", None)
        surface.__dict__.pop("_cached_base_grid_key", None)
        surface.grid.batch_size = new_count * (packed_count or 1)
        if parent_sizes is not None:
            surface.grid.parent_batch_sizes = torch.full_like(parent_sizes, new_count)
        surface._memory_per_timestep_cache = None
        return surface

    def _reconcile_grid_pair(self, mine, theirs):
        width = max(mine.grid_width, theirs.grid_width)
        height = max(mine.grid_height, theirs.grid_height)
        self._resample_surface_to(mine, width, height)
        self._resample_surface_to(theirs, width, height)

    def _align_cubic_geometry(self, mine, theirs):
        def get_sub_circuits(value):
            starts = (
                (
                    (value[..., 0, :] - value.roll(1, -3)[..., -1, :]).abs().sum(-1)
                    > 1e-6
                )
                .nonzero(as_tuple=False)
                .flatten()
                .tolist()
            )
            if not starts:
                return [value]
            return [
                value[
                    starts[index] : starts[index + 1]
                    if index + 1 < len(starts)
                    else value.shape[-3]
                ]
                for index in range(len(starts))
            ]

        def get_parent_circuits(mob):
            segments = unsquish(mob.location, -2, 4).squeeze(0)
            sizes = mob.parent_batch_sizes
            if sizes is None:
                return [segments]
            if (
                bool((sizes % 4 != 0).any())
                or int(sizes.sum()) != mob.location.shape[-2]
            ):
                raise RuntimeError(
                    "parent_batch_sizes does not match cubic control points"
                )
            return list(segments.split((sizes // 4).tolist(), dim=-3))

        had_parent_batches = (
            mine.parent_batch_sizes is not None or theirs.parent_batch_sizes is not None
        )
        my_parents = get_parent_circuits(mine)
        their_parents = get_parent_circuits(theirs)
        difference = len(their_parents) - len(my_parents)
        if difference > 0:
            my_parents = mine._expand_n_list(
                my_parents, difference, counterparts=their_parents
            )
        elif difference < 0:
            their_parents = theirs._expand_n_list(
                their_parents, -difference, counterparts=my_parents
            )

        my_parent_batches = []
        their_parent_batches = []
        parent_batch_sizes = []
        for my_parent, their_parent in zip(my_parents, their_parents):
            my_paths = get_sub_circuits(my_parent)
            their_paths = get_sub_circuits(their_parent)
            difference = len(their_paths) - len(my_paths)
            if difference > 0:
                my_paths = mine._expand_n_list(
                    my_paths, difference, counterparts=their_paths
                )
            elif difference < 0:
                their_paths = theirs._expand_n_list(
                    their_paths, -difference, counterparts=my_paths
                )

            aligned_mine = []
            aligned_theirs = []
            for my_path, their_path in zip(my_paths, their_paths):
                difference = their_path.shape[-3] - my_path.shape[-3]
                if difference > 0:
                    my_path = mine._expand_n_tensor(my_path, difference)
                elif difference < 0:
                    their_path = theirs._expand_n_tensor(their_path, -difference)
                aligned_mine.append(my_path)
                aligned_theirs.append(their_path)
            my_batch = torch.cat(aligned_mine, dim=-3)
            their_batch = torch.cat(aligned_theirs, dim=-3)
            my_parent_batches.append(my_batch)
            their_parent_batches.append(their_batch)
            parent_batch_sizes.append(my_batch.shape[-3] * 4)

        segment_count = sum(parent_batch_sizes) // 4
        my_count = mine.location.shape[-2] // 4
        their_count = theirs.location.shape[-2] // 4
        if segment_count > my_count:
            mine._expand_n_batch(segment_count - my_count)
        if segment_count > their_count:
            theirs._expand_n_batch(segment_count - their_count)
        mine._setattr_and_rebatch_without_record(
            "location",
            squish(torch.cat(my_parent_batches, dim=-3), -3, -2).unsqueeze(0),
        )
        theirs._setattr_and_rebatch_without_record(
            "location",
            squish(torch.cat(their_parent_batches, dim=-3), -3, -2).unsqueeze(0),
        )
        if had_parent_batches:
            reference = (
                mine.parent_batch_sizes
                if mine.parent_batch_sizes is not None
                else theirs.parent_batch_sizes
            )
            metadata = torch.tensor(
                parent_batch_sizes,
                dtype=reference.dtype,
                device=reference.device,
            )
            mine.parent_batch_sizes = metadata
            theirs.parent_batch_sizes = metadata.clone()

    def _prepare_same_kind_node(self, mine, theirs, minimize_movement):
        if mine._morph_family == "grid":
            self._reconcile_grid_pair(mine, theirs)

        my_children = mine.get_non_component_children()
        their_children = theirs.get_non_component_children()
        difference = len(their_children) - len(my_children)
        if difference > 0:
            mine._expand_n_children(difference, counterparts=their_children)
        elif difference < 0:
            theirs._expand_n_children(-difference, counterparts=my_children)

        if mine.num_points_per_object == 4:
            self._align_cubic_geometry(mine, theirs)
        else:
            difference = (
                theirs.location.shape[-2] - mine.location.shape[-2]
            ) // mine.num_points_per_object
            if difference > 0:
                mine._expand_n_batch(difference)
            elif difference < 0:
                theirs._expand_n_batch(-difference)
            if minimize_movement:
                theirs._reorder_batch_to_minimize_movement(mine)

    def _record_same_kind_morph(
        self,
        mine,
        theirs,
        *,
        minimize_movement,
        strategy,
        replacement_allowed,
    ):
        am = mine.animation_manager
        with Seq(animation_manager=am):
            with (
                Off(spawn_at_end=False, animation_manager=am),
                NoExtra(priority_level=1, animation_manager=am),
            ):
                self._prepare_same_kind_node(mine, theirs, minimize_movement)

            my_children = mine.get_non_component_children()
            their_children = theirs.get_non_component_children()
            with Sync(animation_manager=am):
                if minimize_movement and my_children:
                    my_locations = torch.stack(
                        [
                            mid_point(child.location, -2).squeeze()
                            for child in my_children
                        ]
                    )
                    their_locations = torch.stack(
                        [
                            mid_point(child.location, -2).squeeze()
                            for child in their_children
                        ]
                    )
                    row_inds, column_inds = linear_sum_assignment(
                        torch.cdist(my_locations, their_locations).cpu().numpy()
                    )
                    child_pairs = [
                        (my_children[i], their_children[j])
                        for i, j in zip(row_inds, column_inds)
                    ]
                else:
                    child_pairs = list(zip(my_children, their_children))

                for my_child, their_child in child_pairs:
                    self._dispatch_become(
                        my_child,
                        their_child,
                        minimize_movement=minimize_movement,
                        strategy=strategy,
                        replacement_allowed=replacement_allowed,
                    )
                for my_component, their_component in zip(
                    list(mine.components), list(theirs.components)
                ):
                    # Surface grid rows are a fixed UV topology, not an
                    # unordered point batch. Reordering a collapsed grid by
                    # distance makes every assignment equally good and
                    # scrambles the target's triangle adjacency.
                    component_minimize_movement = (
                        minimize_movement and mine._morph_family != "grid"
                    )
                    self._dispatch_become(
                        my_component,
                        their_component,
                        minimize_movement=component_minimize_movement,
                        strategy=strategy,
                        replacement_allowed=replacement_allowed,
                    )

                values = {
                    attr: getattr(theirs, attr)
                    for attr in mine.animatable_attrs
                    if hasattr(mine, attr) and hasattr(theirs, attr)
                }
                if values:
                    mine.set_non_recursive(**values)

            with (
                Off(spawn_at_end=False, animation_manager=am),
                NoExtra(priority_level=1, animation_manager=am),
            ):
                mine._adopt_structural_attrs(theirs)
        return mine

    def _splice_replacement(self, source, replacement, *, carry_identity=True):
        """Put ``replacement`` in every slot ``source`` holds in its parents.

        ``carry_identity`` also records that ``replacement`` now stands where
        ``source`` stood, for the object-ID pass (:func:`_link_stand_in`). It
        is False only where ``source`` is a target-side template being swapped
        for the morph result that already carries the source's identity.
        """
        parents = list(source.parents)
        if carry_identity:
            _link_stand_in(source, replacement)
            _keep_pass_parents(source, parents)
        for parent in parents:
            children = [
                replacement if child is source else child for child in parent.children
            ]
            parent.replace_children(children)
            parent.components = [
                replacement if component is source else component
                for component in parent.components
            ]
            for name, value in list(parent.__dict__.items()):
                if value is source:
                    object.__setattr__(parent, name, replacement)
        return replacement

    @staticmethod
    def _capture_parent_slots(source):
        slots = []
        for parent in list(source.parents):
            slots.append(
                (
                    parent,
                    [
                        index
                        for index, child in enumerate(parent.children)
                        if child is source
                    ],
                    [
                        index
                        for index, component in enumerate(parent.components)
                        if component is source
                    ],
                    [
                        name
                        for name, value in list(parent.__dict__.items())
                        if value is source
                    ],
                )
            )
        return slots

    @staticmethod
    def _fill_captured_parent_slots(slots, replacement):
        for parent, child_indices, component_indices, attr_names in slots:
            children = list(parent.children)
            for index in child_indices:
                children[index] = replacement
            parent.replace_children(children)

            components = list(parent.components)
            for index in component_indices:
                components[index] = replacement
            parent.components = components
            for name in attr_names:
                object.__setattr__(parent, name, replacement)
        return replacement

    @staticmethod
    def _fit_bbox(mob, reference):
        size = mob.get_bounding_box_size()
        target_size = reference.get_bounding_box_size().to(size)
        epsilon = torch.finfo(size.dtype).eps
        scale = torch.ones_like(size)
        valid = (size > epsilon) & (target_size > epsilon)
        scale[valid] = target_size[valid] / size[valid]
        mob.scale(scale)
        mob.move_center_to(reference.get_center())
        return mob

    def _prepare_piecewise_dissolve(self, source, replacement, target):
        source_parts = source.get_non_component_children()
        replacement_parts = replacement.get_non_component_children()
        target_parts = target.get_non_component_children()
        piecewise = (
            source._morph_family is None
            and target._morph_family is None
            and source_parts
            and target_parts
        )
        if not piecewise:
            self._fit_bbox(replacement, source)
            # A SCALAR, NOT ``zeros_like(root.opacity)``. This ``set`` recurses,
            # and on a packed Mob the root carries one opacity row per member
            # while its descendants carry one per point -- so a tensor shaped
            # like the root's meets a descendant of a different width and the
            # subtraction that records the change raises. Every dissolve into a
            # Text or a Tex hit it. A scalar broadcasts to whatever each
            # descendant's rows are.
            replacement.set(opacity=0.0)
            return False

        for index, part in enumerate(replacement_parts):
            source_part = source_parts[
                (index * len(source_parts)) // len(replacement_parts)
            ]
            self._fit_bbox(part, source_part)
            part.set(opacity=0.0)
        replacement.set_non_recursive(opacity=torch.zeros_like(replacement.opacity))
        return True

    def _animate_source_dissolve(self, source, target, piecewise):
        source_parts = source.get_non_component_children()
        target_parts = target.get_non_component_children()
        if piecewise:
            for index, part in enumerate(source_parts):
                target_part = target_parts[
                    (index * len(target_parts)) // len(source_parts)
                ]
                self._fit_bbox(part, target_part)
                part.set(opacity=0.0)
            source.set_non_recursive(opacity=torch.zeros_like(source.opacity))
        else:
            self._fit_bbox(source, target)
            source.set(opacity=0.0)

    @staticmethod
    def _lone_object(node):
        """``(object, containers above it)`` for what ``node`` stands for.

        ``node`` itself if it is not a pure grouping node; otherwise the one
        object it holds, found by walking down through pure grouping nodes,
        with the grouping nodes strictly between, outermost first. ``(None,
        [])`` when a grouping node holds no object or several.
        """
        from algan.rendering.pass_identity import is_container

        if not is_container(node):
            return node, []
        found, pending, via = [], [node], {}
        while pending and len(found) < 2:
            parent = pending.pop()
            for child in parent.get_non_component_children():
                via[id(child)] = parent
                (pending if is_container(child) else found).append(child)
        if len(found) != 1:
            return None, []
        chain, step = [], via[id(found[0])]
        while step is not node:
            chain.append(step)
            step = via[id(step)]
        return found[0], chain[::-1]

    @staticmethod
    def _link_dissolve_identity(source, replacement, *, spliced):
        """Record, for the object-ID pass, who a dissolve's replacement is.

        Two pure grouping nodes are paired part by part, the way the
        dissolve animates them -- source part ``i`` of ``S`` fades toward
        replacement part ``i * R // S`` -- made one-to-one: the first source
        part to reach a replacement part is paired with it, and a surplus part
        on either side is another object, keeping its own ID and
        ``pass_index`` as a hierarchy morph's sinks and surrogates do. Any
        other pair is one object each side, looking through a grouping node
        that holds a single object (:meth:`_lone_object`): that object
        continues as the other side's, and a composite is never split into
        its parts. A side holding several objects hands on its tag only.

        Spliced (the replacement takes ``source``'s place), each source-side
        object is linked to its counterpart and hands it its ``pass_index``,
        the tags of the grouping nodes above it first so the nearest wins; the
        caller splices the root itself. Unspliced, the replacement stays
        beside ``source`` and resolves through its own hierarchy: a
        counterpart object is linked to its source object, and otherwise the
        replacement node reaches its source node's tags through
        ``_pass_tag_parents``, a tags-only route. See :func:`_link_stand_in`.
        """
        from algan.rendering.pass_identity import is_container

        if not spliced:
            replacement._pass_tag_parents = (source,)
        pending = [(source, replacement)]
        while pending:
            part, counterpart = pending.pop()
            is_root = part is source
            if is_container(part) and is_container(counterpart):
                if not is_root:
                    if spliced:
                        _link_stand_in(part, counterpart)
                    else:
                        counterpart._pass_tag_parents = (part,)
                parts = part.get_non_component_children()
                counterparts = counterpart.get_non_component_children()
                taken = set()
                for index, child in enumerate(parts if counterparts else ()):
                    other = counterparts[(index * len(counterparts)) // len(parts)]
                    if id(other) not in taken:
                        taken.add(id(other))
                        pending.append((child, other))
                continue
            obj, chain = MobMorphMixin._lone_object(part)
            other, _ = MobMorphMixin._lone_object(counterpart)
            if not spliced:
                if obj is not None and other is not None:
                    _link_stand_in(other, obj, carry_tag=False)
                elif not is_root:
                    counterpart._pass_tag_parents = (part,)
                continue
            if not is_root:
                if is_container(part):
                    # Its tag; linked only if the counterpart is a grouping
                    # node too, which here it is not.
                    _link_stand_in(part, counterpart)
                elif other is None and part._pass_index is not None:
                    counterpart._pass_index = part._pass_index
            if obj is not None and other is not None:
                for node in chain:
                    if node._pass_index is not None:
                        other._pass_index = node._pass_index
                _link_stand_in(obj, other)

    def _record_dissolve(
        self,
        source,
        target,
        *,
        minimize_movement,
        replacement_allowed,
    ):
        from algan.rendering.pass_identity import is_container

        am = source.animation_manager
        replacement = target.clone(add_to_scene=False, spawn=False)
        with Off(animation_manager=am):
            piecewise = self._prepare_piecewise_dissolve(source, replacement, target)
            self._register_hierarchy_for_render(replacement)
            if not replacement_allowed:
                self._link_dissolve_identity(source, replacement, spliced=False)
            elif is_container(source):
                # The root first, so a member's own tag, handed on after,
                # outranks the group's that the splice carries.
                self._splice_replacement(source, replacement)
                self._link_dissolve_identity(source, replacement, spliced=True)
            else:
                # The object first, so that the link to what it continues as
                # -- the replacement, or the one object a replacement Group
                # holds -- is the one the splice finds already made.
                self._link_dissolve_identity(source, replacement, spliced=True)
                self._splice_replacement(source, replacement)

        with Seq(animation_manager=am):
            with (
                Off(spawn_at_end=False, animation_manager=am),
                NoExtra(priority_level=1, animation_manager=am),
            ):
                replacement.spawn(animate=False)
            with Sync(animation_manager=am):
                self._animate_source_dissolve(source, target, piecewise)
                self._record_same_kind_morph(
                    replacement,
                    target,
                    minimize_movement=minimize_movement,
                    strategy="auto",
                    replacement_allowed=True,
                )
            with (
                Off(spawn_at_end=False, animation_manager=am),
                NoExtra(priority_level=1, animation_manager=am),
            ):
                source.despawn(animate=False)
        return replacement if replacement_allowed else source

    def _record_pn_morph(
        self,
        source,
        target,
        *,
        minimize_movement,
    ):
        from algan.animatable_base.morph_conversions import (
            convert_to_pn_soup,
            get_morph_conversion,
        )

        am = source.animation_manager
        source_conversion = get_morph_conversion(source._morph_family)
        target_conversion = get_morph_conversion(target._morph_family)
        source_soup = convert_to_pn_soup(source, add_to_scene=False)
        target_soup = convert_to_pn_soup(target, add_to_scene=False)
        # The two soups are diced as one primitive, so they have to agree on
        # the bound: the finer of the two, which cannot over-relax either.
        pixel_tolerance = min(
            source_soup.render_tolerance_pixels, target_soup.render_tolerance_pixels
        )
        source_soup.render_tolerance_pixels = pixel_tolerance
        target_soup.render_tolerance_pixels = pixel_tolerance

        difference = (
            target_soup.location.shape[-2] - source_soup.location.shape[-2]
        ) // 3
        if difference > 0:
            source_soup._expand_n_batch(difference)
        elif difference < 0:
            target_soup._expand_n_batch(-difference)

        # The soups are paired triangle by triangle, and by default that pairing
        # was the order they happened to be built in -- so a solid morphing into
        # a solid tore into visibly separated strips while independent triangles
        # crossed each other on their way to unrelated counterparts. Pairing
        # each with its nearest counterpart keeps the surface together.
        #
        # It is a Hungarian solve over an N x N distance matrix, so it is capped
        # rather than unconditional. Measured on this CPU: 0.07s at 1024
        # triangles, 0.66s at 2048, 0.97s at 2500, 2.2s at 3200, 3.3s at 4096 --
        # a 2x in size costs 5-6x in time. The cap sits where a morph still pays
        # about a second, and covers every solid measured (Sphere 462, Torus
        # 1716, a Square's triangulation 2178) while leaving out text-sized
        # soups (Text("hello") is 4379) where the solve runs away.
        triangles = source_soup.location.shape[-2] // 3
        if minimize_movement or triangles <= self._REORDER_TRIANGLE_CAP:
            target_soup._reorder_batch_to_minimize_movement(source_soup)
        else:
            get_logger().log(
                PERF,
                "become: %d triangles is over the %d cap for proximity pairing; "
                "the morph may show seams. Pass minimize_movement=True to pair "
                "anyway.",
                triangles,
                self._REORDER_TRIANGLE_CAP,
            )

        replacement = target.clone(add_to_scene=False, spawn=False)
        target_border = None
        if target_conversion.post_animate is not None and hasattr(
            replacement, "stroke_width"
        ):
            target_border = target.stroke_width.clone()
            replacement.set_non_recursive(
                stroke_width=torch.zeros_like(replacement.stroke_width)
            )

        self._register_hierarchy_for_render(source_soup)
        self._register_hierarchy_for_render(replacement)
        # The soup draws the morph frames of ``source``, which the splice
        # below hands on to ``replacement``: one object throughout.
        _link_stand_in(source_soup, source, carry_tag=False)
        self._splice_replacement(source, replacement)

        source_has_border = (
            source_conversion.pre_animate is not None
            and hasattr(source, "stroke_width")
            and bool((source.stroke_width.abs() > 1e-8).any())
        )
        target_has_border = target_border is not None and bool(
            (target_border.abs() > 1e-8).any()
        )
        border_fraction = 0.3
        border_phases = int(source_has_border) + int(target_has_border)
        morph_fraction = 1.0 - border_fraction * border_phases
        if morph_fraction <= 0:
            morph_fraction = 0.4
        unit = am.context.runtime_per_part

        with Seq(animation_manager=am):
            if source_has_border:
                with Sync(
                    runtime=border_fraction * unit,
                    easing=easings.identity,
                    animation_manager=am,
                ):
                    source_conversion.pre_animate(source, target)
            with (
                Off(spawn_at_end=False, animation_manager=am),
                NoExtra(priority_level=1, animation_manager=am),
            ):
                source.despawn(animate=False)
                source_soup.spawn(animate=False)
            with Sync(runtime=morph_fraction * unit, animation_manager=am):
                values = {
                    attr: getattr(target_soup, attr)
                    for attr in source_soup.animatable_attrs
                    if hasattr(source_soup, attr) and hasattr(target_soup, attr)
                }
                source_soup.set_non_recursive(**values)
            with (
                Off(spawn_at_end=False, animation_manager=am),
                NoExtra(priority_level=1, animation_manager=am),
            ):
                source_soup.despawn(animate=False)
                replacement.spawn(animate=False)
            if target_has_border:
                with Sync(
                    runtime=border_fraction * unit,
                    easing=easings.identity,
                    animation_manager=am,
                ):
                    target_conversion.post_animate(replacement, target)
        return replacement

    @staticmethod
    def _structural_members(source, source_primitives, source_hierarchy):
        """The structural nodes strictly between ``source`` and its primitives.

        ``[(node, {primitive indices below it}, depth)]``, deepest first, read
        before the morph rearranges anything; ``depth`` counts the steps up to
        ``source``. Each primitive is walked up through the parent that
        belongs to ``source``'s hierarchy, so a Mob also parented elsewhere is
        still counted where the author put it in ``source``.
        """
        hierarchy_ids = {id(mob) for mob in source_hierarchy}
        members = {}
        depths = {}
        for index, primitive in enumerate(source_primitives):
            chain, node = [], primitive
            while node is not None and node is not source:
                node = next(
                    (
                        parent
                        for parent in getattr(node, "parents", None) or ()
                        if id(parent) in hierarchy_ids
                    ),
                    None,
                )
                if node is None or _identity_contains(chain, node):
                    chain = None
                    break
                if node is not source:
                    chain.append(node)
            for steps, node in enumerate(reversed(chain or ()), start=1):
                members.setdefault(id(node), set()).add(index)
                depths.setdefault(id(node), steps)
        # A parent precedes its descendants in a traversal, so the reverse
        # lists every node after all of those below it.
        return [
            (node, members[id(node)], depths[id(node)])
            for node in reversed(source_hierarchy)
            if id(node) in members
        ]

    @staticmethod
    def _hand_on_structure_identity(
        source,
        final_root,
        final_ids,
        structure,
        source_primitives,
        paired,
        external_parents,
        grown_results=(),
        source_object=None,
    ):
        """Record, for the object-ID pass, what the source's structure became.

        The pairs are linked already: a replacement's splice carries its
        source primitive's ``pass_index``. What a hierarchy morph also retires
        is structure -- ``source`` and the nodes between it and its primitives
        (an inner Group, a Text inside a Group) -- in favour of the target's
        nesting. Deepest first, so the nearest tag wins as it did in the
        source, each such node goes to its counterpart: the final node at its
        own depth when all its results share that node and no other source's
        results lie below it, else the lowest final node below the root that
        holds only its results. Grouping nodes wrapped around a single object
        are looked through on both sides (:meth:`_lone_object`): a Group
        holding one Text that becomes a Text hands the Text on to the Text,
        and depths below are counted from those objects. A node's
        ``pass_index`` goes along unless a nearer source's tag (a primitive's
        own, a deeper node's) already holds the counterpart, in which case a
        composite is linked for ownership only (``_pass_owner_identity``: it
        is still the same object, but its other members must not take that
        tag). A node with no counterpart -- its results land among others' --
        links nowhere and puts its tag on the highest nodes below the root
        that hold only its results and nothing grown for a surplus target
        instead, and a node still parented outside ``source`` has not been
        retired, so only its tag moves.

        ``source`` itself goes to ``final_root``: its tag, unless a nearer
        source's already holds it, and a link unless ``source`` is still in
        the result (a leaf kept in the Group it became did not stop existing).
        :func:`_link_stand_in` declines the link for a Group that became a
        drawn Mob -- the members left shrinking away in it are other objects
        -- and then, like a root left unlinked because a nearer tag holds the
        result, it keeps resolving through the parents it was taken out of
        (:func:`_keep_pass_parents`).

        ``paired`` is ``[(source primitive index, its result)]``;
        ``external_parents`` are the parents ``source`` had outside itself;
        ``grown_results`` are the Mobs grown for surplus targets;
        ``source_object`` is ``_lone_object(source)`` read before the morph
        moved anything out of ``source``. Private attributes only: nothing
        renders differently.
        """
        from algan.rendering.pass_identity import is_container

        def final_parent(node):
            return next(
                (
                    parent
                    for parent in getattr(node, "parents", None) or ()
                    if id(parent) in final_ids
                ),
                None,
            )

        # Final nodes that hold a source-side tag, which nothing farther from
        # the geometry may overwrite. A target-side tag (one the target's
        # clone brought) is fair game: the source's tag wins.
        claimed = set()
        paths = {}
        below = {}
        for source_index, result in paired:
            if (
                getattr(source_primitives[source_index], "_pass_index", None)
                is not None
            ):
                claimed.add(id(result))
            path, node = [], result
            while node is not None and not _identity_contains(path, node):
                path.append(node)
                if node is final_root:
                    break
                node = final_parent(node)
            paths[source_index] = path
            for node in path:
                below.setdefault(id(node), set()).add(source_index)

        # Final nodes on the way up from a Mob the morph grew for a surplus
        # target: a tag handed down from a node with no counterpart must not
        # climb onto them, since that Mob was never below it.
        grown = set()
        for result in grown_results:
            node, seen = result, set()
            while node is not None and id(node) not in seen:
                seen.add(id(node))
                grown.add(id(node))
                node = None if node is final_root else final_parent(node)

        # Pure grouping wrappers around a single object on either side (a
        # Group holding one Text becoming a Text) are looked through: the
        # object inside the source becomes the object of the result, and
        # depths below it are counted from there.
        depth_of = {id(node): depth for node, _, depth in structure}
        result_object, _ = MobMorphMixin._lone_object(final_root)
        through = (
            source_object is not source
            and source_object is not None
            and id(source_object) in depth_of
            and result_object is not None
            and id(result_object) in final_ids
        )
        object_depth = depth_of[id(source_object)] if through else 0
        anchor = result_object if through else final_root

        structural_ids = {id(source)} | set(depth_of)
        for node, members, depth in structure:
            covered = [paths[index] for index in sorted(members) if index in paths]
            if not covered:
                # Every primitive below it shrinks away still hanging from it.
                continue
            counterpart = None
            if through and depth <= object_depth:
                counterpart = result_object
            else:
                steps = depth - object_depth
                level = {}
                for path in covered:
                    at = next(
                        (i for i, step in enumerate(path) if step is anchor), None
                    )
                    if at is None or at < steps:
                        level = None
                        break
                    level[id(path[at - steps])] = path[at - steps]
                if level and len(level) == 1:
                    (candidate,) = level.values()
                    if below[id(candidate)] <= members:
                        counterpart = candidate
            if counterpart is None:
                shared = set.intersection(
                    *({id(step) for step in path} for path in covered)
                )
                lowest = next((step for step in covered[0] if id(step) in shared), None)
                if (
                    lowest is not None
                    and lowest is not final_root
                    and below[id(lowest)] <= members
                ):
                    counterpart = lowest
            tag = getattr(node, "_pass_index", None)
            if counterpart is not None:
                shared_elsewhere = any(
                    id(parent) not in structural_ids
                    for parent in getattr(node, "parents", None) or ()
                )
                if id(counterpart) in claimed:
                    # A nearer tag holds it: a link would hand that tag to
                    # this node's other members. A composite is still the same
                    # object, so ownership alone follows.
                    if not shared_elsewhere and not is_container(node):
                        node._pass_owner_identity = counterpart
                elif shared_elsewhere:
                    if tag is not None:
                        counterpart._pass_index = tag
                else:
                    _link_stand_in(node, counterpart, within=final_root)
                if tag is not None:
                    claimed.add(id(counterpart))
                continue
            if tag is None:
                continue
            # The tag goes on the highest node below the root holding only
            # this node's results, not on each result: a deeper node linked to
            # its own counterpart hands its leftover members on through that
            # counterpart, so they must meet the tag on its way up. A
            # target-side tag stops the climb and is overridden.
            for path in covered:
                level = 0
                while (
                    level + 2 < len(path)
                    and not (
                        getattr(path[level], "_pass_index", None) is not None
                        and id(path[level]) not in claimed
                    )
                    and below[id(path[level + 1])] <= members
                    and id(path[level + 1]) not in grown
                ):
                    level += 1
                if id(path[level]) not in claimed:
                    path[level]._pass_index = tag
                    claimed.add(id(path[level]))

        kept = id(source) in final_ids
        if id(final_root) not in claimed:
            if not kept:
                _link_stand_in(source, final_root, within=final_root)
            elif (
                source is not final_root
                and getattr(source, "_pass_index", None) is not None
            ):
                final_root._pass_index = source._pass_index
        elif not kept and not is_container(source):
            source._pass_owner_identity = final_root
        if not kept:
            _keep_pass_parents(source, external_parents)

    def _record_primitive_hierarchy_morph(
        self,
        source,
        target,
        *,
        minimize_movement,
        strategy,
    ):
        """Morph renderer-facing units independently, then install target nesting."""
        source_primitives = self._collect_morph_primitives(source)
        target_primitives = self._collect_morph_primitives(target)
        if (
            not source_primitives
            and not target_primitives
            and source._morph_kind == target._morph_kind
        ):
            # Neither side draws anything -- two empty Groups, say. There is no
            # pair to record, and a context whose block records no event never
            # advances its cursor, so this morph alone used to take zero time
            # and pull everything after it in a Seq a second early. The roots
            # still have attributes of their own (location, opacity, color):
            # morphing those is both the right thing to animate and what makes
            # the morph occupy its runtime like every other route. Guarded on
            # the kinds matching because ``_record_same_kind_morph`` is only
            # defined for a matching pair; anything else falls through to the
            # ordinary path, which is what it did before.
            return self._record_same_kind_morph(
                source,
                target,
                minimize_movement=minimize_movement,
                strategy=strategy,
                replacement_allowed=True,
            )
        pairs, unmatched_sources, unmatched_targets = self._pair_primitive_indices(
            source_primitives, target_primitives, minimize_movement
        )

        planned_pairs = [
            (source_primitives[source_index], target_primitives[target_index])
            for source_index, target_index in pairs
        ]
        planned_pairs.extend(
            (
                target_primitives[target_index],
                target_primitives[target_index],
            )
            for target_index in unmatched_targets
        )
        planned_pairs.extend(
            (source_primitives[source_index], source_primitives[source_index])
            for source_index in unmatched_sources
        )
        for planned_source, planned_target in planned_pairs:
            if self._pair_supports_geometric_morph(planned_source, planned_target):
                continue
            if strategy == "auto" and "image" in {
                planned_source._morph_family,
                planned_target._morph_family,
            }:
                continue
            raise NotImplementedError(
                "A geometric hierarchy morph cannot pair "
                f"{type(planned_source).__name__} "
                f"({planned_source._morph_family!r}) with "
                f"{type(planned_target).__name__} "
                f"({planned_target._morph_family!r})"
            )

        am = source.animation_manager
        source_hierarchy = list(source.get_descendants())
        structure = self._structural_members(
            source, source_primitives, source_hierarchy
        )
        # Read now: the morph moves the members it keeps out of ``source``.
        source_object, _ = self._lone_object(source)
        parent_slots = self._capture_parent_slots(source)
        external_parents = [slot[0] for slot in parent_slots]
        pair_specs = []

        with Seq(animation_manager=am):
            with (
                Off(spawn_at_end=False, animation_manager=am),
                NoExtra(priority_level=1, animation_manager=am),
            ):
                pair_specs.extend(
                    (
                        source_primitives[source_index],
                        target_primitives[target_index],
                        target_index,
                    )
                    for source_index, target_index in pairs
                )

                for target_index in unmatched_targets:
                    target_primitive = target_primitives[target_index]
                    surrogate = target_primitive.clone(add_to_scene=False, spawn=False)
                    anchor = self._nearest_geometry_point(
                        source_primitives,
                        target_primitive,
                        source.get_center(),
                    )
                    self._collapse_hierarchy_at(surrogate, anchor)
                    # A COLLAPSED SEED IS A POINT, AND A POINT AT FULL OPACITY
                    # IS A BRIGHT SPECK THAT CAME FROM NOWHERE. The seed carries
                    # the target's color and material, so at zero size it
                    # rendered as a hard dot sitting at an unrelated source
                    # vertex for a third of the morph before inflating into a
                    # solid. Growth still says where the new geometry comes
                    # from; the fade is what makes it arrive rather than pop.
                    # Its counterpart is the sink below, which fades as it
                    # shrinks for the same reason.
                    self._zero_hierarchy_opacity(surrogate)
                    self._register_hierarchy_for_render(surrogate)
                    surrogate.spawn(animate=False)
                    pair_specs.append((surrogate, target_primitive, target_index))

                for source_index in unmatched_sources:
                    source_primitive = source_primitives[source_index]
                    sink = source_primitive.clone(add_to_scene=False, spawn=False)
                    if target_primitives:
                        anchor = min(
                            target_primitives,
                            key=lambda candidate: float(
                                (
                                    self._morph_center(source_primitive)
                                    - self._morph_center(candidate)
                                ).norm()
                            ),
                        ).get_center()
                    else:
                        anchor = target.get_center()
                    self._collapse_hierarchy_at(sink, anchor)
                    # Symmetric with the surrogate above: a surplus source that
                    # only shrinks ends as a full-brightness speck and then is
                    # despawned, so the last thing the viewer sees of it is a
                    # dot blinking out. Fading it as it shrinks is the same
                    # treatment images already got here.
                    self._zero_hierarchy_opacity(sink)
                    pair_specs.append((source_primitive, sink, None))

            results_by_target = {}
            cleanup_results = []
            with Sync(animation_manager=am):
                for pair_source, pair_target, target_index in pair_specs:
                    # "morph" here is this route forcing the geometric path on
                    # its own pairs, not the caller asking for it -- so a pair
                    # the caller left on "auto" still gets the cross-fade when
                    # one end is a stroke-only circuit whose PN soup is empty.
                    if (
                        "image"
                        in {
                            pair_source._morph_family,
                            pair_target._morph_family,
                        }
                        or (
                            strategy == "auto"
                            and pair_source._morph_kind != pair_target._morph_kind
                            and self._pair_wants_crossfade(pair_source, pair_target)
                        )
                        # A same-kind pair that crosses an untravellable
                        # attribute is the other case this route must not force
                        # onto the geometric path: forcing it plays the whole
                        # morph in the endpoint's version of the shape.
                        or (
                            strategy == "auto"
                            and pair_source._morph_structural_break(pair_target)
                        )
                    ):
                        pair_strategy = "dissolve"
                    else:
                        pair_strategy = "morph"
                    result = self._dispatch_become(
                        pair_source,
                        pair_target,
                        minimize_movement=minimize_movement,
                        strategy=pair_strategy,
                        replacement_allowed=True,
                    )
                    if target_index is None:
                        cleanup_results.append(result)
                    else:
                        results_by_target[target_index] = result

            with (
                Off(spawn_at_end=False, animation_manager=am),
                NoExtra(priority_level=1, animation_manager=am),
            ):
                final_root = target
                for target_index, target_primitive in enumerate(target_primitives):
                    result = results_by_target[target_index]
                    self._detach_from_parents(result, preserve=external_parents)
                    if target_primitive is target:
                        final_root = result
                    else:
                        self._splice_replacement(
                            target_primitive, result, carry_identity=False
                        )

                self._fill_captured_parent_slots(parent_slots, final_root)
                final_ids = {id(mob) for mob in final_root.get_descendants()}
                self._hand_on_structure_identity(
                    source,
                    final_root,
                    final_ids,
                    structure,
                    source_primitives,
                    [
                        (source_index, results_by_target[target_index])
                        for source_index, target_index in pairs
                    ],
                    external_parents,
                    [results_by_target[index] for index in unmatched_targets],
                    source_object,
                )
                obsolete_ids = {
                    id(mob) for mob in source_hierarchy if id(mob) not in final_ids
                }
                obsolete_roots = [
                    mob
                    for mob in source_hierarchy
                    if id(mob) in obsolete_ids
                    and not any(id(parent) in obsolete_ids for parent in mob.parents)
                ]
                for obsolete in obsolete_roots:
                    obsolete.despawn(animate=False)
                for cleanup in cleanup_results:
                    cleanup.despawn(animate=False)

                self._register_hierarchy_for_render(final_root)
                final_root.spawn(animate=False)
                bump_hierarchy_version()
        return final_root

    def _dispatch_become(
        self,
        source,
        target,
        *,
        minimize_movement,
        strategy,
        replacement_allowed,
    ):
        same_kind = source._morph_kind == target._morph_kind
        if strategy == "dissolve":
            return self._record_dissolve(
                source,
                target,
                minimize_movement=minimize_movement,
                replacement_allowed=replacement_allowed,
            )
        image_pair = "image" in {source._morph_family, target._morph_family}
        if image_pair:
            if strategy == "morph":
                raise NotImplementedError(
                    "ImageMob transitions do not have a geometric morph adapter"
                )
            return self._record_dissolve(
                source,
                target,
                minimize_movement=minimize_movement,
                replacement_allowed=replacement_allowed,
            )
        requires_grid_conversion = (
            same_kind
            and source._morph_family == "grid"
            and type(source) is not type(target)
        )
        if (
            strategy == "auto"
            and replacement_allowed
            and source._morph_structural_break(target)
        ):
            # An untravellable attribute cannot be shown changing (see
            # ``Mob._MORPH_UNTRAVELLABLE_ATTRS``), so a geometric morph across
            # one plays entirely in the endpoint's version of the shape. The
            # cross-fade is what the pair actually looks like: the source keeps
            # its fill while it fades, and the target arrives with its own.
            # Gated on ``replacement_allowed`` because the endpoint has to be a
            # target-class Mob; ``detach_history=False`` keeps identity and
            # therefore keeps the old behaviour.
            return self._record_dissolve(
                source,
                target,
                minimize_movement=minimize_movement,
                replacement_allowed=replacement_allowed,
            )
        if (
            strategy == "auto"
            and (not same_kind or requires_grid_conversion)
            and self._pair_wants_crossfade(source, target)
        ):
            return self._record_dissolve(
                source,
                target,
                minimize_movement=minimize_movement,
                replacement_allowed=replacement_allowed,
            )
        if same_kind and not requires_grid_conversion:
            return self._record_same_kind_morph(
                source,
                target,
                minimize_movement=minimize_movement,
                strategy=strategy,
                replacement_allowed=replacement_allowed,
            )

        from algan.animatable_base.morph_conversions import (
            MorphConversionError,
            get_morph_conversion,
        )

        source_conversion = get_morph_conversion(source._morph_family)
        target_conversion = get_morph_conversion(target._morph_family)
        can_convert = source_conversion is not None and target_conversion is not None
        if strategy == "morph" and not can_convert:
            raise NotImplementedError(
                "A forced geometric morph requires PN converters for both "
                f"{source._morph_family!r} and {target._morph_family!r}"
            )
        if replacement_allowed and can_convert:
            try:
                return self._record_pn_morph(
                    source,
                    target,
                    minimize_movement=minimize_movement,
                )
            except MorphConversionError:
                if replacement_allowed:
                    raise
        if not replacement_allowed:
            return self._record_dissolve(
                source,
                target,
                minimize_movement=minimize_movement,
                replacement_allowed=False,
            )
        raise NotImplementedError(
            "A geometric morph requires PN converters for both "
            f"{source._morph_family!r} and {target._morph_family!r}; "
            "only ImageMob falls back to a dissolve automatically"
        )

    @_rejecting_timing_kwargs
    def become(
        self,
        other_mob: Mob,
        *,
        detach_history: bool = True,
        minimize_movement: bool = False,
        strategy: str = "auto",
    ) -> Mob:
        """Transform any Mob hierarchy into any other Mob hierarchy.

        Structural containers are transparent to pairing: renderer-facing
        primitives are matched by concrete type, primitive family and either
        traversal order or spatial proximity. Same-kind pairs use structural point
        alignment and different primitive families convert through a cubic-PN
        triangle soup. ImageMob pairs cross-dissolve because image textures have no
        geometric adapter, and so does a pair that changes whether a circuit is
        filled: that flag is read once per render rather than per frame, so no
        morph can show it changing. The pairing prefers a counterpart that does
        not force the change. ``strategy`` may be ``"auto"``, ``"morph"`` (reject
        every non-geometric pair), or ``"dissolve"`` (dissolve the whole root).

        With the default ``detach_history=True``, the returned Mob has the target's
        hierarchy and is spliced into this Mob's parent slot; use it for later
        animation. Surplus target primitives fade in as they grow from collapsed,
        target-shaped geometry at nearby source points, while surplus sources fade
        out as they shrink to points.
        ``detach_history=False`` keeps identity for compatibility and therefore
        uses a dissolve where a target-class replacement would be required.
        Updaters remain attached to replaced sources rather than being migrated
        to replacements.

        Notes
        -----
        Cross-kind geometric morphs pair independent PN triangles, so a surface
        can show seams while triangles move to new counterparts. Bezier outlines
        are triangulated at the primitive-family swap, making the silhouette at
        that instant an approximation. A mesh-to-bezier morph likewise travels
        through the target's filled triangulation rather than growing cubic curves.
        Cached glyph views held before a cross-kind replacement remain views of the
        replaced source; reacquire them from the returned Mob when it is text.
        """
        if strategy not in {"auto", "morph", "dissolve"}:
            raise AlganConfigurationError(
                f"strategy must be 'auto', 'morph', or 'dissolve'; got {strategy!r}"
            )
        # Checked before anything is read off it: `become(None)` otherwise
        # surfaces as an AttributeError about `.scene`, which names neither
        # become nor the argument that was wrong.
        from algan.animatable_base.mob import Mob as _Mob

        if not isinstance(other_mob, _Mob):
            raise AlganConfigurationError(
                "become() takes the Mob to turn into; got "
                f"{other_mob!r}. To make a Mob vanish, use despawn()."
            )
        if other_mob.scene is not self.scene:
            raise AlganConfigurationError(
                "become requires source and target Mobs in the same Scene"
            )
        if (
            strategy == "morph"
            and self._morph_kind != other_mob._morph_kind
            and not detach_history
        ):
            raise NotImplementedError(
                "A cross-kind geometric morph requires detach_history=True "
                "so become can return and splice the target-class replacement"
            )

        am = self.animation_manager
        with Off(animation_manager=am):
            source = self
            if detach_history:
                source = self.detach_history()
            target = other_mob.clone(add_to_scene=False, spawn=False)
        if (
            detach_history
            and strategy != "dissolve"
            and (
                source._morph_family is None
                or target._morph_family is None
                or not source.is_primitive
                or not target.is_primitive
            )
        ):
            result = self._record_primitive_hierarchy_morph(
                source,
                target,
                minimize_movement=minimize_movement,
                strategy=strategy,
            )
        else:
            result = self._dispatch_become(
                source,
                target,
                minimize_movement=minimize_movement,
                strategy=strategy,
                replacement_allowed=detach_history,
            )
        # The object-ID pass needs no link here: every route that replaces
        # this Mob has already recorded who stands where (a replacement's
        # splice, a hierarchy morph's _hand_on_structure_identity), and the
        # others return this Mob itself.
        return result
