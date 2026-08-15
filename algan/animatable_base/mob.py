from __future__ import annotations

import difflib
from collections import defaultdict

import torch
import torch.nn.functional as F

from algan.animatable_base.animatable import (
    ANIMATABLE_PROPERTY_VERSION,
    Animatable,
    animated_function,
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

    def _adopt_structural_attrs(self, target):
        """Take target-side plain geometry metadata at a morph endpoint."""
        return self

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
    def _apply_change(self, attr, change, recursive=True, interpolation=1.0):
        change = change * interpolation
        current_value = self.get_animated_attribute(
            attr, include_descendants=recursive, copy=False
        )
        new_value = current_value + change
        return self._setattr_and_record_modification(
            attr, new_value, include_descendants=recursive
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
        :meth:`~.Mob.set`.

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

        current_value = self.get_animated_attribute(
            attr, include_descendants=recursive, default=value, copy=False
        )
        change = value - current_value
        self._apply_change(attr, change, recursive=recursive)
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
        local_coords = map_global_to_local_coords(my_loc, my_basis, child_loc)
        new_child_location = map_local_to_global_coords(my_loc, new_basis, local_coords)

        child_basis = self.get_animated_attribute(
            "basis", include_descendants=recursive, copy=False
        )
        new_child_basis = relation(child_basis, interpolated_change)

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
        return (
            self.parent_batch_sizes.shape[-1]
            if (
                hasattr(self, "parent_batch_sizes")
                and self.parent_batch_sizes is not None
            )
            else 0
        )

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
