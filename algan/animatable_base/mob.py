from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn.functional as F

from algan.animatable_base.animatable import (
    Animatable,
    animated_function,
)
from algan.animation_timeline.animation_contexts import AnimationContext, NoExtra, Off, Sync, Seq
from algan.animatable_base.mob_hierarchy import MobHierarchyMixin
from algan.animatable_base.mob_orientation import MobOrientationMixin
from algan.animatable_base.mob_movement import MobMovementMixin
from algan.animatable_base.mob_layout import MobLayoutMixin, DEFAULT_BUFFER  # noqa: F401 -- DEFAULT_BUFFER re-exported
from algan.animatable_base.mob_morph import MobMorphMixin
from algan.animatable_base.mob_materials import (  # noqa: F401 -- exception re-exported
    MobMaterialsMixin,
    ModifiedProtectedAttributeError,
)
from algan.constants.spatial import *
from algan.geometry.geometry import (
    get_rotation_between_bases,
    map_global_to_local_coords,
    map_local_to_global_coords,
)
from algan.utils.animation_utils import animate_lagged_by_location
from algan.utils.tensor_utils import (
    cast_to_tensor,
    squish,
    unsquish,
)



class Mob(MobHierarchyMixin, MobOrientationMixin, MobMovementMixin,
        MobLayoutMixin, MobMorphMixin, MobMaterialsMixin, Animatable):
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
        by :meth:`~.Mob.get_default_color()` .
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

        render_to_file()

    Create a mob with a specific color and scale:

    .. algan:: Example2Mob

        from algan import *

        circle = Circle(color=BLUE).scale(2).spawn()

        render_to_file()
    """

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
        additive_relation = (lambda x, y: x + y,
                             lambda x, y: y - x
                             )
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
                    lambda x, y: (x * y),
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

    def _init_default_attr(self, attr, value):
        """Allocate ``attr``'s attribute-timeline buffer directly to ``value``
        during construction, bypassing the get/change/apply machinery of the
        normal property setter. Valid for a fresh mob (no children yet, not
        spawned, buffer not yet allocated) whose setter would only establish
        the initial value -- the state inside :meth:`__init__`. Falls back to
        the full setter if any precondition does not hold."""
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
        animated_args={"interpolation": 0.0}, unique_args=["key", "recursive", "relative"]
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
        """Applies an animated change to an attribute, interpolating between two target values.

        The interpolation first moves from the current value towards `change1` from
        t=0 to 0.5, then moves from `change1` to `change2` from t=0.5 to 1.

        Parameters
        ----------
        key : str
            The name of the attribute to change (e.g., 'location', 'color').
        change1 : Any
            The first target value for the attribute.
        change2 : Any, optional
            The second target value for the attribute. If None, each affected
            part returns to its own current (pre-animation) value — the right
            choice for pulses on composite mobs, where a single target value
            would overwrite per-descendant attributes.
        interpolation : float, optional
            The interpolation factor used for animation.
        recursive : bool, optional
            If True, applies the change recursively to all child Mobs.
            Defaults to True.
        relative : bool, optional
            If True, `change1`/`change2` are multipliers of each part's current
            value instead of absolute targets (e.g. a scale pulse to 1.2x).

        Returns
        -------
        Mob
            The Mob instance itself, allowing for method chaining.

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
            change1 if (not relative and change1 is not None)
            else cast_to_tensor(getattr(self, key))
        )
        default = cast_to_tensor(default)
        if not recursive and default.shape[-2] == 1:
            default = default.expand(
                *([-1] * (default.dim() - 2)), self.location.shape[-2], -1
            )
        current_value = self.get_animated_attribute(key, include_descendants=recursive, default=default)
        if relative:
            change1 = current_value * cast_to_tensor(change1)
            change2 = current_value if change2 is None else current_value * cast_to_tensor(change2)
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

        self.setattr_and_record_modification(key, interpolated_value, include_descendants=recursive)
        return self

    def set_opacity_via_color(self, opacity):
        with Sync(animation_manager=self.animation_manager):
            for d in self.get_descendants():
                d._original_color_set_opacity_via_color = d.color
                d.set_non_recursive(color=d.color.set_opacity(opacity))
        return self

    def pulse_color(self, color: torch.Tensor = None, opacity: bool = None, recursive=True, new_color=None) -> Mob:
        """Animates a color pulse effect.

        The Mob's color changes to the target ``color`` and then animates to
        ``new_color`` (its current color by default), as a two-stage keyframe
        animation.

        Parameters
        ----------
        color
            The color to pulse to. If None, only opacity is pulsed.
        opacity
            If given, the opacity to pulse to (held for both stages).
        recursive
            Whether to apply the pulse to all descendants as well.
        new_color
            The color to end on after the pulse. Defaults to the current color.

        Returns
        -------
        Mob
            The Mob instance itself, allowing for method chaining.

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
                self.apply_absolute_change_two(
                    "color", cast_to_tensor(color), new_color, recursive=recursive)
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
        **kwargs,
    ) -> Mob:
        """Applies a color wave effect across the Mob and its descendants.

        The color change propagates spatially across the mob's constituent parts.

        Parameters
        ----------
        color
            The target color for the wave.
        wave_length
            Controls the spatial extent (length) of the wave. A smaller value
            means a more compressed wave. Defaults to 2.
        reverse
            If True, the wave propagates in the opposite direction.
        direction
            The 3-D vector defining the direction of wave propagation.
            If None, uses the Mob's upwards direction.
        lag_duration
            Time offset (seconds) between the first and last part starting
            their pulse.
        **kwargs
            Additional keyword arguments passed to :meth:`pulse_color` for
            each individual part of the wave animation.

        Returns
        -------
        Mob
            The Mob instance itself, allowing for method chaining.

        """
        if direction is None:
            direction = self.get_upwards_direction()
        with AnimationContext(run_time_unit=wave_length / lag_duration, animation_manager=self.animation_manager):
            # Filters for primitive parts to ensure the wave animates on individual rendering elements
            # TODO change this to use non_recursive set
            primitive_mobs = [
                _
                for _ in self.get_descendants()
                if (_.is_primitive and not _.ignore_wave_animations)
            ]
            kwargs['recursive'] = False
            animate_lagged_by_location(
                primitive_mobs,
                lambda x: x.pulse_color(color, **kwargs),
                direction * (-1 if reverse else 1),
                lag_duration=lag_duration,
            )
        return self

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
            self.data_sub_inds is not None
            and current_inds.shape[0] == self.batch_size
        )
        if (
            shared_view_has_full_buffer
            or current_inds.shape[0] == value.shape[-2]
            or value.shape[-2] == 1
        ):
            return self
        current_value = self.get_animated_attribute(key, default=None, include_descendants=False)
        if current_value.shape[-2] != 1:
            raise ValueError(f"Attempting to set {key} which currently has value of shape {current_value.shape}"
                             f"to new value with shape {value.shape}, which is not broadcastable.")
        # Indexed mobs share their source's timeline rows.  ``data_sub_inds``
        # is expressed in that full source index space, so expanding only to
        # the selected view's local row count would leave later indexing out
        # of bounds (for example a packed text glyph's control-point color).
        target_size = (
            self.batch_size
            if self.data_sub_inds is not None
            else value.shape[-2]
        )
        expanded = current_value.expand(
            *([-1] * (current_value.dim() - 2)), target_size, -1
        )
        tl.add(self, expanded, overwrite=True)
        return self

    @animated_function(animated_args={"interpolation": 0.0})
    def _apply_change(self, attr, change, recursive=True, interpolation=1.0):
        change = change * interpolation
        current_value = self.get_animated_attribute(attr, include_descendants=recursive, copy=False)
        new_value = current_value + change
        return self.setattr_and_record_modification(attr, new_value, include_descendants=recursive)

    @animated_function(animated_args={"interpolation": 0.0})
    def _apply_set(self, attr, value, recursive=True, interpolation=1.0):
        new_value = value * interpolation
        return self.setattr_and_record_modification(attr, new_value, include_descendants=recursive)

    def set_animated_attribute(self, attr, value, recursive=True):
        if self._prevent_recursive_sets:
            recursive = False
        value = cast_to_tensor(value)

        current_value = self.get_animated_attribute(attr, include_descendants=recursive, default=value, copy=False)
        change = value - current_value
        self._apply_change(attr, change, recursive=recursive)
        return self

    @property
    def location(self) -> torch.Tensor:
        """The 3-D location of the Mob in world space.

        When set, it triggers an animated change to the new location,
        maintaining child Mob positions relative to the parent..

        """
        return self.get_animated_attribute("location")

    @location.setter
    def location(self, location: torch.Tensor):
        recursive = not self._prevent_recursive_sets
        value = cast_to_tensor(location)
        attr = "location"

        current_value = self.get_animated_attribute(
            attr, include_descendants=False, default=value, copy=False
        )
        change = value - current_value
        self._apply_change(attr, change, recursive=recursive)
        return self

    @property
    def basis(self) -> torch.Tensor:
        """The flattened 3x3 matrix representing the Mob's orientation and scale.

        The rows of the unflattened matrix correspond to the right, upwards,
        and forwards directions of the Mob's local coordinate system.
        Their norms indicate the scaling along those axes.
        When accessed,
        When set, it triggers an animated interpolation to the new basis,
        maintaining child Mob positions relative to the parent.

        """
        return self.get_animated_attribute("basis")

    @property
    def normalized_basis(self) -> torch.Tensor:
        """The Mob's basis matrix with all its row vectors normalized to unit length.
        This represents only the orientation (rotation) without any scaling.

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
        self._apply_basis_change(
            change, default_basis=value, recursive=recursive
        )

    @animated_function(animated_args={'interpolation': 0.0})
    def _apply_basis_change(
        self, change, default_basis=None, recursive=True, interpolation=1.0
    ):
        attr = "basis"
        relation, inverse_relation = self.attr_to_relations[attr]

        my_basis = self.get_animated_attribute('basis', include_descendants=False, default=default_basis, copy=False)
        my_loc = self.get_animated_attribute('location', include_descendants=False, copy=False)

        identity = inverse_relation(my_basis, my_basis)
        interpolated_change = torch.lerp(identity, change, interpolation)
        new_basis = relation(my_basis, interpolated_change)

        child_loc = self.get_animated_attribute('location', include_descendants=recursive, copy=False)
        local_coords = map_global_to_local_coords(my_loc, my_basis, child_loc)
        new_child_location = map_local_to_global_coords(my_loc, new_basis, local_coords)

        child_basis = self.get_animated_attribute('basis', include_descendants=recursive, copy=False)
        new_child_basis = relation(child_basis, interpolated_change)

        self._apply_set("location", new_child_location, recursive=recursive)
        self._apply_set("basis", new_child_basis, recursive=recursive)

    @property
    def scale_coefficient(self) -> torch.Tensor:
        """The scaling factor of the Mob along its local axes, derived from the basis.
        It is the norm of the basis vectors.

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
        """Alias for :meth:`~.Mob.get_forward_direction()` .

        Returns
        -------
        torch.Tensor
            The normalized forward direction vector of the Mob.

        """
        return self.get_forward_direction()

    def set_location(self, location: torch.Tensor, recursive: bool = True) -> Mob:
        """Sets the location of the Mob.

        Parameters
        ----------
        location : torch.Tensor
            The target 3-D location.
        recursive : bool, optional
            If True, also affects the locations of child Mobs to maintain
            their relative positions. Defaults to True.

        Returns
        -------
        Mob
            The Mob instance itself, allowing for method chaining.

        """
        if recursive:
            self.location = location
        else:
            self.set_non_recursive(location=location)
        return self

    def get_parts_as_mobs(self) -> list[Mob]:
        """
        Recursively flattens the Mob and its children into a list of individual Mobs.

        Returns
        -------
        list[:class:`~.Mob`]
            A list containing this Mob and all its descendant Mobs.

        """
        parts = [self]
        for child in self.children:
            parts.extend(child.get_parts_as_mobs())
        return parts

    def scale(
        self, scale_factor: float | torch.Tensor, recursive: bool = True
    ) -> Mob:
        """Scales the Mob by a factor `scale_factor` relative to its current scale.

        Parameters
        ----------
        scale_factor
            The scaling factor. For example, `2` for double size,
            `0.5` for half size.
        recursive
            If True, applies scaling recursively to all descendant Mobs.

        Returns
        -------
        :class:`~.Mob`
            The Mob instance itself, allowing for method chaining.

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
        """Sets the absolute scale of the Mob to a specific value.

        Parameters
        ----------
        scale
            The target absolute scaling factor.
        recursive
            If True, applies scaling recursively to all descendant Mobs.

        Returns
        -------
        :class:`~.Mob`
            The Mob instance itself, allowing for method chaining.

        """
        return (
            self.set(scale_coefficient=scale)
            if recursive
            else self.set_non_recursive(scale_coefficient=scale)
        )

    def refresh_history(self):
        """Resets the modification history and spawn time for this Mob and all its descendants.
        This effectively clears all animation data and makes them behave as if newly created.
        """
        for mob in self.get_descendants():
            mob.lifespan.start = lambda: -1

    def detach_history(self):
        """Detaches the Mob's current animation history into a new, independent clone of this Mob.

        This is useful when you want to make a change to an animatable attribute that would not be
        animatable (interpolable), for example changing the resolution of a Surface Mob with a simple
        assignment would result in an error when the old resolution is attempted to be interpolated
        with the new resolution (shapes mis-match), so you must detatch the history before changing
        resolution.
        """
        detach_time = self.animation_manager.context.timespan.current_time
        with Off(animation_manager=self.animation_manager), NoExtra(priority_level=1, animation_manager=self.animation_manager):
            clone_mob = self.clone(reset_history=False, spawn=False)
            descendant_map = dict(zip(self.get_descendants(), clone_mob.get_descendants()))

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
            for f in timeline.function_timeline.function_applications:
                if (
                    f.caller in descendant_map
                    # Functions without captured row edits replay against the
                    # caller's current topology.  If such a function begins at
                    # the detach boundary, it belongs to the replacement mob,
                    # not the historical clone whose lifespan ends there.
                    and not (
                        not f.recorded_edits
                        and f.time.start >= detach_time
                    )
                ):
                    f.caller = descendant_map[f.caller]

            for orig, clone in descendant_map.items():
                # The clone inherits the original's spawn time (this mob is
                # re-spawned at the current time below).
                clone.lifespan.start = orig.lifespan.start
                timeline.register_spawn(clone, clone.lifespan)
            clone_mob.despawn(animate=False)
            self.refresh_history()
            self.spawn(animate=False)
            return self

    def check_properties_are_valid(self, property_names):
        # TODO: consider caching this union on the owning timeline.
        available_attrs = set([*self.animatable_attrs, *self.scene.timeline_manager.attr_to_timeline.keys()])
        for p in property_names:
            if not hasattr(self, p) and (p not in available_attrs):
                raise AttributeError(f'"{p}" is not recognized as an animatable Mob property. '
                                     f'Available properties are: {self.animatable_attrs}.')

    def set_non_recursive(self, **kwargs) -> Mob:
        """Sets multiple attributes non-recursively (i.e., only for this Mob, not its children).
        This is useful for applying changes that should not propagate down the hierarchy.

        Parameters
        ----------
        **kwargs
            Keyword arguments where keys are attribute names (e.g., 'color', 'opacity')
            and values are the new values for those attributes.

        Returns
        -------
        :class:`~.Mob`
            The Mob instance itself, allowing for method chaining.

        """
        prs = self._prevent_recursive_sets
        self._prevent_recursive_sets = True
        self.set(**kwargs)
        self._prevent_recursive_sets = prs
        return self

    def set(self, **kwargs) -> Mob:
        """Sets multiple attributes, applying changes recursively to descendants.

        Parameters
        ----------
        **kwargs
            Keyword arguments where keys are attribute names (e.g., 'location', 'color')
            and values are the new values for those attributes. These changes will
            be animated and propagated to children.

        Returns
        -------
        :class:`~.Mob`
            The Mob instance itself, allowing for method chaining.

        Examples
        ---------
        Move a square to the right and change its color to blue:

        .. algan:: Example1MobSet

            from algan import *

            mob = Square().spawn()
            mob.set(location=ORIGIN+RIGHT, color=BLUE)

            render_to_file()

        """
        self.check_properties_are_valid(kwargs.keys())
        with Sync(animation_manager=self.animation_manager):
            for key, value in kwargs.items():
                self.__setattr__(
                    key, value
                )  # Calls the property setters, which handle animation and recursion
        return self

    def on_create(self):
        opacity = self.opacity
        with Seq(animation_manager=self.animation_manager):
            prs = self._prevent_recursive_sets
            self._prevent_recursive_sets = True
            with Off(animation_manager=self.animation_manager):
                self.opacity = 0
            self.opacity = opacity
            self._prevent_recursive_sets = prs

    def on_destroy(self):
        self.opacity = torch.tensor((0.0,)).view(1)

    def set_data_sub_inds(self, data_sub_inds: list[int] | slice):
        """Sets the sub-indices that this Mob will use when reading and writing
        its rows of the shared attribute timelines. This is used for implementing
        indexing of batched mobs to retrieve sub-mobs that share the same
        underlying data.

        Parameters
        ----------
        data_sub_inds : list[int] or slice
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
            c.set_data_sub_inds(data_sub_inds)

    def __getitem__(self, item: int | slice) -> Mob:
        """Allows accessing a part of a batched Mob using slice notation (e.g., `my_mob[0]`, `my_mob[1:3]`).

        Returns a new Mob instance that represents the specified sub-part(s).
        This new Mob shares the underlying animation data with the original,
        but its `data_sub_inds` are set appropriately to only operate on the
        selected batch elements. This is efficient as it avoids data duplication.

        Parameters
        ----------
        item : int or slice
            The index or slice for selecting elements from the batch
            dimension.

        Returns
        -------
        Mob
            A new Mob instance representing the selected sub-part(s) of the
            original Mob.
        """
        # Clone the mob without cloning its data, but recursively for children structure
        cloned_mob = self.clone(
            add_to_scene=False, clone_data=False, recursive=True, animate_creation=False
        )
        # Set the data sub-indices for the cloned mob to point to the desired batch elements
        cloned_mob.set_data_sub_inds([item] if isinstance(item, int) else item)
        return cloned_mob
