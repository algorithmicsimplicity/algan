from __future__ import annotations

import math
import warnings
from collections import defaultdict

import torch
import torch.nn.functional as F

from algan.animation.animatable import (
    Animatable,
    animated_function,
)
from algan.animation.animation_contexts import AnimationContext, NoExtra, Off, Sync
from algan.animation.timeline import STRUCTURE_VERSION, TimelineManager, bump_structure_version
from algan.mobs.mob_layout import MobLayoutMixin, DEFAULT_BUFFER  # noqa: F401 -- DEFAULT_BUFFER re-exported
from algan.mobs.mob_morph import MobMorphMixin
from algan.mobs.mob_materials import (  # noqa: F401 -- exception re-exported
    MobMaterialsMixin,
    ModifiedProtectedAttributeError,
)
from algan.constants.math import RADIANS_TO_DEGREES
from algan.constants.rate_funcs import ease_out_exp, inversed
from algan.constants.spatial import *
from algan.geometry.geometry import (
    get_rotation_around_axis,
    get_rotation_between_3d_vectors,
    get_rotation_between_bases,
    map_global_to_local_coords,
    map_local_to_global_coords,
    project_point_onto_line,
    rotate_vector_around_axis,
)
from algan.utils.animation_utils import animate_lagged_by_location
from algan.utils.python_utils import traverse
from algan.utils.tensor_utils import (
    broadcast_cross_product,
    cast_to_tensor,
    dot_product,
    squish,
    unsquish,
)



class Mob(MobLayoutMixin, MobMorphMixin, MobMaterialsMixin, Animatable):
    """Base class for all objects that have a location and orientation in 3-D space.

    A Mob is an Animatable that exists in a 3-D scene,
    possessing properties like location, orientation (basis), and color.
    It can have child Mobs, forming a hierarchy, and supports various
    transformations and animations.

    Parameters
    ----------
    location
        Initial location in 3-D world space.
        Shape: `(*, 3)` where `*` denotes any number of batch dimensions.
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
        The maximum opacity of the Mob (0.0 for fully transparent to 1.0 for fully opaque).
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
        glow_radius: float = 0.2,
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
                "max_opacity",
                "glow",
                "glow_radius",
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
                # 'glow': additive_relation, # Currently commented out, but could be additive.
                "basis": (
                    lambda x, y: squish(
                        unsquish(y, -1, 3) @ unsquish(x, -1, 3), -2, -1
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
        self.location = cast_to_tensor(location)
        self.basis = cast_to_tensor(basis)

        if color is None:
            color = self.get_default_color()
        self.color = color
        self.max_opacity = cast_to_tensor(opacity)
        self.opacity = cast_to_tensor(opacity)#cast_to_tensor(1)  # Current opacity, can be animated
        self.glow = cast_to_tensor(glow)
        self.glow_radius = cast_to_tensor(glow_radius)
        self.num_points_per_object = 1
        self.shader = None

    def get_points_evenly_along_direction(self, direction, num_points=3):
        e, s = (
            self.get_boundary_edge_point(direction),
            self.get_boundary_edge_point(-direction),
        )
        return [s * t + (1 - t) * e for t in torch.linspace(0, 1, num_points + 2)[1:-1]]

    def reset_basis(self):
        """Resets the Mob's basis to the identity matrix (no rotation, unit scale)."""
        self.basis = cast_to_tensor(cast_to_tensor(squish(torch.eye(3))))

    def register_attrs_as_animatable(self, attrs: list[str], my_class=None):
        """
        Registers attributes as animatable, meaning their changes can be tracked
        and interpolated over time for animation.

        This method dynamically creates property getters and setters for the
        specified attributes if they don't already exist, allowing them to be
        controlled by the animation system. When an animatable attribute is
        modified, the change is recorded on the global timeline
        (:class:`~algan.animation.timeline.AnimationTimeline`).

        Parameters
        ----------
        attrs : set[str] or str
            A collection of attribute names (or a single attribute name) to
            register as animatable.
        my_class : type, optional
            The class to which the property getters and setters should be
            attached. Defaults to the current Mob's class.
        """
        if isinstance(attrs, str):
            attrs = {
                attrs,
            }
        # if not isinstance(attrs, set):
        #    attrs = set(attrs)
        if not hasattr(self, "animatable_attrs"):
            self.animatable_attrs = []
        if my_class is None:
            my_class = self.__class__
        for attr in attrs:
            self.add_property_getter_and_setter(attr, my_class)
        self.animatable_attrs.extend(
            [_ for _ in attrs if _ not in self.animatable_attrs]
        )  # update(attrs)

    def add_property_getter_and_setter(
        self, property_name: str, class_to_attach_to=None
    ):
        """Dynamically adds a property with a getter and setter for a given attribute name.

        The getter retrieves the current (potentially animated) value of the
        attribute from the global attribute timeline; the setter writes the
        value to the timeline, recording the modification so it can be
        replayed at render time.

        Parameters
        ----------
        property_name
            The name of the property to create (e.g., 'location', 'color').
        class_to_attach_to : (type, optional)
            The class to which this property
            will be added. Defaults to the instance's own class.

        """
        if class_to_attach_to is None:
            class_to_attach_to = self.__class__
        if hasattr(class_to_attach_to, property_name):
            return

        tensor_subclass = Color if property_name == 'color' else torch.Tensor

        @property
        def prop(self):
            return self.get_animated_attribute(property_name).as_subclass(tensor_subclass)

        @prop.setter
        def prop(self, value):
            return self.set_animated_attribute(property_name, value)

        setattr(class_to_attach_to, property_name, prop)

    def get_children(self, generation=0, include_components=True):
        children = self.children
        if not include_components:
            children = [_ for _ in children if _ not in self.components]
        if generation <= 0:
            return children
        children = [_.get_children(generation - 1) for _ in children]
        return [x for l in children for x in l]

    def get_descendants(self, include_self: bool = True) -> list[Mob]:
        """Retrieves a list all descendant Mobs in the hierarchy, optionally including itself.

        The traversal is cached against the global structure version (bumped
        by any hierarchy change), because recorded-function replay re-reads it
        for every event of every frame batch.

        Parameters
        ----------
        include_self
            If True, the current Mob instance
            is included in the returned list.

        Returns
        -------
        list[Mob]
            A flat list containing the Mob and all its children,
            grandchildren, and so on.

        """
        cache = getattr(self, "_descendants_cache", None)
        if cache is not None and cache[0] == STRUCTURE_VERSION[0]:
            descendants = cache[1]
        else:
            descendants = list(
                traverse(
                    [
                        self,
                        [c.get_descendants() for c in self.children]
                        if hasattr(self, "children")
                        else [],
                    ]
                )
            )
            object.__setattr__(
                self, "_descendants_cache", (STRUCTURE_VERSION[0], descendants)
            )
        return list(descendants) if include_self else descendants[1:]

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
        animated_args={"interpolation": 0.0}, unique_args=["key", "recursive"]
    )
    def apply_absolute_change_two(
        self,
        key: str,
        change1: any,
        change2: any,
        interpolation: float = 1.0,
        recursive: bool = True,
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
        change2 : Any
            The second target value for the attribute.
        interpolation : float, optional
            The interpolation factor used for animation.
        recursive : bool, optional
            If True, applies the change recursively to all child Mobs.
            Defaults to True.

        Returns
        -------
        Mob
            The Mob instance itself, allowing for method chaining.

        """
        current_value = self.get_animated_attribute(key, include_descendants=recursive, default=change1)
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
        with Sync():
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
        if new_color is None:
            new_color = self.color
        with Sync():
            for attr, v1, v2 in [("color", color, new_color), ("opacity", opacity, opacity)]:
                if v1 is None:
                    continue
                n = self.location.shape[-2]
                self.apply_absolute_change_two(attr, *[cast_to_tensor(_).expand(-1,n,-1)
                    for _ in [v1, v2]], recursive=recursive)
            #if color is not None:
            #    self.apply_absolute_change_two("color", color, new_color, recursive=recursive)
            #if opacity is not None:
            #    self.apply_absolute_change_two("opacity", opacity, opacity, recursive=recursive)
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
        with AnimationContext(run_time_unit=wave_length / lag_duration):
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
        tm = TimelineManager.instance()
        tm.add_mob_attr(self, key, value, add_mob=False)
        tl = tm.attr_to_timeline[key]
        if self.id not in tl.mob_id_to_inds:
            self._try_add_to_timeline(key, value)
            return self
        current_inds = tl.mob_id_to_inds[self.id]
        value = cast_to_tensor(value)
        if (current_inds.shape[0] == value.shape[-2]) or (value.shape[-2] == 1):
            return self
        current_value = self.get_animated_attribute(key, default=None, include_descendants=False)
        if current_value.shape[-2] != 1:
            raise ValueError(f"Attempting to set {key} which currently has value of shape {current_value.shape}"
                             f"to new value with shape {value.shape}, which is not broadcastable.")
        tl.add(self, current_value.expand(-1, value.shape[-2], -1), overwrite=True)
        return self

    @animated_function(animated_args={"interpolation": 0.0})
    def _apply_change(self, attr, change, recursive=True, interpolation=1.0):
        change = change * interpolation
        current_value = self.get_animated_attribute(attr, include_descendants=recursive)
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

        current_value = self.get_animated_attribute(attr, include_descendants=recursive, default=value)
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

        current_value = self.get_animated_attribute(attr, include_descendants=False, default=value)
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
        my_basis = self.get_animated_attribute('basis', include_descendants=False, default=value)
        change = inverse_relation(my_basis, value)
        # recursive must be passed as an explicit kwarg (not read from
        # self._prevent_recursive_sets inside _apply_basis_change) so that it
        # is recorded with the function application and replays correctly at
        # render time, when _prevent_recursive_sets has been restored.
        self._apply_basis_change(
            change, default_basis=value, recursive=not self._prevent_recursive_sets
        )

    @animated_function(animated_args={'interpolation': 0.0})
    def _apply_basis_change(
        self, change, default_basis=None, recursive=True, interpolation=1.0
    ):
        attr = "basis"
        relation, inverse_relation = self.attr_to_relations[attr]

        my_basis = self.get_animated_attribute('basis', include_descendants=False, default=default_basis)
        my_loc = self.get_animated_attribute('location', include_descendants=False)

        identity = inverse_relation(my_basis, my_basis)
        interpolated_change = torch.lerp(identity, change, interpolation)
        new_basis = relation(my_basis, interpolated_change)

        child_loc = self.get_animated_attribute('location', include_descendants=recursive)
        local_coords = map_global_to_local_coords(my_loc, my_basis, child_loc)
        new_child_location = map_local_to_global_coords(my_loc, new_basis, local_coords)

        child_basis = self.get_animated_attribute('basis', include_descendants=recursive)
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

    def move_between(self, loc1, loc2):
        loc1, loc2 = [_.get_center() if hasattr(_, 'get_center') else _ for _ in [loc1, loc2]]
        return self.move_to((loc1 + loc2) / 2)

    def set_center(self, location):
        return self.move_to(self.location - self.get_center() + location)

    def move_to(
        self, location: torch.Tensor, path_arc_angle: float | None = None, **kwargs
    ) -> Mob:
        """Moves the Mob to a specified location.

        If `path_arc_angle` is provided, the Mob moves along a circular arc.
        Otherwise, it moves in a straight line.

        Parameters
        ----------
        location : torch.Tensor
            The target 3-D location.
        path_arc_angle : float, optional
            The angle of the arc in degrees for curved movement. If None,
            movement is linear. Defaults to None.
        **kwargs
            Additional arguments passed to `set_location` or
            `move_to_point_along_arc`.

        Returns
        -------
        Mob
            The Mob instance itself.
        """
        if path_arc_angle is None:
            return self.set_location(location, **kwargs)
        return self.move_to_point_along_arc(location, path_arc_angle, **kwargs)

    def move(self, displacement: torch.Tensor, **kwargs) -> Mob:
        """Moves the Mob by a given displacement vector from its current location.

        Parameters
        ----------
        displacement : torch.Tensor
            The 3-D vector by which to move the Mob.
        **kwargs
            Additional arguments passed to `move_to` (e.g., `path_arc_angle`).

        Returns
        -------
        Mob
            The Mob instance itself, allowing for method chaining.
        """
        self.move_to(self.location + cast_to_tensor(displacement), **kwargs)
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

    @animated_function(animated_args={"num_degrees": 0}, unique_args=["axis"])
    def rotate(
        self, num_degrees: float | torch.Tensor, axis: torch.Tensor = OUT
    ) -> Mob:
        """Rotates the Mob by a number of degrees around a given axis passing through the mob's center.

        Parameters
        ----------
        num_degrees
            The angle of rotation in degrees.
        axis
            3-D axis of rotation (e.g., `OUT` for Z-axis, `UP` for Y-axis).
            This vector does not need to be normalized. Defaults to `OUT`.

        Returns
        -------
        :class:`~.Mob`
            The Mob instance itself, allowing for method chaining.

        """
        normalized_axis = F.normalize(cast_to_tensor(axis), p=2, dim=-1)
        # Get the rotation matrix for the specified degrees and axis
        rotation_matrix = get_rotation_around_axis(num_degrees, normalized_axis, dim=-1)
        # Apply the rotation to the Mob's basis matrix
        self.basis = squish(unsquish(self.basis, -1, 3) @ rotation_matrix, -2, -1)
        return self

    @animated_function(animated_args={"num_degrees": 0}, unique_args=["axis"])
    def rotate_and_scale(
        self,
        num_degrees: float | torch.Tensor,
        axis: torch.Tensor,
        scale: float | torch.Tensor,
        interpolation: float = 1,
    ) -> Mob:
        """Performs both rotation and scaling simultaneously.

        Parameters
        ----------
        num_degrees : float or torch.Tensor
            The total angle of rotation in degrees.
        axis : torch.Tensor
            The 3-D axis of rotation.
        scale : float or torch.Tensor
            The target absolute scale factor.
        interpolation : float, optional
            The interpolation factor for the animation. Defaults to 1.

        Returns
        -------
        Mob
            The Mob instance itself, allowing for method chaining.
        """
        # Apply interpolated rotation
        interpolated_degrees = num_degrees * interpolation
        self.rotate(interpolated_degrees, axis)

        # Apply interpolated scale
        target_scale = cast_to_tensor(scale)
        interpolated_scale = (
            self.scale_coefficient * (1 - interpolation)
            + interpolation * target_scale * self.scale_coefficient
        )
        self.set_scale(interpolated_scale)
        return self

    def rotate_around_line(self, line_point, line_direction, *args, **kwargs):
        rotation_point = project_point_onto_line(
            self.location, line_direction, line_point
        )
        kwargs["axis"] = line_direction
        return self.rotate_around_point(rotation_point, *args, **kwargs)

    @animated_function(animated_args={"num_degrees": 0}, unique_args=["axis"])
    def rotate_around_point(
        self,
        point: torch.Tensor,
        num_degrees: float | torch.Tensor,
        axis: torch.Tensor = OUT,
    ) -> Mob:
        """Rotates the Mob around an arbitrary point in space.

        Parameters
        ----------
        point
            The 3-D point to rotate around.
        num_degrees
            The angle of rotation in degrees.
        axis
            The 3-D axis of rotation (passing through `point`).
            This vector does not need to be normalized. Defaults to `OUT`.

        Returns
        -------
        :class:`~.Mob`
            The Mob instance itself, allowing for method chaining.

        """
        # Calculate displacement from the rotation point to the Mob's current location
        displacement_from_point = self.location - point
        # Rotate this displacement vector
        rotated_displacement = rotate_vector_around_axis(
            displacement_from_point, num_degrees, axis, dim=-1
        )
        # Calculate the new location by adding the rotated displacement back to the point
        new_location = rotated_displacement + point
        self.location = (
            new_location  # This setter handles recursive rotation and updates
        )
        return self

    def orbit_around_point(self, point, num_degrees, axis):
        with Sync():
            self.rotate_around_point(point, num_degrees, axis)
            self.rotate(num_degrees, axis)
        return self

    def orbit_around_line(self, line_point, line_direction, *args, **kwargs):
        rotation_point = project_point_onto_line(
            self.location, line_direction, line_point
        )
        kwargs["axis"] = line_direction
        return self.orbit_around_point(rotation_point, *args, **kwargs)

    @animated_function(animated_args={"num_degrees": 0}, unique_args=["axis"])
    def rotate_around_point_non_recursive(
        self,
        point: torch.Tensor,
        num_degrees: float | torch.Tensor,
        axis: torch.Tensor = OUT,
    ) -> Mob:
        """Rotates the Mob around an arbitrary point in space without affecting its children.

        Parameters
        ----------
        point
            The 3-D point to rotate around.
        num_degrees
            The angle of rotation in degrees.
        axis
            The 3-D axis of rotation (passing through `point`).
            Defaults to `OUT`.

        Returns
        -------
        :class:`~.Mob`
            The Mob instance itself, allowing for method chaining.

        """
        displacement_from_point = self.location - point
        rotated_displacement = rotate_vector_around_axis(
            displacement_from_point, num_degrees, axis, dim=-1
        )
        new_location = rotated_displacement + point
        self.set_non_recursive(location=new_location)
        return self

    def move_to_point_along_arc(
        self,
        point: torch.Tensor,
        arc_angle_degrees: float | torch.Tensor,
        arc_normal: torch.Tensor = OUT,
        recursive: bool = True,
    ) -> Mob:
        # TODO: This is bugged and needs to be fixed. The mathematical implementation for arc center calculation might be unstable or incorrect for all cases.
        """Moves the Mob to a target point along a circular arc. ***Currently bugged***

        Parameters
        ----------
        point : torch.Tensor
            The target 3-D location.
        arc_angle_degrees : float or torch.Tensor
            The angle subtended by the arc, in degrees. The sign determines
            the direction of rotation along the arc
            (clockwise/counter-clockwise).
        arc_normal : torch.Tensor, optional
            The normal vector to the plane of the arc. Defaults to `OUT`
            (positive Z-axis).
        recursive : bool, optional
            If True, applies the rotation recursively to children,
            maintaining their relative positions. Defaults to True.

        Returns
        -------
        Mob
            The Mob instance itself, allowing for method chaining.
        """
        warnings.warn(
            "move_to_point_along_arc (also reached via move_to(path_arc_angle=...)) "
            "is known to be bugged: the arc-center calculation can be unstable or "
            "wrong for some configurations.",
            stacklevel=2,
        )
        my_location = self.location
        displacement_unnormalized = point - my_location
        # Normalize the displacement for consistent direction calculations
        displacement_normalized = F.normalize(displacement_unnormalized, p=2, dim=-1)

        # Calculate a vector orthogonal to both displacement and arc_normal, which will define one axis for arc plane
        displacement_normal_orthogonal = F.normalize(
            broadcast_cross_product(displacement_normalized, arc_normal), p=2, dim=-1
        )

        angle_sign = cast_to_tensor(arc_angle_degrees).sign()
        abs_arc_angle_degrees = (
            abs(arc_angle_degrees)
            if not isinstance(arc_angle_degrees, torch.Tensor)
            else arc_angle_degrees.abs()
        )

        # Calculate two vectors `in1` and `in2` that define the tangents or radii for arc center calculation.
        # These are rotated versions of the normalized displacement, used to form a geometric intersection.
        in1 = F.normalize(
            rotate_vector_around_axis(
                displacement_normalized, abs_arc_angle_degrees - 90, arc_normal, -1
            ),
            p=2,
            dim=-1,
        )
        in2 = F.normalize(
            rotate_vector_around_axis(
                displacement_normalized, -(abs_arc_angle_degrees + 90), arc_normal, -1
            ),
            p=2,
            dim=-1,
        )

        # Calculate the angle of the full circumference based on the dot product of in1 and in2
        arc_circumference_angle = (
            dot_product(-in1, -in2).clamp_(min=-1, max=1).arccos_()
        )

        # Handle edge cases where angle is exactly 180 degrees or displacement is zero,
        # which can lead to division by zero or ambiguous arc centers.
        # In such cases, a simple midpoint is used as the arc center.
        zero_displacement_mask = (
            ((math.pi - arc_circumference_angle).abs() <= 1e-5)
            | (displacement_unnormalized.norm(p=2, dim=-1, keepdim=True) <= 1e-5)
        ).float()

        # Calculate arc center candidates using geometric intersection formulas.
        # These involve solving linear equations based on the dot products of vectors.
        arc_center1 = (
            my_location + point
        ) * 0.5  # Midpoint for 180-degree or zero-displacement cases

        x1, y1 = 0.0, 0.0
        x2, y2 = (
            dot_product(in1, displacement_normal_orthogonal),
            dot_product(in1, displacement_normalized),
        )
        x3, y3 = (
            dot_product(displacement_normalized, displacement_normal_orthogonal),
            dot_product(displacement_normalized, displacement_normalized),
        )
        x4, y4 = (
            dot_product(in2, displacement_normal_orthogonal),
            dot_product(in2, displacement_normalized),
        )

        # Solving for intersection point in a 2D plane defined by displacement_normal_orthogonal and displacement_normalized
        # These are standard formulas for line-line intersection, adapted for vector components.
        intersect_x = (
            (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)
        ) / ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
        intersect_y = (
            (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)
        ) / ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))

        # Reconstruct the arc center from the intersection point and the initial location
        arc_center2 = (
            my_location
            + intersect_x * displacement_normal_orthogonal
            + intersect_y * displacement_normalized
        )
        arc_center2 = arc_center2.nan_to_num_(
            0, 0, 0
        )  # Handle potential NaNs from division by zero

        # Select the appropriate arc center based on the edge case mask
        final_arc_center = (
            arc_center1 * (zero_displacement_mask)
            + (1 - zero_displacement_mask) * arc_center2
        )

        # Perform the rotation around the calculated arc center
        if recursive:
            return self.rotate_around_point(
                final_arc_center,
                arc_circumference_angle * RADIANS_TO_DEGREES * angle_sign,
                arc_normal,
            )
        else:
            return self.rotate_around_point_non_recursive(
                final_arc_center,
                arc_circumference_angle * RADIANS_TO_DEGREES * angle_sign,
                arc_normal,
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
        with Off(), NoExtra(priority_level=1):
            clone_mob = self.clone(reset_history=False, spawn=False)
            descendant_map = dict(zip(self.get_descendants(), clone_mob.get_descendants()))

            # Hand this mob's recorded history over to the clone. All recorded
            # attribute edits reference this mob's current rows in the global
            # attribute timelines, so the clone takes ownership of those rows,
            # while this mob keeps the fresh rows allocated during cloning
            # (which hold the current values and no history). Past function
            # applications are re-targeted at the clone so that at render time
            # they replay onto the old rows.
            timeline = TimelineManager.instance()
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
                        del attr_timeline.mob_id_to_inds[orig.id]
                        attr_timeline.mob_id_to_inds[clone.id] = orig_inds
                        attr_timeline.add(
                            orig, attr_timeline.get(orig_inds), overwrite=True
                        )
                    else:
                        attr_timeline.mob_id_to_inds[orig.id] = clone_inds
                        attr_timeline.mob_id_to_inds[clone.id] = orig_inds
            for f in timeline.function_timeline.function_applications:
                if f.caller in descendant_map:
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
        #TODO this: available_attrs = union(self.animatable_attrs, TimelineManager.attr_to_timeline.keys())
        for p in property_names:
            if not hasattr(self, p) and (p not in self.animatable_attrs):
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
        with Sync():
            for key, value in kwargs.items():
                self.__setattr__(
                    key, value
                )  # Calls the property setters, which handle animation and recursion
        return self

    def get_forward_basis(self):
        return unsquish(self.basis, -1, 3)[..., 2, :]

    def get_right_basis(self):
        return unsquish(self.basis, -1, 3)[..., 0, :]

    def get_upwards_basis(self):
        return unsquish(self.basis, -1, 3)[..., 1, :]

    def get_forward_direction(self) -> torch.Tensor:
        """Gets the Mob's current forward direction vector (normalized).
        This corresponds to the third column of its normalized basis matrix.

        Returns
        -------
        torch.Tensor
            A 3-D vector representing the forward direction.

        """
        return F.normalize(unsquish(self.basis, -1, 3)[..., 2, :], p=2, dim=-1)

    def get_right_direction(self) -> torch.Tensor:
        """Gets the Mob's current right direction vector (normalized).
        This corresponds to the first column of its normalized basis matrix.

        Returns
        -------
        torch.Tensor
            A 3-D vector representing the right direction.

        """
        return F.normalize(unsquish(self.basis, -1, 3)[..., 0, :], p=2, dim=-1)

    def get_upwards_direction(self) -> torch.Tensor:
        """Gets the Mob's current upwards direction vector (normalized).
        This corresponds to the second column of its normalized basis matrix.

        Returns
        -------
        torch.Tensor
            A 3-D vector representing the upwards direction.

        """
        return F.normalize(unsquish(self.basis, -1, 3)[..., 1, :], p=2, dim=-1)

    def look(self, direction: torch.Tensor, axis: int = 2) -> Mob:
        """Rotates the Mob so that one of its local axes points in the given direction.

        Parameters
        ----------
        direction
            The target 3-D direction vector that the specified
            local axis should point towards. This vector does not need to be normalized.
        axis
            The index of the local axis to align.
            0 for right (X-axis), 1 for up (Y-axis), 2 for forward (Z-axis).
            Defaults to 2 (forward vector).

        Returns
        -------
        :class:`~.Mob`
            The Mob instance itself, allowing for method chaining.

        """
        # Get the rotation parameters (angle and axis) needed to align the current local axis
        # with the target direction.
        rotation_angle_degrees, rotation_axis = get_rotation_between_3d_vectors(
            unsquish(self.normalized_basis, -1, 3)[
                ..., axis, :
            ],  # Current orientation of specified axis
            F.normalize(direction, p=2, dim=-1),  # Normalized target direction
            dim=-1,
        )
        # Apply the rotation
        return self.rotate(rotation_angle_degrees, rotation_axis)

    def look_and_scale(
        self, direction: torch.Tensor, scale: float | torch.Tensor, axis: int = 2
    ) -> Mob:
        """Rotates the Mob to look in a specific direction and simultaneously scales it.

        Parameters
        ----------
        direction : torch.Tensor
            The target 3-D direction vector to look at.
        scale : float or torch.Tensor
            The target absolute scale factor.
        axis : int, optional
            The index of the local axis to align (0: right, 1: up,
            2: forward). Defaults to 2 (forward).

        Returns
        -------
        Mob
            The Mob instance itself, allowing for method chaining.

        """
        # Get rotation parameters from the 'look' logic
        rotation_angle_degrees, rotation_axis = get_rotation_between_3d_vectors(
            unsquish(self.normalized_basis, -1, 3)[..., axis, :],
            F.normalize(direction, p=2, dim=-1),
            dim=-1,
        )
        # Apply both rotation and scale using the combined animated function
        return self.rotate_and_scale(rotation_angle_degrees, rotation_axis, scale)

    def look_at(self, point: torch.Tensor, axis: int = 2) -> Mob:
        """Rotates the Mob to face a specific 3-D point.
        The Mob's "forward" direction (or the specified `axis`) will be oriented towards the point.

        Parameters
        ----------
        point
            The 3-D point to look at.
        axis
            The index of the local axis to align (0: right, 1: up, 2: forward).
            Defaults to 2 (forward vector).

        Returns
        -------
        :class:`~.Mob`
            The Mob instance itself, allowing for method chaining.

        """
        # Calculate the direction vector from the Mob's current location to the target point
        direction_to_point = point - self.location
        return self.look(direction_to_point, axis=axis)

    def spawn_tilewise_recursive(self):
        """
        Animates the spawning of the Mob and its primitive children in a lagged, "tile-wise" manner.
        Each tile/primitive appears from a random direction with an ease-out-exponential effect.
        """
        # Collect all primitive tiles from the Mob and its descendants
        tiles = [
            mob.tiles
            for mob in traverse(self.get_descendants())
            if hasattr(mob, "tiles") and not mob.tiles.is_spawned()
        ]
        with AnimationContext(run_time=3):
            # Animate each tile/primitive appearing from a random direction
            animate_lagged_by_location(
                tiles,
                lambda m: m.spawn_from_random_direction(),
                F.normalize(RIGHT * 1.5 + DOWN, p=2, dim=-1),
            )
        return self

    def despawn_tilewise_recursive(self):
        """
        Animates the despawning of the Mob and its primitive children in a lagged, "tile-wise" manner.
        Each tile/primitive disappears into a random direction with an ease-out-exponential effect.
        """
        # Collect all primitive tiles from the Mob and its descendants
        tiles = [
            mob.tiles
            for mob in traverse(self.get_descendants())
            if hasattr(mob, "tiles")
        ]
        with AnimationContext(run_time=3):
            # Animate each tile/primitive disappearing into a random direction
            animate_lagged_by_location(
                tiles,
                lambda m: m.despawn_from_random_direction(),
                F.normalize(RIGHT * 1.5 + DOWN, p=2, dim=-1),
            )
        return self

    def spawn_from_random_direction(self, travel_distance: float = 0.1):
        """
        Animates the Mob appearing from a random direction, fading in and optionally rotating.
        This sets the initial opacity to 0 and then animates it to 1.
        """
        with Off():  # Ensure initial state setting is not recorded as an animation
            self.opacity = 0
        self._create_recursive(
            animate=False
        )  # Mark as created without immediate animation
        with Sync(
            run_time=None, rate_func=ease_out_exp
        ):  # Synchronized animation with ease-out
            # Example of potential animated properties (currently commented out)
            # self.location = loc
            # self.rotate(720, F.normalize(torch.randn_like(self.location), p=2, dim=-1))
            self.opacity = 1  # Animate opacity to full
            # with Synchronized(run_time=2, rate_func=tan):
        return self

    def __len__(self) -> int:
        """Returns the batch size of the Mob, typically derived from its location tensor.
        This allows Mobs to behave somewhat like batched data structures.

        """
        return self.location.shape[-2] if hasattr(self, "location") else 1

    def despawn_from_random_direction(self, travel_distance: float = 0.1):
        """Animates the Mob disappearing into a random direction, fading out and optionally rotating.
        This animates the opacity to 0 and then marks the Mob as destroyed.

        """
        with Sync(
            run_time=None, rate_func=inversed(ease_out_exp)
        ):  # Synchronized animation with inversed ease-out
            current_location = self.location
            # Example of potential animated properties (currently commented out)
            # self.location = current_location + torch.randn_like(current_location) * travel_distance
            # self.rotate(720, F.normalize(torch.randn_like(self.location), p=2, dim=-1))
            self.opacity = 0  # Animate opacity to zero
            # self._destroy_recursive(animate=False)  # Mark as destroyed without immediate animation
            # with Synchronized(run_time=2, rate_func=tan):
            # self.destroy()
        return self

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
