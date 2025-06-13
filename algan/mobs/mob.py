from __future__ import annotations

import math
from collections import defaultdict

#from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F
from torch import Tensor, TensorType

from algan.animation.animatable import Animatable, animated_function, ModificationHistory, TimeInterval
from algan.animation.animation_contexts import Seq, Off, Sync, AnimationContext, NoExtra
from algan.constants.spatial import *
from algan.geometry.geometry import rotate_vector_around_axis, get_rotation_between_3d_vectors, project_point_onto_line, get_rotation_around_axis, map_global_to_local_coords, map_local_to_global_coords, \
    get_rotation_between_bases
from algan.constants.rate_funcs import ease_out_exp, inversed, identity
from algan.defaults.style_defaults import DEFAULT_BUFFER
from algan.utils.animation_utils import animate_lagged_by_location
from algan.utils.python_utils import traverse
from algan.utils.tensor_utils import dot_product, broadcast_gather, unsqueeze_right, unsquish, squish, broadcast_cross_product, cast_to_tensor


class Mob(Animatable):
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

    def __init__(self, location: torch.Tensor = ORIGIN, basis: torch.Tensor = squish(torch.eye(3)),
                 color: Color | None = None, opacity: float = 1, glow: float = 0, *args, **kwargs):
        self.register_attrs_as_animatable(
            {'location', 'basis', 'scale_coefficient', 'color', 'opacity', 'max_opacity', 'glow'}, Mob)
        self.recursing = True
        self.exclude_from_boundary = False
        super().__init__(*args, **kwargs)
        # Defines how attributes changes are inherited by children Mobs (e.g., additive for location, multiplicative for scale)
        self.attr_to_relations = defaultdict(lambda: (lambda x, y: y, lambda x, y: y))
        additive_relation = (lambda x, y: x + y, lambda x, y: y - x)
        self.attr_to_relations.update({'location': additive_relation,
                                       # 'glow': additive_relation, # Currently commented out, but could be additive.
                                       'basis': (lambda x, y: squish(unsquish(x, -1, 3) @ unsquish(y, -1, 3), -2, -1),
                                                 lambda x, y: squish(
                                                     get_rotation_between_bases(unsquish(x, -1, 3), unsquish(y, -1, 3)),
                                                     -2, -1)),
                                       'scale_coefficient': (lambda x, y: (x * y),
                                                             lambda x, y: squish(
                                                                 (unsquish(y, -1, 3).norm(p=2, dim=-1, keepdim=True) /
                                                                  unsquish(x, -1, 3).norm(p=2, dim=-1,
                                                                                          keepdim=True)).expand(
                                                                     *([-1] * (x.dim())), 3), -2, -1))})
        self.location = cast_to_tensor(location)
        self.basis = cast_to_tensor(basis)

        if color is None:
            color = self.get_default_color()
        self.color = color
        self.max_opacity = cast_to_tensor(opacity)
        self.opacity = cast_to_tensor(1)  # Current opacity, can be animated
        self.glow = cast_to_tensor(glow)
        self.num_points_per_object = 1

    def reset_basis(self):
        """Resets the Mob's basis to the identity matrix (no rotation, unit scale)."""
        self.basis = cast_to_tensor(cast_to_tensor(squish(torch.eye(3))))

    def register_attrs_as_animatable(self, attrs: set[str], my_class=None):
        """
        Registers attributes as animatable, meaning their changes can be tracked
        and interpolated over time for animation.

        This method dynamically creates property getters and setters for the
        specified attributes if they don't already exist, allowing them to be
        controlled by the animation system. When an animatable attribute is
        modified, the change is recorded in the mob's `ModificationHistory`.

        Parameters
        ----------
            attrs (set[str] or str): A collection of attribute names (or a single
                attribute name) to register as animatable.
            my_class (type, optional): The class to which the property getters
                and setters should be attached. Defaults to the current Mob's class.
        """
        if isinstance(attrs, str):
            attrs = {attrs, }
        if not isinstance(attrs, set):
            attrs = set(attrs)
        if not hasattr(self, 'animatable_attrs'):
            self.animatable_attrs = set()
        if my_class is None:
            my_class = self.__class__
        for attr in attrs:
            self.add_property_getter_and_setter(attr, my_class)
        self.animatable_attrs.update(attrs)

    def add_property_getter_and_setter(self, property_name: str, class_to_attach_to=None):
        """Dynamically adds a property with a getter and setter for a given attribute name.

        The getter will retrieve the current (potentially animated) value of the attribute
        from the `AnimatableData` dict. The setter will set the value of the
        attribute in the `AnimatableData` dict, recording the change in the
         `ModificationHistory` for animation.

        Parameters
        ----------
            property_name (str): The name of the property to create (e.g., 'location', 'color').
            class_to_attach_to (type, optional): The class to which this property
                will be added. Defaults to the current Mob's class.

        """
        if class_to_attach_to is None:
            class_to_attach_to = self.__class__
        if hasattr(class_to_attach_to, property_name):
            return

        @property
        def prop(self):
            return self.getattribute_animated(property_name)

        @prop.setter
        def prop(self, value):
            return self.setattr_absolute(property_name, value)

        setattr(class_to_attach_to, property_name, prop)

    def get_descendants(self, include_self: bool = True) -> list['Mob']:
        """Retrieves all descendant Mobs in the hierarchy, optionally including itself.

        Parameters
        ----------
            include_self (bool, optional): If True, the current Mob instance
                is included in the returned list. Defaults to True.

        Returns
        -------
            list[Mob]: A flat list containing the Mob and all its children,
                grandchildren, and so on.

        """
        return list(traverse([*([self] if include_self else []),
                              [c.get_descendants() for c in self.children] if hasattr(self, 'children') else []]))

    def set_time_inds_to(self, mob: 'Mob'):
        """Synchronizes the animation time indices of this Mob with another Mob.

        This is used internally to ensure consistent animation states between
        mobs at animation time.

        parameters
        -----------
            mob (Mob): The Mob whose time indices will be copied.

        """
        time_inds = mob.data
        if self.data.time_inds_materialized is None and time_inds.time_inds_materialized is not None:
            self.data.animatable.set_state_pre_function_applications(time_inds.time_inds_materialized.amin(),
                                                                     time_inds.time_inds_materialized.amax() + 1)
        self.data.time_inds_active = time_inds.time_inds_active

    def _expand_batch_if_necessary(self, value: torch.Tensor) -> torch.Tensor:
        """ Internal helper to expand a tensor's batch dimension if it's a singleton
        and the parent has a larger batch size.
        """
        if value.shape[-2] == 1 and self.parent_batch_sizes is not None:
            return value.expand(*([-1 for _ in range(value.dim() - 2)]), len(self.parent_batch_sizes), -1).contiguous()
        return value

    @animated_function(animated_args={'interpolation': 0.0}, unique_args=['key', 'recursive'])
    def apply_absolute_change_two(self, key: str, change1: any, change2: any, interpolation: float = 1.0,
                                  recursive: str = 'True'):
        """Applies an animated change to an attribute, interpolating between two target values.

        The interpolation first moves from the current value towards `change1` from
        t=0 to 0.5, then moves from `change1` to `change2` from t=0.5 to 1.

        Parameters
        ----------
            key (str): The name of the attribute to change (e.g., 'location', 'color').
            change1 (Any): The first target value for the attribute.
            change2 (Any): The second target value for the attribute.
            interpolation (float, optional): The interpolation factor used for animation.
            recursive (str, optional): If equal to "True", applies the change recursively
                to all child Mobs. Defaults to "True".

        Returns
        -------
            Mob: The Mob instance itself, allowing for method chaining.

        """
        relation_func = self.attr_to_relations[key][0]
        current_value = self.__getattribute__(key)
        interpolation = cast_to_tensor(interpolation) * 2  # Double interpolation for 2-stage animation

        # Calculate the interpolated value based on the two changes
        # m is a mask for when interpolation goes beyond 1.0
        mask_interp_gt_1 = (interpolation > 1).float()
        interpolated_value = (current_value * (1 - interpolation) + interpolation * change1) * (1 - mask_interp_gt_1) + \
                             mask_interp_gt_1 * (change1 * (2 - interpolation) + (interpolation - 1) * change2)

        self.setattr_and_record_modification(key, relation_func(current_value, interpolated_value))

        if recursive == 'True':
            for c in self.children:
                c.set_time_inds_to(self)
                if c.parent_batch_sizes is not None:
                    def expand(x):
                        if x.shape[-2] == 1:
                            x = x.expand(*([-1 for _ in range(x.dim() - 2)]), len(c.parent_batch_sizes),
                                         -1).contiguous()
                        return x

                    interpolated_value = torch.repeat_interleave(expand(interpolated_value), c.parent_batch_sizes, -2)
                c.apply_relative_change(key, interpolated_value, interpolation=1, recursive=recursive)
        return self

    def pulse_color(self, color: torch.Tensor, set_opaque: bool = False) -> 'Mob':
        """Animates a color pulse effect.

        The Mob's color changes to the target `color` and then animates back to its
        original color. This uses `apply_absolute_change_two` internally for the two-stage
        color change.

        parameters
        ----------
            color (torch.Tensor): The color to pulse to.
            set_opaque (bool, optional): If True, also animates opacity to 1.0
                during the pulse. Defaults to False.

        Returns
        -------
            Mob: The Mob instance itself, allowing for method chaining.

        """
        with Sync():
            self.apply_absolute_change_two('color', color, self.color)
            if set_opaque:
                self.apply_absolute_change_two('opacity', 1, 1)
        return self

    def wave_color(self, color: torch.Tensor, wave_length: float = 1, reverse: bool = False,
                   direction: torch.Tensor | None = None, **kwargs) -> 'Mob':
        """Applies a color wave effect across the Mob and its descendants.

        The color change propagates spatially across the mob's constituent parts.

        Parameters
        ----------
            color (torch.Tensor): The target color for the wave.
            wave_length (float, optional): Controls the spatial extent (length) of the wave.
                A smaller value means a faster-propagating or more compressed wave.
                Defaults to 1.
            reverse (bool, optional): If True, the wave propagates in the
                opposite direction. Defaults to False.
            direction (torch.Tensor, optional): The 3D vector defining the
                direction of wave propagation. If None, uses the Mob's
                upwards direction. Defaults to None.
            **kwargs: Additional keyword arguments passed to `pulse_color`
                for each individual part of the wave animation.

        Returns
        -------
            Mob: The Mob instance itself, allowing for method chaining.

        """
        if direction is None:
            direction = self.get_upwards_direction()
        with AnimationContext(run_time_unit=3, rate_func=identity):
            # Filters for primitive parts to ensure the wave animates on individual rendering elements
            #TODO change this to use non_recursive set
            primitive_mobs = [_ for _ in self.get_descendants() if _.is_primitive]
            animate_lagged_by_location(primitive_mobs, lambda x: x.pulse_color(color, **kwargs),
                                       direction * (-1 if reverse else 1), 1.5)
        return self

    @animated_function(animated_args={'interpolation': 0.0}, unique_args=['key', 'recursive', 'relation_key'])
    def apply_relative_change(self, key: str, change: any, interpolation: float = 1.0, recursive: str = "True",
                              relation_key: str = "None") -> 'Mob':
        """Applies an animated relative change to an attribute.

        The `change` is scaled by `interpolation` and then combined with the
        current attribute value using a predefined relation (e.g., addition
        for location, multiplication for scale).

        parameters
        ----------
            key (str): The name of the attribute to change.
            change (Any): The relative change to apply (e.g., a displacement vector, a scaling factor).
            interpolation (float, optional): The interpolation factor for the change.
                A value of 0.0 means no change; 1.0 applies the full `change`.
                Defaults to 1.0.
            recursive (str, optional): If equal to "True", applies the change recursively
                to all child Mobs. Defaults to "True".
            relation_key (str, optional): The key to look up the specific relation function
                (how the `change` is combined with the current value). If "None",
                `key` is used as the relation key. Defaults to "None".

        Returns
        -------
            Mob: The Mob instance itself, allowing for method chaining.

        """
        change = change * interpolation
        relation = self.attr_to_relations[relation_key][0]
        current_value = self.__getattribute__(key)

        self.setattr_and_record_modification(key, relation(current_value, change))
        if recursive == 'True':
            for c in self.children:
                c.set_time_inds_to(self)
                change2 = change
                if c.parent_batch_sizes is not None:
                    def expand(x):
                        if x.shape[-2] == 1:
                            x = x.expand(*([-1 for _ in range(x.dim() - 2)]), len(c.parent_batch_sizes), -1).contiguous()
                        return x
                    change2 = torch.repeat_interleave(expand(change2), c.parent_batch_sizes, -2)
                c.apply_relative_change(key, change2, interpolation=1, recursive=recursive, relation_key=relation_key)
        return self

    @animated_function(animated_args={'interpolation': 0.0}, unique_args=['key'])
    def apply_set_value(self, key: str, change: any, interpolation: float = 1.0) -> 'Mob':
        """Sets an attribute's value, interpolating from its current value to the target `change`.

        This is a direct linear interpolation (lerp) from the current value to the target,
        rather than applying a relative change.

        parameters
        ----------
            key (str): The name of the attribute to set.
            change (Any): The target value for the attribute (e.g., a specific location, a final color).
            interpolation (float, optional): The interpolation factor.
                0.0 means the attribute remains its current value; 1.0 means it becomes `change`.
                Defaults to 1.0.

        Returns
        -------
            Mob: The Mob instance itself, allowing for method chaining.

        """
        current_value = self.__getattribute__(key)
        try:
            # Direct linear interpolation
            interpolated_value = current_value * (1 - interpolation) + interpolation * change
        except RuntimeError:
            # Handle cases where interpolation tensor dimensions don't match exactly,
            # typically for batched values where interpolation might need expansion.
            # This logic ensures the interpolation tensor matches the batch dimensions of the value.
            interpolation = torch.cat((interpolation, torch.zeros_like(
                current_value[..., -(current_value.shape[-2] - interpolation.shape[-2]):, :])), -2)
            interpolated_value = current_value * (1 - interpolation) + interpolation * change
        self.setattr_and_record_modification(key, interpolated_value)
        return self

    @animated_function(animated_args={'interpolation': 0.0}, unique_args=['key', 'recursive'])
    def apply_absolute_change(self, key: str, change: any, interpolation: float = 1.0, recursive: str = "True") -> 'Mob':
        """Applies an animated absolute change to an attribute, interpolating to a target value.

        This method smoothly transitions the attribute's value from its current state
        to the specified `change` value over time, according to the `interpolation` factor.

        parameters
        ----------
            key (str): The name of the attribute to change (e.g., 'location', 'opacity').
            change (Any): The target absolute value for the attribute.
            interpolation (float, optional): The interpolation factor.
                0.0 means no change from the current value; 1.0 means the attribute becomes `change`.
                Defaults to 1.0.
            recursive (str, optional): If equal to "True", applies the change recursively
                to all child Mobs. Defaults to "True".

        Returns
        -------
            Mob: The Mob instance itself, allowing for method chaining.

        """
        original_change_value = change  # Store original to propagate to children
        if hasattr(self, key):
            relation_func = self.attr_to_relations[key][0]
            current_value = self.__getattribute__(key)
            try:
                # Direct linear interpolation
                interpolated_value = current_value * (1 - interpolation) + interpolation * change
            except RuntimeError:
                # Adjust interpolation tensor dimensions if necessary for batching
                interpolation = torch.cat((interpolation, torch.zeros_like(
                    current_value[..., -(current_value.shape[-2] - interpolation.shape[-2]):, :])), -2)
                interpolated_value = current_value * (1 - interpolation) + interpolation * change
            self.setattr_and_record_modification(key, relation_func(current_value, interpolated_value))

        if recursive == 'True':
            change = original_change_value
            for c in self.children:
                c.set_time_inds_to(self)
                change2 = change
                interpolation2 = interpolation
                if c.parent_batch_sizes is not None:
                    def expand(x):
                        if x.shape[-2] == 1:
                            x = x.expand(*([-1 for _ in range(x.dim() - 2)]), len(c.parent_batch_sizes),
                                         -1).contiguous()
                        return x

                    change2 = torch.repeat_interleave(expand(change2), c.parent_batch_sizes, -2)
                    if isinstance(interpolation2, torch.Tensor):
                        interpolation2 = torch.repeat_interleave(expand(interpolation2), c.parent_batch_sizes, -2)
                c.apply_absolute_change(key, change2, interpolation=interpolation2, recursive=recursive)
        return self

    def setattr_basic(self, key: str, value: any) -> 'Mob':
        """Sets an attribute's value directly without complex animation logic.

        If the attribute is animatable and an animation context is active,
        this will still record the change as a step-change (instantaneous transition).
        This method does not support recursive application to children. For animated or
        recursive changes, use `setattr_absolute` or `setattr_relative`.

        parameters
        ----------
            key (str): The name of the attribute to set.
            value (Any): The new value for the attribute.

        Returns
        -------
            Mob: The Mob instance itself, allowing for method chaining.

        """
        value = cast_to_tensor(value)
        if not hasattr(self, 'data'):
            # If data is not initialized, set attribute directly
            self.__setattr__(f'_{key}', value)
            return self
        if not hasattr(self, key):
            # If attribute doesn't exist yet as a property, store in data_dict_active
            self.data.data_dict_active[key] = value
            return self

        # For existing animatable attributes, record as an instantaneous "set value"
        self.apply_set_value(key, value)
        return self

    def setattr_relative(self, key: str, value: any, relation_key: str | None = None) -> 'Mob':
        """Sets an attribute by applying a relative change.

        This method calculates the `change` needed to transition from the current
        value to the target `value` based on the inverse of the predefined relation.
        It then applies this `change` relatively to all children.

        parameters
        ----------
            key (str): The name of the attribute to set.
            value (Any): The target value for the attribute.
            relation_key (str, optional): The key to use for looking up the relation functions.
                If None, `key` itself is used. Defaults to None.

        Returns
        -------
            Mob: The Mob instance itself, allowing for method chaining.

        """
        if self.animation_manager.context.trace_mode:
            self.animation_manager.context.traced_mobs = self.animation_manager.context.traced_mobs.union(set(self.get_descendants()))
            return self
        value = cast_to_tensor(value)
        if not hasattr(self, 'data'):
            self.__setattr__(f'_{key}', value)
            return self
        if not hasattr(self, key):
            self.data.data_dict_active[key] = value
            return self

        resolved_relation_key = key if relation_key is None else relation_key
        # Get the relative and inverse relative functions for the attribute
        _, inverse_relation_func = self.attr_to_relations[resolved_relation_key]
        current_value = self.__getattribute__(key)
        # Calculate the 'change' that, when applied relatively, results in 'value'
        change = inverse_relation_func(current_value, value)
        # Apply the calculated relative change, respecting recursion flag
        return self.apply_relative_change(key, change, recursive="True" if self.recursing else "False", relation_key=resolved_relation_key)

    def setattr_absolute(self, key: str, value: any) -> 'Mob':
        """Sets an attribute to a value absolutely, animating the transition.

        This method directly interpolates the attribute's value from its current
        state to the specified `value`.

        Parameters
        ----------
            key (str): The name of the attribute to set.
            value (Any): The target absolute value.

        Returns
        -------
            Mob: The Mob instance itself, allowing for method chaining.

        """
        if self.animation_manager.context.trace_mode:
            self.animation_manager.context.traced_mobs = self.animation_manager.context.traced_mobs.union(set(self.get_descendants()))
            return self
        value = cast_to_tensor(value)
        if not hasattr(self, 'data'):
            self.__setattr__(f'_{key}', value)
            return self
        if not hasattr(self, key):
            self.data.data_dict_active[key] = value
            return self

        return self.apply_absolute_change(key, value, recursive="True" if self.recursing else "False")

    @property
    def location(self) -> torch.Tensor:
        """The 3D location of the Mob in world space.

        When set, it triggers an animated change to the new location,
        maintaining child Mob positions relative to the parent..

        """
        return self.getattribute_animated('location')

    @location.setter
    def location(self, location: torch.Tensor):
        self.setattr_relative('location', location)

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
        return self.getattribute_animated('basis')

    @property
    def normalized_basis(self) -> torch.Tensor:
        """The Mob's basis matrix with all its row vectors normalized to unit length.
        This represents only the orientation (rotation) without any scaling.

        """
        return squish(unsquish(self.basis, -1, 3) / self.scale_coefficient.unsqueeze(-1), -2, -1)

    def set_basis_inner(self, parent_location: torch.Tensor, old_basis: torch.Tensor, new_basis: torch.Tensor) -> 'Mob':
        """Internal method to set the basis of a child Mob relative to its parent,
        ensuring its global position is maintained despite parent's basis change.

        """
        if self.parent_batch_sizes is not None:
            # Expand parent data to match child's batch size for batched operations
            def expand_for_child(x: torch.Tensor) -> torch.Tensor:
                return x.expand(-1, self.parent_batch_sizes.shape[0], -1)

            parent_location, old_basis, new_basis = [
                torch.repeat_interleave(expand_for_child(val), self.parent_batch_sizes, -2) for val in
                [parent_location, old_basis, new_basis]]

        local_coords = map_global_to_local_coords(parent_location, old_basis, self.location)
        new_global_location = map_local_to_global_coords(parent_location, new_basis, local_coords)
        self.setattr_and_record_modification('location', new_global_location)

        if self.recursing:
            for child in self.children:
                child.set_basis_inner(parent_location, old_basis, new_basis)
        return self

    def set_basis_interpolated(self, *args, **kwargs) -> 'Mob':
        """Wrapper around `_set_basis_interpolated` to handle recursive flag.
        Sets the Mob's basis, interpolating from the current basis to the target.

        """
        if self.animation_manager.context.trace_mode:
            self.animation_manager.context.traced_mobs = self.animation_manager.context.traced_mobs.union(set(self.get_descendants()))
            return self
        return self._set_basis_interpolated(*args, **kwargs, recursive='True' if self.recursing else 'False')

    @animated_function(animated_args={'interpolation': 0}, unique_args=['relation_key', 'recursive'])
    def _set_basis_interpolated(self, basis: torch.Tensor, interpolation: float = 1, relation_key: str = 'basis',
                                recursive: str = "True") -> 'Mob':
        """Internal method to set the Mob's basis, interpolating from the current basis to the target.

        This method also ensures that child Mobs maintain their positions
        relative to this Mob during the basis change by adjusting their locations.

        parameters
        -----------
            basis (torch.Tensor): The target 3x3 basis matrix (flattened to 9 elements).
            interpolation (float, optional): Interpolation factor (0.0 to 1.0).
                0.0 means current basis, 1.0 means target `basis`. Defaults to 1.
            relation_key (str, optional): Key for the relation function,
                typically 'basis' or 'scale_coefficient'. Defaults to 'basis'.
            recursive (str, optional): If "True", applies the rotation recursively
                to children, maintaining their relative positions. Defaults to "True".

        Returns
        -------
            Mob: The Mob instance itself, allowing for method chaining.
        """
        if recursive == 'True':
            ds = (self.get_descendants(include_self=False))
            [d.set_time_inds_to(self) for d in ds]
        old_basis = self.basis if hasattr(self, 'basis') else basis
        interpolated_basis = old_basis * (1 - interpolation) + interpolation * basis

        # Temporarily set recursing flag to control setattr_relative behavior
        original_recursing_state = self.recursing
        self.recursing = recursive == 'True'
        self.setattr_relative('basis', interpolated_basis, relation_key)
        self.recursing = original_recursing_state  # Restore original state

        if recursive == 'True':
            # Adjust children's locations to maintain relative positions after basis change
            for child in self.children:
                child.set_basis_inner(self.location, old_basis, interpolated_basis)
        return self

    @basis.setter
    def basis(self, basis: torch.Tensor):
        self.set_basis_interpolated(basis)

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
        new_basis = squish(F.normalize(unsquish(self.basis, -1, 3), p=2, dim=-1) * scale_coefficient.unsqueeze(-1), -2,
                           -1)
        self.set_basis_interpolated(new_basis, relation_key='scale_coefficient')
        return self

    def clear_cache(self):
        """Clears cached animation data.

        This is typically used internally when animation states are reset or recalculated,
        ensuring that subsequent rendering uses fresh data.

        """
        if self.free_cache:
            self.attr_to_values_full = dict()
            self.attr_to_values = dict()
            self.time_stamps_full = None
            self.time_stamps = None
            self.time_inds_full = None
            self.time_inds = None

    def get_normal(self) -> torch.Tensor:
        """Alias for `get_forward_direction()`.

        Returns
        -------
            the normalized forward direction vector of the Mob.

        """
        return self.get_forward_direction()

    def get_center(self) -> torch.Tensor:
        """Gets the logical center of the Mob.
        Currently, this is simply the `location` attribute.

        """
        # TODO: Make this get mid point of bounding box surrounding self + children.
        return self.location

    def set_location(self, location: torch.Tensor, recursive: bool = True) -> 'Mob':
        """Sets the location of the Mob.

        Parameters
        ----------
            location (torch.Tensor): The target 3D location.
            recursive (bool, optional): If True, also affects the locations of child Mobs
                to maintain their relative positions. Defaults to True.

        Returns
        -------
            Mob: The Mob instance itself, allowing for method chaining.

        """
        if recursive:
            self.location = location
        else:
            self.setattr_non_recursive('location', location)
        return self

    def setattr_non_recursive(self, key: str, value: any):
        """Sets an attribute's value without applying the change to child Mobs.

        This temporarily disables the recursive behavior of attribute setting
        for the duration of this specific attribute modification.

        Parameters
        ----------
            key (str): The name of the attribute to set.
            value (Any): The new value for the attribute.

        """
        if self.animation_manager.context.trace_mode:
            self.animation_manager.context.traced_mobs.add(self)
            return self
        original_recursing_state = self.recursing
        self.recursing = False
        self.__setattr__(key, value)  # Calls the property setter, which then calls apply_absolute_change/set_relative
        self.recursing = original_recursing_state  # Restore original state

    def move_to(self, location: torch.Tensor, path_arc_angle: float | None = None, **kwargs) -> 'Mob':
        """Moves the Mob to a specified location.

        If `path_arc_angle` is provided, the Mob moves along a circular arc.
        Otherwise, it moves in a straight line.

        Parameters
        ----------
            location (torch.Tensor): The target 3D location.
            path_arc_angle (float, optional): The angle of the arc in degrees
                for curved movement. If None, movement is linear. Defaults to None.
            **kwargs: Additional arguments passed to `set_location` or
                `move_to_point_along_arc`.

        Returns
        -------
            Mob: The Mob instance itself.
        """
        if path_arc_angle is None:
            return self.set_location(location, **kwargs)
        return self.move_to_point_along_arc(location, path_arc_angle, **kwargs)

    def move(self, displacement: torch.Tensor, **kwargs) -> 'Mob':
        """Moves the Mob by a given displacement vector from its current location.

        Parameters
        ----------
            displacement (torch.Tensor): The 3D vector by which to move the Mob.
            **kwargs: Additional arguments passed to `move_to` (e.g., `path_arc_angle`).

        Returns
        -------
            Mob: The Mob instance itself, allowing for method chaining.
        """
        self.move_to(self.location + displacement, **kwargs)
        return self

    def get_axis_aligned_lower_corner(self):
        return self.location.amin(-2, keepdim=True)

    def get_axis_aligned_upper_corner(self):
        return self.location.amax(-2, keepdim=True)

    def _get_bounding_box_recursive(self, lower_corner, upper_corner):
        if not self.exclude_from_boundary:
            lower_corner = torch.minimum(lower_corner, self.get_axis_aligned_lower_corner())
            upper_corner = torch.maximum(upper_corner, self.get_axis_aligned_upper_corner())
        for c in self.children:
            lower_corner, upper_corner = c._get_bounding_box_recursive(lower_corner, upper_corner)
        return lower_corner, upper_corner

    def get_bounding_box(self):
        lower_corner, upper_corner = self._get_bounding_box_recursive(self.location.amin(-2, keepdim=True), self.location.amax(-2, keepdim=True))
        out = torch.empty(*lower_corner.shape[:-2], 8, 3)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    a = torch.tensor((i,j,k), device=lower_corner.device)
                    out[..., i*4+j*2+k,:] = lower_corner * (1-a) + (a) * upper_corner
        return out

    def get_boundary_points(self) -> torch.Tensor:
        """Returns the current location of the Mob, serving as its boundary point.
        For more complex Mobs, this should be overridden to provide actual boundary points.
        """
        return self.location

    def get_boundary_points_recursive(self) -> torch.Tensor:
        """Recursively collects boundary points from this Mob and all its descendants.

        Returns
        -------
            torch.Tensor: A concatenated tensor of boundary points from all
                relevant Mobs in the hierarchy.

        """
        num_children = len(self.children)
        if num_children == 0:
            return self.get_boundary_points()
        elif num_children == 1:
            return self.children[0].get_boundary_points_recursive()
        return torch.cat([child.get_boundary_points_recursive() for child in self.children], -2)

    def get_boundary_edge_point(self, direction: torch.Tensor) -> torch.Tensor:
        """Finds the point on the Mob's recursive boundary that is furthest in a given direction.

        Parameters
        ----------
            direction (torch.Tensor): The 3D vector indicating the direction
                along which to find the extreme boundary point.

        Returns
        -------
            torch.Tensor: The 3D coordinate of the boundary point furthest in `direction`.

        """
        all_boundary_points = self.get_boundary_points_recursive()
        # Project all boundary points onto the direction vector and find the one with max projection
        best_index = dot_product(all_boundary_points, direction, dim=-1, keepdim=True).argmax(-2, keepdim=True)
        # Use broadcast_gather to retrieve the actual point
        return broadcast_gather(all_boundary_points, -2, best_index, keepdim=True)

    def get_boundary_in_direction(self, direction: torch.Tensor) -> torch.Tensor:
        """Gets the point on the Mob's boundary (including children) that lies along
        the given direction from its center, and is furthest in that direction.

        Parameters
        ----------
            direction (torch.Tensor): The 3D vector defining the direction.

        Returns
        -------
            torch.Tensor: The 3D coordinate of the boundary point.

        """
        edge_point = self.get_boundary_edge_point(direction)

        def get_median_location(tensor_values: torch.Tensor) -> torch.Tensor:
            """Calculates the median (midpoint of min/max) of a tensor's values."""
            max_val = tensor_values.amax(-2, keepdim=True)
            min_val = tensor_values.amin(-2, keepdim=True)
            return (max_val + min_val) * 0.5

        # Get the logical center of the Mob (or its current location if no complex center is defined)
        mob_center = get_median_location(self.location)
        # Project the offset from the center to the edge point onto the direction
        # and add it back to the center to get the boundary point in that direction.
        return project_point_onto_line(edge_point - mob_center, direction, dim=-1) + mob_center

    def set_x_coord(self, x_coord: torch.Tensor):
        """Sets the x-coordinate of the Mob's location, preserving y and z."""
        new_location = self.location.clone()
        new_location[..., 0] = x_coord[..., 0]
        self.location = new_location

    def set_y_coord(self, y_coord: torch.Tensor):
        """Sets the y-coordinate of the Mob's location, preserving x and z."""
        new_location = self.location.clone()
        new_location[..., 1] = y_coord[..., 0]
        self.location = new_location

    def set_x_y_coord(self, xy_coords: torch.Tensor):
        """Sets the x and y coordinates of the Mob's location, preserving z."""
        new_location = self.location.clone()
        new_location[..., :2] = xy_coords[..., :2]
        self.location = new_location

    def move_next_to(self, target_mob: 'Mob' | torch.Tensor, direction: torch.Tensor, buffer: float = DEFAULT_BUFFER,
                     **kwargs) -> 'Mob':
        """Moves this Mob to be adjacent to another Mob (or a point) in a given direction.

        Parameters
        ----------
            target_mob (Mob or torch.Tensor): The target Mob or a 3D point (torch.Tensor)
                to move next to.
            direction (torch.Tensor): The 3D vector indicating the direction
                from `target_mob` towards where this Mob should be placed.
                This vector does not need to be normalized.
            buffer (float, optional): The minimum distance to maintain between
                the closest edges of the two Mobs. Defaults to `DEFAULT_BUFFER`.
            **kwargs: Additional arguments passed to `move_to`.

        Returns
        -------
            Mob: The Mob instance itself, allowing for method chaining.

        """
        normalized_direction = F.normalize(direction, p=2, dim=-1)
        # Get the boundary point of the target_mob along the given direction
        target_edge_point = target_mob.get_boundary_in_direction(normalized_direction) if not isinstance(target_mob,
                                                                                                         torch.Tensor) else target_mob
        # Get the boundary point of this mob in the opposite direction
        my_edge_point = self.get_boundary_in_direction(-normalized_direction)

        # Calculate the required displacement to move 'my_edge_point' to 'target_edge_point'
        # plus the buffer distance, and then apply it to the Mob's current location.
        displacement_to_align_edges = target_edge_point + normalized_direction * buffer - my_edge_point
        self.move_to(self.location + displacement_to_align_edges, **kwargs)
        return self

    def get_length_in_direction(self, direction: torch.Tensor) -> torch.Tensor:
        """Calculates the spatial extent of the Mob along a given direction.
        This is the distance between the furthest points on its boundary
        in that direction and its opposite.

        Parameters
        ----------
            direction (torch.Tensor): The 3D vector defining the direction.

        Returns
        -------
            torch.Tensor: The length of the Mob along the specified direction.

        """
        # Get the boundary points in the positive and negative directions and calculate their distance
        return (self.get_boundary_in_direction(direction) - self.get_boundary_in_direction(-direction)).norm(p=2,
                                                                                                             dim=-1,
                                                                                                             keepdim=True)

    def move_inline_with_edge(self, mob: 'Mob', direction: torch.Tensor, edge: torch.Tensor | None = None,
                              buffer: float = DEFAULT_BUFFER, **kwargs) -> 'Mob':
        """Moves this Mob so its specified edge is aligned with another Mob's edge
        along a given direction, while maintaining a buffer.

        Parameters
        ----------
            mob (Mob): The target Mob to align with.
            direction (torch.Tensor): The primary direction along which the alignment
                should occur (e.g., `RIGHT`, `UP`).
            edge (torch.Tensor, optional): If specified, this direction is used
                to determine "which side" of *this* Mob to use for alignment.
                If None, `direction` is used for both. Defaults to None.
            buffer (float, optional): The buffer distance to maintain between the edges.
                Defaults to `DEFAULT_BUFFER`.
            **kwargs: Additional arguments for `move`.

        Returns
        -------
            Mob: The Mob instance itself, allowing for method chaining.
        """
        # Calculate the target location for this Mob if it were moved next to itself
        # using the specified `edge` direction and `buffer`. This acts as a reference point.
        old_location_reference = Mob(add_to_scene=False).move_next_to(self, direction if edge is None else edge,
                                                                      buffer).location
        # Calculate the target location for this Mob if it were moved next to the `mob`
        # using the primary `direction` and `buffer`.
        new_location_target = Mob(add_to_scene=False).move_next_to(mob, direction, buffer).location
        # Calculate the displacement needed to move from the reference point to the target point,
        # projected onto the `direction` to ensure alignment only along that axis.
        displacement = project_point_onto_line(new_location_target - old_location_reference, direction)
        self.move(displacement, **kwargs)
        return self

    def move_inline_with_center(self, mob: 'Mob', direction: torch.Tensor, buffer: float = DEFAULT_BUFFER) -> 'Mob':
        """Moves this Mob so its center is aligned with another Mob's center
        along a given direction.

        Parameters
        ----------
            mob (Mob): The target Mob whose center will be aligned with.
            direction (torch.Tensor): The 3D vector specifying the alignment direction.
            buffer (float, optional): Buffer distance (currently seems unused in this specific
                implementation, as it aligns centers, not edges). Defaults to `DEFAULT_BUFFER`.

        Returns
        -------
            Mob: The Mob instance itself, allowing for method chaining.
        """
        # Calculate the displacement vector from this Mob's center to the target Mob's center.
        displacement_to_target_center = mob.location - self.location
        # Project this displacement onto the `direction` to get the movement needed for alignment.
        alignment_displacement = project_point_onto_line(displacement_to_target_center, direction)
        self.location = self.location + alignment_displacement
        return self

    def move_inline_with_mob(self, mob: 'Mob', align_direction: torch.Tensor, center: bool = False,
                             from_mob: 'Mob' | None = None, buffer: float = DEFAULT_BUFFER) -> 'Mob':
        """
        Moves this Mob to align with another Mob along a specific direction,
        either by their edges or by their centers.

        Parameters
        ----------
            mob (Mob): The target Mob to align with.
            align_direction (torch.Tensor): The 3D vector defining the direction
                along which alignment should occur.
            center (bool, optional): If True, aligns the centers of the Mobs.
                If False, aligns their edges. Defaults to False.
            from_mob (Mob, optional): The Mob whose edge/center is considered
                the starting point for calculating displacement. If None,
                this Mob itself is used. Defaults to None.
            buffer (float, optional): Buffer distance between aligned edges (only relevant
                if `center` is False). Defaults to `DEFAULT_BUFFER`.

        Returns
        -------
            Mob: The Mob instance itself, allowing for method chaining.
        """
        if center:
            # Align centers
            mob_reference_point = mob.location
            from_mob_reference_point = self.location if from_mob is None else from_mob.location
        else:
            # Align edges
            mob_reference_point = mob.get_boundary_in_direction(align_direction)
            from_mob_reference_point = (self.get_boundary_in_direction(-align_direction)
                                        if from_mob is None else from_mob.get_boundary_in_direction(-align_direction))

        # Calculate the overall displacement needed for alignment
        displacement = mob_reference_point - from_mob_reference_point
        # Normalize the alignment direction
        normalized_align_direction = F.normalize(align_direction, p=2, dim=-1)
        # Project the displacement onto the normalized direction to ensure movement only along that axis
        return self.move(dot_product(displacement, normalized_align_direction) * normalized_align_direction)

    def get_displacement_to_boundary(self, mob: 'Mob', direction: torch.Tensor) -> torch.Tensor:
        """
        Calculates the vector displacement required to move this Mob's boundary
        to match another Mob's boundary along a given direction.

        Parameters
        ----------
            mob (Mob): The target Mob.
            direction (torch.Tensor): The direction along which to calculate the displacement.

        Returns
        -------
            torch.Tensor: The displacement vector.
        """
        my_boundary = self.get_boundary_in_direction(direction)
        other_boundary = mob.get_boundary_in_direction(direction)
        return other_boundary - my_boundary

    def move_inline_with_boundary(self, mob: 'Mob', direction: torch.Tensor) -> 'Mob':
        """
        Moves this Mob so its boundary aligns with another Mob's boundary
        along a specific direction.

        Parameters
        ----------
            mob (Mob): The target Mob whose boundary will be aligned with.
            direction (torch.Tensor): The direction along which to align the boundaries.

        Returns
        -------
            Mob: The Mob instance itself, allowing for method chaining.
        """
        return self.move(self.get_displacement_to_boundary(mob, direction))

    def move_to_edge(self, edge: torch.Tensor, buffer: float = DEFAULT_BUFFER) -> 'Mob':
        """Moves the Mob to an edge of the screen.

        Parameters
        ----------
            edge (torch.Tensor): A 3D vector indicating the screen edge direction
                (e.g., `RIGHT`, `LEFT`, `UP`, `DOWN`).
            buffer (float, optional): Distance to maintain from the screen border
                after moving. Defaults to `DEFAULT_BUFFER`.

        Returns
        -------
            Mob: The Mob instance itself, allowing for method chaining.
        """
        normalized_edge = F.normalize(edge, p=2, dim=-1)
        # Get the boundary point of this Mob that is furthest towards the 'edge' direction
        mob_boundary_point = self.get_boundary_in_direction(normalized_edge)
        # Project this point onto the screen border to find the target point on the border
        edge_point_on_screen = self.scene.camera.project_point_onto_screen_border(mob_boundary_point, normalized_edge)
        # Calculate the final target location for the Mob, accounting for the buffer
        target_location = edge_point_on_screen + F.normalize(mob_boundary_point - edge_point_on_screen, p=2,
                                                             dim=-1) * buffer
        # Calculate the displacement needed and move the Mob
        displacement = target_location - mob_boundary_point
        self.move(displacement)
        return self

    def move_to_corner(self, edge1: torch.Tensor, edge2: torch.Tensor, buffer: float = DEFAULT_BUFFER) -> 'Mob':
        """Moves the Mob to a corner of the screen, defined by two intersecting edge directions.

        Parameters
        ----------
            edge1 (torch.Tensor): Vector for the first screen edge.
            edge2 (torch.Tensor): Vector for the second screen edge.
            buffer (float, optional): Distance to maintain from both screen borders.
                Defaults to `DEFAULT_BUFFER`.

        Returns
        -------
            Mob: The Mob instance itself, allowing for method chaining.
        """
        # Chain two calls to move_to_edge to reach the corner
        return self.move_to_edge(edge1, buffer=buffer).move_to_edge(edge2, buffer=buffer)

    def move_out_of_screen(self, edge: torch.Tensor, buffer: float = DEFAULT_BUFFER, despawn: bool = True) -> 'Mob':
        """Animates the Mob moving off-screen in a given edge direction and then optionally despawns it.

        Parameters
        ----------
            edge (torch.Tensor): Vector indicating the direction to move off-screen.
            buffer (float, optional): Additional distance beyond the screen edge
                to move the Mob. Defaults to `DEFAULT_BUFFER`.
            despawn (bool, optional): If True, the Mob is despawned immediately
                after moving off-screen.

        Returns
        -------
            Mob: The Mob instance itself, allowing for method chaining.

        """
        bbox = self.get_bounding_box()

        points_on_screen_edge = self.scene.camera.project_point_onto_screen_border(bbox, edge)

        disps = points_on_screen_edge - bbox
        largest_disp = broadcast_gather(disps, -2, disps.norm(p=2, dim=-1, keepdim=True).argmax(-2, keepdim=True), keepdim=True)

        with Seq():  # Ensure movement and despawn happen sequentially
            self.move(largest_disp + buffer * F.normalize(edge, p=2,dim=-1))
            if despawn:
                self.despawn(animate=False)
        return self

    def move_to_point_along_square(self, destination: torch.Tensor, displacement: torch.Tensor) -> 'Mob':
        """Moves the Mob to a destination in a two-step "square" path.
        First, it moves by the `displacement` vector. Then, it moves orthogonally
        to align with the `destination` point, and finally reaches the `destination`.
        This creates an L-shaped or Z-shaped path.

        Parameters
        ----------
            destination (torch.Tensor): The final target 3D location.
            displacement (torch.Tensor): The initial 3D displacement vector for the
                first segment of the path.

        Returns
        -------
            Mob: The Mob instance itself, allowing for method chaining.

        """
        # Vector from current location to destination
        destination_displacement = destination - self.location
        # Normalize the initial displacement direction
        normalized_displacement_direction = F.normalize(displacement, p=2, dim=-1)
        # Calculate the orthogonal component of the destination displacement relative to the initial displacement
        orthogonal_displacement = destination_displacement - dot_product(destination_displacement,
                                                                         normalized_displacement_direction) * normalized_displacement_direction

        with Seq(run_time=1):  # Perform steps sequentially over the same runtime
            self.move(displacement)  # First step: move by initial displacement
            self.move(orthogonal_displacement)  # Second step: move orthogonally
            self.location = destination  # Final step: snap to destination (or animate if runtime allows)
        return self

    def get_length_along_direction(self, direction: torch.Tensor) -> torch.Tensor:
        """Calculates the total length of the Mob (and its children) when projected
        onto a given direction.

        Parameters
        ----------
            direction (torch.Tensor): The 3D direction vector.

        Returns
        -------
            torch.Tensor: The length of the Mob along the specified direction.

        """
        # Collect all boundary points from self and children
        all_boundary_points = torch.cat([c.get_boundary_points() for c in self.children], -2)
        # Translate points relative to the Mob's location for projection
        all_boundary_points -= self.location.unsqueeze(-2)
        # Project points onto the direction vector
        projections = dot_product(all_boundary_points, direction.unsqueeze(-2))
        # The length is the difference between the max and min projections
        return projections.amax(-2) - projections.amin(-2)

    def get_parts_as_mobs(self) -> list['Mob']:
        """
        Recursively flattens the Mob and its children into a list of individual Mobs.

        Returns
        -------
            list[Mob]: A list containing this Mob and all its descendant Mobs.
        """
        parts = [self]
        for child in self.children:
            parts.extend(child.get_parts_as_mobs())
        return parts

    def scale(self, scale_factor: float | torch.Tensor, recursive: bool = True) -> 'Mob':
        """Scales the Mob by a factor `scale_factor` relative to its current scale.

        Parameters
        ----------
            scale_factor (float or torch.Tensor): The scaling factor. Can be a scalar or
                a tensor for per-axis/per-batch scaling. For example, `2` for double size,
                `0.5` for half size.
            recursive (bool, optional): If True, applies scaling recursively
                to all child Mobs. Defaults to True.

        Returns
        =======
            Mob: The Mob instance itself, allowing for method chaining.
        """
        # Calculate the new absolute scale coefficient
        new_scale = scale_factor * self.scale_coefficient
        # Use the 'set' method to apply the new scale coefficient, which handles animation and recursion
        return self.set(scale_coefficient=new_scale) if recursive else self.set_non_recursive(
            scale_coefficient=new_scale)

    def set_scale(self, scale: float | torch.Tensor, recursive: bool = True) -> 'Mob':
        """Sets the absolute scale of the Mob to a specific value.

        Parameters
        ----------
            scale (float or torch.Tensor): The target absolute scaling factor.
            recursive (bool, optional): If True, applies scaling recursively
                to all child Mobs. Defaults to True.

        Returns
        -------
            Mob: The Mob instance itself, allowing for method chaining.

        """
        return self.set(scale_coefficient=scale) if recursive else self.set_non_recursive(scale_coefficient=scale)

    @animated_function(animated_args={'num_degrees': 0}, unique_args=['axis'])
    def rotate(self, num_degrees: float | torch.Tensor, axis: torch.Tensor = OUT) -> 'Mob':
        """Rotates the Mob by a number of degrees around a given axis passing through the mob's center.

        Parameters
        ----------
            num_degrees (float or torch.Tensor): The angle of rotation in degrees.
                Can be a scalar or a tensor for batched rotations.
            axis (torch.Tensor, optional): The 3D axis of rotation (e.g., `OUT` for Z-axis, `UP` for Y-axis).
                This vector does not need to be normalized. Defaults to `OUT`.

        Returns
        -------
            Mob: The Mob instance itself, allowing for method chaining.

        """
        normalized_axis = F.normalize(axis, p=2, dim=-1)
        # Get the rotation matrix for the specified degrees and axis
        rotation_matrix = get_rotation_around_axis(num_degrees, normalized_axis, dim=-1)
        # Apply the rotation to the Mob's basis matrix
        self.basis = squish(unsquish(self.basis, -1, 3) @ rotation_matrix, -2, -1)
        return self

    @animated_function(animated_args={'num_degrees': 0}, unique_args=['axis'])
    def rotate_and_scale(self, num_degrees: float | torch.Tensor, axis: torch.Tensor, scale: float | torch.Tensor,
                         interpolation: float = 1) -> 'Mob':
        """Performs both rotation and scaling simultaneously.

        Parameters
        ----------
            num_degrees (float or torch.Tensor): The total angle of rotation in degrees.
            axis (torch.Tensor): The 3D axis of rotation.
            scale (float or torch.Tensor): The target absolute scale factor.
            interpolation (float, optional): The interpolation factor for the animation.
                Defaults to 1.

        Returns
        -------
            Mob: The Mob instance itself, allowing for method chaining.
        """
        # Apply interpolated rotation
        interpolated_degrees = num_degrees * interpolation
        self.rotate(interpolated_degrees, axis)

        # Apply interpolated scale
        target_scale = cast_to_tensor(scale)
        interpolated_scale = self.scale_coefficient * (
                    1 - interpolation) + interpolation * target_scale * self.scale_coefficient
        self.set_scale(interpolated_scale)
        return self

    @animated_function(animated_args={'num_degrees': 0}, unique_args=['axis'])
    def rotate_around_point(self, point: torch.Tensor, num_degrees: float | torch.Tensor,
                            axis: torch.Tensor = OUT) -> 'Mob':
        """Rotates the Mob around an arbitrary point in space.

        Parameters
        ----------
            point (torch.Tensor): The 3D point to rotate around.
            num_degrees (float or torch.Tensor): The angle of rotation in degrees.
            axis (torch.Tensor, optional): The 3D axis of rotation (passing through `point`).
                This vector does not need to be normalized. Defaults to `OUT`.

        Returns
        -------
            Mob: The Mob instance itself, allowing for method chaining.
        """
        # Calculate displacement from the rotation point to the Mob's current location
        displacement_from_point = self.location - point
        # Rotate this displacement vector
        rotated_displacement = rotate_vector_around_axis(displacement_from_point, num_degrees, axis, dim=-1)
        # Calculate the new location by adding the rotated displacement back to the point
        new_location = rotated_displacement + point
        self.location = new_location  # This setter handles recursive rotation and updates
        return self

    @animated_function(animated_args={'num_degrees': 0}, unique_args=['axis'])
    def rotate_around_point_non_recursive(self, point: torch.Tensor, num_degrees: float | torch.Tensor,
                                          axis: torch.Tensor = OUT) -> 'Mob':
        """Rotates the Mob around an arbitrary point in space without affecting its children.

        Parameters
        ----------
            point (torch.Tensor): The 3D point to rotate around.
            num_degrees (float or torch.Tensor): The angle of rotation in degrees.
            axis (torch.Tensor, optional): The 3D axis of rotation (passing through `point`).
                Defaults to `OUT`.

        Returns
        -------
            Mob: The Mob instance itself, allowing for method chaining.

        """
        displacement_from_point = self.location - point
        rotated_displacement = rotate_vector_around_axis(displacement_from_point, num_degrees, axis, dim=-1)
        new_location = rotated_displacement + point
        # Use setattr_non_recursive to ensure only this Mob's location is changed
        self.setattr_non_recursive('location', new_location)
        return self

    def move_to_point_along_arc(self, point: torch.Tensor, arc_angle_degrees: float | torch.Tensor,
                                arc_normal: torch.Tensor = OUT, recursive: bool = True) -> 'Mob':
        # TODO: This is bugged and needs to be fixed. The mathematical implementation for arc center calculation might be unstable or incorrect for all cases.
        """Moves the Mob to a target point along a circular arc. ***Currently bugged***

        Parameters
        ----------
            point (torch.Tensor): The target 3D location.
            arc_angle_degrees (float or torch.Tensor): The angle subtended by the arc, in degrees.
                The sign determines the direction of rotation along the arc (clockwise/counter-clockwise).
            arc_normal (torch.Tensor, optional): The normal vector to the plane
                of the arc. Defaults to `OUT` (positive Z-axis).
            recursive (bool, optional): If True, applies the rotation recursively
                to children, maintaining their relative positions. Defaults to True.

        Returns
        -------
            Mob: The Mob instance itself, allowing for method chaining.
        """
        my_location = self.location
        displacement_unnormalized = point - my_location
        # Normalize the displacement for consistent direction calculations
        displacement_normalized = F.normalize(displacement_unnormalized, p=2, dim=-1)

        # Calculate a vector orthogonal to both displacement and arc_normal, which will define one axis for arc plane
        displacement_normal_orthogonal = F.normalize(broadcast_cross_product(displacement_normalized, arc_normal), p=2,
                                                     dim=-1)

        angle_sign = cast_to_tensor(arc_angle_degrees).sign()
        abs_arc_angle_degrees = abs(arc_angle_degrees) if not isinstance(arc_angle_degrees,
                                                                         torch.Tensor) else arc_angle_degrees.abs()

        # Calculate two vectors `in1` and `in2` that define the tangents or radii for arc center calculation.
        # These are rotated versions of the normalized displacement, used to form a geometric intersection.
        in1 = F.normalize(
            rotate_vector_around_axis(displacement_normalized, abs_arc_angle_degrees - 90, arc_normal, -1), p=2, dim=-1)
        in2 = F.normalize(
            rotate_vector_around_axis(displacement_normalized, -(abs_arc_angle_degrees + 90), arc_normal, -1), p=2,
            dim=-1)

        # Calculate the angle of the full circumference based on the dot product of in1 and in2
        arc_circumference_angle = dot_product(-in1, -in2).clamp_(min=-1, max=1).arccos_()

        # Handle edge cases where angle is exactly 180 degrees or displacement is zero,
        # which can lead to division by zero or ambiguous arc centers.
        # In such cases, a simple midpoint is used as the arc center.
        zero_displacement_mask = (((math.pi - arc_circumference_angle).abs() <= 1e-5) | (
                    displacement_unnormalized.norm(p=2, dim=-1, keepdim=True) <= 1e-5)).float()

        # Calculate arc center candidates using geometric intersection formulas.
        # These involve solving linear equations based on the dot products of vectors.
        arc_center1 = (my_location + point) * 0.5  # Midpoint for 180-degree or zero-displacement cases

        x1, y1 = 0.0, 0.0
        x2, y2 = dot_product(in1, displacement_normal_orthogonal), dot_product(in1, displacement_normalized)
        x3, y3 = dot_product(displacement_normalized, displacement_normal_orthogonal), dot_product(
            displacement_normalized, displacement_normalized)
        x4, y4 = dot_product(in2, displacement_normal_orthogonal), dot_product(in2, displacement_normalized)

        # Solving for intersection point in a 2D plane defined by displacement_normal_orthogonal and displacement_normalized
        # These are standard formulas for line-line intersection, adapted for vector components.
        intersect_x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / (
                    (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
        intersect_y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / (
                    (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))

        # Reconstruct the arc center from the intersection point and the initial location
        arc_center2 = my_location + intersect_x * displacement_normal_orthogonal + intersect_y * displacement_normalized
        arc_center2 = arc_center2.nan_to_num_(0, 0, 0)  # Handle potential NaNs from division by zero

        # Select the appropriate arc center based on the edge case mask
        final_arc_center = arc_center1 * (zero_displacement_mask) + (1 - zero_displacement_mask) * arc_center2

        # Perform the rotation around the calculated arc center
        if recursive:
            return self.rotate_around_point(final_arc_center, arc_circumference_angle * RADIANS_TO_DEGREES * angle_sign,
                                            arc_normal)
        else:
            return self.rotate_around_point_non_recursive(final_arc_center,
                                                          arc_circumference_angle * RADIANS_TO_DEGREES * angle_sign,
                                                          arc_normal)

    def refresh_history(self):
        """Resets the modification history and spawn time for this Mob and all its descendants.
        This effectively clears all animation data and makes them behave as if newly created.
        """
        for mob in self.get_descendants():
            mob.data.history = ModificationHistory()
            mob.data.spawn_time = lambda: -1

    def detach_history(self):
        """"Detaches" the Mob's current state into a new, independent animation history.
        This is useful when you want to take a snapshot of a Mob's state and start
        animating it independently, without its previous history affecting new animations.
        The original history is effectively "frozen" and the new Mob begins a fresh history.
        #TODO explain why we use this (changing tensor dimensions isn't inerpolable).
        """
        with Off():
            with NoExtra(priority_level=1):
                if self.data.spawn_time() >= 0:
                    # If the mob has a history, clone its current state
                    old_self = self.clone(reset_history=False)
                    # Advance its time slightly to ensure it's "past" the current point
                    old_self.wait((1 / self.scene.frames_per_second) + 1e-5)
                    old_self.despawn(animate=False)  # Despawn the old clone without animation
                self.refresh_history()  # Reset current mob's history
                self.spawn(animate=False)  # Re-spawn current mob without animation to start new history
                return self

    def expand_n_children(self, n: int):
        """Expands the number of children by cloning existing ones to reach `n` additional children.
        This is used internally by `become` for smooth transformations between Mobs with different
        numbers of sub-parts.

        Args:
            n (int): The number of additional children to add.
        """
        current_children_count = len(self.children)
        target_children_count = current_children_count + n
        # Determine how many times each existing child needs to be repeated/cloned
        repeat_indices = (torch.arange(target_children_count) * current_children_count) // target_children_count
        split_factors = [(repeat_indices == i).sum() for i in range(current_children_count)]

        new_submobs = []
        for submob, factor in zip(self.children, split_factors):
            new_submobs.append(submob)  # Add the original child
            for _ in range(1, factor):
                new_submobs.append(submob.clone())  # Add clones
        self.children = new_submobs
        return self

    def expand_n_batch(self, n: int):
        """Expands the batch size of the Mob's attributes by cloning existing batch elements.
        This is used internally by `become` to match the batch dimensions when transforming
        between Mobs with different numbers of primitive points.

        Args:
            n (int): The number of additional batch elements to add.
        """
        # Current number of logical objects in the batch (points / points_per_object)
        current_batch_size = self.location.shape[-2] // self.num_points_per_object
        target_batch_size = current_batch_size + n
        # Determine how many times each existing batch element needs to be repeated
        repeat_indices = (torch.arange(target_batch_size) * current_batch_size) // target_batch_size
        split_factors = [(repeat_indices == i).sum() for i in range(current_batch_size)]

        # Iterate over animatable attributes and expand their batch dimensions
        for attr in ['location', 'opacity', 'color', 'basis', 'glow']:
            value = self.__getattribute__(attr)[0]  # Get the current value (first time step)
            if value.shape[-2] == 1:  # If already a singleton batch, no expansion needed
                continue

            # Unsquish to separate individual objects in the batch if needed
            value_per_object = unsquish(value, -2, self.num_points_per_object)
            new_batched_values = []
            for sub_object_data, factor in zip(value_per_object, split_factors):
                new_batched_values.append(sub_object_data)  # Add original sub-object data
                for _ in range(1, factor):
                    # Clone the last point of the sub-object data to expand
                    new_batched_values.append(
                        sub_object_data[..., -1:, :].expand(*([-1 for _ in range(sub_object_data.dim() - 2)]),
                                                            self.num_points_per_object, -1))
            # Stack the new batched values and squish back to original shape for storage
            self.data.data_dict[attr] = squish(torch.stack(new_batched_values, -3), -3, -2).unsqueeze(0)
        return self

    def become(self, other_mob: 'Mob', move_to: bool = False, detach_history: bool = True) -> 'Mob':
        """Transforms this Mob into another Mob (`other_mob`).

        This involves animating changes in location, opacity, color, basis, etc.,
        to match `other_mob`. It intelligently attempts to match parts of this Mob
        to parts of `other_mob` for a smoother transition, especially for complex Mobs
        with multiple children or batched primitive points.

        Parameters
        ----------
            other_mob (Mob): The Mob to transform into. The type of this Mob must be
                compatible with the current Mob (e.g., both should be `Mob` or derived
                from it, and have the same `num_points_per_object` if applicable).
            move_to (bool, optional): If True, the transformation explicitly includes
                a movement animation to `other_mob`'s final location. Note that
                attribute alignment typically includes location changes anyway.
                Defaults to False.
            detach_history (bool, optional): If True, the original Mob's animation
                history is "detached" and this Mob starts a fresh animation history
                from its transformed state. If False, the transformation is recorded
                within the existing history. Defaults to True.

        Returns
        -------
            Mob: The (transformed) Mob instance itself, or the `other_mob_original`
            if `detach_history` is True (as it will be the "new" main mob).

        Raises
        ------
            NotImplementedError: If attempting to transform between mobs with
                different underlying primitive types (e.g., changing a triangle-based
                mob to a bezier-circuit-based mob).
        """
        # Temporarily turn off animation recording for setup steps
        with Off():
            if detach_history:
                self.detach_history()  # Detach current mob's history
                other_mob_original = other_mob  # Keep a reference to the original target mob
                other_mob = other_mob.clone(add_to_scene=False)  # Clone target to avoid modifying it directly

            # Adjust child counts to match for smooth transitions
            child_difference = len(other_mob.children) - len(self.children)
            if child_difference > 0:
                self.expand_n_children(child_difference)
            elif child_difference < 0:
                other_mob.expand_n_children(-child_difference)

        with Seq():
            with Sync():
                # Recursively apply 'become' to children to handle nested transformations
                for my_child, other_child in zip(self.children, other_mob.children):
                    my_child.become(other_child, detach_history=False)  # Children do not detach their history

                # Check for compatibility of primitive types
                if other_mob.num_points_per_object != self.num_points_per_object:
                    raise NotImplementedError(
                        "You are trying to change an object of one primitive type (e.g., triangle) "
                        "to another type (e.g., cubic bezier circuit). This is not supported. "
                        "When using become(), the target mob must be of the same primitive type as the original."
                    )

                # Adjust batch size (number of points per object) for smooth transitions
                batch_difference = (other_mob.location.shape[-2] - self.location.shape[
                    -2]) // self.num_points_per_object
                if batch_difference > 0:
                    self.expand_n_batch(batch_difference)
                elif batch_difference < 0:
                    other_mob.expand_n_batch(-batch_difference)

                # Set all animatable attributes non-recursively to match the target mob's values
                for attr_name in ['location', 'opacity', 'color', 'basis', 'glow']:
                    # Use getattr to safely access attributes, as not all mobs may have all listed attributes
                    self.setattr_non_recursive(attr_name, getattr(other_mob, attr_name))

            if detach_history:
                # If history was detached, despawn the transforming mob and spawn the original target mob
                with Off():
                    self.despawn(animate=False)
                    return other_mob_original.spawn(animate=False)
            return self

    def _become_recursive(self, other_mob: 'Mob', move_to: bool = False):
        """
        Internal recursive helper for the `become` method.
        Handles the transformation logic for children mobs.
        """
        other_children = list(other_mob.children)
        my_children = list(self.children)

        with Sync():
            if len(other_children) > 0:
                if len(my_children) < len(other_children):
                    with Off():  # Avoid recording intermediate clones
                        # Clone additional children from the target mob
                        new_children = [c.clone(animate_creation=False, recursive=True) for c in
                                        other_children[len(my_children):]]
                        my_children.extend(new_children)
                        self.add_children(new_children)
                        # Initialize new children's local coordinates to origin
                        [c.set_recursive(local_coords=ORIGIN) for c in new_children]
                elif len(other_children) < len(my_children):
                    # Trim excess children if current mob has more than target
                    my_children = my_children[:len(other_children)]

                # Recursively call _become_recursive for matched children
                for i in reversed(range(len(my_children))):  # Iterate in reverse for stable child list modification
                    my_children[i]._become_recursive(other_children[i], move_to=True)

            if not move_to:  # Only apply location change if explicitly requested
                return self

            # Handle location transformation for the current mob (parent in this recursive context)
            other_location = other_mob.location
            my_location = self.location
            my_batch_size = my_location.shape[-2]
            other_batch_size = other_location.shape[-2]

            if other_batch_size > my_batch_size:
                with Off(record_funcs=False, record_attr_modifications=False):
                    # Expand current location to match target batch size if smaller
                    expanded_location = torch.cat(
                        [my_location, my_location[..., -1:, :].expand(-1, other_batch_size - my_batch_size, -1)], -2)
                    self.setattr_regular('_location', expanded_location)  # Direct set to avoid recursion issues here
                    self.batch_size = max(self.batch_size, self.location.shape[-2])
                    self.parent_batch_sizes = other_mob.parent_batch_sizes
            elif other_batch_size < my_batch_size:
                # Pad target location with zeros if it's smaller
                other_location = torch.cat((other_location, torch.zeros(
                    (other_location.shape[0], my_batch_size - other_batch_size, other_location.shape[2]))), -2)
            self.location = other_location  # Set location, triggering animation if needed
        return self

    def set_non_recursive(self, **kwargs) -> 'Mob':
        """Sets multiple attributes non-recursively (i.e., only for this Mob, not its children).
        This is useful for applying changes that should not propagate down the hierarchy.

        Parameters
        ----------
            **kwargs: Keyword arguments where keys are attribute names (e.g., 'color', 'opacity')
                and values are the new values for those attributes.

        Returns
        -------
            Mob: The Mob instance itself, allowing for method chaining.
        """
        with Sync():  # Ensure all these attribute sets happen in one synchronized animation step
            for key, value in kwargs.items():
                self.setattr_non_recursive(key, value)
        return self

    def set(self, **kwargs) -> 'Mob':
        """Sets multiple attributes, applying changes recursively to children by default.
        This is the primary method for changing a Mob's properties and
        triggering animations for those changes.

        Parameters
        ----------
        **kwargs
            Keyword arguments where keys are attribute names (e.g., 'location', 'color')
            and values are the new values for those attributes. These changes will
            be animated and propagated to children unless `recursing` is set to False
            (via `setattr_non_recursive` or `_set_recursing_state`).

        Returns
        -------
            Mob: The Mob instance itself, allowing for method chaining.

        Examples
        ---------
        Move a square to the right and change its color to blue:

        .. algan:: Example1MobSet

            from algan import *

            mob = Square().spawn()
            mob.set(location=RIGHT, color=BLUE)

            render_to_file()
        """
        with Sync():
            for key, value in kwargs.items():
                self.__setattr__(key, value)  # Calls the property setters, which handle animation and recursion
        return self

    def set_recursive_from_parent(self, parent: 'Mob', **kwargs):
        """Sets attributes recursively for this Mob based on a parent Mob's attributes,
        handling batching and time synchronization. This is used internally for
        propagating changes down a hierarchy.

        Args:
            parent (Mob): The parent Mob from which attribute values might be derived or synced.
            **kwargs: Attributes to set, typically passed from the parent.

        """
        if self.parent_batch_sizes is not None:
            # Helper to cast values to tensor and expand if necessary for child's batching
            def cast_and_expand(value_to_cast: any) -> torch.Tensor:
                if isinstance(value_to_cast, torch.Tensor):
                    return value_to_cast
                # Convert scalar to tensor and expand for batch dimensions
                return torch.tensor((value_to_cast,)).view(1, 1, 1).expand(len(self.parent_batch_sizes), -1, -1)

            # Synchronize time indices with parent if parent has materialized times
            if self.data.time_inds_materialized is None and parent.data.time_inds_materialized is not None:
                self.set_state_full(parent.data.time_inds_materialized.amin(),
                                    parent.data.time_inds_materialized.amax() + 1)
            self.data.time_inds_active = parent.data.time_inds_active

            # Expand kwargs values to match child's batch dimensions
            kwargs = {key: torch.repeat_interleave(cast_and_expand(value), self.parent_batch_sizes, 0) for key, value in
                      kwargs.items()}

        with AnimationContext(dont_record_funcs=True):  # Do not record function applications during this recursive set
            return self.set_recursive(**kwargs)  # Recursively set attributes

    def set_recursive(self, **kwargs) -> 'Mob':
        """Sets multiple attributes for this Mob and then recursively propagates
        the changes to all its children.

        Parameters
        ----------
            **kwargs: Keyword arguments for attributes to set.

        Returns
        -------
            Mob: The Mob instance itself, allowing for method chaining.

        """
        self.set(**kwargs)  # Set attributes for the current mob
        for child in self.children:
            child.set_recursive_from_parent(self, **kwargs)  # Propagate to children
        return self

    def get_forward_direction(self) -> torch.Tensor:
        """Gets the Mob's current forward direction vector (normalized).
        This corresponds to the third column of its normalized basis matrix.

        Returns
        -------
            torch.Tensor: A 3D vector representing the forward direction.

        """
        return F.normalize(unsquish(self.basis, -1, 3)[..., 2, :], p=2, dim=-1)

    def get_right_direction(self) -> torch.Tensor:
        """Gets the Mob's current right direction vector (normalized).
        This corresponds to the first column of its normalized basis matrix.

        Returns
        -------
            torch.Tensor: A 3D vector representing the right direction.

        """
        return F.normalize(unsquish(self.basis, -1, 3)[..., 0, :], p=2, dim=-1)

    def get_upwards_direction(self) -> torch.Tensor:
        """Gets the Mob's current upwards direction vector (normalized).
        This corresponds to the second column of its normalized basis matrix.

        Returns
        -------
            torch.Tensor: A 3D vector representing the upwards direction.

        """
        return F.normalize(unsquish(self.basis, -1, 3)[..., 1, :], p=2, dim=-1)

    def look(self, direction: torch.Tensor, axis: int = 2) -> 'Mob':
        """Rotates the Mob so that one of its local axes points in the given direction.

        Parameters
        ----------
            direction (torch.Tensor): The target 3D direction vector that the specified
                local axis should point towards. This vector does not need to be normalized.
            axis (int, optional): The index of the local axis to align.
                0 for right (X-axis), 1 for up (Y-axis), 2 for forward (Z-axis).
                Defaults to 2 (forward vector).

        Returns
        -------
            Mob: The Mob instance itself, allowing for method chaining.

        """
        # Get the rotation parameters (angle and axis) needed to align the current local axis
        # with the target direction.
        rotation_angle_degrees, rotation_axis = get_rotation_between_3d_vectors(
            unsquish(self.normalized_basis, -1, 3)[..., axis, :],  # Current orientation of specified axis
            F.normalize(direction, p=2, dim=-1),  # Normalized target direction
            dim=-1
        )
        # Apply the rotation
        return self.rotate(rotation_angle_degrees, rotation_axis)

    def look_and_scale(self, direction: torch.Tensor, scale: float | torch.Tensor, axis: int = 2) -> 'Mob':
        """Rotates the Mob to look in a specific direction and simultaneously scales it.

        Parameters
        ----------
            direction (torch.Tensor): The target 3D direction vector to look at.
            scale (float or torch.Tensor): The target absolute scale factor.
            axis (int, optional): The index of the local axis to align (0: right, 1: up, 2: forward).
                Defaults to 2 (forward).

        Returns
        -------
            Mob: The Mob instance itself, allowing for method chaining.

        """
        # Get rotation parameters from the 'look' logic
        rotation_angle_degrees, rotation_axis = get_rotation_between_3d_vectors(
            unsquish(self.normalized_basis, -1, 3)[..., axis, :],
            F.normalize(direction, p=2, dim=-1),
            dim=-1
        )
        # Apply both rotation and scale using the combined animated function
        return self.rotate_and_scale(rotation_angle_degrees, rotation_axis, scale)

    def look_at(self, point: torch.Tensor, axis: int = 2) -> 'Mob':
        """Rotates the Mob to face a specific 3D point.
        The Mob's "forward" direction (or the specified `axis`) will be oriented towards the point.

        Parameters
        ----------
            point (torch.Tensor): The 3D point to look at.
            axis (int, optional): The index of the local axis to align (0: right, 1: up, 2: forward).
                Defaults to 2 (forward vector).

        Returns
        -------
            Mob: The Mob instance itself, allowing for method chaining.

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
        tiles = [mob.tiles for mob in traverse(self.get_descendants()) if
                 hasattr(mob, 'tiles') and not mob.tiles.time_inds.created]
        with AnimationContext(run_time=3):
            # Animate each tile/primitive appearing from a random direction
            animate_lagged_by_location(tiles, lambda m: m.spawn_from_random_direction(),
                                       F.normalize(RIGHT * 1.5 + DOWN, p=2, dim=-1))
        return self

    def despawn_tilewise_recursive(self):
        """
        Animates the despawning of the Mob and its primitive children in a lagged, "tile-wise" manner.
        Each tile/primitive disappears into a random direction with an ease-out-exponential effect.
        """
        # Collect all primitive tiles from the Mob and its descendants
        tiles = [mob.tiles for mob in traverse(self.get_descendants()) if hasattr(mob, 'tiles')]
        with AnimationContext(run_time=3):
            # Animate each tile/primitive disappearing into a random direction
            animate_lagged_by_location(tiles, lambda m: m.despawn_from_random_direction(),
                                       F.normalize(RIGHT * 1.5 + DOWN, p=2, dim=-1))
        return self

    def spawn_from_random_direction(self, travel_distance: float = 0.1):
        """
        Animates the Mob appearing from a random direction, fading in and optionally rotating.
        This sets the initial opacity to 0 and then animates it to 1.
        """
        with Off():  # Ensure initial state setting is not recorded as an animation
            self.opacity = 0
        self._create_recursive(animate=False)  # Mark as created without immediate animation
        with Sync(run_time=None, rate_func=ease_out_exp):  # Synchronized animation with ease-out
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
        return self.location.shape[-2] if hasattr(self, 'location') else 1

    def despawn_from_random_direction(self, travel_distance: float = 0.1):
        """Animates the Mob disappearing into a random direction, fading out and optionally rotating.
        This animates the opacity to 0 and then marks the Mob as destroyed.

        """
        with Sync(run_time=None, rate_func=inversed(ease_out_exp)):  # Synchronized animation with inversed ease-out
            current_location = self.location
            # Example of potential animated properties (currently commented out)
            # self.location = current_location + torch.randn_like(current_location) * travel_distance
            # self.rotate(720, F.normalize(torch.randn_like(self.location), p=2, dim=-1))
            self.opacity = 0  # Animate opacity to zero
            self._destroy_recursive(animate=False)  # Mark as destroyed without immediate animation
            # with Synchronized(run_time=2, rate_func=tan):
            # self.destroy()
        return self

    def set_data_sub_inds(self, data_sub_inds: list[int] | slice):
        """Sets the sub-indices that this Mob will use when reading and writing from
        the shared data dictionaries (`data.data_dict_active`, `data.data_dict_materialized`).
        This is used for implementing indexing of batched mobs to retrieve sub-mobs that share
        the same underlying data.

        Parameters:
        -----------
            data_sub_inds (list[int] or slice): The indices or slice to apply to the
                batch dimension of the shared data tensors.

        """
        self.batch_size = max(self.batch_size, self.location.shape[1])
        if self.parent_batch_sizes is not None:
            sub_pbs = self.parent_batch_sizes[data_sub_inds]
            inds = torch.arange(self.batch_size).split([_.item() for _ in self.parent_batch_sizes])
            data_sub_inds = torch.cat(
                [inds[d] for d in data_sub_inds] if not isinstance(data_sub_inds, slice) else inds[data_sub_inds])
        else:
            sub_pbs = self.parent_batch_sizes
        self.data_sub_inds = data_sub_inds
        self.parent_batch_sizes = sub_pbs
        for c in self.children:
            c.set_data_sub_inds(data_sub_inds)

    def __getitem__(self, item: int | slice) -> 'Mob':
        """Allows accessing a part of a batched Mob using slice notation (e.g., `my_mob[0]`, `my_mob[1:3]`).

        Returns a new Mob instance that represents the specified sub-part(s).
        This new Mob shares the underlying animation data with the original,
        but its `data_sub_inds` are set appropriately to only operate on the
        selected batch elements. This is efficient as it avoids data duplication.

        Parameters
        ----------
            item (int or slice): The index or slice for selecting elements from the
                batch dimension.

        Returns
        -------
            Mob: A new Mob instance representing the selected sub-part(s) of the original Mob.
        """
        # Clone the mob without cloning its data, but recursively for children structure
        cloned_mob = self.clone(add_to_scene=False, clone_data=False, recursive=True, animate_creation=False)
        # Set the data sub-indices for the cloned mob to point to the desired batch elements
        cloned_mob.set_data_sub_inds([item] if isinstance(item, int) else item)
        return cloned_mob