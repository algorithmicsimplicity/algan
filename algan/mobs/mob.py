import math
from collections import defaultdict

from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F

from algan.animation.animatable import Animatable, animated_function, ModificationHistory
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
    """Base class for all objects that have a location and orientation in 3d space.

    A Mob is an Animatable that exists in a 3D scene,
    possessing properties like location, orientation (basis), and color.
    It can have child Mobs, forming a hierarchy, and supports various
    transformations and animations.

    Parameters
    ----------
    location: Tensor[*,3]
        Location in 3D world space.
    basis: Tensor[*,9]
        Flattened 3x3 matrix in which the rows specify the
        right, upwards, and forwards directions for this mob's orientation, and the row norms
        represent the mob's scale in those directions. Defaults to identify matrix.
    color
        The color of the Mob. If None, uses `get_default_color()`. Defaults to None.
    opacity
        The maximum opacity of the Mob (0.0 to 1.0). Defaults to 1.
    glow
        The glow intensity of the Mob. Defaults to 0.
    *args
        Additional arguments for the Animatable base class.
    **kwargs
        Additional keyword arguments for the Animatable base class.

    Examples
    --------
    .. algan:: Example1Mob

        from algan import *

        square = Square().spawn()
        square.move(LEFT)

        render_to_file()

    """
    def __init__(self, location:torch.Tensor=ORIGIN, basis:torch.Tensor=squish(torch.eye(3)), color:Color|None=None, opacity:float=1, glow:float=0, *args, **kwargs):
        self.register_attrs_as_animatable({'location', 'basis', 'scale_coefficient', 'color', 'opacity', 'max_opacity', 'glow'}, Mob)
        self.recursing = True
        super().__init__(*args, **kwargs)
        self.attr_to_relations = defaultdict(lambda: (lambda x, y: y, lambda x, y: y))
        additive = (lambda x, y: x+y, lambda x, y: y-x)
        self.attr_to_relations.update({'location': additive,
                                       #'glow': additive,
                                  'basis': (lambda x, y: squish(unsquish(x, -1, 3) @ unsquish(y, -1, 3), -2, -1),
                                            lambda x, y: squish(get_rotation_between_bases(unsquish(x, -1, 3), unsquish(y, -1, 3)), -2, -1)),
                                       'scale_coefficient': (lambda x, y: (x*y),
                                                             lambda x, y: squish((unsquish(y,-1,3).norm(p=2,dim=-1,keepdim=True)/
                                                                                  unsquish(x,-1,3).norm(p=2,dim=-1,keepdim=True)).expand(*([-1] * (x.dim())), 3),-2,-1))})
        self.location = cast_to_tensor(location)
        self.basis = cast_to_tensor(basis)
        if color is None:
            color = self.get_default_color()
        self.color = color
        self.max_opacity = cast_to_tensor(opacity)
        self.opacity = cast_to_tensor(1)
        self.glow = cast_to_tensor(glow)
        self.num_points_per_object = 1

    def reset_basis(self):
        """Resets the Mob's basis to the identity matrix (no rotation, unit scale)."""
        self.basis = cast_to_tensor(cast_to_tensor(squish(torch.eye(3))))

    def register_attrs_as_animatable(self, attrs, my_class=None):
        """
        Registers attributes as animatable, meaning their changes can be tracked
        and interpolated over time.

        This method dynamically creates property getters and setters for the
        specified attributes if they don't already exist.

        Args:
            attrs (iterable[str] or str): A collection of attribute names (or a single
                attribute name) to register as animatable.
            my_class (type, optional): The class to which the property getters
                and setters should be attached. Defaults to the current Mob's class.
        """
        if isinstance(attrs, str):
            attrs = {attrs,}
        if not isinstance(attrs, set):
            attrs = set(attrs)
        if not hasattr(self, 'animatable_attrs'):
            self.animatable_attrs = set()
        if my_class is None:
            my_class = self.__class__
        for attr in attrs:
            self.add_property_getter_and_setter(attr, my_class)
        self.animatable_attrs.update(attrs)

    def add_property_getter_and_setter(self, property_name, class_to_attach_to=None):
        """
        Dynamically adds a property with a getter and setter for a given attribute name.

        The getter will retrieve the animated value of the attribute.
        The setter will set the absolute value of the attribute, recording the change
        for animation.

        Args:
            property_name (str): The name of the property to create.
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

    def get_descendants(self, include_self=True):
        """
        Retrieves all descendant Mobs in the hierarchy, optionally including self.

        Args:
            include_self (bool, optional): Whether to include the current Mob
                in the returned list. Defaults to True.

        Returns:
            list[Mob]: A flat list of descendant Mobs.
        """
        return list(traverse([*([self] if include_self else []), [c.get_descendants() for c in self.children] if hasattr(self, 'children') else []]))

    def set_time_inds_to(self, mob):
        """
        Synchronizes the animation time indices of this Mob with another Mob.

        This is used internally to ensure consistent animation states between
        mobs in a hierarchy.

        Args:
            mob (Mob): The Mob whose time indices will be copied.
        """
        time_inds = mob.data
        if self.data.time_inds_materialized is None and time_inds.time_inds_materialized is not None:
            self.data.animatable.set_state_pre_function_applications(time_inds.time_inds_materialized.amin(), time_inds.time_inds_materialized.amax() + 1)
        self.data.time_inds_active = time_inds.time_inds_active

    @animated_function(animated_args={'interpolation': 0.0}, unique_args=['key', 'recursive'])
    def apply_absolute_change_two(self, key, change1, change2, interpolation=1.0, recursive='True'):
        """
        Applies an animated change to an attribute, interpolating between two target values.

        The interpolation first moves from the current value towards `change1`,
        and if `interpolation` goes beyond 1.0, it then moves from `change1`
        towards `change2`.

        Args:
            key (str): The name of the attribute to change.
            change1 (Any): The first target value for the attribute.
            change2 (Any): The second target value for the attribute.
            interpolation (float, optional): The interpolation factor.
            recursive (str, optional): If 'True', applies the change recursively
                to child Mobs. Defaults to 'True'.

        Returns:
            Mob: The Mob instance itself.
        """
        relation = self.attr_to_relations[key][0]
        current_value = self.__getattribute__(key)
        interpolation = cast_to_tensor(interpolation) * 2
        m = (interpolation > 1).float()
        change = (current_value * (1 - interpolation) + interpolation * change1) * (1-m) + m * (change1 * (2 - (interpolation)) + (interpolation-1) * change2)
        self.setattr_and_record_modification(key, relation(current_value, change))
        if recursive == 'True':
            for c in self.children:
                c.set_time_inds_to(self)
                change3 = change
                if c.parent_batch_sizes is not None:
                    def expand(x):
                        if x.shape[-2] == 1:
                            x = x.expand(*([-1 for _ in range(x.dim() - 2)]), len(c.parent_batch_sizes),
                                         -1).contiguous()
                        return x

                    change3 = torch.repeat_interleave(expand(change3), c.parent_batch_sizes, -2)
                c.apply_relative_change(key, change3, interpolation=1, recursive=recursive)
        return self

    def pulse_color(self, color, set_opaque=False):
        """
        Animates a color pulse effect.

        The Mob's color changes to the target `color` and then back to its
        original color.

        Args:
            color (torch.Tensor): The color to pulse to.
            set_opaque (bool, optional): If True, also animates opacity to 1
                during the pulse. Defaults to False.

        Returns:
            Mob: The Mob instance itself.
        """
        with Sync():
            self.apply_absolute_change_two('color', color, self.color)
            if set_opaque:
                self.apply_absolute_change_two('opacity', 1, 1)
        return self

    def wave_color(self, color, wave_length=1, reverse=False, direction=None, **kwargs):
        """
        Applies a color wave effect across the Mob and its descendants.

        The color change propagates spatially.

        Args:
            color (torch.Tensor): The target color for the wave.
            wave_length (float, optional): Controls the spatial extent of the wave.
                Defaults to 1.
            reverse (bool, optional): If True, the wave propagates in the
                opposite direction. Defaults to False.
            direction (torch.Tensor, optional): The 3D vector defining the
                direction of wave propagation. If None, uses the Mob's
                upwards direction. Defaults to None.
            **kwargs: Additional keyword arguments passed to `pulse_color`
                for each part of the wave.

        Returns:
            Mob: The Mob instance itself.
        """
        if direction is None:
            direction = self.get_upwards_direction()
        with AnimationContext(run_time_unit=3, rate_func=identity):
            animate_lagged_by_location([_ for _ in self.get_descendants() if _.is_primitive], lambda x: x.pulse_color(color, **kwargs), direction * (-1 if reverse else 1), 1.5)

    @animated_function(animated_args={'interpolation': 0.0}, unique_args=['key', 'recursive', 'relation_key'])
    def apply_relative_change(self, key, change, interpolation=1.0, recursive='True', relation_key="None"):
        """
        Applies an animated relative change to an attribute.

        The `change` is scaled by `interpolation` and then applied to the
        current attribute value using a predefined relation (e.g., addition
        for location, multiplication for scale).

        Args:
            key (str): The name of the attribute to change.
            change (Any): The relative change to apply.
            interpolation (float, optional): The interpolation factor for the change.
                Defaults to 1.0.
            recursive (str, optional): If 'True', applies the change recursively
                to child Mobs. Defaults to 'True'.
            relation_key (str, optional): The key to look up the relation function
                (how the change is combined with the current value). Defaults to "None",
                which means it uses `key`.

        Returns:
            Mob: The Mob instance itself.
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
    def apply_set_value(self, key, change, interpolation=1.0):
        """
        Sets an attribute's value, interpolating from its current value to the target `change`.

        This is a direct interpolation rather than a relative one.

        Args:
            key (str): The name of the attribute to set.
            change (Any): The target value for the attribute.
            interpolation (float, optional): The interpolation factor.
                0.0 means no change, 1.0 means the attribute becomes `change`.
                Defaults to 1.0.

        Returns:
            Mob: The Mob instance itself.
        """
        current_value = self.__getattribute__(key)
        try:
            change = current_value * (1 - interpolation) + interpolation * change
        except RuntimeError:
            interpolation = torch.cat((interpolation, torch.zeros_like(current_value[..., -(current_value.shape[-2] - interpolation.shape[-2]):, :])), -2)
            change = current_value * (1 - interpolation) + interpolation * change
        self.setattr_and_record_modification(key, change)

    @animated_function(animated_args={'interpolation': 0.0}, unique_args=['key', 'recursive'])
    def apply_absolute_change(self, key, change, interpolation=1.0, recursive='True'):
        """
        Applies an animated absolute change to an attribute, interpolating to a target value.

        Args:
            key (str): The name of the attribute to change.
            change (Any): The target absolute value for the attribute.
            interpolation (float, optional): The interpolation factor.
                0.0 means no change from current, 1.0 means attribute becomes `change`.
                Defaults to 1.0.
            recursive (str, optional): If 'True', applies the change recursively
                to child Mobs. Defaults to 'True'.

        Returns:
            Mob: The Mob instance itself.
        """
        orig_value = change
        if hasattr(self, key):
            relation = self.attr_to_relations[key][0]
            current_value = self.__getattribute__(key)
            try:
                change = current_value * (1-interpolation) + interpolation * change
            except RuntimeError:
                interpolation = torch.cat((interpolation, torch.zeros_like(current_value[...,-(current_value.shape[-2]-interpolation.shape[-2]):,:])), -2)
                change = current_value * (1 - interpolation) + interpolation * change
            self.setattr_and_record_modification(key, relation(current_value, change))
        if recursive == 'True':
            change = orig_value
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

    def setattr_basic(self, key, value):
        """
        Sets an attribute's value directly without complex animation logic.

        If the attribute is animatable and an animation context is active,
        this will still record the change as a step-change.

        Args:
            key (str): The name of the attribute to set.
            value (Any): The new value for the attribute.

        Returns:
            Mob: The Mob instance itself.
        """
        value = cast_to_tensor(value)
        if not hasattr(self, 'data'):
            self.__setattr__(f'_{key}', value)
            return self
        if not hasattr(self, key):
            self.data.data_dict_active[key] = value
            return self

        self.apply_set_value(key, value)
        return self

    def setattr_relative(self, key, value, relation_key=None):
        """
        Sets an attribute by applying a relative change.

        This calculates the difference (`change`) needed to get from the current
        value to the target `value` based on the inverse relation, then
        applies that `change` relatively.

        Args:
            key (str): The name of the attribute to set.
            value (Any): The target value for the attribute.
            relation_key (str, optional): The key for the relation functions.
                Defaults to `key`.

        Returns:
            Mob: The Mob instance itself.
        """
        value = cast_to_tensor(value)
        if not hasattr(self, 'data'):
            self.__setattr__(f'_{key}', value)
            return self
        if not hasattr(self, key):
            self.data.data_dict_active[key] = value
            return self

        if relation_key is None:
            relation_key = key
        rel, rel_inv = self.attr_to_relations[relation_key]
        current_value = self.__getattribute__(key)
        change = rel_inv(current_value, value)
        return self.apply_relative_change(key, change, recursive="True" if self.recursing else "False", relation_key=relation_key)

    def setattr_absolute(self, key, value):
        """
        Sets an attribute to an absolute value, animating the transition.

        Args:
            key (str): The name of the attribute to set.
            value (Any): The target absolute value.

        Returns:
            Mob: The Mob instance itself.
        """
        value = cast_to_tensor(value)
        if not hasattr(self, 'data'):
            self.__setattr__(f'_{key}', value)
            return self
        if not hasattr(self, key):
            self.data.data_dict_active[key] = value
            return self

        return self.apply_absolute_change(key, value, recursive="True" if self.recursing else "False")

    @property
    def location(self):
        return self.getattribute_animated('location')

    @location.setter
    def location(self, location):
        self.setattr_relative('location', location)

    @property
    def basis(self):
        return self.getattribute_animated('basis')

    @property
    def normalized_basis(self):
        return squish(unsquish(self.basis, -1, 3) / self.scale_coefficient.unsqueeze(-1), -2, -1)

    def set_basis_inner(self, parent_location, old_basis, new_basis):
        if self.parent_batch_sizes is not None:
            def expand(x):
                return x.expand(-1,self.parent_batch_sizes.shape[0], -1)
            parent_location, old_basis, new_basis = [torch.repeat_interleave(expand(_), self.parent_batch_sizes, -2) for _ in [parent_location, old_basis, new_basis]]
        local_coords = map_global_to_local_coords(parent_location, old_basis, self.location)
        self.setattr_and_record_modification('location', map_local_to_global_coords(parent_location, new_basis, local_coords))
        if self.recursing:
            for c in self.children:
                c.set_basis_inner(parent_location, old_basis, new_basis)
        return self

    def set_basis_interpolated(self, *args, **kwargs):
        return self._set_basis_interpolated(*args, **kwargs, recursive='True' if self.recursing else 'False')

    @animated_function(animated_args={'interpolation': 0}, unique_args=['relation_key', 'recursive'])
    def _set_basis_interpolated(self, basis, interpolation=1, relation_key='basis', recursive='True'):
        """
        Sets the Mob's basis, interpolating from the current basis to the target.

        This method also ensures that child Mobs maintain their positions
        relative to this Mob during the basis change.

        Args:
            basis (torch.Tensor): The target 3x3 basis matrix.
            interpolation (float, optional): Interpolation factor (0 to 1).
                Defaults to 1.
            relation_key (str, optional): Key for the relation function,
                typically 'basis' or 'scale_coefficient'. Defaults to 'basis'.

        Returns:
            Mob: The Mob instance itself.
        """
        if recursive == 'True':
            ds = (self.get_descendants(include_self=False))
            [d.set_time_inds_to(self) for d in ds]
        old_basis = self.basis if hasattr(self, 'basis') else basis
        basis = old_basis * (1-interpolation) + interpolation * basis
        recursing = self.recursing
        self.recursing = recursive == 'True'
        self.setattr_relative('basis', basis, relation_key)
        self.recursing = recursing
        if recursive == 'True':
            for c in self.children:
                c.set_basis_inner(self.location, old_basis, basis)
        return self

    @basis.setter
    def basis(self, basis):
        self.set_basis_interpolated(basis)

    @property
    def scale_coefficient(self):
        return unsquish(self.basis, -1, 3).norm(p=2, dim=-1, keepdim=False)

    @scale_coefficient.setter
    def scale_coefficient(self, scale_coefficient):
        scale_coefficient = cast_to_tensor(scale_coefficient)
        self.set_basis_interpolated(squish(F.normalize(unsquish(self.basis, -1, 3), p=2, dim=-1) * scale_coefficient.unsqueeze(-1), -2, -1), relation_key='scale_coefficient')
        return self

    def clear_cache(self):
        """
        Clears cached animation data.

        This is typically used internally when animation states are reset or recalculated.
        """
        if self.free_cache:
            self.attr_to_values_full = dict()
            self.attr_to_values = dict()
            self.time_stamps_full = None
            self.time_stamps = None
            self.time_inds_full = None
            self.time_inds = None

    def get_normal(self):
        """
        Alias for get_forward_direction()
        """
        return self.get_forward_direction()

    def get_center(self):
        #TODO make this get mid point of bounding box surrounding self + children.
        return self.location

    def set_location(self, location, recursive=True):
        if recursive:
            self.location = location
        else:
            self.setattr_non_recursive('location', location)
        return self

    def setattr_non_recursive(self, key, value):
        """
        Sets an attribute's value without applying the change to child Mobs.

        This temporarily disables the recursive behavior of attribute setting.

        Args:
            key (str): The name of the attribute to set.
            value (Any): The new value for the attribute.
        """
        recursing = self.recursing
        self.recursing = False
        self.__setattr__(key, value)
        self.recursing = recursing

    def move_to(self, location, path_arc_angle=None, **kwargs):
        """
        Moves the Mob to a specified location.

        If `path_arc_angle` is provided, the Mob moves along an arc.
        Otherwise, it moves in a straight line.

        Args:
            location (torch.Tensor): The target 3D location.
            path_arc_angle (float, optional): The angle of the arc in degrees
                for curved movement. If None, movement is linear. Defaults to None.
            **kwargs: Additional arguments passed to `set_location` or
                `move_to_point_along_arc`.

        Returns:
            Mob: The Mob instance itself.
        """
        if path_arc_angle is None:
            return self.set_location(location, **kwargs)
        return self.move_to_point_along_arc(location, path_arc_angle, **kwargs)

    def move(self, displacement, **kwargs):
        """
        Moves the Mob by a given displacement vector.

        Args:
            displacement (torch.Tensor): The 3D vector by which to move the Mob.
            **kwargs: Additional arguments passed to `move_to`.

        Returns:
            Mob: The Mob instance itself.
        """
        self.move_to(self.location + displacement, **kwargs)
        return self

    def get_boundary_points(self):
        return self.location

    def get_boundary_points_recursive(self):
        num_c = len(self.children)
        if num_c == 0:
            return self.get_boundary_points()
        elif num_c == 1:
            return self.children[0].get_boundary_points_recursive()
        return torch.cat([c.get_boundary_points_recursive() for c in self.children], -2)

    def get_boundary_edge_point(self, direction):
        bp = self.get_boundary_points_recursive()
        best_ind = dot_product(bp, direction, dim=-1, keepdim=True).argmax(-2, keepdim=True)
        return broadcast_gather(bp, -2, best_ind, keepdim=True)

    def get_boundary_in_direction(self, direction):
        edge_point = self.get_boundary_edge_point(direction)

        def med(x):
            mx = x.amax(-2, keepdim=True)
            mn = x.amin(-2, keepdim=True)
            return (mx + mn)*0.5
        loc = med(self.location)
        return project_point_onto_line(edge_point - loc, direction, dim=-1) + loc

    def set_x_coord(self, loc):
        new_loc = self.location.clone()
        new_loc[...,-1] = loc[...,-1]
        self.location = new_loc

    def set_y_coord(self, loc):
        new_loc = self.location.clone()
        new_loc[...,-2] = loc[...,-2]
        self.location = new_loc

    def set_x_y_coord(self, loc):
        new_loc = self.location.clone()
        new_loc[..., -2:] = loc[..., -2:]
        self.location = new_loc

    def move_next_to(self, mob, direction, buffer=DEFAULT_BUFFER, **kwargs):
        """
        Moves this Mob to be adjacent to another Mob (or a point) in a given direction.

        Args:
            mob (Mob or torch.Tensor): The target Mob or 3D point to move next to.
            direction (torch.Tensor): The 3D vector indicating the direction
                from `mob` to this Mob.
            buffer (float, optional): The distance to maintain between the Mobs.
                Defaults to DEFAULT_BUFFER.
            **kwargs: Additional arguments passed to `move_to`.

        Returns:
            Mob: The Mob instance itself.
        """
        direction = F.normalize(direction, p=2, dim=-1)
        mob_edge = mob.get_boundary_in_direction(direction) if not isinstance(mob, torch.Tensor) else mob
        my_edge = self.get_boundary_in_direction(-direction)
        self.move_to(mob_edge + direction * buffer + (self.location-my_edge), **kwargs)
        return self

    def get_length_in_direction(self, direction):
        return (self.get_boundary_in_direction(direction) - self.get_boundary_in_direction(-direction)).norm(p=2,dim=-1,keepdim=True)

    def move_inline_with_edge(self, mob, direction, edge=None, buffer=DEFAULT_BUFFER, **kwargs):
        """
        Moves this Mob so its edge is aligned with another Mob's edge along a direction.

        Args:
            mob (Mob): The target Mob to align with.
            direction (torch.Tensor): The primary direction for alignment.
            edge (torch.Tensor, optional): If specified, this direction is used
                to determine "which side" of this Mob to use for alignment.
                If None, `direction` is used. Defaults to None.
            buffer (float, optional): Buffer distance. Defaults to DEFAULT_BUFFER.
            **kwargs: Additional arguments for `move`.

        Returns:
            Mob: The Mob instance itself.
        """
        old_loc = Mob(add_to_scene=False).move_next_to(self, direction if edge is None else edge, buffer).location
        new_loc = Mob(add_to_scene=False).move_next_to(mob, direction, buffer).location
        self.move(project_point_onto_line(new_loc - old_loc, direction), **kwargs)
        return self

    def move_inline_with_center(self, mob, direction, buffer=DEFAULT_BUFFER):
        """
        Moves this Mob so its center is aligned with another Mob's center
        along a given direction.

        Args:
            mob (Mob): The target Mob.
            direction (torch.Tensor): The alignment direction.
            buffer (float, optional): Buffer distance (seems unused in current impl).
                                      Defaults to DEFAULT_BUFFER.

        Returns:
            Mob: The Mob instance itself.
        """
        old_loc = self.location
        self.location = old_loc + project_point_onto_line(mob.location-old_loc, direction)
        return self

    def move_inline_with_mob(self, mob, align_direction, center=False, from_mob=None, buffer=DEFAULT_BUFFER):
        mob_edge = mob.get_boundary_in_direction(align_direction) if not center else mob.location
        if from_mob is None:
            from_mob = self
        from_edge = from_mob.get_boundary_in_direction(-align_direction) if not center else from_mob.location
        displacement = mob_edge - from_edge
        align_direction = F.normalize(align_direction, p=2, dim=-1)
        return self.move(dot_product(displacement, align_direction) * align_direction)

    def get_displacement_to_boundary(self, mob, direction):
        my_boundary = self.get_boundary_in_direction(direction)
        other_boundary = mob.get_boundary_in_direction(direction)
        return other_boundary - my_boundary

    def move_inline_with_boundary(self, mob, direction):
        return self.move(self.get_displacement_to_boundary(mob, direction))

    def move_to_edge(self, edge, buffer=DEFAULT_BUFFER):
        """
        Moves the Mob to the edge of the screen.

        Args:
            edge (torch.Tensor): A 3D vector indicating the screen edge
                (e.g., RIGHT, LEFT, UP, DOWN).
            buffer (float, optional): Distance to maintain from the screen border.
                Defaults to DEFAULT_BUFFER.

        Returns:
            Mob: The Mob instance itself.
        """
        edge = F.normalize(edge, p=2, dim=-1)
        bp = self.get_boundary_in_direction(edge)
        edge_point = self.scene.camera.project_point_onto_screen_border(bp, edge)
        target_location = edge_point + F.normalize(bp - edge_point, p=2, dim=-1) * buffer
        displacement = target_location - bp
        self.move(displacement)
        return self

    def move_to_corner(self, edge1, edge2, buffer=DEFAULT_BUFFER):
        """
        Moves the Mob to a corner of the screen, defined by two edge directions.

        Args:
            edge1 (torch.Tensor): Vector for the first screen edge.
            edge2 (torch.Tensor): Vector for the second screen edge.
            buffer (float, optional): Distance from the screen borders.
                Defaults to DEFAULT_BUFFER.

        Returns:
            Mob: The Mob instance itself.
        """
        return self.move_to_edge(edge1, buffer=buffer).move_to_edge(edge2, buffer=buffer)

    def move_out_of_screen(self, edge, buffer=DEFAULT_BUFFER, despawn=True):
        """
        Animates the Mob moving off-screen in a given edge direction and then despawns it.

        Args:
            edge (torch.Tensor): Vector indicating the direction to move off-screen.
            buffer (float, optional): Additional distance beyond the screen edge.
                Defaults to DEFAULT_BUFFER.
            despawn (bool, optional): If True, despawns the Mob after
                moving it. Defaults to True.

        Returns:
            Mob: The Mob instance itself.
        """
        bp = self.get_boundary_edge_point(-edge)
        edge_point = self.scene.camera.project_point_onto_screen_border(bp, edge)
        target_location = edge_point - F.normalize(bp - edge_point, p=2, dim=-1) * buffer
        displacement = target_location - bp
        with Seq():
            self.location = self.location + displacement
            if despawn:
                self.despawn(animate=False)
        return self

    def move_to_point_along_square(self, destination, displacement):
        """
        Moves the Mob to a destination in a two-step "square" path.
        First moves by `displacement`, then moves orthogonally to be in line with destination,
        then moves by `-destination`.

        Args:
            destination (torch.Tensor): The final target 3D location.
            displacement (torch.Tensor): The initial 3D displacement vector.

        Returns:
            Mob: The Mob instance itself.
        """
        dest_disp = destination - self.location
        dn = F.normalize(displacement, p=2, dim=-1)
        orth_disp = dest_disp - dot_product(dest_disp, dn) * dn
        with Seq(run_time=1):
            self.move(displacement)
            self.move(orth_disp)
            self.location = destination
        return self

    def get_length_along_direction(self, direction):
        all_boundary_points = torch.cat([c.get_boundary_points() for c in self.children], -2)
        all_boundary_points -= self.location.unsqueeze(-2)
        dots = dot_product(all_boundary_points, direction.unsqueeze(-2))
        return dots.amax(-2) - dots.amin(-2)

    def get_parts_as_mobs(self):
        parts = [self]
        for c in self.children:
            parts.extend(c.get_parts_as_mobs())
        return parts

    def scale(self, scale_factor, recursive=True):
        """
        Scales the Mob by a factor `scale_factor` relative to its current scale.

        Args:
            scale_factor (float or torch.Tensor): The scaling factor. Can be a scalar or
                a tensor for per-axis/per-batch scaling.
            recursive (bool, optional): If True, applies scaling recursively.
                Defaults to True.

        Returns:
            Mob: The Mob instance itself.
        """
        s = scale_factor * self.scale_coefficient
        return self.set(scale_coefficient=s) if recursive else self.set_non_recursive(scale_coefficient=s)

    def set_scale(self, s, recursive=True):
        return self.set(scale_coefficient=s) if recursive else self.set_non_recursive(scale_coefficient=s)

    '''def set_scale(self, s):
        with Synchronized():
            for c in self.children:
                """c.anchors_enabled = False
                c.local_coord_update_enabled = False
                disp = (self.location - (c.location + displacement))*s + displacement
                c.move(disp)
                c.scale(s, disp)
                c.local_coord_update_enabled = True
                c.anchors_enabled = True"""
                c.set_scale(s)
            self.scale_coefficient = s
        return self'''

    @animated_function(animated_args={'num_degrees': 0}, unique_args=['axis'])
    def rotate(self, num_degrees, axis=OUT):
        """
        Rotates the Mob by a number of degrees around a given axis.

        Args:
            num_degrees (float or torch.Tensor): The angle of rotation in degrees.
            axis (torch.Tensor): The 3D axis of rotation.

        Returns:
            Mob: The Mob instance itself.
        """
        axis = F.normalize(axis, p=2, dim=-1)
        R = get_rotation_around_axis(num_degrees, axis, dim=-1)
        self.basis = squish(unsquish(self.basis, -1, 3) @ R, -2, -1)
        return self

    @animated_function(animated_args={'interpolation': 0}, unique_args=['axis'])
    def rotate_and_scale(self, num_degrees, axis, scale, interpolation=1):
        num_degrees = num_degrees * interpolation
        self.rotate(num_degrees, axis)
        scale = cast_to_tensor(scale)
        scale = self.scale_coefficient * (1 - interpolation) + interpolation * scale * self.scale_coefficient
        self.set_scale(scale)
        return self

    @animated_function(animated_args={'num_degrees': 0}, unique_args=['axis'])
    def rotate_around_point(self, point, num_degrees, axis=OUT):
        """
        Rotates the Mob around an arbitrary point in space.

        Args:
            point (torch.Tensor): The 3D point to rotate around.
            num_degrees (float or torch.Tensor): The angle of rotation in degrees.
            axis (torch.Tensor): The 3D axis of rotation (passing through `point`).

        Returns:
            Mob: The Mob instance itself.
        """
        displacement = self.location - point
        t = rotate_vector_around_axis(displacement, num_degrees, axis, dim=-1) + point
        self.location = t
        return self

    @animated_function(animated_args={'num_degrees': 0}, unique_args=['axis'])
    def rotate_around_point_non_recursive(self, point, num_degrees, axis=OUT):
        displacement = self.location - point
        t = rotate_vector_around_axis(displacement, num_degrees, axis, dim=-1) + point
        self.setattr_non_recursive('location', t)
        return self

    def move_to_point_along_arc(self, point, arc_angle_degrees, arc_normal=OUT, recursive=True):
        #TODO this is bugged
        """
        Moves the Mob to a target point along a circular arc. ***Currently bugged***

        Args:
            point (torch.Tensor): The target 3D location.
            arc_angle_degrees (float or torch.Tensor): The angle subtended by the arc, in degrees.
                The sign determines the direction of rotation along the arc.
            arc_normal (torch.Tensor, optional): The normal vector to the plane
                of the arc. Defaults to OUT.
            recursive (bool, optional): If True, applies the rotation recursively
                to children. Defaults to True.

        Returns:
            Mob: The Mob instance itself.
        """
        my_loc = self.location
        disp_unnnorm = point - my_loc
        disp = disp_unnnorm#F.normalize(disp_unnnorm, p=2, dim=-1)
        disp_normal = F.normalize(broadcast_cross_product(disp, arc_normal), p=2, dim=-1)
        angle_sign = cast_to_tensor(arc_angle_degrees).sign()
        arc_angle_degrees = abs(arc_angle_degrees) if not isinstance(arc_angle_degrees, torch.Tensor) else arc_angle_degrees.abs()
        in1 = F.normalize(rotate_vector_around_axis(disp, arc_angle_degrees-90, arc_normal, -1), p=2, dim=-1)
        in2 = F.normalize(rotate_vector_around_axis(disp, -(arc_angle_degrees + 90), arc_normal, -1), p=2, dim=-1)
        arc_circumference = dot_product(-in1, -in2).clamp_(min=-1, max=1).arccos_()
        m = (((math.pi - arc_circumference).abs() <= 1e-5) | (disp_unnnorm.norm(p=2,dim=-1,keepdim=True) <= 1e-5)).float()
        #if angle is exactly 180 then below formulas divide by 0, so handle edge case here.
        arc_center1 = (my_loc + point) * 0.5
        x1, y1 = 0, 0
        x2, y2 = dot_product(in1, disp_normal), dot_product(in1, disp)
        x3, y3 = dot_product(disp, disp_normal), dot_product(disp, disp)
        x4, y4 = dot_product(in2, disp_normal), dot_product(in2, disp)
        intersect_x = ((x1*y2-y1*x2)*(x3-x4) - (x1-x2)*(x3*y4-y3*x4)) / ((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))
        intersect_y = ((x1*y2-y1*x2)*(y3-y4) - (y1-y2)*(x3*y4-y3*x4)) / ((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))
        arc_center2 = my_loc + intersect_x * disp_normal + intersect_y * disp
        arc_center2 = arc_center2.nan_to_num_(0, 0, 0)
        arc_center = arc_center1 * (m) + (1-m) * arc_center2
        if recursive:
            return self.rotate_around_point(arc_center, arc_circumference * RADIANS_TO_DEGREES * angle_sign, arc_normal)
        else:
            return self.rotate_around_point_non_recursive(arc_center, arc_circumference * RADIANS_TO_DEGREES * angle_sign, arc_normal)

    def refresh_history(self):
        for _ in self.get_descendants():
            _.data.history = ModificationHistory()
            _.data.spawn_time = lambda: -1

    def detach_history(self):
        with Off():
            with NoExtra(priority_level=1):
                if self.data.spawn_time() >= 0:
                    old_self = self.clone(reset_history=False)
                    old_self.wait((1 / self.scene.frames_per_second) + 1e-5)
                    old_self.despawn(animate=False)
                self.refresh_history()
                self.spawn(animate=False)
                return self

    def expand_n_children(self, n):
        curr = len(self.children)
        target = curr + n
        repeat_indices = (torch.arange(target) * curr) // target
        split_factors = [(repeat_indices == i).sum() for i in range(curr)]
        new_submobs = []
        for submob, sf in zip(self.children, split_factors):
            new_submobs.append(submob)
            for _ in range(1, sf):
                new_submobs.append(submob.clone())

        self.children = new_submobs
        return self

    def expand_n_batch(self, n):
        curr = self.location.shape[-2] // self.num_points_per_object
        target = curr + n
        repeat_indices = (torch.arange(target) * curr) // target
        split_factors = [(repeat_indices == i).sum() for i in range(curr)]
        for attr in ['location', 'opacity', 'color', 'basis', 'glow']:
            value = self.__getattribute__(attr)[0]
            if value.shape[-2] == 1:
                continue
            value = unsquish(value, -2, self.num_points_per_object)
            new_submobs = []
            for submob, sf in zip(value, split_factors):
                new_submobs.append(submob)
                for _ in range(1, sf):
                    new_submobs.append(submob[...,-1:,:].expand(*[-1 for _ in range(submob.dim()-2)], self.num_points_per_object, -1))
            self.data.data_dict[attr] = squish(torch.stack(new_submobs,-3), -3, -2).unsqueeze(0)
        return self

    def become(self, other_mob, move_to=False, detach_history=True):
        """
        Transforms this Mob into another Mob (`other_mob`).

        This involves animating changes in location, opacity, color, basis, etc.,
        to match `other_mob`. It attempts to match parts of this Mob to parts
        of `other_mob` for a smoother transition, especially for complex Mobs.

        Args:
            other_mob (Mob): The Mob to transform into.
            move_to (bool, optional): If True, explicitly moves this Mob to
                `other_mob`'s location as part of the transformation.
                (The current implementation seems to always align attributes
                including location). Defaults to False.

        Returns:
            Mob: The (transformed) Mob instance itself.
        """
        #TODO When doing become multiple times into Text despawn, the triangles are fadedout out in the wrong order

        with Off():
            if detach_history:
                self.detach_history()
                other_mob_original = other_mob
                other_mob = other_mob.clone(add_to_scene=False)#.detatch_history()

            child_diff = len(other_mob.children) - len(self.children)
            if child_diff > 0:
                self.expand_n_children(child_diff)
            elif child_diff < 0:
                other_mob.expand_n_children(-child_diff)

        with Seq():
            with Sync():
                for my_c, other_c in zip(self.children, other_mob.children):
                    my_c.become(other_c, detach_history=False)

                if other_mob.num_points_per_object != self.num_points_per_object:
                    raise NotImplementedError("You are trying to change an object of one primitive type (e.g. triangle) to another type (e.g. cubic bezier circuit), this it not supported. When using become() the target mob must be of the same type as the original.")
                batch_diff = (other_mob.location.shape[-2] - self.location.shape[-2]) // self.num_points_per_object
                if batch_diff > 0:
                    self.expand_n_batch(batch_diff)
                elif batch_diff < 0:
                    other_mob.expand_n_batch(-batch_diff)

                for attr in ['location', 'opacity', 'color', 'basis', 'glow']:
                    self.setattr_non_recursive(attr, getattr(other_mob, attr))

            if detach_history:
                with Off():
                    self.despawn(animate=False)
                    return other_mob_original.spawn(animate=False)#self.detach_history()
            return self
        #with Off():
        #    self.despawn(animate=False)
        #    return other_mob_original.detach_history()

    def _become_recursive(self, other_mob, move_to=False):
        other_c = list(other_mob.children)
        my_c = list(self.children)
        with Sync():
            if len(other_c) > 0:
                if len(my_c) < len(other_c):
                    with Off():
                        new_c = [c.clone(animate_creation=False, recursive=True) for c in other_c[len(my_c):]]
                        my_c.extend(new_c)
                        self.add_children(new_c)
                        [c.set_recursive(local_coords=ORIGIN) for c in new_c]
                elif len(other_c) < len(my_c):
                    my_c = my_c[:len(other_c)]
                [my_c[i]._become_recursive(other_c[i], move_to=True) for i in reversed(range(len(my_c)))]
            if not move_to:
                return self

            other_loc = other_mob.location
            my_loc = self.location
            my_n = my_loc.shape[-2]
            other_n = other_loc.shape[-2]

            if other_n > my_n:
                with Off(record_funcs=False, record_attr_modifications=False):
                    self.setattr_regular('_location', torch.cat([my_loc, my_loc[..., -1:, :].expand(-1, other_n - my_n, -1)], -2))
                    self.batch_size = max(self.batch_size, self.location.shape[-2])
                    self.parent_batch_sizes = other_mob.parent_batch_sizes
            elif other_n < my_n:
                other_loc = torch.cat((other_loc, torch.zeros((other_loc.shape[0], my_n-other_n, other_loc.shape[2]))), -2)
            self.location = other_loc
        return self

    def set_non_recursive(self, **kwargs):
        """Sets multiple attributes non-recursively (i.e., only for this Mob, not children).

        Args:
            **kwargs: Keyword arguments where keys are attribute names and
                values are the new values for those attributes.

        Returns:
            Mob: The Mob instance itself.
        """
        with Sync():
            for k, v in kwargs.items():
                self.setattr_non_recursive(k, v)
        return self

    def set(self, **kwargs):
        """Sets multiple attributes, applying changes recursively to children by default.

        Parameters
        ----------
        **kwargs
            Keyword arguments where keys are attribute names and values are the new values.

        Returns
        -------
            Mob: The Mob instance itself.
        Examples
        ---------

        .. algan:: Example1MobSet

            from algan import *

            mob = Square().spawn()
            mob.set(location=RIGHT, color=BLUE)

            render_to_file()
        """
        with Sync():
            for k, v in kwargs.items():
                self.__setattr__(k, v)
        return self

    def set_recursive_from_parent(self, parent, **kwargs):
        if self.parent_batch_sizes is not None:
            def cast_to_tensor(x):
                if isinstance(x, torch.Tensor):
                    return x
                return torch.tensor((x,)).view(1,1,1).expand(len(self.parent_batch_sizes),-1, -1)
            if self.time_inds_materialized is None and parent.time_inds_materialized is not None:
                self.set_state_full(parent.time_inds_materialized.amin(), parent.time_inds_materialized.amax()+1)
            self.time_inds_active = parent.time_inds_active
            kwargs = {k: torch.repeat_interleave(cast_to_tensor(v), self.parent_batch_sizes, 0) for k, v in kwargs.items()}
        with AnimationContext(dont_record_funcs=True):
            return self.set_recursive(**kwargs)

    def set_recursive(self, **kwargs):
        self.set(**kwargs)
        for c in self.children:
            c.set_recursive_from_parent(self, **kwargs)
        return self

    def get_forward_direction(self):
        """
        Gets the Mob's current forward direction vector (normalized).
        This is the third column of its normalized basis matrix.

        Returns:
            torch.Tensor: The 3D forward vector.
        """
        return F.normalize(unsquish(self.basis, -1, 3)[...,2,:], p=2, dim=-1)

    def get_right_direction(self):
        """
        Gets the Mob's current right direction vector (normalized).
        This is the first column of its normalized basis matrix.

        Returns:
            torch.Tensor: The 3D right vector.
        """
        return F.normalize(unsquish(self.basis, -1, 3)[...,0,:], p=2, dim=-1)

    def get_upwards_direction(self):
        """
        Gets the Mob's current upwards direction vector (normalized).
        This is the second column of its normalized basis matrix.

        Returns:
            torch.Tensor: The 3D upwards vector.
        """
        return F.normalize(unsquish(self.basis, -1, 3)[...,1,:], p=2, dim=-1)

    def look(self, direction, axis=2):
        """
        Rotates the Mob so that one of its local axes points in the given direction.

        Args:
            direction (torch.Tensor): The target 3D direction vector.
            axis (int, optional): The index of the local axis to align.
                0 for right, 1 for up, 2 for forward.
                Defaults to 2 (forward vector).

        Returns:
            Mob: The Mob instance itself.
        """
        return self.rotate(*get_rotation_between_3d_vectors(unsquish(self.normalized_basis, -1, 3)[..., axis,:], F.normalize(direction, p=2,dim=-1), dim=-1))

    def look_and_scale(self, direction, scale, axis=2):
        return self.rotate_and_scale(*get_rotation_between_3d_vectors(unsquish(self.normalized_basis, -1, 3)[..., axis,:], F.normalize(direction, p=2,dim=-1), dim=-1), scale)

    def look_at(self, point, axis=2):
        """
        Looks in the direction from this mobs location to the given point.
        The Mob's "forward" direction ok`'s default axis)
        will be oriented towards the point.

        Args:
            point (torch.Tensor): The 3D point to look at.
            axis (int, optional): The index of the local axis to align.
                0 for right, 1 for up, 2 for forward.
                Defaults to 2 (forward vector).

        Returns:
            Mob: The Mob instance itself.
        """
        direction = point - self.location
        return self.look(direction, axis=axis)

    def spawn_tilewise_recursive(self):
        tiles = [_.tiles for _ in traverse(self.get_descendants()) if hasattr(_, 'tiles') and not (_.tiles.time_inds.created)]
        with AnimationContext(run_time=3):
            animate_lagged_by_location(tiles, lambda m: m.spawn_from_random_direction(), F.normalize(RIGHT * 1.5 + DOWN, p=2, dim=-1))
        return self

    def despawn_tilewise_recursive(self):
        tiles = [_.tiles for _ in traverse(self.get_descendants()) if hasattr(_, 'tiles')]
        with AnimationContext(run_time=3):
            animate_lagged_by_location(tiles, lambda m: m.despawn_from_random_direction(), F.normalize(RIGHT * 1.5 + DOWN, p=2, dim=-1))
        return self

    def spawn_from_random_direction(self, travel_distance=0.1):
        with Off():
            self.opacity = 0
        self._create_recursive(animate=False)
        with Sync(run_time=None, rate_func=ease_out_exp):
            #self.location = loc
            #self.rotate(720, F.normalize(torch.randn_like(self.location), p=2, dim=-1))
            self.opacity = 1#set_recursive(opacity=1)
            #with Synchronized(run_time=2, rate_func=tan):

    def __len__(self):
        return self.location.shape[-2] if hasattr(self, 'location') else 1

    def despawn_from_random_direction(self, travel_distance=0.1):
        with Sync(run_time=None, rate_func=inversed(ease_out_exp)):
            loc = self.location
            #self.location = loc + torch.randn_like(loc) * travel_distance
            #self.rotate(720, F.normalize(torch.randn_like(self.location), p=2, dim=-1))
            self.opacity = 0#set_recursive(opacity=0)
            self._destroy_recursive(animate=False)
            # with Synchronized(run_time=2, rate_func=tan):
            #self.destroy()

    def set_data_sub_inds(self, data_sub_inds):
        self.batch_size = max(self.batch_size, self.location.shape[1])
        if self.parent_batch_sizes is not None:
            sub_pbs = self.parent_batch_sizes[data_sub_inds]
            inds = torch.arange(self.batch_size).split([_.item() for _ in self.parent_batch_sizes])
            data_sub_inds = torch.cat([inds[d] for d in data_sub_inds] if not isinstance(data_sub_inds, slice) else inds[data_sub_inds])
        else:
            sub_pbs = self.parent_batch_sizes
        self.data_sub_inds = data_sub_inds
        self.parent_batch_sizes = sub_pbs
        for c in self.children:
            c.set_data_sub_inds(data_sub_inds)

    def __getitem__(self, item):
        """
        Allows accessing a part of a batched Mob using slice notation.

        Returns a new Mob instance that represents the specified sub-part(s),
        sharing animation data but with `data_sub_inds` set appropriately.

        Args:
            item (int or slice): The index or slice for the batch dimension.

        Returns:
            Mob: A new Mob representing the selected part(s).
        """
        m = self.clone(add_to_scene=False, clone_data=False, recursive=True, animate_creation=False)
        m.set_data_sub_inds([item] if isinstance(item, int) else item)
        return m