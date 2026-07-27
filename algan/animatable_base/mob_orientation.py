"""Screen-relative layout and bounding-box queries for :class:`~algan.mobs.mob.Mob`.

Split out of ``mob.py`` for readability; :class:`MobLayoutMixin` is mixed into
``Mob`` and is not useful standalone (``self`` is always a Mob).
"""
from __future__ import annotations

import torch.nn.functional as F

from algan import animated_function
from algan.animation_timeline.animation_contexts import Sync
from algan.constants.spatial import *
from algan.geometry.geometry import project_point_onto_line, rotate_vector_around_axis, get_rotation_around_axis, \
    get_rotation_between_3d_vectors
from algan.settings.style_defaults import STYLE_DEFAULTS
from algan.utils.tensor_utils import cast_to_tensor, squish, unsquish

DEFAULT_BUFFER = STYLE_DEFAULTS.buffer


class MobOrientationMixin:
    """Rotations. """

    def reset_basis(self):
        """Reset the basis to identity and return ``self``."""
        self.basis = cast_to_tensor(squish(torch.eye(3)))
        return self

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
        with Sync(animation_manager=self.animation_manager):
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