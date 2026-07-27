"""Screen-relative layout and bounding-box queries for :class:`~algan.mobs.mob.Mob`.

Split out of ``mob.py`` for readability; :class:`MobLayoutMixin` is mixed into
``Mob`` and is not useful standalone (``self`` is always a Mob).
"""
from __future__ import annotations

import torch.nn.functional as F

from algan import animated_function
from algan.constants.spatial import *
from algan.geometry.geometry import (
    get_rotation_around_axis,
    get_rotation_between_3d_vectors,
    rotate_vector_around_axis,
)
from algan.settings.style_defaults import STYLE_DEFAULTS
from algan.utils.tensor_utils import cast_to_tensor, squish, unsquish

DEFAULT_BUFFER = STYLE_DEFAULTS.buffer


class MobOrientationMixin:
    """Rotations."""

    def reset_basis(self):
        """Reset the basis to identity and return ``self``."""
        self.basis = cast_to_tensor(squish(torch.eye(3)))
        return self

    @animated_function(
        animated_args={"num_degrees": 0},
        unique_args=["axis", "about_point"],
    )
    def rotate(
        self,
        num_degrees: float | torch.Tensor,
        axis: torch.Tensor = OUT,
        about_point: torch.Tensor | None = None,
    ) -> Mob:
        """Rotate the Mob around an axis, optionally about a point in space.

        When ``about_point`` is ``None``, only the basis changes. Otherwise,
        the location also moves around the axis through ``about_point``.
        """
        axis = F.normalize(cast_to_tensor(axis), p=2, dim=-1)
        rotation_matrix = get_rotation_around_axis(num_degrees, axis, dim=-1)
        self.basis = squish(
            unsquish(self.basis, -1, 3) @ rotation_matrix, -2, -1
        )
        if about_point is not None:
            about_point = cast_to_tensor(about_point)
            self.location = (
                rotate_vector_around_axis(
                    self.location - about_point,
                    num_degrees,
                    axis,
                    dim=-1,
                )
                + about_point
            )
        return self

    @animated_function(
        animated_args={"num_degrees": 0},
        unique_args=["axis", "about_point"],
    )
    def orbit(
        self,
        num_degrees: float | torch.Tensor,
        axis: torch.Tensor = OUT,
        about_point: torch.Tensor | None = None,
    ) -> Mob:
        """Move the Mob around a point without changing its basis.

        When ``about_point`` is ``None``, the Mob is unchanged.
        """
        if about_point is None:
            return self
        axis = F.normalize(cast_to_tensor(axis), p=2, dim=-1)
        about_point = cast_to_tensor(about_point)
        self.location = (
            rotate_vector_around_axis(
                self.location - about_point,
                num_degrees,
                axis,
                dim=-1,
            )
            + about_point
        )
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
