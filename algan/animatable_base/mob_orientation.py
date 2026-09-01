"""Rotation and orientation methods for :class:`~algan.animatable_base.mob.Mob`.

Split out of ``mob.py`` for readability; :class:`MobOrientationMixin` is mixed
into ``Mob`` and is not useful standalone (``self`` is always a Mob).
"""

from __future__ import annotations

import math

import torch.nn.functional as F

from algan import animated_function
from algan.constants.spatial import *
from algan.errors import AlganConfigurationError
from algan.geometry.geometry import (
    get_rotation_around_axis,
    get_rotation_between_3d_vectors,
    rotate_vector_around_axis,
)
from algan.utils.tensor_utils import (
    cast_to_direction,
    cast_to_tensor,
    squish,
    unsquish,
)

_RADIANS_TO_DEGREES = 180.0 / math.pi

#: Local axis names accepted by ``look`` / ``look_at``, mapped to the basis row
#: they select.
_AXIS_NAMES = {"right": 0, "up": 1, "forward": 2}


class MobOrientationMixin:
    """Methods for rotating and orienting Mobs, mixed into
    :class:`~algan.animatable_base.mob.Mob`.
    """

    def reset_basis(self) -> Mob:
        """Reset the Mob's orientation and scale to the default.

        The basis is set back to the identity matrix, which undoes every
        :meth:`~algan.animatable_base.mob_orientation.MobOrientationMixin.rotate`
        **and** every :meth:`~algan.animatable_base.mob.Mob.scale` applied so far.
        The Mob's location is unaffected.

        Animation
        ---------
        Recorded as an animation: the Mob rotates and scales back to its default
        over the current context's duration (1 second by default). Applies to
        this Mob and its descendants.

        Returns
        -------
        :class:`~algan.animatable_base.mob.Mob`
            This Mob, so calls can be chained.
        """
        self.basis = cast_to_tensor(squish(torch.eye(3)))
        return self

    @animated_function(
        animated_args={"angle": 0},
        unique_args=["axis", "about", "degrees"],
    )
    def rotate(
        self,
        angle: float | torch.Tensor,
        axis: torch.Tensor = OUTWARD,
        about: torch.Tensor | None = None,
        *,
        degrees: bool = True,
    ) -> Mob:
        """Rotate the Mob about an axis, optionally around a point in space.

        With the default ``about=None`` only the Mob's orientation changes
        and it stays where it is. Given an ``about`` point, the Mob also travels
        around the axis through that point, like a planet spinning as it orbits.
        To move around a point *without* re-orienting the Mob, use
        :meth:`~algan.animatable_base.mob_orientation.MobOrientationMixin.orbit`.

        Animation
        ---------
        Recorded as an animation: the rotation sweeps from 0 to ``angle``
        over the current context's duration (1 second by default), so the Mob
        turns rather than snapping. Retime it with
        ``with Seq(duration=3): mob.rotate(90)``, or apply it instantly with
        ``with Off(): mob.rotate(90)``. Applies to this Mob and its descendants.

        Parameters
        ----------
        angle
            How far to rotate, counter-clockwise when looking down ``axis``, in
            degrees unless ``degrees`` is False. Accepts a tensor of shape
            ``(*, 1)`` to give each Mob of a batch its own angle.
        axis
            Axis to rotate around; need not be normalized. Defaults to ``OUTWARD``
            (the +z axis, pointing out of the screen), which spins a flat 2-D
            shape in the screen plane.
        about
            Point to rotate around, shape ``(*, 3)``. Defaults to ``None``,
            meaning rotate in place about the Mob's own center.
        degrees
            Whether ``angle`` is in degrees. Defaults to True; pass False to give
            it in radians.

        Returns
        -------
        :class:`~algan.animatable_base.mob.Mob`
            This Mob, so calls can be chained.

        Examples
        --------
        .. algan:: Example1MobRotate

            from algan import *

            square = Square().spawn()
            square.rotate(90)
            square.rotate(180, axis=UP)
            square.rotate(90, about=RIGHT * 2)

            Scene.save_video()
        """
        # angle has already been through cast_to_tensor in prepare_kwargs,
        # so a swapped rotate(OUTWARD, 90) arrives here as a (*, 3) angle and a
        # scalar axis. Catching it here names the parameters; left alone it
        # surfaces as an IndexError from deep inside the rotation matrix build.
        if isinstance(angle, torch.Tensor) and angle.shape[-1] != 1:
            raise AlganConfigurationError(
                "angle must be an angle, not a vector; the "
                "signature is rotate(angle, axis)"
            )
        # Scaling commutes with the decorator's interpolation from 0, so
        # converting the already-interpolated value here is exact.
        num_degrees = angle if degrees else angle * _RADIANS_TO_DEGREES
        axis = F.normalize(cast_to_direction("axis", axis), p=2, dim=-1)
        rotation_matrix = get_rotation_around_axis(num_degrees, axis, dim=-1)
        self.basis = squish(unsquish(self.basis, -1, 3) @ rotation_matrix, -2, -1)
        if about is not None:
            self.orbit(num_degrees, axis, cast_to_direction("about", about))
        return self

    @animated_function(
        animated_args={"angle": 0},
        unique_args=["axis", "about", "degrees"],
    )
    def orbit(
        self,
        angle: float | torch.Tensor,
        axis: torch.Tensor = OUTWARD,
        about: torch.Tensor | None = None,
        *,
        degrees: bool = True,
    ) -> Mob:
        """Move the Mob around a point without turning it.

        The Mob's location swings around the axis while its orientation is left
        unchanged.
        For an orbit that also turns the object as it moves, use
        :meth:`~algan.animatable_base.mob_orientation.MobOrientationMixin.rotate` with an
        ``about`` point.

        Animation
        ---------
        Recorded as an animation: the orbit sweeps from 0 to ``angle`` over
        the current context's duration (1 second by default). Applies to this Mob
        and its descendants.

        Parameters
        ----------
        angle
            How far around to travel, counter-clockwise when looking down
            ``axis``, in degrees unless ``degrees`` is False.
        axis
            Axis to orbit around; need not be normalized. Defaults to ``OUTWARD``
            (the +z axis, out of the screen).
        about
            Point to orbit around, shape ``(*, 3)``. Defaults to ``None``, which
            makes the call a **no-op** -- orbiting the Mob's own center would not
            move it, since its orientation is held fixed.
        degrees
            Whether ``angle`` is in degrees. Defaults to True; pass False to give
            it in radians.

        Returns
        -------
        :class:`~algan.animatable_base.mob.Mob`
            This Mob, so calls can be chained.
        """
        if about is None:
            return self
        num_degrees = angle if degrees else angle * _RADIANS_TO_DEGREES
        axis = F.normalize(cast_to_direction("axis", axis), p=2, dim=-1)
        about = cast_to_direction("about", about)
        self.location = (
            rotate_vector_around_axis(
                self.location - about,
                num_degrees,
                axis,
                dim=-1,
            )
            + about
        )
        return self

    def get_forward_basis(self) -> torch.Tensor:
        """Get the Mob's forward basis vector, scale included.

        This is the third row of the Mob's basis matrix, so its length is the
        Mob's scale along that axis rather than 1. For a unit-length direction,
        use
        :meth:`~algan.animatable_base.mob_orientation.MobOrientationMixin.get_forward_direction`.

        Returns
        -------
        torch.Tensor
            The forward basis vector, shape ``(*, 3)``, not normalized.
        """
        return unsquish(self.basis, -1, 3)[..., 2, :]

    def get_right_basis(self) -> torch.Tensor:
        """Get the Mob's rightward basis vector, scale included.

        This is the first row of the Mob's basis matrix, so its length is the
        Mob's scale along that axis rather than 1. For a unit-length direction,
        use
        :meth:`~algan.animatable_base.mob_orientation.MobOrientationMixin.get_right_direction`.

        Returns
        -------
        torch.Tensor
            The rightward basis vector, shape ``(*, 3)``, not normalized.
        """
        return unsquish(self.basis, -1, 3)[..., 0, :]

    def get_up_basis(self) -> torch.Tensor:
        """Get the Mob's upward basis vector, scale included.

        This is the second row of the Mob's basis matrix, so its length is the
        Mob's scale along that axis rather than 1. For a unit-length direction,
        use
        :meth:`~algan.animatable_base.mob_orientation.MobOrientationMixin.get_up_direction`.

        Returns
        -------
        torch.Tensor
            The upward basis vector, shape ``(*, 3)``, not normalized.
        """
        return unsquish(self.basis, -1, 3)[..., 1, :]

    def get_forward_direction(self) -> torch.Tensor:
        """Get the direction the Mob is facing.

        The normalized third row of the Mob's basis, i.e. its local +z axis in
        world space -- which for an unrotated Mob is ``OUTWARD``, towards the
        viewer. Scale is divided out, so this is always unit length.

        Returns
        -------
        torch.Tensor
            Unit forward direction, shape ``(*, 3)``.
        """
        return F.normalize(unsquish(self.basis, -1, 3)[..., 2, :], p=2, dim=-1)

    def get_right_direction(self) -> torch.Tensor:
        """Get the Mob's own rightward direction.

        The normalized first row of the Mob's basis, i.e. its local +x axis in
        world space. Scale is divided out, so this is always unit length. Note
        this is the Mob's right, which is only the screen's ``RIGHT`` while the
        Mob is unrotated.

        Returns
        -------
        torch.Tensor
            Unit rightward direction, shape ``(*, 3)``.
        """
        return F.normalize(unsquish(self.basis, -1, 3)[..., 0, :], p=2, dim=-1)

    def get_up_direction(self) -> torch.Tensor:
        """Get the Mob's own upward direction.

        The normalized second row of the Mob's basis, i.e. its local +y axis in
        world space. Scale is divided out, so this is always unit length.

        Returns
        -------
        torch.Tensor
            Unit upward direction, shape ``(*, 3)``.
        """
        return F.normalize(unsquish(self.basis, -1, 3)[..., 1, :], p=2, dim=-1)

    #: The three direction getters also answer to a bare property, so
    #: ``mob.up`` reads as well as ``mob.get_up_direction()``. This is the one
    #: place ``Mob`` carries a deliberate alias (see ``CLAUDE.md``); the basis
    #: getters have no property spelling on purpose, because they carry the
    #: Mob's scale and a scaled vector reads wrongly as ``mob.up``.
    right = property(get_right_direction)
    up = property(get_up_direction)
    forward = property(get_forward_direction)

    @staticmethod
    def _resolve_axis(with_axis: str) -> int:
        """Map ``'right'`` / ``'up'`` / ``'forward'`` onto a basis row index."""
        try:
            return _AXIS_NAMES[str(with_axis).lower()]
        except KeyError:
            raise AlganConfigurationError(
                f"with_axis must be one of {sorted(_AXIS_NAMES)}, got {with_axis!r}"
            ) from None

    def look(self, direction: torch.Tensor, with_axis: str = "forward") -> Mob:
        """Turn the Mob so one of its own axes points a given way.

        The rotation taken is the shortest one that lines the chosen local axis
        up with ``direction``; the Mob's spin about that axis is otherwise left
        as it was.

        Animation
        ---------
        Recorded as an animation: the turn is performed by
        :meth:`~algan.animatable_base.mob_orientation.MobOrientationMixin.rotate`,
        so it sweeps over the current context's duration
        (1 second by default). Applies to this Mob and its descendants.

        Parameters
        ----------
        direction
            World-space direction the chosen axis should point along, shape
            ``(*, 3)``; need not be normalized.
        with_axis
            Which of the Mob's local axes to aim: ``'right'``, ``'up'`` or
            ``'forward'``, matched case-insensitively. Defaults to
            ``'forward'``.

        Returns
        -------
        :class:`~algan.animatable_base.mob.Mob`
            This Mob, so calls can be chained.

        Raises
        ------
        :class:`.AlganConfigurationError`
            If ``with_axis`` is not one of the three axis names.

        See Also
        --------
        :meth:`~algan.animatable_base.mob_orientation.MobOrientationMixin.look_at`
            Aim at a point rather than along a direction.
        """
        # Get the rotation parameters (angle and axis) needed to align the current local axis
        # with the target direction.
        rotation_angle_degrees, rotation_axis = get_rotation_between_3d_vectors(
            unsquish(self.normalized_basis, -1, 3)[
                ..., self._resolve_axis(with_axis), :
            ],  # Current orientation of specified axis
            F.normalize(direction, p=2, dim=-1),  # Normalized target direction
            dim=-1,
        )
        # Apply the rotation
        return self.rotate(rotation_angle_degrees, rotation_axis)

    def look_at(self, point: torch.Tensor, with_axis: str = "forward") -> Mob:
        """Turn the Mob to face a point in space.

        Equivalent to
        :meth:`~algan.animatable_base.mob_orientation.MobOrientationMixin.look`
        along the direction from the Mob to
        ``point``, so the Mob's location is unchanged -- only where it is aimed.

        Animation
        ---------
        Recorded as an animation over the current context's duration (1 second by
        default). The direction is resolved when the call is recorded, so a Mob
        aimed at a moving target will not track it (use an updater for that).
        Applies to this Mob and its descendants.

        Parameters
        ----------
        point
            World-space point to face, shape ``(*, 3)``.
        with_axis
            Which of the Mob's local axes to aim at ``point``: ``'right'``,
            ``'up'`` or ``'forward'``, matched case-insensitively. Defaults to
            ``'forward'``.

        Returns
        -------
        :class:`~algan.animatable_base.mob.Mob`
            This Mob, so calls can be chained.

        Raises
        ------
        :class:`.AlganConfigurationError`
            If ``with_axis`` is not one of the three axis names.
        """
        # Calculate the direction vector from the Mob's current location to the target point
        direction_to_look = point - self.location
        return self.look(direction_to_look, with_axis=with_axis)
