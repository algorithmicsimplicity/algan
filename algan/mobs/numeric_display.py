"""A number on screen that animates between values.

:class:`DecimalNumber` renders a number as text and makes its ``value`` an
animatable attribute, so ``counter.value = 100`` counts smoothly from wherever it
was over the surrounding context's runtime -- re-rendering the glyphs each frame
rather than interpolating their outlines.

``decimal_places`` and ``integer_places`` fix the format so the display
does not jitter in width as digits come and go.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from algan.animatable_base.animatable import animated_function
from algan.animatable_base.mob import Mob
from algan.animation_timeline.animation_contexts import (
    NoExtra,
    Off,
    Seq,
    Sync,
    active_scene_for_new_mob,
)
from algan.geometry.geometry import (
    map_global_to_local_coords,
    map_local_to_global_coords,
)
from algan.mobs.text import Tex
from algan.utils.tensor_utils import cast_to_tensor


class DecimalNumber(Mob):
    """A number drawn on screen, which counts when you assign to it.

    :attr:`value` is an animatable attribute, so ``counter.value = 100``
    is recorded like a move or a color change and the display counts from
    wherever it was to 100 over the surrounding context's runtime. The glyphs
    are re-typeset each frame rather than morphed into one another, which is
    what makes ``7`` become ``8`` rather than bending into it.

    The number is laid out to a fixed width from ``decimal_places`` and
    ``integer_places``, so it does not jitter sideways as digits come and go.
    Leading-digit slots are added automatically when the value outgrows them.

    Animation
    ---------
    Constructing one records nothing; the Mob joins the active Scene unspawned,
    and :meth:`~algan.animatable_base.animatable.Animatable.spawn` makes it
    appear. Assigning to :attr:`value` afterwards is recorded and interpolates
    from the current value over the current context's runtime (1 second by
    default) -- ``with Seq(runtime=3): counter.value = 100`` to change that, or
    ``with Off():`` to jump.

    Parameters
    ----------
    value
        The number shown at construction.
    decimal_places
        Digits after the decimal point. Defaults to ``2``; ``0`` shows no
        decimal point at all.
    integer_places
        Minimum digits before the decimal point. Defaults to ``None``, meaning
        take it from ``value`` (at least 1). More are allocated automatically
        when the value grows past the reserved width.
    **kwargs
        Passed to :class:`~algan.mobs.text.Tex`, which typesets the digits --
        notably ``color`` and ``font_size`` -- and on to :class:`~.Mob`.

    Attributes
    ----------
    value
        The number displayed, shape ``(*, 1)``. Assigning to it records a
        counting animation.

    Examples
    --------
    A counter that runs from 0 to 50 over two seconds:

    .. algan:: Example1DecimalNumber

        from algan import *

        counter = DecimalNumber(0, decimal_places=1).spawn()
        with Seq(runtime=2):
            counter.value = 50

        Scene.save_video()
    """

    def __init__(self, value, decimal_places=2, integer_places=None, **kwargs):
        if kwargs.get("scene") is None:
            kwargs["scene"] = active_scene_for_new_mob()
        value = cast_to_tensor(value)
        self.decimal_places = decimal_places
        required_integer_places = self._required_integer_places(value)
        if integer_places is None:
            integer_places = required_integer_places
        else:
            integer_places = max(integer_places, required_integer_places)
        self.integer_places = integer_places
        num_i, num_d = integer_places, decimal_places
        animation_manager = kwargs["scene"].animation_manager
        with (
            Off(animation_manager=animation_manager),
            NoExtra(priority_level=1, animation_manager=animation_manager),
        ):
            self.placeholder = Tex(
                "-" + "0" * num_i + ("." + "0" * num_d if num_d > 0 else ""), **kwargs
            )
            ct = self.placeholder.animation_manager.context.timespan.current_time
            self.placeholder.animation_manager.context.rewind(
                1 / self.placeholder.scene.frames_per_second + 1e-3
            )
            self.placeholder.opacity = 0
            self.placeholder.animation_manager.context.timespan.current_time = ct

            # Placeholder glyphs: [0]='-', [1..num_i]=integer digits,
            # [num_i+1]='.', [num_i+2..]=decimal digits.
            self.decimal = self.placeholder[num_i + 1] if num_d > 0 else None
            self.negative_sign = self.placeholder[0]
            if num_i >= 2:
                digit_advance_points = (
                    self.placeholder[1].location,
                    self.placeholder[2].location,
                )
            elif num_d >= 2:
                digit_advance_points = (
                    self.placeholder[num_i + 2].location,
                    self.placeholder[num_i + 3].location,
                )
            else:
                spacing_kwargs = dict(kwargs)
                spacing_kwargs["add_to_scene"] = False
                spacing_template = Tex("00", **spacing_kwargs)
                digit_advance_points = (
                    spacing_template[0].location,
                    spacing_template[1].location,
                )
            self.digit_mobs = []
            for _ in range(num_i + num_d):
                self.digit_mobs.append(Tex("0123456789", **kwargs))
                self.digit_mobs[-1].set(opacity=0)
            for i in range(len(self.digit_mobs)):
                location = self.placeholder[
                    1 + i + (1 if (num_d > 0 and i >= num_i) else 0)
                ].location
                for d in self.digit_mobs[i].character_mobs:
                    d.location = location
            super().__init__(**kwargs)
            self.register_attrs_as_animatable(["value"], DecimalNumber)
            self._setattr_without_record("value", value)

            parent_location, parent_basis = self.location, self.basis

            def to_local(point):
                return map_global_to_local_coords(parent_location, parent_basis, point)

            self._rightmost_integer_location = to_local(
                self.digit_mobs[num_i - 1].character_mobs[0].location
            )
            self._digit_advance = to_local(digit_advance_points[1]) - to_local(
                digit_advance_points[0]
            )
            leftmost_integer_location = self._rightmost_integer_location - (
                self._digit_advance * (num_i - 1)
            )
            self._negative_sign_offset = (
                to_local(self.negative_sign.location) - leftmost_integer_location
            )
            self.update_display(self.value)
        if self.decimal is not None:
            self.add_children(self.digit_mobs, self.decimal, self.negative_sign)
        else:
            self.add_children(self.digit_mobs, self.negative_sign)
        for c in self.children:
            c.on_create = lambda c=c: c
        # self.components = [*self.digit_mobs, self.decimal, self.negative_sign]

    @property
    def value(self):
        return self.get_animated_attribute("value")

    @value.setter
    def value(self, value):
        return self.set_value(value)

    def on_create(self):
        with Sync(animation_manager=self.animation_manager):
            for c in self.get_descendants():
                o = c.opacity
                with Seq(animation_manager=self.animation_manager):
                    with Off(animation_manager=self.animation_manager):
                        c.set_non_recursive(opacity=0)
                    c.set_non_recursive(opacity=o)

    def get_value(self):
        return self.value

    def set_value(self, value, interpolation=1):
        value = cast_to_tensor(value)
        self._ensure_integer_places(value)
        return self._set_value(value, interpolation=interpolation)

    @animated_function(animated_args={"interpolation": 0})
    def _set_value(self, value, interpolation=1):
        value = cast_to_tensor(value)
        old_value = self.value
        interpolated_value = old_value * (1 - interpolation) + interpolation * value
        self.update_display(interpolated_value)
        self._setattr_and_record_modification("value", value)
        return self

    def _required_integer_places(self, value):
        """Return the slots needed by ``value`` after decimal rounding."""
        required = 1
        for item in cast_to_tensor(value).reshape(-1):
            scalar = abs(float(item.item()))
            if not math.isfinite(scalar):
                continue
            formatted = f"{scalar:.{self.decimal_places}f}"
            required = max(required, len(formatted.partition(".")[0]))
        return required

    def _ensure_integer_places(self, value):
        """Prepend enough integer slots for a newly assigned finite value."""
        required = self._required_integer_places(value)
        if required <= self.integer_places:
            return

        old_integer_places = self.integer_places
        source_digit = self.digit_mobs[0]
        new_digits_near_to_far = []
        with Off(
            record_funcs=False,
            record_attr_modifications=False,
            animation_manager=self.animation_manager,
        ):
            for place_from_right in range(old_integer_places, required):
                digit_mob = source_digit.clone(spawn=False)
                digit_mob.set(opacity=0)
                slot_location = map_local_to_global_coords(
                    self.location,
                    self.basis,
                    self._rightmost_integer_location
                    - self._digit_advance * place_from_right,
                )
                for digit in digit_mob.character_mobs:
                    digit.location = slot_location
                digit_mob.on_create = lambda digit_mob=digit_mob: digit_mob
                new_digits_near_to_far.append(digit_mob)

        new_digits = list(reversed(new_digits_near_to_far))
        self.digit_mobs = (
            new_digits
            + self.digit_mobs[:old_integer_places]
            + self.digit_mobs[old_integer_places:]
        )
        self.integer_places = required
        self.add_children(new_digits)

        if self.is_spawned() and not self.is_despawned():
            for digit_mob in new_digits:
                digit_mob.spawn(animate=False)

    def update_display(self, value):
        value = cast_to_tensor(value)
        neg_opacity = torch.where((value < 0), 1, 0)
        value = value.abs()
        num_i, num_d = self.integer_places, self.decimal_places
        # Largest value the digit slots can show; anything bigger is clamped
        # (also protects against rounding carrying into a non-existent digit).
        limit = (10**num_i) - ((10**-num_d) if num_d > 0 else 1)
        visible_integer_places = []

        def get_opacities(value):
            all_opacities = []
            for v in value:
                x = float(v.item())
                if x != x:  # NaN guard
                    x = 0.0
                x = min(x, limit)
                value_string = f"{x:.{num_d}f}"
                int_part, _, frac_part = value_string.partition(".")
                visible_integer_places.append(len(int_part))
                int_part = int_part.rjust(num_i, "0")
                # Blank out leading zeros, always keeping the last integer digit.
                num_leading = min(len(int_part) - len(int_part.lstrip("0")), num_i - 1)
                for k, digit in enumerate(int_part + frac_part):
                    if k < num_leading:
                        all_opacities.append(torch.zeros(10, 1, dtype=torch.long))
                    else:
                        all_opacities.append(
                            F.one_hot(torch.tensor((int(digit),)), 10).transpose(0, 1)
                        )
            return torch.stack(all_opacities)

        all_opacities = torch.stack([get_opacities(_) for _ in value], -3)
        visible_integer_places = torch.tensor(
            visible_integer_places,
            dtype=value.dtype,
            device=value.device,
        ).reshape(value.shape)
        sign_location = map_local_to_global_coords(
            self.location,
            self.basis,
            self._rightmost_integer_location
            - self._digit_advance * (visible_integer_places - 1)
            + self._negative_sign_offset,
        )

        with Sync(animation_manager=self.animation_manager):
            if self.decimal is not None:
                self.decimal.opacity = 1
            self.negative_sign.set(
                location=sign_location,
                opacity=neg_opacity,
            )
            for i in range(len(self.digit_mobs)):
                for j in range(10):
                    self.digit_mobs[i].character_mobs[j].set(
                        opacity=all_opacities[i, :, j].unsqueeze(-2),
                    )
