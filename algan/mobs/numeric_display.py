from __future__ import annotations

import torch
import torch.nn.functional as F

from algan.animatable_base.animatable import animated_function
from algan.animation_timeline.animation_contexts import NoExtra, Off, Sync, Seq
from algan.animatable_base.mob import Mob
from algan.mobs.text import Tex
from algan.utils.tensor_utils import cast_to_tensor


class NumericDisplay(Mob):
    def __init__(self, value, num_decimal_places=2, num_integer_places=None, **kwargs):
        """An animated numeric counter with a fixed number of digit slots.

        Parameters
        ----------
        value
            The initial value displayed.
        num_decimal_places
            Number of digits after the decimal point. With 0, no decimal
            point is shown.
        num_integer_places
            Number of digits before the decimal point. If None, derived from
            the initial value (at least 1). Values beyond the digit capacity
            are clamped to the largest displayable value, so pass this
            explicitly for counters that grow (e.g. 4 to count up to 9999).
        """
        value = cast_to_tensor(value)
        self.num_decimal_places = num_decimal_places
        if num_integer_places is None:
            num_integer_places = max(
                1, len(str(int(abs(float(value.reshape(-1)[0])))))
            )
        self.num_integer_places = num_integer_places
        num_i, num_d = num_integer_places, num_decimal_places
        with Off(), NoExtra(priority_level=1):
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
            self.digit_mobs = []
            for _ in range(num_i + num_d):
                self.digit_mobs.append(Tex("0123456789", **kwargs))
                self.digit_mobs[-1].set(opacity=0)
            for i in range(len(self.digit_mobs)):
                l = self.placeholder[
                    1 + i + (1 if (num_d > 0 and i >= num_i) else 0)
                ].location
                for d in self.digit_mobs[i].character_mobs:
                    d.location = l
            super().__init__(**kwargs)
            self.register_attrs_as_animatable(["value"], NumericDisplay)
            self.setattr_without_record("value", value)
            self.update_display(self.value)
        if self.decimal is not None:
            self.add_children(self.digit_mobs, self.decimal, self.negative_sign)
        else:
            self.add_children(self.digit_mobs, self.negative_sign)
        for c in self.children:
            c.on_create = lambda c=c: c
        #self.components = [*self.digit_mobs, self.decimal, self.negative_sign]

    def on_create(self):
        with Sync():
            for c in self.get_descendants():
                o = c.opacity
                with Seq():
                    with Off():
                        c.set_non_recursive(opacity = 0)
                    c.set_non_recursive(opacity = o)

    def set_value(self, value):
        return self.change_value(value)

    def get_value(self):
        return self.value

    @animated_function(animated_args={"interpolation": 0})
    def change_value(self, value, interpolation=1):
        value = cast_to_tensor(value)
        old_value = self.value
        interpolated_value = old_value * (1 - interpolation) + interpolation * value
        self.update_display(interpolated_value)
        self.setattr_and_record_modification("value", value)
        return self

    def update_display(self, value):
        value = cast_to_tensor(value)
        neg_opacity = torch.where((value < 0), 1, 0)
        value = value.abs()
        num_i, num_d = self.num_integer_places, self.num_decimal_places
        # Largest value the digit slots can show; anything bigger is clamped
        # (also protects against rounding carrying into a non-existent digit).
        limit = (10 ** num_i) - ((10 ** -num_d) if num_d > 0 else 1)

        def get_opacities(value):
            all_opacities = []
            for v in value:
                x = float(v.item())
                if x != x:  # NaN guard
                    x = 0.0
                x = min(x, limit)
                value_string = f"{x:.{num_d}f}"
                int_part, _, frac_part = value_string.partition(".")
                int_part = int_part.rjust(num_i, "0")
                # Blank out leading zeros, always keeping the last integer digit.
                num_leading = min(
                    len(int_part) - len(int_part.lstrip("0")), num_i - 1
                )
                for k, digit in enumerate(int_part + frac_part):
                    if k < num_leading:
                        all_opacities.append(torch.zeros(10, 1, dtype=torch.long))
                    else:
                        all_opacities.append(
                            F.one_hot(torch.tensor((int(digit),)), 10).transpose(0, 1)
                        )
            return torch.stack(all_opacities)

        all_opacities = torch.stack([get_opacities(_) for _ in value], -3)

        with Sync():
            if self.decimal is not None:
                self.decimal.opacity = 1
            self.negative_sign.set(opacity=neg_opacity)
            for i in range(len(self.digit_mobs)):
                for j in range(10):
                    self.digit_mobs[i].character_mobs[j].set(
                        opacity=all_opacities[i, :, j].unsqueeze(-2),
                    )
