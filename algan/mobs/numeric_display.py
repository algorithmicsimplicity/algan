from collections import defaultdict

import torch
import torch.nn.functional as F

from algan.animation.animatable import animated_function
from algan.animation.animation_contexts import Off, Sync, Seq, NoExtra
from algan.mobs.mob import Mob
from algan.mobs.text import Tex
from algan.utils.tensor_utils import cast_to_tensor


class NumericDisplay(Mob):
    def __init__(self, value, num_decimal_places=2, **kwargs):
        value = cast_to_tensor(value)
        self.num_decimal_places = num_decimal_places
        with Off():
            with NoExtra(priority_level=1):
                self.placeholder = Tex(
                    f"-0.{''.join(['0' for _ in range(num_decimal_places)])}", **kwargs
                )
                ct = self.placeholder.animation_manager.context.current_time
                self.placeholder.animation_manager.context.rewind(
                    1 / self.placeholder.scene.frames_per_second + 1e-3
                )
                self.placeholder.opacity = 0
                self.placeholder.animation_manager.context.current_time = ct

                self.decimal = self.placeholder[2]
                self.negative_sign = self.placeholder[0]
                self.digit_mobs = []
                for _ in range(num_decimal_places + 1):
                    self.digit_mobs.append(Tex("0123456789", **kwargs))
                    # ct = self.placeholder.animation_manager.context.current_time
                    # self.placeholder.animation_manager.context.rewind(1 / self.placeholder.scene.frames_per_second + 1e-3)
                    self.digit_mobs[-1].set(opacity=0, max_opacity=0)
                    # self.placeholder.animation_manager.context.current_time = ct
                for i in range(len(self.digit_mobs)):
                    self.digit_mobs[i].character_mobs.location = self.placeholder[
                        i + 2 if i > 0 else 1
                    ].location
                self._value = value
                super().__init__(**kwargs)
                self.update_display(self.value)
                self.register_attrs_as_animatable(["value"], NumericDisplay)
        self.add_children(self.digit_mobs, self.decimal, self.negative_sign)
        #self.components = [*self.digit_mobs, self.decimal, self.negative_sign]

    @property
    def value(self):
        return self._value

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
        self.setattr_and_record_modification("_value", value)
        return self

    def update_display(self, value):
        value = cast_to_tensor(value)
        neg_opacity = torch.where((value < 0), 1, 0)
        value = value.abs()

        def get_opacities(value):
            all_opacities = []
            for v in value:
                value_string = f"{v.item():.{self.num_decimal_places}f}"
                value_digits = [value_string[0], *value_string[2:]]
                for i, digit in enumerate(value_digits):
                    digit = int(digit)
                    all_opacities.append(
                        F.one_hot(torch.tensor((digit,)), 10).transpose(0, 1)
                    )
            return torch.stack(all_opacities)

        all_opacities = torch.stack([get_opacities(_) for _ in value], -3)

        def prep(mob):
            mob.set_time_inds_to(self)
            return mob

        with Sync():
            self.negative_sign.set_time_inds_to(self)
            self.negative_sign.set(opacity=neg_opacity, max_opacity=neg_opacity)
            for i in range(len(self.digit_mobs)):
                self.digit_mobs[i].set_time_inds_to(self)
                prep(self.digit_mobs[i].character_mobs).set(
                    opacity=all_opacities[i], max_opacity=all_opacities[i]
                )
