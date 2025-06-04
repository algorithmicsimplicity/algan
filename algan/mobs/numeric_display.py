from collections import defaultdict

import torch
import torch.nn.functional as F

from algan.animation.animatable import animated_function
from algan.animation.animation_contexts import Off, Sync, Seq, NoExtra
from algan.mobs.mob import Mob
from algan.mobs.text import Tex
from algan.utils.tensor_utils import cast_to_tensor


class NumericDisplay(Mob):
    def __init__(self, value, num_digits=2, **kwargs):
        value = cast_to_tensor(value)
        self.num_digits = num_digits
        with Off():
            with NoExtra(priority_level=1):
                self.placeholder = Tex(f'0.{"".join(["0" for _ in range(num_digits)])}', create=False, init=False, **kwargs)
                self.placeholder.spawn(animate=False)
                ct = self.placeholder.animation_manager.context.current_time
                self.placeholder.animation_manager.context.rewind(1 / self.placeholder.scene.frames_per_second + 1e-3)
                self.placeholder.opacity = 0
                self.placeholder.animation_manager.context.current_time = ct

                self.decimal = self.placeholder[1]
                self.digit_mobs = []
                for _ in range(num_digits+1):
                    self.digit_mobs.append(Tex('0123456789', create=False, init=False, **kwargs).spawn(animate=False))
                    ct = self.placeholder.animation_manager.context.current_time
                    self.placeholder.animation_manager.context.rewind(1 / self.placeholder.scene.frames_per_second + 1e-3)
                    self.digit_mobs[-1].set(opacity=0)
                    self.placeholder.animation_manager.context.current_time = ct
                for i in range(len(self.digit_mobs)):
                    self.digit_mobs[i].character_mobs.location = self.placeholder[i+1 if i > 0 else 0].location
                self._value = value
                self.update_display(self.value)
                kwargs2 = {k: v for k, v in kwargs.items()}
                kwargs2['create'] = False
                kwargs2['init'] = False
                self.animatable_attrs = {'value'}
                super().__init__(**kwargs2)
        self.add_children(self.digit_mobs, self.decimal)
        if not ('init' in kwargs and not kwargs['init'] or self.animation_manager.context.delay_init):
            self.init()
        if not ('create' in kwargs and not kwargs['create'] or self.animation_manager.context.delay_creation):
            self.spawn()

    def spawn(self, animate=True):
        if self.time_inds.created:
            return self
        with Off():
            super().spawn(animate=False)
            ct = self.placeholder.animation_manager.context.current_time
            self.placeholder.animation_manager.context.rewind(1 / self.placeholder.scene.frames_per_second + 1e-3)
            self.opacity = 0
            self.placeholder.animation_manager.context.current_time = ct
            #self.setattr_non_recursive('opacity', 1)
        self.placeholder.animation_manager.context.current_time = ct + 1e-3
        with Sync():
            self.update_display(self.value)
            self.decimal.opacity = 1
            #self.decimal.opacity = 1
        return self

    def destroy2(self, animate=True):
        if self.time_inds.destroyed:
            return self
        with Seq():
            with Sync():
                self.opacity = 0
            #self.skip()
            with Off():
                super().despawn(animate=False)
        return self

    @property
    def value(self):
        return self._value

    @animated_function(animated_args={'interpolation': 0})
    def change_value(self, value, interpolation=1):
        value = cast_to_tensor(value)
        old_value = self.value
        try:
            interpolated_value = old_value * (1 - interpolation) + interpolation * value
        except RuntimeError:
            interpolated_value = old_value * (1 - interpolation) + interpolation * value
        # interpolated_value = self._data_dict()[bkey][self.time_inds_active if not self.time_inds_materialized is None else slice(None)]
        self.update_display(interpolated_value)
        #self.value = interpolated_value
        #self.__setattr__('value', value)
        self.setattr_and_record_modification('_value', value)
        return self

    def update_display(self, value):
        value = cast_to_tensor(value)
        #with Off():
        #    self.placeholder.opacity = 0
        all_opacities = defaultdict(list)
        for v in value:
            value_string = f'{v.item():.{self.num_digits}f}'
            value_digits = [value_string[0], *value_string[2:]]
            for i, digit in enumerate(value_digits):
                digit = int(digit)
                all_opacities[i].append(F.one_hot(torch.tensor((digit,)), 10).transpose(0,1))

        def prep(mob):
            if hasattr(self, 'time_inds') and mob.time_inds.time_inds_materialized is None and self.time_inds.time_inds_materialized is not None:
                mob.time_inds.original_animatable.set_state_full(self.time_inds.time_inds_materialized.amin(), self.time_inds.time_inds_materialized.amax() + 1)
                mob.time_inds.time_inds_active = self.time_inds.time_inds_active
            return mob

        for i in range(len(self.digit_mobs)):
            prep(self.digit_mobs[i].character_mobs).opacity = torch.stack(all_opacities[i])
