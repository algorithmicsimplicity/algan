import torch
from manim import Mobject

from algan.mobs.bezier_circuit import BezierCircuitCubic
from algan.mobs.group import Group
from algan.utils.tensor_utils import unsquish


class ManimMob(BezierCircuitCubic):
    """Constructs an equivalent Algan Mob from a given Manim Mobject.

    Parameters
    ----------
    manim_mob
        The Manim Mobject which will be converted into an Algan Mob. It must be
        a bezier-circuit based object.
    **kwargs
        Passed to :class:`~.BezierCircuitCubic` .

    """
    def __init__(self, manim_mob:Mobject, **kwargs):
        manim_scale_factor = 1
        children = []
        for submob in manim_mob.submobjects:
            if submob.n_points_per_curve != 4 or submob.n_points_per_cubic_curve != 4:
                raise NotImplementedError('ManimMob does not support Mobjects which do not have n_points_per_curve == 4')
            children.append(ManimMob(submob))

        if len(manim_mob.points) == 0:
            control_points = torch.from_numpy(manim_mob.get_center()).float()
            control_points = torch.stack([control_points for _ in range(4)], -2)
        else:
            control_points = unsquish(torch.from_numpy(manim_mob.points).float(), -2, 4)

        def convert_manim_color(manim_color, opacity):
            c = torch.from_numpy(manim_color.to_rgba()).float()
            if opacity is not None:
                c[-1] *= opacity
            return torch.cat((c[:-1], torch.tensor((0,)), c[-1:]))
        super().__init__(control_points * manim_scale_factor, color=convert_manim_color(manim_mob.fill_color, opacity=manim_mob.fill_opacity),
                         border_color=convert_manim_color(manim_mob.stroke_color, manim_mob.stroke_opacity),
                         border_width=manim_mob.stroke_width,
                         filled=not hasattr(manim_mob, 'end'), **kwargs)
        if len(children) > 0:
            self.add_children(Group(children))
        self.submobjects = children
