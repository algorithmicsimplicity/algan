import torch

from algan.constants.color import Color
from algan.mobs.bezier_circuit import BezierCircuitCubic
from algan.mobs.image_mob import ImageMob
from algan.mobs.group import Group
from algan.utils.tensor_utils import unsquish
from manim import ImageMobject, VectorizedPoint


class ManimMob(BezierCircuitCubic):
    """Constructs an equivalent Algan Mob from a given Manim Mobject.

    Parameters
    ----------
    manim_mob : manim.Mobject
        The Manim Mobject which will be converted into an Algan Mob. It must be
        a bezier-circuit based object.
    **kwargs
        Passed to :class:`~.BezierCircuitCubic` .

    """
    def __init__(self, manim_mob, **kwargs):
        manim_scale_factor = 1
        children = []
        for submob in manim_mob.submobjects:
            if isinstance(submob, ImageMobject):
                mob = ImageMob(submob)
                children.append(mob)
                continue
            if submob.n_points_per_curve != 4 or submob.n_points_per_cubic_curve != 4:
                raise NotImplementedError('ManimMob does not support Mobjects which do not have n_points_per_curve == 4')
            children.append(ManimMob(submob, **kwargs))

        empty = False
        if len(manim_mob.points) == 0:
            control_points = torch.from_numpy(manim_mob.get_center()).float()
            control_points = torch.stack([control_points for _ in range(4)], -2)
            empty = True
        else:
            control_points = torch.from_numpy(manim_mob.points)
            if len(control_points) == 1:
                control_points = control_points.expand(*([-1] * (control_points.dim() - 2)), 4, -1)
                empty = True
            control_points = unsquish(control_points.float(), -2, 4)

        def convert_manim_color(manim_color, opacity):
            rgba = manim_color.to_rgba()
            rgb = rgba[:3]
            a = rgba[-1]
            if opacity is not None:
                a = a * opacity
            return Color(rgb, glow=0, opacity=a)
        super().__init__(control_points * manim_scale_factor, color=convert_manim_color(manim_mob.fill_color, opacity=manim_mob.fill_opacity), opacity=1,
                         border_color=convert_manim_color(manim_mob.stroke_color, manim_mob.stroke_opacity),
                         border_width=manim_mob.stroke_width,
                         filled=(not hasattr(manim_mob, 'end')) and (manim_mob.fill_opacity is not None and manim_mob.fill_opacity > 1e-5),
                         empty=empty, **kwargs)
        if len(children) > 0:
            self.add_children(Group(children))
        self.submobjects = children
