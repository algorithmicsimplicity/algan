import torch

from algan.constants.color import Color, BLACK
from algan.mobs.bezier_circuit import BezierCircuitCubic
from algan.mobs.image_mob import ImageMob
from algan.mobs.group import Group
from algan.utils.tensor_utils import unsquish
from algan.utils.mob_utils import batch_mobs
from algan.utils.lazy_import import LazyModule

# Deferred: a ManimMob wraps an already-constructed manim mobject, so manim
# is inevitably imported by the caller first; keeping it lazy here means
# ``import algan`` does not pay manim's ~2 s dependency chain. The svg-cache
# module patches manim, so it must ride along on the first load.
_manim = LazyModule("manim", extras=("algan.utils.manim_svg_cache",))


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

    def __init__(self, manim_mob, batch=False, _add_to_scene=None, **kwargs):
        manim_scale_factor = 1
        children = []
        orig_add_to_scene = _add_to_scene
        if _add_to_scene is None:
            _add_to_scene = not batch
        else:
            kwargs['add_to_scene'] = _add_to_scene
        for submob in manim_mob.submobjects:
            if isinstance(submob, _manim.ImageMobject):
                mob = ImageMob(submob, add_to_scene=_add_to_scene)
                children.append(mob)
                continue
            if submob.n_points_per_curve != 4 or submob.n_points_per_cubic_curve != 4:
                raise NotImplementedError(
                    "ManimMob does not support Mobjects which do not have n_points_per_curve == 4"
                )
            # if isinstance(submob, VectorizedPoint):# or isinstance(submob, ThreeDVMobject):
            #    continue
            children.append(ManimMob(submob, batch=False, _add_to_scene=_add_to_scene, **kwargs))

        empty = False
        if len(manim_mob.points) == 0:
            control_points = torch.from_numpy(manim_mob.get_center()).float().to(torch.get_default_device())
            control_points = torch.stack([control_points for _ in range(4)], -2)
            empty = True
        else:
            control_points = torch.from_numpy(manim_mob.points).to(torch.get_default_device())
            if len(control_points) == 1:
                control_points = control_points.expand(
                    *([-1] * (control_points.dim() - 2)), 4, -1
                )
                empty = True
            control_points = unsquish(control_points.float(), -2, 4)

        def convert_manim_color(manim_color, opacity):
            if manim_color is None:
                return BLACK
            rgba = manim_color.to_rgba()
            rgb = rgba[:3]
            a = rgba[-1]
            if opacity is not None:
                a = a * opacity
            return Color(rgb, glow=0, opacity=a)

        if orig_add_to_scene is not None:
            kwargs['add_to_scene'] = orig_add_to_scene

        super().__init__(
            control_points * manim_scale_factor,
            color=convert_manim_color(
                manim_mob.fill_color, opacity=manim_mob.fill_opacity
            ),
            opacity=1,
            border_color=convert_manim_color(
                manim_mob.stroke_color, manim_mob.stroke_opacity
            ),
            border_width=manim_mob.stroke_width / 2,
            filled=(not hasattr(manim_mob, "end"))
            and (manim_mob.fill_opacity is not None and manim_mob.fill_opacity > 1e-5),
            empty=empty,
            **kwargs,
        )
        self.singleton_batch_indexing = True
        if len(children) > 0:
            if 'add_to_scene' not in kwargs:
                add_to_scene = True
            else:
                add_to_scene = kwargs['add_to_scene']
            self.add_children(batch_mobs(children, add_to_scene=add_to_scene) if batch else Group(children, add_to_scene=add_to_scene))
        self.submobjects = children
