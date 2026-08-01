from __future__ import annotations

import torch

from algan.animation_timeline.animation_contexts import active_scene_for_new_mob
from algan.constants.color import BLACK, Color
from algan.mobs.bezier_circuit import BezierCircuitCubic
from algan.mobs.group import Group
from algan.mobs.image_mob import ImageMob
from algan.utils.lazy_import import LazyModule
from algan.utils.mob_utils import batch_mobs
from algan.utils.tensor_utils import unsquish

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
        if kwargs.get("scene") is None:
            kwargs["scene"] = active_scene_for_new_mob()
        # Retain the source object so compatibility Mobs can delegate Manim-specific
        # query/build methods (for example Axes.plot and NumberLine.n2p) and can
        # resynchronise their converted geometry after a delegated mutation.
        self.manim_mobject = manim_mob
        manim_scale_factor = 1
        children = []
        # One Manim Mobject converts to a whole Algan sub-hierarchy, and Algan
        # collects render primitives from the Scene's actor list rather than by
        # walking the hierarchy, so every renderable part has to be registered
        # in its own right: ``add_to_scene`` governs the entire converted
        # subtree.  ``_add_to_scene`` carries that resolved decision down the
        # recursion.  Batching is the one case where the children stay
        # unregistered, because the single batched Mob renders in their place.
        if _add_to_scene is None:
            _add_to_scene = bool(kwargs.get("add_to_scene", True))
        kwargs["add_to_scene"] = _add_to_scene
        child_add_to_scene = _add_to_scene and not batch
        for submob in manim_mob.submobjects:
            if isinstance(submob, _manim.ImageMobject):
                mob = ImageMob(
                    submob,
                    scene=kwargs["scene"],
                    add_to_scene=child_add_to_scene,
                )
                children.append(mob)
                continue
            if submob.n_points_per_curve != 4 or submob.n_points_per_cubic_curve != 4:
                raise NotImplementedError(
                    "ManimMob does not support Mobjects which do not have n_points_per_curve == 4"
                )
            # if isinstance(submob, VectorizedPoint):# or isinstance(submob, ThreeDVMobject):
            #    continue
            children.append(
                ManimMob(
                    submob, batch=False, _add_to_scene=child_add_to_scene, **kwargs
                )
            )

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

        fill_opacity = getattr(manim_mob, "fill_opacity", None)
        stroke_opacity = getattr(manim_mob, "stroke_opacity", None)
        stroke_width = getattr(manim_mob, "stroke_width", 0)
        if stroke_width is None:
            stroke_width = 0

        has_visible_fill = fill_opacity is not None and bool(
            torch.as_tensor(fill_opacity).max().item() > 1e-5
        )

        super().__init__(
            control_points * manim_scale_factor,
            color=convert_manim_color(
                manim_mob.fill_color, opacity=fill_opacity
            ),
            opacity=1,
            border_color=convert_manim_color(
                manim_mob.stroke_color, stroke_opacity
            ),
            border_width=stroke_width / 2,
            filled=(not hasattr(manim_mob, "end"))
            and has_visible_fill,
            empty=empty,
            **kwargs,
        )
        self.singleton_batch_indexing = True
        if len(children) > 0:
            grouped = (
                batch_mobs(children, add_to_scene=_add_to_scene)
                if batch
                else Group(
                    children,
                    scene=self.scene,
                    add_to_scene=_add_to_scene,
                )
            )
            self.add_children(grouped)
        self.submobjects = children
