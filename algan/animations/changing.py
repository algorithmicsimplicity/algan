"""Animations whose geometry changes continuously for an indefinite duration."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from algan.animatable_base.mob import Mob
from algan.animation_timeline.animation_contexts import Off
from algan.constants import easings
from algan.constants.color import BLUE_B, BLUE_D, BLUE_E, GREY_BROWN, Color
from algan.mobs.bezier_circuit import BezierCircuitCubic
from algan.mobs.group import Group
from algan.utils.tensor_utils import cast_to_tensor


def _bezier_family(mobject: Mob) -> list[BezierCircuitCubic]:
    return [
        descendant
        for descendant in mobject.get_descendants()
        if isinstance(descendant, BezierCircuitCubic) and not descendant.empty
    ]


def _color_rows(colors: Sequence, reference: torch.Tensor) -> torch.Tensor:
    rows = []
    for color in colors:
        if isinstance(color, str):
            value = Color(color)
        elif hasattr(color, "to_rgb"):
            value = cast_to_tensor(color.to_rgb())
        else:
            value = cast_to_tensor(color)
        value = Color.add_defaults(value).as_subclass(torch.Tensor)
        rows.append(value.reshape(-1, value.shape[-1])[0].to(reference))
    return torch.stack(rows)


def _animated_boundary_update(boundary, elapsed):
    elapsed = cast_to_tensor(elapsed)
    frame_count = elapsed.shape[0]
    cycle_time = elapsed.reshape(frame_count, 1, 1) * boundary.cycle_rate
    cycle_index = torch.floor(cycle_time).to(torch.long)
    alpha = cycle_time - torch.floor(cycle_time)
    draw_alpha = boundary.draw_easing(alpha)
    fade_alpha = boundary.fade_easing(alpha)

    colors = _color_rows(boundary.colors, elapsed)
    growing_color = colors[(cycle_index.reshape(-1) % len(colors))].view(
        frame_count, 1, -1
    )
    fading_color = colors[((cycle_index.reshape(-1) - 1) % len(colors))].view(
        frame_count, 1, -1
    )

    if boundary.back_and_forth:
        reverse = (cycle_index % 2) == 1
        lower = torch.where(reverse, 1.0 - draw_alpha, torch.zeros_like(draw_alpha))
        upper = torch.where(reverse, torch.ones_like(draw_alpha), draw_alpha)
    else:
        lower = torch.zeros_like(draw_alpha)
        upper = draw_alpha

    fade_width = torch.where(
        cycle_time >= 1.0,
        (1.0 - fade_alpha) * boundary.max_stroke_width,
        torch.zeros_like(fade_alpha),
    )

    for source, growing, fading in zip(
        boundary._source_paths,
        boundary._growing_paths,
        boundary._fading_paths,
    ):
        full_points = source.control_points.location
        if full_points.shape[0] != frame_count:
            full_points = full_points.expand(frame_count, -1, -1)
        growing.set_control_points_to_partial(full_points, lower, upper)
        fading.set_control_points_to_partial(
            full_points, torch.zeros_like(draw_alpha), torch.ones_like(draw_alpha)
        )
        growing.stroke_color = growing_color
        fading.stroke_color = fading_color
        growing.stroke_width = torch.full_like(
            draw_alpha, float(boundary.max_stroke_width)
        )
        fading.stroke_width = fade_width
    return boundary


class AnimatedBoundary(Group):
    """A cycling highlight that repeatedly travels around a Bezier Mob.

    This is one of the few Manim animation helpers that is not reduced to a
    simple Algan context composition: it is an indefinite, absolute-time
    updater which keeps following the source Mob even while the source changes
    shape.

    Parameters mirror Manim Community's ``AnimatedBoundary`` where practical.
    The returned object is an ordinary Algan :class:`~algan.mobs.group.Group`;
    call ``spawn()`` to display it and :meth:`stop` to freeze the updater.
    """

    def __init__(
        self,
        vmobject: Mob,
        colors: Sequence = (BLUE_D, BLUE_B, BLUE_E, GREY_BROWN),
        max_stroke_width: float = 3,
        cycle_rate: float = 0.5,
        back_and_forth: bool = True,
        draw_easing=easings.smooth,
        fade_easing=easings.smooth,
        **kwargs,
    ):
        if not isinstance(vmobject, Mob):
            raise TypeError("AnimatedBoundary expects an Algan Mob.")
        if not colors:
            raise ValueError("colors must contain at least one color")
        source_paths = _bezier_family(vmobject)
        if not source_paths:
            raise TypeError("AnimatedBoundary requires cubic Bezier geometry.")

        self.vmobject = vmobject
        self.colors = tuple(colors)
        # Algan's own units, like every other stroke width in the root
        # namespace. Manim's ``AnimatedBoundary`` means twice this by the same
        # number; ``algan.manim`` is where that conversion lives.
        self.max_stroke_width = float(max_stroke_width)
        self.cycle_rate = float(cycle_rate)
        self.back_and_forth = bool(back_and_forth)
        self.draw_easing = draw_easing
        self.fade_easing = fade_easing

        with Off(animation_manager=vmobject.animation_manager):
            # The two layers ARE the boundary's visible geometry, so they have to
            # be Scene actors: Algan builds render primitives from the Scene's
            # actor list, not by walking this Group's hierarchy. They are cloned
            # unspawned so that spawning the Group is what puts them on screen.
            growing_copy = vmobject.clone(spawn=False)
            fading_copy = vmobject.clone(spawn=False)
            growing_paths = _bezier_family(growing_copy)
            fading_paths = _bezier_family(fading_copy)
            if not (len(source_paths) == len(growing_paths) == len(fading_paths)):
                raise RuntimeError(
                    "AnimatedBoundary clone hierarchy did not match its source."
                )
            for path in [*growing_paths, *fading_paths]:
                path.color = path.color.as_subclass(Color).set_opacity(0.0)
                path.stroke_width = 0.0
            super().__init__(growing_copy, fading_copy, **kwargs)

        self.boundary_copies = (growing_copy, fading_copy)
        self._source_paths = source_paths
        self._growing_paths = growing_paths
        self._fading_paths = fading_paths
        self.updater_id = self.add_updater(_animated_boundary_update)

    def stop(self):
        """Remove the boundary updater, leaving its current appearance fixed."""
        self.remove_updater(self.updater_id)
        return self


__all__ = ["AnimatedBoundary"]
