"""Render coverage for :class:`~.AnimatedBoundary`.

The boundary's two layers are clones of the source Mob, and they used to be
built detached from the Scene.  Since Algan renders registered actors rather
than walking the Group hierarchy, that made the travelling highlight invisible
at every stroke width -- and, over a borderless source, produced completely
empty frames.  Rendering it here turns a regression into a pixel diff.
"""

from algan.animation_timeline.animation_contexts import Off
from algan.animations.changing import AnimatedBoundary
from algan.constants.color import BLUE, TRANSPARENT
from algan.mobs.shapes_2d import Square
from algan.utils.algan_utils import render_all_funcs


def test_animated_boundary_over_borderless_source():
    # The source contributes no pixels of its own, so every visible pixel comes
    # from the boundary layers.
    with Off():
        square = Square(color=TRANSPARENT, border_width=0).scale(1.5).spawn()
        AnimatedBoundary(square, max_stroke_width=14, cycle_rate=1.0).spawn()
    square.wait(2)


def test_animated_boundary_over_visible_source():
    with Off():
        square = Square(color=BLUE).scale(1.2).spawn()
        AnimatedBoundary(square, max_stroke_width=6, cycle_rate=0.5).spawn()
    square.wait(2)


render_all_funcs(__name__)
