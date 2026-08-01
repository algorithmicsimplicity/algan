"""Render coverage for :class:`~.Paragraph` and :class:`~.Code`.

Both are Groups whose text lines carry all the geometry, and both used to build
those lines detached from the Scene.  Because only registered actors reach the
renderer, every frame came out empty.  These scenes keep them covered.
"""

from algan.animation_timeline.animation_contexts import Off
from algan.constants.spatial import *  # RIGHT, LEFT, UP, DOWN, OUT
from algan.mobs.text import Code, Paragraph
from algan.utils.algan_utils import render_all_funcs


def test_paragraph_lines():
    paragraph = Paragraph("hello", "world", alignment="left").scale(0.7).spawn()
    paragraph.move(UP * 0.3)


def test_code_with_window_background():
    with Off():
        code = (
            Code(
                code_string="x = 1\ny = x + 2",
                background="window",
            )
            .scale(0.5)
            .spawn()
        )
    code.wait(0.5)


render_all_funcs(__name__)
