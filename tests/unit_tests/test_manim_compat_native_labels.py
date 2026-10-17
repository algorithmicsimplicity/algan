"""Native text handed to a Manim-backed Mob keeps its glyphs.

Manim draws a label as it is handed one: ``Axes.get_axis_labels``,
``get_graph_label`` and ``LabeledDot`` place the Mobject they are given and
draw it. A native :class:`~algan.mobs.text.Tex` or
:class:`~algan.mobs.text.Text` used to reach Manim as its bounding box, so the
label drew as a rectangle where the text should have been.
"""

import manim as mn
import numpy as np
import pytest

from algan import RED, Axes, LabeledDot, Scene, Tex, Text, UnsupportedFeatureWarning
from algan.mobs.manim_compat import to_manim


def _glyph_points(text):
    return [
        glyph.control_points.location[0].detach().cpu().numpy()
        for glyph in text.character_mobs
    ]


def test_a_tex_reaches_manim_as_one_outline_per_glyph():
    with Scene():
        tex = Tex("ab", "c", color=RED, add_to_scene=False)
        converted = to_manim(tex)

        assert isinstance(converted, mn.VGroup)
        assert len(converted) == 3
        for outline, points in zip(converted, _glyph_points(tex)):
            np.testing.assert_allclose(outline.points, points)
            assert outline.get_fill_color().to_hex() == to_manim(RED)
            assert outline.get_fill_opacity() == 1


def test_axis_labels_given_as_native_text_draw_the_text():
    with Scene():
        ax = Axes(x_range=[0, 5], y_range=[0, 5])
        x_label = Tex("ab", add_to_scene=False)
        y_label = Text("y", color=RED, add_to_scene=False)
        labels = ax.get_axis_labels(x_label=x_label, y_label=y_label)

        drawn_x, drawn_y = labels.manim_mobject
        assert [len(drawn_x), len(drawn_y)] == [2, 1]
        # Placing a label moves it whole, so each glyph keeps its shape.
        for drawn, label in ((drawn_x, x_label), (drawn_y, y_label)):
            for outline, points in zip(drawn, _glyph_points(label)):
                assert outline.points.shape == points.shape
                offset = outline.points - points
                np.testing.assert_allclose(offset, offset[:1].repeat(len(offset), 0))
        assert drawn_y[0].get_fill_color().to_hex() == to_manim(RED)


def test_a_labeled_dot_draws_a_native_tex_label():
    with Scene():
        label = Tex("A", add_to_scene=False)
        dot = LabeledDot(label=label)

        drawn_label = dot.manim_mobject.submobjects[0]
        assert isinstance(drawn_label, mn.VGroup)
        assert len(drawn_label) == 1
        assert drawn_label[0].points.shape == _glyph_points(label)[0].shape


def test_text_image_characters_are_left_out_with_a_warning():
    with Scene():
        text = Text("Hi \N{WAVING HAND SIGN}", add_to_scene=False)
        if not text.image_mobs:
            pytest.skip("this font setup draws the emoji as an outline")
        with pytest.warns(UnsupportedFeatureWarning, match="emoji"):
            converted = to_manim(text)

        assert len(converted) == len(text.character_mobs)
