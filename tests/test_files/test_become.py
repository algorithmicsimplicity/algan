from algan.mobs.shapes_2d import Circle, Square, Polygon, RegularPolygon
from algan.mobs.text import Tex
from algan.utils.algan_utils import render_all_funcs


def test_become_shapes_2d():
    x = Circle().spawn()
    x = x.become(RegularPolygon(10))
    x.wait()
    x = x.become(Square())
    x.wait()


def test_become_text():
    x = Tex("Hello", font_size=90).spawn()
    # minimize_movement pairs each triangle with the closest triangle in the target,
    # so the glyphs deform smoothly in place instead of breaking apart and reforming.
    x.become(Tex("Holloo", font_size=90), minimize_movement=True).become(
        Tex("Hollo", font_size=90), minimize_movement=True
    ).become(Tex("Hllo", font_size=90), minimize_movement=True).become(
        Tex("World", font_size=90), minimize_movement=True
    )


render_all_funcs(__name__)
