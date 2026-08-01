import manim as mn

from algan.animations.manim_animations import draw_border_then_fill
from algan.mobs.manim_mob import ManimMob
from algan.utils.algan_utils import render_all_funcs


def test_write_animation():
    x = ManimMob(mn.Text("Hello")).spawn()
    draw_border_then_fill(x.children[2])
    x.wait(0.5)
    draw_border_then_fill(x.children[2])
    x.wait(0.5)


if __name__ == "__main__":
    render_all_funcs(__name__)
