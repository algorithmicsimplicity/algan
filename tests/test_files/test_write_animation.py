import manim as mn
from algan.constants.color import PURE_RED
from algan.animations.manim_animations import write
from algan.mobs.manim_mob import ManimMob
from algan.utils.algan_utils import render_all_funcs

def test_write_animation():
    x = ManimMob(mn.Text("Hello")).spawn()
    write(x)
    x.wait(0.5)
    write(x)
    x.wait(0.5)

if __name__ == "__main__":
    render_all_funcs(__name__)
