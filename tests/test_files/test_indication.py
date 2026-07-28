import math

from algan.animations.indication import (
    ApplyWave,
    Blink,
    Circumscribe,
    Flash,
    FocusOn,
    Indicate,
    ShowPassingFlash,
    ShowPassingFlashWithThinningStrokeWidth,
    Wiggle,
)
from algan.constants.color import BLUE, PURE_GREEN, PURE_RED, YELLOW
from algan.constants.spatial import RIGHT, UP
from algan.mobs.shapes_2d import Circle, RegularPolygon, Square
from algan.utils.algan_utils import render_all_funcs


def test_indicate():
    x = Circle(radius=0.5, color=BLUE).spawn()
    Indicate(x)
    x.wait(0.2)


def test_wiggle():
    x = Square(side_length=0.8, color=PURE_RED).spawn()
    Wiggle(x, scale_value=1.2, rotation_angle=0.05 * math.pi * 2, n_wiggles=4)
    x.wait(0.2)


def test_blink():
    x = RegularPolygon(3, color=PURE_GREEN).scale(0.5).spawn()
    Blink(x, time_on=0.2, time_off=0.2, blinks=2)
    x.wait(0.2)


def test_focus_on():
    x = Circle(radius=0.1, color=YELLOW, location=RIGHT).spawn()
    FocusOn(x, run_time=1.0)
    x.wait(0.2)


def test_show_passing_flash():
    x = Circle(
        radius=0.6, border_color=PURE_GREEN, border_width=4, filled=False
    ).spawn()
    ShowPassingFlash(x, time_width=0.2, run_time=1.0)
    x.wait(0.2)


def test_show_passing_flash_thinning():
    x = Square(side_length=1.0, border_color=BLUE, border_width=6, filled=False).spawn()
    ShowPassingFlashWithThinningStrokeWidth(
        x, n_segments=5, time_width=0.3, run_time=1.0
    )
    x.wait(0.2)


def test_flash():
    x = Circle(radius=0.1, color=YELLOW).spawn()
    Flash(x, line_length=0.3, num_lines=8, flash_radius=0.2, run_time=1.0)
    x.wait(0.2)


def test_circumscribe():
    x = RegularPolygon(3, color=BLUE).scale(0.6).spawn()
    Circumscribe(x, shape=Square, buff=0.1, run_time=1.0)
    x.wait(0.2)
    Circumscribe(x, shape=Circle, buff=0.1, fade_in=True, fade_out=True, run_time=1.0)
    x.wait(0.2)
    Circumscribe(x, shape=Circle, buff=0.1, fade_in=False, fade_out=True, run_time=1.0)
    x.wait(0.2)


def test_apply_wave():
    x = Square(side_length=1.0, color=PURE_RED).spawn()
    ApplyWave(x, direction=UP, amplitude=0.3, ripples=2, run_time=1.5)
    x.wait(0.2)


if __name__ == "__main__":
    render_all_funcs(__name__)
