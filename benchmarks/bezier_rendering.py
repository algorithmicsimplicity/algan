from __future__ import annotations

import manim as mn

from algan import *


def render_static_beziers():
    mobs = ManimMob(mn.Text("abcdefir\nsbmbbkl")).scale(2).spawn()
    mobs.wait(2)


render_all_funcs(__name__, HD)
