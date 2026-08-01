from __future__ import annotations

import manim as mn

from algan import *


def render_static_text():
    mobs = Group([ManimMob(mn.Text('a')) for _ in range(250)]).arrange_in_grid().scale(1/10).spawn()
    mobs.wait(2)


SETTINGS.computing.set(max_animation_batch_size=1000)
render_all_funcs(__name__, HD)
