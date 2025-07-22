from __future__ import annotations

from algan import *


def render_static_triangles():
    Group([Sphere() for _ in range(9)]).arrange_in_grid().spawn()
    # mobs.wait(2)


render_all_funcs(__name__, PREVIEW)
