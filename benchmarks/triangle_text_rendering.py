from algan import *
import manim as mn


def render_static_triangulated_text():
    with Off():
        mobs = Text("abcdefir\nsbmbbkl\nmbnmcllc\nqwereqtqet").set(border_color=RED, border_width=6).scale(2).spawn()
    mobs.wait(100)


set_log_level('DEBUG')
q = PREVIEW
q.anti_alias_level = 1
render_all_funcs(__name__, q)
