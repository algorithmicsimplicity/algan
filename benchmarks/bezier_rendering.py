from algan import *
import manim as mn


def render_static_beziers():
    with Off():
        mobs = ManimMob(mn.Text("abcdefir\nsbmbbkl\nmbnmcllc\nqwereqtqet", stroke_color=mn.RED, stroke_width=6)).scale(2).spawn()
    mobs.wait(100)


LOGGING_DEFAULTS.verbosity = 'max'
q = PREVIEW
q.anti_alias_level = 1
render_all_funcs(__name__, q)
