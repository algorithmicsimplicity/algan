from __future__ import annotations

import os

# Benchmarks must never be measured inside a warm daemon: it keeps adaptive
# renderer state (the memory model's batch-size fit) across runs, so one
# benchmark would be timed against whatever ran before it.
os.environ.setdefault("ALGAN_USE_DAEMON", "0")

from algan import *


def render_static_triangulated_text():
    with Off():
        mobs = (
            Text("abcdefir\nsbmbbkl\nmbnmcllc\nqwereqtqet")
            .set(border_color=RED, border_width=6)
            .scale(2)
            .spawn()
        )
    mobs.wait(100)


set_log_level("DEBUG")
q = PREVIEW
q.super_sampling_anti_aliasing = 1
render_all_funcs(__name__, q)
