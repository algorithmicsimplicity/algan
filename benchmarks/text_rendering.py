from __future__ import annotations

import os

import manim as mn

# Benchmarks must never be measured inside a warm daemon: it keeps adaptive
# renderer state (the memory model's batch-size fit) across runs, so one
# benchmark would be timed against whatever ran before it.
os.environ.setdefault("ALGAN_USE_DAEMON", "0")

from algan import *  # noqa: E402


def render_static_text():
    mobs = (
        Group([ManimMob(mn.Text("a")) for _ in range(250)])
        .arrange_in_grid()
        .scale(1 / 10)
        .spawn()
    )
    mobs.wait(2)


SETTINGS.computing.set(max_animation_batch_size=1000)
render_all_funcs(__name__, HD)
