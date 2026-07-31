"""Render coverage for the native ``Axes(...).plot(...)`` idiom.

Mobs handed back by a delegated Manim method used to be built without being
registered as actors on the owning Scene, so a plotted curve was silently absent
from the render while the axes around it drew fine.  These scenes render the
returned geometry, so a regression shows up as missing pixels rather than as a
quiet no-op.
"""
import numpy as np

from algan.animation_timeline.animation_contexts import Off, Sync
from algan.constants.color import BLUE, YELLOW
from algan.constants.spatial import *  # RIGHT, LEFT, IN, OUT, ORIGIN, UP
from algan.mobs.manim_compat import Axes
from algan.utils.algan_utils import render_all_funcs


def test_axes_plot_curve():
    axes = Axes(
        x_range=(-3, 3, 1),
        y_range=(-1.5, 1.5, 0.5),
        x_length=8,
        y_length=4,
    )
    graph = axes.plot(lambda x: np.sin(x), color=YELLOW)
    with Off():
        axes.spawn()
        graph.spawn()
    graph.wait()


def test_axes_plot_two_curves_and_animate():
    axes = Axes(
        x_range=(-3, 3, 1),
        y_range=(-1.5, 1.5, 0.5),
        x_length=7,
        y_length=3.5,
    )
    sine = axes.plot(lambda x: np.sin(x), color=YELLOW)
    cosine = axes.plot(lambda x: np.cos(x), color=BLUE)
    with Off():
        axes.spawn()
        sine.spawn()
        cosine.spawn()
    # A returned Mob has to be animatable as well as renderable: both curves
    # own timeline rows of their own, so they animate independently.
    with Sync():
        sine.color = BLUE
        cosine.color = YELLOW


render_all_funcs(__name__)
