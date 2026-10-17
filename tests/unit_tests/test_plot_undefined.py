"""A function plotted where it is undefined leaves a gap.

Manim samples a plotted function across the whole axis, so ``ax.plot(np.sqrt)``
on axes that run through negative numbers produces NaN points, which Manim's
renderer leaves undrawn. Converting them crashed with ``IndexError: index 560
is out of bounds for axis 1 with size 560``.
"""

import warnings

import numpy as np
import torch

from algan import Axes, Scene
from algan.mobs.bezier_circuit import BezierCircuitCubic


def _plot(function):
    with warnings.catch_warnings():
        # NumPy says so itself, once per undefined sample.
        warnings.simplefilter("ignore", RuntimeWarning)
        axes = Axes(x_range=[-4, 4], y_range=[-2, 2], x_length=8, y_length=4)
        return axes.plot(function)


def _drawn_x(graph):
    points = torch.cat(
        [
            m.control_points.location.reshape(-1, 3)
            for m in graph.get_family()
            if isinstance(m, BezierCircuitCubic)
        ]
    )
    assert torch.isfinite(points).all()
    return points[:, 0]


def test_a_square_root_is_drawn_where_it_is_defined():
    with Scene():
        x = _drawn_x(_plot(np.sqrt))

    assert x.min() > -0.1
    assert x.max() > 3.9


def test_an_undefined_stretch_in_the_middle_is_a_gap():
    with Scene():
        x = _drawn_x(_plot(lambda t: np.sqrt(abs(t) - 1)))

    assert not ((x > -0.95) & (x < 0.95)).any()
    assert x.min() < -3.9
    assert x.max() > 3.9
