"""Painting a grid of values onto a named colour.

``WHITE.mult_opacity(values)`` is how you turn a colour into a gradient: one
opacity per vertex, or per texel, computed from whatever the values mean --
height above the ground, distance from a point, a field being visualised.

A named colour is a single ``[5]`` row, though, so until the colour has the
value's leading axes there is nothing for the assignment to land in, and the
call used to raise ``expand(...): the number of sizes provided (1) must be
greater or equal to the number of dimensions in the tensor``. :meth:`Color.prep_set`
now gives the colour those axes -- but only when the value really carries a
grid, since a scalar opacity arrives padded to ``[1, 1, 1]`` and every existing
caller of ``WHITE.set_opacity(0.5)`` expects a ``[5]`` colour back.

Feature tests for the colour helpers: unmarked, so outside the fast suite.
"""

import torch

from algan.constants.color import BLUE, RED, WHITE
from algan.mobs.shapes_3d import Sphere
from algan.scene_manager import SceneManager


def test_a_named_colour_takes_a_value_per_vertex():
    SceneManager.reset()
    sphere = Sphere(radius=1.0)
    height = sphere.grid.location[..., 1:2]

    colored = WHITE.mult_opacity(height)

    assert colored.shape == (*height.shape[:-1], 5)
    assert torch.allclose(colored[..., -1:], height)
    sphere.grid.color = colored


def test_a_named_colour_takes_a_value_per_texel():
    """The same call with a texture-shaped value, which is what makes a map
    buildable from :meth:`~.Surface.get_texture_locations`.
    """
    values = torch.rand(16, 12, 1)

    colored = BLUE.mult_opacity(values)

    assert colored.shape == (16, 12, 5)
    assert torch.allclose(colored[..., -1:], values)
    assert torch.allclose(colored[..., :3], BLUE[:3].expand(16, 12, 3))


def test_a_scalar_still_gives_back_a_single_colour():
    """The shape every existing caller gets: broadening the grid case must not
    inflate this one.
    """
    assert WHITE.set_opacity(0.5).shape == (5,)
    assert WHITE.mult_opacity(0.5).shape == (5,)
    assert WHITE.set_glow(0.3).shape == (5,)
    assert RED.set_rgb(torch.rand(3)).shape == (5,)


def test_a_colour_that_already_has_axes_keeps_them():
    SceneManager.reset()
    sphere = Sphere(radius=1.0)
    height = sphere.grid.location[..., 1:2]

    colored = sphere.grid.color.mult_opacity(height)

    assert colored.shape == sphere.grid.color.shape
