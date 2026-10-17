"""Every point and direction argument takes every spelling ``move`` takes.

``mob.move((1, 0, 0))``, ``mob.move([1, 0, 0])`` and ``mob.move(np.array(...))``
all cast through :func:`~algan.utils.tensor_utils.cast_to_direction`. These
methods did arithmetic on the argument as given instead, so only a tensor got
through and anything else failed with an error about tuples, ``dim`` or
``shape`` that never named the argument.

A feature test, outside the fast suite: each case breaks only when its own
method changes.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from algan import RED, Circle, Group, Square
from algan.errors import AlganConfigurationError
from algan.scene_manager import SceneManager

VECTOR = (1.0, 0.5, 0.0)

SPELLINGS = pytest.mark.parametrize(
    "spell",
    [tuple, list, np.array],
    ids=["tuple", "list", "ndarray"],
)


def _tensor(vector):
    return torch.tensor(vector, dtype=torch.get_default_dtype())


@pytest.fixture(autouse=True)
def _fresh_scenes():
    SceneManager.reset()
    yield
    SceneManager.reset()


def materialize(*times: float):
    SceneManager.instance().current_scene.timeline_manager.set_state_to_times(
        torch.tensor(times, dtype=torch.get_default_dtype())
    )


def _shape():
    # Off-centre and irregular, so a direction that is ignored or mangled shows.
    mob = Square(add_to_scene=False).spawn(False)
    mob.location = (0.3, -0.2, 0.0)
    mob.scale((2.0, 1.0, 1.0))
    return mob


@SPELLINGS
@pytest.mark.parametrize(
    "query",
    [
        lambda mob, d: mob.get_edge_point(d),
        lambda mob, d: mob.get_edge_point(d, recursive=False),
        lambda mob, d: mob.get_length_in_direction(d),
        lambda mob, d: torch.stack(mob.sample_points_in_direction(d)),
    ],
    ids=["edge point", "own edge point", "length", "sampled points"],
)
def test_a_layout_query_takes_every_spelling(query, spell):
    """REGRESSION. ``get_edge_point((1, 0, 0))`` failed with "'tuple' object has
    no attribute 'dim'", and ``get_length_in_direction`` with "bad operand type
    for unary -: 'tuple'".
    """
    mob = _shape()
    materialize(1.0)
    torch.testing.assert_close(query(mob, spell(VECTOR)), query(mob, _tensor(VECTOR)))


def _settled(animate, spell, read):
    SceneManager.reset()
    mob = _shape()
    animate(mob, spell(VECTOR))
    materialize(10.0)
    return read(mob).clone()


@SPELLINGS
@pytest.mark.parametrize(
    "animate",
    [
        lambda mob, d: mob.look(d),
        lambda mob, d: mob.look(d, with_axis="up"),
        lambda mob, d: mob.look_at(d),
    ],
    ids=["look", "look with up", "look_at"],
)
def test_aiming_takes_every_spelling(animate, spell):
    """REGRESSION. ``look((1, 0, 0))`` failed with "'tuple' object has no
    attribute 'norm'", and ``look_at`` with "unsupported operand type(s) for -:
    'tuple' and 'Tensor'".
    """

    def basis(mob):
        return mob.basis

    reference = _settled(animate, _tensor, basis)
    assert not torch.allclose(reference, _settled(lambda m, d: None, _tensor, basis))
    torch.testing.assert_close(_settled(animate, spell, basis), reference)


@SPELLINGS
@pytest.mark.parametrize("end", ["start", "end"])
def test_move_between_takes_every_spelling(spell, end):
    """REGRESSION. ``move_between((0, 0, 0), RIGHT)`` failed with "can only
    concatenate tuple (not "Tensor") to tuple".
    """

    def location(mob):
        return mob.location

    def between(given):
        def animate(mob, point):
            other = _tensor((-1.0, 0.0, 0.0))
            pair = (point, other) if end == "start" else (other, point)
            mob.move_between(*pair)

        return _settled(animate, given, location)

    torch.testing.assert_close(between(_tensor).reshape(-1), _tensor((0.0, 0.25, 0.0)))
    torch.testing.assert_close(between(spell), between(_tensor))


def test_move_between_still_takes_a_mob():
    mob = Square(add_to_scene=False).spawn(False)
    left = Circle(add_to_scene=False).spawn(False)
    left.location = (-2.0, 0.0, 0.0)
    mob.move_between(left, (2.0, 2.0, 0.0))
    materialize(10.0)
    torch.testing.assert_close(mob.location.reshape(-1), _tensor((0.0, 1.0, 0.0)))


@SPELLINGS
def test_arrange_between_points_takes_every_spelling(spell):
    """REGRESSION. ``Group.arrange_between_points((0, 0, 0), (4, 0, 0))`` failed
    with "unsupported operand type(s) for -: 'tuple' and 'Tensor'".
    """

    def arranged(given):
        SceneManager.reset()
        members = [Square(add_to_scene=False).spawn(False) for _ in range(3)]
        group = Group(members)
        group.arrange_between_points(given((-2.0, 1.0, 0.0)), given((2.0, 1.0, 0.0)))
        materialize(10.0)
        return torch.stack([member.get_center().reshape(-1) for member in members])

    torch.testing.assert_close(arranged(spell), arranged(_tensor))
    torch.testing.assert_close(arranged(_tensor)[:, 0], _tensor((-1.0, 0.0, 1.0)))


@SPELLINGS
def test_wave_color_takes_every_direction_spelling(spell):
    """REGRESSION. ``wave_color(RED, direction=(1, 0, 0))`` failed with "'tuple'
    object has no attribute 'shape'".
    """
    times = (0.5, 1.0, 1.5)

    def swept(given):
        SceneManager.reset()
        mob = _shape()
        mob.wave_color(RED, direction=given(VECTOR), refine_resolution=False)
        materialize(*times)
        return mob.color.clone()

    torch.testing.assert_close(swept(spell), swept(_tensor))


@pytest.mark.parametrize(
    "call",
    [
        lambda mob: mob.get_edge_point(2.0),
        lambda mob: mob.look((1.0, 0.0)),
        lambda mob: mob.look_at("origin"),
        lambda mob: mob.move_between((0.0, 0.0), (1.0, 0.0, 0.0)),
    ],
    ids=["scalar", "2-D direction", "string", "2-D start"],
)
def test_a_point_that_is_not_3_d_is_refused_by_name(call):
    mob = _shape()
    with pytest.raises(AlganConfigurationError, match=r"must be a 3-D vector"):
        call(mob)
