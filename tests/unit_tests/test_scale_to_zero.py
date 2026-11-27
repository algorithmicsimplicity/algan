"""Scaling a Mob to nothing and back.

Render-free: the state is read at authoring time and at materialized frame
times, the way the renderer reads it. ``scale(0)`` is legal -- a script can end
on a shrink to nothing -- but it used to be final: a Mob's basis rows were
scaled to zero, a later ``set_scale`` or basis assignment multiplied zero by its
target, and the Mob stayed invisible for the rest of the video with no error.
"""

from __future__ import annotations

import math

import pytest
import torch

import algan
from algan.geometry.geometry import map_global_to_local_coords
from algan.scene_manager import SceneManager


def _empty_scene(scene):
    scene.camera = None
    scene.light_sources = []


@pytest.fixture
def scene():
    SceneManager.reset()
    current = algan.Scene(scene_initializer=_empty_scene)
    yield current
    current._terminate()
    SceneManager.reset()


def _materialize(scene, time):
    with algan.Off(
        record_attr_modifications=False,
        record_funcs=False,
        priority_level=math.inf,
    ):
        scene.timeline_manager.set_state_to_times(
            torch.tensor([time], dtype=torch.get_default_dtype())
        )


def _shape(mob):
    """Every location row of the subtree in the Mob's own frame, in float64.

    Scaling or turning a Mob about its pivot leaves these unchanged, so they
    are what a Mob that grows back from nothing must give back.
    """
    rows = mob.get_animated_attribute("location", include_descendants=True)
    return map_global_to_local_coords(
        mob.location.reshape(1, 3).double(),
        mob.basis.reshape(1, 9).double(),
        rows.reshape(-1, 3).double(),
    )


def _turned_square():
    square = algan.Square().move(algan.RIGHT * 2).spawn(animate=False)
    with algan.Off():
        square.rotate(30, algan.OUT)
        square.rotate(20, algan.RIGHT)
    return square


def _assert_same(actual, expected, atol=1e-5):
    torch.testing.assert_close(actual.double(), expected.double(), rtol=0, atol=atol)


@pytest.mark.parametrize("regrow", ["set_scale", "scale_coefficient", "basis"])
def test_a_mob_scaled_to_nothing_grows_back_to_its_size_and_orientation(scene, regrow):
    square = _turned_square()
    basis = square.basis.clone()
    scale = square.scale_coefficient.clone()
    shape = _shape(square)
    with algan.Sync(runtime=1, easing=algan.easings.identity):
        square.scale(0)
    assert float(square.get_width()) == 0.0
    with algan.Sync(runtime=1, easing=algan.easings.identity):
        if regrow == "set_scale":
            square.set_scale(scale)
        elif regrow == "scale_coefficient":
            square.scale_coefficient = scale
        else:
            square.basis = basis
    _assert_same(square.basis, basis)
    _assert_same(_shape(square), shape)

    # As the renderer sees it: half grown, half way through.
    _materialize(scene, 1.5)
    _assert_same(square.scale_coefficient, 0.5 * scale)
    _assert_same(square.normalized_basis, basis / scale.repeat_interleave(3, -1))
    _assert_same(_shape(square), shape)
    _materialize(scene, 2.0)
    _assert_same(square.basis, basis)
    _assert_same(_shape(square), shape)


def test_a_mob_moved_while_it_was_nothing_grows_back_where_it_is(scene):
    square = _turned_square()
    scale = square.scale_coefficient.clone()
    shape = _shape(square)
    square.scale(0)
    square.move(algan.UP * 3)
    square.set_scale(scale)
    _assert_same(_shape(square), shape)
    _assert_same(square.location.reshape(3), torch.tensor([2.0, 3.0, 0.0]))


def test_a_group_scaled_to_nothing_grows_its_members_back(scene):
    members = [algan.Square().move(algan.LEFT * 2), algan.Circle().move(algan.UP)]
    group = algan.Group(members).spawn(animate=False)
    before = [_shape(member) for member in members]
    locations = [member.location.clone() for member in members]
    group.scale(0)
    group.set_scale(1)
    for member, shape, location in zip(members, before, locations):
        _assert_same(_shape(member), shape)
        _assert_same(member.location, location)


def test_a_shrink_to_nothing_can_still_end_the_video(scene):
    square = _turned_square()
    scale = square.scale_coefficient.clone()
    with algan.Sync(runtime=1, easing=algan.easings.identity):
        square.scale(0)
    _materialize(scene, 0.5)
    _assert_same(square.scale_coefficient, 0.5 * scale)
    _materialize(scene, 1.0)
    assert float(square.scale_coefficient.abs().max()) == 0.0


def test_scaling_nothing_by_a_factor_leaves_nothing(scene):
    # There is no size left to multiply: only an absolute size (set_scale, an
    # assigned scale_coefficient or basis) brings a collapsed Mob back.
    square = _turned_square()
    square.scale(0)
    square.scale(2)
    assert float(square.get_width()) == 0.0


def test_a_mob_flattened_along_one_axis_and_turned_regrows_turned(scene):
    # Two surviving axes fix the turn, so the flattened axis comes back along
    # where the turn carried it, not where it pointed before.
    square = _turned_square()
    scale = square.scale_coefficient.clone()
    shape = _shape(square)
    square.scale(torch.tensor([1.0, 0.0, 1.0]))
    square.rotate(90, algan.OUT)
    square.set_scale(scale)
    reference = _turned_square()
    with algan.Off():
        reference.rotate(90, algan.OUT)
    _assert_same(square.basis, reference.basis)
    _assert_same(_shape(square), shape)


def test_an_ordinary_write_remembers_no_collapse(scene):
    square = _turned_square()
    square.scale(0.5)
    square.rotate(45, algan.UP)
    assert "_collapse" not in square.__dict__
    square.scale(0)
    assert "_collapse" in square.__dict__
    square.set_scale(1)
    assert "_collapse" not in square.__dict__
