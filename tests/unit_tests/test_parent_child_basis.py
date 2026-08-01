from __future__ import annotations

import math

import pytest
import torch

import algan
from algan.geometry.geometry import get_rotation_around_axis
from algan.scene_manager import SceneManager


def _empty_scene(scene):
    scene.camera = None
    scene.light_sources = []


@pytest.fixture
def scene():
    SceneManager.reset()
    current = algan.Scene(scene_initializer=_empty_scene)
    yield current
    current.terminate()
    SceneManager.reset()


def _materialize(scene, *times):
    with algan.Off(
        record_attr_modifications=False,
        record_funcs=False,
        priority_level=math.inf,
    ):
        scene.timeline_manager.set_state_to_times(
            torch.tensor(times, dtype=torch.get_default_dtype())
        )


def test_synchronized_parent_child_rotations_preserve_descendant_bases(scene):
    group = algan.Group([algan.Square()]).arrange_in_grid().spawn(animate=False)
    square = group[0]
    descendants = square.get_descendants(include_self=True)
    initial_bases = {mob.id: mob.basis.clone().reshape(-1, 3, 3) for mob in descendants}

    with algan.Sync(run_time=1, rate_func=algan.rate_funcs.identity):
        group.rotate(180, algan.UP)
        square.rotate(180, algan.RIGHT)

    _materialize(scene, 0.25, 0.5, 0.75)

    for mob in descendants:
        actual = mob.basis.reshape(3, -1, 3, 3)
        expected = torch.stack(
            [
                initial_bases[mob.id]
                @ get_rotation_around_axis(180 * time, algan.UP, dim=-1)
                @ get_rotation_around_axis(180 * time, algan.RIGHT, dim=-1)
                for time in (0.25, 0.5, 0.75)
            ]
        )
        torch.testing.assert_close(actual, expected, atol=5e-6, rtol=0)
