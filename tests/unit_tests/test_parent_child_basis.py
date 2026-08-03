from __future__ import annotations

import math

import pytest
import torch

import algan
from algan.geometry.geometry import (
    get_rotation_around_axis,
    get_rotation_between_bases,
)
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


def _sheared_basis():
    """A basis whose rows are neither unit length nor mutually orthogonal."""
    return torch.tensor(
        [
            [0.02, 0.0, 0.0],
            [0.3, 1.1, 0.0],
            [0.0, 0.4, 0.05],
        ]
    )


def _flat(basis):
    """A 3x3 basis in the flat 9-channel layout ``Mob.basis`` stores."""
    return basis.reshape(9)


def test_rotation_between_bases_maps_a_sheared_basis_onto_the_target():
    source = _sheared_basis()
    target = torch.tensor(
        [
            [0.0, 0.7, 0.1],
            [-1.2, 0.2, 0.0],
            [0.05, 0.0, 0.3],
        ]
    )

    change = get_rotation_between_bases(source, target)

    torch.testing.assert_close(source @ change, target, atol=1e-6, rtol=0)


def test_assigning_a_basis_to_itself_does_not_drift(scene):
    # Mob.basis's setter records an absolute basis as a change relative to the
    # current one, so an inexact change is re-applied to the value it was
    # measured from. That used to amplify the float-noise shear of an
    # orthogonal basis roughly threefold per assignment -- and every
    # detach_history clone performs one, so wave_color's resolution refinement
    # collapsed a Cylinder's basis after a couple of dozen waves.
    square = algan.Square().spawn(animate=False)
    square.basis = _flat(_sheared_basis())
    original = square.basis.clone()

    for _ in range(40):
        square.basis = square.basis

    torch.testing.assert_close(square.basis, original, atol=1e-6, rtol=0)


def test_repeated_history_detachment_preserves_a_sheared_basis(scene):
    square = algan.Square().spawn(animate=False)
    square.basis = _flat(_sheared_basis())
    original = square.basis.clone()

    for _ in range(20):
        square.detach_history()

    torch.testing.assert_close(square.basis, original, atol=1e-6, rtol=0)
