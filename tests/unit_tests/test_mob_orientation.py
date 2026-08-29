from __future__ import annotations

import math

import pytest
import torch

from algan import ORIGIN, OUT, Mob, Scene, SceneManager

# In the fast suite: rotation is where location and basis have to stay
# consistent with each other, and every 3-D transform composes through it.
pytestmark = pytest.mark.fast


def _empty_scene(scene):
    scene.camera = None
    scene.light_sources = []


@pytest.fixture(autouse=True)
def fresh_scene_stack():
    SceneManager.reset()
    yield
    SceneManager.reset()


def test_rotate_without_about_point_only_changes_basis():
    scene = Scene(scene_initializer=_empty_scene)
    mob = Mob(location=[1, 0, 0], add_to_scene=False)
    initial_location = mob.location.clone()
    initial_basis = mob.basis.clone()

    mob.rotate(90, OUT)

    torch.testing.assert_close(mob.location, initial_location)
    assert not torch.equal(mob.basis, initial_basis)
    scene.terminate()


def test_rotate_about_point_changes_basis_and_location():
    scene = Scene(scene_initializer=_empty_scene)
    mob = Mob(location=[1, 0, 0], add_to_scene=False)
    initial_basis = mob.basis.clone()

    mob.rotate(90, OUT, about=ORIGIN)

    torch.testing.assert_close(
        mob.location,
        torch.tensor([[[0.0, 1.0, 0.0]]]),
        atol=1e-6,
        rtol=0,
    )
    assert not torch.equal(mob.basis, initial_basis)
    scene.terminate()


def test_orbit_only_changes_location():
    scene = Scene(scene_initializer=_empty_scene)
    mob = Mob(location=[1, 0, 0], add_to_scene=False)
    initial_basis = mob.basis.clone()

    mob.orbit(90, OUT, about=ORIGIN)

    torch.testing.assert_close(
        mob.location,
        torch.tensor([[[0.0, 1.0, 0.0]]]),
        atol=1e-6,
        rtol=0,
    )
    torch.testing.assert_close(mob.basis, initial_basis)
    scene.terminate()


def test_animated_rotate_about_point_replays_location_and_basis_together():
    scene = Scene(scene_initializer=_empty_scene)
    mob = Mob(location=[1, 0, 0], add_to_scene=False).spawn(animate=False)
    initial_basis = mob.basis.clone()

    mob.rotate(90, OUT, about=ORIGIN)
    scene.timeline_manager.set_state_to_times(torch.tensor([0.5]))

    diagonal = math.sqrt(0.5)
    torch.testing.assert_close(
        mob.location,
        torch.tensor([[[diagonal, diagonal, 0.0]]]),
        atol=2e-6,
        rtol=0,
    )
    assert not torch.equal(mob.basis, initial_basis)
    scene.terminate()
