"""``resolution`` reads the same way round on every revolved solid.

Manim's ``Cylinder`` and ``Cone`` both take ``resolution`` as (patches along the
axis, patches round it). Algan's ``Cone`` did, its ``Cylinder`` read the pair the
other way round, and ``Arrow3D`` hands one pair to a ``Cylinder`` shaft and a
``Cone`` tip -- so any non-square pair flattened one of the two: ``(12, 2)``
left the tip two patches round, a fin with its base disc as a crossbar, and
``(2, 16)`` turned the shaft into a ribbon.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from algan.constants.spatial import ORIGIN, RIGHT
from algan.mobs.shapes_3d import Arrow3D, Cone, Cylinder
from algan.scene_manager import SceneManager


@pytest.fixture(autouse=True)
def reset_scene():
    SceneManager.reset()
    yield
    SceneManager.reset()


def _roundness(mob, axis) -> float:
    """How round the grid's cross-section is: 1 for a circle, 0 for a fin.

    The ratio of the two principal spreads of the vertices once the axial
    component is taken out. A part only two patches round has every vertex on
    one plane through its axis, so its second spread is zero.
    """
    points = mob.grid.location.detach().reshape(-1, 3)
    relative = points - points.mean(0)
    axis = F.normalize(axis.to(relative), dim=-1)
    relative = relative - (relative @ axis)[:, None] * axis
    spreads = torch.linalg.svdvals(relative)
    return float(spreads[1] / spreads[0])


@pytest.mark.parametrize("resolution", [(2, 16), (5, 12)])
def test_cylinder_and_cone_read_along_then_round(resolution):
    along, around = resolution
    cylinder = Cylinder(resolution=resolution, add_to_scene=False)
    cone = Cone(resolution=resolution, add_to_scene=False)

    # Algan's grids count vertices; the cylinder carries its azimuth on the
    # first grid axis, the cone on the second.
    assert (cylinder.grid_width, cylinder.grid_height) == (around + 1, along + 1)
    assert (cone.grid_height, cone.grid_width) == (around + 1, along + 1)


@pytest.mark.parametrize("resolution", [(2, 16), (12, 2), 16])
def test_arrow3d_shaft_and_tip_agree_on_one_resolution(resolution):
    arrow = Arrow3D(
        start=ORIGIN, end=RIGHT * 2, resolution=resolution, add_to_scene=False
    ).spawn()

    assert arrow.tail.grid_width == arrow.head.grid_height
    assert arrow.tail.grid_height == arrow.head.grid_width
    shaft = _roundness(arrow.tail, RIGHT)
    tip = _roundness(arrow.head, RIGHT)
    assert shaft == pytest.approx(tip, abs=0.05)


def test_arrow3d_with_few_patches_along_is_round():
    arrow = Arrow3D(
        start=ORIGIN, end=RIGHT * 2, resolution=(2, 16), add_to_scene=False
    ).spawn()

    assert _roundness(arrow.tail, RIGHT) > 0.9
    assert _roundness(arrow.head, RIGHT) > 0.9
