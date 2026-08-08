"""The exported point-cloud family reaches the renderer as packed spheres.

``DotCloud``, ``PointCloudDot``, ``TrueDot`` and ``PGroup`` construct their
points and register as actors. They delegate rendering to a hidden batched
``Dot3D`` collection, retaining the point-array API without requiring a separate
point-sprite path in the renderer.

The whole module is marked ``slow`` and so sits outside the fast suite. It
constructs several dense packed sphere collections, which is not worth paying
for on every unrelated change; the full suite still runs it.
"""

from __future__ import annotations

import pytest

from algan import (
    BLUE_A,
    GREEN_A,
    YELLOW,
    DotCloud,
    Off,
    PGroup,
    PointCloudDot,
    Scene,
    TrueDot,
)

pytestmark = pytest.mark.slow

BUILDERS = {
    "DotCloud": lambda **kwargs: DotCloud(
        color=YELLOW, radius=0.6, density=12, **kwargs
    ),
    "PointCloudDot": lambda **kwargs: PointCloudDot(
        radius=0.5, density=14, color=BLUE_A, **kwargs
    ),
    "TrueDot": lambda **kwargs: TrueDot(color=GREEN_A, **kwargs),
}


@pytest.fixture
def scene():
    with Scene() as active:
        yield active


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_point_cloud_mob_builds_points_and_registers_as_an_actor(scene, name):
    with Off():
        cloud = BUILDERS[name]().spawn()
    assert cloud.points is not None
    assert len(cloud.points) > 0
    assert cloud in scene.actors


def test_pgroup_collects_point_clouds(scene):
    with Off():
        group = PGroup(
            *(builder(add_to_scene=False) for builder in BUILDERS.values())
        ).spawn()
    assert len(group.children) == len(BUILDERS)
    assert group.get_render_primitives()


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_point_cloud_mob_produces_render_primitives(scene, name):
    with Off():
        cloud = BUILDERS[name]().spawn()
    assert hasattr(cloud, "get_render_primitives"), f"{name} cannot reach the renderer"
    primitives = cloud.get_render_primitives()
    assert primitives
    assert sum(primitive.corners.numel() for primitive in primitives) > 0
    assert cloud.get_memory_used_per_timestep() > 0
