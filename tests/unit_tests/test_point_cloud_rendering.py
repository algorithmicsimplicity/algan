"""The point-cloud family is exported but cannot currently reach the renderer.

``DotCloud``, ``PointCloudDot``, ``TrueDot`` and ``PGroup`` construct their
points and register as actors, but none of them defines
``get_render_primitives``, which is the method the render loop calls to turn an
actor into geometry.  A scene built out of them therefore renders empty frames
with no error.

The construction tests below pass today.  The rendering test is an expected
failure that pins the gap: if someone implements the primitives, it XPASSes and
fails the suite, which is the signal to move these classes out of ``EXEMPT`` in
``test_render_coverage_audit.py`` and into a full-render scene.
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

BUILDERS = {
    "DotCloud": lambda: DotCloud(color=YELLOW, radius=0.6, density=12),
    "PointCloudDot": lambda: PointCloudDot(radius=0.5, density=14, color=BLUE_A),
    "TrueDot": lambda: TrueDot(color=GREEN_A),
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
        group = PGroup(*(builder() for builder in BUILDERS.values()))
    assert len(group.children) == len(BUILDERS)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "point-cloud Mobs define no get_render_primitives, so they render "
        "nothing; remove this xfail and the EXEMPT entries in "
        "test_render_coverage_audit.py once they do"
    ),
)
@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_point_cloud_mob_produces_render_primitives(scene, name):
    with Off():
        cloud = BUILDERS[name]().spawn()
    assert hasattr(cloud, "get_render_primitives"), (
        f"{name} cannot reach the renderer"
    )
    assert cloud.get_render_primitives()
