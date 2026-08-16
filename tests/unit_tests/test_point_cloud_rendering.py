"""The exported point-cloud family reaches the renderer as packed spheres.

``DotCloud``, ``PointCloudDot``, ``TrueDot`` and ``PGroup`` construct their
points and register as actors. They delegate rendering to a hidden batched
``Dot3D`` collection, retaining the point-array API without requiring a separate
point-sprite path in the renderer.

This is a per-family regression, so it sits outside the fast suite: it only
breaks when the point-cloud family or the batched-sphere path it delegates to is
worked on, and it constructs several dense packed sphere collections to do it.
The full suite still runs it.
"""

from __future__ import annotations

import pytest
import torch

from algan import (
    BLUE_A,
    GREEN_A,
    YELLOW,
    Dot3D,
    DotCloud,
    Off,
    PGroup,
    PointCloudDot,
    Scene,
    TrueDot,
)
from algan.mobs.surfaces.surface import get_render_primitives_batched

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
    assert cloud._get_memory_used_per_timestep() > 0


def test_dot3d_and_point_cloud_spheres_use_automatic_resolution(scene):
    with Off():
        cloud = DotCloud(points=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        dot = Dot3D(radius=cloud.point_radius, add_to_scene=False)

    geometry = cloud.children[0]
    assert dot.resolution is None
    assert dot._geometry_auto_resolution_enabled
    assert geometry.resolution is None
    assert geometry._geometry_auto_resolution_enabled
    assert (geometry.grid_width, geometry.grid_height) == (
        dot.grid_width,
        dot.grid_height,
    )


def test_dot_cloud_spheres_have_disconnected_triangle_topology(scene):
    points = torch.tensor(
        [
            [-1.5, -1.0, 0.0],
            [1.5, -1.0, 0.0],
            [0.0, 1.5, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    with Off():
        cloud = DotCloud(points=points, stroke_width=10).spawn()

    geometry = cloud.children[0]
    primitive = cloud.get_render_primitives()[0]
    triangles = primitive.corners.reshape(-1, 3, 3)
    triangles_per_sphere = 2 * (geometry.grid_width - 1) * (geometry.grid_height - 1)

    assert len(triangles) == len(points) * triangles_per_sphere

    nearest_center = (
        (triangles.unsqueeze(-2) - points.reshape(1, 1, -1, 3))
        .norm(dim=-1)
        .argmin(dim=-1)
    )
    assert torch.all(nearest_center == nearest_center[:, :1])

    deferred_primitive = get_render_primitives_batched([geometry])[0]
    assert torch.equal(deferred_primitive.corners, primitive.corners)
    assert torch.equal(deferred_primitive.normals, primitive.normals)


def test_point_cloud_memory_estimate_scales_with_sphere_count(scene):
    with Off():
        one_point = DotCloud(points=[[0.0, 0.0, 0.0]]).spawn()
        four_points = DotCloud(
            points=[
                [-1.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
                [-1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
        ).spawn()

    assert (
        four_points._get_memory_used_per_timestep()
        == 4 * one_point._get_memory_used_per_timestep()
    )
