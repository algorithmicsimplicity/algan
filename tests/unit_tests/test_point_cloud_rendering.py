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

from algan import BLUE_A, GREEN_A, YELLOW, Dot3D, Off, Scene
from algan.manim import DotCloud, PGroup, PointCloudDot, TrueDot
from algan.mobs.point_cloud import OpenGLPGroup, PMobject
from algan.mobs.surfaces.surface import (
    get_grid_to_triangle_indices,
    get_render_primitives_batched,
    surface_weld_flags,
)

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


@pytest.mark.parametrize("group_type", [PGroup, OpenGLPGroup])
@pytest.mark.parametrize("style", ["color", "stroke_width"])
@pytest.mark.parametrize("family", [False, True])
@pytest.mark.parametrize("spawned", [False, True])
def test_pgroup_styling_preserves_nested_members_and_transforms(
    scene, group_type, style, family, spawned
):
    with Off():
        a = PMobject(points=[[-1.0, 0.0, 0.0]], color=BLUE_A, add_to_scene=False)
        b = PMobject(points=[[1.0, 0.0, 0.0]], color=BLUE_A, add_to_scene=False)
        nested = group_type(a, add_to_scene=False)
        group = group_type(nested, b)
        if spawned:
            group.spawn()
        # Warm the hierarchy cache, so retained links must also remain usable.
        group.get_descendants()
        initial_rgbas = [mob.rgbas.clone() for mob in (a, b)]
        if style == "color":
            assert group.set_color(GREEN_A, family=family) is group
        else:
            assert group.set_stroke_width(12, family=family) is group

        assert group.children == [nested, b]
        assert nested.children == [a]
        for parent, child in ((group, nested), (group, b), (nested, a)):
            assert any(item is parent for item in child.parents)
        for mob, rgba in zip((a, b), initial_rgbas):
            if style == "color" and family:
                expected_rgb = GREEN_A.rgb.reshape(-1, 3)[0]
                assert torch.allclose(mob.rgbas[:, :3], expected_rgb.expand(1, 3))
            else:
                assert torch.equal(mob.rgbas, rgba)
            expected_width = 12 if style == "stroke_width" and family else 4
            assert mob.get_stroke_width() == expected_width
            assert mob.point_radius == pytest.approx(expected_width * 0.01)

        corners = [p.corners.clone() for p in group.get_render_primitives()]
        assert len(corners) == 2
        locations = [mob.location.clone() for mob in (a, b)]
        shift = torch.tensor([0.75, 0.25, -0.5])
        group.move(shift)
        for mob, location in zip((a, b), locations):
            assert torch.allclose(mob.location, location + shift)
        for primitive, before in zip(group.get_render_primitives(), corners):
            assert torch.allclose(primitive.corners, before + shift, atol=1e-6)


def test_rebuilding_group_points_only_replaces_its_generated_geometry(scene):
    with Off():
        member = PMobject(points=[[1.0, 0.0, 0.0]], add_to_scene=False)
        group = PGroup(member)
        group.add_points([[0.0, 0.0, 0.0]])
        old_geometry = group.children[0]

        group.set_color(GREEN_A, family=False)

        assert len(group.children) == 2
        assert group.children[1] is member
        assert group.children[0] is not old_geometry
        assert all(parent is not group for parent in old_geometry.parents)
        assert any(parent is group for parent in member.parents)
        group.reset_points()
        assert group.children == [member]


def test_ingesting_group_members_still_replaces_them_with_merged_points(scene):
    with Off():
        a = PMobject(points=[[-1.0, 0.0, 0.0]], add_to_scene=False)
        b = PMobject(points=[[1.0, 0.0, 0.0]], add_to_scene=False)
        group = PGroup(a, b)

        group.ingest_submobjects()

        assert torch.equal(group.points, torch.cat((a.points, b.points)))
        assert len(group.children) == 1
        for member in (a, b):
            assert all(child is not member for child in group.children)
            assert all(parent is not group for parent in member.parents)


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
    # Derived from the builder rather than restated as 2*(W-1)*(H-1). That
    # formula is the UNWELDED count, so it hard-codes an answer that
    # ALGAN_WELD_SURFACE_SEAMS changes -- a Sphere's two pole fans are
    # degenerate and the weld drops them. What this test is about is that each
    # sphere's triangles stay disconnected from its neighbours', which is true
    # either way; the count is a means to that, so it asks the same builder the
    # renderer asked.
    per_sphere_indices = get_grid_to_triangle_indices(
        geometry.grid_width,
        geometry.grid_height,
        primitive.corners.device,
        surface_weld_flags(geometry._reshape_grid_for_render(geometry.grid.location)),
    )
    triangles_per_sphere = len(per_sphere_indices) // 3

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
