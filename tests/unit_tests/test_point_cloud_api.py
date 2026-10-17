"""The Manim point-cloud API on Algan's packed-sphere point clouds.

``PMobject`` keeps Manim's point and color arrays, but what renders is a hidden
packed ``Dot3D`` it rebuilds from them after every edit. These tests pin both
halves: what each array method computes, and that a rebuild keeps what the
spheres already show -- the transforms applied to them, and the frames recorded
before the edit.

A per-family feature test, so it sits outside the fast suite: it only breaks
when the point-cloud family is worked on.
"""

from __future__ import annotations

import pytest
import torch

from algan import BLUE, GREEN, RED, RIGHT, UP, WHITE, YELLOW, Dot3D, Off, Scene
from algan.errors import AlganConfigurationError
from algan.manim import DotCloud, PGroup, PointCloudDot, TrueDot
from algan.mobs.point_cloud import Mobject1D, PMobject, _as_rgba


@pytest.fixture
def scene():
    with Scene() as active:
        yield active


def _rgb(color):
    return _as_rgba(color)[:3]


def _peak_opacity(mob, frames):
    """Each frame's highest opacity over ``mob``'s rows, rounded."""
    values = mob.opacity.reshape(frames, -1).max(-1).values
    return [round(float(value), 3) for value in values]


def _cloud(points, **kwargs):
    with Off():
        return PMobject(points=points, **kwargs)


# --- Rebuilds keep what the spheres already show ------------------------------


def test_points_report_where_a_transform_moved_the_cloud(scene):
    """REGRESSION. ``get_points()`` reported the points the cloud was built
    with, wherever it had since been moved.
    """
    cloud = _cloud([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], color=RED)
    with Off():
        cloud.spawn()
        cloud.move(UP * 2)

    expected = torch.tensor([[0.0, 2.0, 0.0], [1.0, 2.0, 0.0]])
    assert torch.allclose(cloud.get_points(), expected)
    assert torch.allclose(cloud.points, expected)


def test_an_edit_after_a_move_starts_from_the_moved_points(scene):
    """REGRESSION. The arrays never saw the move, so the rebuild after
    ``add_points`` put the cloud back where it was built.
    """
    cloud = _cloud([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], color=RED)
    with Off():
        cloud.spawn()
    cloud.move(UP * 2)
    cloud.add_points([[5.0, 0.0, 0.0]])

    spheres = cloud._point_geometry.location.reshape(-1, 3)
    assert torch.allclose(
        spheres, torch.tensor([[0.0, 2.0, 0.0], [1.0, 2.0, 0.0], [5.0, 0.0, 0.0]])
    )


def test_a_color_animated_onto_the_cloud_survives_the_next_edit(scene):
    """REGRESSION. The rebuild after ``add_points`` took the colors the cloud
    was built with, so the color animated onto it was undone.
    """
    cloud = _cloud([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], color=RED)
    with Off():
        cloud.spawn()
    cloud.color = BLUE
    cloud.add_points([[2.0, 0.0, 0.0]], color=GREEN)

    rgbas = cloud.get_all_rgbas()
    assert torch.allclose(rgbas[:2, :3], _rgb(BLUE).expand(2, 3), atol=1e-6)
    assert torch.allclose(rgbas[2, :3], _rgb(GREEN), atol=1e-6)


def test_an_in_place_edit_to_the_points_reaches_the_next_rebuild(scene):
    cloud = _cloud([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    cloud.points[0, 0] = -3.0
    cloud.set_color(RED)

    spheres = cloud._point_geometry.location.reshape(-1, 3)
    assert torch.allclose(spheres[:, 0], torch.tensor([-3.0, 1.0]))


def test_an_edit_after_spawning_keeps_the_entrance(scene):
    """REGRESSION. The rebuild dropped the spheres that had faded in and spawned
    new ones at the edit, so every frame before ``set_color`` lost the cloud.
    """
    cloud = PointCloudDot(radius=0.5, density=5).spawn()  # fades in over 0..1
    entering = cloud._point_geometry
    cloud.set_color(BLUE)
    cloud.wait(1)
    recolored = cloud._point_geometry

    scene.timeline_manager.set_state_to_times(torch.tensor([0.5, 0.9, 1.5]))
    assert recolored is not entering
    entrance = _peak_opacity(entering, 3)
    assert 0 < entrance[0] < entrance[1] < 1
    assert entrance[2] == 0
    assert _peak_opacity(recolored, 3) == [0, 0, 1]


def test_an_edit_after_despawning_keeps_the_cloud_before_it(scene):
    """REGRESSION. As after spawning: the rebuild took the despawned spheres out
    of the hierarchy, so the frames before the despawn lost the cloud.
    """
    cloud = _cloud([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], color=RED)
    with Off():
        cloud.spawn()
    cloud.wait(1)
    with Off():
        cloud.despawn()
    shown = cloud._point_geometry
    cloud.set_color(BLUE)

    # Still in the hierarchy, so the frames before the despawn draw it.
    assert any(child is shown for child in cloud.children)
    assert not cloud._point_geometry.is_spawned()
    scene.timeline_manager.set_state_to_times(torch.tensor([0.5, 1.5]))
    assert _peak_opacity(shown, 2) == [1, 0]


def test_spheres_replaced_the_instant_they_spawned_are_dropped(scene):
    with Off():
        cloud = PMobject(points=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]).spawn()
        cloud.set_color(BLUE)

    assert cloud.children == [cloud._point_geometry]


def test_retired_spheres_leave_the_layout(scene):
    cloud = _cloud([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    with Off():
        cloud.spawn()
    cloud.wait(1)
    cloud.set_points([[10.0, 0.0, 0.0], [12.0, 0.0, 0.0]])

    assert len(cloud.children) == 2
    assert float(cloud.get_center().reshape(-1)[0]) == pytest.approx(11.0, abs=0.05)


def test_retired_spheres_are_only_built_for_frames_they_can_show(scene):
    cloud = _cloud([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    with Off():
        cloud.spawn()
    cloud.wait(1)
    cloud.set_color(BLUE)
    cloud.wait(1)

    timeline = scene.timeline_manager
    timeline.set_state_to_times(torch.tensor([0.25, 0.5]))
    assert len(cloud.get_render_primitives()) == 2
    timeline.set_state_to_times(torch.tensor([1.25, 1.5]))
    assert len(cloud.get_render_primitives()) == 1


# --- The array API -------------------------------------------------------------


def test_add_points_appends_with_its_own_color_and_opacity(scene):
    cloud = _cloud([[0.0, 0.0, 0.0]], color=RED)
    cloud.add_points([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], color=BLUE, opacity=0.25)

    assert cloud.get_num_points() == 3
    rgbas = cloud.get_all_rgbas()
    assert torch.allclose(rgbas[0, :3], _rgb(RED))
    assert torch.allclose(rgbas[1:, :3], _rgb(BLUE).expand(2, 3))
    assert rgbas[1:, 3].tolist() == [0.25, 0.25]


def test_add_points_rejects_a_color_per_point_of_the_wrong_length(scene):
    cloud = _cloud([[0.0, 0.0, 0.0]])
    with pytest.raises(AlganConfigurationError, match="same length"):
        cloud.add_points([[1.0, 0.0, 0.0]], rgbas=torch.ones((2, 4)))


def test_set_points_resizes_the_colors(scene):
    cloud = _cloud([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], color=RED)
    cloud.set_points([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    assert cloud.get_all_rgbas().shape == (3, 4)
    assert torch.allclose(cloud.get_all_rgbas()[:, :3], _rgb(RED).expand(3, 3))

    empty = _cloud(None, color=GREEN)
    empty.set_points([[0.0, 0.0, 0.0]])
    assert torch.allclose(empty.get_all_rgbas()[0, :3], _rgb(GREEN))


def test_set_color_by_gradient_runs_from_the_first_point_to_the_last(scene):
    cloud = _cloud([[float(x), 0.0, 0.0] for x in range(5)])
    cloud.set_color_by_gradient(RED, BLUE)

    rgbas = cloud.get_all_rgbas()
    assert torch.allclose(rgbas[0, :3], _rgb(RED), atol=1e-6)
    assert torch.allclose(rgbas[-1, :3], _rgb(BLUE), atol=1e-6)
    assert torch.allclose(rgbas[2, :3], (_rgb(RED) + _rgb(BLUE)) / 2, atol=1e-6)


def test_set_colors_by_radial_gradient_fades_out_from_the_center(scene):
    cloud = _cloud([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [3.0, 0.0, 0.0]])
    cloud.set_colors_by_radial_gradient(
        center=[0.0, 0.0, 0.0], radius=1, inner_color=WHITE, outer_color=RED
    )

    rgbas = cloud.get_all_rgbas()
    assert torch.allclose(rgbas[0, :3], _rgb(WHITE), atol=1e-6)
    assert torch.allclose(rgbas[1, :3], (_rgb(WHITE) + _rgb(RED)) / 2, atol=1e-6)
    assert torch.allclose(rgbas[2, :3], _rgb(RED), atol=1e-6)


def test_match_colors_resamples_the_other_clouds_colors(scene):
    source = _cloud([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    source.set_color_by_gradient(RED, BLUE)
    target = _cloud([[float(x), 1.0, 0.0] for x in range(3)])
    target.match_colors(source)

    rgbas = target.get_all_rgbas()
    assert torch.allclose(rgbas[0, :3], _rgb(RED), atol=1e-6)
    assert torch.allclose(rgbas[2, :3], _rgb(BLUE), atol=1e-6)


def test_filter_thin_and_sort_rearrange_points_and_colors_together(scene):
    cloud = _cloud([[float(x), 0.0, 0.0] for x in (3, 1, 4, 0, 2)])
    cloud.set_color_by_gradient(RED, BLUE)
    color_of = {
        float(point[0]): rgba.clone()
        for point, rgba in zip(cloud.get_points(), cloud.get_all_rgbas())
    }

    cloud.sort_points()
    assert cloud.get_points()[:, 0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    cloud.filter_out(lambda point: point[0] == 2.0)
    assert cloud.get_points()[:, 0].tolist() == [0.0, 1.0, 3.0, 4.0]
    cloud.thin_out(2)
    assert cloud.get_points()[:, 0].tolist() == [0.0, 3.0]
    for point, rgba in zip(cloud.get_points(), cloud.get_all_rgbas()):
        assert torch.allclose(rgba, color_of[float(point[0])], atol=1e-6)


def test_fade_to_blends_every_point_toward_a_color(scene):
    cloud = _cloud([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], color=RED)
    cloud.fade_to(BLUE, 0.25)

    expected = _rgb(RED) * 0.75 + _rgb(BLUE) * 0.25
    assert torch.allclose(cloud.get_all_rgbas()[:, :3], expected.expand(2, 3))


def test_point_from_proportion_indexes_along_the_points(scene):
    cloud = _cloud([[float(x), 0.0, 0.0] for x in range(5)])
    assert cloud.point_from_proportion(0).tolist() == [0.0, 0.0, 0.0]
    assert cloud.point_from_proportion(0.5).tolist() == [2.0, 0.0, 0.0]
    assert cloud.point_from_proportion(1).tolist() == [4.0, 0.0, 0.0]
    with pytest.raises(IndexError):
        _cloud(None).point_from_proportion(0.5)


def test_align_points_with_larger_matches_the_point_count(scene):
    larger = _cloud([[float(x), 0.0, 0.0] for x in range(4)])
    smaller = _cloud([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    smaller.align_points_with_larger(larger)
    assert smaller.get_num_points() == 4
    assert {tuple(point) for point in smaller.get_points().tolist()} == {
        (0.0, 1.0, 0.0),
        (1.0, 1.0, 0.0),
    }

    empty = _cloud(None)
    empty.align_points_with_larger(larger)
    assert torch.equal(empty.get_points(), torch.zeros((4, 3)))


def test_pointwise_become_partial_takes_a_slice_of_the_other_cloud(scene):
    whole = _cloud([[float(x), 0.0, 0.0] for x in range(4)])
    part = _cloud(None)
    part.pointwise_become_partial(whole, 0.25, 0.75)
    assert part.get_points()[:, 0].tolist() == [1.0, 2.0]


def test_interpolate_color_blends_colors_and_stroke_widths(scene):
    with Off():
        first = PMobject(points=[[0.0, 0.0, 0.0]], color=RED, stroke_width=2)
        second = PMobject(points=[[0.0, 0.0, 0.0]], color=BLUE, stroke_width=6)
        mixed = PMobject(points=[[0.0, 0.0, 0.0]])
    mixed.interpolate_color(first, second, 0.5)

    assert torch.allclose(
        mixed.get_all_rgbas()[0, :3], (_rgb(RED) + _rgb(BLUE)) / 2, atol=1e-6
    )
    assert mixed.get_stroke_width() == pytest.approx(4)
    assert mixed.point_radius == pytest.approx(0.04)


def test_get_point_mobject_sits_at_the_mean_point(scene):
    cloud = _cloud([[0.0, 0.0, 0.0], [2.0, 4.0, 0.0]])
    point = cloud.get_point_mobject()
    assert torch.allclose(point.location.reshape(-1), torch.tensor([1.0, 2.0, 0.0]))


def test_stroke_width_sets_the_sphere_radius(scene):
    cloud = _cloud([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    cloud.set_stroke_width(12)
    assert cloud.get_stroke_width() == 12
    assert cloud.point_radius == pytest.approx(0.12)


def test_the_color_of_an_empty_cloud_is_the_one_it_was_given(scene):
    cloud = _cloud(None, color=GREEN)
    assert cloud.get_color() is GREEN
    cloud.set_color(RED)
    assert cloud.get_color() is RED


def test_reset_points_removes_the_spheres(scene):
    cloud = _cloud([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    cloud.reset_points()
    assert cloud.get_num_points() == 0
    assert cloud.children == []
    assert cloud.get_render_primitives() is None


def test_add_line_samples_points_at_the_density(scene):
    with Off():
        line = Mobject1D(density=4)
    line.add_line([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    assert line.get_points()[:, 0].tolist() == [0.0, 0.25, 0.5, 0.75]

    line.add_line([5.0, 0.0, 0.0], [5.0, 0.0, 0.0])
    assert line.get_points()[-1].tolist() == [5.0, 0.0, 0.0]


def test_point_cloud_dot_fills_a_disc(scene):
    with Off():
        dot = PointCloudDot(center=RIGHT, radius=0.5, density=10)
    offsets = dot.get_points() - torch.tensor([1.0, 0.0, 0.0])
    assert dot.get_num_points() > 20
    assert float(offsets[:, :2].norm(dim=-1).max()) < 0.5
    assert torch.all(offsets[:, 2] == 0)


def test_pgroup_only_holds_point_clouds(scene):
    with Off():
        cloud = PMobject(points=[[0.0, 0.0, 0.0]], add_to_scene=False)
        with pytest.raises(AlganConfigurationError, match="PMobject"):
            PGroup(cloud, Dot3D(add_to_scene=False))


def test_dot_cloud_to_grid_lays_points_out_at_the_height(scene):
    with Off():
        cloud = DotCloud(color=YELLOW)
    cloud.to_grid(3, 2, height=4)

    points = cloud.get_points()
    assert len(points) == 6
    assert len({round(float(x), 4) for x in points[:, 0]}) == 2
    assert len({round(float(y), 4) for y in points[:, 1]}) == 3
    assert float(points[:, 1].max() - points[:, 1].min()) == pytest.approx(4)
    assert torch.allclose(points.mean(0), torch.zeros(3), atol=1e-6)


def test_true_dot_is_one_point_at_its_center(scene):
    with Off():
        dot = TrueDot(center=UP)
    assert dot.get_points().tolist() == [[0.0, 1.0, 0.0]]
