from __future__ import annotations

import numpy as np
import pytest
import torch

import algan
from algan.scene_manager import SceneManager


@pytest.fixture(autouse=True)
def reset_scene():
    SceneManager.reset()
    yield
    SceneManager.reset()


def materialize(*times: float):
    SceneManager.instance().current_scene.timeline_manager.set_state_to_times(
        torch.tensor(times, dtype=torch.get_default_dtype())
    )


def test_high_value_animation_api_is_exported():
    expected = {
        "MoveAlongPath",
        "ApplyPointwiseFunction",
        "ApplyMatrix",
        "ApplyComplexFunction",
        "Homotopy",
        "ComplexHomotopy",
        "PhaseFlow",
        "AnimatedBoundary",
    }
    assert expected <= set(vars(algan))


def test_apply_wave_materializes_vectorized_point_geometry():
    circle = algan.Circle(add_to_scene=False).spawn(False)
    initial = circle.control_points.location.clone()

    assert (
        algan.ApplyWave(
            circle,
            direction=algan.UP,
            amplitude=0.2,
            duration=1,
        )
        is circle
    )

    materialize(0.5)
    assert circle.control_points.location.shape == initial.shape
    assert not torch.allclose(circle.control_points.location, initial)


def test_show_passing_flash_broadcasts_static_curve_across_frames():
    square = algan.Square(add_to_scene=False)
    point_count = square.control_points.location.shape[-2]

    assert (
        algan.ShowPassingFlash(
            square,
            time_width=0.2,
            duration=1,
        )
        is square
    )

    materialize(0.2, 0.5, 0.8)
    assert square.control_points.location.shape == (3, point_count, 3)


def test_move_along_path_uses_arc_length_and_materializes_batches():
    path = algan.Line(algan.LEFT, algan.RIGHT, add_to_scene=False).spawn(False)
    dot = algan.Dot(add_to_scene=False).spawn(False)

    algan.MoveAlongPath(
        dot,
        path,
        duration=1,
        rate_func=algan.rate_funcs.identity,
    )
    materialize(0, 0.5, 0.999999)

    assert torch.allclose(
        dot.location[:, 0],
        torch.tensor([[-1, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=dot.location.dtype),
        atol=2e-5,
    )

    SceneManager.reset()
    path = algan.Line(algan.LEFT, algan.RIGHT, add_to_scene=False).spawn(False)
    dot = algan.Dot(add_to_scene=False).spawn(False)
    with algan.Sync(duration=1, rate_func=algan.rate_funcs.identity):
        path.move(algan.UP)
        algan.MoveAlongPath(dot, path, duration=1, rate_func=algan.rate_funcs.identity)
    materialize(0.5)
    assert torch.allclose(dot.location[0, 0], torch.tensor([0.0, 0.5, 0.0]), atol=2e-5)


def test_apply_matrix_supports_manim_argument_order_and_midpoint_state():
    square = algan.Square(add_to_scene=False).spawn(False)
    initial = square.control_points.location.clone()

    algan.ApplyMatrix(
        [[2, 0], [0, 3]],
        square,
        duration=1,
        rate_func=algan.rate_funcs.identity,
    )
    materialize(0, 0.5, 0.999999)

    scale = torch.tensor([2.0, 3.0, 1.0], dtype=initial.dtype)
    expected_final = initial * scale
    assert torch.allclose(square.control_points.location[0], initial[0], atol=1e-6)
    assert torch.allclose(
        square.control_points.location[1],
        torch.lerp(initial[0], expected_final[0], 0.5),
        atol=2e-5,
    )
    assert torch.allclose(
        square.control_points.location[2], expected_final[0], atol=2e-5
    )


def test_pointwise_function_has_vectorized_and_scalar_callback_paths():
    vectorized = algan.Line([0, 0, 0], [1, 0, 0], add_to_scene=False).spawn(False)
    scalar = algan.Line([0, 0, 0], [1, 0, 0], add_to_scene=False).spawn(False)
    vectorized_initial = vectorized.control_points.location.clone()
    scalar_initial = scalar.control_points.location.clone()

    def scalar_only(point):
        point = np.asarray(point)
        if point.ndim != 1:
            raise TypeError("one point at a time")
        return point + np.array([-1.0, 0.5, 0.0])

    with algan.Sync():
        algan.ApplyPointwiseFunction(
            vectorized,
            lambda points: points + torch.tensor([1.0, 2.0, 0.0]),
            rate_func=algan.rate_funcs.identity,
        )
        algan.ApplyPointwiseFunction(
            scalar_only,
            scalar,
            rate_func=algan.rate_funcs.identity,
        )
    materialize(0.999999)

    assert torch.allclose(
        vectorized.control_points.location,
        vectorized_initial + torch.tensor([1.0, 2.0, 0.0]),
        atol=2e-5,
    )
    assert torch.allclose(
        scalar.control_points.location,
        scalar_initial + torch.tensor([-1.0, 0.5, 0.0]),
        atol=2e-5,
    )


def test_homotopy_accepts_manim_scalar_api_and_surface_geometry():
    circle = algan.Circle(add_to_scene=False).spawn(False)
    circle_initial = circle.control_points.location.clone()
    algan.Homotopy(
        lambda x, y, z, t: (x, y + t, z),
        circle,
        duration=1,
        rate_func=algan.rate_funcs.identity,
    )
    materialize(0, 0.5, 0.999999)
    offsets = circle.control_points.location - circle_initial.expand(3, -1, -1)
    assert torch.allclose(
        offsets[..., 1].mean(-1), torch.tensor([0.0, 0.5, 1.0]), atol=2e-5
    )

    SceneManager.reset()
    surface = algan.Surface(
        lambda u, v: (u, v, 0),
        resolution=(2, 2),
        add_to_scene=False,
    ).spawn(False)
    initial_grid = surface.grid.location.clone()
    algan.Homotopy(
        surface,
        lambda points, t: (
            points + torch.cat((torch.zeros_like(t), torch.zeros_like(t), t), dim=-1)
        ),
        duration=1,
        rate_func=algan.rate_funcs.identity,
    )
    materialize(0.5)
    assert torch.allclose(
        surface.grid.location[..., 2],
        initial_grid[..., 2] + 0.5,
        atol=2e-5,
    )
    assert surface.get_render_primitives() is not None


def test_complex_transforms_preserve_z_and_accept_numpy_callbacks():
    line = algan.Line([1, 0, 2], [2, 0, 2], add_to_scene=False).spawn(False)
    initial = line.control_points.location.clone()
    algan.ApplyComplexFunction(
        lambda z: np.asarray(z) * 1j,
        line,
        duration=1,
        rate_func=algan.rate_funcs.identity,
    )
    materialize(0.999999)

    result = line.control_points.location
    assert torch.allclose(result[..., 0], -initial[..., 1], atol=2e-5)
    assert torch.allclose(result[..., 1], initial[..., 0], atol=2e-5)
    assert torch.allclose(result[..., 2], initial[..., 2], atol=2e-5)

    SceneManager.reset()
    line = algan.Line([1, 0, 3], [2, 0, 3], add_to_scene=False).spawn(False)
    initial = line.control_points.location.clone()
    algan.ComplexHomotopy(
        lambda z, t: z * torch.exp(1j * torch.pi * t / 2),
        line,
        duration=1,
        rate_func=algan.rate_funcs.identity,
    )
    materialize(0.5)
    expected = initial[..., 0] / np.sqrt(2)
    assert torch.allclose(line.control_points.location[..., 0], expected, atol=2e-5)
    assert torch.allclose(line.control_points.location[..., 1], expected, atol=2e-5)
    assert torch.allclose(
        line.control_points.location[..., 2], initial[..., 2], atol=2e-5
    )


def test_phase_flow_is_deterministic_across_frame_batches():
    line = algan.Line([0, 0, 0], [1, 0, 0], add_to_scene=False).spawn(False)
    initial = line.control_points.location.clone()
    algan.PhaseFlow(
        lambda points: torch.ones_like(points) * torch.tensor([1.0, 2.0, 0.0]),
        line,
        virtual_time=2,
        integration_steps=4,
        duration=1,
    )

    materialize(0.25, 0.75)
    first_batch = line.control_points.location.clone()
    materialize(0.75)
    isolated = line.control_points.location.clone()

    assert torch.allclose(first_batch[1], isolated[0], atol=1e-6)
    assert torch.allclose(
        isolated,
        initial + torch.tensor([1.5, 3.0, 0.0]),
        atol=2e-5,
    )

    SceneManager.reset()
    line = algan.Line([1, 0, 0], [2, 0, 0], add_to_scene=False).spawn(False)
    initial = line.control_points.location.clone()
    algan.PhaseFlow(
        lambda points: points * torch.tensor([1.0, 0.0, 0.0]),
        line,
        virtual_time=1,
        integration_steps=32,
        duration=1,
    )
    materialize(0.999999)
    assert torch.allclose(
        line.control_points.location[..., 0],
        initial[..., 0] * torch.e,
        rtol=2e-5,
        atol=2e-5,
    )


def test_recursive_replay_uses_rows_captured_before_descendant_rebatch():
    """A later descendant rebatch must not change an earlier edit's targets."""
    child = algan.Mob()
    group = algan.Group([child]).spawn(animate=False)
    start = SceneManager.instance().current_scene.animation_manager.context.timespan.current_time
    opacity_timeline = (
        SceneManager.instance().current_scene.timeline_manager.attr_to_timeline[
            "opacity"
        ]
    )
    old_child_rows = opacity_timeline.mob_id_to_inds[child.id].clone()

    with algan.Sync(rate_func=algan.rate_funcs.identity):
        group.opacity = 0
        child.set_non_recursive(opacity=torch.ones((1, 3, 1)))

    new_child_rows = opacity_timeline.mob_id_to_inds[child.id]
    assert new_child_rows.numel() == 3
    assert not torch.equal(old_child_rows, new_child_rows)

    materialize(start + 0.5)
    assert torch.allclose(group.opacity, torch.full_like(group.opacity, 0.5))
    assert torch.allclose(child.opacity, torch.full_like(child.opacity, 0.5))


def test_replay_distinguishes_recursive_edits_from_nonrecursive_reads():
    cylinder = algan.Cylinder(resolution=(4, 4), add_to_scene=False).spawn(
        animate=False
    )
    cylinder.set_start_point(algan.LEFT)

    materialize(0.5)

    assert cylinder.location.shape == (1, 1, 3)
    assert cylinder.basis.shape == (1, 1, 9)


def test_surface_logical_pn_topology_is_fixed_during_animation():
    cylinder = algan.Cylinder(add_to_scene=False).spawn(animate=False)
    initial_basis = cylinder.basis.clone()
    initial_resolution = (cylinder.grid_width, cylinder.grid_height)
    initial_grid_rows = cylinder.grid.location.shape[-2]

    with algan.Sync(duration=1, rate_func=algan.rate_funcs.identity):
        cylinder.rotate(720, algan.OUT)
        cylinder.move_off_screen(algan.LEFT, despawn=False)

    assert (cylinder.grid_width, cylinder.grid_height) == initial_resolution
    assert cylinder.grid.location.shape[-2] == initial_grid_rows
    rotate_event = next(
        event
        for event in (
            SceneManager.instance().current_scene.timeline_manager.function_timeline.function_applications
        )
        if event.function.__name__ == "rotate"
    )
    assert rotate_event.caller is cylinder

    materialize(0.125)
    assert not torch.allclose(cylinder.basis, initial_basis)
    assert (cylinder.grid_width, cylinder.grid_height) == initial_resolution
    assert cylinder.grid.location.shape[-2] == initial_grid_rows


def test_surface_fixed_topology_preserves_parent_rotation_and_scale():
    fixed_group = algan.Group(
        [
            algan.Square(),
            algan.Circle(),
            algan.Sphere(),
            algan.Cylinder(),
        ]
    ).arrange_in_grid()
    fixed_group.spawn(animate=False)
    auto_group = (
        algan.Group(
            [
                algan.Square(),
                algan.Circle(),
                algan.Sphere(),
                algan.Cylinder(),
            ]
        )
        .arrange_in_grid()
        .spawn(animate=False)
    )
    initial_resolutions = [
        (mob.grid_width, mob.grid_height)
        for mob in auto_group
        if isinstance(mob, algan.Surface)
    ]

    with algan.Sync(duration=1, rate_func=algan.rate_funcs.identity):
        fixed_group.rotate(180, algan.UP).scale(0.75)
        auto_group.rotate(180, algan.UP).scale(0.75)

    materialize(0.25, 0.5, 0.75)

    for actual, expected in zip(auto_group, fixed_group):
        torch.testing.assert_close(
            actual.location,
            expected.location,
            atol=2e-5,
            rtol=0,
            msg=type(actual).__name__,
        )
    assert [
        (mob.grid_width, mob.grid_height)
        for mob in auto_group
        if isinstance(mob, algan.Surface)
    ] == initial_resolutions


def test_animated_boundary_tracks_source_and_can_stop():
    source = algan.Circle(add_to_scene=False).spawn(False)
    boundary = algan.AnimatedBoundary(
        source,
        colors=("#FF0000", algan.BLUE),
        cycle_rate=1,
        max_stroke_width=2,
        back_and_forth=False,
        add_to_scene=False,
    ).spawn(False)
    source.move(algan.RIGHT)

    materialize(0.25, 0.75, 1.25)
    growing = boundary._growing_paths[0]
    fading = boundary._fading_paths[0]

    assert growing.control_points.location.shape[0] == 3
    assert torch.allclose(
        growing.control_points.location[:, 0],
        source.control_points.location[:, 0],
        atol=1e-6,
    )
    assert torch.allclose(
        growing.stroke_width.reshape(3, -1)[:, 0],
        torch.full((3,), 2.0),
    )
    assert fading.stroke_width.reshape(3, -1)[0, 0] == 0
    assert fading.stroke_width.reshape(3, -1)[2, 0] > 0
    assert growing.get_render_primitives() is not None

    SceneManager.reset()
    source = algan.Circle(add_to_scene=False).spawn(False)
    boundary = algan.AnimatedBoundary(source, add_to_scene=False).spawn(False)
    assert boundary.stop() is boundary
    updater = SceneManager.instance().current_scene.timeline_manager.function_timeline.updaters[
        boundary.updater_id
    ]
    assert updater.time.end_event is not None


def test_invalid_geometry_and_parameters_fail_early():
    empty = algan.Mob(add_to_scene=False)
    with pytest.raises(TypeError, match="deformable"):
        algan.ApplyPointwiseFunction(empty, lambda points: points)
    with pytest.raises(ValueError, match="shape"):
        algan.ApplyMatrix(algan.Circle(add_to_scene=False), [[1, 0, 0]])
    with pytest.raises(ValueError, match="at least 1"):
        algan.PhaseFlow(
            algan.Circle(add_to_scene=False),
            lambda points: points,
            integration_steps=0,
        )
