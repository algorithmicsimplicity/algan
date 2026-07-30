from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from algan.constants.color import BLUE
from algan.mobs.shapes_3d import Sphere
from algan.rendering.logical_pn import (
    evaluate_logical_pn,
    evaluate_logical_pn_normals,
    logical_pn_control_points,
    logical_pn_normal_control_points,
    subdivision_triangle_uvs,
)
from algan.rendering.raytracing.primitives import (
    LogicalPNTrianglePrimitive,
    RayTracedPNTrianglePrimitive,
    RayTracedTrianglePrimitive,
)
from algan.scene_manager import SceneManager
from algan.utils.memory_utils import ManualMemory


@pytest.fixture(autouse=True)
def reset_scene():
    SceneManager.reset()
    yield
    SceneManager.reset()


def _logical_patch(corners, normals, render_tolerance=0.5):
    source = LogicalPNTrianglePrimitive(
        corners=corners.reshape(1, 3, 3),
        normals=normals.reshape(1, 3, 3),
        colors=BLUE,
        render_tolerance=render_tolerance,
    )
    return LogicalPNTrianglePrimitive(triangle_collection=[source])


def _camera(z_positions, *, screen_height=480, device=None):
    z_positions = torch.as_tensor(z_positions, dtype=torch.float32, device=device)
    origins = torch.zeros((len(z_positions), 1, 3), device=device)
    origins[..., 2] = z_positions.view(-1, 1)
    screen_points = origins.clone()
    screen_points[..., 2] += 1.0
    return SimpleNamespace(
        ray_origin=origins,
        screen_point=screen_points,
        screen_basis=torch.eye(3, device=device)
        .unsqueeze(0)
        .repeat(len(z_positions), 1, 1),
        screen_width=640,
        screen_height=screen_height,
        output_screen_width=640,
        output_screen_height=screen_height,
        analytic_raster=False,
    )


def _curved_patch_inputs(device=None):
    corners = torch.tensor(
        [[-1.0, -0.7, 0.0], [1.0, -0.7, 0.0], [0.0, 1.0, 0.0]],
        device=device,
    )
    normals = torch.nn.functional.normalize(
        torch.tensor(
            [[-0.8, 0.0, 1.0], [0.8, 0.0, 1.0], [0.0, 0.8, 1.0]],
            device=device,
        ),
        dim=-1,
    )
    return corners, normals


def test_standard_pn_patch_is_flat_for_coplanar_equal_normals():
    corners = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    normals = torch.tensor([[0.0, 0.0, 1.0]]).repeat(3, 1)
    position_controls = logical_pn_control_points(
        corners.view(1, 1, 3, 3), normals.view(1, 1, 3, 3)
    )
    normal_controls = logical_pn_normal_control_points(
        corners.view(1, 1, 3, 3), normals.view(1, 1, 3, 3)
    )
    triangle_uv = subdivision_triangle_uvs(
        3, device=corners.device, dtype=corners.dtype
    )

    positions = evaluate_logical_pn(position_controls, triangle_uv)
    evaluated_normals = evaluate_logical_pn_normals(normal_controls, triangle_uv)

    assert position_controls.shape[-2] == 10
    assert normal_controls.shape[-2] == 6
    assert torch.count_nonzero(positions[..., 2]) == 0
    torch.testing.assert_close(
        evaluated_normals,
        torch.tensor([0.0, 0.0, 1.0]).expand_as(evaluated_normals),
    )


def test_adjacent_logical_pn_patches_share_their_curved_edge():
    p0 = torch.tensor([-1.0, 0.0, 0.0])
    p1 = torch.tensor([1.0, 0.0, 0.0])
    p2 = torch.tensor([0.0, 1.0, 0.2])
    p3 = torch.tensor([0.0, -1.0, -0.2])
    n0 = torch.nn.functional.normalize(torch.tensor([-0.4, 0.0, 1.0]), dim=0)
    n1 = torch.nn.functional.normalize(torch.tensor([0.4, 0.0, 1.0]), dim=0)
    n2 = torch.nn.functional.normalize(torch.tensor([0.0, 0.4, 1.0]), dim=0)
    n3 = torch.nn.functional.normalize(torch.tensor([0.0, -0.4, 1.0]), dim=0)
    corners = torch.stack(
        (torch.stack((p0, p1, p2)), torch.stack((p1, p0, p3)))
    ).unsqueeze(0)
    normals = torch.stack(
        (torch.stack((n0, n1, n2)), torch.stack((n1, n0, n3)))
    ).unsqueeze(0)
    position_controls = logical_pn_control_points(corners, normals)
    normal_controls = logical_pn_normal_control_points(corners, normals)
    t = torch.linspace(0.0, 1.0, 65)
    forward_uv = torch.stack((t, torch.zeros_like(t)), dim=-1)
    reverse_uv = torch.stack((1.0 - t, torch.zeros_like(t)), dim=-1)

    positions = evaluate_logical_pn(position_controls, forward_uv)
    reverse_positions = evaluate_logical_pn(
        position_controls[:, 1:], reverse_uv
    )
    evaluated_normals = evaluate_logical_pn_normals(
        normal_controls, forward_uv
    )
    reverse_normals = evaluate_logical_pn_normals(
        normal_controls[:, 1:], reverse_uv
    )

    torch.testing.assert_close(positions[:, :1], reverse_positions)
    torch.testing.assert_close(evaluated_normals[:, :1], reverse_normals)


def test_camera_distance_selects_per_frame_subdivision_and_batch_padding():
    corners, normals = _curved_patch_inputs()
    primitive = _logical_patch(corners, normals, render_tolerance=0.0005)
    camera = _camera([-30.0, -3.0], device=primitive.corners.device)

    primitive._dice_logical_pn(camera)

    far_level, close_level = primitive._logical_pn_subdivision_levels.tolist()
    assert close_level > far_level
    assert primitive.corners.shape[1] == 4**close_level
    assert primitive._logical_pn_padding[0].sum() > 0
    assert primitive._logical_pn_padding[1].sum() == 0


def test_batch_max_padding_does_not_change_per_frame_flat_mesh():
    corners, normals = _curved_patch_inputs()
    together = _logical_patch(corners, normals, render_tolerance=0.5)
    together._dice_logical_pn(
        _camera([-30.0, -3.0], device=together.corners.device)
    )

    for frame, camera_z in enumerate((-30.0, -3.0)):
        separate = _logical_patch(corners, normals, render_tolerance=0.5)
        separate._dice_logical_pn(
            _camera([camera_z], device=separate.corners.device)
        )
        count = separate.corners.shape[1]
        assert (
            together._logical_pn_subdivision_levels[frame]
            == separate._logical_pn_subdivision_levels[0]
        )
        torch.testing.assert_close(
            together.corners[frame, :count], separate.corners[0]
        )
        torch.testing.assert_close(
            together.normals[frame, :count], separate.normals[0]
        )
        torch.testing.assert_close(
            together.colors[frame, :count], separate.colors[0]
        )


def test_selected_flat_mesh_meets_dense_output_pixel_error_check():
    tolerance = 0.0005
    corners, normals = _curved_patch_inputs()
    primitive = _logical_patch(corners, normals, render_tolerance=tolerance)
    camera = _camera([-3.0], device=primitive.corners.device)
    position_controls = logical_pn_control_points(primitive.corners, primitive.normals)
    cam_o = camera.ray_origin.reshape(1, 3)
    screen_point = camera.screen_point.reshape(1, 3)
    level = int(
        primitive._required_subdivision_levels(
            position_controls,
            cam_o,
            screen_point,
            camera.screen_basis,
            camera.output_screen_height,
        ).item()
    )
    triangle_uv = subdivision_triangle_uvs(
        level,
        device=primitive.corners.device,
        dtype=primitive.corners.dtype,
    )
    denominator = 16
    sample_weights = torch.tensor(
        [
            (i / denominator, j / denominator, (denominator - i - j) / denominator)
            for i in range(denominator + 1)
            for j in range(denominator + 1 - i)
        ],
        device=primitive.corners.device,
        dtype=primitive.corners.dtype,
    )
    sample_uv = torch.einsum("sk,mka->msa", sample_weights, triangle_uv)
    exact = evaluate_logical_pn(position_controls, sample_uv)
    vertices = evaluate_logical_pn(position_controls, triangle_uv)
    approximated = torch.einsum("sk,tpmkc->tpmsc", sample_weights, vertices)
    exact_pixels, _ = primitive._project_to_output_pixels(
        exact,
        cam_o,
        screen_point,
        camera.screen_basis,
        camera.output_screen_height,
    )
    approximated_pixels, _ = primitive._project_to_output_pixels(
        approximated,
        cam_o,
        screen_point,
        camera.screen_basis,
        camera.output_screen_height,
    )

    assert (exact_pixels - approximated_pixels).norm(dim=-1).max() <= (tolerance * camera.output_screen_height)


def test_logical_pn_packs_only_regular_flat_triangle_geometry():
    corners, normals = _curved_patch_inputs()
    primitive = _logical_patch(corners, normals, render_tolerance=0.5)
    camera = _camera([-30.0, -3.0], device=primitive.corners.device)
    primitive.memory = ManualMemory(0, device=primitive.corners.device, managed=False)

    result = primitive.project_to_screen(camera, [])

    assert result is primitive
    assert isinstance(primitive, RayTracedTrianglePrimitive)
    assert not isinstance(primitive, RayTracedPNTrianglePrimitive)
    assert hasattr(primitive, "_rt_tri_pos")
    assert not hasattr(primitive, "_rt_pn_ctrl")
    assert primitive._rt_tri_pos.shape[0] == 2
    padded = primitive._logical_pn_padding
    empty = primitive._rt_frame_lo[..., 0] > primitive._rt_frame_hi[..., 0]
    assert torch.equal(empty, padded)


def test_geometry_tolerance_is_absolute_at_construction_scale():
    tolerance = 0.05
    unit = Sphere(
        radius=1,
        geometry_tolerance=tolerance,
        max_grid_resolution=80,
    )
    doubled = Sphere(
        radius=2,
        geometry_tolerance=tolerance,
        max_grid_resolution=80,
    )

    unit_error = unit._compute_pn_geometry_error(
        unit.coord_function_active, unit.grid_width, unit.grid_height
    )
    doubled_error = doubled._compute_pn_geometry_error(
        doubled.coord_function_active,
        doubled.grid_width,
        doubled.grid_height,
    )

    assert unit_error <= tolerance
    assert doubled_error <= tolerance
    assert doubled.grid_width >= unit.grid_width
    assert doubled.grid_height >= unit.grid_height


def test_surface_builds_new_logical_pn_primitive_with_both_tolerances():
    sphere = Sphere(
        geometry_tolerance=0.04,
        render_tolerance=0.75,
        max_grid_resolution=80,
    )

    primitive = sphere.get_render_primitives()

    assert isinstance(primitive, LogicalPNTrianglePrimitive)
    assert not isinstance(primitive, RayTracedPNTrianglePrimitive)
    assert sphere.geometry_tolerance == 0.04
    assert sphere.render_tolerance == 0.75
    assert primitive.render_tolerance == 0.75
