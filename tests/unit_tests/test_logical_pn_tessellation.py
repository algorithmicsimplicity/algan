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
    logical_pn_edge_control_points,
    logical_pn_normal_control_points,
    snap_boundary_values,
    subdivision_boundary_map,
    subdivision_triangle_indices,
    subdivision_triangle_uvs,
    subdivision_vertex_uvs,
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
    """A one-patch primitive; ``corners``/``normals`` are ``[3, 3]``."""
    return _logical_patches(
        corners.reshape(1, 3, 3), normals.reshape(1, 3, 3), render_tolerance
    )


def _logical_patches(corners, normals, render_tolerance=0.5):
    """A multi-patch primitive; ``corners``/``normals`` are ``[P, 3, 3]``."""
    source = LogicalPNTrianglePrimitive(
        corners=corners.reshape(1, -1, 3),
        normals=normals.reshape(1, -1, 3),
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

    (far_level,), (close_level,) = (
        primitive._logical_pn_subdivision_levels.tolist()
    )
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
        assert torch.equal(
            together._logical_pn_subdivision_levels[frame],
            separate._logical_pn_subdivision_levels[0],
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


def _dense_sample_weights(denominator, device, dtype):
    return torch.tensor(
        [
            (i / denominator, j / denominator, (denominator - i - j) / denominator)
            for i in range(denominator + 1)
            for j in range(denominator + 1 - i)
        ],
        device=device,
        dtype=dtype,
    )


def test_selected_flat_mesh_meets_dense_output_pixel_error_check():
    tolerance = 0.0005
    corners, normals = _curved_patch_inputs()
    primitive = _logical_patch(corners, normals, render_tolerance=tolerance)
    camera = _camera([-3.0], device=primitive.corners.device)
    device = primitive.corners.device
    dtype = primitive.corners.dtype
    position_controls = logical_pn_control_points(
        primitive.corners, primitive.normals
    )
    cam = (
        camera.ray_origin.reshape(1, 3),
        camera.screen_point.reshape(1, 3),
        camera.screen_basis,
    )
    levels, edge_levels = _levels_for(primitive, camera)
    level = int(levels[0, 0])

    # Re-measure the chosen dice against a far denser sample set than the level
    # search uses, both as the interior approximation alone and as the geometry
    # the renderer actually emits (boundary snapping included).
    triangle_uv = subdivision_triangle_uvs(level, device=device, dtype=dtype)
    triangle_indices = subdivision_triangle_indices(level, device=device)
    interior = evaluate_logical_pn(
        position_controls,
        subdivision_vertex_uvs(level, device=device, dtype=dtype),
    )[0]
    emitted = snap_boundary_values(
        interior,
        level,
        edge_levels[0],
        subdivision_boundary_map(level, device=device),
    )
    sample_weights = _dense_sample_weights(16, device, dtype)
    sample_uv = torch.einsum("sk,mka->msa", sample_weights, triangle_uv)
    exact_pixels, _ = primitive._project_to_output_pixels(
        evaluate_logical_pn(position_controls, sample_uv),
        *cam,
        camera.output_screen_height,
    )

    def pixel_error(vertices):
        approximated = torch.einsum(
            "sk,pmkc->pmsc", sample_weights, vertices[:, triangle_indices]
        ).unsqueeze(0)
        approximated_pixels, _ = primitive._project_to_output_pixels(
            approximated, *cam, camera.output_screen_height
        )
        return (exact_pixels - approximated_pixels).norm(dim=-1).max()

    limit = tolerance * camera.output_screen_height
    # The interior dice is held to the tolerance on its own...
    assert pixel_error(interior) <= limit
    # ...and the boundary snap, itself within the tolerance, can only add to it.
    assert pixel_error(emitted) <= 2 * limit


def _levels_for(primitive, camera):
    """The (patch, edge) levels the primitive would dice this camera at."""
    # _dice_logical_pn broadcasts the source geometry across the camera's
    # frames before the level search sees it.
    frames = camera.ray_origin.shape[0]
    corners, normals = primitive.corners, primitive.normals
    controls = logical_pn_control_points(corners, normals)
    edges = logical_pn_edge_control_points(corners, normals)
    return primitive._required_subdivision_levels(
        controls.expand(frames, *controls.shape[1:]),
        edges.expand(frames, *edges.shape[1:]),
        camera.ray_origin.reshape(-1, 3),
        camera.screen_point.reshape(-1, 3),
        camera.screen_basis,
        camera.output_screen_height,
    )


def _patch_levels_for(primitive, camera):
    return _levels_for(primitive, camera)[0]


def _sideways_camera(offsets, *, screen_height=480):
    """Cameras displaced sideways but still aimed down +z, as `orbit` leaves
    them: the subject swings out of frame without the camera following.
    """
    offsets = torch.as_tensor(offsets, dtype=torch.float32)
    origins = torch.zeros((len(offsets), 1, 3))
    origins[..., 0] = offsets.view(-1, 1)
    origins[..., 2] = -3.0
    screen_points = origins.clone()
    screen_points[..., 2] += 1.0
    return SimpleNamespace(
        ray_origin=origins,
        screen_point=screen_points,
        screen_basis=torch.eye(3).unsqueeze(0).repeat(len(offsets), 1, 1),
        screen_width=640,
        screen_height=screen_height,
        output_screen_width=640,
        output_screen_height=screen_height,
        analytic_raster=False,
    )


def test_off_frame_geometry_does_not_drive_subdivision():
    corners, normals = _curved_patch_inputs()
    primitive = _logical_patch(corners, normals, render_tolerance=0.0005)

    # Framed, then swung progressively further out of frame. Screen-space
    # error grows without bound as the patch approaches the camera plane, so
    # without the guard box these levels would climb monotonically.
    levels = _patch_levels_for(
        primitive, _sideways_camera([0.0, 20.0, 400.0]))[:, 0].tolist()

    assert levels[0] > 0, "in-frame patch should still be subdivided"
    assert levels[2] == 0, "wholly off-frame patch needs no subdivision"
    assert levels[1] <= levels[0]


def test_off_frame_camera_plane_straddler_needs_no_subdivision():
    # A patch spanning the camera plane has no finite screen error, so it used
    # to be forced to max_subdivision_level -- 4**8 triangles per patch, which
    # is unallocatable for any real mesh -- however far off frame it was. Its
    # in-front samples decide now, and here they all project way outside.
    corners = torch.tensor(
        [[299.0, -0.7, -6.0], [301.0, -0.7, 4.0], [300.0, 1.0, 4.0]])
    normals = torch.nn.functional.normalize(
        torch.tensor([[-0.8, 0.0, 1.0], [0.8, 0.0, 1.0], [0.0, 0.8, 1.0]]),
        dim=-1)
    primitive = _logical_patch(corners, normals, render_tolerance=0.0005)

    assert int(_patch_levels_for(primitive, _camera([-3.0]))[0, 0]) == 0


def test_subdivision_level_is_capped_by_the_triangle_budget():
    corners, normals = _curved_patch_inputs()
    primitive = _logical_patch(corners, normals, render_tolerance=0.0005)
    uncapped = int(_patch_levels_for(primitive, _camera([-3.0]))[0, 0])
    assert uncapped > 1

    primitive.max_diced_triangles = 4  # one patch, so 4**1 triangles
    with pytest.warns(RuntimeWarning):
        capped = int(_patch_levels_for(primitive, _camera([-3.0]))[0, 0])

    assert capped == 1


def test_budget_cap_does_not_depend_on_the_frame_window():
    # A level that moved with the render batch's frame count would pop at
    # batch boundaries.
    corners, normals = _curved_patch_inputs()
    primitive = _logical_patch(corners, normals, render_tolerance=0.0005)
    primitive.max_diced_triangles = 16

    with pytest.warns(RuntimeWarning):
        alone = _patch_levels_for(primitive, _camera([-3.0])).tolist()
    with pytest.warns(RuntimeWarning):
        batched = _patch_levels_for(
            primitive, _camera([-30.0, -3.0, -3.0])).tolist()

    assert batched[1:] == alone * 2


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


def _shifted(points, offset):
    return points + torch.tensor(offset, dtype=points.dtype)


def test_patches_in_one_frame_choose_their_own_subdivision_levels():
    # A near patch and a far one, side by side in the same frame. Under the old
    # per-frame level both cost 4 ** max(levels) triangles.
    corners, normals = _curved_patch_inputs()
    primitive = _logical_patches(
        torch.stack((corners, _shifted(corners, (0.0, 0.0, 240.0)))),
        torch.stack((normals, normals)),
        render_tolerance=0.0005,
    )
    camera = _camera([-3.0], device=primitive.corners.device)

    primitive._dice_logical_pn(camera)

    near, far = primitive._logical_pn_subdivision_levels[0].tolist()
    assert near > far, "the near patch must be diced more finely"
    assert primitive.corners.shape[1] == 4**near + 4**far
    assert primitive.corners.shape[1] < 2 * 4**near


def test_shared_edge_controls_do_not_depend_on_patch_orientation():
    # The crack-free guarantee rests on both patches deriving a shared edge's
    # level from bit-identical inputs, whichever way round they see it.
    corners, normals = _adjacent_patch_inputs()
    edge_controls = logical_pn_edge_control_points(corners, normals)

    # Edge 0 of patch 0 is (P0, P1); edge 0 of patch 1 is (P1, P0).
    assert torch.equal(edge_controls[0, 0, 0], edge_controls[0, 1, 0])


def _adjacent_patch_inputs(device=None):
    """Two patches sharing edge (p0, p1), seen in opposite orientations.

    The second patch is far more strongly curved, so the two want different
    interior levels for the same camera.
    """
    def normalize(v):
        return torch.nn.functional.normalize(v, dim=-1)

    p0 = torch.tensor([-1.0, 0.0, 0.0], device=device)
    p1 = torch.tensor([1.0, 0.0, 0.0], device=device)
    p2 = torch.tensor([0.0, 1.0, 0.05], device=device)
    p3 = torch.tensor([0.0, -1.0, -0.05], device=device)
    n0 = normalize(torch.tensor([-0.05, 0.0, 1.0], device=device))
    n1 = normalize(torch.tensor([0.05, 0.0, 1.0], device=device))
    n2 = normalize(torch.tensor([0.0, 0.05, 1.0], device=device))
    n3 = normalize(torch.tensor([0.6, -0.9, 1.0], device=device))
    corners = torch.stack(
        (torch.stack((p0, p1, p2)), torch.stack((p1, p0, p3)))
    ).unsqueeze(0)
    normals = torch.stack(
        (torch.stack((n0, n1, n2)), torch.stack((n1, n0, n3)))
    ).unsqueeze(0)
    return corners, normals


def _distance_to_polyline(points, polyline):
    """Distance from each of ``points`` to the polyline through ``polyline``."""
    start = polyline[:-1].unsqueeze(0)
    direction = (polyline[1:] - polyline[:-1]).unsqueeze(0)
    offset = points.unsqueeze(1) - start
    length_squared = (direction * direction).sum(-1).clamp_min(1e-20)
    t = ((offset * direction).sum(-1) / length_squared).clamp(0.0, 1.0)
    closest = start + t.unsqueeze(-1) * direction
    return (points.unsqueeze(1) - closest).norm(dim=-1).amin(-1)


def test_adjacent_patches_stay_watertight_at_different_levels():
    corners, normals = _adjacent_patch_inputs()
    primitive = _logical_patches(
        corners[0], normals[0], render_tolerance=0.0008
    )
    camera = _camera([-3.0], device=primitive.corners.device)

    primitive._dice_logical_pn(camera)

    levels = primitive._logical_pn_subdivision_levels[0].tolist()
    edge_levels = primitive._logical_pn_edge_levels[0]
    assert levels[0] != levels[1], "test needs the two patches to disagree"
    # Both patches must have derived the same level for the curve they share.
    assert int(edge_levels[0, 0]) == int(edge_levels[1, 0])

    # These patches are built so the shared curve is the whole of y == 0 and
    # nothing else in either patch touches it.
    counts = [4**level for level in levels]
    blocks = (
        primitive.corners[0, : counts[0]].reshape(-1, 3),
        primitive.corners[0, counts[0]: counts[0] + counts[1]].reshape(-1, 3),
    )
    seams = [block[block[:, 1].abs() < 1e-6] for block in blocks]
    for seam, level in zip(seams, levels):
        assert seam.unique(dim=0).shape[0] == 2**level + 1

    # Both patches must have laid their seam vertices on the *same* polyline;
    # any disagreement is a crack the background shows through.
    ordered = [seam[seam[:, 0].argsort()] for seam in seams]
    assert _distance_to_polyline(ordered[0], ordered[1]).max() < 1e-6
    assert _distance_to_polyline(ordered[1], ordered[0]).max() < 1e-6


def test_boundary_snap_is_a_no_op_when_the_levels_agree():
    device = torch.device("cpu")
    values = torch.randn(5, 15, 3, device=device)
    boundary = subdivision_boundary_map(2, device=device)
    levels = torch.full((5, 3), 2, dtype=torch.long, device=device)

    assert snap_boundary_values(values, 2, levels, boundary) is values

    levels[2, 1] = 1
    snapped = snap_boundary_values(values, 2, levels, boundary)
    changed = (snapped != values).any(-1).any(-1)
    assert changed.tolist() == [False, False, True, False, False]


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
