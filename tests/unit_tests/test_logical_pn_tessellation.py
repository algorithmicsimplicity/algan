from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from algan.constants.color import BLUE
from algan.mobs.shapes_3d import Cone, Cylinder, Sphere, Torus
from algan.rendering.logical_pn import (
    EDGE_CORNERS,
    OPPOSITE_EDGE,
    dice_pattern,
    dice_triangle_count,
    evaluate_logical_pn,
    evaluate_logical_pn_normals,
    interpolate_patch_attribute,
    interpolate_patch_vertex_attribute,
    logical_pn_control_points,
    logical_pn_edge_control_points,
    logical_pn_normal_control_points,
    mean_patch_edge_length,
    snap_boundary_values,
    subdivision_boundary_map,
    subdivision_triangle_indices,
    subdivision_triangle_uvs,
    subdivision_vertex_uvs,
)
from algan.rendering.raytracing.primitives import (
    LogicalPNTrianglePrimitive,
    RayTracedTrianglePrimitive,
)
from algan.scene_manager import SceneManager
from algan.settings import SETTINGS
from algan.utils.memory_utils import ManualMemory


@pytest.fixture(autouse=True)
def reset_scene():
    SceneManager.reset()
    yield
    SceneManager.reset()


def _logical_patch(
    corners, normals, render_tolerance=0.5, render_tolerance_pixels=None
):
    """A one-patch primitive; ``corners``/``normals`` are ``[3, 3]``."""
    return _logical_patches(
        corners.reshape(1, 3, 3),
        normals.reshape(1, 3, 3),
        render_tolerance,
        render_tolerance_pixels,
    )


def _logical_patches(
    corners, normals, render_tolerance=0.5, render_tolerance_pixels=None
):
    """A multi-patch primitive; ``corners``/``normals`` are ``[P, 3, 3]``."""
    source = LogicalPNTrianglePrimitive(
        corners=corners.reshape(1, -1, 3),
        normals=normals.reshape(1, -1, 3),
        colors=BLUE,
        render_tolerance=render_tolerance,
        render_tolerance_pixels=render_tolerance_pixels,
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
    reverse_positions = evaluate_logical_pn(position_controls[:, 1:], reverse_uv)
    evaluated_normals = evaluate_logical_pn_normals(normal_controls, forward_uv)
    reverse_normals = evaluate_logical_pn_normals(normal_controls[:, 1:], reverse_uv)

    torch.testing.assert_close(positions[:, :1], reverse_positions)
    torch.testing.assert_close(evaluated_normals[:, :1], reverse_normals)


def test_camera_distance_selects_per_frame_subdivision_and_batch_padding():
    corners, normals = _curved_patch_inputs()
    primitive = _logical_patch(corners, normals, render_tolerance=0.0005)
    camera = _camera([-30.0, -3.0], device=primitive.corners.device)

    primitive._dice_logical_pn(camera)

    (far_level,), (close_level,) = primitive._logical_pn_subdivision_levels.tolist()
    counts = primitive._logical_pn_triangle_counts
    assert close_level > far_level
    # The batch is padded to its widest frame and no wider.
    assert primitive.corners.shape[1] == int(counts.sum(1).amax())
    assert primitive._logical_pn_padding[0].sum() > 0
    assert primitive._logical_pn_padding[1].sum() == 0


def test_batch_max_padding_does_not_change_per_frame_flat_mesh():
    corners, normals = _curved_patch_inputs()
    together = _logical_patch(corners, normals, render_tolerance=0.5)
    together._dice_logical_pn(_camera([-30.0, -3.0], device=together.corners.device))

    for frame, camera_z in enumerate((-30.0, -3.0)):
        separate = _logical_patch(corners, normals, render_tolerance=0.5)
        separate._dice_logical_pn(_camera([camera_z], device=separate.corners.device))
        count = separate.corners.shape[1]
        assert torch.equal(
            together._logical_pn_subdivision_levels[frame],
            separate._logical_pn_subdivision_levels[0],
        )
        torch.testing.assert_close(together.corners[frame, :count], separate.corners[0])
        torch.testing.assert_close(together.normals[frame, :count], separate.normals[0])
        torch.testing.assert_close(together.colors[frame, :count], separate.colors[0])


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
    position_controls = logical_pn_control_points(primitive.corners, primitive.normals)
    cam = (
        camera.ray_origin.reshape(1, 3),
        camera.screen_point.reshape(1, 3),
        camera.screen_basis,
    )
    levels, edge_levels, apex, across = _levels_for(primitive, camera)
    pattern = dice_pattern(
        int(levels[0, 0]),
        int(across[0, 0]),
        int(apex[0, 0]),
        device=device,
        dtype=dtype,
    )

    # Re-measure the chosen dice against a far denser sample set than the level
    # search uses, both as the interior approximation alone and as the geometry
    # the renderer actually emits (boundary snapping included). Whatever shape
    # the search settled on -- uniform or anisotropic -- is what is measured.
    triangle_indices = pattern.triangle_indices
    triangle_uv = pattern.vertex_uv[triangle_indices]
    interior = evaluate_logical_pn(position_controls, pattern.vertex_uv)[0]
    emitted = snap_boundary_values(
        interior,
        pattern.edge_levels,
        edge_levels[0],
        pattern.boundary,
    )
    sample_weights = _dense_sample_weights(16, device, dtype)
    sample_uv = torch.einsum("sk,mka->msa", sample_weights, triangle_uv)
    exact_pixels, _, _ = primitive._project_to_output_pixels(
        evaluate_logical_pn(position_controls, sample_uv),
        *cam,
        camera.output_screen_height,
    )

    def pixel_error(vertices):
        approximated = torch.einsum(
            "sk,pmkc->pmsc", sample_weights, vertices[:, triangle_indices]
        ).unsqueeze(0)
        approximated_pixels, _, _ = primitive._project_to_output_pixels(
            approximated, *cam, camera.output_screen_height
        )
        return (exact_pixels - approximated_pixels).norm(dim=-1).max()

    limit = tolerance * camera.output_screen_height
    # The interior dice is held to the tolerance on its own...
    assert pixel_error(interior) <= limit
    # ...and the boundary snap, itself within the tolerance, can only add to it.
    assert pixel_error(emitted) <= 2 * limit


def _levels_for(primitive, camera):
    """The (patch, edge, apex, across) levels the primitive would dice at."""
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
    levels = _patch_levels_for(primitive, _sideways_camera([0.0, 20.0, 400.0]))[
        :, 0
    ].tolist()

    assert levels[0] > 0, "in-frame patch should still be subdivided"
    assert levels[2] == 0, "wholly off-frame patch needs no subdivision"
    assert levels[1] <= levels[0]


def test_off_frame_camera_plane_straddler_needs_no_subdivision():
    # A patch spanning the camera plane has no finite screen error, so it used
    # to be forced to max_subdivision_level -- 4**8 triangles per patch, which
    # is unallocatable for any real mesh -- however far off frame it was. Its
    # in-front samples decide now, and here they all project way outside.
    corners = torch.tensor([[299.0, -0.7, -6.0], [301.0, -0.7, 4.0], [300.0, 1.0, 4.0]])
    normals = torch.nn.functional.normalize(
        torch.tensor([[-0.8, 0.0, 1.0], [0.8, 0.0, 1.0], [0.0, 0.8, 1.0]]), dim=-1
    )
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
        batched = _patch_levels_for(primitive, _camera([-30.0, -3.0, -3.0])).tolist()

    assert batched[1:] == alone * 2


def test_logical_pn_packs_only_regular_flat_triangle_geometry():
    corners, normals = _curved_patch_inputs()
    primitive = _logical_patch(corners, normals, render_tolerance=0.5)
    camera = _camera([-30.0, -3.0], device=primitive.corners.device)
    primitive.memory = ManualMemory(0, device=primitive.corners.device, managed=False)

    result = primitive.project_to_screen(camera, [])

    assert result is primitive
    assert isinstance(primitive, RayTracedTrianglePrimitive)
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
    near_count, far_count = primitive._logical_pn_triangle_counts[0].tolist()
    assert near > far, "the near patch must be diced more finely"
    assert primitive.corners.shape[1] == near_count + far_count
    assert primitive.corners.shape[1] < 2 * near_count


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
    primitive = _logical_patches(corners[0], normals[0], render_tolerance=0.0008)
    camera = _camera([-3.0], device=primitive.corners.device)

    primitive._dice_logical_pn(camera)

    levels = primitive._logical_pn_subdivision_levels[0].tolist()
    edge_levels = primitive._logical_pn_edge_levels[0]
    apex = primitive._logical_pn_apex[0].tolist()
    across = primitive._logical_pn_across_levels[0].tolist()
    assert levels[0] != levels[1], "test needs the two patches to disagree"
    # Both patches must have derived the same level for the curve they share.
    assert int(edge_levels[0, 0]) == int(edge_levels[1, 0])

    # These patches are built so the shared curve is the whole of y == 0 and
    # nothing else in either patch touches it.
    counts = primitive._logical_pn_triangle_counts[0].tolist()
    blocks = (
        primitive.corners[0, : counts[0]].reshape(-1, 3),
        primitive.corners[0, counts[0] : counts[0] + counts[1]].reshape(-1, 3),
    )
    seams = [block[block[:, 1].abs() < 1e-6] for block in blocks]
    # Each patch cuts the shared curve at its own dice's level for that edge --
    # which is the across level when its rows run parallel to the seam.
    for seam, level, patch_apex, patch_across in zip(seams, levels, apex, across):
        seam_level = patch_across if OPPOSITE_EDGE[patch_apex] == 0 else level
        assert seam.unique(dim=0).shape[0] == 2**seam_level + 1

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


def _materialized_per_frame(primitive, num_frames):
    """Give ``primitive`` one source row per frame, as materialization does.

    A mob that does not move still reaches the dice as ``num_frames``
    byte-identical rows -- the ``[1, ...]`` sources these fixtures build
    otherwise are not what a real render hands it.
    """
    for name in ("corners", "normals", "colors", *primitive._surface_params):
        value = getattr(primitive, name, None)
        if value is not None and value.shape[0] == 1:
            setattr(
                primitive, name, value.expand(num_frames, *value.shape[1:]).contiguous()
            )
    if primitive.uvs is not None and primitive.uvs.shape[0] == 1:
        primitive.uvs = primitive.uvs.expand(
            num_frames, *primitive.uvs.shape[1:]
        ).contiguous()
    primitive.shader_param_values = [
        value.expand(num_frames, *value.shape[1:]).contiguous()
        if value.shape[0] == 1
        else value
        for value in primitive.shader_param_values
    ]
    return primitive


def test_vertex_attribute_interpolation_matches_the_per_corner_form():
    # The dice interpolates attributes on the shared subdivision vertices and
    # gathers them through the triangle indices, which is only sound because a
    # microtriangle's corners ARE those vertices. That identity is exact and is
    # asserted as such; the interpolated values are only compared closely,
    # because the two forms contract [V, 3] and [M * 3, 3] weight matrices and
    # a BLAS is free to order a three-term sum differently between the two
    # shapes -- measured at one ulp here from level 2 up.
    torch.manual_seed(0)
    values = torch.randn(23, 3, 4)
    for level in range(4):
        corner_uv = subdivision_triangle_uvs(
            level, device=values.device, dtype=values.dtype
        )
        vertex_uv = subdivision_vertex_uvs(
            level, device=values.device, dtype=values.dtype
        )
        indices = subdivision_triangle_indices(level, device=values.device)
        assert torch.equal(vertex_uv[indices], corner_uv)
        torch.testing.assert_close(
            interpolate_patch_vertex_attribute(values, vertex_uv)[:, indices],
            interpolate_patch_attribute(values, corner_uv),
            rtol=1e-6,
            atol=1e-6,
        )


def test_frame_invariant_sources_dice_to_the_same_values(monkeypatch):
    # A mesh that does not move arrives as N identical source rows, and the
    # dice collapses them: one control net instead of N, and one evaluation per
    # distinct patch instead of one per (frame, patch). Nothing it writes may
    # move as a result.
    corners, normals = _curved_patch_inputs()
    two_patches = (
        torch.stack((corners, _shifted(corners, (0.0, 0.0, 240.0)))),
        torch.stack((normals, normals)),
    )
    frames = [-30.0, -6.0, -3.0]

    def fixture():
        return _materialized_per_frame(
            _logical_patches(*two_patches, render_tolerance=0.0005), len(frames)
        )

    shared, per_frame = fixture(), fixture()
    with monkeypatch.context() as patch:
        # Report every source as frame-varying: the pre-collapse code path.
        patch.setattr(
            LogicalPNTrianglePrimitive,
            "_collapse_redundant_frames",
            staticmethod(lambda value: (value, False)),
        )
        per_frame._dice_logical_pn(_camera(frames, device=per_frame.corners.device))
    shared._dice_logical_pn(_camera(frames, device=shared.corners.device))

    levels = shared._logical_pn_subdivision_levels
    assert levels.unique().numel() > 1, (
        "the fixture must dice its patches at more than one level"
    )
    assert shared._logical_pn_padding.any(), "the fixture must pad some frames"
    for name in ("corners", "normals", "colors", *shared._surface_params):
        assert torch.equal(getattr(shared, name), getattr(per_frame, name)), name
    for name in (
        "_logical_pn_padding",
        "_logical_pn_subdivision_levels",
        "_logical_pn_edge_levels",
        "_logical_pn_tri_obj",
    ):
        assert torch.equal(getattr(shared, name), getattr(per_frame, name)), name


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


def _dense_implicit_surface_error(shape):
    primitive = shape.get_render_primitives()
    controls = logical_pn_control_points(
        primitive.corners.reshape(-1, 3, 3),
        primitive.normals.reshape(-1, 3, 3),
    )
    dense_uv = subdivision_vertex_uvs(
        5,
        device=primitive.corners.device,
        dtype=primitive.corners.dtype,
    )
    dense_points = evaluate_logical_pn(controls, dense_uv)
    return shape._pn_geometry_deviation(dense_points, None, None).max()


def test_default_sphere_uses_compact_geometry_accurate_logical_topology():
    sphere = Sphere()

    error = sphere._compute_pn_geometry_error(
        sphere.coord_function_active,
        sphere.grid_width,
        sphere.grid_height,
    )

    dense_error = _dense_implicit_surface_error(sphere)

    assert sphere.grid_width * sphere.grid_height < 500
    assert error <= sphere.geometry_tolerance
    assert dense_error <= sphere.geometry_tolerance


@pytest.mark.parametrize(
    ("shape_type", "vertex_limit"),
    [
        pytest.param(Cylinder, 100, id="cylinder"),
        # A cone is straight along its slant, so it pays for its azimuth alone.
        # It used to tie the two axes together and spend 520 vertices here.
        pytest.param(Cone, 200, id="cone"),
        pytest.param(Torus, 1000, id="torus"),
    ],
)
def test_other_analytic_surfaces_use_compact_geometry_accurate_topology(
    shape_type,
    vertex_limit,
):
    shape = shape_type()

    assert shape._geometry_auto_resolution_enabled
    assert shape.grid_width * shape.grid_height < vertex_limit
    assert _dense_implicit_surface_error(shape) <= shape.geometry_tolerance


# --- the absolute-pixel render tolerance ------------------------------------


def _levels_at(screen_height, **tolerances):
    corners, normals = _curved_patch_inputs()
    primitive = _logical_patch(corners, normals, **tolerances)
    primitive._dice_logical_pn(
        _camera([-3.0], screen_height=screen_height, device=primitive.corners.device)
    )
    return primitive


@pytest.mark.parametrize("screen_height", [396, 486])
def test_pixel_tolerance_is_inert_where_the_screen_fraction_is_finer(screen_height):
    """At low resolutions the fraction already asks for sub-pixel triangles.

    ``render_tolerance=0.001`` is worth 0.4 px at ``PREVIEW`` and 0.49 px at
    ``LD``, so a 1 px absolute bound cannot bind there and the dice -- every
    vertex of it -- has to be the one the fraction alone chose.
    """
    without = _levels_at(screen_height, render_tolerance=0.001)
    with_pixels = _levels_at(
        screen_height, render_tolerance=0.001, render_tolerance_pixels=1.0
    )

    assert torch.equal(
        without._logical_pn_subdivision_levels,
        with_pixels._logical_pn_subdivision_levels,
    )
    assert torch.equal(without.corners, with_pixels.corners)


def test_pixel_tolerance_refines_the_dice_at_high_resolution():
    """Once the frame is large, the absolute bound is what binds.

    ``render_tolerance=0.001`` is worth 2.16 px at ``UHD``; asking for half a
    pixel there costs a subdivision level the fraction would not have bought.
    """
    without = _levels_at(2160, render_tolerance=0.001)
    with_pixels = _levels_at(2160, render_tolerance=0.001, render_tolerance_pixels=0.5)

    assert (
        with_pixels._logical_pn_subdivision_levels.amax()
        > without._logical_pn_subdivision_levels.amax()
    )
    assert with_pixels.corners.shape[1] > without.corners.shape[1]


@pytest.mark.parametrize("pixels", [2.16, 1.0, 0.5, 0.25])
def test_pixel_tolerance_dices_exactly_as_the_equivalent_fraction_would(pixels):
    """``p`` pixels of an ``H`` pixel frame is the fraction ``p / H``.

    The two spellings are the same bound, so they must reach the same dice --
    vertex for vertex -- rather than merely the same level.
    """
    by_pixels = _levels_at(2160, render_tolerance=0.5, render_tolerance_pixels=pixels)
    by_fraction = _levels_at(2160, render_tolerance=pixels / 2160)

    assert torch.equal(
        by_pixels._logical_pn_subdivision_levels,
        by_fraction._logical_pn_subdivision_levels,
    )
    assert torch.equal(by_pixels.corners, by_fraction.corners)


def test_pixel_tolerance_does_not_relax_a_finer_screen_fraction():
    """The finer of the two bounds wins; neither can loosen the other."""
    tight = _levels_at(480, render_tolerance=0.0005)
    with_loose_pixels = _levels_at(
        480, render_tolerance=0.0005, render_tolerance_pixels=8.0
    )

    assert torch.equal(
        tight._logical_pn_subdivision_levels,
        with_loose_pixels._logical_pn_subdivision_levels,
    )


def test_merged_primitive_takes_the_finest_pixel_tolerance_of_its_members():
    corners, normals = _curved_patch_inputs()
    members = [
        LogicalPNTrianglePrimitive(
            corners=corners.reshape(1, -1, 3),
            normals=normals.reshape(1, -1, 3),
            colors=BLUE,
            render_tolerance_pixels=pixels,
        )
        for pixels in (2.0, None, 0.5)
    ]

    merged = LogicalPNTrianglePrimitive(triangle_collection=members)

    assert merged.render_tolerance_pixels == 0.5
    # Members that differ only in this must not share a merge bucket.
    assert members[0].get_batch_identifier() != members[2].get_batch_identifier()


def test_absent_pixel_tolerance_leaves_the_screen_fraction_alone():
    primitive = _logical_patch(
        *_curved_patch_inputs(), render_tolerance=0.001, render_tolerance_pixels=None
    )

    assert primitive.render_tolerance_pixels == float("inf")
    assert primitive._pixel_threshold(2160) == pytest.approx(2.16)


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan")])
def test_non_positive_pixel_tolerance_is_rejected(value):
    with pytest.raises(ValueError, match="render_tolerance_pixels"):
        Sphere(render_tolerance_pixels=value)


def test_surface_defaults_to_a_one_pixel_absolute_tolerance():
    sphere = Sphere()

    primitive = sphere.get_render_primitives()

    assert sphere.render_tolerance_pixels == 1.0
    assert primitive.render_tolerance_pixels == 1.0
    # PREVIEW and LD are decided by the fraction; HD and above by the pixel.
    assert primitive._pixel_threshold(396) == pytest.approx(0.396)
    assert primitive._pixel_threshold(1080) == 1.0


def test_surface_builds_new_logical_pn_primitive_with_both_tolerances():
    sphere = Sphere(
        geometry_tolerance=0.04,
        render_tolerance=0.75,
        max_grid_resolution=80,
    )

    primitive = sphere.get_render_primitives()

    assert isinstance(primitive, LogicalPNTrianglePrimitive)
    assert sphere.geometry_tolerance == 0.04
    assert sphere.render_tolerance == 0.75
    assert primitive.render_tolerance == 0.75
    assert primitive.render_tolerance_pixels == sphere.render_tolerance_pixels


# --- per-dimension dicing ---------------------------------------------------


@pytest.mark.parametrize("level", [0, 1, 2, 3, 4])
def test_dice_pattern_reproduces_the_uniform_grid_when_both_levels_agree(level):
    """A patch that wants the same detail both ways dices as it always has."""
    device = torch.device("cpu")
    pattern = dice_pattern(level, level, 0, device=device, dtype=torch.float32)

    torch.testing.assert_close(
        pattern.vertex_uv[pattern.triangle_indices],
        subdivision_triangle_uvs(level, device=device, dtype=torch.float32),
        rtol=0,
        atol=0,
    )
    assert pattern.edge_levels == (level, level, level)


@pytest.mark.parametrize(
    ("along", "across"), [(3, 0), (3, 1), (4, 0), (4, 2), (5, 3), (6, 1)]
)
@pytest.mark.parametrize("apex", [0, 1, 2])
def test_anisotropic_dice_cuts_each_edge_at_its_own_level(along, across, apex):
    device = torch.device("cpu")
    pattern = dice_pattern(along, across, apex, device=device, dtype=torch.float32)
    uv = pattern.vertex_uv
    barycentric = torch.cat((1.0 - uv.sum(-1, keepdim=True), uv), -1)

    for edge, corners in enumerate(EDGE_CORNERS):
        opposite_corner = 3 - sum(corners)
        on_edge = barycentric[:, opposite_corner].abs() < 1e-6
        expected = across if edge == OPPOSITE_EDGE[apex] else along
        assert int(on_edge.sum()) - 1 == 2**expected
        assert pattern.edge_levels[edge] == expected

    # Every microtriangle keeps the uniform grid's winding, and together they
    # tile the patch exactly once (cross products sum to the unit triangle's).
    a, b, c = uv[pattern.triangle_indices].unbind(1)
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (
        c[:, 0] - a[:, 0]
    )
    assert bool((cross > 0).all())
    torch.testing.assert_close(cross.sum(), torch.tensor(1.0), rtol=0, atol=1e-5)
    assert pattern.triangle_count == int(
        dice_triangle_count(torch.tensor(along), torch.tensor(across))
    )
    assert pattern.triangle_count < 4**along


def _patch_boundary_points(primitive, source_corners, source_normals, patch, edge):
    """The points one patch emits along one of its edges, in edge order."""
    device = source_corners.device
    dtype = source_corners.dtype
    pattern = dice_pattern(
        int(primitive._logical_pn_subdivision_levels[0, patch]),
        int(primitive._logical_pn_across_levels[0, patch]),
        int(primitive._logical_pn_apex[0, patch]),
        device=device,
        dtype=dtype,
    )
    controls = logical_pn_control_points(
        source_corners[patch : patch + 1], source_normals[patch : patch + 1]
    )
    snapped = snap_boundary_values(
        evaluate_logical_pn(controls.unsqueeze(0), pattern.vertex_uv)[0],
        pattern.edge_levels,
        primitive._logical_pn_edge_levels[0, patch : patch + 1],
        pattern.boundary,
    )[0]
    ids = pattern.boundary.edge_vertex_ids[edge, : (1 << pattern.edge_levels[edge]) + 1]
    return snapped[ids]


@pytest.mark.parametrize(
    "build",
    [lambda: Cylinder(radius=0.5, height=2.0), Sphere],
    ids=["cylinder", "sphere"],
)
def test_a_whole_diced_mesh_stays_watertight_across_every_seam(build):
    """Every edge two patches share is the same polyline seen from both sides.

    The per-dimension dice hands neighbouring patches genuinely different
    tessellations -- different levels, different row directions, different
    numbers of knots on the curve they share -- so this walks a real mesh and
    checks each seam from both sides rather than trusting the construction.
    """
    mob = build()
    primitive = LogicalPNTrianglePrimitive(
        triangle_collection=[mob.get_render_primitives()]
    )
    source_corners = primitive.corners.reshape(-1, 3, 3).clone()
    source_normals = primitive.normals.reshape(-1, 3, 3).clone()
    primitive._dice_logical_pn(_camera([-2.5], device=source_corners.device))

    # Two patches are neighbours when they name the same pair of corners.
    seams = {}
    keys = (source_corners * 1e5).round().to(torch.int64)
    for patch in range(source_corners.shape[0]):
        for edge, (first, second) in enumerate(EDGE_CORNERS):
            key = tuple(
                sorted(
                    (
                        tuple(keys[patch, first].tolist()),
                        tuple(keys[patch, second].tolist()),
                    )
                )
            )
            seams.setdefault(key, []).append((patch, edge))

    shared = [sides for sides in seams.values() if len(sides) == 2]
    assert len(shared) > 20, "the mesh should have plenty of interior seams"

    def emitted_level(patch, edge):
        apex = int(primitive._logical_pn_apex[0, patch])
        if OPPOSITE_EDGE[apex] == edge:
            return int(primitive._logical_pn_across_levels[0, patch])
        return int(primitive._logical_pn_subdivision_levels[0, patch])

    cracks = 0
    uneven = 0
    for (patch_a, edge_a), (patch_b, edge_b) in shared:
        points_a = _patch_boundary_points(
            primitive, source_corners, source_normals, patch_a, edge_a
        )
        points_b = _patch_boundary_points(
            primitive, source_corners, source_normals, patch_b, edge_b
        )
        gap = max(
            float(_distance_to_polyline(points_a, points_b).max()),
            float(_distance_to_polyline(points_b, points_a).max()),
        )
        cracks += gap > 1e-5
        uneven += emitted_level(patch_a, edge_a) != emitted_level(patch_b, edge_b)
    assert cracks == 0
    # Keep the check honest: a mesh whose neighbours all happened to agree
    # would pass this without exercising the snap at all.
    assert uneven > 0


def test_a_developable_surface_dices_along_its_curved_direction_only():
    """A cylinder is straight along its axis, so the dice stays coarse there."""
    mob = Cylinder(radius=0.3, height=8.0)
    primitive = LogicalPNTrianglePrimitive(
        triangle_collection=[mob.get_render_primitives()]
    )

    primitive._dice_logical_pn(_camera([-2.0], device=primitive.corners.device))

    levels = primitive._logical_pn_subdivision_levels[0]
    across = primitive._logical_pn_across_levels[0]
    counts = primitive._logical_pn_triangle_counts[0]
    assert bool((across <= levels).all())
    assert int((across < levels).sum()) > levels.numel() // 2
    # ...and the saving is real: strictly fewer microtriangles than the uniform
    # dice at the level the interior actually needs.
    assert int(counts.sum()) < int((4**levels).sum())


def test_anisotropic_dice_can_be_switched_off():
    mob = Cylinder(radius=0.3, height=8.0)

    def dice():
        primitive = LogicalPNTrianglePrimitive(
            triangle_collection=[mob.get_render_primitives()]
        )
        primitive._dice_logical_pn(_camera([-2.0], device=primitive.corners.device))
        return primitive

    with SETTINGS.raytracing.experimental.override(pn_anisotropic_dice=False):
        uniform = dice()
    anisotropic = dice()

    assert bool(
        (
            uniform._logical_pn_across_levels == uniform._logical_pn_subdivision_levels
        ).all()
    )
    torch.testing.assert_close(
        uniform._logical_pn_triangle_counts,
        4**uniform._logical_pn_subdivision_levels,
    )
    assert int(anisotropic._logical_pn_triangle_counts.sum()) < int(
        uniform._logical_pn_triangle_counts.sum()
    )


# --- stopping at the logical surface's own accuracy --------------------------


def test_geometry_slack_stops_the_search_at_the_surfaces_own_accuracy():
    """The dice does not resolve detail the PN patch itself does not have."""
    mob = Sphere(radius=0.8)

    def dice(**overrides):
        primitive = LogicalPNTrianglePrimitive(
            triangle_collection=[mob.get_render_primitives()]
        )
        with SETTINGS.raytracing.experimental.override(**overrides):
            primitive._dice_logical_pn(_camera([-1.4], device=primitive.corners.device))
        return primitive

    strict = dice(pn_geometry_slack=False)
    relaxed = dice()

    assert mob._geometry_slack_ratio > 0
    assert bool(
        (
            relaxed._logical_pn_subdivision_levels
            <= strict._logical_pn_subdivision_levels
        ).all()
    )
    assert int(relaxed._logical_pn_triangle_counts.sum()) < int(
        strict._logical_pn_triangle_counts.sum()
    )


def test_a_patch_soup_is_its_own_surface_and_gets_no_slack():
    corners, normals = _curved_patch_inputs()

    primitive = _logical_patch(corners, normals)

    assert primitive.geometry_slack_ratio == 0.0


def test_geometry_slack_follows_a_scaled_surface():
    """The accuracy is a world-space length, so it scales with the mob."""
    mob = Sphere(radius=0.8)
    primitive = LogicalPNTrianglePrimitive(
        triangle_collection=[mob.get_render_primitives()]
    )
    corners = primitive.corners

    full = mean_patch_edge_length(corners) * primitive.geometry_slack_ratio
    tenth = mean_patch_edge_length(corners * 0.1) * primitive.geometry_slack_ratio

    torch.testing.assert_close(full, mob.geometry_tolerance * torch.ones_like(full))
    torch.testing.assert_close(tenth, full * 0.1)
