import torch

from algan.animation_timeline.animation_contexts import Off
from algan.mobs.shapes_2d import Line, Square
from algan.rendering.raytracing.primitives import (
    RayTracedBezierCircuitPrimitive,
    _bezier_connection_visibility,
    _evaluate_cubic_bezier_batch,
    _packed_uniform_cubic_parameters,
    _uniform_cubic_subcurves,
)
from algan.rendering.raytracing.raytrace_kernels_taichi import (
    _M_IOR,
    _M_REFLECTIVITY,
    _M_ROUGHNESS,
    _M_TRANSMISSION,
    _M_WIDTH,
)
from algan.scene_manager import SceneManager


def _sampler(tolerance=1.0):
    primitive = RayTracedBezierCircuitPrimitive.__new__(RayTracedBezierCircuitPrimitive)
    primitive.num_pixels_per_sample = tolerance
    primitive.max_samples_per_segment = 512
    return primitive


def _camera(num_frames=1):
    camera_origin = torch.tensor([[0.0, 0.0, 10.0]]).repeat(num_frames, 1)
    screen_point = torch.tensor([[0.0, 0.0, 0.0]]).repeat(num_frames, 1)
    screen_basis = torch.eye(3).unsqueeze(0).repeat(num_frames, 1, 1)
    return camera_origin, screen_point, screen_basis


def _dense_screen_error(corners, num_chords, screen_height=200):
    t = torch.linspace(0.0, 1.0, 8193).view(-1, 1)
    curve = _evaluate_cubic_bezier_batch(corners, t)[..., :2]

    chord_index = (t[:, 0] * num_chords).floor().long()
    chord_index.clamp_max_(num_chords - 1)
    chord_t0 = (chord_index / num_chords).view(-1, 1)
    chord_t1 = ((chord_index + 1) / num_chords).view(-1, 1)
    chord_start = _evaluate_cubic_bezier_batch(corners, chord_t0)[..., :2]
    chord_end = _evaluate_cubic_bezier_batch(corners, chord_t1)[..., :2]
    chord = chord_end - chord_start
    along = ((curve - chord_start) * chord).sum(-1, keepdim=True)
    along /= chord.square().sum(-1, keepdim=True).clamp_min(1e-20)
    closest = chord_start + along.clamp(0.0, 1.0) * chord
    return ((curve - closest) * (screen_height / 2)).norm(dim=-1).max()


def _project_to_screen(
    points, camera_origin, screen_point, screen_basis, screen_height
):
    rays = points - camera_origin
    normal = screen_basis[2]
    distance = ((screen_point - camera_origin) * normal).sum()
    projected = camera_origin + rays * (
        distance / (rays * normal).sum(-1, keepdim=True)
    )
    relative = projected - screen_point
    return torch.stack(
        (
            (relative * screen_basis[0]).sum(-1),
            (relative * screen_basis[1]).sum(-1),
        ),
        dim=-1,
    ) * (screen_height / 2)


def _dense_perspective_error(
    corners, num_chords, camera_origin, screen_point, screen_basis, screen_height
):
    t = torch.linspace(0.0, 1.0, 8193).view(-1, 1)
    curve = _project_to_screen(
        _evaluate_cubic_bezier_batch(corners, t),
        camera_origin,
        screen_point,
        screen_basis,
        screen_height,
    )
    chord_index = (t[:, 0] * num_chords).floor().long()
    chord_index.clamp_max_(num_chords - 1)
    chord_t0 = (chord_index / num_chords).view(-1, 1)
    chord_t1 = ((chord_index + 1) / num_chords).view(-1, 1)
    chord_start = _project_to_screen(
        _evaluate_cubic_bezier_batch(corners, chord_t0),
        camera_origin,
        screen_point,
        screen_basis,
        screen_height,
    )
    chord_end = _project_to_screen(
        _evaluate_cubic_bezier_batch(corners, chord_t1),
        camera_origin,
        screen_point,
        screen_basis,
        screen_height,
    )
    chord = chord_end - chord_start
    along = ((curve - chord_start) * chord).sum(-1, keepdim=True)
    along /= chord.square().sum(-1, keepdim=True).clamp_min(1e-20)
    closest = chord_start + along.clamp(0.0, 1.0) * chord
    return (curve - closest).norm(dim=-1).max()


def test_straight_cubic_uses_one_chord_regardless_of_length():
    straight = torch.tensor(
        [
            [-100.0, 0.0, 0.0],
            [-100.0 / 3.0, 0.0, 0.0],
            [100.0 / 3.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
        ]
    )
    curved = torch.tensor(
        [
            [-1.0, 0.0, 0.0],
            [-1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    corners = torch.stack((straight, curved)).unsqueeze(0)
    camera_origin, screen_point, screen_basis = _camera()

    chord_counts = _sampler()._compute_samples_per_segment(
        corners, camera_origin, screen_point, screen_basis, screen_h=200
    )

    # One packed chord evaluates the cubic's two endpoints.  The curved cubic
    # must be subdivided independently rather than sharing the straight one's
    # count (or forcing that count onto it).
    assert chord_counts[0].item() == 1
    assert chord_counts[1].item() > 1


def test_checked_subcurves_are_the_uniform_chords_emitted_by_renderer():
    curve = torch.tensor(
        [
            [
                [-1.0, 0.0, 0.0],
                [-1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        ]
    )
    num_chords = 8
    checked = _uniform_cubic_subcurves(curve.unsqueeze(0), num_chords)[0, 0]
    emitted_t = _packed_uniform_cubic_parameters(
        torch.tensor([num_chords]), curve.dtype
    ).view(-1, 1)
    emitted_starts = _evaluate_cubic_bezier_batch(curve[0], emitted_t)
    emitted_ends = _evaluate_cubic_bezier_batch(curve[0], emitted_t + 1.0 / num_chords)

    assert torch.equal(checked[:, 0], emitted_starts)
    assert torch.equal(checked[:, 3], emitted_ends)


def test_closed_wraparound_border_is_visible_but_fill_closure_is_not():
    segments = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.7, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [0.7, 0.0, 0.0], [0.3, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ]
    ).unsqueeze(0)
    wraparound = torch.tensor([[1, 0]])

    closed = _bezier_connection_visibility(segments, wraparound)
    assert closed.tolist() == [[True, True]]

    segments[0, 1, 3, 0] = 2.0
    open_path = _bezier_connection_visibility(segments, wraparound)
    assert open_path.tolist() == [[True, False]]


def test_each_cubic_meets_screen_space_error_tolerance():
    tolerance = 1.0
    curves = torch.tensor(
        [
            [[-1.0, 0.0, 0.0], [-1.0, 0.1, 0.0], [1.0, 0.1, 0.0], [1.0, 0.0, 0.0]],
            [[-1.0, 0.0, 0.0], [-1.0, 1.5, 0.0], [1.0, -1.5, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [2.0, 3.0, 0.0], [-2.0, 3.0, 0.0], [0.0, 0.0, 0.0]],
        ]
    )
    camera_origin, screen_point, screen_basis = _camera()
    chord_counts = _sampler(tolerance)._compute_samples_per_segment(
        curves.unsqueeze(0),
        camera_origin,
        screen_point,
        screen_basis,
        screen_h=200,
    )

    for curve, count in zip(curves, chord_counts.tolist()):
        assert _dense_screen_error(curve, count) <= tolerance + 1e-4


def test_perspective_projected_cubic_meets_error_tolerance():
    tolerance = 1.0
    screen_height = 200
    curve = torch.tensor(
        [
            [-1.0, 0.0, 0.0],
            [-1.0, 2.0, 3.0],
            [1.0, -2.0, 5.0],
            [1.0, 0.0, 1.0],
        ]
    )
    camera_origin = torch.tensor([[0.0, 0.0, 10.0]])
    screen_point = torch.tensor([[0.0, 0.0, 7.0]])
    screen_basis = torch.eye(3).unsqueeze(0)
    chord_count = (
        _sampler(tolerance)
        ._compute_samples_per_segment(
            curve.view(1, 1, 4, 3),
            camera_origin,
            screen_point,
            screen_basis,
            screen_h=screen_height,
        )
        .item()
    )

    error = _dense_perspective_error(
        curve,
        chord_count,
        camera_origin[0],
        screen_point[0],
        screen_basis[0],
        screen_height,
    )
    assert error <= tolerance + 1e-4


def test_sampling_uses_worst_frame_for_each_segment():
    curve = torch.tensor(
        [
            [-1.0, 0.0, 0.0],
            [-1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    ).view(1, 1, 4, 3)
    camera_origin, screen_point, screen_basis = _camera(num_frames=2)
    camera_origin[0, 2] = 100.0
    screen_point[0, 2] = 97.0
    camera_origin[1, 2] = 10.0
    screen_point[1, 2] = 7.0

    both_frames = _sampler()._compute_samples_per_segment(
        curve, camera_origin, screen_point, screen_basis, screen_h=200
    )
    close_frame = _sampler()._compute_samples_per_segment(
        curve,
        camera_origin[1:],
        screen_point[1:],
        screen_basis[1:],
        screen_h=200,
    )

    assert torch.equal(both_frames, close_frame)


def _polyline(mob, chords_per_segment=1):
    """Pack ``mob``'s circuit into plane-space edges at a fixed chord count."""
    single = mob.get_render_primitives()
    # The renderer merges same-kind primitives into one collection before
    # projecting them, and that merge is what resolves each mob's RELATIVE
    # next_segment_inds into absolute ones.  Build the same thing here.
    primitive = type(single)(triangle_collection=[single])
    corners = primitive.corners.float().contiguous()
    chords = torch.full(
        (corners.shape[1],), chords_per_segment, dtype=torch.long, device=corners.device
    )
    primitive._build_circuit_geometry(corners, chords)
    return primitive._rt_edges[0]  # [V, 5]: u0, v0, u1, v1, border_visible


def test_circuit_material_metadata_matches_kernel_layout():
    """The host packer and every Taichi renderer share this channel contract."""
    SceneManager.reset()
    with Off(record_funcs=False, record_attr_modifications=False):
        square = Square(add_to_scene=False)

    single = square.get_render_primitives()
    primitive = type(single)(triangle_collection=[single])
    corners = primitive.corners.float().contiguous()
    chords = torch.ones(corners.shape[1], dtype=torch.long, device=corners.device)
    primitive._build_circuit_geometry(corners, chords)

    meta = primitive._rt_circuit_meta
    assert meta.shape[-1] == _M_WIDTH == 24
    assert torch.equal(
        meta[..., _M_REFLECTIVITY : _M_REFLECTIVITY + 1], primitive.reflectivity
    )
    assert torch.equal(meta[..., _M_ROUGHNESS : _M_ROUGHNESS + 1], primitive.roughness)
    assert torch.equal(meta[..., _M_IOR : _M_IOR + 1], primitive.refractive_index)
    assert torch.equal(
        meta[..., _M_TRANSMISSION : _M_TRANSMISSION + 1], primitive.transmission
    )


def test_open_subpath_polyline_carries_its_own_endpoint():
    """A ``Line`` is one straight cubic, so it flattens to a single chord.

    The packed polyline samples ``t = k/n`` for ``k < n`` and takes each cubic's
    endpoint from the first vertex of the segment it connects to -- but an open
    subpath connects back to its own start, so the endpoint belonged to nobody
    and the final chord was dropped.  At ``n = 1`` that is the whole outline:
    the circuit collapsed to a point and a ``Line`` rendered nothing at all.
    """
    SceneManager.reset()
    start = torch.tensor([-1.0, -0.5, 0.0])
    end = torch.tensor([1.0, 0.5, 0.0])
    with Off(record_funcs=False, record_attr_modifications=False):
        line = Line(start, end, stroke_width=2, add_to_scene=False)

    edges = _polyline(line)

    # Two vertices: t = 0 and the explicit t = 1, not one self-closing point.
    assert edges.shape[0] == 2
    drawn = edges[edges[:, 4] > 0.5]
    assert drawn.shape[0] == 1
    length = (drawn[0, 2:4] - drawn[0, :2]).norm()
    assert torch.isclose(length, (end - start).norm(), atol=1e-5)
    # The edge back to the start is the fill closure and stays undrawn.
    closure = edges[edges[:, 4] <= 0.5]
    assert closure.shape[0] == 1
    assert torch.allclose(closure[0, 2:4], drawn[0, :2], atol=1e-6)


def test_closed_circuit_polyline_is_unchanged_by_the_endpoint_rule():
    """Every connection of a closed circuit is continuous, so no segment needs
    an endpoint of its own and the packed vertex count stays at one per chord.
    """
    SceneManager.reset()
    with Off(record_funcs=False, record_attr_modifications=False):
        square = Square(stroke_width=2, add_to_scene=False)

    edges = _polyline(square)
    segments = square.get_render_primitives().corners.shape[1]

    assert edges.shape[0] == segments
    assert bool((edges[:, 4] > 0.5).all())
