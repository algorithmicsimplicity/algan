import numpy as np
import taichi as ti

from algan.rendering.raytracing.raster_taichi import (
    _AA_EXACT_MODE,
    _exact_circuit_boundary_moments,
    _pixel_clip_moments,
    _ss_pixel,
)


@ti.kernel
def _clip_moments(values: ti.types.ndarray(), result: ti.types.ndarray()):
    for i in range(values.shape[0]):
        area, cx, cy = _pixel_clip_moments(
            ti.math.vec3(values[i, 0], values[i, 2], values[i, 4]),
            ti.math.vec3(values[i, 1], values[i, 3], values[i, 5]),
        )
        result[i, 0] = area
        result[i, 1] = area * cx
        result[i, 2] = area * cy


@ti.kernel
def _one_exact_fragment(
    screen: ti.types.ndarray(), world: ti.types.ndarray(), result: ti.types.ndarray()
):
    sm = ti.Matrix.zero(ti.f32, 3, 3)
    vm = ti.Matrix.zero(ti.f32, 3, 3)
    for i in ti.static(range(3)):
        sm[0, i] = screen[i, 0]
        sm[1, i] = screen[i, 1]
        sm[2, i] = 1.0
        for j in ti.static(range(3)):
            vm[i, j] = world[i, j]
    ok, depth, b1, b2, coverage, _mask = _ss_pixel(
        0,
        0,
        sm,
        vm,
        ti.math.vec3(0.0, 0.0, 0.0),
        ti.math.vec3(0.0, 0.0, 0.0),
        _AA_EXACT_MODE,
    )
    result[0] = ok
    result[1] = depth
    result[2] = b1
    result[3] = b2
    result[4] = coverage


@ti.kernel
def _circuit_moments(
    edges: ti.types.ndarray(),
    offsets: ti.types.ndarray(),
    origins: ti.types.ndarray(),
    pixels: ti.types.ndarray(),
    result: ti.types.ndarray(),
):
    for i in range(pixels.shape[0]):
        area, cx, cy = _exact_circuit_boundary_moments(
            0,
            0,
            pixels[i, 0],
            pixels[i, 1],
            edges,
            offsets,
            origins,
        )
        result[i, 0] = area
        result[i, 1] = area * cx
        result[i, 2] = area * cy


def _clip_polygon(points):
    polygon = [np.asarray(point, dtype=np.float64) for point in points]
    for axis, bound, keep_greater in (
        (0, -0.5, True),
        (0, 0.5, False),
        (1, -0.5, True),
        (1, 0.5, False),
    ):
        clipped = []
        for start, end in zip(polygon, polygon[1:] + polygon[:1]):
            start_in = start[axis] >= bound if keep_greater else start[axis] <= bound
            end_in = end[axis] >= bound if keep_greater else end[axis] <= bound
            if start_in != end_in:
                t = (bound - start[axis]) / (end[axis] - start[axis])
                crossing = start + t * (end - start)
                if start_in:
                    clipped.append(crossing)
                else:
                    clipped.append(crossing)
            if end_in:
                clipped.append(end)
        polygon = clipped
        if not polygon:
            break
    return polygon


def _reference_moments(triangle):
    polygon = _clip_polygon(triangle)
    if len(polygon) < 3:
        return np.zeros(3, dtype=np.float64)
    points = np.asarray(polygon)
    following = np.roll(points, -1, axis=0)
    cross = points[:, 0] * following[:, 1] - following[:, 0] * points[:, 1]
    signed_twice_area = cross.sum()
    area = abs(signed_twice_area) * 0.5
    mx = ((points[:, 0] + following[:, 0]) * cross).sum() / 6.0
    my = ((points[:, 1] + following[:, 1]) * cross).sum() / 6.0
    sign = 1.0 if signed_twice_area >= 0.0 else -1.0
    return np.array((area, sign * mx, sign * my))


def test_triangle_clip_area_and_first_moments_match_double_reference():
    rng = np.random.default_rng(9324)
    triangles = []
    for scale in (0.01, 0.1, 1.0, 10.0, 100.0):
        centers = rng.uniform(-1.0, 1.0, (80, 1, 2))
        triangles.extend(centers + rng.uniform(-scale, scale, (80, 3, 2)))
    triangles = np.asarray(triangles, dtype=np.float64)
    packed = triangles.astype(np.float32).reshape(-1, 6)
    result = np.empty((packed.shape[0], 3), dtype=np.float32)

    _clip_moments(packed, result)

    expected = np.stack(
        [_reference_moments(triangle) for triangle in packed.reshape(-1, 3, 2)]
    )
    np.testing.assert_allclose(result[:, 0], expected[:, 0], atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(result[:, 1:], expected[:, 1:], atol=1e-5, rtol=2e-5)


def test_triangle_clip_conserves_a_pixel_partition():
    triangles = np.array(
        [
            [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5]],
            [[-0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]],
        ],
        dtype=np.float32,
    )
    result = np.empty((2, 3), dtype=np.float32)
    _clip_moments(triangles.reshape(-1, 6), result)

    assert abs(float(result[:, 0].sum()) - 1.0) <= 1e-6
    np.testing.assert_allclose(result[:, 1:].sum(0), 0.0, atol=1e-6)


def test_exact_fragment_uses_clipped_area_and_an_interior_centroid():
    screen = np.array([[0.3, 0.3], [1.7, 0.3], [0.3, 1.7]], dtype=np.float32)
    world = np.column_stack((screen, np.ones(3, dtype=np.float32)))
    result = np.empty(5, dtype=np.float32)

    _one_exact_fragment(screen, world, result)

    reference = _reference_moments(screen - 0.5)
    assert result[0] == 1.0
    assert abs(float(result[4]) - reference[0]) <= 1e-6
    assert result[2] >= 0.0
    assert result[3] >= 0.0
    assert result[2] + result[3] <= 1.0 + 1e-6


def test_exact_fragment_keeps_a_positive_sub_min_alpha_triangle():
    # Coverage smaller than the renderer's material-alpha cutoff is still real
    # geometry.  Exact mode must retain it; coverage is applied during resolve,
    # not used to reject the fragment during raster discovery.
    screen = np.array(
        [[0.499, 0.499], [0.501, 0.499], [0.499, 0.501]], dtype=np.float32
    )
    world = np.column_stack((screen, np.ones(3, dtype=np.float32)))
    result = np.empty(5, dtype=np.float32)

    _one_exact_fragment(screen, world, result)

    assert result[0] == 1.0
    assert 0.0 < result[4] < 1e-3


def _oriented_edges(*contours):
    return np.concatenate(
        [np.column_stack((points, np.roll(points, -1, axis=0))) for points in contours]
    ).astype(np.float32)


def _reference_boundary_moments(contours, origin, pixel):
    shift = np.asarray(pixel, dtype=np.float64) + 0.5
    origin = np.asarray(origin, dtype=np.float64) - shift
    total = np.zeros(3, dtype=np.float64)
    for points in contours:
        local = points - shift
        for start, end in zip(local, np.roll(local, -1, axis=0)):
            first = start - origin
            second = end - origin
            winding = first[0] * second[1] - first[1] * second[0]
            if abs(winding) > 1e-14:
                total += np.sign(winding) * _reference_moments(
                    np.stack((origin, start, end))
                )
    return total


def test_circuit_boundary_integral_matches_concave_polygon_with_hole():
    # CCW concave outer boundary and CW hole: the exact kernel integrates the
    # complete logical region once, however many contour edges cross a pixel.
    outer = np.array(
        [
            [-0.7, -0.4],
            [2.6, -0.4],
            [2.6, 2.4],
            [1.3, 1.1],
            [-0.7, 2.4],
        ],
        dtype=np.float64,
    )
    hole = np.array(
        [[0.1, 0.2], [0.1, 0.9], [0.8, 0.9], [0.8, 0.2]],
        dtype=np.float64,
    )
    edges = _oriented_edges(outer, hole)[None]
    offsets = np.array([0, edges.shape[1]], dtype=np.int32)
    origins = np.array([[[0.8, 0.8]]], dtype=np.float32)
    pixels = np.array(
        [(x, y) for y in range(-1, 3) for x in range(-1, 3)], dtype=np.int32
    )
    result = np.empty((len(pixels), 3), dtype=np.float32)

    _circuit_moments(edges, offsets, origins, pixels, result)

    expected = np.asarray(
        [
            _reference_boundary_moments((outer, hole), origins[0, 0], pixel)
            for pixel in pixels
        ]
    )
    np.testing.assert_allclose(result, expected, atol=1e-5, rtol=2e-5)

    # Conservation over the pixel partition equals the exact polygon area.
    outer_area = abs(
        (
            outer[:, 0] * np.roll(outer[:, 1], -1)
            - outer[:, 1] * np.roll(outer[:, 0], -1)
        ).sum()
        * 0.5
    )
    hole_area = abs(
        (
            hole[:, 0] * np.roll(hole[:, 1], -1) - hole[:, 1] * np.roll(hole[:, 0], -1)
        ).sum()
        * 0.5
    )
    assert abs(float(result[:, 0].sum()) - (outer_area - hole_area)) <= 1e-5
