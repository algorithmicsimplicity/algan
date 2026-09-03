"""Taichi kernels for the logical PN / bezier subdivision-level criteria.

The level searches in :mod:`algan.rendering.raytracing.primitives` are
reductions: they evaluate a patch (or a curve) at a few hundred thousand sample
points and reduce the whole lot to *one scalar per patch*, the peak projected
pixel deviation of that level's flat stand-in.  Written in torch that is ~30
elementwise passes over ``[K, N, 3]`` scratch -- two full perspective
projections, a guard box, a clamp, a norm and a masked max -- so it moves
hundreds of megabytes per call to produce a handful of floats, and is bound by
device bandwidth rather than by the arithmetic.

Fused into a kernel the intermediates never leave registers: one thread per
(patch, microtriangle) evaluates that microtriangle's corners and interior
samples, projects both, and reduces to a single ``atomic_max`` on the patch's
error.  The traffic is then the control points in and one float out.

Determinism and crack-freeness
------------------------------
The level a boundary curve settles on must come out *bit-identical* for the two
patches that share it, or the mesh cracks open along the seam
(``benchmarks/_logical_pn_crack_check.py``).  Two properties give that here:

* both patches present the same canonically ordered controls
  (:func:`~algan.rendering.logical_pn.logical_pn_edge_control_points`) to the
  same kernel code, so the arithmetic is identical operation for operation; and
* the cross-thread reduction is ``max``, which is exact and order-independent
  for floats -- unlike a sum, a differently ordered ``atomic_max`` cannot
  produce a different answer.

These kernels are *not* bit-identical to the torch path they replace: Taichi
initialises with ``fast_math`` on, so the fused expressions round differently in
the last bits and borderline patches flip to a neighbouring level.  That moves
geometry (within ``render_tolerance_pixels`` by construction), not just pixel
rounding.
``ALGAN_PN_CRITERION_KERNEL=0`` restores the torch path for A/B.
"""

from algan.taichi_compat import ti

# Taichi is NOT initialized here. The arch depends on
# ``SETTINGS.computing.render_device``, which a script may still change at
# this point, so the program is created at the start of a render by
# ``taichi_runtime.ensure_taichi_for_render()``. Defining a kernel needs no
# program -- ``@ti.kernel`` only registers it; materialization at first launch
# is what needs one, and by then a render has selected the arch. Anything that
# launches these kernels outside a render (a benchmark, a unit test) must call
# ``init_taichi()`` itself.

# Matches ``LogicalPNTrianglePrimitive._guarded_pixel_error``: a projected
# sample at or behind the camera plane has no finite screen position and cannot
# steer subdivision, so it is dropped and the in-front samples decide.
_MIN_FRONT_DEPTH = 1e-7


@ti.func
def _screen_pixels(point, cam_origin, screen_point, sx, sy, sz, half_height):
    """Perspective-project one world point; returns ``[pixel_x, pixel_y, depth]``.

    The same construction as
    ``LogicalPNTrianglePrimitive._project_to_output_pixels``: intersect the view
    ray with the screen plane, take raw dot products with the (generally
    non-orthogonal) screen basis rows, and scale by half the output height.
    """
    ray = point - cam_origin
    depth = ray.dot(sz)
    screen_distance = (screen_point - cam_origin).dot(sz)
    projected = cam_origin + (screen_distance / depth) * ray
    relative = projected - screen_point
    return ti.Vector(
        [
            relative.dot(sx) * half_height,
            relative.dot(sy) * half_height,
            depth,
        ]
    )


@ti.func
def _guarded_error(
    exact,
    approximated,
    cam_origin,
    screen_point,
    sx,
    sy,
    sz,
    half_height,
    guard,
    sign,
    slack,
):
    """Guarded projected pixel deviation between one pair of matching points.

    Deviation that happens entirely outside the guard box costs nothing, and
    both projections are clamped into the box before being compared, so
    geometry that has left the frame cannot drive the level up (see
    ``_guarded_pixel_error`` for why ``camera.orbit`` needs that).

    ``slack`` is the world-space accuracy of the reference surface itself,
    projected at the exact point's depth and subtracted from the deviation, so
    the search stops where the reference stops being meaningful. Zero measures
    against the PN patch exactly. Matches ``_guarded_pixel_error``.
    """
    e = _screen_pixels(exact, cam_origin, screen_point, sx, sy, sz, half_height)
    a = _screen_pixels(approximated, cam_origin, screen_point, sx, sy, sz, half_height)
    result = 0.0
    inside = (ti.abs(e[0]) <= guard and ti.abs(e[1]) <= guard) or (
        ti.abs(a[0]) <= guard and ti.abs(a[1]) <= guard
    )
    if (e[2] * sign > _MIN_FRONT_DEPTH) and (a[2] * sign > _MIN_FRONT_DEPTH) and inside:
        dx = ti.min(ti.max(e[0], -guard), guard) - ti.min(ti.max(a[0], -guard), guard)
        dy = ti.min(ti.max(e[1], -guard), guard) - ti.min(ti.max(a[1], -guard), guard)
        screen_distance = (screen_point - cam_origin).dot(sz)
        allowance = slack * ti.abs(screen_distance / e[2]) * half_height
        result = ti.max(ti.sqrt(dx * dx + dy * dy) - allowance, 0.0)
    return result


@ti.func
def _pn_evaluate(controls, u, v):
    """Cubic logical PN patch at barycentric ``(u, v)``; ``controls`` is 10x3.

    Term order matches :func:`~algan.rendering.logical_pn.evaluate_logical_pn`.
    """
    w = 1.0 - u - v
    b0 = w * w * w
    b1 = u * u * u
    b2 = v * v * v
    b3 = 3.0 * w * w * u
    b4 = 3.0 * w * u * u
    b5 = 3.0 * u * u * v
    b6 = 3.0 * u * v * v
    b7 = 3.0 * w * v * v
    b8 = 3.0 * w * w * v
    b9 = 6.0 * w * u * v
    result = ti.Vector([0.0, 0.0, 0.0])
    for c in ti.static(range(3)):
        result[c] = (
            b0 * controls[0, c]
            + b1 * controls[1, c]
            + b2 * controls[2, c]
            + b3 * controls[3, c]
            + b4 * controls[4, c]
            + b5 * controls[5, c]
            + b6 * controls[6, c]
            + b7 * controls[7, c]
            + b8 * controls[8, c]
            + b9 * controls[9, c]
        )
    return result


@ti.func
def _cubic_evaluate(controls, t):
    """Cubic bezier curve at ``t``; ``controls`` is 4x3.

    Term order matches :func:`~algan.rendering.logical_pn.evaluate_cubic_curve`.
    """
    s = 1.0 - t
    w0 = s * s * s
    w1 = 3.0 * s * s * t
    w2 = 3.0 * s * t * t
    w3 = t * t * t
    result = ti.Vector([0.0, 0.0, 0.0])
    for c in ti.static(range(3)):
        result[c] = (
            w0 * controls[0, c]
            + w1 * controls[1, c]
            + w2 * controls[2, c]
            + w3 * controls[3, c]
        )
    return result


@ti.func
def _cubic_derivative(controls, t):
    """Derivative of a cubic bezier at ``t``; ``controls`` is 4x3.

    Term order matches ``primitives._evaluate_cubic_bezier_derivative_batch``.
    """
    s = 1.0 - t
    w0 = s * s
    w1 = 2.0 * s * t
    w2 = t * t
    result = ti.Vector([0.0, 0.0, 0.0])
    for c in ti.static(range(3)):
        result[c] = 3.0 * (
            w0 * (controls[1, c] - controls[0, c])
            + w1 * (controls[2, c] - controls[1, c])
            + w2 * (controls[3, c] - controls[2, c])
        )
    return result


@ti.kernel
def pn_patch_flatness_error(
    control_points: ti.types.ndarray(),  # [Tc, P, 10, 3] f32
    frame_stride: ti.i32,  # 0 when the control net is shared by all frames
    rows: ti.types.ndarray(),  # [R, 2] i32 -- (frame, patch)
    corner_uv: ti.types.ndarray(),  # [M, 3, 2] f32
    weights: ti.types.ndarray(),  # [S, 3] f32
    cam_origin: ti.types.ndarray(),  # [T, 3] f32
    screen_point: ti.types.ndarray(),  # [T, 3] f32
    screen_basis: ti.types.ndarray(),  # [T, 3, 3] f32
    front_sign: ti.types.ndarray(),  # [T] f32
    slack: ti.types.ndarray(),  # [T] f32 -- surface accuracy, 0 = measure exactly
    error: ti.types.ndarray(),  # [R] f32, pre-zeroed
    half_height: ti.f32,
    guard: ti.f32,
):
    """Peak pixel deviation of each selected patch's level dice.

    One thread per (selected patch, microtriangle).  The thread evaluates the
    microtriangle's three dice corners on the true patch, then walks the
    interior sample weights comparing the true patch against the flat
    interpolation of those corners, and folds its own peak into the patch's
    error with a single atomic.  ``error`` must arrive zeroed.
    """
    num_micro = corner_uv.shape[0]
    num_samples = weights.shape[0]
    for flat in range(rows.shape[0] * num_micro):
        r = flat // num_micro
        m = flat - r * num_micro
        t = rows[r, 0]
        p = rows[r, 1]

        controls = ti.Matrix.zero(ti.f32, 10, 3)
        for k in ti.static(range(10)):
            for c in ti.static(range(3)):
                controls[k, c] = control_points[t * frame_stride, p, k, c]

        origin = ti.Vector(
            [cam_origin[t, 0], cam_origin[t, 1], cam_origin[t, 2]]
        )
        plane = ti.Vector(
            [screen_point[t, 0], screen_point[t, 1], screen_point[t, 2]]
        )
        sx = ti.Vector(
            [screen_basis[t, 0, 0], screen_basis[t, 0, 1], screen_basis[t, 0, 2]]
        )
        sy = ti.Vector(
            [screen_basis[t, 1, 0], screen_basis[t, 1, 1], screen_basis[t, 1, 2]]
        )
        sz = ti.Vector(
            [screen_basis[t, 2, 0], screen_basis[t, 2, 1], screen_basis[t, 2, 2]]
        )
        sign = front_sign[t]
        allowance = slack[t]

        # The dice's own vertices: the flat stand-in this level is judged on is
        # the plane through these three points.
        corners = ti.Matrix.zero(ti.f32, 3, 3)
        for k in ti.static(range(3)):
            point = _pn_evaluate(controls, corner_uv[m, k, 0], corner_uv[m, k, 1])
            for c in ti.static(range(3)):
                corners[k, c] = point[c]

        worst = 0.0
        for s in range(num_samples):
            u = 0.0
            v = 0.0
            approximated = ti.Vector([0.0, 0.0, 0.0])
            for k in ti.static(range(3)):
                weight = weights[s, k]
                u += weight * corner_uv[m, k, 0]
                v += weight * corner_uv[m, k, 1]
                for c in ti.static(range(3)):
                    approximated[c] += weight * corners[k, c]
            exact = _pn_evaluate(controls, u, v)
            worst = ti.max(
                worst,
                _guarded_error(
                    exact,
                    approximated,
                    origin,
                    plane,
                    sx,
                    sy,
                    sz,
                    half_height,
                    guard,
                    sign,
                    allowance,
                ),
            )
        ti.atomic_max(error[r], worst)


@ti.func
def _segment_distance_squared(point, start, delta, length_squared):
    """Squared 2D distance from ``point`` to the finite segment ``start+delta``.

    Matches ``primitives._point_to_segment_distance_squared``, including its
    ``1e-20`` floor on the segment length (a chord of zero length collapses to
    its start point rather than dividing by zero).
    """
    along = (point - start).dot(delta) / ti.max(length_squared, 1e-20)
    closest = start + ti.min(ti.max(along, 0.0), 1.0) * delta
    return (point - closest).dot(point - closest)


@ti.kernel
def bezier_chord_hull_error(
    corners: ti.types.ndarray(),  # [Tc, S, 4, 3] f32 -- world cubic controls
    frame_stride: ti.i32,  # 0 when one control set serves every frame
    active: ti.types.ndarray(),  # [A] i32 -- segment indices still searching
    cam_origin: ti.types.ndarray(),  # [T, 3] f32
    screen_point: ti.types.ndarray(),  # [T, 3] f32
    screen_basis: ti.types.ndarray(),  # [T, 3, 3] f32
    error_squared: ti.types.ndarray(),  # [A] f32, pre-zeroed
    num_frames: ti.i32,
    num_subdivisions: ti.i32,
    half_height: ti.f32,
):
    """Peak squared control-hull deviation of each active cubic segment.

    One thread per (active segment, frame, uniform subcurve). The thread splits
    the segment's cubic to that subcurve exactly -- endpoint positions and
    derivatives determine the restricted cubic's four controls -- projects those
    four controls, and measures how far the two interior controls sit from the
    endpoint chord.

    A perspective-projected cubic whose controls all lie on one side of the
    camera plane is a rational Bezier with positive weights, so it stays inside
    its projected control hull and that distance bounds the curve-to-chord
    error. A subcurve touching or crossing the plane has no such bound: it
    reports ``inf``, which keeps its segment active and falls back to the hard
    cap, exactly as the torch path does.
    """
    scale = 1.0 / ti.cast(num_subdivisions, ti.f32)
    derivative_scale = scale / 3.0
    per_segment = num_frames * num_subdivisions
    for flat in range(active.shape[0] * per_segment):
        a = flat // per_segment
        within = flat - a * per_segment
        f = within // num_subdivisions
        d = within - f * num_subdivisions
        s = active[a]

        controls = ti.Matrix.zero(ti.f32, 4, 3)
        for k in ti.static(range(4)):
            for c in ti.static(range(3)):
                controls[k, c] = corners[f * frame_stride, s, k, c]

        origin = ti.Vector([cam_origin[f, 0], cam_origin[f, 1], cam_origin[f, 2]])
        plane = ti.Vector(
            [screen_point[f, 0], screen_point[f, 1], screen_point[f, 2]]
        )
        sx = ti.Vector(
            [screen_basis[f, 0, 0], screen_basis[f, 0, 1], screen_basis[f, 0, 2]]
        )
        sy = ti.Vector(
            [screen_basis[f, 1, 0], screen_basis[f, 1, 1], screen_basis[f, 1, 2]]
        )
        sz = ti.Vector(
            [screen_basis[f, 2, 0], screen_basis[f, 2, 1], screen_basis[f, 2, 2]]
        )

        t0 = ti.cast(d, ti.f32) * scale
        t1 = t0 + scale
        q0 = _cubic_evaluate(controls, t0)
        q3 = _cubic_evaluate(controls, t1)
        q1 = q0 + derivative_scale * _cubic_derivative(controls, t0)
        q2 = q3 - derivative_scale * _cubic_derivative(controls, t1)

        hull = ti.Matrix.zero(ti.f32, 4, 3)
        for c in ti.static(range(3)):
            hull[0, c] = q0[c]
            hull[1, c] = q1[c]
            hull[2, c] = q2[c]
            hull[3, c] = q3[c]

        pixels = ti.Matrix.zero(ti.f32, 4, 2)
        depth_low = ti.math.inf
        depth_high = -ti.math.inf
        bounded = True
        for k in ti.static(range(4)):
            source = ti.Vector([hull[k, 0], hull[k, 1], hull[k, 2]])
            projected = _screen_pixels(
                source, origin, plane, sx, sy, sz, half_height
            )
            pixels[k, 0] = projected[0]
            pixels[k, 1] = projected[1]
            depth_low = ti.min(depth_low, projected[2])
            depth_high = ti.max(depth_high, projected[2])
            # Comparison-based, so it survives fast_math where an isfinite
            # intrinsic may not: both inf and nan fail it.
            bounded = bounded and (
                ti.abs(projected[0]) < 1e30 and ti.abs(projected[1]) < 1e30
            )

        result = ti.math.inf
        if bounded and (depth_low > 1e-8 or depth_high < -1e-8):
            start = ti.Vector([pixels[0, 0], pixels[0, 1]])
            delta = ti.Vector([pixels[3, 0] - start[0], pixels[3, 1] - start[1]])
            length_squared = delta.dot(delta)
            result = ti.max(
                _segment_distance_squared(
                    ti.Vector([pixels[1, 0], pixels[1, 1]]),
                    start,
                    delta,
                    length_squared,
                ),
                _segment_distance_squared(
                    ti.Vector([pixels[2, 0], pixels[2, 1]]),
                    start,
                    delta,
                    length_squared,
                ),
            )
        ti.atomic_max(error_squared[a], result)


@ti.kernel
def pn_edge_chord_error(
    edge_controls: ti.types.ndarray(),  # [Te, P, 3, 4, 3] f32
    frame_stride: ti.i32,  # 0 when the control net is shared by all frames
    active: ti.types.ndarray(),  # [A] i32 -- flat indices into [T, P, 3]
    samples: ti.types.ndarray(),  # [S] f32 -- parameters within each chord
    cam_origin: ti.types.ndarray(),  # [T, 3] f32
    screen_point: ti.types.ndarray(),  # [T, 3] f32
    screen_basis: ti.types.ndarray(),  # [T, 3, 3] f32
    front_sign: ti.types.ndarray(),  # [T] f32
    slack: ti.types.ndarray(),  # [T] f32 -- surface accuracy, 0 = measure exactly
    error: ti.types.ndarray(),  # [A] f32, pre-zeroed
    num_patches: ti.i32,
    segments: ti.i32,
    half_height: ti.f32,
    guard: ti.f32,
):
    """Peak pixel deviation of each active boundary curve from its chord polyline.

    One thread per (active curve, chord).  Both of a chord's knots are
    re-evaluated per thread rather than shared with the neighbouring chord: the
    curve evaluation is a handful of FMAs and the sharing would cost either a
    second pass or scratch, which is the traffic this kernel exists to avoid.

    The curve is judged on its canonical controls alone, so the two patches
    sharing it reach a bit-identical answer -- see the module docstring.
    """
    num_samples = samples.shape[0]
    steps = ti.cast(segments, ti.f32)
    for flat in range(active.shape[0] * segments):
        a = flat // segments
        i = flat - a * segments
        index = active[a]
        t = index // (num_patches * 3)
        within = index - t * (num_patches * 3)
        p = within // 3
        e = within - p * 3

        controls = ti.Matrix.zero(ti.f32, 4, 3)
        for k in ti.static(range(4)):
            for c in ti.static(range(3)):
                controls[k, c] = edge_controls[t * frame_stride, p, e, k, c]

        origin = ti.Vector(
            [cam_origin[t, 0], cam_origin[t, 1], cam_origin[t, 2]]
        )
        plane = ti.Vector(
            [screen_point[t, 0], screen_point[t, 1], screen_point[t, 2]]
        )
        sx = ti.Vector(
            [screen_basis[t, 0, 0], screen_basis[t, 0, 1], screen_basis[t, 0, 2]]
        )
        sy = ti.Vector(
            [screen_basis[t, 1, 0], screen_basis[t, 1, 1], screen_basis[t, 1, 2]]
        )
        sz = ti.Vector(
            [screen_basis[t, 2, 0], screen_basis[t, 2, 1], screen_basis[t, 2, 2]]
        )
        sign = front_sign[t]
        allowance = slack[t]

        low = _cubic_evaluate(controls, ti.cast(i, ti.f32) / steps)
        high = _cubic_evaluate(controls, ti.cast(i + 1, ti.f32) / steps)

        worst = 0.0
        for s in range(num_samples):
            blend = samples[s]
            exact = _cubic_evaluate(controls, (ti.cast(i, ti.f32) + blend) / steps)
            chord = low * (1.0 - blend) + high * blend
            worst = ti.max(
                worst,
                _guarded_error(
                    exact,
                    chord,
                    origin,
                    plane,
                    sx,
                    sy,
                    sz,
                    half_height,
                    guard,
                    sign,
                    allowance,
                ),
            )
        ti.atomic_max(error[a], worst)
