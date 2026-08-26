"""CPU batch-prep kernels for the batched surface build.

``get_render_primitives_batched`` is the largest single stage of a render
(``DESIGN_optimization_targets.md``, P10/P10b). Two of its rows are here, both
dispatched only when Taichi's arch is the CPU -- see
:func:`algan.rendering.taichi_runtime.cpu_prep_kernel_enabled` for why a GPU arch
must keep the torch path.

``grid_normals_sides_crosses``
    The sides-and-crosses block of :func:`compute_grid_vertex_normals`, ~35% of
    the stage. The torch form is arithmetically cheap and structurally
    expensive: four ``_wrapped_difference`` buffers, four cross-product buffers
    and one accumulator, so nine full-size tensors are written to produce one at
    roughly 57 flops per grid point, every intermediate read exactly once. One
    stencil pass reads the grid and writes the normals.

``gather_grid_to_triangles``
    :func:`grid_to_triangle_vertices`' gather, ~20% of the stage across its two
    call sites. Pure permutation, so this one *is* byte-identical: it copies the
    same elements the advanced index copies, and writing them from a kernel
    skips torch's index-expansion machinery.

**Watertightness.** The gather is an exact copy, so vertex positions are
unchanged and the mesh topology cannot move. The normals kernel is *not*
bit-identical to torch (see below), but it cannot open a seam either: it
produces the same ``unnormalized_normals`` buffer the seam-merge and pole-fan
code then consumes, and that code assigns one shared value to both sides of a
closed seam and one shared value to a whole pole row. Two grid points that must
agree still read the same element afterwards. This matters beyond shading --
logical PN patches build their curvature from corner normals, so a seam whose
two sides disagreed would crack the geometry, not just the shading.
``tests/unit_tests/test_surface_prep_kernels.py`` asserts it on both closed
axes and both poles.

**Why not bit-identical.** The block evaluates the same four differences, the
same four cross products from the same operand pairs and the same three
additions in the same ``((A + B) + C) + D`` order, with a gated triangle
contributing an explicit ``0.0`` rather than being dropped (``-0.0 + 0.0`` is
``+0.0`` and ``NaN + 0.0`` is ``NaN``, so dropping the add is observable). It
still differs on ~4% of elements by 1-2 ulp, and the cause is ``torch.cross``,
not Taichi: rebuilding with ``fast_math=False`` changes nothing, and on the
cross product's third component -- ``a0 * b1 - a1 * b0``, the one that
catastrophically cancels for a sphere's tangential sides -- ``torch.cross``
matches neither that expression in float32 nor its products taken exactly in
double and rounded once. Taichi 1.7.4 exposes no FMA intrinsic, so no
formulation reproduces it. ``benchmarks/_grid_normals_kernel_ab.py`` measures
both the deviation and the speedup.
"""

import taichi as ti


@ti.kernel
def grid_normals_sides_crosses(
        grid: ti.types.ndarray(dtype=ti.f32, ndim=4),  # [B, W, H, 3]
        out: ti.types.ndarray(dtype=ti.f32, ndim=4),  # [B, W, H, 3]
):
    """Accumulated (unnormalized) area-weighted vertex normals, in one pass.

    Parallel over every grid point. Neighbours wrap, matching
    ``_wrapped_difference``'s two-piece write, and the four triangles carry the
    same boundary gates the torch form applies as slice assignments: ``xm_ym``
    dies on the low x or low y edge, ``ym_xp`` on high x or low y, ``xp_yp`` on
    high x or high y, ``yp_xm`` on low x or high y.
    """
    W = grid.shape[1]
    H = grid.shape[2]
    for b, x, y in ti.ndrange(grid.shape[0], W, H):
        c0 = grid[b, x, y, 0]
        c1 = grid[b, x, y, 1]
        c2 = grid[b, x, y, 2]

        xm_i = x - 1
        if xm_i < 0:
            xm_i = W - 1
        xp_i = x + 1
        if xp_i >= W:
            xp_i = 0
        ym_i = y - 1
        if ym_i < 0:
            ym_i = H - 1
        yp_i = y + 1
        if yp_i >= H:
            yp_i = 0

        # grid.roll(shift, axis) - grid, never materialized.
        xm0 = grid[b, xm_i, y, 0] - c0
        xm1 = grid[b, xm_i, y, 1] - c1
        xm2 = grid[b, xm_i, y, 2] - c2
        ym0 = grid[b, x, ym_i, 0] - c0
        ym1 = grid[b, x, ym_i, 1] - c1
        ym2 = grid[b, x, ym_i, 2] - c2
        xp0 = grid[b, xp_i, y, 0] - c0
        xp1 = grid[b, xp_i, y, 1] - c1
        xp2 = grid[b, xp_i, y, 2] - c2
        yp0 = grid[b, x, yp_i, 0] - c0
        yp1 = grid[b, x, yp_i, 1] - c1
        yp2 = grid[b, x, yp_i, 2] - c2

        low_x = x == 0
        high_x = x == W - 1
        low_y = y == 0
        high_y = y == H - 1

        # A = cross(x_minus, y_minus)
        a0 = 0.0
        a1 = 0.0
        a2 = 0.0
        if not (low_x or low_y):
            a0 = xm1 * ym2 - xm2 * ym1
            a1 = xm2 * ym0 - xm0 * ym2
            a2 = xm0 * ym1 - xm1 * ym0

        # B = cross(y_minus, x_plus)
        b0 = 0.0
        b1 = 0.0
        b2 = 0.0
        if not (high_x or low_y):
            b0 = ym1 * xp2 - ym2 * xp1
            b1 = ym2 * xp0 - ym0 * xp2
            b2 = ym0 * xp1 - ym1 * xp0

        # C = cross(x_plus, y_plus)
        c_0 = 0.0
        c_1 = 0.0
        c_2 = 0.0
        if not (high_x or high_y):
            c_0 = xp1 * yp2 - xp2 * yp1
            c_1 = xp2 * yp0 - xp0 * yp2
            c_2 = xp0 * yp1 - xp1 * yp0

        # D = cross(y_plus, x_minus)
        d0 = 0.0
        d1 = 0.0
        d2 = 0.0
        if not (low_x or high_y):
            d0 = yp1 * xm2 - yp2 * xm1
            d1 = yp2 * xm0 - yp0 * xm2
            d2 = yp0 * xm1 - yp1 * xm0

        # ((A + B) + C) + D, the order the in-place accumulation takes.
        out[b, x, y, 0] = ((a0 + b0) + c_0) + d0
        out[b, x, y, 1] = ((a1 + b1) + c_1) + d1
        out[b, x, y, 2] = ((a2 + b2) + c_2) + d2


@ti.kernel
def gather_grid_to_triangles(
        flat_grid: ti.types.ndarray(dtype=ti.f32, ndim=3),  # [B, W * H, C]
        indices: ti.types.ndarray(dtype=ti.i64, ndim=1),  # [L], L = triangles * 3
        out: ti.types.ndarray(dtype=ti.f32, ndim=3),  # [B, L, C]
):
    """``flat_grid[..., indices, :]``, written directly.

    ``get_grid_to_triangle_indices`` returns its table already flattened to
    ``[triangles * 3]`` (its ``stacked.reshape(-1)``), so the corner axis is not
    separate here and the result keeps the ``[..., L, C]`` layout the advanced
    index produces.

    Byte-identical by construction: every output element is a copy of the
    element the advanced index would have selected. The welded index table is
    what makes a closed seam or a collapsed pole a *shared* vertex rather than
    coincident duplicates, and it arrives here already built, so welding is
    carried through unchanged.
    """
    C = flat_grid.shape[2]
    for b, i in ti.ndrange(flat_grid.shape[0], indices.shape[0]):
        source = indices[i]
        for c in range(C):
            out[b, i, c] = flat_grid[b, source, c]
