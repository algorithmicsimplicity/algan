"""One-pass kernel for the sides-and-crosses block of ``compute_grid_vertex_normals``.

``DESIGN_optimization_targets.md`` (P10b) puts this block at **~35% of
``get_render_primitives_batched``** -- the largest single item left in a render --
and records that the only *large* win available to the torch form is the identity
``cross(xm,ym) + cross(ym,xp) + cross(xp,yp) + cross(yp,xm) = cross(xm - xp, ym - yp)``,
which collapses four cross products into one but is **not bit-identical** and
breaks at the grid's boundaries.

A kernel gets a bigger reduction without that trade. The torch form is
arithmetically cheap and structurally expensive: four ``_wrapped_difference``
buffers, four cross-product buffers and one accumulator, so **nine full-size
tensors** are written to produce one, at roughly 57 flops per grid point. Every
one of those intermediates exists only to be read once by the next operation.
Fusing the block into a single stencil pass reads the grid and writes the
normals, and nothing else.

This is written to be bit-identical and **is not**, for a reason that is worth
recording because it is not the obvious one. The structure is the same as
P11/P11b's: the same four differences, the same four cross products from the same
operand pairs, the same three additions in the same ``((A + B) + C) + D`` order,
nothing reassociated, and no term skipped -- a gated triangle contributes an
explicit ``0.0`` rather than being dropped, because ``-0.0 + 0.0`` is ``+0.0``
and ``NaN + 0.0`` is ``NaN``, so dropping the add is observable.

It still differs from the torch block on ~4% of elements by 1-2 ulp, and the
cause is **``torch.cross``**, not Taichi. Measured (see the A/B script):

* Not ``fast_math``. Algan runs Taichi with ``fast_math=True``, which would
  permit contracting ``a * b - c * d`` into an FMA -- but rebuilding this kernel
  with ``fast_math=False`` changes nothing: the same elements differ by the same
  amount.
* ``torch.cross`` is not the textbook expression in float32. On the third
  component alone -- ``a0 * b1 - a1 * b0``, the one that catastrophically
  cancels for a sphere's tangential sides -- it matches neither ``a0 * b1 -
  a1 * b0`` evaluated in float32 nor the same products evaluated exactly in
  double and rounded once. The other two components match both. So the
  divergence is ATen's rounding on the cancelling term, and there is no
  formulation of the kernel that reproduces it: Taichi 1.7.4 exposes no FMA
  intrinsic (no ``fma``/``mad`` in ``ti`` or ``ti.math``).

That makes this the same *kind* of decision ``DESIGN_optimization_targets.md``
already records against the four-crosses-to-one identity -- a real win that costs
bit-identity and needs baselines regenerated -- but on much better terms: 1-2 ulp
on ~4% of vertex normals rather than the algebraic collapse's error, and
8.4-11.3x rather than the ~4x that identity's traffic saving implies. No render was made
here, so the pixel consequence is unmeasured.

Restricted to float32 and to a contiguous ``[B, W, H, 3]`` view, which is what
the batched build passes (``torch.stack`` of same-shaped surface grids, leading
dims flattened by the caller). The float64 fixture in ``_grid_normals_ab.py`` is
a robustness case for the torch arm and is skipped here.
"""

import taichi as ti


@ti.kernel
def grid_normals_sides_crosses(
        grid: ti.types.ndarray(dtype=ti.f32, ndim=4),  # [B, W, H, 3]
        out: ti.types.ndarray(dtype=ti.f32, ndim=4),  # [B, W, H, 3]
):
    """Accumulated (unnormalized) vertex normals, one pass over the grid.

    Parallel over every grid point. The four neighbours wrap, matching
    ``_wrapped_difference``'s two-piece write; the four triangles are gated by
    the same boundary rules the torch form applies as slice assignments:
    ``xm_ym`` dies on the low x or low y edge, ``ym_xp`` on high x or low y,
    ``xp_yp`` on high x or high y, ``yp_xm`` on low x or high y.
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

        # grid.roll(shift, axis) - grid, evaluated in place.
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
