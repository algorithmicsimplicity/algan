"""CPU batch-prep kernel for :class:`TrianglePrimitive`'s vertex-colour bake.

``DESIGN_optimization_targets.md`` P10b measures this at **13.5% of
``get_render_primitives_batched``** -- the largest row of the stage's
per-surface tail, and one no earlier plan had named. The torch form is

    self.colors = colors.clone()
    self.colors[..., -2:-1] += glow
    self.colors[..., -1:] *= opacity

which is one full-size clone of the ``[T, M, 5]`` colours followed by two
in-place passes over strided one-channel views of it. Three passes to produce
one buffer, where the arithmetic is one add and one multiply per row.

``glow`` and ``opacity`` reach that code as ``broadcast_all`` results, i.e.
``expand``ed stride-0 views: measured on a real build, ``opacity`` comes from a
single element and ``glow`` carries one value per row. The kernel takes them as
a contiguous ``[N]`` or ``[1]`` buffer with a row stride of 1 or 0, so the
single-element case is never materialized into a full-size tensor just to be
read once.

Byte-identical to the torch form: each output channel is either a copy, one
``+``, or one ``*`` of exactly the operands torch uses, and no reduction or
reassociation is involved.

**It does not pay, and ships off by default** (see
:data:`algan.rendering.taichi_runtime._CPU_PREP_KERNELS_ON_BY_DEFAULT`; opt in
with ``ALGAN_OPT_ENABLE=cpucolors``). Measured at **0.89-0.92x** against the
clone-plus-two-passes form on ``[50, 100000, 5]`` and ``[50, 500000, 5]`` --
0.79-0.81x before the loop was flattened to 1-D offsets, which is the same fix
that rescued the gather but is not enough here.

The "three passes to one" reasoning that made this look promising overcounted:
the two in-place passes touch *one channel each*, not the full width, so torch
moves about ``14N`` floats where the kernel moves ``10N`` -- a 1.4x traffic
saving, not 3x, and not enough to cover a kernel launch against torch's
vectorized ``clone``. Unrolling the channel copy at compile time
(``channels: ti.template()``) changed almost nothing, confirming it is
bandwidth-bound rather than codegen-bound.

This one is against the structural floor rather than a fixable loop shape:
``benchmarks/_taichi_loop_shapes_taichi.py`` measures even a flat Taichi copy at
0.25-0.75x of torch's vectorized ``copy_``, and a bake that is one add and one
multiply over a full-width copy cannot make that back. Kept because it is
correct and the measurement should be reproducible elsewhere;
``benchmarks/_cpu_prep_kernels_ab.py`` is the harness.

Dispatched only when Taichi's arch is the CPU -- see
:func:`algan.rendering.taichi_runtime.cpu_prep_kernel_enabled`.
"""

import taichi as ti


@ti.kernel
def apply_glow_and_opacity(
        colors: ti.types.ndarray(dtype=ti.f32, ndim=1),  # [N * D]
        glow: ti.types.ndarray(dtype=ti.f32, ndim=1),  # [N] or [1]
        opacity: ti.types.ndarray(dtype=ti.f32, ndim=1),  # [N] or [1]
        out: ti.types.ndarray(dtype=ti.f32, ndim=1),  # [N * D]
        glow_stride: ti.i32,  # 1 per-row, 0 broadcast from one element
        opacity_stride: ti.i32,
        channels: ti.template(),  # D, as a template so the copy unrolls
):
    """Copy ``colors``, adding glow to channel ``D - 2`` and scaling ``D - 1``.

    The strides carry the broadcast rather than an expanded input: a stride of
    0 reads element 0 for every row, which is what indexing a stride-0 view
    does, without a tensor of that size existing.
    """
    # Flat offsets rather than 2-D indexing: measured on a pure copy of the same
    # bytes, multi-dimensional ndarray addressing runs at roughly a third of a
    # flat 1-D loop's throughput. Compile-time channel range (Taichi specializes
    # on template arguments), so the untouched channels unroll.
    for n in range(colors.shape[0] // channels):
        base = n * channels
        for d in ti.static(range(channels - 2)):
            out[base + d] = colors[base + d]
        out[base + channels - 2] = colors[base + channels - 2] + glow[n * glow_stride]
        out[base + channels - 1] = colors[base + channels - 1] * opacity[n * opacity_stride]
