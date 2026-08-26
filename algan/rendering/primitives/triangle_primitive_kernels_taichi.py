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
with ``ALGAN_OPT_ENABLE=cpucolors``). Measured at **0.79-0.81x** against the
clone-plus-two-passes form on ``[50, 100000, 5]`` and ``[50, 500000, 5]``.

The "three passes to one" reasoning that made this look promising overcounted:
the two in-place passes touch *one channel each*, not the full width, so torch
moves about ``14N`` floats where the kernel moves ``10N`` -- a 1.4x traffic
saving, not 3x, and not enough to cover a kernel launch against torch's
vectorized ``clone``. Unrolling the channel copy at compile time
(``channels: ti.template()``) changed almost nothing, confirming it is
bandwidth-bound rather than codegen-bound. Kept because it is correct and the
measurement should be reproducible elsewhere; ``benchmarks/_cpu_prep_kernels_ab.py``
is the harness.

Dispatched only when Taichi's arch is the CPU -- see
:func:`algan.rendering.taichi_runtime.cpu_prep_kernel_enabled`.
"""

import taichi as ti


@ti.kernel
def apply_glow_and_opacity(
        colors: ti.types.ndarray(dtype=ti.f32, ndim=2),  # [N, D]
        glow: ti.types.ndarray(dtype=ti.f32, ndim=1),  # [N] or [1]
        opacity: ti.types.ndarray(dtype=ti.f32, ndim=1),  # [N] or [1]
        out: ti.types.ndarray(dtype=ti.f32, ndim=2),  # [N, D]
        glow_stride: ti.i32,  # 1 per-row, 0 broadcast from one element
        opacity_stride: ti.i32,
        channels: ti.template(),  # D, as a template so the copy unrolls
):
    """Copy ``colors``, adding glow to channel ``D - 2`` and scaling ``D - 1``.

    The strides carry the broadcast rather than an expanded input: a stride of
    0 reads element 0 for every row, which is what indexing a stride-0 view
    does, without a tensor of that size existing.
    """
    for n in range(colors.shape[0]):
        # Compile-time range (Taichi specializes on template arguments), so the
        # untouched channels unroll instead of running a runtime-bounded loop.
        for d in ti.static(range(channels - 2)):
            out[n, d] = colors[n, d]
        out[n, channels - 2] = colors[n, channels - 2] + glow[n * glow_stride]
        out[n, channels - 1] = colors[n, channels - 1] * opacity[n * opacity_stride]
