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

Dispatched only when Taichi's arch is the CPU -- see
:func:`algan.rendering.taichi_runtime.cpu_prep_kernel_enabled`.

Byte-identical to the torch form: each output channel is either a copy, one
``+``, or one ``*`` of exactly the operands torch uses, and no reduction or
reassociation is involved.
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
):
    """Copy ``colors``, adding glow to channel ``D - 2`` and scaling ``D - 1``.

    The strides carry the broadcast rather than an expanded input: a stride of
    0 reads element 0 for every row, which is what indexing a stride-0 view
    does, without a tensor of that size existing.
    """
    D = colors.shape[1]
    for n in range(colors.shape[0]):
        for d in range(D - 2):
            out[n, d] = colors[n, d]
        out[n, D - 2] = colors[n, D - 2] + glow[n * glow_stride]
        out[n, D - 1] = colors[n, D - 1] * opacity[n * opacity_stride]
