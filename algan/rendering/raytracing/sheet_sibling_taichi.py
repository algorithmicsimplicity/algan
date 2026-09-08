"""Sibling coverage in two passes, without per-operation Torch temporaries."""

from algan.rendering.raytracing.raster_taichi import (
    _AA_MASK_ALL,
    _AA_NUM_SAMPLES,
    _AA_SLIVER_BIT,
)
from algan.taichi_compat import ti


@ti.kernel
def sibling_band_counts(
    band: ti.types.ndarray(),
    n: ti.i32,
    counts: ti.types.ndarray(),
):
    # Integer atomics preserve the original member/run counts in any order.
    # Counts start at zero; column 0 is members, column 1 is disjoint runs.
    for i in range(n):
        b = band[i]
        ti.atomic_add(counts[b, 0], 1)
        start = i == 0
        if i > 0:
            start = b != band[i - 1]
        if start:
            ti.atomic_add(counts[b, 1], 1)


@ti.kernel
def sibling_coverage_weights(
    band: ti.types.ndarray(),
    cov: ti.types.ndarray(),
    mask: ti.types.ndarray(),
    area: ti.types.ndarray(),
    union: ti.types.ndarray(),
    correction: ti.types.ndarray(),
    counts: ti.types.ndarray(),
    n: ti.i32,
    weights: ti.types.ndarray(),
    masks: ti.types.ndarray(),
    acc: ti.template(),
):
    for i in range(n):
        b = band[i]
        weight = cov[i]
        word = mask[i]
        # An interleaved band must keep each sheet's original coverage/mask.
        # Only an uninterrupted multi-sheet run may defer its occlusion.
        if counts[b, 0] > 1 and counts[b, 1] == 1:
            a = ti.max(ti.cast(area[b], acc), ti.cast(1e-12, acc))
            share = ti.cast(cov[i], acc) / a
            p = ti.cast(correction[b], acc) * share
            u = ti.cast(union[b], ti.i32)
            if u != _AA_MASK_ALL:
                pop = 0
                for lane in ti.static(range(_AA_NUM_SAMPLES)):
                    pop += (u >> lane) & 1
                p = p * ti.cast(ti.max(pop, 1), acc) / ti.cast(_AA_NUM_SAMPLES, acc)
            weight = ti.cast(p, ti.f32)
            if i + 1 < n:
                if band[i + 1] == b:
                    weight = -weight
            word = u | (word & ~_AA_MASK_ALL & ~_AA_SLIVER_BIT)
        weights[i] = weight
        masks[i] = word
