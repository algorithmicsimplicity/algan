"""Exact copies from compaction's working arrays into persistent sheet records."""

from algan.taichi_compat import ti


@ti.kernel
def copy_sheet_records(
    final: ti.types.ndarray(), nearest: ti.types.ndarray(), representative: ti.types.ndarray(),
    frag_key: ti.types.ndarray(), frag_ref: ti.types.ndarray(), frag_ab: ti.types.ndarray(),
    frag_cap: ti.types.ndarray(), weights: ti.types.ndarray(), masks: ti.types.ndarray(),
    out_key: ti.types.ndarray(), out_ref: ti.types.ndarray(), out_ab: ti.types.ndarray(),
    out_cov: ti.types.ndarray(), out_msk: ti.types.ndarray(), out_cap: ti.types.ndarray(),
    n: ti.i32,
):
    for i in range(n):
        sheet = final[i]
        near = nearest[sheet]
        rep = representative[sheet]
        # Packed keys are copied as integers, never numerically cast through f32.
        out_key[i] = frag_key[near]
        out_ref[i] = frag_ref[rep]
        out_ab[i, 0] = frag_ab[rep, 0]
        out_ab[i, 1] = frag_ab[rep, 1]
        out_cap[i] = frag_cap[rep]
        out_cov[i] = weights[i]
        out_msk[i] = masks[i]


@ti.kernel
def write_sheet_offsets(
    covered: ti.types.ndarray(), sheet_pixels: ti.types.ndarray(),
    offsets: ti.types.ndarray(), num_covered: ti.i32, num_sheets: ti.i32,
):
    # Every covered pixel has a nonempty sorted sheet run. Its lower bound is
    # its start; the explicit terminal also handles a completely empty stream.
    for i in range(num_covered + 1):
        value = num_sheets
        if i < num_covered:
            target = covered[i]
            lo, hi = 0, num_sheets
            while lo < hi:
                mid = lo + (hi - lo) // 2
                if sheet_pixels[mid] < target:
                    lo = mid + 1
                else:
                    hi = mid
            value = lo
        offsets[i] = value
