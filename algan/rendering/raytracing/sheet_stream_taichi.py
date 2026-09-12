"""Integer/copy-only passes for sheet streams; no coverage reduction changes."""

from algan.taichi_compat import ti


@ti.kernel
def gather_group_stream(
    order: ti.types.ndarray(), pixel: ti.types.ndarray(),
    group: ti.types.ndarray(), depth: ti.types.ndarray(),
    coverage: ti.types.ndarray(), mask: ti.types.ndarray(),
    pixel_out: ti.types.ndarray(), depth_out: ti.types.ndarray(),
    coverage_out: ti.types.ndarray(), mask_out: ti.types.ndarray(),
    group_start: ti.types.ndarray(), n: ti.i32,
):
    """Gather sorted fields and detect groups without a sorted group-key array."""
    for i in range(n):
        src = ti.cast(order[i], ti.i32)
        pixel_out[i] = pixel[src]
        depth_out[i] = depth[src]
        coverage_out[i] = coverage[src]
        mask_out[i] = mask[src]
        first = i == 0
        if i > 0:
            prev = ti.cast(order[i - 1], ti.i32)
            first = pixel[src] != pixel[prev] or group[src] != group[prev]
        group_start[i] = ti.cast(first, ti.u8)


@ti.kernel
def gather_sheet_records(
    final: ti.types.ndarray(), nearest: ti.types.ndarray(),
    representative: ti.types.ndarray(), key: ti.types.ndarray(),
    ref: ti.types.ndarray(), ab: ti.types.ndarray(), cap: ti.types.ndarray(),
    cov: ti.types.ndarray(), mask: ti.types.ndarray(),
    nfrag: ti.types.ndarray(), fused: ti.types.ndarray(), band: ti.types.ndarray(),
    out_key: ti.types.ndarray(), out_pix: ti.types.ndarray(),
    out_ref: ti.types.ndarray(), out_ab: ti.types.ndarray(), out_cap: ti.types.ndarray(),
    out_cov: ti.types.ndarray(), out_mask: ti.types.ndarray(),
    out_nfrag: ti.types.ndarray(), out_fused: ti.types.ndarray(), out_band: ti.types.ndarray(),
    n: ti.i32, has_band: ti.template(),
):
    """Compose both permutations while gathering final records once.

    Packed keys are copied as integers, including on Metal. The depth comes
    from the nearest positioned fragment, not the dominant shading fragment.
    """
    for i in range(n):
        sheet = ti.cast(final[i], ti.i32)
        near = ti.cast(nearest[sheet], ti.i32)
        rep = ti.cast(representative[sheet], ti.i32)
        k = key[near]
        out_key[i] = k
        out_pix[i] = k >> 32
        out_ref[i] = ref[rep]
        out_ab[i, 0] = ab[rep, 0]
        out_ab[i, 1] = ab[rep, 1]
        out_cap[i] = cap[rep]
        out_cov[i] = cov[sheet]
        out_mask[i] = mask[sheet]
        out_nfrag[i] = nfrag[sheet]
        out_fused[i] = fused[sheet]
        if ti.static(has_band):
            out_band[i] = band[sheet]


@ti.kernel
def pixel_run_flags(key: ti.types.ndarray(), flags: ti.types.ndarray(), n: ti.i32):
    """Detect pixel boundaries directly from packed keys (no decoded stream)."""
    for i in range(n):
        first = i == 0
        if i > 0:
            first = (key[i] >> 32) != (key[i - 1] >> 32)
        flags[i] = ti.cast(first, ti.i32)


@ti.kernel
def write_pixel_runs(
    key: ti.types.ndarray(), prefix: ti.types.ndarray(),
    covered: ti.types.ndarray(), offsets: ti.types.ndarray(), n: ti.i32, nr: ti.i32,
):
    """Scatter boundary positions using an exact integer inclusive scan."""
    for i in range(n):
        first = i == 0
        if i > 0:
            first = prefix[i] != prefix[i - 1]
        if first:
            run = prefix[i] - 1
            covered[run] = key[i] >> 32
            offsets[run] = i
        if i == 0:
            offsets[nr] = n


@ti.kernel
def retained_pixel_counts(
    opaque: ti.types.ndarray(), offsets: ti.types.ndarray(),
    counts: ti.types.ndarray(), nr: ti.i32,
):
    """Each run retains its first opaque fragment, inclusive, or its full length."""
    for run in range(nr):
        start = ti.cast(offsets[run], ti.i32)
        end = ti.cast(offsets[run + 1], ti.i32)
        stop = end
        i = start
        while i < end:
            if opaque[i] != 0:
                stop = i + 1
                break
            i += 1
        counts[run] = stop - start


@ti.kernel
def retained_pixel_indices(
    old_offsets: ti.types.ndarray(), new_offsets: ti.types.ndarray(),
    indices: ti.types.ndarray(), nr: ti.i32,
):
    """Build the prefix gather without a full-stream keep mask or nonzero scan."""
    for run in range(nr):
        dst = ti.cast(new_offsets[run], ti.i32)
        end = ti.cast(new_offsets[run + 1], ti.i32)
        src = ti.cast(old_offsets[run], ti.i32)
        for j in range(dst, end):
            indices[j] = src + j - dst
