"""Stable sheet-group ordering within the emission's existing pixel runs."""

from algan.taichi_compat import ti


@ti.func
def _after(a, b, group: ti.template(), depth: ti.template(), depth_key: ti.template()):
    # Original position is the final key, reproducing a stable sort even in
    # the heap arm. Inspect NaN bits without relying on fast-math comparisons.
    result = group[a] > group[b]
    if group[a] == group[b]:
        result = a > b
        if ti.static(depth_key):
            da, db = depth[a], depth[b]
            na = (ti.bit_cast(da, ti.u32) & ti.u32(0x7fffffff)) > ti.u32(0x7f800000)
            nb = (ti.bit_cast(db, ti.u32) & ti.u32(0x7fffffff)) > ti.u32(0x7f800000)
            result = False
            if na:
                result = not nb or a > b
            elif not nb:
                result = da > db or (da == db and a > b)
    return result


@ti.func
def _sift(start, root, size, order: ti.template(), group: ti.template(), depth: ti.template(), depth_key: ti.template()):
    value = ti.cast(order[start + root], ti.i32)
    node = root
    walking = True
    while walking and 2 * node + 1 < size:
        child = 2 * node + 1
        if child + 1 < size:
            if _after(ti.cast(order[start + child + 1], ti.i32),
                      ti.cast(order[start + child], ti.i32), group, depth, depth_key):
                child += 1
        other = ti.cast(order[start + child], ti.i32)
        if _after(other, value, group, depth, depth_key):
            order[start + node] = other
            node = child
        else:
            walking = False
    order[start + node] = value


@ti.func
def _sort_run(start, end, order: ti.template(), group: ti.template(), depth: ti.template(), depth_key: ti.template()):
    size = end - start
    if size <= 16:
        for i in range(start + 1, end):
            value = ti.cast(order[i], ti.i32)
            j = i
            walking = True
            while walking and j > start:
                prev = ti.cast(order[j - 1], ti.i32)
                if _after(prev, value, group, depth, depth_key):
                    order[j] = prev
                    j -= 1
                else:
                    walking = False
            order[j] = value
    else:
        root = size // 2 - 1
        while root >= 0:
            _sift(start, root, size, order, group, depth, depth_key)
            root -= 1
        remaining = size - 1
        while remaining > 0:
            last = order[start + remaining]
            order[start + remaining] = order[start]
            order[start] = last
            _sift(start, 0, remaining, order, group, depth, depth_key)
            remaining -= 1


@ti.kernel
def pixel_group_order(
    offsets: ti.types.ndarray(),
    group: ti.types.ndarray(),
    depth: ti.types.ndarray(),
    order: ti.types.ndarray(),
    num_pixels: ti.i32,
):
    """Sort each disjoint pixel run by (group, depth, original position).

    Small runs use insertion sort; longer runs use in-place heapsort, keeping
    work O(n log n) without a fragment-count ceiling or per-thread arrays.
    Only the output permutation is allocated. Input fragment arrays stay put.
    """
    ti.loop_config(block_dim=128)
    for pixel in range(num_pixels):
        start = ti.cast(offsets[pixel], ti.i32)
        end = ti.cast(offsets[pixel + 1], ti.i32)
        for i in range(start, end):
            order[i] = i
        _sort_run(start, end, order, group, depth, True)


@ti.kernel
def key_run_order(
    run_key: ti.types.ndarray(),
    group: ti.types.ndarray(),
    depth: ti.types.ndarray(),
    order: ti.types.ndarray(),
    n: ti.i32,
    initialize: ti.template(),
    depth_key: ti.template(),
):
    """Order immutable key runs, optionally starting from a supplied permutation."""
    ti.loop_config(block_dim=128)
    for start in range(n):
        first = start == 0
        if start > 0:
            first = run_key[start] != run_key[start - 1]
        if first:
            end = start + 1
            while end < n:
                if run_key[end] != run_key[start]:
                    break
                end += 1
            if ti.static(initialize):
                for i in range(start, end):
                    order[i] = i
            _sort_run(start, end, order, group, depth, depth_key)


@ti.kernel
def pixel_bin_counts(
    frag_key: ti.types.ndarray(),
    n: ti.i32,
    num_bins: ti.i32,
    counts: ti.types.ndarray(),
    outside: ti.types.ndarray(),
):
    """Fragments per pixel bin (``counts`` PRE-ZEROED), bin = ``key >> 32``.

    A key whose pixel falls outside ``[0, num_bins)`` is not counted; it sets
    ``outside[0]`` instead so the caller can fall back to a comparison sort.
    """
    for i in range(n):
        p = frag_key[i] >> 32
        if p >= 0 and p < num_bins:
            ti.atomic_add(counts[ti.cast(p, ti.i32)], 1)
        else:
            outside[0] = 1


@ti.kernel
def pixel_bin_scatter(
    frag_key: ti.types.ndarray(),
    n: ti.i32,
    cursor: ti.types.ndarray(),
    order: ti.types.ndarray(),
):
    """Scatter each fragment's index into its pixel bin's run of ``order``.

    ``cursor`` holds each bin's first slot on entry. The order WITHIN a bin is
    whatever the atomics made it; :func:`bin_run_order` then sorts every bin
    by a total key ending in the original index, so it cannot leak out.
    """
    for i in range(n):
        slot = ti.atomic_add(cursor[ti.cast(frag_key[i] >> 32, ti.i32)], 1)
        order[slot] = i


@ti.kernel
def bin_run_order(
    offsets: ti.types.ndarray(),
    group: ti.types.ndarray(),
    order: ti.types.ndarray(),
    num_bins: ti.i32,
):
    """Sort each bin's run of ``order`` by (group, original position)."""
    ti.loop_config(block_dim=128)
    for b in range(num_bins):
        start = ti.cast(offsets[b], ti.i32)
        end = ti.cast(offsets[b + 1], ti.i32)
        if end - start > 1:
            _sort_run(start, end, order, group, group, False)
