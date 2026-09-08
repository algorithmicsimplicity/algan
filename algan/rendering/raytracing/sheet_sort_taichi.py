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
