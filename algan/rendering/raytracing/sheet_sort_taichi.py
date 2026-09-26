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
def fragment_bin_keys(
    frag_key: ti.types.ndarray(),
    frag_ref: ti.types.ndarray(),
    n: ti.i32,
    num_bins: ti.i32,
    inv_eps: ti.f32,
    bez_shift: ti.i32,
    layer_offset: ti.i32,
    counts: ti.types.ndarray(),
    group: ti.types.ndarray(),
    outside: ti.types.ndarray(),
):
    """:func:`pixel_bin_counts` plus each fragment's within-pixel sort key.

    ``group[i] = (depth_bin << 32) | (0x7FFFFFFF - layer)``, computed exactly
    as the torch expressions in ``raster_pipeline`` compute it ON CUDA, where
    ``t / eps`` by a host scalar is ``t * (1 / eps)`` in f32 (so ``inv_eps``
    must be that f32 reciprocal): one multiply, a floor, and the clamp to
    ``[0, 2**31)`` spelled as comparisons, which is what the int64 cast and
    clamp give for every input including NaN and the infinities. The layer is
    the circuit id above the border bits for a circuit fragment and the
    offset triangle index otherwise, both in wrapping i32 like the tensors.
    """
    for i in range(n):
        k = frag_key[i]
        p = k >> 32
        if p >= 0 and p < num_bins:
            ti.atomic_add(counts[ti.cast(p, ti.i32)], 1)
        else:
            outside[0] = 1
        t = ti.bit_cast(ti.cast(k, ti.u32), ti.f32)  # the low 32 bits
        x = ti.floor(t * inv_eps)
        db = ti.cast(0, ti.i64)
        if x >= 2147483648.0:
            db = ti.cast(0x7FFFFFFF, ti.i64)
        elif x >= 0.0:
            db = ti.cast(x, ti.i64)
        r = frag_ref[i]
        layer = r + layer_offset
        if r < 0:
            layer = (-r - 1) >> bez_shift
        group[i] = (db << 32) | (ti.cast(0x7FFFFFFF, ti.i64) - ti.cast(layer, ti.i64))


@ti.kernel
def pixel_bin_scatter(
    frag_key: ti.types.ndarray(),
    n: ti.i32,
    cursor: ti.types.ndarray(),
    order: ti.types.ndarray(),
):
    """Scatter each fragment's index into its pixel bin's run of ``order``.

    ``cursor`` holds each bin's first slot on entry. The order WITHIN a bin is
    whatever the atomics made it; :func:`bin_run_sort_pairs` then sorts every bin
    by a total key ending in the original index, so it cannot leak out.
    """
    for i in range(n):
        slot = ti.atomic_add(cursor[ti.cast(frag_key[i] >> 32, ti.i32)], 1)
        order[slot] = i


@ti.func
def _pair_after(keys: ti.template(), order: ti.template(), i, key, idx):
    """Whether slot ``i`` sorts after the pair ``(key, idx)``."""
    k = keys[i]
    return k > key or (k == key and order[i] > idx)


@ti.func
def _pair_sift(keys: ti.template(), order: ti.template(), start, root, size):
    key = keys[start + root]
    idx = order[start + root]
    node = root
    walking = True
    while walking and 2 * node + 1 < size:
        child = 2 * node + 1
        if child + 1 < size:
            if _pair_after(keys, order, start + child + 1, keys[start + child],
                           order[start + child]):
                child += 1
        if _pair_after(keys, order, start + child, key, idx):
            keys[start + node] = keys[start + child]
            order[start + node] = order[start + child]
            node = child
        else:
            walking = False
    keys[start + node] = key
    order[start + node] = idx


@ti.kernel
def bin_run_sort_pairs(
    offsets: ti.types.ndarray(),
    keys: ti.types.ndarray(),
    order: ti.types.ndarray(),
    num_bins: ti.i32,
):
    """Sort each bin's run of ``(keys, order)`` pairs in place by (key, order).

    ``keys`` is the run-local copy of each slot's sort key (``group[order]``),
    so a run's comparisons read contiguous memory instead of chasing
    ``order`` into a fragment-indexed table. ``order`` holds original
    positions, which break key ties: the result is the stable order whatever
    order the slots arrived in. Insertion sort for short runs, in-place
    heapsort for long ones, as in :func:`_sort_run`.
    """
    ti.loop_config(block_dim=128)
    for b in range(num_bins):
        start = ti.cast(offsets[b], ti.i32)
        end = ti.cast(offsets[b + 1], ti.i32)
        size = end - start
        if size > 1:
            if size <= 16:
                for i in range(start + 1, end):
                    key = keys[i]
                    idx = order[i]
                    j = i
                    walking = True
                    while walking and j > start:
                        if _pair_after(keys, order, j - 1, key, idx):
                            keys[j] = keys[j - 1]
                            order[j] = order[j - 1]
                            j -= 1
                        else:
                            walking = False
                    keys[j] = key
                    order[j] = idx
            else:
                root = size // 2 - 1
                while root >= 0:
                    _pair_sift(keys, order, start, root, size)
                    root -= 1
                remaining = size - 1
                while remaining > 0:
                    last_key = keys[start + remaining]
                    last_idx = order[start + remaining]
                    keys[start + remaining] = keys[start]
                    order[start + remaining] = order[start]
                    keys[start] = last_key
                    order[start] = last_idx
                    _pair_sift(keys, order, start, 0, remaining)
                    remaining -= 1
