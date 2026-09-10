"""Integer-only grouping of already ordered sheet runs."""

from algan.taichi_compat import ti


@ti.kernel
def unique_boundaries(keys: ti.types.ndarray(), starts: ti.types.ndarray(), n: ti.i32):
    ti.loop_config(block_dim=128)
    for i in range(n):
        first = i == 0
        if i > 0:
            first = keys[i] != keys[i - 1]
        starts[i] = ti.cast(first, ti.i32)


@ti.kernel
def unique_scatter(
    keys: ti.types.ndarray(), prefix: ti.types.ndarray(),
    unique: ti.types.ndarray(), inverse: ti.types.ndarray(), n: ti.i32,
):
    ti.loop_config(block_dim=128)
    for i in range(n):
        group = prefix[i] - 1
        inverse[i] = group
        first = i == 0
        if i > 0:
            first = prefix[i] != prefix[i - 1]
        if first:
            unique[group] = keys[i]


@ti.func
def _after(a, b, band: ti.template(), cls: ti.template()):
    result = band[a] > band[b]
    if band[a] == band[b]:
        result = cls[a] > cls[b] or (cls[a] == cls[b] and a > b)
    return result


@ti.func
def _sift(start, root, size, order: ti.template(), band: ti.template(), cls: ti.template()):
    value = order[start + root]
    node = root
    walking = True
    # Test parenthood before doubling the index: the public row bound is
    # int32, but 2 * node + 1 can overflow for a leaf in a very large run.
    while walking and node < size // 2:
        child = 2 * node + 1
        if child + 1 < size:
            if _after(order[start + child + 1], order[start + child], band, cls):
                child += 1
        other = order[start + child]
        if _after(other, value, band, cls):
            order[start + node] = other
            node = child
        else:
            walking = False
    order[start + node] = value


@ti.func
def _sort_run(start, end, order: ti.template(), band: ti.template(), cls: ti.template()):
    size = end - start
    if size <= 16:
        for i in range(start + 1, end):
            value = order[i]
            j = i
            walking = True
            while walking and j > start:
                previous = order[j - 1]
                if _after(previous, value, band, cls):
                    order[j] = previous
                    j -= 1
                else:
                    walking = False
            order[j] = value
    else:
        root = size // 2 - 1
        while root >= 0:
            _sift(start, root, size, order, band, cls)
            root -= 1
        remaining = size - 1
        while remaining > 0:
            last = order[start + remaining]
            order[start + remaining] = order[start]
            order[start] = last
            _sift(start, 0, remaining, order, band, cls)
            remaining -= 1


@ti.kernel
def class_order(
    starts: ti.types.ndarray(), band: ti.types.ndarray(), cls: ti.types.ndarray(),
    order: ti.types.ndarray(), boundaries: ti.types.ndarray(), n: ti.i32,
):
    """Stable (band, class) order in disjoint original surface/facing runs.

    Run band ranges must be disjoint and increasing. Insertion sort handles
    short runs; heapsort bounds long-run work without a fragment-count cap.
    """
    ti.loop_config(block_dim=128)
    for start in range(n):
        if start == 0 or starts[start] != 0:
            end = start + 1
            while end < n:
                if starts[end] != 0:
                    break
                end += 1
            for i in range(start, end):
                order[i] = i
            _sort_run(start, end, order, band, cls)
            boundaries[start] = 1
            for i in range(start + 1, end):
                a, b = order[i - 1], order[i]
                boundaries[i] = ti.cast(band[a] != band[b] or cls[a] != cls[b], ti.i32)


@ti.kernel
def class_scatter(
    band: ti.types.ndarray(), order: ti.types.ndarray(), prefix: ti.types.ndarray(),
    inverse: ti.types.ndarray(), group_band: ti.types.ndarray(), n: ti.i32,
):
    ti.loop_config(block_dim=128)
    for i in range(n):
        original = order[i]
        group = prefix[i] - 1
        inverse[original] = group
        first = i == 0
        if i > 0:
            first = prefix[i] != prefix[i - 1]
        if first:
            group_band[group] = band[original]


@ti.kernel
def fragment_metadata(
    keys: ti.types.ndarray(), refs: ti.types.ndarray(), masks: ti.types.ndarray(),
    objects: ti.types.ndarray(), pixels: ti.types.ndarray(), depth: ti.types.ndarray(),
    frames: ti.types.ndarray(), triangles: ti.types.ndarray(), safe_refs: ti.types.ndarray(),
    positions: ti.types.ndarray(), groups: ti.types.ndarray(),
    n: ti.i32, pixels_per_frame: ti.i64, time_start: ti.i32, backface_bit: ti.i32,
):
    ti.loop_config(block_dim=128)
    for i in range(n):
        pixel = keys[i] >> 32
        frame = pixel // pixels_per_frame
        ref = ti.max(refs[i], 0)
        row = ti.cast((frame + time_start) % objects.shape[0], ti.i32)
        surface = objects[row, ti.cast(ref, ti.i32)]
        front = (masks[i] & backface_bit) != 0
        group = -(ti.cast(i, ti.i64) + 2)
        if refs[i] >= 0:
            group = ti.cast(surface, ti.i64) * 2 + ti.cast(front, ti.i64)
        pixels[i] = pixel
        depth[i] = ti.bit_cast(ti.cast(keys[i], ti.i32), ti.f32)
        frames[i] = frame
        triangles[i] = ti.cast(refs[i] >= 0, ti.u8)
        safe_refs[i] = ref
        positions[i] = i
        groups[i] = group
