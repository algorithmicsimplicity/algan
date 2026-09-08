"""Per-pixel competing-surface depth reduction without expanded lane sorts."""

from algan.rendering.raytracing.raster_taichi import _AA_NUM_SAMPLES
from algan.taichi_compat import ti


@ti.kernel
def sheet_lane_depths_inplace(
    first: ti.types.ndarray(),
    depth: ti.types.ndarray(),
    n: ti.i32,
):
    # The owner reduction has finished. Each thread now consumes only its own
    # slot, so that int32 storage can become the float32 depth table without
    # another allocation. Keep one read/write argument (no aliased ndarrays).
    for slot in first:
        index = first[slot]
        value = float("inf")
        if index < n:
            value = depth[index]
        first[slot] = ti.bit_cast(value, ti.i32)


@ti.kernel
def sheet_lane_depths(
    first: ti.types.ndarray(),
    depth: ti.types.ndarray(),
    n: ti.i32,
    out: ti.types.ndarray(),
):
    for sheet, lane in out:
        index = first[sheet * _AA_NUM_SAMPLES + lane]
        value = float("inf")
        if index < n:
            value = depth[index]
        out[sheet, lane] = value


@ti.kernel
def sheet_depth_lose(
    pixel: ti.types.ndarray(),
    surface: ti.types.ndarray(),
    depths: ti.types.ndarray(),
    mask: ti.types.ndarray(),
    subject: ti.types.ndarray(),
    enforcer: ti.types.ndarray(),
    n: ti.i32,
    epsilon: ti.f32,
    cede: ti.f32,
    lose_shift: ti.i32,
    lose: ti.types.ndarray(),
):
    # Only the first sheet of each pixel runs. Both walks are unbounded:
    # no overlap ceiling or assumption about the number of surfaces.
    for first in range(n):
        start = first == 0
        if first > 0:
            start = pixel[first] != pixel[first - 1]
        if start:
            end = first + 1
            while end < n:
                if pixel[end] != pixel[first]:
                    break
                end += 1
            best = ti.Vector([float("inf")] * _AA_NUM_SAMPLES)
            second = ti.Vector([float("inf")] * _AA_NUM_SAMPLES)
            owner = ti.Vector([ti.cast(-1, ti.i64)] * _AA_NUM_SAMPLES)
            for j in range(first, end):
                if enforcer[j] != 0:
                    sid = surface[j]
                    for lane in ti.static(range(_AA_NUM_SAMPLES)):
                        d = depths[j, lane]
                        if d < best[lane]:
                            if sid != owner[lane]:
                                second[lane] = best[lane]
                            best[lane] = d
                            owner[lane] = sid
                        elif sid != owner[lane] and d < second[lane]:
                            second[lane] = d
            for j in range(first, end):
                word = 0
                if subject[j] != 0:
                    own_count = 0
                    lose_count = 0
                    for lane in ti.static(range(_AA_NUM_SAMPLES)):
                        if ((mask[j] >> lane) & 1) != 0:
                            own_count += 1
                            other = best[lane]
                            if surface[j] == owner[lane]:
                                other = second[lane]
                            if other < depths[j, lane] - epsilon:
                                word |= 1 << lane
                                lose_count += 1
                    if ti.cast(lose_count, ti.f32) <= cede * ti.cast(own_count, ti.f32):
                        word = 0
                lose[j] = word << lose_shift
