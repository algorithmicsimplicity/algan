"""Integer-preserving row gathers into explicitly owned contiguous storage."""

from algan.taichi_compat import ti


@ti.kernel
def gather_rows_into(
    source: ti.types.ndarray(), indices: ti.types.ndarray(),
    output: ti.types.ndarray(), rows: ti.i32, columns: ti.i32,
):
    for row, column in ti.ndrange(rows, columns):
        output[row * columns + column] = source[indices[row] * columns + column]
