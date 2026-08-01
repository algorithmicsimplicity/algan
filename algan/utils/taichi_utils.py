import taichi as ti


def elementwise(user_fn):
    op = ti.func(user_fn)

    @ti.kernel
    def run(
        src: ti.types.ndarray(),
        dst: ti.types.ndarray(),
    ):
        for i in src:
            dst[i] = op(src[i])

    return run
