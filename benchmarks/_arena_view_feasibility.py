"""Can an arena slice be handed to the EXISTING ti.funcs, unchanged?

``impl.subscript`` evaluates a non-Taichi ``value`` in Python
(``value.__getitem__(indices)``), and every array parameter of algan's render
``ti.func``s is annotated ``ti.template()`` -- which inlines whatever object it
is given. So a plain Python object holding (arena AnyArray, base offset, shape)
should be indexable in Taichi scope AND passable into the existing funcs with
no edit to them.

This probe checks each thing that has to be true for that to work:

1. constructing the view inside kernel scope;
2. reading through it, 2-D and 3-D;
3. *writing* through it -- needs ``__getitem__`` to return a genuine lvalue;
4. ``view.shape[0]`` (the funcs' frame-wrap idiom, ``f % tri_pos.shape[0]``);
5. passing it into a REAL, unmodified algan ti.func (``_triangle_normal``);
6. bit-for-bit agreement with the plain-ndarray spelling.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from algan.rendering.taichi_runtime import taichi_init_kwargs  # noqa: E402
from algan.taichi_compat import submodule, ti  # noqa: E402

_ti_impl = submodule("lang.impl")

ti.init(**taichi_init_kwargs())


class View(tuple):
    """A window into a flat arena, quacking like an ndarray parameter.

    Constructed in Taichi scope: ``base`` is a Taichi Expr (or a Python int)
    and ``shape`` is a tuple of Python ints. ``__getitem__`` returns the
    arena's own IndexExpression, which is an lvalue -- so stores through the
    view work exactly like stores through the array it replaces.

    Subclasses ``tuple`` for one reason: ``ti.static`` accepts a tuple and
    passes it through untouched, which is what lets the view be bound to a
    local NAME in kernel scope. Without that, Taichi's assignment builder tries
    to create a Taichi local variable of type ``View`` and fails; the view then
    has to be spelled out at every call site instead.

    ``shape`` being a compile-time constant is strictly *better* than what the
    funcs get today: an ndarray's ``.shape`` is a runtime value, so
    ``f % tri_pos.shape[0]`` currently emits a runtime modulo.
    """

    def __new__(cls, buf, base, shape):
        return super().__new__(cls, (buf, base, tuple(shape)))

    @property
    def buf(self):
        return tuple.__getitem__(self, 0)

    @property
    def base(self):
        return tuple.__getitem__(self, 1)

    @property
    def shape(self):
        return tuple.__getitem__(self, 2)

    def __getitem__(self, idx):
        if not isinstance(idx, tuple):
            idx = (idx,)
        shape = self.shape
        assert len(idx) == len(shape), (len(idx), shape)
        flat = idx[0]
        for d in range(1, len(idx)):
            flat = flat * shape[d] + idx[d]
        # Python scope here: AnyArray has no Python __getitem__, so go
        # through Taichi's own subscript builder. The IndexExpression it
        # returns is an lvalue, which is what makes stores work.
        return _ti_impl.subscript(None, self.buf, self.base + flat)


@ti.kernel
def k_ref(n: ti.i32, a: ti.types.ndarray(), out: ti.types.ndarray()):
    for i in range(n):
        v = ti.math.vec3(a[i, 0], a[i, 1], a[i, 2])
        a[i, 1] = v[0] + v[2]  # store through the array
        out[i] = v[0] * 2.0 + v[1] - v[2] + a[i, 1]


@ti.kernel
def k_view(
    n: ti.i32,
    arena: ti.types.ndarray(),
    off: ti.types.ndarray(),
    rows: ti.template(),
    out: ti.types.ndarray(),
):
    a = ti.static(View(arena, off[0], (rows, 3)))
    for i in range(n):
        v = ti.math.vec3(a[i, 0], a[i, 1], a[i, 2])
        a[i, 1] = v[0] + v[2]  # store through the VIEW
        out[i] = v[0] * 2.0 + v[1] - v[2] + a[i, 1]


def main():
    from algan.rendering.raytracing.raytrace_kernels_taichi import (
        _triangle_normal,
    )

    arch = ti.lang.impl.current_cfg().arch
    dev = "cuda" if arch == ti.cuda else "cpu"
    print("arch:", arch)

    n = 4096
    pad = 128  # non-zero base, so the view is not trivially the whole arena
    gen = torch.Generator(device=dev).manual_seed(7)
    src = torch.rand(n * 3, device=dev, generator=gen, dtype=torch.float32)

    a_ref = src.clone().view(n, 3)
    out_ref = torch.zeros(n, device=dev, dtype=torch.float32)
    k_ref(n, a_ref, out_ref)

    arena = torch.zeros(pad + n * 3, device=dev, dtype=torch.float32)
    arena[pad:] = src
    off = torch.tensor([pad], device=dev, dtype=torch.int32)
    out_view = torch.zeros(n, device=dev, dtype=torch.float32)
    k_view(n, arena, off, n, out_view)

    print("2-D reads agree:      ", torch.equal(out_ref, out_view))
    print("stores through view:  ", torch.equal(a_ref.reshape(-1), arena[pad:]))

    # --- 5. a REAL algan ti.func, unmodified --------------------------------
    # _triangle_normal does 3-D template indexing AND reads ``.shape[0]``.
    @ti.kernel
    def k_real_ref(
        m: ti.i32,
        frames: ti.i32,
        tri_norm: ti.types.ndarray(),
        tri_pos: ti.types.ndarray(),
        o: ti.types.ndarray(),
    ):
        for i in range(m):
            nrm = _triangle_normal(i % frames, i, 0.2, 0.3, 0.5, tri_norm, tri_pos)
            o[i] = nrm[0] + nrm[1] * 2.0 + nrm[2] * 3.0

    @ti.kernel
    def k_real_view(
        m: ti.i32,
        frames: ti.i32,
        arena_: ti.types.ndarray(),
        off_: ti.types.ndarray(),
        fr: ti.template(),
        pr: ti.template(),
        o: ti.types.ndarray(),
    ):
        tri_norm = ti.static(View(arena_, off_[0], (fr, pr, 9)))
        tri_pos = ti.static(View(arena_, off_[1], (fr, pr, 9)))
        for i in range(m):
            nrm = _triangle_normal(i % frames, i, 0.2, 0.3, 0.5, tri_norm, tri_pos)
            o[i] = nrm[0] + nrm[1] * 2.0 + nrm[2] * 3.0

    frames, prims = 4, 512
    numel = frames * prims * 9
    # Half the normals are deliberately degenerate so the geometric-normal
    # fallback -- the branch that reads tri_pos and tri_pos.shape[0] -- runs.
    tn_src = torch.rand(numel, device=dev, generator=gen, dtype=torch.float32)
    tn_src.view(frames, prims, 9)[:, ::2, :] = 0.0
    tp_src = torch.rand(numel, device=dev, generator=gen, dtype=torch.float32)

    o1 = torch.zeros(prims, device=dev, dtype=torch.float32)
    k_real_ref(
        prims,
        frames,
        tn_src.clone().view(frames, prims, 9),
        tp_src.clone().view(frames, prims, 9),
        o1,
    )

    arena2 = torch.zeros(pad + 2 * numel, device=dev, dtype=torch.float32)
    arena2[pad : pad + numel] = tn_src
    arena2[pad + numel :] = tp_src
    off2 = torch.tensor([pad, pad + numel], device=dev, dtype=torch.int32)
    o2 = torch.zeros(prims, device=dev, dtype=torch.float32)
    k_real_view(prims, frames, arena2, off2, frames, prims, o2)

    print(
        "real ti.func agrees:  ",
        torch.equal(o1, o2),
        "(max |d| =",
        float((o1 - o2).abs().max().item()),
        ")",
    )

    # --- 7. RUNTIME shapes --------------------------------------------------
    # Compile-time shapes would specialize the kernel per scene -- a
    # non-starter when a cold megakernel compile is minutes. The view has to
    # work with shape entries that are Taichi Exprs read from a metadata
    # ndarray, so nothing about the geometry is baked into the kernel.
    @ti.kernel
    def k_real_view_rt(
        m: ti.i32,
        frames: ti.i32,
        arena_: ti.types.ndarray(),
        off_: ti.types.ndarray(),
        shp: ti.types.ndarray(),
        o: ti.types.ndarray(),
    ):
        tri_norm = ti.static(View(arena_, off_[0], (shp[0], shp[1], shp[2])))
        tri_pos = ti.static(View(arena_, off_[1], (shp[0], shp[1], shp[2])))
        for i in range(m):
            nrm = _triangle_normal(i % frames, i, 0.2, 0.3, 0.5, tri_norm, tri_pos)
            o[i] = nrm[0] + nrm[1] * 2.0 + nrm[2] * 3.0

    shp = torch.tensor([frames, prims, 9], device=dev, dtype=torch.int32)
    o3 = torch.zeros(prims, device=dev, dtype=torch.float32)
    k_real_view_rt(prims, frames, arena2, off2, shp, o3)
    print(
        "runtime-shape view:   ",
        torch.equal(o1, o3),
        "(max |d| =",
        float((o1 - o3).abs().max().item()),
        ")",
    )


if __name__ == "__main__":
    main()
