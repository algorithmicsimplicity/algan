"""Step 0 of roadmap 6a-bis: can a REMAPPING view over ``light_col`` be built
in kernel scope and read through the existing, unmodified shading funcs?

The authored-appearance branch of ``pt_shade`` wants to hand
``_run_frag_pipeline`` a *subset* of the packed light rows -- the ambient rows
plus a few sampled ones -- with each sampled row's radiance pre-scaled by its
Monte Carlo weight, WITHOUT touching ``shading_taichi.py`` or the 16-argument
stage signature. The mechanism proposed is a read-only view in the
``ArenaView`` idiom: a tuple subclass whose ``__getitem__`` rewrites
``view[tl, slot, c]`` into ``inner[tl, rows[slot], c]`` (times a per-slot
weight for the three radiance channels), where ``rows`` and ``scale`` are
per-thread ``ti.Vector`` locals filled at the crossing.

Every property that has to hold is checked here, because a Taichi scoping
failure is a compile error the host cannot see:

1. ``ti.Vector`` i32 / f32 locals declared BEFORE a loop and written inside it
   with a RUNTIME index (the vectors are filled per surface crossing);
2. a tuple-subclass view constructed in kernel scope over an ``ArenaView``
   (which is itself a view -- so this is a view over a view);
3. reading through it with the runtime slot index and Python-literal channel;
4. ``view.shape[0]`` / ``.shape[2]``, which every callee uses for the frame
   wrap and the compact/extended row test;
5. passing it into the REAL, unmodified ``shading_taichi._light_eval``;
6. bit-for-bit agreement with the plain spelling, weight included.

Run it once per compiler:

    uv run python benchmarks/_pt_sampled_light_view_spike.py
    ALGAN_TAICHI_BACKEND=taichi uv run python benchmarks/_pt_sampled_light_view_spike.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from algan.rendering.raytracing.arena_args_taichi import ArenaView  # noqa: E402
from algan.rendering.raytracing.shading_taichi import _light_eval  # noqa: E402
from algan.rendering.taichi_runtime import taichi_init_kwargs  # noqa: E402
from algan.taichi_compat import BACKEND, submodule, ti  # noqa: E402

_ti_impl = submodule("lang.impl")

ti.init(**taichi_init_kwargs())


class SampledLightView(tuple):
    """``inner`` re-indexed by ``rows``, radiance channels scaled by ``scale``.

    Read-only: the scaled channels return an rvalue, so this is deliberately
    NOT the lvalue an ``ArenaView`` subscript is.
    """

    __slots__ = ()

    def __new__(cls, inner, rows, scale):
        return super().__new__(cls, (inner, rows, scale))

    @property
    def inner(self):
        return tuple.__getitem__(self, 0)

    @property
    def rows(self):
        return tuple.__getitem__(self, 1)

    @property
    def scale(self):
        return tuple.__getitem__(self, 2)

    @property
    def shape(self):
        return self.inner.shape

    def __getitem__(self, idx):
        tl, slot, c = idx
        # ``ti.Vector`` locals are matrix-typed ``Expr``s in kernel scope, not
        # Python ``Matrix`` objects, so they are indexed through the compiler's
        # own builder rather than with ``[]`` (which is exactly what
        # ``ArenaView`` does for the arena buffer).
        row = _ti_impl.subscript(None, self.rows, slot)
        val = self.inner[tl, row, c]
        w = self.scale
        if w is None:
            return val
        ws = _ti_impl.subscript(None, w, slot)
        if isinstance(c, int):
            return val * ws if c < 3 else val
        return val * ti.select(c < 3, ws, 1.0)


@ti.func
def _read_c(view: ti.template(), tl, slot, c):
    """A nested ti.func reading the view -- the ``_light_eval`` shape."""
    return view[tl, slot, c]


@ti.kernel
def probe(
    arena_f32: ti.types.ndarray(),
    base: ti.i32,
    rows_n: ti.i32,
    cols: ti.i32,
    n_slots: ti.i32,
    out: ti.types.ndarray(),
    out2: ti.types.ndarray(),
):
    for _t in range(1):
        inner = ti.static(ArenaView(arena_f32, base, (1, rows_n, cols)))
        # (1) locals declared before the loop, written inside it at a runtime
        # index -- the per-crossing fill.
        lrow = ti.Vector([0] * 4)
        lscale = ti.Vector([0.0] * 4)
        # (2) the view over the view, constructed ONCE (the vectors are
        # mutated in place afterwards and read at every use).
        view = ti.static(SampledLightView(inner, lrow, lscale))
        for s in range(n_slots):
            lrow[s] = (rows_n - 1) - s
            lscale[s] = 1.0 + 0.5 * ti.cast(s, ti.f32)
        for s in range(n_slots):
            # (3) + (4): runtime slot, literal channel, forwarded shape.
            tl = 0 % view.shape[0]
            for c in ti.static(range(3)):
                out[s, c] = _read_c(view, tl, s, c)
            out[s, 3] = view[tl, s, 3]
            out[s, 4] = ti.cast(view.shape[2], ti.f32)
        # (5) the real, unmodified _light_eval through the view.
        pos = ti.math.vec3(0.0, 0.0, 0.0)
        nrm = ti.math.vec3(0.0, 1.0, 0.0)
        for s in range(n_slots):
            ld, lc, sw, frac = _light_eval(view, view, 0, s, pos, nrm)
            out2[s, 0] = lc[0]
            out2[s, 1] = lc[1]
            out2[s, 2] = lc[2]
            out2[s, 3] = ld[0]
            out2[s, 4] = sw
            out2[s, 5] = frac


def main():
    rows_n, cols, n_slots = 6, 16, 4
    host = torch.arange(rows_n * cols, dtype=torch.float32).reshape(rows_n, cols)
    host[:, 3] = 0.0  # every row a point light
    host[:, 4] = 0.0  # no decay
    host[:, 5] = 0.0  # no range fade
    host[:, 15] = 1.0  # whole light
    base = 7
    arena = torch.zeros(base + rows_n * cols, dtype=torch.float32)
    arena[base:] = host.reshape(-1)
    out = torch.zeros((n_slots, 5), dtype=torch.float32)
    out2 = torch.zeros((n_slots, 6), dtype=torch.float32)
    probe(arena, base, rows_n, cols, n_slots, out, out2)

    ok = True
    for s in range(n_slots):
        row = (rows_n - 1) - s
        w = 1.0 + 0.5 * s
        want = [host[row, c].item() * (w if c < 3 else 1.0) for c in range(4)]
        got = [out[s, c].item() for c in range(4)]
        if any(abs(a - b) > 1e-5 for a, b in zip(want, got)):
            ok = False
            print(f"  slot {s}: remap/scale MISMATCH want={want} got={got}")
        if int(out[s, 4].item()) != cols:
            ok = False
            print(f"  slot {s}: .shape[2] MISMATCH {out[s, 4].item()} != {cols}")
        want_lc = [host[row, c].item() * w for c in range(3)]
        got_lc = [out2[s, c].item() for c in range(3)]
        if any(abs(a - b) > 1e-4 for a, b in zip(want_lc, got_lc)):
            ok = False
            print(f"  slot {s}: _light_eval MISMATCH want={want_lc} got={got_lc}")
        if abs(out2[s, 5].item() - 1.0) > 1e-5:
            ok = False
            print(f"  slot {s}: frac (unscaled column 15) MISMATCH {out2[s, 5]}")
    print(
        f"backend {BACKEND} {ti.__version__}: "
        f"{'PASS' if ok else 'FAIL'} (remap + scale + real _light_eval)"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
