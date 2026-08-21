"""Bitwise A/B for the batched-axis ``get_orthonormal_vector``.

The rewrite collapses the per-axis Python loop (~40 small dispatches) into one
pass over a new axis dim (~16), keeping the arithmetic and the sequential
strictly-greater selection exactly -- so the result must be **bit-identical**,
including on the degenerate inputs where the difference would hide: seeds
lying exactly in the span of the inputs (axis-aligned cylinders are the common
case), zero vectors, near-parallel pairs, and NaN (the reason selection is a
where-chain, not argmax).

The reference below is the pre-rewrite implementation, inlined verbatim.

    .venv/Scripts/python.exe benchmarks/_orthonormal_bitwise_ab.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

os.environ.setdefault("ALGAN_RENDER_DEVICE", "cpu")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from algan.geometry.geometry import get_orthonormal_vector  # noqa: E402
from algan.utils.tensor_utils import dot_product  # noqa: E402


def reference(*vectors):
    """The per-axis loop this change replaced, verbatim."""
    vectors = [F.normalize(v, p=2, dim=-1) for v in vectors]
    v0 = vectors[0]
    best = torch.zeros_like(v0)
    best_norm = torch.zeros_like(v0[..., :1])
    for axis in range(v0.shape[-1]):
        r = torch.zeros_like(v0)
        r[..., axis] = 1.0
        for vn in vectors:
            r = r - dot_product(r, vn) * vn
        n = r.norm(p=2, dim=-1, keepdim=True)
        take = n > best_norm
        best = torch.where(take, r, best)
        best_norm = torch.where(take, n, best_norm)
    return F.normalize(best, p=2, dim=-1)


def cases():
    g = torch.Generator().manual_seed(0)

    def rand(*shape):
        return torch.randn(*shape, generator=g)

    yield "random [50,1,3] x1", (rand(50, 1, 3),)
    yield "random [50,1,3] x2", (rand(50, 1, 3), rand(50, 1, 3))
    yield "random [7,3] x1", (rand(7, 3),)
    yield "unbatched [3]", (rand(3),)
    # Axis-aligned: the seed is annihilated exactly (r == 0 with signed-zero
    # subtleties) -- the case the selection order exists for.
    for sign in (1.0, -1.0):
        for axis in range(3):
            e = torch.zeros(4, 3)
            e[:, axis] = sign
            yield f"axis-aligned e{axis} sign {sign:+.0f}", (e,)
    # Two vectors spanning a coordinate plane: two of three seeds annihilate.
    yield (
        "spans xy-plane",
        (
            torch.tensor([[1.0, 0.0, 0.0]]),
            torch.tensor([[0.0, 1.0, 0.0]]),
        ),
    )
    # Near-parallel pair: ill-conditioned second projection.
    v = rand(16, 3)
    yield "near-parallel pair", (v, v + 1e-6 * rand(16, 3))
    yield "zero vector", (torch.zeros(5, 3),)
    yield "with NaN", (torch.tensor([[float("nan"), 1.0, 0.0], [0.0, 1.0, 0.0]]),)
    yield "float64", (rand(6, 3).double(),)


def main():
    for name, vectors in cases():
        a = reference(*vectors)
        b = get_orthonormal_vector(*vectors)
        same = (a == b) | (a.isnan() & b.isnan())
        assert bool(same.all()), f"{name}: results differ"
        # == treats -0.0 and +0.0 as equal; require the same bit pattern too.
        assert torch.equal(
            a.float().view(torch.int32), b.float().view(torch.int32)
        ) or bool(a.isnan().any()), f"{name}: signed-zero / bit pattern differs"
        print(f"  {name:<28} bit-identical")

    # Timing on the updater's real shape: [T, 1, 3] with T frames.
    shapes = [(50, 1, 3)]
    for shape in shapes:
        g = torch.Generator().manual_seed(1)
        u = torch.randn(*shape, generator=g)
        w = torch.randn(*shape, generator=g)
        for label, fn in (
            ("reference", reference),
            ("batched", get_orthonormal_vector),
        ):
            fn(u, w)  # warm
            t0 = time.perf_counter()
            n = 300
            for _ in range(n):
                fn(u)
                fn(u, w)
            dt = (time.perf_counter() - t0) / n
            print(f"  {shape} {label:<10} {dt * 1e6:8.1f} us per (x1 + x2) pair")

    print("\nbatched-axis get_orthonormal_vector is bit-identical")


if __name__ == "__main__":
    main()
