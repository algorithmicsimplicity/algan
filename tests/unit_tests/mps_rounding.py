"""MPS's integer gather, emulated on any machine.

On an Apple GPU, ``index_select`` returns an int32 or int64 as
``float32(value)``, which is exact only through 2**24
(``mps_compat._MPS_EXACT_INT_BITS``; DESIGN_mps_support.md §2.3b and §4.9).
``torch.gather`` did the same on the build §2.3b measured. While
:func:`round_integer_gathers` is in effect both answer that way on the CPU
too, so a test can show that a gather site rounds, or no longer does, without
an Apple GPU. Advanced indexing ``v[i]``, which ``mps_compat.gather_exact``
uses in MPS-friendly mode, is left exact, as it measured on the GPU.
"""

from __future__ import annotations

import torch

#: Exact through 2**24, so the first integer a float32 cannot hold.
FIRST_ROUNDED = 2**24 + 1


def _rounding(gather):
    def rounded(*args, **kwargs):
        out = gather(*args, **kwargs)
        if out.dtype in (torch.int32, torch.int64):
            out.copy_(out.to(torch.float32).to(out.dtype))
        return out

    return rounded


def round_integer_gathers(monkeypatch) -> None:
    """Make ``index_select`` and ``gather`` round every integer they return
    through a float32, as functions and as methods, until ``monkeypatch``
    undoes it.

    Raises ``AssertionError`` if the rounding is not in effect afterwards,
    because a test that relies on it would then prove nothing.
    """
    for owner in (torch.Tensor, torch):
        for name in ("index_select", "gather"):
            monkeypatch.setattr(owner, name, _rounding(getattr(owner, name)))
    word = torch.tensor([FIRST_ROUNDED], dtype=torch.int32)
    if int(word.index_select(0, torch.zeros(1, dtype=torch.int64))) == FIRST_ROUNDED:
        raise AssertionError(
            "the simulated MPS rounding is not in effect, so the test proves nothing"
        )
