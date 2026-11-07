"""MPS's integer gather, emulated on any machine.

On an Apple GPU, ``index_select`` returns an int32 or int64 as
``float32(value)``, which is exact only through 2**24
(``mps_compat._MPS_EXACT_INT_BITS``; DESIGN_mps_support.md §2.3b and §4.9).
``torch.gather`` did the same on the build §2.3b measured. While
:func:`round_integer_gathers` is in effect both answer that way on the CPU
too, so a test can show that a gather site rounds, or no longer does, without
an Apple GPU. Advanced indexing ``v[i]``, which ``mps_compat.gather_exact``
uses in MPS-friendly mode, is left exact, as it measured on the GPU.

:func:`merge_wide_uniques` does the same for ``torch.unique``, whose inverse
and ``dim`` forms merge distinct integers past 2**24 on an Apple GPU.
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


def _merging(unique):
    def merged(input, sorted=True, return_inverse=False, return_counts=False, dim=None):
        kwargs = {"sorted": sorted, "return_inverse": return_inverse}
        kwargs.update(return_counts=return_counts, dim=dim)
        if not (return_inverse or dim is not None) or input.dtype not in (
            torch.int32,
            torch.int64,
        ):
            return unique(input, **kwargs)
        out = unique(input.to(torch.int32).to(torch.float32), **kwargs)
        if isinstance(out, tuple):
            return (out[0].to(input.dtype), *out[1:])
        return out.to(input.dtype)

    return merged


def merge_wide_uniques(monkeypatch) -> None:
    """Make ``torch.unique`` answer as it does on an Apple GPU whenever it is
    asked for an inverse or given a ``dim``, until ``monkeypatch`` undoes it.

    On torch 2.13.0's MPS those forms return what the CPU returns for
    ``float32(int32(x))``: an int64 keeps its low 32 bits, and every value is
    rounded to a float32, so distinct integers past 2**24 merge and come back
    rounded. The values-only and counts-only forms are exact there
    (``benchmarks/_mps_unique_probe.py``, run 42351444222). ``Tensor.unique``
    calls ``torch.unique``, so it merges too. ``unique_consecutive`` is left
    alone: it groups exactly at both widths, and only its values go wrong, as
    ``int32(x)``, for an int64 from 2**31 on (run 42355723203).

    Raises ``AssertionError`` if the merging is not in effect afterwards.
    """
    monkeypatch.setattr(torch, "unique", _merging(torch.unique))
    pair = torch.tensor([FIRST_ROUNDED - 1, FIRST_ROUNDED], dtype=torch.int64)
    if torch.unique(pair, return_inverse=True)[0].numel() != 1:
        raise AssertionError(
            "the simulated MPS unique is not in effect, so the test proves nothing"
        )
