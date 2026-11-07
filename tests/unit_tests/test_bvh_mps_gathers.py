"""The BVH builders' integer gathers, past the 2**24 an MPS gather is exact to.

On an Apple GPU ``index_select`` hands an int32 or int64 back as
``float32(value)``, exact only through 2**24 (``mps_compat._MPS_EXACT_INT_BITS``,
``DESIGN_mps_support.md`` §2.3b). Two builders that run on the render device
for a deferred build move positions that pass it:

* ``refit_bvh._build_level`` keeps each node range's start and count, which
  are positions in the tree's primitive order, so they pass 2**24 once a tree
  has that many primitives;
* ``sliver_split._strip_rows`` numbers each strip from its sliver's first
  strip, a running count of strips, and names the triangle each strip cuts,
  so the one passes 2**24 once a scene has that many strips (524,288 slivers
  cut into the default 32 pieces) and the other once it has that many
  triangles.

Rounded, a range slides onto its neighbour's primitives and the tree loses
some and holds others twice; a strip slides along its sliver or tests the
wrong triangle. The CPU tests round the gathers as MPS does
(``mps_rounding.round_integer_gathers``) and ask, in MPS-friendly mode, for
exactly what the exact gathers give. The ``apple_gpu`` tests build on the
Apple GPU: the strips whenever the machine has one, and the BVH level when MPS
is the render device.
"""

import pytest
import torch

from algan.rendering.raytracing import refit_bvh as R
from algan.rendering.raytracing.sliver_split import _strip_rows
from algan.settings import SETTINGS
from tests.unit_tests.mps_rounding import FIRST_ROUNDED, round_integer_gathers

#: The GPU of a machine that has one, otherwise the CPU (as in
#: ``test_bezier_polyline_mps_repeat``).
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


@pytest.fixture
def mps_friendly(monkeypatch):
    """MPS-friendly mode, whatever ``ALGAN_MPS_FRIENDLY`` says."""
    monkeypatch.delenv("ALGAN_MPS_FRIENDLY", raising=False)
    SETTINGS.computing.set(mps_friendly=True)


def _level(offset, device):
    """One build level over six node ranges that start ``offset`` positions
    into the primitive order, with the offset taken back off what it returns.

    The ranges are contiguous from ``offset``, so past 2**24 most of their
    starts are odd and a float32 cannot hold them. Only the positions move
    with ``offset``: the boxes and the primitives in each range are the same,
    so an exact build returns the same level wherever the ranges sit.
    """
    g = torch.Generator().manual_seed(9)
    counts = torch.tensor([2, 3, 41, 7, 64, 9])
    total = int(counts.sum())
    starts = offset + torch.cumsum(counts, 0) - counts
    order = torch.zeros(offset + total, dtype=torch.long)
    order[offset:] = torch.randperm(total, generator=g)
    lo = torch.rand((total, 3), generator=g) * 10
    hi = lo + torch.rand((total, 3), generator=g) * 0.5 + 0.01
    order, starts, counts, lo, hi = (
        t.to(device) for t in (order, starts, counts, lo, hi)
    )
    kind, ref, next_start, next_count = R._build_level(
        order, starts, counts, 0, 6, (lo + hi) * 0.5, lo, hi, R.bvh_arity
    )
    return {
        "child kinds": kind.cpu(),
        "child refs": ref.cpu(),
        "next starts": next_start.cpu() - offset,
        "next counts": next_count.cpu(),
        "order": order[offset:].cpu(),
    }


def _differences(got, want):
    return [
        f"{name}: {int((got[name] != want[name]).sum())} of {want[name].numel()} differ"
        for name in want
        if got[name].shape != want[name].shape or not torch.equal(got[name], want[name])
    ]


def test_a_bvh_level_keeps_its_ranges_past_the_ceiling(mps_friendly, monkeypatch):
    """A level whose ranges start past 2**24 is the level they make at 0.

    Through the rounding, the ranges' odd starts come back as their even
    float32 neighbours, so a range takes in a primitive of the one beside it,
    and a split reads a position past the end of the order: an ``IndexError``
    here, and the same index as an ``AcceleratorError`` on the Mac runner's
    GPU (run 42356233755).
    """
    want = _level(0, "cpu")
    round_integer_gathers(monkeypatch)
    got = _level(FIRST_ROUNDED, "cpu")
    assert not _differences(got, want), _differences(got, want)


def test_a_bvh_level_on_the_apple_gpu_keeps_its_ranges(mps_friendly):
    """The same, built on the Apple GPU when MPS is the render device.

    Both levels are built on the one device, so the two differ only in the
    integer positions, never in how the device divides. Elsewhere this
    compares the CPU with itself. The level step needs the torch the MPS arm
    installs (2.13): the macOS CPU arm keeps the locked 2.7.1, whose MPS has no
    int64 ``index_add_`` (run 42382117024), and Algan renders nothing on MPS
    with that torch. The strips' plain ops run on 2.7.1's MPS too, so their
    test below asks any Apple machine.
    """
    device = "mps" if SETTINGS.computing.render_device.type == "mps" else "cpu"
    want = _level(0, device)
    got = _level(FIRST_ROUNDED, device)
    assert not _differences(got, want), _differences(got, want)


def _strips_past_the_ceiling():
    """``(n, sidx)``: 524,288 slivers of 32 strips, then three shorter ones.

    The long slivers bring the strip count to 2**24, so the short ones'
    first strips sit at 2**24, 2**24 + 3 and 2**24 + 8; every triangle index
    is odd and past 2**24.
    """
    n = torch.cat((torch.full((524_288,), 32), torch.tensor([3, 5, 7])))
    sidx = FIRST_ROUNDED + 2 * torch.arange(n.numel())
    return n, sidx


def _strip_differences(owner, k, parent, n, sidx):
    """What ``_strip_rows`` got wrong, checked without a second copy of it."""
    wrong = []
    long_k = k[: 524_288 * 32].view(-1, 32)
    if not torch.equal(long_k, torch.arange(32).expand_as(long_k)):
        wrong.append("a long sliver's strips are misnumbered")
    long_parent = parent[: 524_288 * 32].view(-1, 32)
    if not torch.equal(long_parent, sidx[:524_288, None].expand_as(long_parent)):
        wrong.append("a long sliver's strips name the wrong triangle")
    tail = torch.cat([torch.arange(int(c)) for c in n[-3:]])
    if not torch.equal(k[-15:], tail):
        wrong.append(f"the short slivers' strips are numbered {k[-15:].tolist()}")
    tail_parent = torch.repeat_interleave(sidx[-3:], n[-3:])
    if not torch.equal(parent[-15:], tail_parent):
        wrong.append(f"the short slivers' strips name {parent[-15:].tolist()}")
    if not torch.equal(
        owner[-15:], torch.repeat_interleave(torch.arange(524_288, 524_291), n[-3:])
    ):
        wrong.append(f"the short slivers' strips are owned by {owner[-15:].tolist()}")
    return wrong


def test_sliver_strips_keep_their_place_past_the_ceiling(mps_friendly, monkeypatch):
    """Every strip past 2**24 is numbered from its own sliver's first strip
    and names its own triangle.

    Through the rounding, the second short sliver's first strip, 2**24 + 3,
    comes back as 2**24 + 4, so its strips are numbered from -1, and every
    strip names the even neighbour of its triangle.
    """
    n, sidx = _strips_past_the_ceiling()
    round_integer_gathers(monkeypatch)
    owner, k, parent = _strip_rows(n, sidx)
    assert k.shape == (524_288 * 32 + 15,)
    wrong = _strip_differences(owner, k, parent, n, sidx)
    assert not wrong, wrong


def test_sliver_strips_on_the_apple_gpu_keep_their_place(mps_friendly):
    """The same strips numbered on the Apple GPU whenever the machine has one."""
    n, sidx = _strips_past_the_ceiling()
    owner, k, parent = _strip_rows(n.to(DEVICE), sidx.to(DEVICE))
    wrong = _strip_differences(owner.cpu(), k.cpu(), parent.cpu(), n, sidx)
    assert not wrong, wrong
