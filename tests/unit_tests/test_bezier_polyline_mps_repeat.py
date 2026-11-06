"""The Bezier polylines, through MPS's ``repeat_interleave``.

Sampling a batch of circuits numbers every polyline vertex, and two steps
spread a per-run integer over the run with ``repeat_interleave(values,
counts)``: the parameters ``k / n`` take each segment's first vertex
(``primitives._packed_uniform_cubic_parameters`` and
``bezier_circuit_primitive.batch_arange``), and the geometry cache moves each
circuit's edges to its slots in the whole table (``bezier_geometry_cache``).
torch computes that call as ``values.index_select(0,
repeat_interleave(counts))``, and on an Apple GPU ``index_select`` hands an
integer back as ``float32(value)``, exact only through 2**24
(``mps_compat._MPS_EXACT_INT_BITS``). A first vertex or a slot passes 2**24
once a batch has that many vertices or edges.

The ordinary MPS route samples on the CPU, since projection moves to the
render device only on CUDA (``settings.project_on_gpu_active``), so most of
these tests round the CPU's ``repeat_interleave`` as MPS's does
(``test_bezier_header_mps_gather._round_integer_repeats``) and, in
MPS-friendly mode, ask for exactly what the exact call gives. The
``apple_gpu`` test samples on the Apple GPU whenever the machine has one.
"""

import pytest
import torch

from algan.rendering.primitives.bezier_circuit_primitive import batch_arange
from algan.rendering.raytracing.bezier_geometry_cache import (
    _build_cached_circuit_edges,
    _build_frame_local_circuit_edges,
)
from algan.rendering.raytracing.primitives import _packed_uniform_cubic_parameters
from algan.settings import SETTINGS
from tests.unit_tests.mps_rounding import FIRST_ROUNDED
from tests.unit_tests.test_bezier_header_mps_gather import _round_integer_repeats


@pytest.fixture
def mps_friendly(monkeypatch):
    """MPS-friendly mode, whatever ``ALGAN_MPS_FRIENDLY`` says."""
    monkeypatch.delenv("ALGAN_MPS_FRIENDLY", raising=False)
    SETTINGS.computing.set(mps_friendly=True)


def _segments_past_the_ceiling():
    """2.2 million segments' chord counts and vertex counts.

    1 to 16 chords each, and one segment in twenty closing an open sub-path
    with an extra vertex: 18,800,000-odd vertices, so the first vertices of
    the last quarter million segments pass 2**24.
    """
    generator = torch.Generator().manual_seed(4)
    chords = torch.randint(1, 17, (2_200_000,), generator=generator)
    closing = torch.rand(chords.shape, generator=generator) < 0.05
    return chords, chords + closing.long()


def _moved(got, want, what):
    return f"{int((got != want).sum())} of {want.numel()} {what}"


def test_the_vertex_parameters_survive_the_mps_repeat_interleave(
    mps_friendly, monkeypatch
):
    """Every vertex keeps its ``k / n``, with the total known or not.

    Through the rounding, a segment whose first vertex passes 2**24 starts
    one or two vertices early or late, which slides its vertices along the
    cubic and past its ends: 1,020,086 of the 18,816,186 parameters moved,
    62,493 of them off the cubic.
    """
    chords, vertices = _segments_past_the_ceiling()
    total = int(vertices.sum())
    assert total > FIRST_ROUNDED, "the vertices should pass 2**24"
    want = (
        _packed_uniform_cubic_parameters(chords, torch.float32, vertices, total=total),
        _packed_uniform_cubic_parameters(chords, torch.float32, vertices),
    )
    _round_integer_repeats(monkeypatch)
    got = (
        _packed_uniform_cubic_parameters(chords, torch.float32, vertices, total=total),
        _packed_uniform_cubic_parameters(chords, torch.float32, vertices),
    )
    for spelling, g, w in zip(("with", "without"), got, want):
        outside = int(((g < 0) | (g > 1)).sum())
        assert torch.equal(g, w), (
            f"{_moved(g, w, 'vertex parameters')} moved {spelling} the total,"
            f" {outside} of them off the cubic"
        )


def test_the_vertex_parameters_on_the_apple_gpu_are_the_cpu_ones(mps_friendly):
    """Each vertex's ``k``, numbered on an Apple GPU, is the CPU's.

    On ``torch.backends.mps``'s device whenever the machine has one, as in
    ``test_bezier_header_mps_gather``. ``k`` is read back out of ``k / n``
    rather than compared as a float, so only the numbering is asked to match,
    not the device's division.
    """
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    chords, vertices = _segments_past_the_ceiling()
    total = int(vertices.sum())
    want = batch_arange(vertices)
    got = batch_arange(vertices.to(device)).cpu()
    assert torch.equal(got, want), _moved(got, want, f"vertex numbers on {device}")
    n = torch.repeat_interleave(chords, vertices).float()
    t = _packed_uniform_cubic_parameters(
        chords.to(device), torch.float32, vertices.to(device), total=total
    ).cpu()
    got = (t * n).round().long()
    assert torch.equal(got, want), _moved(got, want, f"vertex k on {device}")


def _tagged_build(corners, samples, segments, next_inds, centers, bu, bv, inward):
    """A stand-in for ``_sample_circuit_edges`` over one-segment circuits.

    Each circuit gets ``samples`` edges, every one a row of the tag in its
    first control point, so a slot holds its circuit's tag or reads wrong.
    """
    counts = samples.long()
    tags = torch.repeat_interleave(corners[0, :, 0, 0], counts)
    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    return tags.view(1, -1, 1).expand(-1, -1, 6), offsets.to(torch.int32)


def _circuits_after_a_big_one(first_counts, small=4096):
    """Arguments for the circuit-edge builders, over one-segment circuits.

    The first circuit has ``first_counts`` edges (one entry per frame) and
    changes shape between the two frames; the ``small`` circuits after it,
    1 to 4 edges each, are still. Each circuit's tag is its index.
    """
    generator = torch.Generator().manual_seed(6)
    counts = torch.randint(1, 5, (small,), generator=generator)
    num = small + 1
    samples = torch.cat(
        (
            torch.tensor(first_counts).view(-1, 1),
            counts.view(1, -1).expand(len(first_counts), -1),
        ),
        1,
    )
    corners = torch.zeros((2, num, 4, 3))
    corners[:, :, 0, 0] = torch.arange(num).float()
    corners[:, :, 1, 1] = 1.0
    corners[1, 0, 1, 1] = 2.0
    plane = torch.zeros((2, num, 3))
    basis_u, basis_v = plane.clone(), plane.clone()
    basis_u[..., 0] = 1.0
    basis_v[..., 1] = 1.0
    return (
        corners,
        samples if len(first_counts) > 1 else samples[0],
        torch.ones(num, dtype=torch.long),
        torch.arange(num).expand(2, -1),
        plane,
        basis_u,
        basis_v,
    )


def test_the_cache_places_every_edge_past_the_ceiling(mps_friendly, monkeypatch):
    """The still circuits' edges land in their own slots.

    The still circuits are built apart from the moving one in front of them,
    so each is moved by that circuit's 2**24 + 1 edges, which the rounding
    makes 2**24: every still edge one slot early, over its neighbour's, and
    the table's last slot never written. 4096 slots read another circuit's
    tag, one at each circuit's end.
    """
    args = _circuits_after_a_big_one([FIRST_ROUNDED])
    want, want_offsets = _build_cached_circuit_edges(None, _tagged_build, args, False)
    assert want.shape[1] > FIRST_ROUNDED
    _round_integer_repeats(monkeypatch)
    got, got_offsets = _build_cached_circuit_edges(None, _tagged_build, args, False)
    assert torch.equal(got_offsets, want_offsets)
    assert torch.equal(got, want), _moved(got[..., 0], want[..., 0], "edge slots")


def test_frame_local_edges_keep_their_slots_past_the_ceiling(mps_friendly, monkeypatch):
    """Each frame's edges land in its own circuit's slots.

    The first circuit has 2**24 + 2 edges in frame 0 and one in frame 1, so in
    frame 1 every later circuit is moved past 2**24 + 1 slots of padding,
    which the rounding makes 2**24, and 4096 slots read wrong.
    """
    args = _circuits_after_a_big_one([FIRST_ROUNDED + 1, 1])
    want, want_offsets = _build_frame_local_circuit_edges(
        None, _tagged_build, args, False, cache=False
    )
    assert want.shape[1] > FIRST_ROUNDED
    _round_integer_repeats(monkeypatch)
    got, got_offsets = _build_frame_local_circuit_edges(
        None, _tagged_build, args, False, cache=False
    )
    assert torch.equal(got_offsets, want_offsets)
    assert torch.equal(got, want), _moved(got[..., 0], want[..., 0], "edge slots")
