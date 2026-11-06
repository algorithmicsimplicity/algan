"""The Bezier edge tables' header, through MPS's integer gather.

``build_bezier_edge_acceleration`` gives every (frame, circuit) group a
header -- its edge range, bounds, inverse bin sizes and bin offsets -- and
writes all of them with one gather of a compact int32 table. On an Apple GPU
``index_select`` hands an int32 back as ``float32(value)``, exact only through
2**24 (``mps_compat._MPS_EXACT_INT_BITS``), and almost no header word fits in
24 bits. The bounds and inverse bin sizes are float32 bit patterns, past 2**24
for every float of magnitude between 2**-125 and 2**127, and the bin offsets
pass it once the buffer does. ``mps_compat.gather_exact`` is the gather that
carries them.

The ordinary MPS route builds these tables on the CPU, since the merge moves
to the render device only on CUDA (``settings.merge_on_gpu_active``), and
uploads them byte for byte. So the first three tests reproduce the defect the
way the rest of this family does: ``mps_rounding.round_integer_gathers`` makes
the CPU's gathers round as MPS's do, and in MPS-friendly mode the tables must
come back exactly as the exact gathers build them. The last test builds them
on the Apple GPU whenever the machine has one.
"""

# No ``from __future__ import annotations``: it would turn the probe kernel's
# ndarray annotations into strings, which Taichi cannot compile.
import pytest
import torch

from algan.rendering.raytracing import bezier_acceleration as ba
from algan.rendering.raytracing.bezier_acceleration import (
    BEZIER_ACCEL_HEADER_SIZE,
    BEZIER_MIN_U,
    BEZIER_SCAN_INV_V,
    BEZIER_SCAN_OFFSET_BASE,
    bezier_scan_bins,
    build_bezier_edge_acceleration,
)
from algan.settings import SETTINGS
from tests.unit_tests.mps_rounding import FIRST_ROUNDED, round_integer_gathers
from tests.unit_tests.test_bezier_acceleration_dedup import _batch


@pytest.fixture
def mps_friendly(monkeypatch):
    """MPS-friendly mode, whatever ``ALGAN_MPS_FRIENDLY`` says."""
    monkeypatch.delenv("ALGAN_MPS_FRIENDLY", raising=False)
    SETTINGS.computing.set(mps_friendly=True)


def _headers(buffer, groups):
    return buffer[: groups * BEZIER_ACCEL_HEADER_SIZE].view(
        groups, BEZIER_ACCEL_HEADER_SIZE
    )


def _exact_then_rounded(monkeypatch, edges, offsets):
    """The buffer built with exact gathers, then with gathers that round."""
    want = build_bezier_edge_acceleration(edges, offsets)
    round_integer_gathers(monkeypatch)
    return want, build_bezier_edge_acceleration(edges, offsets)


def _moved(got, want, groups):
    return int((_headers(got, groups) != _headers(want, groups)).sum())


def _page_of_glyphs(frames=100, circuits=1850, seed=0):
    """``circuits`` small triangles over ``frames`` frames, the last one moving.

    185,000 (frame, circuit) groups at the defaults, so 16,835,000 header
    words: past 2**24 on their own, which puts every bin offset past it. A
    batch of 1850 glyphs reaches that in 100 frames when any one of them
    moves, because a moving circuit keeps the batch's frames in ``edges_2d``.

    Every coordinate is a multiple of 2**-20 and every triangle spans exactly
    1/4 each way, so each inverse bin size is a power of two whatever the
    division rounds like. The bounds keep their low bits, which is what the
    gather can lose. No edge is horizontal, so each triangle files an odd
    number of scanline entries and the offsets do not stay even, which a
    float32 would carry exactly up to 2**25.
    """
    generator = torch.Generator().manual_seed(seed)
    corner = torch.randint(-(2**19), 2**19, (circuits, 1, 2), generator=generator)
    legs = torch.tensor([[0, 0], [2**18, 2**17], [2**17, 2**18]])
    corners = (corner + legs).float() / 2**20
    ends = corners.roll(-1, dims=1)
    rows = torch.cat((corners, ends, torch.ones(circuits, 3, 2)), -1)
    edges = rows.reshape(1, 3 * circuits, 6).repeat(frames, 1, 1)
    step = torch.arange(frames, dtype=torch.float32).view(-1, 1, 1) / 2**10
    edges[:, -3:, :4] += step
    offsets = torch.arange(0, 3 * circuits + 1, 3, dtype=torch.int32)
    return edges, offsets


@pytest.mark.parametrize(
    "every_group", [False, True], ids=["shared_bins", "every_group_built"]
)
@pytest.mark.parametrize("seed", range(4))
def test_the_bounds_and_bin_sizes_survive_the_mps_integer_gather(
    mps_friendly, monkeypatch, seed, every_group
):
    """Every group's bounds and inverse bin sizes come through bit for bit.

    They are float32 bit patterns, so past 2**24 in any scene, and the
    rounding gather moves them by up to 64 ulps: 42 to 1102 header words of
    these batches, and 99 of the 1365 of a frame of text, a circle, an arc and
    a stroke. That frame kept every pixel, but a bound or a scale also decides
    which scanline bin a query reads, and
    :func:`test_a_point_inside_a_circuit_reads_inside_through_the_mps_integer_gather`
    is where that goes wrong. ``ALGAN_OPT_DISABLE=acceldedup`` builds every
    group and still gathers its rows, through the identity, so both arms are
    held.
    """
    monkeypatch.setattr(ba, "_dedup_disabled", lambda: every_group)
    edges, offsets = _batch(seed)
    groups = edges.shape[0] * (offsets.numel() - 1)
    want, got = _exact_then_rounded(monkeypatch, edges, offsets)
    floats = _headers(want, groups)[:, BEZIER_MIN_U : BEZIER_SCAN_INV_V + 1]
    assert bool((floats.long().abs() >= FIRST_ROUNDED).any()), (
        "no header word is past the ceiling, so this proves nothing"
    )
    assert torch.equal(got, want), (
        f"{_moved(got, want, groups)} header words moved through the gather"
    )


def test_bin_offsets_past_the_ceiling_survive_the_mps_integer_gather(
    mps_friendly, monkeypatch
):
    """A header that passes 2**24 words keeps every bin's edges.

    All 15,170,000 bin offsets of :func:`_page_of_glyphs` are past the
    ceiling, and the rounding gather moved 7,677,450 of them by one entry of
    the edge lists. A bin then started or ended on its neighbour's edges: 1.6
    million of the 3 million scanline bins and 9 million of the 12 million grid
    cells held a different number of edges. Queried at its own centroid with
    those headers, 72,725 of the 185,000 triangles crossed an even number of
    edges and read as outside.
    """
    edges, offsets = _page_of_glyphs()
    groups = edges.shape[0] * (offsets.numel() - 1)
    want, got = _exact_then_rounded(monkeypatch, edges, offsets)
    bins = _headers(want, groups)[:, BEZIER_SCAN_OFFSET_BASE:]
    assert bool((bins >= FIRST_ROUNDED).all()), "the offsets should pass 2**24"
    assert torch.equal(got, want), (
        f"{_moved(got, want, groups)} header words moved through the gather"
    )


def _wall_split_at_every_bin():
    """A rectangle whose right wall has two vertices at every bin boundary.

    The rectangle is ``[0, 1] x [a, b]``. The builder files an edge in every
    scanline bin from its lower end's to its upper end's, by ``floor((y -
    min_v) * scan_inv_v)``, and a query reads the one bin its own ``v`` maps
    to through the header's ``min_v`` and ``scan_inv_v``. The wall's vertices
    are the last float of each bin and the first of the next, so if a rounded
    bound or scale moves a boundary either way, the queries between where it
    was and where the header says it is read a bin that does not hold the wall
    edge they cross. ``a`` and ``b`` give a bound and a scale whose low bits
    the rounding moves.

    Returns the ``[1, E, 6]`` edges, the ``[2]`` offsets of the one circuit,
    and the first float of each bin above the first.
    """
    a = torch.tensor(-0.399562389, dtype=torch.float32)
    b = torch.tensor(0.3, dtype=torch.float32)
    # The builder's scan_inv_v, bit for bit: the extent is b - a.
    scale = bezier_scan_bins / (b - a)
    firsts = []
    for k in range(1, bezier_scan_bins):
        v = a + k / scale
        while torch.floor((v - a) * scale) >= k:
            v = torch.nextafter(v, a)
        while torch.floor((v - a) * scale) < k:
            v = torch.nextafter(v, b)
        firsts.append(v)
    wall = [a]
    for v in firsts:
        wall += [torch.nextafter(v, a), v]
    wall = torch.stack([*wall, b])
    rows = [(0.0, a, 1.0, a)]
    rows += [(1.0, y0, 1.0, y1) for y0, y1 in zip(wall[:-1], wall[1:])]
    rows += [(1.0, b, 0.0, b), (0.0, b, 0.0, a)]
    edges = torch.tensor([[(*(float(x) for x in r), 1.0, 1.0) for r in rows]])
    return edges, torch.tensor([0, len(rows)], dtype=torch.int32), torch.stack(firsts)


_crossings_probe = None


def _crossings(edges, accel, v):
    """Each query ``(0.5, v)``'s scanline crossings, as the kernels count them."""
    global _crossings_probe
    from algan.rendering.raytracing import raytrace_kernels_taichi as k
    from algan.rendering.taichi_runtime import init_taichi
    from algan.settings._startup import render_device
    from algan.taichi_compat import ti

    # Algan's own init, never a bare ``ti.init`` (test_watertight_triangle.py).
    init_taichi()
    if _crossings_probe is None:

        @ti.kernel
        def probe(
            edges_2d: ti.types.ndarray(dtype=ti.f32, ndim=3),
            edge_accel: ti.types.ndarray(dtype=ti.i32, ndim=1),
            queries: ti.types.ndarray(dtype=ti.f32, ndim=1),
            result: ti.types.ndarray(dtype=ti.i32, ndim=1),
        ):
            for i in range(queries.shape[0]):
                (cr, _md, _cu, _cv, _e1x, _e1y, _s1, _sd, _su, _sv, _e2x, _e2y, _s2) = (
                    k._bezier_point_metrics(
                        0, 0, 0.5, queries[i], 0.0, 1, edges_2d, edge_accel
                    )
                )
                result[i] = cr

        _crossings_probe = probe
    device = render_device()
    out = torch.zeros(v.shape, dtype=torch.int32, device=device)
    _crossings_probe(
        edges.to(device).contiguous(),
        accel.to(device).contiguous(),
        v.to(device).contiguous(),
        out,
    )
    return out.cpu()


def test_a_point_inside_a_circuit_reads_inside_through_the_mps_integer_gather(
    mps_friendly, monkeypatch
):
    """The header's rounded bounds and scale must not move a query's bin.

    Queries sit on the inside of the rectangle of :func:`_wall_split_at_every_bin`,
    from 4 to 128 floats either side of each bin boundary, half a unit from
    the right wall, so each crosses exactly one wall edge. With the rounded
    header 1164 of those 3750 points crossed no edge at all: the inside read
    as outside on a strip at bin boundaries, half a unit from any edge, so a
    filled circuit would drop those pixels whole rather than shade its rim a
    little differently.
    """
    edges, offsets, firsts = _wall_split_at_every_bin()
    steps = torch.cat((torch.arange(-128, -3), torch.arange(4, 129)))
    bits = firsts.view(torch.int32).view(-1, 1) + steps.view(1, -1).to(torch.int32)
    v = bits.reshape(-1).view(torch.float32)
    want, got = _exact_then_rounded(monkeypatch, edges, offsets)
    inside = _crossings(edges, want, v) % 2 == 1
    assert bool(inside.all()), "the exact header itself misfiles a query"
    outside = int((_crossings(edges, got, v) % 2 == 0).sum())
    assert outside == 0, f"{outside} of {v.numel()} inside points read outside"


@pytest.mark.parametrize("case", ["few_frames", "past_the_ceiling"])
def test_the_header_built_on_the_apple_gpu_is_the_cpu_header(mps_friendly, case):
    """Bit for bit, the tables an Apple GPU builds are the CPU's.

    Built from :func:`_page_of_glyphs`, whose arithmetic is exact on any
    device -- extremes, differences of multiples of 2**-20, products and
    quotients by powers of two -- so a word that differs is the gather's.
    ``few_frames`` moves every circuit, so every group after frame 0 is built
    from its own edges, and its header is short of the ceiling;
    ``past_the_ceiling`` is the 185,000-group batch.

    **It only bites on a Mac**, like the gather guards in
    ``test_mps_friendly.py``: it runs on ``torch.backends.mps``'s device rather
    than Algan's configured one, so any Apple machine checks the real gather,
    and everywhere else it compares the CPU with itself.
    """
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if case == "few_frames":
        edges, offsets = _page_of_glyphs(frames=6, circuits=40, seed=1)
        edges[1:, :, :4] += torch.arange(1, 6).view(-1, 1, 1).float() / 2**10
    else:
        edges, offsets = _page_of_glyphs()
    groups = edges.shape[0] * (offsets.numel() - 1)
    want = build_bezier_edge_acceleration(edges, offsets)
    got = build_bezier_edge_acceleration(edges.to(device), offsets.to(device))
    got = got.cpu()
    assert got.shape == want.shape
    assert torch.equal(got, want), (
        f"{_moved(got, want, groups)} of {groups * BEZIER_ACCEL_HEADER_SIZE} "
        f"header words built on {device} differ from the CPU's"
    )
