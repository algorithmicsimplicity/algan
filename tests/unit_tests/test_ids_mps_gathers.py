"""The renderer's circuit, member and surface ids, past the 2**24 an MPS gather
is exact to.

On an Apple GPU ``index_select`` hands an int32 or int64 back as
``float32(value)``, exact only through 2**24 (``mps_compat._MPS_EXACT_INT_BITS``,
``DESIGN_mps_support.md`` §2.3b), and ``repeat_interleave(values, counts)`` is
that gather inside torch: ``values.index_select(0, repeat_interleave(counts))``.
Several places numbered things as ``repeat_interleave(arange(n), counts)``:
each Bezier segment's, edge's and table slot's circuit, and each triangle's
or logical-PN patch's member. An id passes 2**24 once there are that many
circuits or members, and rounded, an odd id past it becomes an even
neighbour's. The counts alone expand to the same ids, and exactly. Each
Bezier vertex's circuit spreads the segments' ids further, and
``repeat_interleave_exact`` now does that. Two more places gathered an id
with ``index_select``: the depth gate's surface ids
(``sheets._sample_depth_lose_reference``) and each promoted color group's
first triangle (``scene_builder._split_promotable``), which ``gather_exact``
now carries.

Only the depth gate runs on the render device. The rest run where projection
and the scene merge do, which on an MPS render is the CPU
(``settings.project_on_gpu_active`` and ``merge_on_gpu_active`` are CUDA-only),
so MPS reaches them only through the unsupported ``ALGAN_ANIMATION_DEVICE=mps``
or a direct call. As in ``test_bezier_polyline_mps_repeat``, these tests round
the CPU's gathers as MPS's do (``mps_rounding.round_integer_gathers``, and
``test_bezier_header_mps_gather._round_integer_repeats`` for the gather inside
``repeat_interleave``) and ask, in MPS-friendly mode, for exactly what the
exact expansions give.

Each test builds its case twice: once alone, and once behind 2**24 empty
circuits or members (or with its ids moved up by 2**24), which makes the ids
it reads 2**24 and 2**24 + 1, one float32 for both. Nothing about the case
depends on what is in front of it, so through the rounding the second must
still come out as the first.
"""

from types import SimpleNamespace

import pytest
import torch

from algan.constants.color import BLUE
from algan.rendering.raytracing import bezier_acceleration as ba
from algan.rendering.raytracing import settings as rt_settings
from algan.rendering.raytracing import sheets
from algan.rendering.raytracing.bezier_geometry_cache import (
    _build_cached_circuit_edges,
    _build_frame_local_circuit_edges,
)
from algan.rendering.raytracing.primitives import (
    LogicalPNTrianglePrimitive,
    RayTracedBezierCircuitPrimitive,
    RayTracedTrianglePrimitive,
)
from algan.rendering.raytracing.raytrace_kernels_taichi import _EXTRA_W
from algan.rendering.raytracing.scene_builder import _split_promotable
from algan.settings import SETTINGS
from tests.unit_tests.mps_rounding import FIRST_ROUNDED, round_integer_gathers
from tests.unit_tests.test_bezier_header_mps_gather import _round_integer_repeats

#: The empty circuits or members in front of a case's two, which are then
#: 2**24, a float32 holds, and 2**24 + 1, which it rounds to 2**24.
EMPTY = FIRST_ROUNDED - 1

_STATIC_CIRCUITS = ba._static_circuits


@pytest.fixture
def mps_friendly(monkeypatch):
    """MPS-friendly mode, whatever ``ALGAN_MPS_FRIENDLY`` says."""
    monkeypatch.delenv("ALGAN_MPS_FRIENDLY", raising=False)
    SETTINGS.computing.set(mps_friendly=True)


def _round_like_mps(monkeypatch):
    """Round every integer the CPU gathers, inside ``repeat_interleave`` too."""
    _round_integer_repeats(monkeypatch)
    round_integer_gathers(monkeypatch)


def _counts(empty, *last):
    """One count per circuit or member: ``empty`` zeros, then ``last``."""
    counts = torch.zeros(empty + len(last), dtype=torch.long)
    counts[empty:] = torch.tensor(last)
    return counts


def _differ(got, want, what):
    if got.shape != want.shape:
        return f"{what}: shape {tuple(got.shape)} against {tuple(want.shape)}"
    return f"{what}: {int((got != want).sum())} of {want.numel()} differ"


def _depth_gate(first_sid):
    """The subjects' lose words in four pixels, each a far subject sheet over
    a near, full, opaque enforcer.

    In pixels 0 and 2 the two sheets are different surfaces, so the subject
    cedes every sample; in 1 and 3 they are one surface, so it cedes none. The
    surface ids are ``first_sid`` plus 0 and 1, 1 and 1, 3 and 4, 3 and 3:
    past 2**24, two neighbours one float32 cannot tell apart, or one id it
    cannot hold.
    """
    sid = first_sid + torch.tensor([0, 1, 1, 1, 3, 4, 3, 3])
    subject = torch.tensor([True, False] * 4)
    depths = torch.where(subject, 10.0, 5.0)
    lose = sheets._sample_depth_lose_reference(
        torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]),
        depths.view(-1, 1).expand(-1, sheets.AA_NUM_SAMPLES).contiguous(),
        sid,
        ~subject,
        subject,
        torch.full((8,), sheets.AA_MASK_ALL, dtype=torch.int32),
    )
    return (lose >> sheets.AA_LOSE_SHIFT)[subject]


def test_the_depth_gate_tells_surfaces_apart_past_the_ceiling(
    mps_friendly, monkeypatch
):
    """A subject cedes its samples to a nearer surface, never to its own.

    ``sheet_sample_depth``'s reference arm, on the render device, which runs
    when ``ALGAN_SHEET_DEPTH_REDUCE_KERNEL=0`` and when a batch is too large
    for the kernel. Its sample keys are
    ``test_sheet_chain_kernels.test_the_reference_depth_gate_keeps_its_samples_apart_in_a_4k_frame``'s;
    this holds its surface ids, which pass 2**24 once a scene has that many
    surfaces. Through the rounding gather three of the four subjects got it
    backwards: pixel 0's kept the samples a nearer surface covers, and pixels
    1 and 3 ceded every sample to their own surface.
    """
    full = sheets.AA_MASK_ALL
    want = _depth_gate(0)
    assert want.tolist() == [full, 0, full, 0], "the case should cede in 0 and 2"
    round_integer_gathers(monkeypatch)
    got = _depth_gate(EMPTY)
    assert torch.equal(got, want), (
        f"lose words {got.tolist()} with the ids past 2**24, against {want.tolist()}"
    )


def _promoted_groups(empty):
    """What ``_split_promotable`` makes of two promoted triangles after
    ``empty`` glowing ones: the promoted triangles (counted from the first of
    the two), their meta rows, and the maps appended for them.

    Every triangle has one color. Glowing keeps a triangle per vertex; the
    last two do not glow, and their roughness differs, so each is the first
    and only triangle of its group.
    """
    n = empty + 2
    extra = torch.zeros((1, n, _EXTRA_W))
    extra[0, :empty, 9:12] = 1.0
    extra[0, -2, [1, 3, 5]] = 0.25
    extra[0, -1, [1, 3, 5]] = 0.75
    primitive = SimpleNamespace(
        _rt_tri_colors=torch.full((1, 1, 3, 5), 0.5).expand(-1, n, -1, -1),
        _rt_tri_extra=extra,
    )
    maps = []

    def append(texels, is_color=False):
        maps.append(texels)
        return (len(maps), 1, 1, texels.shape[0], -1)

    _, promoted, meta = _split_promotable(primitive, append, torch.device("cpu"), {})
    return promoted - empty, meta, torch.cat(maps)


def test_each_promoted_group_takes_its_own_triangle_past_the_ceiling(
    mps_friendly, monkeypatch
):
    """Each promoted group's color and material maps are its own triangle's.

    The merge's, so on the CPU on a Mac. Through the rounding gather triangle
    2**24 + 1's group was given triangle 2**24's maps, and would render with
    roughness 0.25 instead of 0.75.
    """
    want = _promoted_groups(0)
    assert want[0].tolist() == [0, 1], "both triangles should be promoted"
    round_integer_gathers(monkeypatch)
    got = _promoted_groups(EMPTY)
    for name, g, w in zip(("promoted triangles", "meta rows", "maps"), got, want):
        assert torch.equal(g, w), _differ(g, w, name)


class _Numbered(Exception):
    """Raised to stop a build once it has numbered what a test reads."""


def _edge_numbering(monkeypatch, empty):
    """Which circuit ``build_bezier_edge_acceleration`` files each edge under
    (counted from the first non-empty one), and which circuits it finds still.

    Two triangles after ``empty`` empty circuits, over two frames, the second
    triangle moving. Both answers are read where the builder hands them to
    ``_static_circuits``, and the build stops there: past it every circuit
    gets a 91-word header, which for 2**24 of them is 6 GB that nothing here
    reads.
    """
    seen = {}

    def static_circuits(edges_2d, circuit_of_edge, num_circuits):
        seen["circuits"] = circuit_of_edge - empty
        seen["still"] = _STATIC_CIRCUITS(edges_2d, circuit_of_edge, num_circuits)
        raise _Numbered

    monkeypatch.setattr(ba, "_static_circuits", static_circuits)
    triangle = torch.tensor(
        [[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0]]
    )
    edges = torch.cat((triangle, triangle + 2.0)).repeat(2, 1, 1)
    edges[1, 3:] += 0.5
    edges = torch.cat((edges, torch.ones(2, 6, 2)), -1)
    starts = _counts(empty, 3, 3).cumsum(0)
    offsets = torch.cat((starts.new_zeros(1), starts)).to(torch.int32)
    with pytest.raises(_Numbered):
        ba.build_bezier_edge_acceleration(edges, offsets)
    return seen["circuits"], seen["still"][empty:]


def test_the_edge_tables_file_each_edge_under_its_circuit_past_the_ceiling(
    mps_friendly, monkeypatch
):
    """Each edge is filed under its own circuit, and only the moving one moves.

    The merge's, so on the CPU on a Mac. Through the rounding expansion all
    six edges were filed under circuit 2**24: circuit 2**24 + 1's header would
    list none of them, so it would draw nothing, and the moving circuit
    counted as still while the still one counted as moving.
    """
    want = _edge_numbering(monkeypatch, 0)
    assert want[0].tolist() == [0, 0, 0, 1, 1, 1]
    assert want[1].tolist() == [True, False]
    _round_like_mps(monkeypatch)
    got = _edge_numbering(monkeypatch, EMPTY)
    for name, g, w in zip(("edge circuits", "still circuits"), got, want):
        assert torch.equal(g, w), _differ(g, w, name)


def _tag_build(corners, samples, segments, next_inds, centers, bu, bv, inward):
    """A stand-in for ``_sample_circuit_edges``: ``samples`` edges a segment,
    every one a row of the tag in the segment's first control point, frame by
    frame.
    """
    counts = samples.long()
    tags = torch.repeat_interleave(corners[:, :, 0, 0], counts, dim=1)
    edges = torch.zeros_like(segments).index_add_(
        0, torch.repeat_interleave(segments), counts
    )
    offsets = torch.cat((segments.new_zeros(1), edges.cumsum(0)))
    return tags.unsqueeze(-1).expand(-1, -1, 6), offsets.to(torch.int32)


def _circuit_args(empty, samples, corners):
    """The circuit-edge builders' arguments for two one-segment circuits after
    ``empty`` empty ones, over the two frames of ``corners``.

    Every circuit has the same plane, which as a view costs nothing for the
    empty ones.
    """
    plane = torch.zeros((1, 1, 3)).expand(-1, empty + 2, -1)
    unit = torch.eye(3)
    return (
        corners,
        torch.tensor(samples),
        _counts(empty, 1, 1),
        torch.arange(2).expand(2, -1),
        plane,
        unit[0].expand_as(plane),
        unit[1].expand_as(plane),
    )


def _two_circuits(empty, samples, tags):
    """``_circuit_args`` for two segments tagged ``tags`` in each frame,
    ``[2, 2]``; a segment whose tag changes between the frames is moving.
    """
    corners = torch.zeros((2, 2, 4, 3))
    corners[:, :, 0, 0] = torch.tensor(tags)
    corners[:, :, 1, 1] = 1.0
    return _circuit_args(empty, samples, corners)


def test_the_cache_keeps_a_moving_circuit_moving_past_the_ceiling(
    mps_friendly, monkeypatch
):
    """The still circuit is built once and the moving one in every frame.

    On the CPU on a Mac, where projection runs. Through the rounding
    expansion the moving circuit's segment counted towards the still one's
    stillness: the moving circuit went to the still part without its segment,
    and the build raised.
    """
    case = ([2, 3], [[1.0, 2.0], [1.0, 3.0]])
    want, want_offsets = _build_cached_circuit_edges(
        None, _tag_build, _two_circuits(0, *case), False
    )
    assert want[..., 0].tolist() == [[1.0] * 2 + [2.0] * 3, [1.0] * 2 + [3.0] * 3]
    _round_like_mps(monkeypatch)
    got, got_offsets = _build_cached_circuit_edges(
        None, _tag_build, _two_circuits(EMPTY, *case), False
    )
    assert torch.equal(got_offsets[EMPTY:], want_offsets), "the edge ranges moved"
    assert torch.equal(got, want), _differ(got[..., 0], want[..., 0], "edge slots")


def test_frame_local_edges_pad_each_circuit_with_its_own_past_the_ceiling(
    mps_friendly, monkeypatch
):
    """A circuit's spare slots hold its own inert edge.

    On the CPU on a Mac, where projection runs. Circuit 2**24 + 1 has three
    edges in frame 0 and one in frame 1, so in frame 1 two of its slots are
    padding. Through the rounding expansions those two held circuit 2**24's
    padding, a point on the other circuit, which the row scans that walk a
    circuit's every edge would meet.
    """
    case = ([[2, 3], [2, 1]], [[1.0, 2.0], [1.0, 3.0]])
    want, want_offsets = _build_frame_local_circuit_edges(
        None, _tag_build, _two_circuits(0, *case), False, cache=False
    )
    assert want[1, 3:, :4].tolist() == [[3.0] * 4] * 2
    _round_like_mps(monkeypatch)
    got, got_offsets = _build_frame_local_circuit_edges(
        None, _tag_build, _two_circuits(EMPTY, *case), False, cache=False
    )
    assert torch.equal(got_offsets[EMPTY:], want_offsets), "the edge ranges moved"
    assert torch.equal(got, want), _differ(got[..., 0], want[..., 0], "edge slots")


def _closing_plan(empty):
    """How many frames each of ``_build_frame_local_circuit_edges``'s calls to
    the builder holds, for two frames of two one-segment circuits after
    ``empty`` empty ones.

    Circuit 0 is a closed cubic a unit across. Circuit 1 is a thousandth of a
    unit across, and its cubic ends 1e-6 short of its start in frame 0 and
    1e-4 short in frame 1. Measured against its own size
    (``utils.CONNECTION_GAP``), both gaps are open, so both frames close the
    same vertex and one batched call builds them. Measured against the size of
    the two circuits together, the first gap would count as joined.
    """
    corners = torch.zeros((2, 2, 4, 3))
    corners[:, 0] = torch.tensor([[1.0, 0, 0], [2, 0, 0], [2, 1, 0], [1, 0, 0]])
    corners[:, 1, 1:3] = torch.tensor([[1e-3, 0, 0], [1e-3, 1e-3, 0]])
    corners[:, 1, 3, 0] = torch.tensor([1e-6, 1e-4])
    frames = []

    def build(corners, *args):
        frames.append(corners.shape[0])
        return _tag_build(corners, *args)

    _build_frame_local_circuit_edges(
        None, build, _circuit_args(empty, [[2, 3], [2, 3]], corners), False, cache=False
    )
    return frames


def test_a_still_batch_is_planned_by_each_circuits_own_gaps_past_the_ceiling(
    mps_friendly, monkeypatch
):
    """Frames that close the same vertices are built in one batched call.

    On the CPU on a Mac, where projection runs. Through the rounding
    expansion, circuit 2**24 + 1 was measured together with circuit 2**24.
    Its gap in frame 0 then counted as joined, and the batch was built frame
    by frame. The same mistake the other way round would build two frames
    that close different vertices in one call, and one of them would get a
    closing vertex that its own build does not have.
    """
    want = _closing_plan(0)
    assert want == [2], "one batched call should build both frames"
    _round_like_mps(monkeypatch)
    got = _closing_plan(EMPTY)
    assert got == want, f"built in calls of {got} frames, against {want}"


def _circuit_primitive():
    """A bare ``RayTracedBezierCircuitPrimitive`` (as ``test_coplanar_z_index``
    builds one), for the methods that only read what a test sets.
    """
    return RayTracedBezierCircuitPrimitive.__new__(RayTracedBezierCircuitPrimitive)


def _loop(center, size):
    """``[4, 3]``: a closed cubic around ``center``."""
    shape = torch.tensor([[0.0, -1, 0], [2, 1, 0], [-2, 1, 0], [0, -1, 0]])
    return torch.tensor(center) + size * shape


def _sampled_edges(empty):
    """``_sample_circuit_edges`` over two loops after ``empty`` empty
    circuits, each loop in its own plane around its own center.
    """
    centers = torch.zeros((1, empty + 2, 3))
    centers[0, -1] = torch.tensor([10.0, 0.0, 0.0])
    corners = torch.stack((_loop([0.0, 0.0, 0.0], 1.0), _loop([10.0, 0.0, 0.0], 0.5)))
    unit = torch.eye(3)
    return _circuit_primitive()._sample_circuit_edges(
        corners.unsqueeze(0),
        torch.tensor([4, 6]),
        _counts(empty, 1, 1),
        torch.arange(2).view(1, -1),
        centers,
        unit[0].expand_as(centers),
        unit[1].expand_as(centers),
        False,
    )


def test_every_vertex_is_sampled_in_its_own_circuit_past_the_ceiling(
    mps_friendly, monkeypatch
):
    """Each loop's edges are in its own plane's coordinates and its own range.

    On the CPU on a Mac, where projection runs. Through the rounding
    expansions the second loop's six vertices were placed in the first
    circuit's plane, ten units off, and counted as the first circuit's edges,
    leaving circuit 2**24 + 1 with none.
    """
    want, want_offsets = _sampled_edges(0)
    assert want_offsets.tolist() == [0, 4, 10]
    _round_like_mps(monkeypatch)
    got, got_offsets = _sampled_edges(EMPTY)
    assert torch.equal(got_offsets[EMPTY:], want_offsets), "the edge ranges moved"
    assert torch.equal(got, want), _differ(got, want, "edge coordinates")


def _segment_circuits(monkeypatch, empty):
    """The circuit ``_build_circuit_geometry`` files each of two segments
    under, one segment per circuit after ``empty`` empty circuits, counted
    from the first of the two.

    ``_build_frame_bounds`` reduces each segment's box into the box of that
    circuit. The build is stopped where it samples the edges, which
    ``_sampled_edges`` calls on its own. Past that point the build packs
    every circuit's materials and colors, which this bare primitive does not
    have.
    """
    prim = _circuit_primitive()
    prim.num_segments_per_object = _counts(empty, 1, 1)
    prim.next_segment_inds = torch.arange(2).view(1, -1)
    prim.normals = torch.tensor([0.0, 0.0, 1.0]).expand(1, empty + 2, -1)
    prim.mob_center = torch.zeros((1, 1, 3)).expand(1, empty + 2, -1)
    prim.filled = False

    def sample(*args):
        raise _Numbered

    prim._sample_circuit_edges = sample
    monkeypatch.setattr(rt_settings, "bezier_geometry_cache", False)
    corners = torch.stack((_loop([0.0, 0.0, 0.0], 1.0), _loop([10.0, 0.0, 0.0], 0.5)))
    with pytest.raises(_Numbered):
        prim._build_circuit_geometry(corners.unsqueeze(0), torch.tensor([4, 6]))
    return prim._rt_circuit_of_segment - empty


def test_each_segment_bounds_its_own_circuit_past_the_ceiling(
    mps_friendly, monkeypatch
):
    """Each segment's box goes into its own circuit's frame bounds.

    On the CPU on a Mac, where projection runs. Through the rounding
    expansion, both segments were filed under circuit 2**24. Circuit
    2**24 + 1's bounds then stayed empty, so it would never be drawn, and
    circuit 2**24's grew to hold both segments.
    """
    want = _segment_circuits(monkeypatch, 0)
    assert want.tolist() == [0, 1]
    _round_like_mps(monkeypatch)
    got = _segment_circuits(monkeypatch, EMPTY)
    assert torch.equal(got, want), (
        f"segment circuits {got.tolist()} with the ids past 2**24, "
        f"against {want.tolist()}"
    )


def _biased_controls(empty):
    """Two circuits after ``empty`` empty ones, each moved toward the camera
    by its own ``z_index`` along its own eye ray (as
    ``test_coplanar_z_index.test_the_bias_runs_along_each_circuits_own_eye_ray``).
    """
    prim = _circuit_primitive()
    prim._has_z_index = True
    centres = torch.zeros((1, empty + 2, 3))
    centres[0, -2:] = torch.tensor([[3.0, 2.0, 0.0], [-4.0, 1.5, 1.0]])
    prim.mob_center = centres
    prim.z_index = torch.zeros((1, empty + 2, 1))
    prim.z_index[0, -2:, 0] = torch.tensor([1.0e4, 3.0e4])
    prim.num_segments_per_object = _counts(empty, 1, 1)
    corners = centres[:, -2:].unsqueeze(-2).expand(-1, -1, 4, -1)
    return prim._apply_z_index_bias(
        corners.contiguous(),
        torch.tensor([[0.0, 0.0, 20.0]]),
        torch.tensor([[0.0, 0.0, 10.0]]),
    )


def test_the_z_index_moves_each_circuit_by_its_own_bias_past_the_ceiling(
    mps_friendly, monkeypatch
):
    """Each circuit's controls move by its own bias along its own eye ray.

    On the CPU on a Mac, where projection runs. Through the rounding
    expansion circuit 2**24 + 1's controls moved by circuit 2**24's bias,
    along circuit 2**24's ray: one world unit instead of three, and off their
    own pixel.
    """
    want = _biased_controls(0)
    _round_like_mps(monkeypatch)
    got = _biased_controls(EMPTY)
    assert torch.equal(got, want), _differ(got, want, "control coordinates")


def _camera():
    """One frame, looking down +z from z = -5 (a ``_dice_logical_pn`` camera)."""
    origin = torch.tensor([[[0.0, 0.0, -5.0]]])
    return SimpleNamespace(
        ray_origin=origin,
        screen_point=origin + torch.tensor([0.0, 0.0, 1.0]),
        screen_basis=torch.eye(3).unsqueeze(0),
        screen_width=640,
        screen_height=480,
        output_screen_width=640,
        output_screen_height=480,
        analytic_raster=False,
    )


def _member_ids(cls, empty):
    """The surface id of every triangle a ``cls`` collection packs or dices
    from two members of one triangle (or patch) each, after ``empty`` empty
    members, counted from the first of the two.
    """
    corners = torch.tensor([[-1.0, -0.7, 0.0], [1.0, -0.7, 0.0], [0.0, 1.0, 0.0]])
    corners = torch.cat((corners, corners + torch.tensor([3.0, 0.0, 0.0])))
    normals = torch.zeros_like(corners)
    normals[:, 2] = 1.0
    shape = {"corners": corners.view(1, -1, 3), "normals": normals.view(1, -1, 3)}
    if cls is LogicalPNTrianglePrimitive:
        shape["render_tolerance_pixels"] = 500.0
    collection = cls(triangle_collection=[cls(colors=BLUE, **shape)])
    collection._obj_counts = [0] * empty + [1, 1]
    if cls is LogicalPNTrianglePrimitive:
        collection._dice_logical_pn(_camera())
        return collection._logical_pn_tri_obj - empty
    collection._pack_projected_flat_geometry(_camera())
    return collection._rt_tri_obj - empty


@pytest.mark.parametrize(
    "cls",
    [RayTracedTrianglePrimitive, LogicalPNTrianglePrimitive],
    ids=["flat", "logical_pn"],
)
def test_each_member_keeps_its_own_surface_id_past_the_ceiling(
    mps_friendly, monkeypatch, cls
):
    """Every triangle of a member, flat or diced, carries that member's id.

    On the CPU on a Mac, where projection runs. Through the rounding
    expansion members 2**24 and 2**24 + 1 both came out as surface 2**24,
    which the sheet compaction keys as one surface: where the two overlap a
    pixel, one sheet shaded from whichever fragment was larger.
    """
    want = _member_ids(cls, 0)
    assert want.unique().tolist() == [0, 1]
    _round_like_mps(monkeypatch)
    got = _member_ids(cls, EMPTY)
    assert torch.equal(got, want), _differ(got, want, "triangle surface ids")
