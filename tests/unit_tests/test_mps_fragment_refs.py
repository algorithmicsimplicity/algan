"""Fragment refs past 2**24 through MPS's integer gather.

A fragment's ``ref`` is a triangle's index, or for a bezier circuit
``-((circuit << 8) + border weight + 1)`` (``raster_taichi._pack_bez_ref``).
So a circuit's ref passes 2**24 from a batch's 65,536th circuit on, and a
triangle's once a frame holds 2**24 triangle columns. MPS's ``index_select``
returns an integer as a float32, exact only through 2**24
(``mps_compat._MPS_EXACT_INT_BITS``). Rounded, a circuit's filled fragment
(border weight 0) decodes as the previous circuit's full border, and a
triangle index names a neighbouring triangle. ``mps_compat.gather_exact``
carries every gather of a ref.

Off an Apple GPU the rounding is emulated (``tests/unit_tests/mps_rounding``)
in MPS-friendly mode, the mode in which ``gather_exact`` changes anything. On
one, the synthetic tests also run on MPS tensors against the CPU's answer, and
the render runs there with the rounding real.

Feature tests of the renderer: not ``fast``.
"""

from __future__ import annotations

import pytest
import torch

from algan import (
    BLUE,
    GREEN,
    ORANGE,
    RED,
    SETTINGS,
    WHITE,
    YELLOW,
    Cube,
    Off,
    Scene,
    VideoSettings,
)
from algan.mobs.bezier_circuit import BezierCircuitCubic
from algan.rendering import fragment_capture
from algan.rendering.raytracing import raster_pipeline
from algan.rendering.raytracing.raster_taichi import (
    _AA_BACKFACE_BIT as BACKFACE,
)
from algan.rendering.raytracing.raster_taichi import (
    _AA_LOSE_SHIFT as LOSE_SHIFT,
)
from algan.rendering.raytracing.raster_taichi import (
    _AA_MASK_ALL as MASK_ALL,
)
from algan.rendering.raytracing.raster_taichi import (
    _AA_MAT_OPAQUE_BIT as MAT_OPAQUE,
)
from algan.rendering.raytracing.raster_taichi import (
    _BEZ_BORDER_BITS as BORDER_BITS,
)
from algan.rendering.raytracing.sheets import compact_sheets
from algan.scene_manager import SceneManager
from tests.unit_tests.mps_rounding import FIRST_ROUNDED, round_integer_gathers

ON_MPS = SETTINGS.computing.render_device.type == "mps"
#: The CPU with the rounding emulated, and on an Apple GPU the GPU itself.
DEVICES = ["cpu"] + (["mps"] if ON_MPS else [])

#: The first circuit whose ref passes 2**24.
FIRST_WIDE_CIRCUIT = 1 << (24 - BORDER_BITS)


def _bez_ref(circuit, weight):
    """``raster_taichi._pack_bez_ref`` for a border weight already in 0..255."""
    return -((circuit << BORDER_BITS) + weight + 1)


@pytest.fixture(autouse=True)
def _mps_friendly(monkeypatch):
    """The mode an Apple GPU renders in, where ``gather_exact`` is exact."""
    monkeypatch.delenv("ALGAN_MPS_FRIENDLY", raising=False)
    SETTINGS.computing.set(mps_friendly=True)


def _rounding_as_on(device, monkeypatch):
    """Round integer gathers through a float32 on the CPU; the GPU does it."""
    if device == "cpu":
        round_integer_gathers(monkeypatch)


# ---- A render past the 65,536th circuit ------------------------------------

RES = (96, 54)

#: Four overlapping squares, each ``(x, y, z, half side, color)``.
_SQUARES = [
    (-3.0, 0.5, 0.0, 1.6, RED),
    (-1.2, -0.4, 0.5, 1.4, GREEN),
    (0.6, 0.6, -0.5, 1.5, YELLOW),
    (2.6, -0.3, 1.0, 1.3, ORANGE),
]


def _square(x, y, z, half):
    """A square's outline as four straight cubic segments."""
    corners = torch.tensor(
        [
            [x - half, y - half, z],
            [x + half, y - half, z],
            [x + half, y + half, z],
            [x - half, y + half, z],
        ]
    )
    t = torch.tensor([0.0, 1 / 3, 2 / 3, 1.0]).view(1, 4, 1)
    return (corners[:, None] * (1 - t) + corners.roll(-1, 0)[:, None] * t).reshape(
        -1, 3
    )


def _off_screen_loop(x):
    """A one-segment circuit far to the right of the frame."""
    return torch.tensor([[x, 0.0, 0.0], [x + 1, 0, 0], [x, 1, 0], [x, 0, 0]])


def _squares_behind(hidden):
    """The squares as the last circuits of one pack, after ``hidden`` circuits
    the camera never sees, and a translucent cube among them for triangle
    sheets. The pack is the same draw order whatever ``hidden`` is, so the
    picture is too.
    """
    batches = [_off_screen_loop(50.0 + i % 7) for i in range(hidden)]
    batches += [_square(x, y, z, half) for x, y, z, half, _ in _SQUARES]
    pack = BezierCircuitCubic.from_batches(
        batches, color=BLUE, stroke_width=3, stroke_color=WHITE
    )
    for k, (*_, color) in enumerate(_SQUARES):
        pack[hidden + k].color = color
    pack.spawn(animate=False)
    cube = Cube(size=2.5, color=BLUE, opacity=0.5)
    cube.rotate(30, (0, 1, 0)).rotate(20, (1, 0, 0)).move_to((0.3, 0.0, 0.2))
    cube.spawn(animate=False)


def _captured_frame(hidden):
    SceneManager.reset()
    SETTINGS.raytracing.set(shadows=False, samples_per_pixel=1)
    SETTINGS.raytracing.experimental.hybrid_raster = True
    video = VideoSettings(resolution=RES, frames_per_second=10, supersampling=1)
    fragment_capture.arm()
    try:
        with Scene(video_settings=video) as scene:
            with Off():
                _squares_behind(hidden)
            image = torch.cat([f.cpu() for f in scene.get_frames(0, 1)]).float()[0]
    finally:
        records = fragment_capture.disarm()
        SceneManager.reset()
    return image, records


def _round_in_the_raster_pipeline(monkeypatch):
    """MPS's integer gather, emulated where an Apple GPU makes it.

    On the Mac the merged scene is built on the CPU (``merge_on_gpu_active``
    is CUDA-only), and the raster pipeline's two entry points run on the GPU.
    So the gathers round only while one of those runs. Rounding everywhere, as
    ``round_integer_gathers`` does, would also round the merge's gather of the
    circuits' edge headers, int32 words holding float bits
    (``bezier_acceleration``), which the Mac makes exactly.

    Returns how often each entry point ran, for the test to show both did.
    """
    depth = [0]
    for owner in (torch.Tensor, torch):
        for name in ("index_select", "gather"):

            def gather(*args, _real=getattr(owner, name), **kwargs):
                out = _real(*args, **kwargs)
                if depth[0] and out.dtype in (torch.int32, torch.int64):
                    out.copy_(out.to(torch.float32).to(out.dtype))
                return out

            monkeypatch.setattr(owner, name, gather)
    calls = {}
    for name in ("prepare_sparse_raster_coverage", "shade_sparse_raster_coverage"):
        calls[name] = 0

        def entry(*args, _real=getattr(raster_pipeline, name), _name=name, **kwargs):
            calls[_name] += 1
            depth[0] += 1
            try:
                return _real(*args, **kwargs)
            finally:
                depth[0] -= 1

        monkeypatch.setattr(raster_pipeline, name, entry)
    depth[0] += 1
    word = torch.tensor([FIRST_ROUNDED], dtype=torch.int32)
    rounds = int(word.index_select(0, torch.zeros(1, dtype=torch.int64)))
    depth[0] -= 1
    assert rounds != FIRST_ROUNDED, (
        "the simulated MPS rounding is not in effect, so the test proves nothing"
    )
    return calls


def test_circuits_past_the_65536th_render_as_they_do_with_small_ids(monkeypatch):
    """The same four squares, drawn as circuits 0..3 and as circuits 65,536..
    65,539 of a pack whose first 65,536 circuits are off screen, must give the
    same frame and the same sheets, refs shifted by the circuits ahead.

    Through a rounding gather a square's filled fragments (border weight 0)
    come back as the previous circuit's full border, so its inside takes the
    white of the stroke: 1,197 of the frame's pixels, before
    ``_gather_fragment_arrays``, the opaque-prefix truncation and the
    compaction's output gathered refs exactly.
    """
    calls = None if ON_MPS else _round_in_the_raster_pipeline(monkeypatch)
    small, small_records = _captured_frame(0)
    big, big_records = _captured_frame(FIRST_WIDE_CIRCUIT)
    if calls is not None:
        assert min(calls.values()) > 0, calls

    assert len(small_records) == len(big_records) == 1
    small_ref = small_records[0]["sheet_ref"].to(torch.int64)
    big_ref = big_records[0]["sheet_ref"].to(torch.int64)
    assert small_ref.shape == big_ref.shape
    want_ref = torch.where(
        small_ref < 0, small_ref - (FIRST_WIDE_CIRCUIT << BORDER_BITS), small_ref
    )
    # What makes this a test of the ceiling: every circuit sheet's ref is past
    # it, and triangle sheets ride the same stream.
    circuit = want_ref < 0
    assert bool(circuit.any())
    assert bool((~circuit).any())
    assert int(want_ref[circuit].max()) < -(2**24)
    moved_refs = int((big_ref != want_ref).sum())
    moved = (big[..., :3] - small[..., :3]).abs().amax(-1)
    moved_by = (
        f"{moved_refs} of {big_ref.numel()} sheet refs and "
        f"{int((moved > 0).sum())} pixels (by up to {float(moved.max()):.0f}) "
        "moved past the 65,536th circuit"
    )
    assert moved_refs == 0, moved_by
    assert torch.equal(big, small), moved_by
    assert torch.equal(big_records[0]["sheet_key"], small_records[0]["sheet_key"])


# ---- The compaction's own gathers ------------------------------------------

#: ``(pixel, depth, ref, area, mask)`` rows in emission order: circuits past
#: the ceiling whose border weight a float32 rounds to the previous circuit's
#: full border (weight 0, or 1 from 2**25 on), to a neighbouring weight (2) or
#: not at all (1 below 2**25), one ref past 2**30, and triangles sharing their
#: pixels.
_WIDE_CIRCUIT_ROWS = [
    (0, 1.0, 2, 0.6, 0b0000_0011),
    (0, 1.5, _bez_ref(FIRST_WIDE_CIRCUIT, 0), 0.5, MASK_ALL),
    (1, 1.2, _bez_ref(FIRST_WIDE_CIRCUIT + 1, 0), 0.4, 0b0000_1111),
    (1, 1.4, _bez_ref(FIRST_WIDE_CIRCUIT + 2, 2), 0.3, 0b1111_0000),
    (2, 1.0, _bez_ref(2 * FIRST_WIDE_CIRCUIT, 1), 0.7, MASK_ALL),
    (2, 2.0, 5, 0.9, 0b0011_1100),
    (3, 1.0, _bez_ref(5_000_000, 17), 0.2, 0b0101_0101),
    (5, 1.1, _bez_ref(FIRST_WIDE_CIRCUIT + 3, 1), 0.8, MASK_ALL),
]


def _key(pixel, depth):
    bits = int(torch.tensor([depth], dtype=torch.float32).view(torch.int32)[0])
    return (int(pixel) << 32) | (bits & 0xFFFFFFFF)


def _stream(rows, tri_obj, device):
    """A coverage record and merged scene for ``rows`` on a 4x4 frame."""
    pix = torch.tensor([r[0] for r in rows], dtype=torch.int64)
    covered, counts = torch.unique_consecutive(pix, return_counts=True)
    run_offsets = torch.zeros(covered.numel() + 1, dtype=torch.int32)
    run_offsets[1:] = torch.cumsum(counts.to(torch.int32), 0)
    n = len(rows)
    coverage = {
        "frag_key": torch.tensor([_key(r[0], r[1]) for r in rows]),
        "frag_ref": torch.tensor([r[2] for r in rows], dtype=torch.int32),
        "frag_ab": torch.zeros(n, 2),
        "frag_cov": torch.tensor([r[3] for r in rows], dtype=torch.float32),
        "frag_msk": torch.tensor([r[4] for r in rows], dtype=torch.int32),
        "frag_cap": torch.full((n,), 2.0),
        "covered_idx": covered.to(torch.int32),
        "run_offsets": run_offsets,
        "num_fragments": n,
        "num_covered": int(covered.numel()),
    }
    coverage = {
        k: v.to(device) if torch.is_tensor(v) else v for k, v in coverage.items()
    }
    merged = {"tri_obj": tri_obj.to(device)}
    return coverage, merged


def _compact(rows, tri_obj, device, **flags):
    coverage, merged = _stream(rows, tri_obj, device)
    out = compact_sheets(
        coverage,
        merged,
        torch.zeros(1, 3, device=device),
        torch.full((1,), 1e-3, device=device),
        time_start=0,
        width=4,
        height=4,
        band_rule="facing",
        **flags,
    )
    return {k: v.cpu() if torch.is_tensor(v) else v for k, v in out.items()}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("resolve_only", [True, False])
@pytest.mark.parametrize("sample_depth", [True, False])
def test_circuit_refs_past_the_ceiling_leave_the_compaction_exact(
    device, resolve_only, sample_depth, monkeypatch
):
    """Every circuit fragment is a sheet of its own, so its ref is the sheet's
    ref, bit for bit. Rounded, circuits 65,536 and 65,537 (weight 0) come back
    as 65,535 and 65,536 at full border weight, 131,072 (weight 1) as 131,071,
    and 5,000,000 as 4,999,999, and 65,538's weight 2 comes back as 3.
    """
    tri_obj = torch.zeros(1, 8, dtype=torch.int32)
    flags = {"resolve_only": resolve_only, "sample_depth": sample_depth}
    want = _compact(_WIDE_CIRCUIT_ROWS, tri_obj, "cpu", **flags)
    _rounding_as_on(device, monkeypatch)
    got = _compact(_WIDE_CIRCUIT_ROWS, tri_obj, device, **flags)

    refs = [r[2] for r in _WIDE_CIRCUIT_ROWS]
    assert sorted(got["sheet_ref"].tolist()) == sorted(refs)
    for name in ("sheet_ref", "sheet_key", "sheet_offsets", "sheet_wmsk"):
        assert torch.equal(got[name], want[name]), name


#: The sample-depth subject's triangle: past the ceiling and odd, so a
#: float32 rounds it to the even neighbour below it.
_WIDE_TRIANGLE = 2**24 + 1


def _subject_and_enforcer(subject_surface, neighbour_surface):
    """A material-opaque triangle covering pixel 0 (surface 0) in front of a
    triangle whose index is past the ceiling, and the ``tri_obj`` row that
    puts the far one on ``subject_surface`` and its rounded index on
    ``neighbour_surface``. A far triangle on the near one's surface is turned
    back-facing, so the two stay separate sheets.
    """
    tri_obj = torch.zeros(1, _WIDE_TRIANGLE + 1, dtype=torch.int32)
    tri_obj[0, _WIDE_TRIANGLE] = subject_surface
    tri_obj[0, _WIDE_TRIANGLE - 1] = neighbour_surface
    facing = BACKFACE if subject_surface == 0 else 0
    rows = [
        (0, 1.0, 3, 1.0, MASK_ALL | MAT_OPAQUE),
        (0, 2.0, _WIDE_TRIANGLE, 0.5, 0b0000_1111 | facing),
    ]
    return rows, tri_obj


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("resolve_only", [True, False])
@pytest.mark.parametrize(
    ("subject_surface", "cedes"),
    [(1, True), (0, False)],
    ids=["cedes_to_another_surface", "keeps_its_own_surfaces_samples"],
)
def test_a_triangle_ref_past_the_ceiling_keeps_its_surface_in_the_sample_depth_rule(
    device, resolve_only, subject_surface, cedes, monkeypatch
):
    """The sample-depth rule cedes a sheet's samples to a nearer opaque sheet
    of ANOTHER surface, and reads a sheet's surface off its ref. A triangle
    index past 2**24, rounded, names its neighbour and the neighbour's
    surface: the far sheet then keeps samples it should cede, or cedes to
    its own surface. Needs no 2**24-triangle scene: only ``tri_obj`` is that
    wide.
    """
    rows, tri_obj = _subject_and_enforcer(subject_surface, 1 - subject_surface)
    flags = {"resolve_only": resolve_only, "sample_depth": True}
    want = _compact(rows, tri_obj, "cpu", **flags)
    lose = (want["sheet_wmsk"] >> LOSE_SHIFT) & MASK_ALL
    assert lose.tolist() == [0, 0b0000_1111 if cedes else 0]
    _rounding_as_on(device, monkeypatch)
    got = _compact(rows, tri_obj, device, **flags)

    for name in ("sheet_wmsk",) + (() if resolve_only else ("sheet_msk",)):
        assert torch.equal(got[name], want[name]), name
    assert got["sheet_ref"].tolist() == [3, _WIDE_TRIANGLE]


# ---- The emission's sort gather --------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
def test_the_fragment_gather_carries_refs_past_the_ceiling_exactly(device, monkeypatch):
    """``_gather_fragment_arrays``' torch arm, the one MPS takes (the fused
    kernel is opt-in), puts the emission's refs into stream order.
    """
    refs = [
        _bez_ref(FIRST_WIDE_CIRCUIT, 0),
        _bez_ref(FIRST_WIDE_CIRCUIT + 2, 2),
        _bez_ref(5_000_000, 17),
        _WIDE_TRIANGLE,
        2**24 + 3,
        7,
    ]
    n = len(refs)
    idx = torch.tensor([4, 0, 5, 2, 1, 3], device=device)
    arrays = (
        torch.tensor([_key(p, 1.0 + p) for p in range(n)], device=device),
        torch.tensor(refs, dtype=torch.int32, device=device),
        torch.zeros(n, 2, device=device),
        torch.ones(n, device=device),
        torch.full((n,), MASK_ALL, dtype=torch.int32, device=device),
        torch.ones(n, dtype=torch.bool, device=device),
    )
    _rounding_as_on(device, monkeypatch)
    with SETTINGS.raytracing.experimental.override(raster_fused_gather=False):
        ref = raster_pipeline._gather_fragment_arrays(idx, *arrays)[1]
    assert ref.cpu().tolist() == [refs[i] for i in idx.tolist()]
