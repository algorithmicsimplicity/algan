"""The five sheet-chain host-pass kernels against the torch they replaced.

``solid_shell_ceiling``, ``band_stats_reduce``, ``band_stats_rep_orig``,
``pair_expand_count`` and ``pair_expand_write`` each replace a multi-pass
torch block in the sheet chain, and every replacement claims bit-identity.
These tests launch each kernel on small real inputs and compare bitwise
against the exact torch statements they replaced -- the only way to catch a
Taichi scoping/compile error, which no host-side inspection of the kernel
source can see. ``benchmarks/_sheet_kernel_check.py`` covers 4K-scale shapes;
this file pins small deterministic cases, including the routing gate in
``_class_pairs_flat``.

Not marked fast: each test pays a Taichi compile its own module owns (the
convention in ``test_sheet_compaction.py``).
"""

from __future__ import annotations

import inspect

import pytest
import torch

from algan import (
    BLUE,
    GREEN,
    RED,
    RIGHT,
    UP,
    Cube,
    Off,
    Scene,
    Square,
    VideoSettings,
)
from algan.rendering.mps_compat import kernel_index
from algan.rendering.raytracing import sheets
from algan.rendering.raytracing.raster_pipeline import (
    _class_pairs_flat,
    raster_chunk,
)
from algan.rendering.raytracing.raster_taichi import _AA_MASK_ALL as MASK_ALL
from algan.rendering.raytracing.sheet_compact_taichi import (
    SHELL_SPEND_ONE,
    SHELL_SPEND_QUANTUM,
    band_stats_reduce,
    band_stats_rep_orig,
    pair_expand_count,
    pair_expand_write,
    solid_shell_ceiling,
)
from algan.rendering.raytracing.sheets import _lexsort, _segment_exclusive_prefix
from algan.scene_manager import SceneManager
from algan.settings import SETTINGS
from algan.taichi_compat import template_dtype, ti
from tests.unit_tests.mps_rounding import round_integer_gathers

EXPERIMENTAL = SETTINGS.raytracing.experimental
# Dtype template arguments as the renderer passes them (template_dtype): a raw
# compiler dtype is not weakly referenceable and makes Quadrants warn.
ACC_F64 = template_dtype(ti.f64)
ACC_F32 = template_dtype(ti.f32)
IDX_I64 = template_dtype(ti.i64)


def _bits_equal(a, b):
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.dtype.is_floating_point:
        width = torch.int32 if a.dtype == torch.float32 else torch.int64
        return bool((a.view(width) == b.view(width)).all())
    return bool((a == b).all())


# ----------------------------------------------------- solid_shell_ceiling


def _shell_stream(n, seed, zero_areas=False):
    g = torch.Generator().manual_seed(seed)
    # Mostly closed-surface (pixel, surface) segments plus unique-negative
    # pass-through keys, exactly the two kinds compact_sheets emits.
    pix = torch.randint(0, 50, (n,), generator=g)
    sid = torch.randint(0, 3, (n,), generator=g)
    closed = torch.rand(n, generator=g) < 0.75
    key = torch.where(closed, pix * 5 + sid, -(torch.arange(n) + 1))
    depth = torch.rand(n, generator=g)
    o2 = _lexsort(key, depth)
    back = torch.rand(n, generator=g) < 0.5
    cov = torch.rand(n, generator=g)
    if zero_areas:
        cov = torch.full((n,), 1e-13)
        cov[::7] = 0.0
    return key.contiguous(), o2.contiguous(), back, cov.contiguous()


def _torch_shell_ceiling(key, o2, back, c2, cov_src):
    """The verbatim torch block from compact_sheets (toggle-off arm)."""
    n = int(key.numel())
    device = key.device
    k2 = key.index_select(0, o2)
    seg_start = torch.ones(n, dtype=torch.bool, device=device)
    if n > 1:
        seg_start[1:] = k2[1:] != k2[:-1]
    seg = torch.cumsum(seg_start.to(torch.int64), 0) - 1
    nseg = int(seg[-1].item()) + 1
    first = torch.zeros(nseg, dtype=torch.int64, device=device)
    first.scatter_(0, seg[seg_start], torch.nonzero(seg_start).reshape(-1))
    q2 = (c2 * SHELL_SPEND_ONE).to(torch.int64)
    excl_q = torch.cumsum(q2, 0).sub_(q2)
    spent = (
        (excl_q - excl_q.index_select(0, first).index_select(0, seg))
        .to(torch.float64)
        .mul_(SHELL_SPEND_QUANTUM)
    )
    backf2 = back.index_select(0, o2)
    z64 = torch.zeros((), dtype=torch.float64, device=device)
    front = torch.zeros(nseg, dtype=torch.float64, device=device)
    back_s = torch.zeros(nseg, dtype=torch.float64, device=device)
    front.scatter_add_(0, seg, torch.where(backf2, z64, c2))
    back_s.scatter_add_(0, seg, torch.where(backf2, c2, z64))
    cap = torch.maximum(front, back_s).to(torch.float32).to(torch.float64)
    scale = (
        cap.index_select(0, seg)
        .sub_(spent)
        .clamp_min_(0.0)
        .div_(c2.clamp_min_(1e-12))
        .clamp_max_(1.0)
    )
    out = cov_src.clone()
    out.index_copy_(0, o2, (c2 * scale).to(torch.float32))
    return out


@pytest.mark.parametrize("zero_areas", [False, True])
def test_solid_shell_ceiling_matches_torch_segment_clamp(zero_areas, wide_kernel_arms):
    n = 4096
    key, o2, back, cov = _shell_stream(n, seed=11, zero_areas=zero_areas)
    c2 = cov.to(torch.float64).index_select(0, o2)

    want = _torch_shell_ceiling(key, o2, back, c2, cov)

    got = cov.clone()
    solid_shell_ceiling(key, o2, back.contiguous().view(torch.uint8), n, got, ACC_F64)
    assert _bits_equal(want, got), (
        f"max abs diff {(want - got).abs().max().item()}, "
        f"{int((want.view(torch.int32) != got.view(torch.int32)).sum())} bits-differing rows"
    )

    again = cov.clone()
    solid_shell_ceiling(key, o2, back.contiguous().view(torch.uint8), n, again, ACC_F64)
    assert _bits_equal(got, again), "kernel is not deterministic across runs"


def test_solid_shell_ceiling_single_fragment_and_whole_stream_segments(
    wide_kernel_arms,
):
    for n, key_val in ((1, 0), (257, 0)):
        key = torch.full((n,), key_val, dtype=torch.int64)
        o2 = torch.arange(n, dtype=torch.int64)
        back = torch.zeros(n, dtype=torch.bool)
        if n > 1:
            back[::3] = True
        cov = torch.linspace(0.01, 0.9, n)
        c2 = cov.to(torch.float64).index_select(0, o2)
        want = _torch_shell_ceiling(key, o2, back, c2, cov)
        got = cov.clone()
        solid_shell_ceiling(
            key, o2, back.contiguous().view(torch.uint8), n, got, ACC_F64
        )
        assert _bits_equal(want, got), f"n={n}"


def test_solid_shell_ceiling_returns_what_it_does_not_clip_bit_for_bit():
    """A fragment the allowance covers comes back as it went in, on any arch.

    Two kinds, both common in a real stream: a fragment of an undeclared
    surface, which reaches the kernel as a one-fragment segment whose
    allowance is its own area, and the nearer crossing of a closed shell's
    front/back pair. Neither is the ceiling's to change, and the stroke-pixel
    split (``_split_stroke_pixels``) holds only because a stroke pixel's
    fragments come back unchanged when they do pass through here. On an Apple
    GPU they did not: the kernel wrote ``denom * min(left / denom, 1)``,
    Metal's division put ``c / c`` under 1 for about one area in seventy, and
    those came back an ulp low (which is how
    ``test_stroke_sheets.py::test_stroke_pixel_split_holds_with_a_closed_solid``
    failed there). The f32 arm is the one Metal runs, so this runs on every
    render device; the f64 arm is checked wherever it compiles.
    """
    g = torch.Generator().manual_seed(3)
    n = 1 << 16
    half = n // 2
    cov = torch.rand(n, generator=g) * 0.98 + 0.01
    key = torch.empty(n, dtype=torch.int64)
    # The first half undeclared: a unique negative key each, as compact_sheets
    # keys them.
    key[:half] = -(torch.arange(half) + 1)
    # The second half closed: (pixel, surface) pairs of a front and a back
    # crossing of one area, the nearer first, so the cap is that area.
    key[half:] = torch.arange(half) // 2
    cov[half + 1 :: 2] = cov[half::2]
    back = torch.zeros(n, dtype=torch.bool)
    back[half + 1 :: 2] = True
    o2 = _lexsort(key, torch.arange(n, dtype=torch.float32))
    kept = torch.cat([torch.arange(half), torch.arange(half, n, 2)])
    arms = {"f32": ACC_F32}
    if str(SETTINGS.computing.render_device) != "mps":
        arms["f64"] = ACC_F64
    for name, acc in arms.items():
        got = cov.clone()
        # As compact_sheets passes them: the stream order narrowed for Metal
        # (``kernel_index``), the identity off MPS-friendly mode.
        solid_shell_ceiling(key, kernel_index(o2), back.view(torch.uint8), n, got, acc)
        moved = got[kept].view(torch.int32) != cov[kept].view(torch.int32)
        assert not bool(moved.any()), (
            f"{name}: {int(moved.sum())} of {kept.numel()} fragments the allowance "
            "covers came back changed"
        )
        # The farther crossing of each pair is what the ceiling does govern:
        # nothing is left for it (bar the f64 spend's sub-quantum remainder).
        assert bool((got[half + 1 :: 2] < 1e-9).all()), name


def _large_shell_stream(nseg, seed):
    """A frame-scale stream: ``nseg`` closed (pixel, surface) segments of one
    to four crossings each, plus a pass-through fragment per five segments.

    Returns the kernel's inputs plus each row's segment ordinal and each
    segment's first row along ``o2`` (the torch arm's ``seg`` / ``first``).
    """
    g = torch.Generator().manual_seed(seed)
    lengths = torch.randint(1, 5, (nseg,), generator=g)
    closed_key = torch.repeat_interleave(torch.arange(nseg), lengths)
    n_pass = nseg // 5
    key = torch.cat([closed_key, -(torch.arange(n_pass) + 1)])
    n = int(key.numel())
    # Stream order is not segment order: shuffle, as the compaction's sort
    # input is.
    key = key.index_select(0, torch.randperm(n, generator=g)).contiguous()
    o2 = _lexsort(key, torch.rand(n, generator=g)).contiguous()
    back = torch.rand(n, generator=g) < 0.5
    cov = torch.rand(n, generator=g)
    k2 = key.index_select(0, o2)
    seg_start = torch.ones(n, dtype=torch.bool)
    seg_start[1:] = k2[1:] != k2[:-1]
    seg = torch.cumsum(seg_start.to(torch.int64), 0) - 1
    first = torch.nonzero(seg_start).reshape(-1)
    return key, o2, back, cov, seg, first


def test_float32_shell_ceiling_spend_does_not_degrade_with_stream_length(
    wide_kernel_arms,
):
    """MPS-friendly mode (``acc_t`` f32) on a stream the size of a 1080p frame.

    The spend -- the coverage a fragment's in-segment predecessors already
    took -- used to be a difference of two entries of a global prefix over the
    whole stream. Here that prefix would end near 1.5e6, where f32 spacing is
    0.125, so a fragment late in the stream had its spend, and with it the
    coverage it keeps, off by up to 0.24 (measured before the fix). The f64
    arm is the reference: what the f32 arm computes should differ from it
    only by rounding at the scale of one pixel's coverage.
    """
    key, o2, back, cov, _seg, _first = _large_shell_stream(1_200_000, seed=5)
    n = int(key.numel())
    back_u8 = back.contiguous().view(torch.uint8)
    assert float(cov.sum()) > 1e6, "the stream is too small to show the effect"

    want = cov.clone()
    solid_shell_ceiling(key, o2, back_u8, n, want, ACC_F64)
    got = cov.clone()
    solid_shell_ceiling(key, o2, back_u8, n, got, ACC_F32)

    err = (got - want).abs().max().item()
    assert err < 1e-5, f"f32 ceiling is off the f64 one by up to {err:.3g}"


def test_shell_ceiling_clamp_reads_only_its_own_segment(wide_kernel_arms):
    """A segment's clamp is a function of that segment alone, bit for bit.

    ``compact_sheets`` compacts a stream's triangle pixels apart from its
    stroke pixels (``_split_stroke_pixels``) and frames arrive in chunks of
    any size, so the same (pixel, surface) segment can sit behind millions of
    other fragments or none. The f64 spend is Q32 integer arithmetic, so it
    must come out the same either way. (A difference of two rows of a global
    float prefix did not: behind a prefix of ~1e6 its low bits moved.)
    """
    key, o2, back, cov, seg, first = _large_shell_stream(1_200_000, seed=7)
    n = int(key.numel())
    assert float(cov.sum()) > 1e6, "the stream is too small to show the effect"
    whole = cov.clone()
    solid_shell_ceiling(key, o2, back.contiguous().view(torch.uint8), n, whole, ACC_F64)
    clamped = (whole.view(torch.int32) != cov.view(torch.int32)).sum()
    assert int(clamped) > n // 10, "the stream exercises too few clamps"

    # The last 5000 segments along ``o2``, alone: the rows they own, in the
    # same relative order.
    tail_rows = o2[int(first[-5000]) :]
    sub_key = key.index_select(0, tail_rows)
    sub_back = back.index_select(0, tail_rows)
    sub_cov = cov.index_select(0, tail_rows)
    m = int(tail_rows.numel())
    alone = sub_cov.clone()
    solid_shell_ceiling(
        sub_key.contiguous(),
        torch.arange(m, dtype=torch.int64),
        sub_back.contiguous().view(torch.uint8),
        m,
        alone,
        ACC_F64,
    )
    assert _bits_equal(alone, whole.index_select(0, tail_rows))

    # And the torch arm reaches the same integers from the whole stream.
    want = _torch_shell_ceiling(key, o2, back, cov.to(torch.float64)[o2], cov)
    assert _bits_equal(want, whole)


def test_float32_segment_prefix_matches_the_float64_global_difference():
    """The torch arm's f32 spend (``_segment_exclusive_prefix``) on the same
    frame-scale stream, against the f64 global-prefix difference it replaces
    in that mode -- and exactly against a serial walk on a small stream,
    including one-row and empty ones.
    """
    key, o2, _back, cov, seg, first = _large_shell_stream(1_200_000, seed=6)
    c64 = cov.to(torch.float64).index_select(0, o2)
    excl64 = torch.cumsum(c64, 0).sub_(c64)
    want = excl64 - excl64.index_select(0, first).index_select(0, seg)
    got = _segment_exclusive_prefix(cov.index_select(0, o2), seg, first)
    assert got.dtype == torch.float32
    err = (got.to(torch.float64) - want).abs().max().item()
    assert err < 1e-5, f"segmented f32 prefix is off by up to {err:.3g}"

    # The old construction on the same data, to show what the test guards.
    c32 = cov.index_select(0, o2)
    excl32 = torch.cumsum(c32, 0).sub_(c32)
    old = excl32 - excl32.index_select(0, first).index_select(0, seg)
    assert (old.to(torch.float64) - want).abs().max().item() > 1e-2

    for n in (0, 1, 2, 37):
        _k, small_o2, _b, small_cov, small_seg, small_first = _large_shell_stream(
            max(n, 1), seed=n
        )
        values = small_cov.index_select(0, small_o2)[:n]
        sseg, sfirst = small_seg[:n], small_first[small_first < n]
        got = _segment_exclusive_prefix(values, sseg, sfirst)
        serial = torch.zeros_like(values)
        for i in range(1, n):
            if sseg[i] == sseg[i - 1]:
                serial[i] = serial[i - 1] + values[i - 1]
        assert torch.allclose(got, serial, atol=1e-6), n


def test_the_float32_segment_prefix_finds_its_rows_past_the_mps_gather_ceiling(
    monkeypatch,
):
    """The torch arm's float32 spend (``_segment_exclusive_prefix``) places a
    row in its segment by gathering the segment's first row, a stream
    position. Once a stream holds 2**24 rows, MPS's ``index_select`` rounds
    that position through a float32 (``mps_compat._MPS_EXACT_INT_BITS``), and
    a row then counts the previous segment's area as spent, or forgets one of
    its own segment's.

    Three-row segments of unit area past 2**24 rows, in MPS-friendly mode
    with integer gathers rounding as on MPS: each row must have spent exactly
    its place in its segment, which holds only while the gather goes through
    ``gather_exact``. On MPS tensors where MPS is the render device. The
    segment ids are built on the CPU, because MPS also rounds an int64 ``//``.
    """
    monkeypatch.delenv("ALGAN_MPS_FRIENDLY", raising=False)
    SETTINGS.computing.set(mps_friendly=True)
    device = SETTINGS.computing.render_device
    n = (1 << 24) + (1 << 16)
    seg = (torch.arange(n) // 3).to(device)
    first = torch.arange(0, n, 3).to(device)
    round_integer_gathers(monkeypatch)
    got = _segment_exclusive_prefix(torch.ones(n, device=device), seg, first)
    moved = int((got.cpu() != (torch.arange(n) % 3).to(torch.float32)).sum())
    assert not moved, (
        f"{moved} of {n} rows spent another segment's area or lost their own"
    )


# ----------------------- the ceiling past MPS's integer gather ceiling

#: The merged tables ``compact_sheets`` reads.
_COMPACTION_TABLES = ("tri_obj", "tri_closed", "tri_pos", "tri_norm")
#: The 4K frame a captured 96x54 frame is moved into, and the window's first
#: pixel, (column, row): far enough along the frame's pixel ids that a ceiling
#: key passes 2**24.
_FRAME_4K = (3840, 2160)
_WINDOW_ORIGIN = (1800, 1600)


def _two_overlapping_cubes_and_a_pane():
    """Two translucent cubes overlapping on screen and in depth, and a pane
    in front of both. The cubes are two declared closed surfaces that share
    pixels, so the ceiling sees two (pixel, surface) segments side by side;
    the pane is a circuit, and its fragments pass through on keys of their own.
    """
    near = Cube(size=3.0, color=RED, opacity=0.5)
    near.rotate(30, UP).rotate(20, RIGHT).move_to((-0.8, 0.0, 0.0))
    near.spawn(animate=False)
    far = Cube(size=3.0, color=BLUE, opacity=0.5)
    far.rotate(-25, UP).rotate(15, RIGHT).move_to((0.9, 0.3, -1.0))
    far.spawn(animate=False)
    pane = Square(size=3.0, color=GREEN, fill_opacity=0.4, stroke_width=0)
    pane.move_to((0.0, -0.8, 2.0)).spawn(animate=False)


def _copied(value):
    return value.clone() if torch.is_tensor(value) else value


def _captured_compaction(build):
    """One 96x54 frame of ``build``, rendered in MPS-friendly mode, and its
    sheet compaction as the renderer called it.

    Returned as ``compact_sheets``' arguments by name. The fragment stream and
    the merged tables are copies, because they are arena views that the
    renderer reuses once the chunk is done. Settings are restored afterwards,
    so a module-scoped fixture can call this once for every test using it.
    """
    real = sheets.compact_sheets
    calls = []

    def capture(*args, **kwargs):
        if not calls:
            bound = inspect.signature(real).bind(*args, **kwargs).arguments
            call = {
                name: _copied(value)
                for name, value in bound.items()
                if name not in ("coverage", "merged", "memory")
            }
            call["coverage"] = {k: _copied(v) for k, v in bound["coverage"].items()}
            call["merged"] = {
                k: bound["merged"][k].clone()
                for k in _COMPACTION_TABLES
                if k in bound["merged"]
            }
            calls.append(call)
        return real(*args, **kwargs)

    snapshot = SETTINGS.snapshot()
    SceneManager.reset()
    try:
        with pytest.MonkeyPatch.context() as patched:
            patched.delenv("ALGAN_MPS_FRIENDLY", raising=False)
            patched.setattr(sheets, "compact_sheets", capture)
            SETTINGS.computing.set(mps_friendly=True)
            SETTINGS.raytracing.set(shadows=False, samples_per_pixel=1)
            SETTINGS.raytracing.experimental.hybrid_raster = True
            video = VideoSettings(
                resolution=(96, 54), frames_per_second=10, supersampling=1
            )
            with Scene(video_settings=video) as scene:
                with Off():
                    build()
                # A lazy iterator: the frame renders as it is drawn from.
                for _frame in scene.get_frames(0, 1):
                    pass
    finally:
        SceneManager.reset()
        SETTINGS.restore(snapshot)
    assert calls, "the frame was not compacted into sheets"
    return calls[0]


@pytest.fixture(scope="module")
def cubes_in_a_4k_window():
    """``_two_overlapping_cubes_and_a_pane``'s compaction, in a 4K window."""
    return _in_a_4k_window(_captured_compaction(_two_overlapping_cubes_and_a_pane))


def _in_a_4k_window(call):
    """The captured compaction, its pixels moved into a window of a 4K frame.

    ``compact_sheets`` reads a pixel id only to find its frame and to group by
    it, and the band rule reads per-frame tables rather than pixel coordinates,
    so this changes the ids and nothing else. That is the point: the ceiling
    keys each fragment by ``pixel * (surface_bound + 2) + surface``, which here
    passes 2**24 with only two surfaces. The ids are computed on the CPU,
    because MPS also rounds an int64 ``//`` (DESIGN_mps_support.md §2.3b).
    """
    width = call["width"]
    coverage = dict(call["coverage"])
    n, covered = coverage["num_fragments"], coverage["num_covered"]
    x0, y0 = _WINDOW_ORIGIN

    def place(pix):
        assert int(pix.max()) < width * call["height"], "one frame, not a chunk"
        return (y0 + pix // width) * _FRAME_4K[0] + x0 + pix % width

    key = coverage["frag_key"][:n]
    host = key.cpu()
    coverage["frag_key"] = ((place(host >> 32) << 32) | (host & 0xFFFFFFFF)).to(
        key.device
    )
    ids = coverage["covered_idx"][:covered]
    coverage["covered_idx"] = place(ids.cpu().to(torch.int64)).to(ids)
    return {**call, "coverage": coverage, "width": _FRAME_4K[0], "height": _FRAME_4K[1]}


def _ceiling_keys_one_float_holds(call):
    """How many pixels hold two closed surfaces whose ceiling keys round to one
    float32, and the largest key: what makes this stream a test of the gathers.

    The keys are ``compact_sheets``' own, ``pix * K + sid`` with ``K`` the
    batch's largest surface id plus two, rebuilt on the CPU.
    """
    coverage, merged = call["coverage"], call["merged"]
    n = coverage["num_fragments"]
    tri_obj, tri_closed = merged["tri_obj"].cpu(), merged["tri_closed"].cpu()
    assert tri_obj.shape[0] == tri_closed.shape[0] == 1, "a still frame's tables"
    ref = coverage["frag_ref"][:n].cpu().to(torch.int64)
    safe = ref.clamp_min(0)
    closed = (ref >= 0) & (tri_closed[0, safe] > 0.5)
    K = int(tri_obj.max()) + 2
    pix = coverage["frag_key"][:n].cpu() >> 32
    keys = torch.unique((pix * K + tri_obj[0, safe].to(torch.int64))[closed])
    same_pixel = keys[1:] // K == keys[:-1] // K
    one_float = keys[1:].to(torch.float32) == keys[:-1].to(torch.float32)
    return int((same_pixel & one_float).sum()), int(keys.max())


def _compact(call, *, ceiling_kernel, pixel_sort):
    """The captured compaction, run again with the ceiling's arms chosen.

    ``ceiling_kernel`` picks the ceiling's spend (the kernel or its torch arm),
    and ``pixel_sort`` how its segments are ordered (each pixel's run sorted by
    ``key_run_order``, or ``_lexsort``'s torch fallback over the stream). Each
    run gets its own copy of the merged tables, so no run reuses a table
    another one cached on them.
    """
    arguments = {
        **call,
        "merged": dict(call["merged"]),
        "memory": None,
        "persist_output": False,
        "resolve_only": False,
    }
    with SETTINGS.raytracing.experimental.override(
        sheet_shell_ceiling_kernel=ceiling_kernel, sheet_pixel_sort=pixel_sort
    ):
        return sheets.compact_sheets(**arguments)


def _rows_differing(a, b):
    """How many leading-dimension rows of ``a`` and ``b`` differ in any bit."""
    if a.dtype.is_floating_point:
        bits = torch.int32 if a.dtype == torch.float32 else torch.int64
        a, b = a.view(bits), b.view(bits)
    ne = a != b
    return int(ne.reshape(ne.shape[0] if ne.dim() else 1, -1).any(1).sum())


def _assert_same_sheets(got, want, *, bitwise, what):
    """``got`` makes ``want``'s sheets: every output bit for bit, or, where the
    device sums in no fixed order or the arms add a segment's areas in a
    different order, the same sheets with coverage within float32 noise.
    Measured on the CPU, because an MPS tensor has no float64.
    """
    got, want = (
        {k: v.cpu() if torch.is_tensor(v) else v for k, v in out.items()}
        for out in (got, want)
    )
    assert set(got) == set(want), what
    assert got["num_sheets"] == want["num_sheets"], what
    names = sorted(want) if bitwise else ["sheet_key", "sheet_offsets"]
    moved = []
    for name in names:
        a, b = got[name], want[name]
        if not torch.is_tensor(b):
            assert a == b, f"{what} {name}"
            continue
        assert a.dtype == b.dtype, f"{what} {name}"
        assert a.shape == b.shape, f"{what} {name}"
        if not _bits_equal(a, b):
            by = ""
            if a.dtype.is_floating_point:
                by = f" by up to {float((a.double() - b.double()).abs().max()):.3g}"
            moved.append(f"{name} in {_rows_differing(a, b)}{by}")
    off = (got["sheet_cov"].double() - want["sheet_cov"].double()).abs()
    if not bitwise and bool((off > 1e-5).any()):
        moved.append(
            f"sheet_cov in {int((off > 1e-5).sum())} by up to {float(off.max()):.3g}"
        )
    assert not moved, f"{what} {', '.join(moved)} of {want['num_sheets']} sheets"


@pytest.mark.parametrize(
    ("ceiling_kernel", "pixel_sort"),
    [(False, True), (False, False), (True, False)],
    ids=["torch-ceiling", "torch-ceiling-torch-sort", "torch-sort"],
)
def test_the_shell_ceiling_keeps_its_segments_past_the_mps_gather_ceiling(
    monkeypatch, cubes_in_a_4k_window, ceiling_kernel, pixel_sort
):
    """MPS's ``index_select`` returns an integer as ``float32(value)``, exact
    only through 2**24 (``mps_compat._MPS_EXACT_INT_BITS``), and the closed-shell
    ceiling's segment key passes that: it is ``pixel * (surface_bound + 2) +
    surface``, so two solids in one pixel of a 4K frame's last third of pixel
    rows, or of a 1080p frame's last tenth with eight surfaces, can have keys
    one float32 cannot tell apart. Gathered that way, the torch arm merges the
    two segments and spends one allowance across both solids: the near one
    keeps its far shell, and the far one is dropped. ``_lexsort``'s torch
    fallback, which orders the segments when the per-pixel sort kernel is off,
    gathers the key too, and with it rounded the segments interleave, so even
    the kernel arm clamps each crossing on its own.

    A real frame's fragments, moved into a 4K window (``_in_a_4k_window``), in
    MPS-friendly mode with integer gathers rounding as on MPS
    (``round_integer_gathers``): each arm must compact them exactly as it does
    with exact gathers, and as the kernel arm over the per-pixel sort does,
    within float32 noise (in that mode the torch arm adds a segment's areas
    in a different order from the kernel). That holds only while every
    integer gather on the ceiling's path goes through ``gather_exact``. Run on
    MPS tensors where MPS is the render device, where the gathers round of
    their own accord.
    """
    monkeypatch.delenv("ALGAN_MPS_FRIENDLY", raising=False)
    SETTINGS.computing.set(mps_friendly=True)
    call = cubes_in_a_4k_window
    merging, widest = _ceiling_keys_one_float_holds(call)
    assert widest > 2**24, "no ceiling key passes 2**24 here"
    assert merging > 0, (
        "no two surfaces' ceiling keys round to one float32 here, so this "
        "proves nothing"
    )

    want = _compact(call, ceiling_kernel=True, pixel_sort=True)
    exact = _compact(call, ceiling_kernel=ceiling_kernel, pixel_sort=pixel_sort)
    round_integer_gathers(monkeypatch)
    got = _compact(call, ceiling_kernel=ceiling_kernel, pixel_sort=pixel_sort)

    # The ceiling clipped the far shells, so the comparisons are not vacuous.
    raw = float(call["coverage"]["frag_cov"].cpu().double().sum())
    assert float(want["sheet_cov"].cpu().double().sum()) < raw - 100.0

    # Bit for bit on the CPU; elsewhere the float32 sums have no fixed order.
    bitwise = want["sheet_cov"].device.type == "cpu"
    _assert_same_sheets(got, exact, bitwise=bitwise, what="rounding the gathers moved")
    _assert_same_sheets(got, want, bitwise=False, what="the arms disagree on")


# ------------------------------------------------------- band stats fusion


def _band_torch_arms(band, msk, pos_o, cov, nb, positioned):
    """The verbatim scatters from compact_sheets (toggle-off arms)."""
    n = int(band.numel())
    dev = band.device
    fs = torch.full((nb,), n, dtype=torch.int64, device=dev)
    fs.scatter_reduce_(
        0,
        band,
        torch.arange(n, dtype=torch.int64, device=dev),
        reduce="amin",
        include_self=True,
    )
    mp = torch.full((nb,), n, dtype=torch.int64, device=dev)
    mp.scatter_reduce_(0, band, pos_o, reduce="amin", include_self=True)
    fsp = torch.full((nb,), n, dtype=torch.int64, device=dev)
    mpp = torch.full((nb,), n, dtype=torch.int64, device=dev)
    if positioned:
        big = torch.full((), n, dtype=torch.int64, device=dev)
        posn = (msk & MASK_ALL) != 0
        masked = torch.where(posn, torch.arange(n, dtype=torch.int64, device=dev), big)
        fsp.scatter_reduce_(0, band, masked, reduce="amin", include_self=True)
        masked = torch.where(posn, pos_o, big)
        mpp.scatter_reduce_(0, band, masked, reduce="amin", include_self=True)
    cm = torch.zeros(nb, dtype=torch.float32, device=dev)
    cm.scatter_reduce_(0, band, cov, reduce="amax", include_self=True)
    nf = torch.zeros(nb, dtype=torch.int64, device=dev)
    nf.scatter_add_(0, band, torch.ones_like(band))
    is_max = cov >= cm.index_select(0, band)
    cand = torch.where(
        is_max, pos_o, torch.full((n,), n, dtype=torch.int64, device=dev)
    )
    rp = torch.full((nb,), n, dtype=torch.int64, device=dev)
    rp.scatter_reduce_(0, band, cand, reduce="amin", include_self=True)
    return fs, mp, fsp, mpp, cm, nf, rp


@pytest.mark.parametrize("positioned", [False, True])
def test_band_stats_kernels_match_the_five_scatters(positioned, wide_kernel_arms):
    g = torch.Generator().manual_seed(23)
    n, nb = 30_000, 4_001
    band = torch.randint(0, nb, (n,), generator=g)
    pos_o = torch.randperm(n, generator=g)
    msk = torch.randint(0, 1 << 22, (n,), generator=g, dtype=torch.int32)
    cov = torch.rand(n, generator=g)
    cov[::11] = 0.0

    w_fs, w_mp, w_fsp, w_mpp, w_cm, w_nf, w_rp = _band_torch_arms(
        band, msk, pos_o, cov, nb, positioned
    )

    fs = torch.full((nb,), n, dtype=torch.int64)
    mp = torch.full((nb,), n, dtype=torch.int64)
    fsp = torch.full((nb,), n, dtype=torch.int64)
    mpp = torch.full((nb,), n, dtype=torch.int64)
    cm = torch.zeros(nb, dtype=torch.float32)
    nf = torch.zeros(nb, dtype=torch.int64)
    band_stats_reduce(
        band.contiguous(),
        msk.contiguous(),
        pos_o.contiguous(),
        cov.contiguous(),
        n,
        int(MASK_ALL),
        fs,
        mp,
        fsp,
        mpp,
        cm,
        nf,
        bool(positioned),
        IDX_I64,
    )
    rp = torch.full((nb,), n, dtype=torch.int64)
    band_stats_rep_orig(
        band.contiguous(), pos_o.contiguous(), cov.contiguous(), cm, n, rp, IDX_I64
    )

    assert _bits_equal(w_fs, fs), "first_sorted"
    assert _bits_equal(w_mp, mp), "min_pos"
    assert _bits_equal(w_fsp, fsp), "first_sorted_p"
    assert _bits_equal(w_mpp, mpp), "min_pos_p"
    assert _bits_equal(w_cm, cm), "cmax"
    assert _bits_equal(w_nf, nf), "nfrag"
    assert _bits_equal(w_rp, rp), "rep_orig"


def test_band_stats_leaves_unused_band_rows_at_sentinel(wide_kernel_arms):
    g = torch.Generator().manual_seed(29)
    n, nb = 512, 128
    band = torch.randint(0, nb // 2, (n,), generator=g)  # upper half unused
    pos_o = torch.randperm(n, generator=g)
    msk = torch.full((n,), MASK_ALL, dtype=torch.int32)
    cov = torch.rand(n, generator=g)

    w_fs, w_mp, w_fsp, w_mpp, w_cm, w_nf, w_rp = _band_torch_arms(
        band, msk, pos_o, cov, nb, True
    )
    fs = torch.full((nb,), n, dtype=torch.int64)
    mp = torch.full((nb,), n, dtype=torch.int64)
    fsp = torch.full((nb,), n, dtype=torch.int64)
    mpp = torch.full((nb,), n, dtype=torch.int64)
    cm = torch.zeros(nb, dtype=torch.float32)
    nf = torch.zeros(nb, dtype=torch.int64)
    band_stats_reduce(
        band.contiguous(),
        msk.contiguous(),
        pos_o.contiguous(),
        cov.contiguous(),
        n,
        int(MASK_ALL),
        fs,
        mp,
        fsp,
        mpp,
        cm,
        nf,
        True,
        IDX_I64,
    )
    rp = torch.full((nb,), n, dtype=torch.int64)
    band_stats_rep_orig(
        band.contiguous(), pos_o.contiguous(), cov.contiguous(), cm, n, rp, IDX_I64
    )
    assert _bits_equal(w_fs, fs), "first_sorted"
    assert _bits_equal(w_mp, mp), "min_pos"
    assert _bits_equal(w_fsp, fsp), "first_sorted_p"
    assert _bits_equal(w_mpp, mpp), "min_pos_p"
    assert _bits_equal(w_rp, rp)


# ---------------------------------------------------- pair-row expansion


def _torch_pair_rows(mask, x0, x1, y0, y1, f_abs):
    """The verbatim torch body of _class_pairs_flat."""
    ncirc = mask.shape[1]
    idx = mask.reshape(-1).nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return None
    bx0 = x0.reshape(-1)[idx]
    by0 = y0.reshape(-1)[idx]
    bw = x1.reshape(-1)[idx] - bx0 + 1
    bh = y1.reshape(-1)[idx] - by0 + 1
    area = bw * bh
    nch = (area + (raster_chunk - 1)) // raster_chunk
    rep = torch.repeat_interleave(torch.arange(idx.numel()), nch)
    if rep.numel() == 0:
        return None
    base = torch.cumsum(nch, 0) - nch
    off = (torch.arange(rep.shape[0]) - base[rep]) * raster_chunk
    rows = torch.stack(
        [
            (idx % ncirc)[rep],
            f_abs.index_select(0, idx // ncirc)[rep],
            bx0[rep],
            by0[rep],
            bw[rep],
            bh[rep],
            off,
            torch.zeros_like(rep),
        ],
        -1,
    )
    return rows.to(torch.int32).contiguous()


def _kernel_pair_rows(mask, x0, x1, y0, y1, f_abs):
    """The host shape _pair_expand_rows gives the two kernels."""
    numel = int(mask.numel())
    ncirc = mask.shape[1]
    mflat = mask.reshape(-1).contiguous()
    x0f = x0.reshape(-1).contiguous()
    x1f = x1.reshape(-1).contiguous()
    y0f = y0.reshape(-1).contiguous()
    y1f = y1.reshape(-1).contiguous()
    counts = torch.empty(numel, dtype=torch.int64)
    # Box mode: the span arguments are inert (``spans`` 0), as in the host
    # when no projection table is handed over.
    screen = torch.zeros((1, 1, 10), dtype=torch.float32)
    f_abs_c = f_abs.contiguous()
    pair_expand_count(
        mflat.view(torch.uint8),
        x0f,
        x1f,
        y0f,
        y1f,
        numel,
        raster_chunk,
        counts,
        screen,
        f_abs_c,
        ncirc,
        4 * raster_chunk,
        0,
    )
    offs = torch.cumsum(counts, 0) - counts
    total = int(counts.sum().item())
    if total == 0:
        return None
    rows = torch.empty((total, 8), dtype=torch.int32)
    pair_expand_write(
        mflat.view(torch.uint8),
        x0f,
        x1f,
        y0f,
        y1f,
        f_abs_c,
        offs,
        numel,
        ncirc,
        raster_chunk,
        rows,
        screen,
        4 * raster_chunk,
        0,
    )
    return rows


def test_pair_expand_kernels_match_class_pairs_flat_body():
    g = torch.Generator().manual_seed(31)
    ft, ncirc = 3, 64
    # Interleaved candidates and non-candidates, multi-frame so both e % ncirc
    # and e // ncirc carry real values; one candidate's area is an exact
    # multiple of raster_chunk, another's degenerate (x0 == x1, y0 == y1).
    mask = torch.rand(ft, ncirc, generator=g) < 0.4
    mask[0, ::16] = True
    x0 = torch.randint(0, 800, (ft, ncirc), generator=g)
    x1 = x0 + torch.randint(1, 100, (ft, ncirc), generator=g)
    y0 = torch.randint(0, 450, (ft, ncirc), generator=g)
    y1 = y0 + torch.randint(1, 80, (ft, ncirc), generator=g)
    # exact-multiple area: width 32 x height 4 -> 128 = 4 chunks
    mask[1, 5] = True
    x0[1, 5], x1[1, 5], y0[1, 5], y1[1, 5] = 10, 41, 20, 23
    # degenerate one-pixel bbox
    mask[2, 7] = True
    x0[2, 7], x1[2, 7], y0[2, 7], y1[2, 7] = 300, 300, 200, 200
    f_abs = torch.arange(ft) + 17

    want = _torch_pair_rows(mask, x0, x1, y0, y1, f_abs)
    got = _kernel_pair_rows(mask, x0, x1, y0, y1, f_abs)
    assert want is not None
    assert got is not None
    assert got.dtype == torch.int32
    assert got.is_contiguous()
    assert _bits_equal(want, got), (
        f"row mismatch at "
        f"{int((want.view(torch.int32) != got.view(torch.int32)).any(-1).sum())}"
        f"/{want.shape[0]} rows"
    )


def test_pair_expand_empty_window_returns_none():
    mask = torch.zeros(2, 33, dtype=torch.bool)
    zeros = torch.zeros(2, 33, dtype=torch.int64)
    assert _kernel_pair_rows(mask, zeros, zeros, zeros, zeros, torch.arange(2)) is None


def test_class_pairs_flat_routes_by_toggle_and_agrees():
    g = torch.Generator().manual_seed(37)
    ft, ncirc = 2, 128
    mask = torch.rand(ft, ncirc, generator=g) < 0.35
    mask[0, ::9] = True
    x0 = torch.randint(0, 800, (ft, ncirc), generator=g)
    x1 = x0 + torch.randint(1, 90, (ft, ncirc), generator=g)
    y0 = torch.randint(0, 450, (ft, ncirc), generator=g)
    y1 = y0 + torch.randint(1, 60, (ft, ncirc), generator=g)
    f_abs = torch.arange(ft) + 4

    import algan.rendering.raytracing.sheet_compact_taichi as sct

    launches = {"count": 0, "write": 0}
    real_count, real_write = sct.pair_expand_count, sct.pair_expand_write

    def counted_count(*a, **k):
        launches["count"] += 1
        return real_count(*a, **k)

    def counted_write(*a, **k):
        launches["write"] += 1
        return real_write(*a, **k)

    sct.pair_expand_count, sct.pair_expand_write = counted_count, counted_write
    try:
        EXPERIMENTAL.set(raster_pair_expand_kernel=False)
        want = _class_pairs_flat(mask, x0, x1, y0, y1, f_abs, torch.device("cpu"))
        assert launches == {"count": 0, "write": 0}, "torch arm launched a kernel"

        EXPERIMENTAL.set(raster_pair_expand_kernel=True)
        got = _class_pairs_flat(mask, x0, x1, y0, y1, f_abs, torch.device("cpu"))
        assert launches["count"] >= 1, (
            "toggle ON did not route to the pair-expand kernels"
        )
        assert launches["write"] >= 1, (
            "toggle ON did not route to the pair-expand kernels"
        )
    finally:
        sct.pair_expand_count, sct.pair_expand_write = real_count, real_write
        EXPERIMENTAL.set(raster_pair_expand_kernel=True)

    assert want is not None
    assert got is not None
    assert got.dtype == torch.int32
    assert got.is_contiguous()
    assert _bits_equal(want, got)


@pytest.mark.parametrize("refused", ["window", "rows", None])
def test_class_pairs_flat_past_the_extent_limit_expands_in_torch(monkeypatch, refused):
    # Quadrants addresses an array in int32, so the kernel arm needs the window
    # (candidates) and the write pass's [total, 8] rows under the limit. The
    # window is asked before the count pass, the rows only after it; either
    # refusal hands the window to the torch expansion, which gives the same
    # rows.
    import algan.rendering.raytracing.sheet_compact_taichi as sct
    from algan.rendering.raytracing import sheet_stream

    g = torch.Generator().manual_seed(41)
    ft, ncirc = 2, 64
    mask = torch.rand(ft, ncirc, generator=g) < 0.4
    x0 = torch.randint(0, 800, (ft, ncirc), generator=g)
    x1 = x0 + torch.randint(1, 90, (ft, ncirc), generator=g)
    y0 = torch.randint(0, 450, (ft, ncirc), generator=g)
    y1 = y0 + torch.randint(1, 60, (ft, ncirc), generator=g)
    f_abs = torch.arange(ft) + 4
    cpu = torch.device("cpu")
    want = _class_pairs_flat(mask, x0, x1, y0, y1, f_abs, cpu)
    assert mask.numel() < want.numel()
    limit = {"window": mask.numel(), "rows": want.numel(), None: want.numel() + 1}
    launched = []
    for name in ("pair_expand_count", "pair_expand_write"):

        def spy(*args, _kernel=getattr(sct, name), _name=name):
            launched.append(_name)
            return _kernel(*args)

        monkeypatch.setattr(sct, name, spy)
    monkeypatch.setattr(sheet_stream, "KERNEL_EXTENT_LIMIT", limit[refused])
    got = _class_pairs_flat(mask, x0, x1, y0, y1, f_abs, cpu)
    assert ("pair_expand_count" in launched) == (refused != "window")
    assert ("pair_expand_write" in launched) == (refused is None)
    assert _bits_equal(want, got)


def _candidate_pixels(rows):
    """Set of (frame, prim, px, py) a pair table names."""
    out = set()
    for prim, f, x0, y0, bw, bh, off, _z in rows.tolist():
        for o in range(off, min(off + raster_chunk, bw * bh)):
            out.add((f, prim, x0 + o % bw, y0 + o // bw))
    return out


def test_span_candidates_follow_the_triangle_and_cover_every_pixel_it_touches():
    """Row-span mode (raster_span_candidates): a long thin diagonal triangle's
    candidates are a subset of its box's chunks and still contain every pixel
    its projection can touch.
    """
    ft, ncirc = 1, 2
    # Triangle 0: a 400 x 4 pixel diagonal sliver; triangle 1: small (box mode).
    sx = torch.tensor([[[10.0, 410.0, 412.0], [50.0, 58.0, 55.0]]])  # [F, C, 3]
    sy = torch.tensor([[[20.0, 300.0, 302.0], [30.0, 34.0, 37.0]]])
    screen = torch.zeros((ft, ncirc, 13))
    screen[..., 0:3] = sx
    screen[..., 3:6] = sy
    screen[..., 9] = 1.0
    x0 = (sx.amin(-1) - 1).floor().long()
    x1 = (sx.amax(-1) + 1).ceil().long()
    y0 = (sy.amin(-1) - 1).floor().long()
    y1 = (sy.amax(-1) + 1).ceil().long()
    mask = torch.ones(ft, ncirc, dtype=torch.bool)
    f_abs = torch.tensor([3])
    cpu = torch.device("cpu")
    with EXPERIMENTAL.override(
        raster_pair_expand_kernel=True, raster_span_candidates=True
    ):
        spans = _class_pairs_flat(mask, x0, x1, y0, y1, f_abs, cpu, screen=screen)
        boxes = _class_pairs_flat(mask, x0, x1, y0, y1, f_abs, cpu, screen=None)
    span_set = _candidate_pixels(spans)
    box_set = _candidate_pixels(boxes)
    assert span_set <= box_set
    assert len(span_set) < len(box_set) // 20  # the sliver's box is mostly empty
    # Every pixel a dense sampling of the triangle lands in is a candidate.
    g = torch.Generator().manual_seed(7)
    u = torch.rand(20000, generator=g)
    v = torch.rand(20000, generator=g)
    flip = u + v > 1
    u = torch.where(flip, 1 - u, u)
    v = torch.where(flip, 1 - v, v)
    for prim in range(ncirc):
        px = (
            sx[0, prim, 0]
            + u * (sx[0, prim, 1] - sx[0, prim, 0])
            + v * (sx[0, prim, 2] - sx[0, prim, 0])
        )
        py = (
            sy[0, prim, 0]
            + u * (sy[0, prim, 1] - sy[0, prim, 0])
            + v * (sy[0, prim, 2] - sy[0, prim, 0])
        )
        need = {
            (3, prim, int(x), int(y))
            for x, y in zip(px.floor().tolist(), py.floor().tolist())
        }
        assert need <= span_set
    # The small triangle keeps its whole box.
    assert {c for c in box_set if c[1] == 1} == {c for c in span_set if c[1] == 1}
