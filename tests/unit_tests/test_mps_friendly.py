"""MPS-friendly mode: the flag, the substitutions, and a render through them.

``DESIGN_mps_support.md`` §1.2 measured what Metal cannot do -- no float64
anywhere, no int64 atomics, no int64 amin/amax ``scatter_reduce_``, no
``cummax`` -- and ``algan.rendering.mps_compat`` is the one place the renderer
substitutes for each. These tests run the mode **on the CPU**, which is the
only way anything but an Apple machine can check it, and is exactly why the
helpers dispatch on the mode rather than on a tensor's device.

Four things are pinned here:

* the flag itself -- ``'auto'`` follows the render device, ``True``/``False``
  do not, the environment variable wins, and a nonsense value is refused;
* the substitutions in isolation -- the scan against ``cummax``/``cummin``,
  the dtypes against the mode;
* the narrowed **kernel variants** against the wide ones, which is what says
  the ``ti.template()`` dtype arguments really specialise rather than one arm
  silently reusing the other's compiled code (``CLAUDE.md``'s ``ti.static``
  hazard, which is why these are template arguments);
* an end-to-end render in the mode, compared against the same scene rendered
  out of it. That render is what would have caught the aliasing this mode
  introduced the first time: at float32 ``cov.to(accumulate_dtype())`` is the
  identity, so the shell-ceiling kernel's scratch buffer became the coverage
  array it was clamping.

The render is deliberately compared with a **loose** tolerance. MPS-friendly
mode is not bit-identical and is not meant to be -- narrowing the §6.6.4
accumulators is the trade Metal forces -- so what the comparison establishes
is that the mode renders the same picture, not the same bytes.
"""

from __future__ import annotations

import os

import pytest
import taichi as ti
import torch

from algan import LD, OUTWARD, RIGHT, UP, Off, Scene, Sphere, Square
from algan.errors import AlganConfigurationError
from algan.rendering import mps_compat
from algan.rendering.mps_compat import (
    _MPS_EXACT_INT_BITS,
    accumulate_dtype,
    band_class_groups,
    cummax_values,
    cummin_values,
    gather_packed_key,
    mps_friendly,
    reduction_index_dtype,
    reduction_index_sentinel,
    taichi_accumulate_dtype,
    taichi_reduction_index_dtype,
)
from algan.settings import SETTINGS

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture
def computing_settings(monkeypatch):
    """``SETTINGS.computing``, restored, with the environment out of the way.

    ``ALGAN_MPS_FRIENDLY`` overrides the setting by design, so a run that has
    it exported -- which is exactly how this mode is exercised over a CPU
    render device, in ``mps_probe.yaml``'s Linux arm and in any local
    ``ALGAN_MPS_FRIENDLY=1 pytest`` -- would make every test below assert
    against the environment rather than against what it set.
    """
    monkeypatch.delenv("ALGAN_MPS_FRIENDLY", raising=False)
    snapshot = SETTINGS.snapshot()
    try:
        yield SETTINGS.computing
    finally:
        SETTINGS.restore(snapshot)


# ------------------------------------------------------------------ the flag


@pytest.mark.fast
def test_auto_follows_the_render_device(computing_settings, monkeypatch):
    """The default resolves per call, because the device is settable per render."""
    computing_settings.set(mps_friendly="auto")
    monkeypatch.setattr(mps_compat, "render_device", lambda: torch.device("cpu"))
    assert mps_friendly() is False
    monkeypatch.setattr(mps_compat, "render_device", lambda: torch.device("mps"))
    assert mps_friendly() is True


@pytest.mark.fast
@pytest.mark.parametrize("value", [True, False])
def test_an_explicit_value_ignores_the_device(computing_settings, monkeypatch, value):
    computing_settings.set(mps_friendly=value)
    for device in ("cpu", "mps"):
        monkeypatch.setattr(
            mps_compat, "render_device", lambda d=device: torch.device(d)
        )
        assert mps_friendly() is value


@pytest.mark.fast
def test_the_environment_variable_wins(computing_settings, monkeypatch):
    """``ALGAN_MPS_FRIENDLY`` is how a CPU machine runs the mode's own tests."""
    computing_settings.set(mps_friendly=False)
    monkeypatch.setitem(os.environ, "ALGAN_MPS_FRIENDLY", "1")
    assert mps_friendly() is True
    monkeypatch.setitem(os.environ, "ALGAN_MPS_FRIENDLY", "0")
    computing_settings.set(mps_friendly=True)
    assert mps_friendly() is False


@pytest.mark.fast
def test_a_string_spelling_of_the_flag_is_accepted(computing_settings):
    computing_settings.set(mps_friendly="on")
    assert computing_settings.mps_friendly is True
    computing_settings.set(mps_friendly="AUTO")
    assert computing_settings.mps_friendly == "auto"


@pytest.mark.fast
def test_a_nonsense_value_is_refused(computing_settings):
    with pytest.raises(AlganConfigurationError, match="mps_friendly"):
        computing_settings.set(mps_friendly="sometimes")


@pytest.mark.fast
def test_the_flag_survives_a_snapshot_round_trip(computing_settings):
    computing_settings.set(mps_friendly=True)
    snapshot = SETTINGS.snapshot()
    computing_settings.set(mps_friendly="auto")
    SETTINGS.restore(snapshot)
    assert SETTINGS.computing.mps_friendly is True


# ------------------------------------------------------- the dtype selectors


@pytest.mark.fast
def test_the_dtypes_follow_the_mode(computing_settings):
    computing_settings.set(mps_friendly=False)
    assert accumulate_dtype() is torch.float64
    assert reduction_index_dtype() is torch.int64
    assert reduction_index_sentinel() == 1 << 40
    assert taichi_accumulate_dtype() is ti.f64
    assert taichi_reduction_index_dtype() is ti.i64

    computing_settings.set(mps_friendly=True)
    assert accumulate_dtype() is torch.float32
    assert reduction_index_dtype() is torch.int32
    assert reduction_index_sentinel() == 2147483647
    assert taichi_accumulate_dtype() is ti.f32
    assert taichi_reduction_index_dtype() is ti.i32


@pytest.mark.fast
def test_the_sentinel_fits_its_own_dtype(computing_settings):
    """A sentinel that overflowed its slot would wrap to a real position."""
    for value in (False, True):
        computing_settings.set(mps_friendly=value)
        dtype = reduction_index_dtype()
        assert reduction_index_sentinel() <= torch.iinfo(dtype).max
        filled = torch.full((3,), reduction_index_sentinel(), dtype=dtype)
        assert int(filled[0]) == reduction_index_sentinel()


# ---------------------------------------------------------------- the scans


@pytest.mark.fast
@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
def test_the_scan_reproduces_cummax_and_cummin(computing_settings, dim, dtype):
    """The log-step scan is ``cummax``/``cummin``, exactly, not approximately.

    ``maximum`` and ``minimum`` are idempotent, so the doubling's overlapping
    ranges cannot change the answer, and neither op reassociates -- there is
    no float rounding to differ over.
    """
    generator = torch.Generator().manual_seed(7)
    if dtype.is_floating_point:
        x = torch.randn((7, 5, 9), generator=generator, dtype=dtype)
    else:
        x = torch.randint(-50, 50, (7, 5, 9), generator=generator, dtype=dtype)

    computing_settings.set(mps_friendly=True)
    assert torch.equal(cummax_values(x, dim), torch.cummax(x, dim).values)
    assert torch.equal(cummin_values(x, dim), torch.cummin(x, dim).values)


@pytest.mark.fast
@pytest.mark.parametrize("n", [0, 1, 2, 3, 64, 65])
def test_the_scan_handles_every_length(computing_settings, n):
    """Including the ones the doubling's bounds are easiest to get wrong on."""
    computing_settings.set(mps_friendly=True)
    x = torch.arange(n, dtype=torch.float32).flip(0)
    got = cummax_values(x, 0)
    assert got.shape == x.shape
    if n:
        assert torch.equal(got, torch.cummax(x, 0).values)


@pytest.mark.fast
def test_the_scan_does_not_alias_its_input(computing_settings):
    """A returned view would let a later in-place write corrupt the source."""
    computing_settings.set(mps_friendly=True)
    x = torch.tensor([3.0, 1.0, 2.0])
    got = cummax_values(x, 0)
    got += 1.0
    assert torch.equal(x, torch.tensor([3.0, 1.0, 2.0]))


# ------------------------------------------------- the split key gather


def _packed_keys(n, seed=0):
    """A fragment stream's packed keys: ``pixel << 32 | bit_cast(depth)``.

    The same shape ``raster_taichi.py:2039`` writes, and the same magnitude:
    a 1080p pixel index puts these near 2**50, which is where MPS's own
    int64 gather stops being exact.
    """
    g = torch.Generator().manual_seed(seed)
    pixel = torch.randint(0, 1920 * 1080, (n,), generator=g, dtype=torch.int64)
    depth = torch.rand(n, generator=g, dtype=torch.float32) * 4.0 + 4.0
    return (pixel << 32) | depth.view(torch.int32).to(torch.int64), pixel, depth


@pytest.mark.fast
@pytest.mark.parametrize("friendly", [False, True])
def test_the_packed_gather_matches_index_select(computing_settings, friendly):
    """Both arms answer exactly what ``index_select`` answers.

    The substituted form must be ``index_select``'s answer exactly -- the
    packed key back bit for bit, pixel word and depth word both -- or the
    compaction cannot order fragments by depth.
    """
    computing_settings.set(mps_friendly=friendly)
    keys, pixel, depth = _packed_keys(4096)
    order = torch.randperm(4096)

    got = gather_packed_key(keys, order)

    assert torch.equal(got, keys.index_select(0, order))
    assert torch.equal(got >> 32, pixel.index_select(0, order))
    assert torch.equal(
        (got & 0xFFFFFFFF).to(torch.int32).view(torch.float32),
        depth.index_select(0, order),
    )


@pytest.mark.fast
def test_the_packed_gather_is_exact_across_the_whole_int64_range():
    """Word boundaries, both sign bits, and the extremes.

    Algan's keys are non-negative and their low words are float32 depth bits,
    but a gather that silently drops one bit in one corner is exactly the class
    of defect this function exists to fix, so it should not depend on the
    caller's range staying polite.
    """
    keys = torch.tensor(
        [
            0,
            1,
            -1,
            (1 << 50) | 0x40E68475,
            (7 << 32) | 0xFFFFFFFF,
            (1 << 32) | 0x80000000,
            (1 << 63) - 1,
            -(1 << 63),
            0xFFFF,
            0x10000,
        ],
        dtype=torch.int64,
    )
    order = torch.tensor([9, 0, 4, 7, 2, 5, 1, 8, 3, 6])
    for friendly in (False, True):
        SETTINGS.computing.set(mps_friendly=friendly)
        assert torch.equal(gather_packed_key(keys, order), keys.index_select(0, order))


@pytest.mark.fast
def test_advanced_indexing_is_exact_above_the_mps_ceiling(computing_settings):
    """The dispatch ``gather_packed_key`` bets on, checked on real hardware.

    MPS's ``index_select`` and ``torch.gather`` round integer values through a
    float32 -- exact below 2**24, silently wrong above (``mps_compat``'s
    ``_MPS_EXACT_INT_BITS`` has the measurement). Advanced indexing ``v[i]``
    lands on a different aten kernel and is exact, which is the whole reason
    the mode can gather the packed fragment key in one operation instead of
    four 16-bit lanes.

    Nothing in torch's API promises that. If a future version routes ``v[i]``
    to the rounding kernel, every Algan render on an Apple GPU goes quietly
    wrong -- the fragment key loses its depth word, the compaction cannot band
    anything, and the frame comes back flat. This is what turns that into a
    loud test failure instead.

    **It only bites on a Mac**, and deliberately so: it selects the device from
    ``torch.backends.mps.is_available()`` rather than from Algan's configured
    render device, so it guards on any Apple machine -- including one without
    the patched Taichi, where Algan itself would render on the CPU. A green run
    anywhere else does NOT clear this; on those machines it degenerates to a
    correctness check of the helper, because advanced indexing is exact on CPU
    and CUDA whatever the width.
    """
    computing_settings.set(mps_friendly=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # The real key magnitude (~2**50) plus the corners: the ceiling itself and
    # one past it, the widths the probe measured wrong, and both sign bits.
    keys, _, _ = _packed_keys(2048)
    corners = torch.tensor(
        [
            (1 << 24),
            (1 << 24) + 1,
            (1 << 25) + 1,
            18271053,  # the probe's own int32 2**25 counter-example
            976314890686,  # ... and its int64 2**40 one
            (1 << 62) + 6789,
            (1 << 63) - 1,
            -1,
            -(1 << 62) - 1,
            -(1 << 63),
        ],
        dtype=torch.int64,
    )
    values = torch.cat((keys, corners))
    index = torch.randperm(values.numel(), generator=torch.Generator().manual_seed(3))
    # The reference is computed on the CPU, where every gather is exact.
    want = values.index_select(0, index)

    on_device = values.to(device)
    moved = index.to(device)
    assert torch.equal(values, on_device.cpu()), (
        "the values changed on the way to the device, so nothing below is "
        "attributable to the gather"
    )

    raw = on_device[moved].cpu()
    assert torch.equal(raw, want), (
        f"advanced indexing v[i] is no longer exact on {device} for values "
        f"above 2**{_MPS_EXACT_INT_BITS}. mps_compat.gather_packed_key depends "
        "on it; switch that back to the 16-bit lane split (see its docstring) "
        "and report the regression upstream"
    )
    assert torch.equal(gather_packed_key(on_device, moved).cpu(), want)


@pytest.mark.fast
def test_a_band_of_zero_area_hands_its_siblings_a_finite_weight(computing_settings):
    """The divide guard in ``_sibling_weights``, on whatever device is here.

    ``share = cov / sum(area)`` is guarded by a tiny floor, and the floor is
    what MPS does not honour: ``clamp_min(1e-12)`` returns an exact zero
    unchanged there, so a band whose siblings all contributed zero coverage
    divided ``0 / 0`` and handed the resolve a NaN. Nothing downstream catches
    one -- ``eff <= min_alpha`` is false against a NaN, like every comparison --
    so the sheet composited instead of dropping out, and the closed shell of
    ``test_path_tracer.py::test_closed_shell_attenuates_once_at_authored_opacity``
    came back attenuated twice on the one column where a third sheet sits.

    The floor is now spelled ``where(x < eps, eps, x)``, which is ``clamp_min``
    bit for bit and does not go through the call that declines. This is the
    test that says so on hardware; ``benchmarks/_mps_torch_op_probe.py``'s
    ``probe_epsilon_clamp`` is where the two are compared spelling by spelling.

    Two zero-area bands, because the function only takes its sibling path when
    a band holds more than one sheet: a two-sibling band that is entirely empty
    and a three-sibling one, which is the arrangement the cube's near vertical
    edge actually produced.

    **It only bites on a Mac**, like the gather guard above, and for the same
    reason: it runs on ``torch.backends.mps.is_available()``'s device rather
    than Algan's configured one, so any Apple machine checks the real hardware
    and everywhere else it pins the arithmetic.

    The mode is switched on here for the same reason the gather guard switches
    it on: it is the mode that spells the floor as a ``where`` and the §6.6.4
    accumulators as float32, and both are what the Apple arm has to exercise.
    Left off, this reaches Metal asking for a float64 accumulator -- which
    Torch refuses outright, ahead of any guard -- through a ``clamp_min`` the
    hardware would not have honoured anyway.
    """
    from algan.rendering.raytracing.sheets import AA_MASK_ALL, _sibling_weights

    computing_settings.set(mps_friendly=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    # Walk order, two bands: sheets 0-1 in band 0, sheets 2-4 in band 1. Every
    # sheet's own coverage is zero, so both bands sum to zero area.
    sheet_band = torch.tensor([0, 0, 1, 1, 1], dtype=torch.int64, device=device)
    cov = torch.zeros(5, dtype=torch.float32, device=device)
    msk = torch.full((5,), AA_MASK_ALL, dtype=torch.int64, device=device)
    band_area = torch.zeros(2, dtype=torch.float32, device=device)
    band_corr = torch.zeros(2, dtype=torch.float32, device=device)
    band_union = torch.full((2,), AA_MASK_ALL, dtype=torch.int64, device=device)

    wgt, wmsk = _sibling_weights(sheet_band, cov, msk, band_area, band_union, band_corr)

    assert bool(torch.isfinite(wgt).all()), (
        f"a zero-area band produced {wgt.cpu().tolist()} on {device}: the "
        "divide guard in sheets._sibling_weights no longer holds, and a NaN "
        "coverage composites a sheet the resolve should have dropped"
    )
    # Zero area means zero share means zero weight -- the answer the guard was
    # always meant to give, sign included: every sibling but a band's last
    # carries the continuation flag as a negative.
    assert torch.equal(wgt.abs().cpu(), torch.zeros(5))
    assert torch.equal(wmsk.cpu(), torch.full((5,), AA_MASK_ALL, dtype=torch.int64))


@pytest.mark.fast
def test_the_packed_gather_leaves_narrow_dtypes_alone(computing_settings):
    """Only int64 takes the substituted path; everything else is ``index_select``."""
    computing_settings.set(mps_friendly=True)
    order = torch.tensor([2, 0, 1])
    for source in (
        torch.tensor([1.5, 2.5, 3.5]),
        torch.tensor([1, 2, 3], dtype=torch.int32),
    ):
        assert torch.equal(
            gather_packed_key(source, order), source.index_select(0, order)
        )


# ------------------------------------------------- FXAA's border padding


@pytest.mark.fast
@pytest.mark.parametrize(("h", "w"), [(5, 7), (4, 4), (1, 9), (9, 1)])
def test_a_clamped_grid_reproduces_border_padding(h, w):
    """FXAA's substitution, at the level it substitutes.

    ``grid_sampler_2d`` with padding mode 1 (border) raises on MPS --
    ``RuntimeError: MPS: Unsupported Border padding mode`` -- so ``fxaa`` runs
    mode 0 (zeros) over a grid clamped to the edge pixel CENTRES. That is not
    an approximation: with ``align_corners=False`` pixel ``i`` sits at
    ``(2i + 1) / N - 1``, so a coordinate past the first or last centre has its
    whole bilinear footprint clamped onto that edge pixel under border padding,
    which is exactly what sampling at the centre returns.

    Checked here at float64 over a grid reaching well outside the image, which
    is the case the padding mode exists for, and including the single-row and
    single-column shapes where the two clamp bounds meet.
    """
    generator = torch.Generator().manual_seed(h * 31 + w)
    image = torch.randn(2, 3, h, w, generator=generator, dtype=torch.float64)
    grid = torch.rand(2, h, w, 2, generator=generator, dtype=torch.float64) * 4.0 - 2.0

    border = torch.ops.aten.grid_sampler_2d(image, grid, 0, 1, False)
    clamped = grid.clone()
    clamped[..., 0].clamp_(1.0 / w - 1.0, 1.0 - 1.0 / w)
    clamped[..., 1].clamp_(1.0 / h - 1.0, 1.0 - 1.0 / h)
    zeros = torch.ops.aten.grid_sampler_2d(image, clamped, 0, 0, False)

    assert torch.allclose(border, zeros, rtol=0.0, atol=1e-12)


# ------------------------------------------------- the band/class grouping


def _wide_key_reference(band, cls, base):
    """What the composite key answers, which both arms must reproduce."""
    uniq, inverse = torch.unique(band * base + cls, sorted=True, return_inverse=True)
    return int(uniq.numel()), inverse, uniq // base


@pytest.mark.fast
@pytest.mark.parametrize(("bands", "classes"), [(1, 1), (7, 1), (40, 5), (400, 130)])
def test_the_pair_grouping_matches_the_wide_composite_key(
    computing_settings, bands, classes
):
    """The narrow grouping answers exactly what ``band * base + cls`` answers.

    Count, per-fragment group id AND each group's band, because the group ids
    are consumed as indices into per-group tables: an equivalent grouping in a
    different ORDER would pass a count check and still mis-shade every sheet.
    """
    base = 1 << 25
    g = torch.Generator().manual_seed(bands)
    band = torch.randint(0, bands, (2000,), generator=g, dtype=torch.int64)
    cls = torch.randint(0, classes, (2000,), generator=g, dtype=torch.int64)
    want_n, want_inverse, want_band = _wide_key_reference(band, cls, base)

    computing_settings.set(mps_friendly=True)
    got_n, got_inverse, got_band = band_class_groups(band, cls, base)

    assert got_n == want_n
    assert torch.equal(got_inverse, want_inverse)
    assert torch.equal(got_band, want_band)


@pytest.mark.fast
def test_the_pair_grouping_is_the_wide_key_off_the_mode(computing_settings):
    """Off MPS-friendly mode the composite key is what runs, unchanged."""
    base = 1 << 25
    band = torch.tensor([0, 0, 1, 1, 2], dtype=torch.int64)
    cls = torch.tensor([0, 3, 3, 3, 1], dtype=torch.int64)
    computing_settings.set(mps_friendly=False)
    got = band_class_groups(band, cls, base)
    want = _wide_key_reference(band, cls, base)
    assert got[0] == want[0]
    assert torch.equal(got[1], want[1])
    assert torch.equal(got[2], want[2])


@pytest.mark.fast
def test_the_pair_grouping_handles_an_empty_stream(computing_settings):
    """A chunk can rasterise nothing; the grouping must not index into it."""
    computing_settings.set(mps_friendly=True)
    empty = torch.zeros(0, dtype=torch.int64)
    count, inverse, band = band_class_groups(empty, empty, 1 << 25)
    assert count == 0
    assert inverse.numel() == 0
    assert band.numel() == 0


@pytest.mark.fast
def test_the_pair_grouping_survives_a_band_count_that_overflows_the_wide_key():
    """The whole point: a key the composite form could not represent.

    With 2**20 bands the composite reaches 2**45, which is where the Apple
    GPU's ``unique`` merged rows that differ only in their low bits and turned
    40956 sheets into 128. The narrow form never builds that value, so this
    stays exact -- and the reference below is computed on the CPU, where the
    wide key is exact, so it is a real comparison rather than two wrong
    answers agreeing.
    """
    base = 1 << 25
    band = torch.arange(1 << 20, dtype=torch.int64).repeat_interleave(2)
    cls = torch.arange(band.numel(), dtype=torch.int64) % 3
    want_n, want_inverse, want_band = _wide_key_reference(band, cls, base)

    SETTINGS.computing.set(mps_friendly=True)
    got_n, got_inverse, got_band = band_class_groups(band, cls, base)

    assert got_n == want_n
    assert torch.equal(got_inverse, want_inverse)
    assert torch.equal(got_band, want_band)


# ------------------------------------------------- the narrowed kernel arms


def _band_stats_arrays(nb, n, dtype):
    return (
        torch.full((nb,), n, dtype=dtype),
        torch.full((nb,), n, dtype=dtype),
        torch.full((nb,), n, dtype=dtype),
        torch.full((nb,), n, dtype=dtype),
        torch.zeros(nb, dtype=torch.float32),
        torch.zeros(nb, dtype=dtype),
    )


def test_the_int32_band_stats_kernel_answers_the_int64_one(wide_kernel_arms):
    """Positions and counts, so the narrow answer is the wide one exactly.

    Also the check that the ``idx_t`` template really specialises: if Taichi
    reused one variant's compiled code for the other, the int32 launch would
    write through an int64 view of a half-length buffer and these would not
    line up.
    """
    from algan.rendering.raytracing.raster_taichi import _AA_MASK_ALL as MASK_ALL
    from algan.rendering.raytracing.sheet_compact_taichi import (
        band_stats_reduce,
        band_stats_rep_orig,
    )
    from algan.rendering.taichi_runtime import init_taichi

    init_taichi()
    generator = torch.Generator().manual_seed(17)
    n, nb = 4096, 512
    band = torch.randint(0, nb, (n,), generator=generator)
    pos_o = torch.randperm(n, generator=generator)
    msk = torch.randint(0, MASK_ALL + 1, (n,), generator=generator, dtype=torch.int32)
    cov = torch.rand(n, generator=generator)

    results = {}
    for ti_dtype, torch_dtype in ((ti.i64, torch.int64), (ti.i32, torch.int32)):
        first, minp, first_p, minp_p, cmax, nfrag = _band_stats_arrays(
            nb, n, torch_dtype
        )
        band_stats_reduce(
            band.contiguous(),
            msk.contiguous(),
            pos_o.contiguous(),
            cov.contiguous(),
            n,
            int(MASK_ALL),
            first,
            minp,
            first_p,
            minp_p,
            cmax,
            nfrag,
            True,
            ti_dtype,
        )
        rep = torch.full((nb,), n, dtype=torch_dtype)
        band_stats_rep_orig(
            band.contiguous(),
            pos_o.contiguous(),
            cov.contiguous(),
            cmax,
            n,
            rep,
            ti_dtype,
        )
        results[torch_dtype] = [
            t.to(torch.int64) for t in (first, minp, first_p, minp_p, nfrag, rep)
        ] + [cmax]

    for wide, narrow in zip(results[torch.int64], results[torch.int32]):
        assert torch.equal(wide, narrow)
    # Not vacuous: some band really was reduced into.
    assert int(results[torch.int64][4].sum()) == n


def test_the_float32_area_kernel_tracks_the_float64_one(wide_kernel_arms):
    """The narrowed accumulator is the same sum, not the same bits."""
    from algan.rendering.raytracing.raster_taichi import (
        _AA_MASK_ALL as MASK_ALL,
    )
    from algan.rendering.raytracing.raster_taichi import (
        _AA_SLIVER_BIT as SLIVER_BIT,
    )
    from algan.rendering.raytracing.sheet_compact_taichi import sheet_band_reduce
    from algan.rendering.taichi_runtime import init_taichi

    init_taichi()
    generator = torch.Generator().manual_seed(23)
    n, nb = 8192, 256
    band = torch.randint(0, nb, (n,), generator=generator)
    msk = torch.randint(0, MASK_ALL + 1, (n,), generator=generator, dtype=torch.int32)
    cov = torch.rand(n, generator=generator)

    areas = {}
    unions = {}
    for ti_dtype, torch_dtype in ((ti.f64, torch.float64), (ti.f32, torch.float32)):
        area = torch.zeros(nb, dtype=torch_dtype)
        union = torch.zeros(nb, dtype=torch.int32)
        dup = torch.zeros(nb, dtype=torch.int32)
        sliver = torch.zeros(1, dtype=torch.int32)
        sheet_band_reduce(
            band.contiguous(),
            msk.contiguous(),
            cov.contiguous(),
            n,
            int(MASK_ALL),
            int(SLIVER_BIT),
            area,
            union,
            dup,
            sliver,
            False,
            ti_dtype,
        )
        areas[torch_dtype] = area.to(torch.float64)
        unions[torch_dtype] = union

    assert torch.equal(unions[torch.float64], unions[torch.float32])
    wide, narrow = areas[torch.float64], areas[torch.float32]
    assert wide.sum() > 0
    assert torch.allclose(wide, narrow, rtol=1e-5, atol=1e-5)
    # And it is genuinely the f32 sum rather than the f64 one relabelled --
    # a band of many fragments cannot round identically at both widths.
    assert not torch.equal(wide, narrow)


# ------------------------------------------------- nothing reaches past it


#: Spellings Metal cannot run, and the module that is allowed to say them
#: because selecting between them is its job.
_BANNED_ATTRIBUTES = {
    "torch.float64": "float64 does not exist on Metal; use accumulate_dtype()",
    "torch.double": "float64 does not exist on Metal; use accumulate_dtype()",
    "ti.f64": "Taichi's SPIR-V codegen refuses f64; take it as a template arg",
}
#: Method spellings of the same thing, plus the two scans MPS lacks.
_BANNED_METHODS = {
    "double": "float64 does not exist on Metal; use accumulate_dtype()",
    "cummax": "unimplemented on MPS; use mps_compat.cummax_values",
    "cummin": "unimplemented on MPS; use mps_compat.cummin_values",
}

#: Renderer modules that may still say float64, and why each is not a
#: violation. Both build a HOST tensor and never place it on the render
#: device, and torch's CPU backend has float64 on a Mac like anywhere else.
_HOST_ONLY = {
    # A dice pattern's barycentric weights, built once on the CPU and cast to
    # the render dtype by ``.to(dtype)`` before ``.to(device)`` ever runs.
    "rendering/logical_pn.py",
    # One import-time scalar: the PU-encoding normalisation constant.
    "rendering/denoise/denoise.py",
}


def _dotted(node):
    """``torch.float64`` for ``Attribute(Name('torch'), 'float64')``, else None."""
    import ast

    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


@pytest.mark.fast
def test_the_renderer_reaches_float64_only_through_mps_compat():
    """A new float64 accumulator in the renderer is an MPS render that aborts.

    The walk is over the AST rather than the text, because the renderer's
    comments say "float64" constantly -- §6.6.4 is the reason half these
    accumulators exist, and the arguments for them are worth keeping even
    where the mode narrows them.
    """
    import ast
    from pathlib import Path

    import algan

    root = Path(algan.__file__).resolve().parent / "rendering"
    offenders = []
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root.parent).as_posix()
        if path.name == "mps_compat.py" or relative in _HOST_ONLY:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute):
                continue
            dotted = _dotted(node)
            reason = _BANNED_ATTRIBUTES.get(dotted)
            if reason is None and node.attr in _BANNED_METHODS:
                dotted, reason = node.attr, _BANNED_METHODS[node.attr]
            if reason is not None:
                offenders.append(f"{relative}:{node.lineno}: {dotted} -- {reason}")
    assert not offenders, "\n".join(
        ["MPS-friendly mode cannot reach these:", *offenders]
    )


def _literal(node):
    """The float a constant expression denotes, or None if it is not one."""
    import ast

    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _literal(node.operand)
        return None if inner is None else -inner
    return None


@pytest.mark.fast
def test_the_renderer_floors_a_divide_through_clamp_floor():
    """A new ``clamp_min(1e-30)`` in the renderer is a wrong denominator on MPS.

    MPS rounds a clamp's scalar bound through float16, so every bound below
    float16's smallest subnormal comes back as that subnormal instead -- and so
    does every input below it, including inputs comfortably *above* the bound
    that was asked for. ``mps_compat.clamp_floor`` has the measurement.

    This is the defect in this file with the quietest failure mode. The other
    two abort or produce a NaN; this one hands back a plausible number, up to
    twenty-two orders of magnitude off the intended floor, in a denominator.
    Nothing downstream can tell. So the guard is structural: no literal bound
    below the cliff may reach a ``clamp`` in ``algan/rendering`` at all.

    ``*_taichi.py`` is exempt and must be: those bodies compile to MSL through
    Taichi and never touch torch's MPS dispatch, so ``ti.math.clamp`` there is
    a different function with a different defect surface.
    """
    import ast
    from pathlib import Path

    import algan
    from algan.rendering.mps_compat import _MPS_CLAMP_FLOOR

    root = Path(algan.__file__).resolve().parent / "rendering"
    offenders = []
    for path in sorted(root.rglob("*.py")):
        if path.name.endswith("_taichi.py") or path.name == "mps_compat.py":
            continue
        relative = path.relative_to(root.parent).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            name = (
                function.attr
                if isinstance(function, ast.Attribute)
                else getattr(function, "id", None)
            )
            if name not in ("clamp", "clamp_min", "clamp_max", "clip"):
                continue
            bounds = [_literal(a) for a in node.args]
            bounds += [
                _literal(k.value) for k in node.keywords if k.arg in ("min", "max")
            ]
            for bound in bounds:
                if bound is not None and 0 < abs(bound) < _MPS_CLAMP_FLOOR:
                    offenders.append(
                        f"{relative}:{node.lineno}: {name}({bound:g}) -- below "
                        f"{_MPS_CLAMP_FLOOR:g}, so MPS returns that instead"
                    )
    assert not offenders, "\n".join(
        [
            "These floors are silently wrong on MPS. Use "
            "mps_compat.clamp_floor(tensor, bound):",
            *offenders,
        ]
    )


# ------------------------------------------------------------ end to end


def _small_scene():
    with Off():
        Square(size=2.0).spawn()
        Sphere(radius=0.6).move(RIGHT * 1.2 + UP * 0.4).spawn()
        Sphere(radius=0.45).move(OUTWARD * 1.0).spawn()


def _render(tmp_path, name):
    from PIL import Image

    path = tmp_path / name
    with Scene() as scene:
        _small_scene()
        scene.save_frame(str(path), video_settings=LD)
    return torch.from_numpy(
        __import__("numpy").asarray(Image.open(path).convert("RGB")).copy()
    ).to(torch.int16)


def test_a_scene_renders_in_mps_friendly_mode(
    computing_settings, tmp_path, wide_kernel_arms
):
    """The whole renderer, with every float64 accumulator narrowed.

    The tolerance is loose on purpose. What moves between the two arms is the
    §6.6.4 accumulators, and they feed *thresholds* -- a coverage ceiling that
    wobbles in its low bits flips borderline fragments in and out of being
    clipped -- so the differences concentrate on silhouettes rather than
    spreading over the image. Measured on the fast suite's own scene: 99.94%
    of channels identical, 0.019% past a difference of 2, worst 34. So this
    asserts on the SHAPE of that distribution, which a real breakage (a
    corrupted buffer, a dropped clamp) violates immediately, rather than on a
    per-pixel bound that would only be pinning noise.
    """
    computing_settings.set(mps_friendly=False)
    wide = _render(tmp_path, "wide.png")
    computing_settings.set(mps_friendly=True)
    narrow = _render(tmp_path, "narrow.png")

    assert wide.shape == narrow.shape
    difference = (wide - narrow).abs()
    channels = difference.numel()
    assert float((difference == 0).sum()) / channels > 0.99
    assert float((difference > 2).sum()) / channels < 0.005
    assert int(difference.max()) <= 64
