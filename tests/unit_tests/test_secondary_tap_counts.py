"""The secondary continuation-tap tables: positions, ownership, and the clamp.

``analytic_aa_secondary_samples`` used to be snapped to ``8 / 4 / 2``, so a
configured 16 or 32 rendered exactly as 8 and said nothing. Two things capped
it: the position table had hand-written entries only at 1/2/4/8, and each of
the eight coverage samples owned its single nearest position, so no fragment
could ever own more than eight positions however long the table was. The
tables below are what replaced both, and these tests pin them.

Three claims here are load-bearing rather than tidy:

* The four hand-written tap counts must return EXACTLY today's
  ``(position_mask, count)`` for every one of the 256 possible coverage masks
  -- the render baselines were taken through this mapping, so any drift in it
  is a renderer change. The rule is spelled out in this file (not imported
  from the module under test) so the contract cannot rotate underneath the
  test. That includes its quirks: at 8 taps position 1 is nobody's nearest,
  so a fully covered fragment spawns SEVEN rays there, not eight -- pinned
  behaviour, not an accident this file corrects.
* Every other count owns its positions by the inverse rule -- each POSITION
  assigned to its nearest coverage sample -- which partitions all n positions
  across the samples and makes a fully covered fragment spawn all n rays.
  That partition property is what makes a tap count above eight mean anything.
* The setting itself clamps to ``[1, _AA_SEC_MAX]`` instead of snapping, and
  warns once per process when it clamps.

The probe drives the ``@ti.func`` through one small kernel over every mask x
count pair -- the same shape as ``test_dielectric_fresnel.py``. Outside the
fast suite: it compiles Taichi kernels, and nothing elsewhere in the codebase
can break it (see ``tests/README.md`` on what earns a ``fast`` mark).
"""

# No ``from __future__ import annotations`` here, deliberately: the probe
# below defines a real ``@ti.kernel``, and stringised annotations turn
# ``ti.types.ndarray()`` into text that Taichi rejects at compile time.
import logging

import numpy as np
import pytest

from algan.rendering.raytracing import settings as rt_settings
from algan.rendering.raytracing.raster_taichi import (
    _AA_FIXED_SCALE,
    _AA_NUM_SAMPLES,
    _AA_SAMPLES,
    _AA_SEC_JITTER,
    _AA_SEC_JITTER_HANDWRITTEN,
    _AA_SEC_MAX,
    _AA_SEC_OWNER,
    _sec_positions,
)
from algan.taichi_compat import ti

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

#: The tap counts whose tables render baselines pin.
PINNED_COUNTS = tuple(sorted(_AA_SEC_JITTER_HANDWRITTEN))


@ti.kernel
def _probe_all(
    msks: ti.types.ndarray(),
    out_pm: ti.types.ndarray(),
    out_cnt: ti.types.ndarray(),
):
    """``_sec_positions`` for every mask x tap-count pair, one launch."""
    for i in range(msks.shape[0]):
        msk = msks[i]
        for c in ti.static(range(_AA_SEC_MAX)):
            pm, cnt = _sec_positions(msk, c + 1)
            out_pm[i, c] = pm
            out_cnt[i, c] = cnt


@pytest.fixture(scope="module")
def resolved():
    """Every ``(mask, tap count) -> (position mask, count)``, resolved once."""
    msks = np.arange(256, dtype=np.int32)
    pm = np.zeros((256, _AA_SEC_MAX), dtype=np.int32)
    cnt = np.zeros((256, _AA_SEC_MAX), dtype=np.int32)
    _probe_all(msks, pm, cnt)
    return pm, cnt


def _forward_owner_list(n):
    """The forward rule's per-sample owner indices, for tap count ``n``."""
    positions = _AA_SEC_JITTER_HANDWRITTEN[n]
    owners = []
    for ox, oy in (
        (0.5 + sx / _AA_FIXED_SCALE, 0.5 + sy / _AA_FIXED_SCALE)
        for sx, sy in _AA_SAMPLES
    ):
        best, bd = 0, float("inf")
        for j, (jx, jy) in enumerate(positions):
            d = (ox - jx) ** 2 + (oy - jy) ** 2
            if d < bd:
                bd, best = d, j
        owners.append(best)
    return owners


def reference_forward_positions(msk, n):
    """Today's forward-nearest rule, written out rather than imported.

    Each coverage sample sits at ``0.5 + offset / scale``; it owns the single
    hand-written position nearest it (first index winning ties), and a covered
    sample contributes exactly its owner's bit. This is the mapping the render
    baselines pin for tap counts 1/2/4/8.
    """
    owners = _forward_owner_list(n)
    pm = 0
    for k in range(_AA_NUM_SAMPLES):
        if (msk >> k) & 1:
            pm |= 1 << owners[k]
    cnt = 0
    for j in range(n):
        if (pm >> j) & 1:
            cnt += 1
    return pm, cnt


def test_pinned_counts_match_the_forward_rule_for_every_mask(resolved):
    """Tap counts 1/2/4/8 are byte-identical to today, all 256 masks each."""
    pm, cnt = resolved
    for n in PINNED_COUNTS:
        col = n - 1
        for msk in range(256):
            want_pm, want_cnt = reference_forward_positions(msk, n)
            assert int(pm[msk, col]) == want_pm, (n, msk)
            assert int(cnt[msk, col]) == want_cnt, (n, msk)


def test_jitter_table_has_n_distinct_unit_square_entries_for_every_count():
    for n in range(1, _AA_SEC_MAX + 1):
        pos = _AA_SEC_JITTER[n]
        assert len(pos) == n
        assert all(0.0 <= x < 1.0 and 0.0 <= y < 1.0 for x, y in pos), n
        assert len(set(pos)) == n


def test_generated_counts_partition_every_position():
    """Inverse ownership: each position under exactly one sample.

    The pinned counts cannot satisfy "exactly once" even in principle -- they
    keep the forward rule, where several samples may share an owner -- so
    theirs asserts the shape that rule has instead.
    """
    for n in range(1, _AA_SEC_MAX + 1):
        owned = [j for grp in _AA_SEC_OWNER[n] for j in grp]
        if n in PINNED_COUNTS:
            assert all(len(grp) == 1 for grp in _AA_SEC_OWNER[n])
            assert [grp[0] for grp in _AA_SEC_OWNER[n]] == _forward_owner_list(n)
        else:
            assert sorted(owned) == list(range(n)), n


def test_fully_covered_fragment_spawns_every_ray(resolved):
    """A full coverage mask owns all n positions -- except where pinned."""
    _, cnt = resolved
    full = (1 << _AA_NUM_SAMPLES) - 1
    for n in range(1, _AA_SEC_MAX + 1):
        got = int(cnt[full, n - 1])
        if n in PINNED_COUNTS:
            # Pinned behaviour, quirks included: at 8 taps position 1 is
            # nobody's nearest under the forward rule, so a fully covered
            # fragment spawns 7 rays there. Changing it would change renders.
            assert got == reference_forward_positions(full, n)[1], n
        else:
            assert got == n, n


@pytest.fixture
def restore_secondary_setting():
    """Undo the analytic-AA state this module's settings tests write."""
    aa_before = rt_settings.analytic_aa
    samples_before = rt_settings.analytic_aa_secondary_samples
    warned_before = rt_settings._SECONDARY_CLAMP_WARNED
    try:
        yield
    finally:
        rt_settings.set_analytic_aa(aa_before, secondary=samples_before)
        rt_settings._SECONDARY_CLAMP_WARNED = warned_before


class _Record:
    def __init__(self):
        self.records = []


def test_setting_returns_configured_count_and_clamps_over_ceiling(
    restore_secondary_setting,
):
    """16 stays 16 and 32 stays 32; 64 clamps to 32, warning once."""
    record = _Record()
    handler = logging.Handler()
    handler.emit = lambda r: record.records.append(r)
    logger = logging.getLogger("algan")
    logger.addHandler(handler)
    try:
        rt_settings._SECONDARY_CLAMP_WARNED = False

        rt_settings.set_analytic_aa(False, secondary=64)
        assert rt_settings.effective_analytic_aa_secondary_samples() == 1

        rt_settings.set_analytic_aa(True, secondary=16)
        assert rt_settings.effective_analytic_aa_secondary_samples() == 16
        rt_settings.set_analytic_aa(True, secondary=32)
        assert rt_settings.effective_analytic_aa_secondary_samples() == 32
        assert record.records == []

        rt_settings.set_analytic_aa(True, secondary=64)
        assert rt_settings.effective_analytic_aa_secondary_samples() == 32
        assert len(record.records) == 1
        assert "clamped" in record.records[0].getMessage()

        # A second over-ceiling read does not repeat the warning.
        assert rt_settings.effective_analytic_aa_secondary_samples() == 32
        assert len(record.records) == 1
    finally:
        logger.removeHandler(handler)


def test_setting_still_passes_small_counts_through(restore_secondary_setting):
    """Counts inside the supported range come back unchanged, as before."""
    rt_settings._SECONDARY_CLAMP_WARNED = False
    for value in (1, 2, 3, 4, 5, 7, 9, 31):
        rt_settings.set_analytic_aa(True, secondary=value)
        assert rt_settings.effective_analytic_aa_secondary_samples() == value
