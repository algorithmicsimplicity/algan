"""Host-side logic of the shade kernels' compile-time material gating
(``ALGAN_FRAG_PID_GATE`` / ``rt_settings.FRAG_PID_GATE``).

The mask decides which material stages are compiled into the shade kernels,
so an id the kernel can still read but the mask omits would shade that
surface with the wrong material (or none). These assert the two invariants
that keep it safe: the mask is only ever narrowed from real merge-time id
lists, and anything unknown falls back to "every stage compiled in".

Render-level parity of the gate is covered by
``benchmarks/_frag_pid_gate_ab.py`` (byte-identical on both shade kernels).

    .venv/Scripts/python.exe -m pytest tests/unit_tests/test_frag_pid_gate.py
"""

import pytest

from algan.rendering.raytracing import settings as rt_settings
from algan.rendering.raytracing.shading_taichi import (
    _MID_PHYSICAL,
    _MID_STANDARD,
    _MID_UNLIT,
    _USER_PIPELINE_BASE,
    ALL_PIDS,
    solo_pid,
)
from algan.rendering.raytracing.tracer import _frag_pid_mask


@pytest.fixture
def gate_on():
    before = rt_settings.FRAG_PID_GATE
    rt_settings.set_frag_pid_gate(True)
    yield
    rt_settings.set_frag_pid_gate(before)


@pytest.fixture
def gate_off():
    before = rt_settings.FRAG_PID_GATE
    rt_settings.set_frag_pid_gate(False)
    yield
    rt_settings.set_frag_pid_gate(before)


# --------------------------------------------------------------------------
# solo_pid: when may the kernel skip the id fetch entirely?
# --------------------------------------------------------------------------
def test_all_pids_is_never_solo():
    # The ungated sentinel has every bit set, so the runtime switch stays.
    assert solo_pid(ALL_PIDS, 0) == -1


def test_single_builtin_id_is_solo():
    assert solo_pid(1 << _MID_STANDARD, 0) == _MID_STANDARD
    assert solo_pid(1 << _MID_UNLIT, 0) == _MID_UNLIT


def test_two_ids_are_not_solo():
    assert solo_pid((1 << _MID_STANDARD) | (1 << _MID_PHYSICAL), 0) == -1


def test_single_user_pipeline_is_solo_only_when_injected():
    mask = 1 << (_USER_PIPELINE_BASE + 1)
    assert solo_pid(mask, 2) == _USER_PIPELINE_BASE + 1
    # An id with no pipeline behind it must keep the runtime switch: the
    # unconditional call would have nothing to call.
    assert solo_pid(mask, 1) == -1


# --------------------------------------------------------------------------
# _frag_pid_mask: the host bitmask handed to the kernels
# --------------------------------------------------------------------------
def test_mask_is_all_pids_when_the_gate_is_off(gate_off):
    merged = {"tri_material_ids": (_MID_STANDARD,)}
    assert _frag_pid_mask(merged, "tri", 1, _record=False) == ALL_PIDS


def test_mask_covers_every_merged_material_id(gate_on):
    merged = {"tri_material_ids": (_MID_UNLIT, _MID_STANDARD)}
    mask = _frag_pid_mask(merged, "tri", 1, _record=False)
    assert mask == (1 << _MID_UNLIT) | (1 << _MID_STANDARD)
    # Every id in the list must survive as a compiled-in stage.
    for pid in merged["tri_material_ids"]:
        assert (mask >> pid) & 1


def test_mask_is_all_pids_for_an_absent_geometry_type(gate_on):
    # num_pn == 0 leaves a placeholder id list behind; gating on it would be
    # correct but pointless, and ALL_PIDS avoids a needless kernel variant.
    merged = {"pn_material_ids": (0,)}
    assert _frag_pid_mask(merged, "pn", 0, _record=False) == ALL_PIDS


def test_mask_is_all_pids_when_the_scene_predates_the_id_list(gate_on):
    # Externally assembled merged scenes may not carry the merge-time list;
    # guessing would risk deleting a stage the kernel still dispatches to.
    assert _frag_pid_mask({}, "tri", 1, _record=False) == ALL_PIDS
    assert _frag_pid_mask({"tri_material_ids": ()}, "tri", 1, _record=False) == ALL_PIDS
