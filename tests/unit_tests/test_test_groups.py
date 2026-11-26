"""The test groups in tests/groups/ must name tests that still exist.

``pytest --group ci`` and ``--group pre-pr`` select by node id, and a node id
that nothing collects is silently dropped: renaming or deleting a test takes
it out of every group without a word, and the group quietly covers less than
its header says. So each listed test is checked against its file's source --
the file exists and defines the function, at module level or in the named
class -- which is cheap enough to run in the CI group itself.

The pre-PR group is a superset of the CI group by construction (see
scripts/select_test_groups.py); that is checked too, since a hand edit to one
list could break it.

So are the script's pins. ``ALWAYS_CI`` and ``ALWAYS_PRE_PR`` name the tests
the greedy pass would never pick -- source guards, and the regression tests of
a fix, which add no lines of their own -- and ``--group`` runs every test they
name, listed or not (a regeneration that predates the pins dropped the sheet
resolve's regression tests, and nothing noticed for a week). A pin left behind
by a rename names nothing and runs nothing, just as silently.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from functools import cache
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GROUPS = ROOT / "tests" / "groups"


def _members(name: str) -> list[str]:
    lines = (GROUPS / f"{name}.txt").read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]


@cache
def _definitions(path: Path) -> set[str]:
    """``func`` and ``Class::func`` for every test function in a file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    names.add(f"{node.name}::{sub.name}")
    return names


@pytest.mark.parametrize("group", ["ci", "pre-pr"])
def test_every_listed_test_still_exists(group):
    missing = []
    for nodeid in _members(group):
        file, _, rest = nodeid.partition("::")
        path = ROOT / file
        name = rest.split("[", 1)[0]
        if not path.is_file() or name not in _definitions(path):
            missing.append(nodeid)
    assert not missing, (
        f"tests/groups/{group}.txt names {len(missing)} test(s) that no longer "
        f"exist, e.g. {missing[:5]}. Remove those lines, or regenerate the "
        f"groups with scripts/select_test_groups.py."
    )


def test_the_pre_pr_group_contains_the_ci_group():
    missing = sorted(set(_members("ci")) - set(_members("pre-pr")))
    assert not missing, (
        f"{len(missing)} test(s) in the CI group are not in the pre-PR group, "
        f"e.g. {missing[:5]}; a pull request that passed pre-PR could then "
        f"fail CI on a test nobody ran."
    )


@cache
def _selector():
    """scripts/select_test_groups.py, loaded by path: scripts/ is no package."""
    path = ROOT / "scripts" / "select_test_groups.py"
    spec = importlib.util.spec_from_file_location("select_test_groups", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _names_a_test(pin):
    """Whether a pin, a node-id prefix, names any test function.

    A pin may name one parametrized case (``test_x[0-None-40]``); the case id
    is generated at collection, so only its function is checked here.
    """
    file, _, rest = pin.partition("::")
    rest = rest.split("[", 1)[0]
    path = ROOT / file
    return path.is_file() and any(
        name.startswith(rest) and name.split("::")[-1].startswith("test")
        for name in _definitions(path)
    )


class _Item:
    """Just enough of a collected test for ``_test_group_keeps``."""

    def __init__(self, nodeid, *markers):
        self.nodeid = nodeid
        self._markers = {mark.name: mark for mark in markers}

    def get_closest_marker(self, name):
        return self._markers.get(name)


@pytest.mark.parametrize("group", ["ci", "pre-pr"])
def test_a_fast_test_runs_in_every_group_whether_listed_or_not(group):
    """The fast suite is in both groups by definition, not by regeneration.

    The lists seed it only when they are regenerated, so the guards marked
    ``fast`` since -- the kernel control-flow walk's cases for the two newest
    kernel modules, the packed-view cost check -- ran on no pull request.
    """
    keeps = sys.modules["conftest"]._test_group_keeps
    members = set(_members(group))
    pins = sys.modules["conftest"]._test_group_pins(group)
    nodeid = "tests/unit_tests/test_kernel_control_flow.py::test_a_case_no_list_names"
    assert nodeid not in members
    assert not nodeid.startswith(pins)
    # By name: spelled as an attribute, the marker would put this file in the
    # fast suite as far as test_fast_suite_curation.py can tell.
    fast = getattr(pytest.mark, "fast").mark  # noqa: B009
    assert keeps(_Item(nodeid, fast), members, pins)
    assert not keeps(_Item(nodeid), members, pins)


@pytest.mark.parametrize("pins", ["ALWAYS_CI", "ALWAYS_PRE_PR"])
def test_every_pin_still_names_a_test(pins):
    stale = [p for p in getattr(_selector(), pins) if not _names_a_test(p)]
    assert not stale, (
        f"{stale} in {pins} (scripts/select_test_groups.py) match no test "
        f"function any more; point them at the test's new name, or drop them."
    )


#: Portable test files with no test in the pre-PR group, frozen as they stood
#: when the check below was added. The groups were last measured on
#: 2026-10-23 and the greedy pass chose none of these files' tests: their lines
#: were covered more cheaply by others, or they cover none. The next
#: regeneration may change that; drop an entry once its file has a test in the
#: group (the check says so). Never add one: give a new file's tests a group
#: by pinning them instead (tests/README.md, "Test groups").
NOT_IN_GROUPS = frozenset(
    {
        "test_audio_context.py",
        "test_bvh_block_bound_range.py",
        "test_callable_signature.py",
        "test_circuit_frames.py",
        "test_circuit_span_candidates.py",
        "test_color_grid_broadcasting.py",
        "test_daemon_kernel_persistence.py",
        "test_default_shader_albedo.py",
        "test_dielectric_fresnel.py",
        "test_distant_camera_reflections.py",
        "test_doc_builder.py",
        "test_frame_readback.py",
        "test_ggx_albedo.py",
        "test_glass_energy_table.py",
        "test_gpu_harness.py",
        "test_hud_near_camera.py",
        "test_lagged_wave_timing.py",
        "test_lit_bsdf_sidedness.py",
        "test_mac_branch_cleanup.py",
        "test_mac_depth_buffer_ab.py",
        "test_manim_compat_spawned_edits.py",
        "test_manim_shader_render.py",
        "test_mps_arena_build.py",
        "test_nested_ior.py",
        "test_numeric_display_render.py",
        "test_phong_specular_normalization.py",
        "test_primary_shadow_sort.py",
        "test_profiler_stage_attribution.py",
        "test_promoted_maps_batched.py",
        "test_quadrants_distribution.py",
        "test_quadrants_ptx_cache.py",
        "test_quadrants_wheels.py",
        "test_refit_link_decode.py",
        "test_render_determinism.py",
        "test_rough_dielectric.py",
        "test_rough_glass_furnace_render.py",
        "test_shadow_fan_grain.py",
        "test_shadow_ray_budget.py",
        "test_shadow_vis_slots.py",
        "test_source_key_check_harness.py",
        "test_still_output.py",
        "test_surface_parametric_ranges.py",
        "test_surface_prep_identity.py",
        "test_taichi_source_key_fingerprint.py",
        "test_tile_state_fills.py",
        "test_verify_baseline_pointer.py",
        "test_volume_absorption.py",
        "test_wide_attr_device.py",
    }
)


def _portable_test_files():
    for suite in ("unit_tests", "fast"):
        yield from sorted((ROOT / "tests" / suite).glob("test_*.py"))


def _files_with_a_pre_pr_test():
    """Files the pre-PR group runs a test of: listed, pinned, or marked fast."""
    pins = _selector().ALWAYS_CI + _selector().ALWAYS_PRE_PR
    named = {nodeid.split("::", 1)[0] for nodeid in _members("pre-pr") + list(pins)}
    files = set()
    for path in _portable_test_files():
        rel = path.relative_to(ROOT).as_posix()
        # `--group` keeps every `fast` test (tests/conftest.py); a marker is
        # either module-level or on a test, and both spell it this way. Split,
        # as test_fast_suite_curation.py does, so this file does not match.
        marker = "pytest.mark." + "fast"
        if rel in named or marker in path.read_text(encoding="utf-8"):
            files.add(path.name)
    return files


def test_every_portable_test_file_has_a_test_in_the_pre_pr_group():
    """A new test file must not sit in no group until someone regenerates.

    The groups are measured, and the measurement needs a whole-suite coverage
    run, so it happens rarely; five files that landed after the last one --
    the caustics and volume tests among them, 171 tests -- ran on no pull
    request and in no pre-PR run, and nothing said so.
    """
    covered = _files_with_a_pre_pr_test()
    missing = [
        path.relative_to(ROOT).as_posix()
        for path in _portable_test_files()
        if path.name not in covered and path.name not in NOT_IN_GROUPS
    ]
    assert not missing, (
        f"{missing} have no test in the pre-PR group. Pin one in "
        f"scripts/select_test_groups.py: the file's whole `<path>::` in "
        f"ALWAYS_CI if it runs in a fraction of a second, else a representative "
        f"test in ALWAYS_PRE_PR (tests/README.md, 'Test groups')."
    )


def test_the_files_left_out_of_the_groups_are_still_left_out():
    """``NOT_IN_GROUPS`` only shrinks: an entry that is now covered, or gone, goes."""
    names = {path.name for path in _portable_test_files()}
    gone = sorted(NOT_IN_GROUPS - names)
    covered = sorted(NOT_IN_GROUPS & _files_with_a_pre_pr_test())
    assert not gone, f"NOT_IN_GROUPS names files that no longer exist: {gone}"
    assert not covered, (
        f"{covered} now have a test in the pre-PR group; drop them from "
        f"NOT_IN_GROUPS in this file."
    )
