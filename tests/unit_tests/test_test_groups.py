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
    """Whether a pin, a node-id prefix, names any test function."""
    file, _, rest = pin.partition("::")
    path = ROOT / file
    return path.is_file() and any(
        name.startswith(rest) and name.split("::")[-1].startswith("test")
        for name in _definitions(path)
    )


@pytest.mark.parametrize("pins", ["ALWAYS_CI", "ALWAYS_PRE_PR"])
def test_every_pin_still_names_a_test(pins):
    stale = [p for p in getattr(_selector(), pins) if not _names_a_test(p)]
    assert not stale, (
        f"{stale} in {pins} (scripts/select_test_groups.py) match no test "
        f"function any more; point them at the test's new name, or drop them."
    )
