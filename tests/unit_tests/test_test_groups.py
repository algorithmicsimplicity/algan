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
a fix, which add no lines of their own -- and the lists must carry every one of
them. A regeneration that predates the pins dropped the sheet resolve's
regression tests exactly that way, and nothing noticed for a week.
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


def _pinned(prefix):
    """The test functions a pin (a node-id prefix) names, as node ids."""
    file, _, rest = prefix.partition("::")
    path = ROOT / file
    if not path.is_file():
        return []
    return [
        f"{file}::{name}"
        for name in sorted(_definitions(path))
        if name.startswith(rest)
        and all(
            part.startswith("Test") if i < name.count("::") else part.startswith("test")
            for i, part in enumerate(name.split("::"))
        )
    ]


@pytest.mark.parametrize(
    ("group", "pins"),
    [("ci", ("ALWAYS_CI",)), ("pre-pr", ("ALWAYS_CI", "ALWAYS_PRE_PR"))],
    ids=["ci", "pre-pr"],
)
def test_every_pinned_test_is_listed(group, pins):
    """A parametrized test counts as listed when any of its cases is: a
    regeneration lists them all, and checking that needs a collection.
    """
    listed = set(_members(group))
    prefixes = [p for name in pins for p in getattr(_selector(), name)]
    stale = [p for p in prefixes if not _pinned(p)]
    assert not stale, (
        f"{stale} in scripts/select_test_groups.py match no test function any "
        f"more; point them at the test's new name, or drop them."
    )
    missing = [
        nodeid
        for p in prefixes
        for nodeid in _pinned(p)
        if nodeid not in listed and not any(m.startswith(nodeid + "[") for m in listed)
    ]
    assert not missing, (
        f"tests/groups/{group}.txt leaves out {len(missing)} test(s) the "
        f"selector pins, e.g. {missing[:5]}. Add their node ids (every "
        f"parametrized case) -- a regeneration would put them back anyway."
    )
