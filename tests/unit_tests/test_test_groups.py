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
"""

from __future__ import annotations

import ast
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
