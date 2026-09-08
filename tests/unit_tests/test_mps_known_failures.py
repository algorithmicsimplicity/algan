"""The Apple-GPU xfail list names tests that exist.

``tests/mps_known_failures.py`` is what lets the required macOS MPS arm be both
green and honest, and its whole value is that the entries are live: an
``xfail(strict=True)`` on a nodeid nothing collects is a silent no-op, so a
renamed or deleted test would quietly take its own entry out of service and the
count in ``DESIGN_mps_support.md`` §4 would stop meaning anything.

Nothing here needs an Apple GPU -- the list is data, and these are checks about
the data. The arm itself is what checks the *claims*: ``strict`` makes a test
that starts passing turn it red, so the list cannot outlive the defects.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_LIST = _ROOT / "tests" / "mps_known_failures.py"


def _known():
    spec = importlib.util.spec_from_file_location("algan_mps_known_failures", _LIST)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.KNOWN_FAILURES


def _test_functions(path):
    """Every ``def test_*`` in a file, including ones nested in a class."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    }


@pytest.mark.fast
def test_every_entry_names_a_test_that_exists():
    """A typo'd or stale nodeid is an entry that does nothing and says nothing.

    The parametrisation suffix is stripped rather than checked: ``[sheet]`` is
    produced by a fixture or a ``parametrize`` argument and is not in the
    function's name, so demanding it here would mean re-implementing pytest's
    id generation. What this catches is the failure that actually happens --
    a test renamed or moved while its entry stayed behind.
    """
    missing = []
    for nodeid in _known():
        file_part, _, test_part = nodeid.partition("::")
        name = test_part.split("[", 1)[0]
        path = _ROOT / file_part
        if not path.is_file():
            missing.append(f"{nodeid} -- no such file")
        elif name not in _test_functions(path):
            missing.append(f"{nodeid} -- {file_part} has no {name}")
    assert not missing, "\n".join(
        ["stale entries in tests/mps_known_failures.py:", *missing]
    )


def test_every_entry_carries_a_reason_pointing_at_the_measurement():
    """An entry's reason has to point at the measurement behind it.

    Every entry has to name the section of ``DESIGN_mps_support.md`` that holds
    the measurement, because that section is what the next person needs and the
    xfail reason is where they will look for it.
    """
    vague = [
        nodeid
        for nodeid, reason in _known().items()
        if "DESIGN_mps_support.md" not in reason
    ]
    assert not vague, "\n".join(
        ["these entries do not point at a measurement:", *vague]
    )
