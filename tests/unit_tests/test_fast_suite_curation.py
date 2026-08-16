"""The fast suite's membership is hand-picked, so it has to stay written down.

``--fast`` runs the tests marked ``fast`` and nothing else. That makes joining
the suite a deliberate act -- one line in one file -- which is the point, but it
also makes it a quiet one: a ``pytestmark`` added in passing grows the
development loop for everyone, and nothing else in the repository would notice.

So the marker and the table in ``tests/README.md`` are checked against each
other. Adding a module to the fast suite means saying, in that table, what makes
it a canary for changes elsewhere; dropping one means removing the row. Neither
direction can happen by accident.

This is an audit rather than a behaviour test, so it is not itself in the fast
suite -- it fails when you *change the suite*, which is a thing you find out
before pushing.
"""

from __future__ import annotations

import re
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parents[1]
README = TESTS_DIR / "README.md"

# The heading pair bracketing the membership table in tests/README.md.
TABLE_START = "### What is in it"
TABLE_END = "### What is not in it"

# Any path-like `...test_something.py` inside a backtick span.
_FILE_IN_BACKTICKS = re.compile(r"`(?:[\w./-]*/)?(test_\w+\.py)`")


def _files_carrying_the_marker() -> set[str]:
    """Every test file that puts itself in the fast suite, by file name."""
    marker = "pytest.mark." + "fast"  # split so this file does not match itself
    return {
        path.name
        for path in sorted(TESTS_DIR.rglob("test_*.py"))
        if path != Path(__file__) and marker in path.read_text(encoding="utf-8")
    }


def _files_named_in_the_readme() -> set[str]:
    text = README.read_text(encoding="utf-8")
    start = text.index(TABLE_START)
    end = text.index(TABLE_END, start)
    return set(_FILE_IN_BACKTICKS.findall(text[start:end]))


def test_the_membership_table_matches_the_markers():
    marked = _files_carrying_the_marker()
    documented = _files_named_in_the_readme()

    undocumented = sorted(marked - documented)
    stale = sorted(documented - marked)

    assert not undocumented, (
        f"{undocumented} joined the fast suite without being listed in "
        f"{TABLE_START!r} in tests/README.md. Add a row saying which change "
        "elsewhere in the codebase is liable to break it -- or, if the honest "
        "answer is 'only a change to what it tests', drop the marker instead."
    )
    assert not stale, (
        f"tests/README.md lists {stale} as part of the fast suite, but nothing "
        "in those files carries the marker any more. Remove the rows."
    )


def test_the_fast_suite_still_renders_something():
    """The one end-to-end render is the only renderer coverage in the loop."""
    assert "test_fast_render.py" in _files_carrying_the_marker(), (
        "tests/fast/test_fast_render.py has left the fast suite, which leaves "
        "the development loop unable to see any renderer regression at all."
    )
