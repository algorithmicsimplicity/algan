from __future__ import annotations

import sys
from pathlib import Path

import pytest

from docs.make_and_open_docs import _sphinx_command


@pytest.mark.parametrize(
    ("skip_examples", "doctree_name"),
    [
        (False, "doctrees-with-examples"),
        (True, "doctrees-without-examples"),
    ],
)
def test_sphinx_command_separates_example_doctree_caches(
    skip_examples: bool,
    doctree_name: str,
) -> None:
    docs_dir = Path("project") / "docs"

    command = _sphinx_command(docs_dir, skip_examples=skip_examples)

    assert command[:3] == [sys.executable, "-m", "sphinx"]
    assert command[command.index("-d") + 1] == str(docs_dir / "build" / doctree_name)
    assert ("skip-manim" in command) is skip_examples
    assert str(docs_dir / "build") in command
