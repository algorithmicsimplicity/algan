"""Facts that the documentation and the agent skill state about the repository.

``test_doc_examples.py`` checks the code in the documentation. This module
checks the few things around that code which are written down twice and drift
apart without anything failing: the commit and version the agent skill says it
describes.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_ROOT = REPO_ROOT / ".claude" / "skills" / "algan-video"


def _skill_metadata(field: str) -> str:
    header = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8").split("---")[1]
    match = re.search(rf'^\s*{re.escape(field)}:\s*"([^"]*)"', header, re.M)
    assert match, f"SKILL.md's header has no {field!r} field"
    return match.group(1)


def test_skill_pins_one_commit():
    """The skill names the commit it was checked against in two places, and
    links into the source at that commit. They named three different commits
    once, so an agent resolving API drift followed links into a tree two months
    older than the text it was reading."""
    commit = _skill_metadata("algan-source-commit")
    assert re.fullmatch(r"[0-9a-f]{40}", commit), (
        f"algan-source-commit should be one full commit hash, got {commit!r}"
    )
    sources = (SKILL_ROOT / "references" / "api-sources.md").read_text(
        encoding="utf-8"
    )
    named = set(re.findall(r"\b[0-9a-f]{40}\b", sources))
    assert named == {commit}, (
        f"references/api-sources.md names commits {sorted(named)}, but SKILL.md's "
        f"algan-source-commit is {commit}. When the skill is re-checked against "
        "the source, update both, and the links, to the same commit."
    )


def test_skill_version_is_the_package_version():
    """``algan-version`` in the skill is the version in ``pyproject.toml``.

    A version bump fails this until the skill has been re-checked against the
    source: update ``algan-version``, ``algan-source-commit``, ``source-checked``
    and the links in ``references/api-sources.md`` together.
    """
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.M)
    assert match, "pyproject.toml has no version"
    assert _skill_metadata("algan-version") == match.group(1), (
        f"SKILL.md says algan-version {_skill_metadata('algan-version')!r}, "
        f"pyproject.toml says {match.group(1)!r}"
    )
